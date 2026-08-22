import collections
import datetime
import heapq
from typing import List, Optional, Set, Dict, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic Models (Forward Compatible)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str

class ResetRequest(BaseModel):
    clearTransactions: bool = True

class ResetResponse(BaseModel):
    clearTransactions: bool

class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    txId: str
    fromUserId: str
    toUserId: str
    amount: float
    createdAt: str
    ipAddress: Optional[str] = None
    deviceId: Optional[str] = None

class TransactionResult(BaseModel):
    txId: str
    riskScore: float

class TransactionBatchRequest(BaseModel):
    transactions: List[Transaction]

class TransactionBatchResponse(BaseModel):
    transactions: List[TransactionResult]

# ---------------------------------------------------------------------------
# State Engine: Phase 1 (Topology) + Phase 2 (Identity) + Phase 3 (Value)
# ---------------------------------------------------------------------------

class RiskEngine:
    def __init__(self):
        self.WINDOW_SECONDS = 24 * 60 * 60
        self.t_max: float = 0.0
        
        # Idempotency cache: txId -> (payload_hash, risk_score)
        self.tx_cache: Dict[str, Tuple[str, float]] = {}
        
        # Active state tracking
        self.active_txs: Dict[str, Transaction] = {}
        self.tx_timestamps: Dict[str, float] = {}
        
        # Priority Queue for 24h lookback window pruning: (timestamp, txId)
        self.edge_heap: List[Tuple[float, str]] = []
        
        # Structural Graph Adjacency
        self.adj: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self.rev_adj: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        
        # Phase 2 Identity Indices
        self.ip_to_txs: Dict[str, Set[str]] = collections.defaultdict(set)
        self.device_to_txs: Dict[str, Set[str]] = collections.defaultdict(set)

    def clear(self):
        self.__init__()

    def parse_timestamp(self, iso_str: str) -> float:
        s = iso_str.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(s).timestamp()

    def prune(self):
        """Strict 24-hour event-time window pruning across all indices."""
        cutoff = self.t_max - self.WINDOW_SECONDS
        
        while self.edge_heap and self.edge_heap[0][0] < cutoff:
            ts, txId = heapq.heappop(self.edge_heap)
            
            if txId in self.active_txs and self.tx_timestamps.get(txId) == ts:
                tx = self.active_txs.pop(txId)
                del self.tx_timestamps[txId]
                
                # Prune Graph Edges
                u, v = tx.fromUserId, tx.toUserId
                if self.adj[u][v] > 1:
                    self.adj[u][v] -= 1
                else:
                    del self.adj[u][v]
                    if not self.adj[u]: del self.adj[u]
                    
                if self.rev_adj[v][u] > 1:
                    self.rev_adj[v][u] -= 1
                else:
                    del self.rev_adj[v][u]
                    if not self.rev_adj[v]: del self.rev_adj[v]

                # Prune Identity Maps
                if tx.ipAddress and tx.ipAddress in self.ip_to_txs:
                    self.ip_to_txs[tx.ipAddress].discard(txId)
                    if not self.ip_to_txs[tx.ipAddress]:
                        del self.ip_to_txs[tx.ipAddress]
                        
                if tx.deviceId and tx.deviceId in self.device_to_txs:
                    self.device_to_txs[tx.deviceId].discard(txId)
                    if not self.device_to_txs[tx.deviceId]:
                        del self.device_to_txs[tx.deviceId]

    def _get_reachable_nodes(self, start: str, reverse: bool = False) -> Set[str]:
        graph = self.rev_adj if reverse else self.adj
        if start not in graph:
            return set()
            
        visited = set()
        queue = collections.deque([start])
        while queue:
            curr = queue.popleft()
            for nxt in graph.get(curr, {}):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited

    def _calculate_structural_score(self, tx: Transaction) -> float:
        """Phase 1: Graph Topology Evaluation."""
        src, dst = tx.fromUserId, tx.toUserId
        
        if src == dst:
            return 0.85
            
        ancestors_src = self._get_reachable_nodes(src, reverse=True)
        is_return_cycle = dst in ancestors_src or src == dst
        incoming_to_dst = len(self.rev_adj.get(dst, {}))
        
        if is_return_cycle:
            return 0.95 if incoming_to_dst > 1 else 0.80
        elif incoming_to_dst > 0:
            return 0.50  # Convergence
        elif src in self.adj:
            return 0.25  # Extension
            
        return 0.05  # Isolated Edge

    def _calculate_identity_signal(self, tx: Transaction) -> Tuple[float, float]:
        """Phase 2: Identity Fingerprint Evaluation."""
        src, dst = tx.fromUserId, tx.toUserId
        upstream_txs = [
            self.active_txs[t] for t in self.active_txs 
            if self.active_txs[t].toUserId == src
        ]
        
        upstream_ips = {t.ipAddress for t in upstream_txs if t.ipAddress}
        upstream_devices = {t.deviceId for t in upstream_txs if t.deviceId}
        
        mod_multiplier = 1.0
        bonus_risk = 0.0
        
        # Mid-Flow Identity Evasion (Dropped IP or Device)
        if upstream_txs and (upstream_ips or upstream_devices):
            if (tx.ipAddress is None and upstream_ips) or (tx.deviceId is None and upstream_devices):
                bonus_risk += 0.20
                
        # Mid-Flow Identity Shift
        if tx.deviceId and upstream_devices and tx.deviceId not in upstream_devices:
            bonus_risk += 0.15
        if tx.ipAddress and upstream_ips and tx.ipAddress not in upstream_ips:
            bonus_risk += 0.10
            
        # Shared Identity Across Disconnected Components
        connected_cluster = self._get_reachable_nodes(src) | self._get_reachable_nodes(dst) | {src, dst}
        for attr, index in [(tx.ipAddress, self.ip_to_txs), (tx.deviceId, self.device_to_txs)]:
            if attr and attr in index:
                for existing_tx_id in index[attr]:
                    ex_tx = self.active_txs[existing_tx_id]
                    if ex_tx.fromUserId not in connected_cluster and ex_tx.toUserId not in connected_cluster:
                        bonus_risk += 0.15
                        break

        # Consistent Identity Alignment
        if tx.deviceId and tx.deviceId in upstream_devices:
            mod_multiplier *= 1.10
        if tx.ipAddress and tx.ipAddress in upstream_ips:
            mod_multiplier *= 1.05
            
        return mod_multiplier, bonus_risk

    def _calculate_value_signal(self, tx: Transaction) -> float:
        """
        Phase 3: Amount Progression inside Inferred Flow Segments.
        Detects expected value decay vs. value trajectory reversals.
        """
        src = tx.fromUserId
        upstream_txs = [
            self.active_txs[t] for t in self.active_txs 
            if self.active_txs[t].toUserId == src
        ]
        
        if not upstream_txs:
            return 0.0  # Isolated or flow origin
            
        max_upstream_amount = max(t.amount for t in upstream_txs)
        
        # 1. Value Trajectory Reversal (Amount increases mid-flow)
        if tx.amount > max_upstream_amount:
            # Significant risk penalty for breaking expected flow decay
            overage_ratio = (tx.amount - max_upstream_amount) / max_upstream_amount
            return min(0.35, 0.25 + (overage_ratio * 0.10))
            
        # 2. Consistent Value Decay (Expected Layering Pattern)
        decay_ratio = tx.amount / max_upstream_amount
        if 0.85 <= decay_ratio <= 1.0:
            # Standard layering progression -> No extra risk added (lowest relative score)
            return 0.0
            
        # 3. Discontinuous Value Jump/Drop
        if decay_ratio < 0.50:
            return 0.10
            
        return 0.05

    def process_transaction(self, tx: Transaction) -> float:
        ts = self.parse_timestamp(tx.createdAt)
        payload_str = tx.model_dump_json()
        
        # Idempotency Check
        if tx.txId in self.tx_cache:
            cached_payload, cached_score = self.tx_cache[tx.txId]
            if cached_payload == payload_str:
                return cached_score

        # Stream Time Advance & Pruning
        if ts > self.t_max:
            self.t_max = ts
        self.prune()

        # Multi-Signal Calculations
        struct_score = self._calculate_structural_score(tx)
        ident_mult, ident_bonus = self._calculate_identity_signal(tx)
        val_bonus = self._calculate_value_signal(tx)
        
        # Composite Scoring Formula
        raw_score = (struct_score * ident_mult) + ident_bonus + val_bonus
        final_score = round(min(1.0, max(0.0, raw_score)), 4)

        # Update Active Graph & State
        self.active_txs[tx.txId] = tx
        self.tx_timestamps[tx.txId] = ts
        heapq.heappush(self.edge_heap, (ts, tx.txId))
        
        self.adj[tx.fromUserId][tx.toUserId] += 1
        self.rev_adj[tx.toUserId][tx.fromUserId] += 1
        
        if tx.ipAddress:
            self.ip_to_txs[tx.ipAddress].add(tx.txId)
        if tx.deviceId:
            self.device_to_txs[tx.deviceId].add(tx.txId)
            
        self.tx_cache[tx.txId] = (payload_str, final_score)
        return final_score

# Service Instance
engine = RiskEngine()

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@router.get("/ghost-chains/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")

@router.post("/ghost-chains/reset", response_model=ResetResponse)
def reset_state(req: ResetRequest):
    if req.clearTransactions:
        engine.clear()
    return ResetResponse(clearTransactions=req.clearTransactions)

@router.post("/ghost-chains/transactions", response_model=TransactionBatchResponse)
def process_transactions(batch: TransactionBatchRequest):
    results = []
    for tx in batch.transactions:
        score = engine.process_transaction(tx)
        results.append(TransactionResult(txId=tx.txId, riskScore=score))
    return TransactionBatchResponse(transactions=results)