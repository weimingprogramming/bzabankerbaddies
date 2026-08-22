import collections
import datetime
import heapq
from typing import List, Optional, Set, Dict, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic Models
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
# Stateful Streaming Engine with Phase 2 Identity Signal Assessment
# ---------------------------------------------------------------------------

class RiskEngine:
    def __init__(self):
        self.WINDOW_SECONDS = 24 * 60 * 60
        self.t_max: float = 0.0
        
        # Idempotency cache: txId -> (payload_hash, risk_score)
        self.tx_cache: Dict[str, Tuple[str, float]] = {}
        
        # Active transactions store: txId -> Transaction
        self.active_txs: Dict[str, Transaction] = {}
        self.tx_timestamps: Dict[str, float] = {}
        
        # Min-Heap for time-based pruning: (timestamp, txId)
        self.edge_heap: List[Tuple[float, str]] = []
        
        # Graph multi-edge representation
        self.adj: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self.rev_adj: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        
        # Phase 2 Identity Indexing
        self.ip_to_txs: Dict[str, Set[str]] = collections.defaultdict(set)
        self.device_to_txs: Dict[str, Set[str]] = collections.defaultdict(set)

    def clear(self):
        self.__init__()

    def parse_timestamp(self, iso_str: str) -> float:
        s = iso_str.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(s).timestamp()

    def prune(self):
        """Removes transactions and graph/identity state older than 24h from current t_max."""
        cutoff = self.t_max - self.WINDOW_SECONDS
        
        while self.edge_heap and self.edge_heap[0][0] < cutoff:
            ts, txId = heapq.heappop(self.edge_heap)
            
            if txId in self.active_txs and self.tx_timestamps.get(txId) == ts:
                tx = self.active_txs.pop(txId)
                del self.tx_timestamps[txId]
                
                # Prune graph edges
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

                # Prune identity indexes
                if tx.ipAddress and tx.ipAddress in self.ip_to_txs:
                    self.ip_to_txs[tx.ipAddress].discard(txId)
                    if not self.ip_to_txs[tx.ipAddress]:
                        del self.ip_to_txs[tx.ipAddress]
                        
                if tx.deviceId and tx.deviceId in self.device_to_txs:
                    self.device_to_txs[tx.deviceId].discard(txId)
                    if not self.device_to_txs[tx.deviceId]:
                        del self.device_to_txs[tx.deviceId]

    def _get_reachable_nodes(self, start: str, reverse: bool = False) -> Set[str]:
        """BFS to get all reachable nodes in forward or reverse direction."""
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
        """Evaluates graph topology changes (extensions, cycles, convergence, multi-loops)."""
        src, dst = tx.fromUserId, tx.toUserId
        
        if src == dst:
            return 0.85  # Self-loop anomaly
            
        ancestors_src = self._get_reachable_nodes(src, reverse=True)
        is_return_cycle = dst in ancestors_src or src == dst
        
        # Check convergence (dst reachable from multiple upstream nodes)
        incoming_to_dst = len(self.rev_adj.get(dst, {}))
        
        if is_return_cycle:
            # Multi-loop return vs single return
            if incoming_to_dst > 1:
                return 0.95
            return 0.80
        elif incoming_to_dst > 0:
            return 0.50  # Structural convergence
        elif src in self.adj:
            return 0.25  # Extension
            
        return 0.05  # Isolated edge

    def _calculate_identity_signal(self, tx: Transaction) -> Tuple[float, float]:
        """
        Calculates Phase 2 Identity modifiers:
        Returns: (identity_multiplier, risk_additive_bonus)
        """
        src, dst = tx.fromUserId, tx.toUserId
        upstream_txs = [
            self.active_txs[t] for t in self.active_txs 
            if self.active_txs[t].toUserId == src
        ]
        
        upstream_ips = {t.ipAddress for t in upstream_txs if t.ipAddress}
        upstream_devices = {t.deviceId for t in upstream_txs if t.deviceId}
        
        mod_multiplier = 1.0
        bonus_risk = 0.0
        
        # 1. Missing Identity Mid-Flow (Trail Breaking Evasion)
        if upstream_txs and (upstream_ips or upstream_devices):
            missing_ip = tx.ipAddress is None and len(upstream_ips) > 0
            missing_dev = tx.deviceId is None and len(upstream_devices) > 0
            
            if missing_ip or missing_dev:
                bonus_risk += 0.20  # Explicit evasion penalty
                
        # 2. Identity Shift Mid-Flow
        if tx.deviceId and upstream_devices and tx.deviceId not in upstream_devices:
            bonus_risk += 0.15
        if tx.ipAddress and upstream_ips and tx.ipAddress not in upstream_ips:
            bonus_risk += 0.10
            
        # 3. Shared Identity Across Disconnected Components
        reachable_from_src = self._get_reachable_nodes(src) | {src}
        reachable_from_dst = self._get_reachable_nodes(dst) | {dst}
        connected_cluster = reachable_from_src | reachable_from_dst
        
        for attr, index in [(tx.ipAddress, self.ip_to_txs), (tx.deviceId, self.device_to_txs)]:
            if attr and attr in index:
                for existing_tx_id in index[attr]:
                    existing_tx = self.active_txs[existing_tx_id]
                    # Check if the shared identity spans disconnected nodes
                    if (existing_tx.fromUserId not in connected_cluster and 
                        existing_tx.toUserId not in connected_cluster):
                        bonus_risk += 0.15
                        break

        # 4. Consistent Identity Across Active Path (Reinforces structural intent)
        if tx.deviceId and tx.deviceId in upstream_devices:
            mod_multiplier *= 1.15
        if tx.ipAddress and tx.ipAddress in upstream_ips:
            mod_multiplier *= 1.10
            
        return mod_multiplier, bonus_risk

    def process_transaction(self, tx: Transaction) -> float:
        ts = self.parse_timestamp(tx.createdAt)
        payload_str = tx.model_dump_json()
        
        # Idempotency check
        if tx.txId in self.tx_cache:
            cached_payload, cached_score = self.tx_cache[tx.txId]
            if cached_payload == payload_str:
                return cached_score

        # Update streaming max time & prune window
        if ts > self.t_max:
            self.t_max = ts
        self.prune()

        # Compute combined signals
        struct_score = self._calculate_structural_score(tx)
        ident_mult, ident_bonus = self._calculate_identity_signal(tx)
        
        final_score = min(1.0, max(0.0, (struct_score * ident_mult) + ident_bonus))
        final_score = round(final_score, 4)

        # Update state graph & identity indexes
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

# Engine Singleton
engine = RiskEngine()

# ---------------------------------------------------------------------------
# API Endpoints
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