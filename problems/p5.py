import collections
import datetime
import heapq
from typing import List, Optional, Set, Dict, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic Schemas
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
# High-Precision Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    def __init__(self):
        self.WINDOW_SECONDS = 24 * 60 * 60
        self.t_max: float = 0.0
        
        # Idempotency cache: txId -> (payload_json, risk_score)
        self.tx_cache: Dict[str, Tuple[str, float]] = {}
        
        # Active streaming state
        self.active_txs: Dict[str, Transaction] = {}
        self.tx_timestamps: Dict[str, float] = {}
        
        # Priority queue for rolling 24h event-time pruning: (timestamp, txId)
        self.edge_heap: List[Tuple[float, str]] = []
        
        # Graph multi-edge representations storing lists of txIds
        self.adj: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        self.rev_adj: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        
        # Identity index mappings
        self.ip_to_txs: Dict[str, Set[str]] = collections.defaultdict(set)
        self.device_to_txs: Dict[str, Set[str]] = collections.defaultdict(set)

    def clear(self):
        self.__init__()

    def parse_timestamp(self, iso_str: str) -> float:
        s = iso_str.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(s).timestamp()

    def prune(self, current_time: float):
        """Prunes transactions older than 24h relative to current event time."""
        cutoff = current_time - self.WINDOW_SECONDS
        
        while self.edge_heap and self.edge_heap[0][0] < cutoff:
            ts, txId = heapq.heappop(self.edge_heap)
            
            if txId in self.active_txs and self.tx_timestamps.get(txId) == ts:
                tx = self.active_txs.pop(txId)
                del self.tx_timestamps[txId]
                
                u, v = tx.fromUserId, tx.toUserId
                if txId in self.adj[u][v]:
                    self.adj[u][v].remove(txId)
                    if not self.adj[u][v]: del self.adj[u][v]
                    if not self.adj[u]: del self.adj[u]
                    
                if txId in self.rev_adj[v][u]:
                    self.rev_adj[v][u].remove(txId)
                    if not self.rev_adj[v][u]: del self.rev_adj[v][u]
                    if not self.rev_adj[v]: del self.rev_adj[v]

                if tx.ipAddress and tx.ipAddress in self.ip_to_txs:
                    self.ip_to_txs[tx.ipAddress].discard(txId)
                    if not self.ip_to_txs[tx.ipAddress]: del self.ip_to_txs[tx.ipAddress]
                        
                if tx.deviceId and tx.deviceId in self.device_to_txs:
                    self.device_to_txs[tx.deviceId].discard(txId)
                    if not self.device_to_txs[tx.deviceId]: del self.device_to_txs[tx.deviceId]

    def _find_upstream_lineage(self, node: str, max_depth: int = 4) -> List[List[Transaction]]:
        """BFS to retrieve multi-hop path chains leading into node."""
        paths = []
        queue = collections.deque([([node], [])])
        
        while queue:
            node_path, tx_path = queue.popleft()
            curr = node_path[-1]
            
            if len(node_path) > max_depth:
                if tx_path: paths.append(tx_path)
                continue
                
            parents = self.rev_adj.get(curr, {})
            if not parents:
                if tx_path: paths.append(tx_path)
                continue
                
            for parent, tx_ids in parents.items():
                if parent in node_path:
                    continue  # Stop on cycle
                latest_tx_id = tx_ids[-1]  # Most recent transaction on edge
                latest_tx = self.active_txs[latest_tx_id]
                queue.append((node_path + [parent], tx_path + [latest_tx]))
                
        return paths

    def _get_weakly_connected_component(self, start_node: str) -> Set[str]:
        """Returns all nodes connected directly or indirectly ignoring edge direction."""
        component = {start_node}
        queue = collections.deque([start_node])
        
        while queue:
            curr = queue.popleft()
            neighbors = set(self.adj.get(curr, {}).keys()) | set(self.rev_adj.get(curr, {}).keys())
            for nxt in neighbors:
                if nxt not in component:
                    component.add(nxt)
                    queue.append(nxt)
        return component

    def _evaluate_topology(self, u: str, v: str) -> float:
        """Phase 1 Structural Score Engine."""
        if u == v:
            return 0.85
            
        # Check if v can reach u (Cycle Return Path)
        reachable_from_v = set()
        queue = collections.deque([v])
        while queue:
            curr = queue.popleft()
            for nxt in self.adj.get(curr, {}):
                if nxt not in reachable_from_v:
                    reachable_from_v.add(nxt)
                    queue.append(nxt)
                    
        is_return_cycle = u in reachable_from_v
        in_degree_v = len(self.rev_adj.get(v, {}))
        
        if is_return_cycle:
            # Multi-loop vs single return cycle
            out_degree_v = len(self.adj.get(v, {}))
            return 0.92 if (in_degree_v > 1 or out_degree_v > 1) else 0.75
        elif in_degree_v > 0:
            return 0.45  # Convergence
        elif u in self.adj or u in self.rev_adj:
            return 0.20  # Extension
            
        return 0.05  # Isolated

    def _evaluate_identity(self, tx: Transaction, lineages: List[List[Transaction]]) -> Tuple[float, float]:
        """Phase 2 Identity Fingerprint Engine."""
        u, v = tx.fromUserId, tx.toUserId
        mod_multiplier = 1.0
        bonus_risk = 0.0
        
        all_upstream_txs = [t for path in lineages for t in path]
        upstream_ips = {t.ipAddress for t in all_upstream_txs if t.ipAddress}
        upstream_devices = {t.deviceId for t in all_upstream_txs if t.deviceId}
        
        # 1. Trail Breaking (Missing Identity Mid-Flow Evasion)
        if all_upstream_txs:
            if (tx.ipAddress is None and upstream_ips) or (tx.deviceId is None and upstream_devices):
                bonus_risk += 0.20

        # 2. Identity Shift Mid-Flow
        if tx.deviceId and upstream_devices and tx.deviceId not in upstream_devices:
            bonus_risk += 0.12
        if tx.ipAddress and upstream_ips and tx.ipAddress not in upstream_ips:
            bonus_risk += 0.10

        # 3. Shared Identity Across Disconnected Components
        comp_u = self._get_weakly_connected_component(u)
        comp_v = self._get_weakly_connected_component(v)
        cluster = comp_u | comp_v
        
        for attr, index in [(tx.ipAddress, self.ip_to_txs), (tx.deviceId, self.device_to_txs)]:
            if attr and attr in index:
                for ex_tx_id in index[attr]:
                    ex_tx = self.active_txs[ex_tx_id]
                    if ex_tx.fromUserId not in cluster and ex_tx.toUserId not in cluster:
                        bonus_risk += 0.15
                        break

        # 4. Consistent Identity Reinforcement
        if tx.deviceId and tx.deviceId in upstream_devices:
            mod_multiplier *= 1.10
        if tx.ipAddress and tx.ipAddress in upstream_ips:
            mod_multiplier *= 1.05

        return mod_multiplier, bonus_risk

    def _evaluate_value(self, tx: Transaction, lineages: List[List[Transaction]]) -> float:
        """Phase 3 Value Signal Engine."""
        if not lineages:
            return 0.0

        # Examine immediate preceding amounts along lineages
        predecessor_amounts = [path[0].amount for path in lineages if path]
        if not predecessor_amounts:
            return 0.0
            
        avg_prev_amount = sum(predecessor_amounts) / len(predecessor_amounts)
        
        # Value Trajectory Reversal (Amount increases mid-flow)
        if tx.amount > avg_prev_amount:
            ratio = (tx.amount - avg_prev_amount) / avg_prev_amount
            return min(0.35, 0.25 + (ratio * 0.10))
            
        # Consistent Value Decay (Expected Layering -> Reduces Anomaly Risk relative to reversals)
        decay_ratio = tx.amount / avg_prev_amount
        if 0.85 <= decay_ratio <= 1.0:
            return -0.05
            
        return 0.05

    def process_transaction(self, tx: Transaction) -> float:
        ts = self.parse_timestamp(tx.createdAt)
        payload_str = tx.model_dump_json()
        
        # Idempotency check
        if tx.txId in self.tx_cache:
            cached_payload, cached_score = self.tx_cache[tx.txId]
            if cached_payload == payload_str:
                return cached_score

        # Advance timeline & prune state
        if ts > self.t_max:
            self.t_max = ts
        self.prune(ts)

        # Reconstruct active path lineage
        lineages = self._find_upstream_lineage(tx.fromUserId)

        # Compute signal dimensions
        s_struct = self._evaluate_topology(tx.fromUserId, tx.toUserId)
        s_id_mult, s_id_bonus = self._evaluate_identity(tx, lineages)
        s_val_bonus = self._evaluate_value(tx, lineages)
        
        # Composite score
        raw_score = (s_struct * s_id_mult) + s_id_bonus + s_val_bonus
        final_score = round(min(1.0, max(0.0, raw_score)), 4)

        # Apply state updates
        self.active_txs[tx.txId] = tx
        self.tx_timestamps[tx.txId] = ts
        heapq.heappush(self.edge_heap, (ts, tx.txId))
        
        self.adj[tx.fromUserId][tx.toUserId].append(tx.txId)
        self.rev_adj[tx.toUserId][tx.fromUserId].append(tx.txId)
        
        if tx.ipAddress:
            self.ip_to_txs[tx.ipAddress].add(tx.txId)
        if tx.deviceId:
            self.device_to_txs[tx.deviceId].add(tx.txId)
            
        self.tx_cache[tx.txId] = (payload_str, final_score)
        return final_score

# Singleton engine instance
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