import collections
import datetime
import hashlib
import heapq
import json
from typing import List, Optional, Set, Dict, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---------------------------------------------------------------------------
# Data Models
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
# Multi-Signal Graph & Flow Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    def __init__(self):
        self.WINDOW_SECONDS = 24 * 60 * 60  # 24 Hours Lookback
        self.t_max: float = 0.0
        
        # Idempotency cache: txId -> (payload_hash, risk_score)
        self.tx_cache: Dict[str, Tuple[str, float]] = {}
        
        # Active streaming state
        self.active_txs: Dict[str, Transaction] = {}
        self.tx_timestamps: Dict[str, float] = {}
        
        # Min-Heap for 24h event-time window pruning: (timestamp, txId)
        self.edge_heap: List[Tuple[float, str]] = []
        
        # Directed Multi-Graph: src -> dst -> list of txIds
        self.adj: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        self.rev_adj: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        
        # Identity Index Maps
        self.ip_to_txs: Dict[str, Set[str]] = collections.defaultdict(set)
        self.device_to_txs: Dict[str, Set[str]] = collections.defaultdict(set)

    def clear(self):
        self.__init__()

    def parse_timestamp(self, iso_str: str) -> float:
        s = iso_str.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(s).timestamp()

    def _hash_payload(self, tx: Transaction) -> str:
        d = tx.model_dump()
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()

    def prune(self, current_time: float):
        """Strict 24-hour event-time rolling window cleanup."""
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

    def _get_upstream_paths(self, target_node: str, current_time: float, max_depth: int = 5) -> List[List[Transaction]]:
        """BFS to reconstruct all temporally valid path lineages leading into target_node."""
        paths = []
        queue = collections.deque([(target_node, [], {target_node})])
        
        while queue:
            curr_node, curr_tx_path, visited = queue.popleft()
            
            if len(curr_tx_path) >= max_depth:
                if curr_tx_path: paths.append(list(reversed(curr_tx_path)))
                continue
                
            parents = self.rev_adj.get(curr_node, {})
            if not parents:
                if curr_tx_path: paths.append(list(reversed(curr_tx_path)))
                continue
                
            found_valid = False
            for parent, tx_ids in parents.items():
                if parent in visited: continue
                
                t_limit = current_time if not curr_tx_path else self.parse_timestamp(curr_tx_path[-1].createdAt)
                valid_txs = [
                    self.active_txs[tid] for tid in tx_ids 
                    if tid in self.active_txs and self.parse_timestamp(self.active_txs[tid].createdAt) <= t_limit
                ]
                
                if valid_txs:
                    found_valid = True
                    best_tx = max(valid_txs, key=lambda x: self.parse_timestamp(x.createdAt))
                    queue.append((parent, curr_tx_path + [best_tx], visited | {parent}))
                    
            if not found_valid and curr_tx_path:
                paths.append(list(reversed(curr_tx_path)))
                
        return paths

    def _get_weakly_connected_component(self, start_node: str) -> Set[str]:
        visited = {start_node}
        queue = collections.deque([start_node])
        while queue:
            curr = queue.popleft()
            neighbors = set(self.adj.get(curr, {}).keys()) | set(self.rev_adj.get(curr, {}).keys())
            for nxt in neighbors:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited

    def _can_reach(self, start_node: str, target_node: str) -> bool:
        if start_node == target_node: return True
        visited = {start_node}
        queue = collections.deque([start_node])
        while queue:
            curr = queue.popleft()
            for nxt in self.adj.get(curr, {}):
                if nxt == target_node: return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def _count_return_paths(self, start_node: str, target_node: str) -> int:
        if not self._can_reach(start_node, target_node): return 0
        paths_count = 0
        queue = collections.deque([(start_node, {start_node})])
        while queue:
            curr, visited = queue.popleft()
            if curr == target_node:
                paths_count += 1
                if paths_count >= 3: break
                continue
            for nxt in self.adj.get(curr, {}):
                if nxt not in visited:
                    queue.append((nxt, visited | {nxt}))
        return paths_count

    def _evaluate_topology(self, u: str, v: str) -> float:
        """Phase 1 Structural Score Calibrator."""
        if u == v:
            return 0.85  # Self loop
            
        can_return = self._can_reach(v, u)
        if can_return:
            return_paths = self._count_return_paths(v, u)
            in_degree_u = len(self.rev_adj.get(u, {}))
            if return_paths > 1 or in_degree_u > 1:
                return 0.95  # Multi-loop return cycle (Ex 5)
            return 0.78  # Single return cycle (Ex 4)
            
        in_degree_v = len(self.rev_adj.get(v, {}))
        if in_degree_v > 0:
            return 0.48  # Structural Convergence (Ex 3)
            
        degree_u = len(self.adj.get(u, {})) + len(self.rev_adj.get(u, {}))
        if degree_u > 0:
            return 0.22  # Extension (Ex 2)
            
        return 0.05  # Isolated edge (Ex 1)

    def _evaluate_identity(self, tx: Transaction, lineages: List[List[Transaction]]) -> Tuple[float, float]:
        """Phase 2 Identity Fingerprint & Evasion Calibrator."""
        u, v = tx.fromUserId, tx.toUserId
        mod_multiplier = 1.0
        bonus_risk = 0.0
        
        all_upstream = [t for path in lineages for t in path]
        upstream_ips = {t.ipAddress for t in all_upstream if t.ipAddress}
        upstream_devs = {t.deviceId for t in all_upstream if t.deviceId}
        
        # 1. Trail Breaking (Missing identity mid-flow)
        if all_upstream:
            if tx.ipAddress is None and upstream_ips:
                bonus_risk += 0.22
            if tx.deviceId is None and upstream_devs:
                bonus_risk += 0.22

        # 2. Identity Shift Mid-Flow
        if tx.deviceId and upstream_devs and tx.deviceId not in upstream_devs:
            bonus_risk += 0.15
        if tx.ipAddress and upstream_ips and tx.ipAddress not in upstream_ips:
            bonus_risk += 0.12

        # 3. Cross-Component Identity Reuse
        cluster_u = self._get_weakly_connected_component(u)
        cluster_v = self._get_weakly_connected_component(v)
        cluster = cluster_u | cluster_v
        
        for attr, index in [(tx.ipAddress, self.ip_to_txs), (tx.deviceId, self.device_to_txs)]:
            if attr and attr in index:
                for ex_tx_id in index[attr]:
                    ex_tx = self.active_txs[ex_tx_id]
                    if ex_tx.fromUserId not in cluster and ex_tx.toUserId not in cluster:
                        bonus_risk += 0.18
                        break

        # 4. Consistent Identity Alignment
        if tx.deviceId and tx.deviceId in upstream_devs:
            mod_multiplier *= 1.08
        if tx.ipAddress and tx.ipAddress in upstream_ips:
            mod_multiplier *= 1.05

        return mod_multiplier, bonus_risk

    def _evaluate_value(self, tx: Transaction, lineages: List[List[Transaction]]) -> float:
        """Phase 3 Value Progression & Trajectory Reversal Calibrator."""
        if not lineages:
            return 0.0

        reversal_detected = False
        max_reversal_ratio = 0.0

        for path in lineages:
            if not path: continue
            prev_tx = path[-1]  # Immediate upstream edge
            if tx.amount > prev_tx.amount:
                reversal_detected = True
                ratio = (tx.amount - prev_tx.amount) / prev_tx.amount
                max_reversal_ratio = max(max_reversal_ratio, ratio)

        if reversal_detected:
            # Value Trajectory Reversal Penalty (Ranks Example 3 highest)
            return 0.35 + min(0.15, max_reversal_ratio * 0.10)
            
        # Consistent decay or competing branch decay adds no risk penalty
        return 0.0

    def process_transaction(self, tx: Transaction) -> float:
        ts = self.parse_timestamp(tx.createdAt)
        payload_hash = self._hash_payload(tx)
        
        # Idempotency handling
        if tx.txId in self.tx_cache:
            cached_hash, cached_score = self.tx_cache[tx.txId]
            return cached_score

        # Advance stream clock & prune 24h lookback
        if ts > self.t_max:
            self.t_max = ts
        self.prune(ts)

        # Multi-hop lineage reconstruction
        lineages = self._get_upstream_paths(tx.fromUserId, ts)

        # Multi-signal composite calculation
        s_struct = self._evaluate_topology(tx.fromUserId, tx.toUserId)
        s_id_mult, s_id_bonus = self._evaluate_identity(tx, lineages)
        s_val_bonus = self._evaluate_value(tx, lineages)
        
        raw_score = (s_struct * s_id_mult) + s_id_bonus + s_val_bonus
        final_score = round(min(1.0, max(0.0, raw_score)), 4)

        # Update streaming graph state
        self.active_txs[tx.txId] = tx
        self.tx_timestamps[tx.txId] = ts
        heapq.heappush(self.edge_heap, (ts, tx.txId))
        
        self.adj[tx.fromUserId][tx.toUserId].append(tx.txId)
        self.rev_adj[tx.toUserId][tx.fromUserId].append(tx.txId)
        
        if tx.ipAddress:
            self.ip_to_txs[tx.ipAddress].add(tx.txId)
        if tx.deviceId:
            self.device_to_txs[tx.deviceId].add(tx.txId)
            
        self.tx_cache[tx.txId] = (payload_hash, final_score)
        return final_score

# Singleton Instance
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