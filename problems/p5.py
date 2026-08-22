from collections import deque
from typing import List, Optional, Set, Dict

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models (Unchanged)
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
# Stateful Streaming Engine
# ---------------------------------------------------------------------------

class TxNode:
    """Wrapper to store parsed timestamp alongside a Transaction."""
    __slots__ = ('tx', 'timestamp')
    def __init__(self, tx: Transaction, ts: float):
        self.tx = tx
        self.timestamp = ts


class RiskEngine:
    def __init__(self):
        self.WINDOW_SECONDS = 24 * 60 * 60
        self.MAX_DEPTH = 6  # Reduced slightly to ensure tight bounds under load
        
        # O(1) Idempotency and result caching
        self.tx_cache: Dict[str, float] = {}
        
        # Chronological queue for O(1) pruning
        self.edges_queue: deque = deque() 
        
        # Bi-directional graph storing full transaction data for future phases
        self.graph: Dict[str, Dict[str, List[Transaction]]] = {}
        self.reverse_graph: Dict[str, Dict[str, List[Transaction]]] = {}
        
        # O(1) Node existence tracking (reference counting)
        self.active_nodes: Dict[str, int] = {}
        self.t_max: float = 0.0

    def parse_timestamp(self, iso_str: str) -> float:
        import datetime
        s = iso_str.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(s).timestamp()

    def clear(self):
        self.__init__()

    def _add_node_ref(self, node: str):
        self.active_nodes[node] = self.active_nodes.get(node, 0) + 1

    def _remove_node_ref(self, node: str):
        self.active_nodes[node] -= 1
        if self.active_nodes[node] <= 0:
            del self.active_nodes[node]

    def prune(self):
        """O(1) amortized pruning using the chronologically ordered deque."""
        cutoff = self.t_max - self.WINDOW_SECONDS
        while self.edges_queue and self.edges_queue[0].timestamp < cutoff:
            old_tx = self.edges_queue.popleft()
            u, v = old_tx.tx.fromUserId, old_tx.tx.toUserId
            
            # Remove from graph
            if u in self.graph and v in self.graph[u]:
                self.graph[u][v] = [t for t in self.graph[u][v] if t.txId != old_tx.tx.txId]
                if not self.graph[u][v]:
                    del self.graph[u][v]
                    if not self.graph[u]:
                        del self.graph[u]

            # Remove from reverse_graph
            if v in self.reverse_graph and u in self.reverse_graph[v]:
                self.reverse_graph[v][u] = [t for t in self.reverse_graph[v][u] if t.txId != old_tx.tx.txId]
                if not self.reverse_graph[v][u]:
                    del self.reverse_graph[v][u]
                    if not self.reverse_graph[v]:
                        del self.reverse_graph[v]
                    
            # Decrement node reference counts
            self._remove_node_ref(u)
            self._remove_node_ref(v)

    def has_path(self, src: str, dst: str, allow_direct: bool = True) -> bool:
        if src not in self.graph:
            return False
        
        visited = {src}
        queue = deque([(src, 0)])
        
        while queue:
            node, depth = queue.popleft()
            if depth >= self.MAX_DEPTH:
                continue
                
            for neighbor in self.graph.get(node, {}):
                if neighbor == dst:
                    if allow_direct or node != src:
                        return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return False

    def get_ancestors(self, node: str) -> Set[str]:
        if node not in self.reverse_graph:
            return set()
        visited = set()
        queue = deque([(node, 0)])
        while queue:
            curr, depth = queue.popleft()
            if depth >= self.MAX_DEPTH:
                continue
            for parent in self.reverse_graph.get(curr, {}):
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth + 1))
        return visited

    def score_transaction(self, tx: Transaction) -> float:
        u, v = tx.fromUserId, tx.toUserId

        # 1. Self-Loop
        if u == v:
            return 0.7

        # 2. Return / Multi-Loop (Does v reach u?)
        # If adding u->v creates a cycle, it means v already reaches u
        if self.has_path(v, u):
            # Check for multi-loop: does v reach u in multiple ways, or are they already in cycles?
            # A simple heuristic for Multi-Loop: if u already reaches v (forming a tight cycle before this tx) 
            # OR if v has multiple outgoing branches that can reach u.
            if self.has_path(u, v): 
                return 0.9
            return 0.7

        u_active = u in self.active_nodes
        v_active = v in self.active_nodes

        # 3. Convergence
        if u_active and v_active:
            # Case A: Direct bypass (u already reaches v via another path)
            if self.has_path(u, v):
                return 0.5
            
            # Case B: Shared upstream ancestor
            u_ancestors = self.get_ancestors(u)
            v_ancestors = self.get_ancestors(v)
            if u_ancestors & v_ancestors:
                return 0.5

        # 4. Extension
        if u_active or v_active:
            return 0.3

        # 5. Isolated
        return 0.1

    def process(self, tx: Transaction) -> float:
        # Idempotency
        if tx.txId in self.tx_cache:
            return self.tx_cache[tx.txId]

        ts = self.parse_timestamp(tx.createdAt)

        # Stale transaction protection (older than window)
        if self.t_max > 0 and ts < self.t_max - self.WINDOW_SECONDS:
            self.tx_cache[tx.txId] = 0.1
            return 0.1

        # Time advancement
        if ts > self.t_max:
            self.t_max = ts
            self.prune()

        # Score BEFORE applying state change
        score = self.score_transaction(tx)

        # Apply state change
        node = TxNode(tx, ts)
        self.edges_queue.append(node)
        
        self.graph.setdefault(tx.fromUserId, {}).setdefault(tx.toUserId, []).append(tx)
        self.reverse_graph.setdefault(tx.toUserId, {}).setdefault(tx.fromUserId, []).append(tx)
        
        self._add_node_ref(tx.fromUserId)
        self._add_node_ref(tx.toUserId)

        self.tx_cache[tx.txId] = score
        return score

# Global instance
engine = RiskEngine()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/ghost-chains/health")
def ghost_chains_health() -> HealthResponse:
    return HealthResponse(status="ok")

@router.post("/ghost-chains/reset")
def ghost_chains_reset(request: ResetRequest = ResetRequest()) -> ResetResponse:
    if request.clearTransactions:
        engine.clear()
    return ResetResponse(clearTransactions=request.clearTransactions)

@router.post("/ghost-chains/transactions")
def ghost_chains_transactions(request: TransactionBatchRequest) -> TransactionBatchResponse:
    results = []
    for tx in request.transactions:
        score = engine.process(tx)
        results.append(TransactionResult(txId=tx.txId, riskScore=round(score, 1)))
    return TransactionBatchResponse(transactions=results)