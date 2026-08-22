import collections
import datetime
import heapq
from typing import List, Optional, Set, Dict, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models
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

class RiskEngine:
    def __init__(self):
        self.WINDOW_SECONDS = 24 * 60 * 60
        self.t_max: float = 0.0
        
        # Idempotency cache: txId -> (payload_string, risk_score)
        self.tx_cache: Dict[str, Tuple[str, float]] = {}
        
        # Active edges state for O(1) lookups: txId -> (u, v, timestamp)
        self.active_edges: Dict[str, Tuple[str, str, float]] = {}
        
        # Min-Heap for time-based pruning (handles out-of-order arrivals perfectly)
        self.edge_heap: List[Tuple[float, str]] = []
        
        # Multi-graph adjacency lists (tracking edge counts between nodes)
        self.adj: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self.rev_adj: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    def clear(self):
        self.__init__()

    def parse_timestamp(self, iso_str: str) -> float:
        s = iso_str.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(s).timestamp()

    def _remove_edge_from_graph(self, u: str, v: str):
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

    def prune(self):
        """Strict event-time pruning. Removes edges older than or exactly equal to 24h ago."""
        cutoff = self.t_max - self.WINDOW_SECONDS
        
        # Strict pruning: edges older than exactly 24h are removed
        while self.edge_heap and self.edge_heap[0][0] < cutoff:
            ts, txId = heapq.heappop(self.edge_heap)
            
            # Verify this isn't a stale heap entry from a modified payload
            if txId in self.active_edges and self.active_edges[txId][2] == ts:
                u, v, _ = self.active_edges.pop(txId)
                self._remove_edge_from_graph(u, v)

    def has_path(self, src: str, dst: str) -> bool:
        """Standard BFS to detect reachability. Detects if a node is in a cycle if src == dst."""
        if src not in self.adj: 
            return False

        visited = {src}
        queue = collections.deque([src])
        
        while queue:
            curr = queue.popleft()
            for nxt in self.adj.get(curr, {}):
                if nxt == dst:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def get_ancestors(self, node: str) -> Set[str]:
        """Returns all nodes that can reach the given node."""
        if node not in self.rev_adj: return set()
        
        visited = set()
        queue = collections.deque([node])
        
        while queue:
            curr = queue.popleft()
            for parent in self.rev_adj.get(curr, {}):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return visited

    def has_indirect_path(self, src: str, dst: str) -> bool:
        """Checks if src can reach dst ignoring the immediate direct edge (if it exists)."""
        if src not in self.adj: return False
        visited = {src}
        queue = collections.deque([src])
        
        while queue:
            curr = queue.popleft()
            for nxt in self.adj.get(curr, {}):
                if curr == src and nxt == dst:
                    continue  # Skip direct path
                if nxt == dst:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def score_transaction(self, u: str, v: str) -> float:
        # 1. Cycle Detection (Return or Multi-Loop)
        creates_cycle = self.has_path(v, u) or (u == v)
        
        if creates_cycle:
            # Multi-Loop: The new edge completes a cycle, AND one of the endpoints
            # is ALREADY part of an existing independent cycle.
            if self.has_path(v, v) or self.has_path(u, u):
                return 0.9
            return 0.7

        # 2. Convergence Detection
        u_ancestors = self.get_ancestors(u)
        v_ancestors = self.get_ancestors(v)
        
        # Convergence requires either a shared upstream ancestor, OR 
        # a structural indirect path (meaning this new edge creates a 2nd path).
        # A simple repeated direct edge will fail this and fall through to Extension.
        if (u_ancestors & v_ancestors) or self.has_indirect_path(u, v):
            return 0.5

        # 3. Extension vs Isolated
        u_active = (u in self.adj) or (u in self.rev_adj)
        v_active = (v in self.adj) or (v in self.rev_adj)
        
        if u_active or v_active:
            return 0.3
            
        return 0.1

    def process(self, tx: Transaction) -> float:
        # 1. Idempotency & Payload update check
        payload_str = tx.model_dump_json(exclude_unset=True)
        if tx.txId in self.tx_cache:
            cached_payload, cached_score = self.tx_cache[tx.txId]
            if cached_payload == payload_str:
                return cached_score
            
            # If payload differs, treat as update: remove old edge state first
            if tx.txId in self.active_edges:
                old_u, old_v, _ = self.active_edges.pop(tx.txId)
                self._remove_edge_from_graph(old_u, old_v)

        ts = self.parse_timestamp(tx.createdAt)

        # 2. Update time and prune BEFORE scoring
        if ts > self.t_max:
            self.t_max = ts
        
        self.prune()

        # 3. Handle expired transactions (strictly older than window)
        if self.t_max > 0 and ts < self.t_max - self.WINDOW_SECONDS:
            self.tx_cache[tx.txId] = (payload_str, 0.1)
            return 0.1

        # 4. Score based purely on prior state
        u, v = tx.fromUserId, tx.toUserId
        score = self.score_transaction(u, v)

        # 5. Commit to graph state
        self.active_edges[tx.txId] = (u, v, ts)
        heapq.heappush(self.edge_heap, (ts, tx.txId))
        self.adj[u][v] += 1
        self.rev_adj[v][u] += 1
        
        self.tx_cache[tx.txId] = (payload_str, score)
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