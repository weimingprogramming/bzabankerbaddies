import bisect
import heapq
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import RootModel

router = APIRouter()

# --- Constants ---

NEG_INF = float("-inf")
POS_INF = float("inf")
EPS = 1e-6
UNREACHABLE = {"total_duration_sec": None, "arrival_time": None, "path": []}
MAX_STATES = 200_000

# --- Pydantic Models ---

class DeliveryBatch(RootModel[Dict[str, Any]]):
    pass

# --- Helper Functions ---

def parse_iso(s: str) -> float:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def format_iso(t: float) -> str:
    t_rounded = round(t, 3)
    dt = datetime.fromtimestamp(t_rounded, tz=timezone.utc)
    if dt.microsecond == 0:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"

def build_timeline(obs_list):
    """Calculates active speed factors over time. Overlaps combine via min()."""
    if not obs_list:
        return None
    pts = sorted(set([s for s, e, f in obs_list] + [e for s, e, f in obs_list]))
    bounds = [NEG_INF] + pts + [POS_INF]
    rates = [1.0]
    for i in range(len(pts) - 1):
        lo = pts[i]
        active = [f for (s, e, f) in obs_list if s <= lo < e]
        rates.append(min(active) if active else 1.0)
    rates.append(1.0)
    return bounds, rates

def traverse(timeline, duration, entry_time):
    """Simulate traversing an edge with piecewise speed factors."""
    if duration <= 0:
        return entry_time
    if timeline is None:
        return entry_time + duration

    bounds, rates = timeline
    i = max(0, min(bisect.bisect_right(bounds, entry_time) - 1, len(rates) - 1))
    
    remaining = duration
    cur = entry_time
    n = len(rates)
    
    while True:
        if remaining <= EPS:
            return cur
        rate = rates[i]
        if rate <= 0:
            return None  # Waiting not allowed, edge impassable
        
        seg_end = bounds[i + 1]
        if seg_end == POS_INF:
            return cur + remaining / rate
            
        avail = seg_end - cur
        progress = rate * avail
        
        if progress >= remaining - EPS:
            return cur + remaining / rate
            
        remaining -= progress
        cur = seg_end
        i += 1
        if i >= n:
            return None

def static_dijkstra(adj, source):
    """Calculates the static shortest distance from all reachable nodes to `source`."""
    dist = {source: 0.0}
    prev = {}
    pq = [(0.0, source)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, POS_INF):
            continue
            
        for (v, eid, w) in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, POS_INF) - EPS:
                dist[v] = nd
                prev[v] = (u, eid)
                heapq.heappush(pq, (nd, v))

    def get_path(node):
        if node not in dist:
            return None
        path = []
        cur = node
        while cur != source:
            pu, eid = prev[cur]
            path.append(eid)
            cur = pu
        return path

    return dist, get_path

def solve_case(case: dict) -> dict:
    try:
        start = tuple(case["start_coordinate"])
        end = tuple(case["end_coordinate"])
        t0 = parse_iso(case["start_time"])
        edges = case.get("edges", [])
        obstructions = case.get("obstructions", [])

        directed_from: dict = {}
        edge_lookup: dict = {}
        static_adj: dict = {}

        for e in edges:
            n1, n2 = tuple(e["node1"]), tuple(e["node2"])
            eid, dur = e["edge_id"], e["base_duration_sec"]
            for (a, b) in ((n1, n2), (n2, n1)):
                rec = {"to": b, "edge_id": eid, "dur": dur, "obs": [], "timeline": None}
                directed_from.setdefault(a, []).append(rec)
                edge_lookup[(eid, a, b)] = rec
            
            # Static adjacency for heuristic/fallback (traversing backwards from end)
            static_adj.setdefault(n1, []).append((n2, eid, dur))
            static_adj.setdefault(n2, []).append((n1, eid, dur))

        T_max = t0
        for obs in obstructions:
            eid = obs["edge_id"]
            f, tt = tuple(obs["edge"]["from"]), tuple(obs["edge"]["to"])
            rec = edge_lookup.get((eid, f, tt))
            if not rec:
                continue
            s, e_ = parse_iso(obs["start_time"]), parse_iso(obs["end_time"])
            rec["obs"].append((s, e_, float(obs["speed_factor"])))
            T_max = max(T_max, e_)

        for lst in directed_from.values():
            for rec in lst:
                if rec["obs"]:
                    rec["timeline"] = build_timeline(rec["obs"])

        static_dist, get_path = static_dijkstra(static_adj, end)

        if start == end:
            return {"total_duration_sec": 0, "arrival_time": format_iso(t0), "path": []}

        # --- Corrected Time-Dependent Dijkstra ---
        dist = {start: t0}
        came_from = {start: None}
        counter = 0
        heap = [(t0, counter, start)]

        while heap:
            t, _, node = heapq.heappop(heap)

            # Prune obsolete states strictly 
            if t > dist.get(node, POS_INF) + EPS:
                continue

            # Goal reached
            if node == end:
                path = []
                cur = end
                while cur != start:
                    info = came_from.get(cur)
                    if not info: break
                    prev_node, edge_info = info
                    
                    if isinstance(edge_info, tuple) and edge_info[0] == "STATIC":
                        jump_node = edge_info[1]
                        sp = get_path(jump_node) or []
                        path.extend(reversed(sp))
                        cur = jump_node
                    else:
                        path.append(edge_info)
                        cur = prev_node
                
                path.reverse()
                total = t - t0
                total_out = int(round(total)) if abs(total - round(total)) < 1e-6 else round(total, 6)
                return {"total_duration_sec": total_out, "arrival_time": format_iso(t), "path": path}

            # Optimization: If we are past T_max, timeline is normal. Jump directly to end.
            if t >= T_max - EPS:
                sd = static_dist.get(node)
                if sd is not None and sd < POS_INF:
                    arrival = t + sd
                    if arrival < dist.get(end, POS_INF) - EPS:
                        dist[end] = arrival
                        came_from[end] = (node, ("STATIC", node))
                        counter += 1
                        heapq.heappush(heap, (arrival, counter, end))
                continue

            # Standard Edge Relaxation
            for rec in directed_from.get(node, []):
                exit_t = traverse(rec["timeline"], rec["dur"], t)
                if exit_t is None:
                    continue
                
                # Only explore if we found a strictly faster route to the next node
                if exit_t < dist.get(rec["to"], POS_INF) - EPS:
                    dist[rec["to"]] = exit_t
                    came_from[rec["to"]] = (node, rec["edge_id"])
                    counter += 1
                    heapq.heappush(heap, (exit_t, counter, rec["to"]))

        return dict(UNREACHABLE)
    except Exception as e:
        return dict(UNREACHABLE)

# --- Routes ---

@router.post("/kan-cheong-delivery-driver")
def kan_cheong_delivery_driver(request: DeliveryBatch) -> Dict[str, Any]:
    return {case_id: solve_case(case) for case_id, case in request.root.items()}
