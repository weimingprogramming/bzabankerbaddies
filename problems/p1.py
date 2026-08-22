import time
import heapq
from datetime import datetime, timezone
from typing import Dict, Any
from collections import defaultdict
from fastapi import APIRouter

router = APIRouter()

def parse_iso(iso_str: str) -> float:
    """Converts ISO-8601 string to a Unix timestamp."""
    iso_str = iso_str.replace('Z', '+00:00')
    return datetime.fromisoformat(iso_str).timestamp()

def to_iso(timestamp: float) -> str:
    """Converts Unix timestamp back to ISO-8601 string."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

def compute_traversal(base_dur: int, start_t: float, obs_list: list) -> float:
    """Calculates the time it takes to cross an edge, accounting for speed changes."""
    if not obs_list:
        return start_t + base_dur
        
    curr_t = start_t
    d = float(base_dur)
    
    speed = 1.0
    for st, et, sf in obs_list:
        if st <= curr_t < et:
            speed *= sf
    if speed == 0.0:
        return None
        
    while d > 1e-7:
        speed = 1.0
        next_event_t = float('inf')
        
        for st, et, sf in obs_list:
            if st <= curr_t < et:
                speed *= sf
                if et < next_event_t:
                    next_event_t = et
            elif curr_t < st:
                if st < next_event_t:
                    next_event_t = st
                    
        if speed == 0.0:
            if next_event_t == float('inf'):
                return None  # Blocked forever
            curr_t = next_event_t
        else:
            max_d = speed * (next_event_t - curr_t)
            if d <= max_d:
                curr_t += d / speed
                d = 0
            else:
                d -= max_d
                curr_t = next_event_t
                
    return curr_t

@router.post("/kan-cheong-delivery-driver")
def solve_delivery_driver(batch: Dict[str, Any]) -> Dict[str, Any]:
    global_start = time.time()
    responses = {}
    
    for case_id, case in batch.items():
        # STRICT TIMEOUT: Abort at 8.5s to ensure response transmits safely
        if time.time() - global_start > 8.5:
            responses[case_id] = {"total_duration_sec": None, "arrival_time": None, "path": []}
            continue
            
        start_coord = tuple(case["start_coordinate"])
        end_coord = tuple(case["end_coordinate"])
        start_time = parse_iso(case["start_time"])
        
        # Build the graph
        graph = defaultdict(list)
        for e in case.get("edges", []):
            u = tuple(e["node1"])
            v = tuple(e["node2"])
            eid = e["edge_id"]
            dur = e["base_duration_sec"]
            graph[u].append((v, eid, dur))
            graph[v].append((u, eid, dur))
            
        # Map out all obstructions by their directed edge
        obs_dict = defaultdict(list)
        T_max = start_time
        for o in case.get("obstructions", []):
            eid = o["edge_id"]
            u = tuple(o["edge"]["from"])
            v = tuple(o["edge"]["to"])
            st = parse_iso(o["start_time"])
            et = parse_iso(o["end_time"])
            sf = o["speed_factor"]
            obs_dict[(eid, u, v)].append((st, et, sf))
            if et > T_max:
                T_max = et

        # --- A* ENHANCEMENT: Precompute static shortest paths (Reverse Dijkstra) ---
        static_dists = {}
        rev_pq = [(0, end_coord)]
        while rev_pq:
            d, u = heapq.heappop(rev_pq)
            if u in static_dists:
                continue
            static_dists[u] = d
            for v, eid, dur in graph[u]:
                if v not in static_dists:
                    heapq.heappush(rev_pq, (d + dur, v))
                    
        # If the destination is completely unreachable even without traffic, skip early
        if start_coord not in static_dists:
            responses[case_id] = {"total_duration_sec": None, "arrival_time": None, "path": []}
            continue
                
        # --- Time-Dependent A* Search ---
        static_visited = set()
        visited_states = set()
        
        # Priority Queue stores: (estimated_total_time, current_time, current_node, path_so_far)
        pq = [(start_time + static_dists[start_coord], start_time, start_coord, ())]
        best_res = None
        
        while pq:
            if time.time() - global_start > 8.5:
                break
                
            est_total, curr_t, u, path = heapq.heappop(pq)
            
            if u == end_coord:
                best_res = (curr_t, path)
                break
                
            # If current time is past all obstructions, graph behaves statically
            if curr_t >= T_max:
                if u in static_visited:
                    continue
                static_visited.add(u)
            else:
                # Track exact state to prevent infinite loops (round to 2 decimals to prevent float jitter)
                state = (u, round(curr_t, 2)) 
                if state in visited_states:
                    continue
                visited_states.add(state)
                
            for v, eid, base_dur in graph[u]:
                obs_list = obs_dict.get((eid, u, v), [])
                t_next = compute_traversal(base_dur, curr_t, obs_list)
                
                if t_next is not None and v in static_dists:
                    est_t = t_next + static_dists[v]
                    heapq.heappush(pq, (est_t, t_next, v, path + (eid,)))
                    
        if best_res:
            arr_t, path_res = best_res
            responses[case_id] = {
                "total_duration_sec": int(round(arr_t - start_time)),
                "arrival_time": to_iso(arr_t),
                "path": list(path_res)
            }
        else:
            responses[case_id] = {
                "total_duration_sec": None,
                "arrival_time": None,
                "path": []
            }
            
    return responses