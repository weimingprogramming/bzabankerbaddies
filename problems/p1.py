import time
import heapq
import itertools
from datetime import datetime, timezone
from typing import Dict, Any
from collections import defaultdict
from fastapi import APIRouter

router = APIRouter()

def parse_iso(iso_str: str) -> float:
    iso_str = iso_str.replace('Z', '+00:00')
    return datetime.fromisoformat(iso_str).timestamp()

def to_iso(timestamp: float) -> str:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

def compute_traversal(base_dur: float, start_t: float, obs_list: list) -> float:
    if not obs_list:
        return start_t + base_dur
        
    # Strictly enforce "No waiting at nodes"
    initial_speed = 1.0
    for st, et, sf in obs_list:
        if st <= start_t < et:
            initial_speed *= sf
    if initial_speed == 0.0:
        return None
        
    curr_t = start_t
    rem_d = float(base_dur)
    
    # loop_counter prevents infinite loops from floating point drift on 0-duration edges
    loop_counter = 0 
    while rem_d > 1e-6 and loop_counter < 1000:
        loop_counter += 1
        speed = 1.0
        next_t = float('inf')
        
        for st, et, sf in obs_list:
            if st <= curr_t < et:
                speed *= sf
                if et < next_t:
                    next_t = et
            elif curr_t < st < next_t:
                next_t = st
                
        if speed == 0.0:
            if next_t == float('inf'):
                return None
            curr_t = next_t
        else:
            max_d = speed * (next_t - curr_t)
            if rem_d <= max_d + 1e-6:
                return curr_t + (rem_d / speed)
            rem_d -= max_d
            curr_t = next_t
            
    return curr_t if rem_d <= 1e-6 else None

@router.post("/kan-cheong-delivery-driver")
def solve_delivery_driver(batch: Dict[str, Any]) -> Dict[str, Any]:
    global_start = time.time()
    responses = {}
    
    for case_id, case in batch.items():
        case_start = time.time()
        time_elapsed = time.time() - global_start
        
        # Dynamic Batch Protection
        if time_elapsed > 9.0:
            responses[case_id] = {"total_duration_sec": None, "arrival_time": None, "path": []}
            continue
            
        # Give remaining time fairly, up to 2.5s per heavy case
        case_time_limit = min(2.5, 9.2 - time_elapsed)
            
        start_coord = tuple(case["start_coordinate"])
        end_coord = tuple(case["end_coordinate"])
        start_time = parse_iso(case["start_time"])
        
        # Edge Case: Start and End are exactly the same
        if start_coord == end_coord:
            responses[case_id] = {
                "total_duration_sec": 0,
                "arrival_time": to_iso(start_time),
                "path": []
            }
            continue
        
        graph = defaultdict(list)
        for e in case.get("edges", []):
            u = tuple(e["node1"])
            v = tuple(e["node2"])
            eid = e["edge_id"]
            dur = e["base_duration_sec"]
            graph[u].append((v, eid, dur))
            graph[v].append((u, eid, dur))
            
        obs_dict = defaultdict(list)
        T_max = start_time
        global_max_speed = 1.0
        
        for o in case.get("obstructions", []):
            eid = o["edge_id"]
            u = tuple(o["edge"]["from"])
            v = tuple(o["edge"]["to"])
            st = parse_iso(o["start_time"])
            et = parse_iso(o["end_time"])
            sf = float(o["speed_factor"])
            obs_dict[(eid, u, v)].append((st, et, sf))
            
            if et > T_max: T_max = et
            if sf > global_max_speed: global_max_speed = sf

        # Reverse Dijkstra Heuristic
        static_dists = {}
        rev_pq = [(0, end_coord)]
        while rev_pq:
            d, u = heapq.heappop(rev_pq)
            if u in static_dists: continue
            static_dists[u] = d
            for v, eid, dur in graph[u]:
                if v not in static_dists:
                    heapq.heappush(rev_pq, (d + dur, v))
                    
        if start_coord not in static_dists:
            responses[case_id] = {"total_duration_sec": None, "arrival_time": None, "path": []}
            continue
                
        static_visited = set()
        visited_states = set()
        
        counter = itertools.count()
        pq = [(start_time + static_dists[start_coord]/global_max_speed, start_time, next(counter), start_coord, None)]
        best_res = None
        
        while pq:
            if time.time() - case_start > case_time_limit:
                break
            if time.time() - global_start > 9.0:
                break
                
            est_total, curr_t, _, u, path_node = heapq.heappop(pq)
            
            if u == end_coord:
                best_res = (curr_t, path_node)
                break
                
            if curr_t >= T_max:
                if u in static_visited: continue
                static_visited.add(u)
            else:
                # Snap micro-fluctuations to exact 5-decimal grids to optimize state memory
                state = (u, round(curr_t, 5)) 
                if state in visited_states: continue
                visited_states.add(state)
                
            for v, eid, base_dur in graph[u]:
                obs_list = obs_dict.get((eid, u, v), [])
                t_next = compute_traversal(base_dur, curr_t, obs_list)
                
                if t_next is not None and v in static_dists:
                    est_t = t_next + (static_dists[v] / global_max_speed)
                    heapq.heappush(pq, (est_t, t_next, next(counter), v, (eid, path_node)))
                    
        if best_res:
            arr_t, final_path_node = best_res
            arr_t = round(arr_t)
            
            path_res = []
            curr_node = final_path_node
            while curr_node is not None:
                path_res.append(curr_node[0])
                curr_node = curr_node[1]
            path_res.reverse()
            
            responses[case_id] = {
                "total_duration_sec": int(arr_t - start_time),
                "arrival_time": to_iso(arr_t),
                "path": path_res
            }
        else:
            responses[case_id] = {"total_duration_sec": None, "arrival_time": None, "path": []}
            
    return responses