import time
import heapq
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
        
    curr_t = start_t
    rem_d = float(base_dur)
    
    while rem_d > 1e-6:
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
            continue
            
        max_d = speed * (next_t - curr_t)
        if rem_d <= max_d:
            return curr_t + (rem_d / speed)
            
        rem_d -= max_d
        curr_t = next_t
        
    return curr_t

@router.post("/kan-cheong-delivery-driver")
def solve_delivery_driver(batch: Dict[str, Any]) -> Dict[str, Any]:
    global_start = time.time()
    responses = {}
    
    for case_id, case in batch.items():
        if time.time() - global_start > 8.8:
            responses[case_id] = {"total_duration_sec": None, "arrival_time": None, "path": []}
            continue
            
        start_coord = tuple(case["start_coordinate"])
        end_coord = tuple(case["end_coordinate"])
        start_time = parse_iso(case["start_time"])
        
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
        
        # Priority Queue: (est_total, curr_t, current_node, linked_path_node)
        pq = [(start_time + static_dists[start_coord]/global_max_speed, start_time, start_coord, None)]
        best_res = None
        
        while pq:
            if time.time() - global_start > 8.8:
                break
                
            est_total, curr_t, u, path_node = heapq.heappop(pq)
            
            if u == end_coord:
                best_res = (curr_t, path_node)
                break
                
            if curr_t >= T_max:
                if u in static_visited: continue
                static_visited.add(u)
            else:
                state = (u, round(curr_t, 1)) 
                if state in visited_states: continue
                visited_states.add(state)
                
            for v, eid, base_dur in graph[u]:
                obs_list = obs_dict.get((eid, u, v), [])
                t_next = compute_traversal(base_dur, curr_t, obs_list)
                
                if t_next is not None and v in static_dists:
                    est_t = t_next + (static_dists[v] / global_max_speed)
                    heapq.heappush(pq, (est_t, t_next, v, (eid, path_node)))
                    
        if best_res:
            arr_t, final_path_node = best_res
            
            # Unwind the linked path memory struct
            path_res = []
            curr_node = final_path_node
            while curr_node is not None:
                path_res.append(curr_node[0])
                curr_node = curr_node[1]
            path_res.reverse()
            
            responses[case_id] = {
                "total_duration_sec": int(round(arr_t - start_time)),
                "arrival_time": to_iso(arr_t),
                "path": path_res
            }
        else:
            responses[case_id] = {"total_duration_sec": None, "arrival_time": None, "path": []}
            
    return responses