from typing import List, Dict, Any
import collections
import datetime
import hashlib
import heapq
import json
from typing import List, Optional, Set, Dict, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()

def get_q_candidates(max_can_buy, total_items):
    """
    Branching heuristic to prevent massive time complexity on large capital/quantities.
    Checks maximum affordable, slightly less than max, half, and zero.
    """
    if max_can_buy == 0:
        return [0]
    
    # If there are very few choices, checking every quantity is safe and highly optimal
    if total_items <= 2 and max_can_buy <= 50:
        return list(range(max_can_buy, -1, -1))
    if max_can_buy <= 3:
        return list(range(max_can_buy, -1, -1))
        
    return list(sorted(set([max_can_buy, max_can_buy - 1, max_can_buy // 2, 0]), reverse=True))

def _clean_state(inv, tl_state):
    """Canonical, hashable form of (inventory, future timeline) for memoization."""
    clean_inv = tuple(sorted((s, q) for s, q in inv.items() if q > 0))
    clean_tl_list = []
    for y in sorted(tl_state.keys()):
        clean_stocks = tuple(sorted((s, q) for s, q in tl_state[y].items() if q > 0))
        if clean_stocks:
            clean_tl_list.append((y, clean_stocks))
    clean_tl = tuple(clean_tl_list)
    return clean_inv, clean_tl


def solve_single(test_case: Dict[str, Any]) -> List[str]:
    energy_limit = test_case['energy']
    initial_capital = test_case['capital']
    timeline = test_case['timeline']

    prices = {}
    init_tl = {}
    for y, stocks in timeline.items():
        prices[y] = {}
        init_tl[y] = {}
        for s, data in stocks.items():
            prices[y][s] = data['price']
            init_tl[y][s] = data['qty']

    memo = {}

    def dfs_step(curr_year, energy, cap, inv, tl_state):
        # Generator form of the search: rather than recursing directly on each
        # jump, it yields the args of each sub-search and resumes with the
        # result via generator.send(). run_dfs() below drives this with an
        # explicit stack so recursion depth doesn't scale with `energy`
        # (which caused RecursionError -> 500 on large-energy test cases).
        clean_inv, _ = _clean_state(inv, tl_state)

        best_cap = -1
        best_acts = []
        
        # 2. Base Option: Stop here if we are safely in 2037
        if curr_year == "2037":
            final_cap = cap
            sell_acts = []
            for s, q in clean_inv:
                final_cap += prices[curr_year][s] * q
                sell_acts.append(f"s-{s}-{q}")
            best_cap = final_cap
            best_acts = sell_acts
            
        # 3. Sell Phase Setup
        inv_items = list(clean_inv)
        sell_combinations = []
        def gen_sells(idx, current_sells):
            if idx == len(inv_items):
                sell_combinations.append(dict(current_sells))
                return
            s, q = inv_items[idx]
            gen_sells(idx + 1, current_sells) # Branch A: Hold (Sell 0)
            
            current_sells[s] = q
            gen_sells(idx + 1, current_sells) # Branch B: Sell All
            del current_sells[s]
            
        gen_sells(0, {})
        
        for sells in sell_combinations:
            temp_cap = cap
            temp_inv = dict(inv)
            sell_acts_step = []
            for s, q in sells.items():
                temp_cap += prices[curr_year][s] * q
                temp_inv[s] -= q
                if temp_inv[s] == 0:
                    del temp_inv[s]
                sell_acts_step.append(f"s-{s}-{q}")
                
            # 4. Buy Phase Setup (Find items mathematically viable to buy)
            profitable = []
            for s, q_left in tl_state.get(curr_year, {}).items():
                if q_left > 0:
                    max_future_p = 0
                    for fut_y in tl_state.keys():
                        if fut_y == curr_year: continue
                        cost_to_fut = abs(int(curr_year) - int(fut_y))
                        cost_fut_to_2037 = abs(int(fut_y) - 2037)
                        
                        if energy >= cost_to_fut + cost_fut_to_2037:
                            if s in prices[fut_y]:
                                max_future_p = max(max_future_p, prices[fut_y][s])
                    
                    if max_future_p > prices[curr_year][s]:
                        profitable.append((s, prices[curr_year][s], q_left))
                        
            buy_combinations = []
            def gen_buys(idx, current_cap, current_buys):
                if idx == len(profitable):
                    buy_combinations.append((current_cap, dict(current_buys)))
                    return
                s, p, max_avail = profitable[idx]
                max_can_buy = min(max_avail, current_cap // p)
                
                # Using the heuristic boundary array to avoid integer combination explosions
                for q in get_q_candidates(max_can_buy, len(profitable)):
                    if q > 0:
                        current_buys[s] = q
                        gen_buys(idx + 1, current_cap - q * p, current_buys)
                        del current_buys[s]
                    else:
                        gen_buys(idx + 1, current_cap, current_buys)
                        
            gen_buys(0, temp_cap, {})
            
            # 5. Execute Jumps across valid branches
            for next_cap, buys in buy_combinations:
                temp_tl = {y: dict(stocks) for y, stocks in tl_state.items()}
                next_inv = dict(temp_inv)
                buy_acts_step = []
                for s, q in buys.items():
                    next_inv[s] = next_inv.get(s, 0) + q
                    temp_tl[curr_year][s] -= q
                    buy_acts_step.append(f"b-{s}-{q}")
                    
                for next_year in tl_state.keys():
                    if next_year == curr_year:
                        continue
                        
                    cost = abs(int(curr_year) - int(next_year))
                    cost_ret = abs(int(next_year) - 2037)
                    
                    # Ensure jump validity (can we eventually make it back home?)
                    if energy >= cost + cost_ret:
                        res_cap, res_acts = yield (next_year, energy - cost, next_cap, next_inv, temp_tl)
                        if res_cap > best_cap:
                            best_cap = res_cap
                            jump_act = f"j-{curr_year}-{next_year}"
                            best_acts = sell_acts_step + buy_acts_step + [jump_act] + res_acts

        return best_cap, best_acts

    def run_dfs(curr_year, energy, cap, inv, tl_state):
        # Explicit-stack trampoline driving dfs_step generators, so search
        # depth is bounded by heap memory rather than Python's call stack.
        clean_inv, clean_tl = _clean_state(inv, tl_state)
        root_key = (curr_year, energy, cap, clean_inv, clean_tl)
        if root_key in memo:
            return memo[root_key]

        stack = [(dfs_step(curr_year, energy, cap, inv, tl_state), root_key)]
        send_val = None

        while stack:
            gen, key = stack[-1]
            try:
                req = gen.send(send_val)
            except StopIteration as done:
                result = done.value
                memo[key] = result
                stack.pop()
                send_val = result
                continue

            req_clean_inv, req_clean_tl = _clean_state(req[3], req[4])
            req_key = (req[0], req[1], req[2], req_clean_inv, req_clean_tl)
            if req_key in memo:
                send_val = memo[req_key]
            else:
                stack.append((dfs_step(*req), req_key))
                send_val = None

        return memo[root_key]

    max_cap, final_acts = run_dfs("2037", energy_limit, initial_capital, {}, init_tl)
    return final_acts

@router.post("/stonks")
def stonks_endpoint(payload: List[Dict[str, Any]]):
    """
    Takes an array of test cases (JSON objects) and processes them sequentially.
    """
    results = []
    for test_case in payload:
        results.append(solve_single(test_case))
    return results