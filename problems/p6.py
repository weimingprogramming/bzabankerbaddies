from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import copy

router = APIRouter()

class TestCase(BaseModel):
    energy: int
    capital: int
    timeline: Dict[str, Dict[str, Dict[str, int]]]

@router.post("/stonks")
def solve_stonks(test_cases: List[TestCase]):
    results = []
    
    for case in test_cases:
        energy = case.energy
        capital = case.capital
        timeline = case.timeline
        
        # Parse timeline into a more accessible format
        years = sorted([int(y) for y in timeline.keys()])
        
        # Best tracking
        best_capital = -1
        best_actions = []

        # DFS State parameters:
        # current_year, energy_left, capital, inventory, timeline_state, current_actions
        def dfs(curr_year: int, energy_left: int, cap: int, inv: Dict[str, int], t_state: Dict[int, Dict[str, Dict[str, int]]], path: List[str]):
            nonlocal best_capital, best_actions
            
            # If we are at 2037, we can record this state as a potential best
            if curr_year == 2037:
                if cap > best_capital:
                    best_capital = cap
                    best_actions = list(path)
            
            # Action 1: Sell stocks we currently hold in the current year
            if curr_year in t_state:
                market = t_state[curr_year]
                for stock, amount in inv.items():
                    if amount > 0 and stock in market:
                        price = market[stock]['price']
                        # We sell all we have for simplicity in this greedy approach
                        sell_val = price * amount
                        
                        new_inv = dict(inv)
                        new_inv[stock] = 0
                        
                        path.append(f"s-{stock}-{amount}")
                        dfs(curr_year, energy_left, cap + sell_val, new_inv, t_state, path)
                        path.pop()
            
            # Action 2: Buy stocks available in the current year
            if curr_year in t_state:
                market = t_state[curr_year]
                for stock, data in market.items():
                    price = data['price']
                    qty_avail = data['qty']
                    
                    if qty_avail > 0 and cap >= price:
                        # Buy as much as we can afford, up to qty_avail
                        max_can_buy = min(qty_avail, cap // price)
                        cost = max_can_buy * price
                        
                        new_t_state = copy.deepcopy(t_state)
                        new_t_state[curr_year][stock]['qty'] -= max_can_buy
                        
                        new_inv = dict(inv)
                        new_inv[stock] = new_inv.get(stock, 0) + max_can_buy
                        
                        path.append(f"b-{stock}-{max_can_buy}")
                        dfs(curr_year, energy_left, cap - cost, new_inv, new_t_state, path)
                        path.pop()

            # Action 3: Jump to another year
            for target_year in years:
                if target_year != curr_year:
                    cost = abs(target_year - curr_year)
                    if cost <= energy_left:
                        path.append(f"j-{curr_year}-{target_year}")
                        dfs(target_year, energy_left - cost, cap, inv, t_state, path)
                        path.pop()

        # Format timeline state with integer keys for easier processing
        initial_t_state = {int(y): data for y, data in timeline.items()}
        dfs(2037, energy, capital, {}, initial_t_state, [])
        
        results.append(best_actions)
        
    return results