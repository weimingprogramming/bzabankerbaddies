from typing import Any, Dict, List, Tuple

from fastapi import APIRouter

router = APIRouter()

HOME_YEAR = "2037"


def _clean_state(inv: Dict[str, int], tl_state: Dict[str, Dict[str, int]]) -> Tuple[tuple, tuple]:
    """Canonical, hashable form of (inventory, remaining-timeline) for memoization."""
    clean_inv = tuple(sorted((s, q) for s, q in inv.items() if q > 0))
    clean_tl_list = []
    for y in sorted(tl_state.keys()):
        clean_stocks = tuple(sorted((s, q) for s, q in tl_state[y].items() if q > 0))
        if clean_stocks:
            clean_tl_list.append((y, clean_stocks))
    return clean_inv, tuple(clean_tl_list)


def solve_single(test_case: Dict[str, Any]) -> List[str]:
    energy_limit = test_case["energy"]
    initial_capital = test_case["capital"]
    timeline = test_case["timeline"]

    prices: Dict[str, Dict[str, int]] = {}
    init_tl: Dict[str, Dict[str, int]] = {}
    for y, stocks in timeline.items():
        prices[y] = {}
        init_tl[y] = {}
        for s, data in stocks.items():
            prices[y][s] = data["price"]
            init_tl[y][s] = data["qty"]

    # Home year must always be a valid stop even if no stock trades there.
    if HOME_YEAR not in init_tl:
        prices[HOME_YEAR] = {}
        init_tl[HOME_YEAR] = {}

    years = list(init_tl.keys())
    memo: Dict[tuple, Tuple[int, List[str]]] = {}

    def best_reachable_price(stock: str, curr_year: str, energy: int) -> int:
        """
        Highest price `stock` can be sold at in a year we can both reach from
        curr_year and still return home from, within the remaining energy.
        """
        best = 0
        for fut_y in years:
            if fut_y == curr_year:
                continue
            cost_there = abs(int(curr_year) - int(fut_y))
            cost_home = abs(int(fut_y) - int(HOME_YEAR))
            if energy < cost_there + cost_home:
                continue
            p = prices[fut_y].get(stock)
            if p is not None and p > best:
                best = p
        return best

    def dfs_step(curr_year, energy, cap, inv, tl_state):
        """
        Generator form of the state search: rather than recursing directly on
        each jump, it yields the args of the sub-search it needs and resumes
        with the result via generator.send(). run_dfs() below drives this
        with an explicit stack, so search depth is bounded by heap memory
        rather than Python's call stack (a plain recursive dfs overflows for
        large `energy`, since each jump only costs 1+ energy but consumes one
        stack frame).
        """
        clean_inv, _ = _clean_state(inv, tl_state)

        best_cap = -1
        best_acts: List[str] = []

        # 1. Base option: stop here if we're already home.
        if curr_year == HOME_YEAR:
            final_cap = cap
            sell_acts = []
            for s, q in clean_inv:
                p = prices[HOME_YEAR].get(s)
                if p is not None:
                    final_cap += p * q
                    sell_acts.append(f"s-{s}-{q}")
            best_cap = final_cap
            best_acts = sell_acts

        # 2. Sell phase: for each held stock, try holding vs. selling all of
        # it (only stocks with a listed price this year can be sold here).
        inv_items = list(clean_inv)
        sell_combinations: List[Dict[str, int]] = []

        def gen_sells(idx, current_sells):
            if idx == len(inv_items):
                sell_combinations.append(dict(current_sells))
                return
            s, q = inv_items[idx]
            gen_sells(idx + 1, current_sells)  # Branch A: hold
            if s in prices.get(curr_year, {}):
                current_sells[s] = q
                gen_sells(idx + 1, current_sells)  # Branch B: sell all
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

            # 3. Buy phase: greedily allocate capital across stocks worth
            # buying here (a reachable-and-returnable future year offers a
            # strictly higher price). Every unit of a given stock has the
            # same marginal value (best future price - current price)
            # regardless of how many units are bought, and buying it doesn't
            # change any other stock's value or price - so there are no
            # diminishing returns or cross-item interactions. That makes
            # filling the best value-per-dollar stock first, then the next,
            # until capital or stock runs out, an exact optimal allocation
            # for this visit (not just a heuristic).
            scored = []
            for s, q_left in tl_state.get(curr_year, {}).items():
                if q_left <= 0:
                    continue
                price = prices[curr_year][s]
                value = best_reachable_price(s, curr_year, energy) - price
                if value > 0:
                    scored.append((value / price, s, price, q_left))
            scored.sort(reverse=True)

            buys: Dict[str, int] = {}
            next_cap = temp_cap
            for _, s, price, q_left in scored:
                qty = min(q_left, next_cap // price)
                if qty > 0:
                    buys[s] = qty
                    next_cap -= qty * price

            temp_tl = {y: dict(stocks) for y, stocks in tl_state.items()}
            next_inv = dict(temp_inv)
            buy_acts_step = []
            for s, q in buys.items():
                next_inv[s] = next_inv.get(s, 0) + q
                temp_tl[curr_year][s] -= q
                buy_acts_step.append(f"b-{s}-{q}")

            # 4. Jump phase: try every year we can afford to visit and still
            # make it back home from afterwards.
            for next_year in years:
                if next_year == curr_year:
                    continue
                cost = abs(int(curr_year) - int(next_year))
                cost_home = abs(int(next_year) - int(HOME_YEAR))
                if energy < cost + cost_home:
                    continue

                res_cap, res_acts = yield (next_year, energy - cost, next_cap, next_inv, temp_tl)
                if res_cap > best_cap:
                    best_cap = res_cap
                    jump_act = f"j-{curr_year}-{next_year}"
                    best_acts = sell_acts_step + buy_acts_step + [jump_act] + res_acts

        return best_cap, best_acts

    def run_dfs(curr_year, energy, cap, inv, tl_state):
        # Explicit-stack trampoline driving dfs_step generators.
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

    _, final_acts = run_dfs(HOME_YEAR, energy_limit, initial_capital, {}, init_tl)
    return final_acts


@router.post("/stonks")
def stonks_endpoint(payload: List[Dict[str, Any]]):
    """Takes an array of test cases (JSON objects) and processes them sequentially."""
    return [solve_single(test_case) for test_case in payload]
