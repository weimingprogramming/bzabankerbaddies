from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

def evaluate_hand_strength(state: Dict[str, Any]) -> float:
    """Returns normalized hand strength from 0.0 (garbage) to 1.0 (unbeatable)."""
    table_rule = state.get("table_rule", "standard").lower()
    your_num = state.get("your_number")
    comm_num = state.get("community_number")
    
    if your_num is None:
        return 0.5

    # Leg 2: Obsidian (Smallest Wins, pairs disabled)
    if table_rule == "obsidian":
        return (14 - your_num) / 13.0

    # Leg 3: Amaranth (Modular Match distance to community card)
    if table_rule == "amaranth":
        if comm_num is not None:
            if your_num == comm_num:
                return 1.0  # Pair beats non-pairs
            dist = (13 + comm_num - your_num) % 13
            return (13 - dist) / 13.0
        return (14 - your_num) / 13.0

    # Leg 1 & 4: Verdigris / Cinnabar / Standard (Highest Card or Pair)
    if comm_num is not None and your_num == comm_num:
        return 1.0  # Pair beats non-pairs
    
    return your_num / 13.0

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    legal_actions = state.get("legal_actions", [])
    to_call = state.get("to_call", 0)
    your_stack = state.get("your_stack", 200)
    pot = state.get("pot", 1)
    min_raise = state.get("min_raise_to")
    max_raise = state.get("max_raise_to")
    players = state.get("players", [])

    strength = evaluate_hand_strength(state)

    # 1. Detect opponent aggression (all-ins or huge bets relative to stack/pot)
    opponent_all_in = any(p.get("all_in", False) for p in players if p.get("name") != "you")
    is_massive_bet = to_call >= (your_stack * 0.5) or opponent_all_in

    # 2. Facing heavy pressure / All-In: Demand near-nut hand (strength >= 0.92)
    if is_massive_bet:
        if strength >= 0.92:
            if "call" in legal_actions:
                return {"action": "call"}
        if "check" in legal_actions:
            return {"action": "check"}
        return {"action": "fold"}

    # 3. Strong Hands: Value raise or bet against standard action
    if strength >= 0.80:
        if "raise" in legal_actions and min_raise is not None and max_raise is not None:
            target_raise = int(min_raise + 0.3 * (max_raise - min_raise))
            amount = max(min_raise, min(target_raise, max_raise))
            return {"action": "raise", "amount": amount}
        if "bet" in legal_actions and min_raise is not None and max_raise is not None:
            amount = max(min_raise, min(min_raise + 6, max_raise))
            return {"action": "bet", "amount": amount}
        if "call" in legal_actions:
            return {"action": "call"}

    # 4. Moderate Hands: Call small/medium bets using pot odds
    pot_odds_threshold = to_call / max(pot + to_call, 1)
    if strength >= 0.50:
        if "check" in legal_actions:
            return {"action": "check"}
        # Only call if hand strength comfortably exceeds pot odds ratio
        if "call" in legal_actions and pot_odds_threshold <= 0.35 and to_call <= (your_stack * 0.20):
            return {"action": "call"}

    # 5. Weak Hands: Free checks only, otherwise fold
    if "check" in legal_actions:
        return {"action": "check"}

    return {"action": "fold"}