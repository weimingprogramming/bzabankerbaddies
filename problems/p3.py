from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

def calculate_hand_strength(my_card: int, comm_card: int, rule: str) -> float:
    # 1. PRE-REVEAL EVALUATION (No Community Card Yet)
    if comm_card is None:
        if rule == "obsidian":
            return (14 - my_card) / 13.0  # Lower card is better
        elif rule == "amaranth":
            return 0.5                    # Neutral pre-reveal state
        else:
            return my_card / 13.0         # Standard higher card is better

    # 2. POST-REVEAL EVALUATION
    
    # --- LEG 3 (Amaranth): Wrap-Around Modular Math ---
    if rule == "amaranth":
        raw_score = (13 + comm_card - my_card) % 13
        if raw_score == 0:
            raw_score = 13
        return raw_score / 13.0

    # --- LEG 2 (Obsidian): Lowest Card Wins ---
    elif rule == "obsidian":
        return (14 - my_card) / 13.0

    # --- LEG 1 & LEG 4 (Verdigris / Cinnabar / Fallback): Highest Card Wins ---
    else:
        return my_card / 13.0


@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    legal_actions = state.get("legal_actions", [])
    my_card = state.get("your_number")
    comm_card = state.get("community_number")
    rule = state.get("table_rule", "unknown")
    pot = state.get("pot", 0)
    max_raise = state.get("max_raise_to")
    min_raise = state.get("min_raise_to")
    
    if my_card is None:
        return {"action": "check" if "check" in legal_actions else "fold"}

    # Evaluate raw strength based on table rule
    strength = calculate_hand_strength(my_card, comm_card, rule)

    # Pairs check (Active on all rules EXCEPT Obsidian)
    if comm_card is not None and my_card == comm_card and rule != "obsidian":
        strength = 1.0

    # Multiway scaling for 6-player table
    players = state.get("players", [])
    if isinstance(players, dict):
        players = list(players.values())
    active_opps = max(1, sum(1 for p in players if not p.get("folded", False) and not p.get("busted", False)) - 1)
    
    # Scale confidence down based on number of active opponents
    if strength < 1.0:
        confidence = strength ** (active_opps * 0.8)
    else:
        confidence = 1.0

    def get_legal_bet(desired_amount):
        if max_raise is None or min_raise is None: return 0
        return max(min_raise, min(desired_amount, max_raise))

    # --- ACTION EXECUTION ---
    if confidence >= 0.80:
        legal_amount = get_legal_bet(int(pot * 0.6))
        if "raise" in legal_actions: return {"action": "raise", "amount": legal_amount}
        if "bet" in legal_actions: return {"action": "bet", "amount": legal_amount}
        if "call" in legal_actions: return {"action": "call"}

    elif confidence >= 0.45:
        if "check" in legal_actions: return {"action": "check"}
        if "call" in legal_actions: return {"action": "call"}

    # Weak hand: Check if free, fold if forced to bet
    if "check" in legal_actions: return {"action": "check"}
    if "fold" in legal_actions: return {"action": "fold"}

    return {"action": "call" if "call" in legal_actions else "check"}