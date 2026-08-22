from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

def calculate_hand_strength(my_card: int, comm_card: int, rule: str) -> float:
    # 1. NO COMMUNITY CARD YET (Pre-reveal)
    if comm_card is None:
        if rule == "obsidian":
            return (14 - my_card) / 13.0  # Lower is better
        elif rule == "amaranth":
            # Target middle cards (6-9) because high community cards are likely
            return 1.0 - (abs(my_card - 7) / 7.0)
        else:
            return my_card / 13.0  # Standard higher is better

    # 2. POST-REVEAL EVALUATION
    
    # --- RULE: HIGHEST BELOW COMMUNITY CARD (Price is Right) ---
    if rule == "amaranth":  # Or whichever rule code maps to this
        if my_card <= comm_card:
            # Valid hand: Scored between 0.50 and 1.0 based on how high it is
            return 0.5 + (my_card / (2.0 * comm_card))
        else:
            # Busted hand (Over community card): Low score
            return (my_card / 13.0) * 0.49

    # --- RULE: LOWEST CARD WINS ---
    elif rule == "obsidian":
        return (14 - my_card) / 13.0

    # --- RULE: STANDARD HIGHEST CARD WINS ---
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

    # Evaluate raw strength
    strength = calculate_hand_strength(my_card, comm_card, rule)

    # Pairs check (if applicable)
    if comm_card is not None and my_card == comm_card and rule != "obsidian":
        strength = 1.0

    # Multiway scaling for 6-player table
    players = state.get("players", [])
    if isinstance(players, dict):
        players = list(players.values())
    active_opps = max(1, sum(1 for p in players if not p.get("folded", False) and not p.get("busted", False)) - 1)
    
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

    # Weak hand: Check if free, fold if bet to
    if "check" in legal_actions: return {"action": "check"}
    if "fold" in legal_actions: return {"action": "fold"}

    return {"action": "call" if "call" in legal_actions else "check"}