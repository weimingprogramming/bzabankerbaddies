from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

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
        return {"action": "fold" if "fold" in legal_actions else "check"}

    # ==========================================
    # 1. STRICT FACE-VALUE EVALUATION
    # We no longer hardcode pairs as 1.0! If a rule disables pairs, 
    # a pair of 2s is just a 2. We play the card's raw power.
    # ==========================================
    if rule in ["obsidian", "verdigris"]:
        # Lowest Card Wins: 1 is a monster (1.0), 13 is trash
        strength = (14 - my_card) / 13.0
    else:
        # cinnabar, amaranth, or fallback: 13 is a monster (1.0), 1 is trash
        strength = my_card / 13.0

    # ==========================================
    # 2. STRICT POT CONTROL (Pre-reveal Cap)
    # Never raise before the community card is revealed.
    # ==========================================
    if comm_card is None:
        strength = min(strength, 0.60)

    def get_legal_bet(desired_amount):
        if max_raise is None or min_raise is None: return 0
        return max(min_raise, min(desired_amount, max_raise))

    # ==========================================
    # 3. SAFER AGGRESSION (Value Betting)
    # ==========================================
    if strength >= 0.85:
        # We have a top 2 card. Bet 75% of the pot to extract value safely.
        legal_amount = get_legal_bet(int(pot * 0.75))
        
        if "raise" in legal_actions: return {"action": "raise", "amount": legal_amount}
        elif "bet" in legal_actions: return {"action": "bet", "amount": legal_amount}
        elif "call" in legal_actions: return {"action": "call"}
        
    elif strength >= 0.60:
        # We have a decent card. See a cheap showdown.
        if "check" in legal_actions: return {"action": "check"}
        elif "call" in legal_actions: return {"action": "call"}
        
    else:
        # We have trash. Fold immediately to a bet.
        if "check" in legal_actions: return {"action": "check"}
        elif "fold" in legal_actions: return {"action": "fold"}
        
    return {"action": "fold" if "fold" in legal_actions else "check"}