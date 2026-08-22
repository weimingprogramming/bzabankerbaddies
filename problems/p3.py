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
    # 1. PAIRS BEAT EVERYTHING
    # The table rules only apply if NO ONE has a pair.
    # ==========================================
    is_pair = (comm_card is not None and my_card == comm_card)
    
    if is_pair:
        strength = 1.0
        
    else:
        # ==========================================
        # 2. IF NO PAIR, APPLY THE TABLE RULE
        # ==========================================
        if rule in ["obsidian", "verdigris"]:
            # Lowest Card Wins
            strength = (14 - my_card) / 13.0
        else:
            # cinnabar, amaranth, or fallback = Highest Card Wins
            strength = my_card / 13.0

    # ==========================================
    # POT CONTROL (Don't get stacked before the reveal!)
    # ==========================================
    if comm_card is None:
        strength = min(strength, 0.70)

    def get_legal_bet(desired_amount):
        if max_raise is None or min_raise is None: return 0
        return max(min_raise, min(desired_amount, max_raise))

    # ==========================================
    # AGGRESSIVE BETTING
    # ==========================================
    if strength >= 0.95:
        legal_amount = get_legal_bet(pot * 2)
        if "raise" in legal_actions: return {"action": "raise", "amount": legal_amount}
        elif "bet" in legal_actions: return {"action": "bet", "amount": legal_amount}
        elif "call" in legal_actions: return {"action": "call"}
        
    elif strength >= 0.50:
        if "check" in legal_actions: return {"action": "check"}
        elif "call" in legal_actions: return {"action": "call"}
        
    else:
        if "check" in legal_actions: return {"action": "check"}
        elif "fold" in legal_actions: return {"action": "fold"}
        
    return {"action": "fold" if "fold" in legal_actions else "check"}