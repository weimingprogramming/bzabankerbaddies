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
    # THE FINAL DECODER RING
    # ==========================================
    if rule in ["obsidian", "verdigris"]:
        # LOWEST CARD WINS
        # Inverts value: 1 = 1.0 (Monster), 13 = 0.07 (Trash)
        strength = (14 - my_card) / 13.0
        
    elif rule in ["cinnabar", "amaranth"]:
        # HIGHEST CARD WINS
        # Standard value: 13 = 1.0 (Monster), 1 = 0.07 (Trash)
        # We also give Pairs a massive boost here just in case.
        if comm_card is not None and my_card == comm_card:
            strength = 1.0
        else:
            strength = my_card / 13.0
            
    else:
        # Fallback just in case
        strength = my_card / 13.0

    # ==========================================
    # POT CONTROL (Don't get stacked early!)
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
        legal_amount = get_legal_bet(pot * 2)  # Overbet the pot
        
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