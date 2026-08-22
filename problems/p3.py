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
        return {"action": "check" if "check" in legal_actions else "fold"}

    # ==========================================
    # 1. 100% CONFIDENCE FOR PAIRS
    # ==========================================
    if comm_card is not None and my_card == comm_card:
        confidence = 1.0
        
    else:
        # ==========================================
        # 2. THE TRUE RULE MAPPING
        # ==========================================
        if rule == "obsidian":
            # Smallest wins: 1 is best (1.0), 13 is worst (0.0)
            confidence = (14 - my_card) / 13.0
            
        elif rule == "amaranth":
            # Evens Win: Any Even beats any Odd. 
            if my_card % 2 == 0:
                confidence = 0.5 + (my_card / 26.0)  # Evens score 0.5 to 1.0
            else:
                confidence = my_card / 26.0          # Odds score 0.0 to 0.5
                
        elif rule == "cinnabar":
            # Odds Win: Any Odd beats any Even.
            if my_card % 2 != 0:
                confidence = 0.5 + (my_card / 26.0)  # Odds score 0.5 to 1.0
            else:
                confidence = my_card / 26.0          # Evens score 0.0 to 0.5
                
        else:
            # verdigris: Largest wins (Standard)
            confidence = my_card / 13.0

    # Pot Control: Never go crazy before the community card reveals
    if comm_card is None:
        confidence = min(confidence, 0.60)

    def get_legal_bet(desired_amount):
        if max_raise is None or min_raise is None: return 0
        return max(min_raise, min(desired_amount, max_raise))

    # ==========================================
    # 3. SMART ACTION LOGIC
    # ==========================================
    
    if confidence >= 0.85:
        # Safe value bet (50% of pot) instead of wild overbetting
        legal_amount = get_legal_bet(int(pot * 0.5))
        if "raise" in legal_actions: 
            return {"action": "raise", "amount": legal_amount}
        if "bet" in legal_actions: 
            return {"action": "bet", "amount": legal_amount}
            
    if confidence >= 0.50:
        if "check" in legal_actions: return {"action": "check"}
        if "call" in legal_actions: return {"action": "call"}
            
    # Trash hand: Escape immediately
    if "check" in legal_actions: return {"action": "check"}
    if "fold" in legal_actions: return {"action": "fold"}
        
    return {"action": "call" if "call" in legal_actions else "check"}