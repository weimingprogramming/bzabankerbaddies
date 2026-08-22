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
    
    # Failsafe if hand hasn't started properly
    if my_card is None:
        return {"action": "check" if "check" in legal_actions else "fold"}

    # ==========================================
    # 1. 100% CONFIDENCE FOR PAIRS
    # ==========================================
    if comm_card is not None and my_card == comm_card:
        confidence = 1.0
        
    else:
        # ==========================================
        # 2. MATCH RULES TO PROPER NAMES
        # ==========================================
        if rule == "obsidian":
            # Leg 4: Smallest Card Wins
            confidence = (14 - my_card) / 13.0
        else:
            # Legs 1, 2, 3 (Verdigris, Cinnabar, Amaranth): Largest Wins
            confidence = my_card / 13.0

    # Pot Control: Never go all-in blind before the reveal
    if comm_card is None:
        confidence = min(confidence, 0.60)

    def get_legal_bet(desired_amount):
        if max_raise is None or min_raise is None: return 0
        return max(min_raise, min(desired_amount, max_raise))

    # ==========================================
    # 3. ACTION LOGIC (Check > Fold fallback)
    # ==========================================
    
    # HIGH CONFIDENCE (>= 0.85): Bet/Raise to build the pot
    if confidence >= 0.85:
        legal_amount = get_legal_bet(int(pot * 0.75))
        if "raise" in legal_actions: 
            return {"action": "raise", "amount": legal_amount}
        if "bet" in legal_actions: 
            return {"action": "bet", "amount": legal_amount}
            
    # MEDIUM CONFIDENCE (>= 0.50): Stay in the hand
    if confidence >= 0.50:
        if "check" in legal_actions: 
            return {"action": "check"}
        if "call" in legal_actions: 
            return {"action": "call"}
            
    # LOW CONFIDENCE (< 0.50): Try to leave safely
    if "check" in legal_actions:
        return {"action": "check"}  # Always check instead of folding if allowed!
    if "fold" in legal_actions:
        return {"action": "fold"}
        
    # Absolute failsafe 
    return {"action": "check" if "check" in legal_actions else "call"}