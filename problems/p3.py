from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

def evaluate_strength(my_card: int, comm_card: int | None, rule: str) -> float:
    """Returns hand strength from 0.0 (worst) to 1.0 (best)."""
    
    # RULE 1: OBSIDIAN (Lowest card wins)
    if rule == "obsidian":
        # Inverts the value: a 1 becomes 1.0 (perfect), a 13 becomes ~0.07 (trash)
        return (14 - my_card) / 13.0
        
    # RULES 2 & 3: CINNABAR / AMARANTH (Highest card wins)
    elif rule in ["cinnabar", "amaranth"]:
        # Standard rules: Pairs are massive monsters
        if comm_card is not None and my_card == comm_card:
            return 1.0
        return my_card / 13.0
        
    # FALLBACK: Assume standard rules for any unknown leg
    else:
        if comm_card is not None and my_card == comm_card:
            return 1.0
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

    # 1. Evaluate hand strength
    if my_card is None:
        return {"action": "fold" if "fold" in legal_actions else "check"}
        
    strength = evaluate_strength(my_card, comm_card, rule)
    
    # ==========================================================
    # CRITICAL FIX 1: Pot Control Pre-Reveal
    # Cap strength at 0.70 before the community card is flipped.
    # This forces the bot to only 'call' or 'check' early on!
    # ==========================================================
    if comm_card is None:
        strength = min(strength, 0.70)
    
    # Helper to calculate legal aggressive bets
    def get_legal_bet(desired_amount):
        if max_raise is None or min_raise is None:
            return 0
        return max(min_raise, min(desired_amount, max_raise))

    # ==========================================================
    # CRITICAL FIX 2: Safer Monster Threshold
    # Increased from 0.85 to 0.95. We only overbet the pot if 
    # we have a Pair or the absolute highest card possible.
    # ==========================================================
    if strength >= 0.95:
        target_bet = pot * 2  
        legal_amount = get_legal_bet(target_bet)
        
        if "raise" in legal_actions:
            return {"action": "raise", "amount": legal_amount}
        elif "bet" in legal_actions:
            return {"action": "bet", "amount": legal_amount}
        elif "call" in legal_actions:
            return {"action": "call"}
            
    # DECENT HAND: Just call or check to see a cheap showdown
    elif strength >= 0.50:
        if "check" in legal_actions:
            return {"action": "check"}
        elif "call" in legal_actions:
            return {"action": "call"}
            
    # TRASH HAND: Get out immediately
    else:
        if "check" in legal_actions:
            return {"action": "check"}
        elif "fold" in legal_actions:
            return {"action": "fold"}
            
    # Failsafe
    return {"action": "fold" if "fold" in legal_actions else "check"}