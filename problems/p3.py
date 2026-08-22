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
    # 1. COUNT ACTIVE OPPONENTS
    # The more people in the hand, the weaker your card is.
    # ==========================================
    players = state.get("players", [])
    if isinstance(players, dict):
        players = list(players.values())
        
    # Count how many players haven't folded or busted
    active_players = sum(1 for p in players if not p.get("folded", False) and not p.get("busted", False))
    active_opps = max(1, active_players - 1)  # Minimum 1 to avoid math errors

    # ==========================================
    # 2. MULTIWAY MATH EVALUATION
    # ==========================================
    is_pair = (comm_card is not None and my_card == comm_card)
    confidence = 0.0

    if rule == "obsidian":
        # Phase 3 Obsidian: Smallest wins, NO PAIRS.
        # A 1 is a 1.0 base. We raise it to the power of opponents.
        base_conf = (14 - my_card) / 13.0
        confidence = base_conf ** active_opps
        
    else:
        # Standard Rules: Largest wins, Pairs win.
        if is_pair:
            # We hit a pair. Unbeatable monster.
            confidence = 1.0
        else:
            base_conf = my_card / 13.0
            # Exponentially decrease confidence based on crowd size
            confidence = base_conf ** active_opps
            
            # Apply a penalty for the threat of someone else hitting a pair!
            if comm_card is None:
                # Pre-reveal: ~7% chance per opponent they WILL pair
                confidence = confidence * (1.0 - (active_opps * 0.07))
            else:
                # Post-reveal: ~8% chance per opponent they ALREADY paired
                confidence = confidence * (1.0 - (active_opps * 0.08))

    def get_legal_bet(desired_amount):
        if max_raise is None or min_raise is None: return 0
        return max(min_raise, min(desired_amount, max_raise))

    # ==========================================
    # 3. PHASE 3 TIGHT ACTION LOGIC
    # ==========================================
    
    if confidence >= 0.80:
        # We have the nuts. Extract value from the bots.
        legal_amount = get_legal_bet(int(pot * 0.5))
        if "raise" in legal_actions: 
            return {"action": "raise", "amount": legal_amount}
        if "bet" in legal_actions: 
            return {"action": "bet", "amount": legal_amount}
            
    if confidence >= 0.40:
        # Good enough to stick around cheaply.
        if "check" in legal_actions: return {"action": "check"}
        if "call" in legal_actions: return {"action": "call"}
            
    # Trash hand or too many players: Fold immediately and save chips
    if "check" in legal_actions: return {"action": "check"}
    if "fold" in legal_actions: return {"action": "fold"}
        
    return {"action": "call" if "call" in legal_actions else "check"}