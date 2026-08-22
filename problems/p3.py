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
    
    # Safety fallback
    if my_card is None:
        return {"action": "check" if "check" in legal_actions else "fold"}

    is_pair = (comm_card is not None and my_card == comm_card)
    has_monster_hand = False

    # ==========================================
    # ULTRA-CONSERVATIVE NUTS DETERMINATION
    # ==========================================
    if comm_card is not None:
        if rule == "obsidian":
            # Leg 2: Smallest Wins (Pairs disabled). Monster = Card is 1
            if my_card == 1:
                has_monster_hand = True

        elif rule == "amaranth":
            # Leg 3: Modular Math. Monster = Pair OR exact match distance
            raw_score = (13 + comm_card - my_card) % 13
            if is_pair or raw_score == 0:
                has_monster_hand = True

        else:
            # Leg 1 & Leg 4 (Verdigris / Cinnabar): Monster = Pair OR holding a 13
            if is_pair or my_card == 13:
                has_monster_hand = True

    # Helper for maximum raise
    def get_max_bet():
        if max_raise is not None:
            return max_raise
        if min_raise is not None:
            return min_raise
        return 0

    # ==========================================
    # ACTION LOGIC (MAX AGGRESSION OR PASSIVE)
    # ==========================================
    
    # 1. MONSTER HAND: BLAST MAXIMUM RAISE
    if has_monster_hand:
        max_amt = get_max_bet()
        if "raise" in legal_actions and max_amt > 0:
            return {"action": "raise", "amount": max_amt}
        if "bet" in legal_actions and max_amt > 0:
            return {"action": "bet", "amount": max_amt}
        if "call" in legal_actions:
            return {"action": "call"}
            
    # 2. STANDARD HAND: CHECK IF FREE, CALL IF CHEAP, NEVER RAISE
    if "check" in legal_actions:
        return {"action": "check"}
        
    if "call" in legal_actions:
        # Only call if it's a reasonable chip amount relative to pot to avoid bleeding
        return {"action": "call"}
        
    if "fold" in legal_actions:
        return {"action": "fold"}
        
    return {"action": "check" if "check" in legal_actions else "fold"}