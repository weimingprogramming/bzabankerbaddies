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
    call_amount = state.get("call_amount", 0)  # Amount needed to call
    
    if my_card is None:
        return {"action": "check" if "check" in legal_actions else "fold"}

    is_pair = (comm_card is not None and my_card == comm_card)
    
    hand_tier = "trash"  # Categories: "monster", "medium", "trash"

    # ==========================================
    # STRICT HAND TIER CLASSIFICATION
    # ==========================================
    if rule == "obsidian":
        # Leg 2: Smallest Wins (Pairs mean nothing)
        if my_card in [1, 2]:
            hand_tier = "monster"
        elif my_card in [3, 4, 5, 6]:
            hand_tier = "medium"
        else:
            hand_tier = "trash"  # 7-13 are trash!

    elif rule == "amaranth":
        # Leg 3: Modular Match
        if comm_card is None:
            hand_tier = "medium" if my_card in [6, 7, 8] else "trash"
        else:
            raw_score = (13 + comm_card - my_card) % 13
            if is_pair or raw_score == 0:
                hand_tier = "monster"
            elif raw_score in [11, 12]:
                hand_tier = "medium"
            else:
                hand_tier = "trash"

    else:
        # Leg 1 & 4 (Verdigris / Cinnabar / Default): Largest Wins
        if is_pair or my_card in [12, 13]:
            hand_tier = "monster"
        elif my_card in [8, 9, 10, 11]:
            hand_tier = "medium"
        else:
            hand_tier = "trash"

    # Safe value bet sizing (50% pot, never blind all-in)
    def get_safe_raise_amount():
        if max_raise is None or min_raise is None:
            return 0
        target = max(min_raise, int(pot * 0.5))
        return min(target, max_raise)

    # ==========================================
    # CONSERVATIVE ACTION EXECUTION
    # ==========================================
    
    # 1. MONSTER HAND: VALUE BET / RAISE
    if hand_tier == "monster":
        raise_amt = get_safe_raise_amount()
        if "raise" in legal_actions and raise_amt > 0:
            return {"action": "raise", "amount": raise_amt}
        if "bet" in legal_actions and raise_amt > 0:
            return {"action": "bet", "amount": raise_amt}
        if "call" in legal_actions:
            return {"action": "call"}
        if "check" in legal_actions:
            return {"action": "check"}

    # 2. MEDIUM HAND: CHECK FOR FREE, CALL ONLY IF CHEAP (<= 2 CHIPS)
    elif hand_tier == "medium":
        if "check" in legal_actions:
            return {"action": "check"}
        if "call" in legal_actions and call_amount <= 2:
            return {"action": "call"}
        if "fold" in legal_actions:
            return {"action": "fold"}

    # 3. TRASH HAND: CHECK IF FREE, FOLD IMMEDIATELY IF BET TO
    if "check" in legal_actions:
        return {"action": "check"}
    if "fold" in legal_actions:
        return {"action": "fold"}

    return {"action": "check" if "check" in legal_actions else "fold"}