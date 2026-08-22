from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    legal_actions = state.get("legal_actions", [])
    my_card = state.get("your_number")
    comm_card = state.get("community_number")
    rule = state.get("table_rule", "unknown")
    pot = state.get("pot", 1)  # Avoid division by zero
    max_raise = state.get("max_raise_to")
    min_raise = state.get("min_raise_to")
    call_amount = state.get("call_amount", 0)
    
    if my_card is None:
        return {"action": "check" if "check" in legal_actions else "fold"}

    is_pair = (comm_card is not None and my_card == comm_card)
    tier = "trash"  # Categories: "monster", "strong", "trash"

    # ==========================================
    # 1. HARD-CODED HAND TIER CLASSIFICATION
    # ==========================================
    if comm_card is None:
        # Pre-reveal tight classification
        if rule == "obsidian":
            tier = "monster" if my_card in [1, 2] else ("strong" if my_card in [3, 4] else "trash")
        elif rule == "amaranth":
            tier = "strong" if my_card in [6, 7, 8] else "trash"
        else:
            tier = "monster" if my_card == 13 else ("strong" if my_card in [11, 12] else "trash")
    else:
        # Post-reveal exact classification
        if rule == "obsidian":
            # Smallest wins, pairs are worthless
            if my_card in [1, 2]:
                tier = "monster"
            elif my_card in [3, 4]:
                tier = "strong"
            else:
                tier = "trash"

        elif rule == "amaranth":
            # Modular match
            score = (13 + comm_card - my_card) % 13
            if is_pair or score == 0:
                tier = "monster"
            elif score in [11, 12]:
                tier = "strong"
            else:
                tier = "trash"

        else:
            # Leg 1 / Leg 4: Largest wins
            if is_pair or my_card == 13:
                tier = "monster"
            elif my_card in [11, 12]:
                tier = "strong"
            else:
                tier = "trash"

    # Helper for disciplined bet sizing
    def get_value_bet_amount(pct=0.4):
        if max_raise is None or min_raise is None:
            return 0
        desired = int(pot * pct)
        return max(min_raise, min(desired, max_raise))

    # ==========================================
    # 2. DISCIPLINED ACTION LOGIC
    # ==========================================

    # A. TIER 1: MONSTER HANDS -> VALUE BET / RAISE / CALL ANY
    if tier == "monster":
        bet_amt = get_value_bet_amount(0.5)
        if "raise" in legal_actions and bet_amt > 0:
            return {"action": "raise", "amount": bet_amt}
        if "bet" in legal_actions and bet_amt > 0:
            return {"action": "bet", "amount": bet_amt}
        if "call" in legal_actions:
            return {"action": "call"}
        if "check" in legal_actions:
            return {"action": "check"}

    # B. TIER 2: STRONG HANDS -> CHECK FOR FREE, CALL ONLY ON GOOD ODDS
    elif tier == "strong":
        if "check" in legal_actions:
            return {"action": "check"}
            
        # Pot odds filter: ONLY call if the call cost is <= 35% of the pot (3:1 odds)
        # AND the raw call_amount is reasonably small (e.g., <= 5 chips)
        if "call" in legal_actions:
            if call_amount <= (pot * 0.35) and call_amount <= 5:
                return {"action": "call"}
            else:
                # Opponent bet too big -> Fold marginal hands!
                return {"action": "fold"}

    # C. TIER 3: TRASH HANDS -> CHECK IF FREE, IMMEDIATELY FOLD TO BETS
    if "check" in legal_actions:
        return {"action": "check"}
    
    if "fold" in legal_actions:
        return {"action": "fold"}

    return {"action": "check" if "check" in legal_actions else "fold"}