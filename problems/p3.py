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
    call_amount = state.get("call_amount", 0)
    
    if my_card is None:
        return {"action": "check" if "check" in legal_actions else "fold"}

    # Track active non-folded, non-busted opponents
    players = state.get("players", [])
    if isinstance(players, dict):
        players = list(players.values())
    active_opps = max(1, sum(1 for p in players if not p.get("folded", False) and not p.get("busted", False)) - 1)

    is_pair = (comm_card is not None and my_card == comm_card)

    # ==========================================
    # 1. EVALUATE CARD STRENGTH (0.0 to 1.0)
    # ==========================================
    if comm_card is None:
        # Pre-reveal: Evaluate raw rank by rule
        if rule == "obsidian":
            strength = (14 - my_card) / 13.0
        elif rule == "amaranth":
            strength = 1.0 - (abs(my_card - 7) / 7.0)
        else:
            strength = my_card / 13.0
    else:
        # Post-reveal: Evaluate true hand rank
        if rule == "obsidian":
            # Lowest wins, pairs mean nothing
            strength = (14 - my_card) / 13.0
        elif rule == "amaranth":
            if is_pair:
                strength = 1.0
            else:
                score = (13 + comm_card - my_card) % 13
                strength = 1.0 if score == 0 else (score / 13.0)
        else:
            # Standard High Card / Pairs win
            if is_pair:
                strength = 1.0
            else:
                strength = my_card / 13.0

    # Multiway probability adjustment
    if strength < 1.0:
        confidence = strength ** (active_opps * 0.7)
    else:
        confidence = 1.0

    def get_balanced_bet(pct=0.4):
        if max_raise is None or min_raise is None:
            return 0
        desired = int(pot * pct)
        return max(min_raise, min(desired, max_raise))

    # ==========================================
    # 2. DECISION ENGINE
    # ==========================================

    # PREMIUM HAND (Confidence >= 0.75): Bet/Raise for value
    if confidence >= 0.75:
        bet_amt = get_balanced_bet(0.45)
        if "raise" in legal_actions and bet_amt > 0:
            return {"action": "raise", "amount": bet_amt}
        if "bet" in legal_actions and bet_amt > 0:
            return {"action": "bet", "amount": bet_amt}
        if "call" in legal_actions:
            return {"action": "call"}

    # MARGINAL HAND (Confidence >= 0.40): Opportunistic play / Pot control
    elif confidence >= 0.40:
        if "check" in legal_actions:
            # Late position pot steal pre-reveal if pot is small
            if comm_card is None and active_opps <= 3 and "bet" in legal_actions:
                steal_amt = get_balanced_bet(0.25)
                if steal_amt > 0:
                    return {"action": "bet", "amount": steal_amt}
            return {"action": "check"}
        
        # Only call small bets relative to strength
        if "call" in legal_actions and call_amount <= 3:
            return {"action": "call"}

    # WEAK HAND: Free check or fold immediately
    if "check" in legal_actions:
        return {"action": "check"}
    if "fold" in legal_actions:
        return {"action": "fold"}

    return {"action": "check" if "check" in legal_actions else "fold"}