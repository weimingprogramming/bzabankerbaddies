import logging
from fastapi import APIRouter
from typing import Dict, Any

# Set up a logger that blasts straight through to Render's console
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

router = APIRouter()

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Extract the intel
    recent = state.get("recent_hands", [])
    if recent:
        last_hand = recent[-1]
        
        # Only log if we saw their cards
        if last_hand.get("showdown"):
            rule = state.get("table_rule", "unknown")
            leg = state.get("leg_number", "?")
            
            my_cards = last_hand["showdown"].get("your_cards", [])
            opp_cards = last_hand["showdown"].get("opponent_cards", [])
            chip_delta = last_hand.get("chip_delta", 0)
            
            winner = "TIE"
            if chip_delta > 0: winner = "WE WON"
            elif chip_delta < 0: winner = "OPPONENT WON"
            
            # WARNING level guarantees it prints immediately
            logger.warning(
                f"LEG {leg} | RULE: {rule} | Me: {my_cards} | Opp: {opp_cards} | {winner}"
            )

    # 2. Force the showdown
    valid_actions = state.get("valid_actions", [])
    
    if "check" in valid_actions:
        return {"action": "check"}
    elif "call" in valid_actions:
        return {"action": "call"}
    else:
        return {"action": valid_actions[0] if valid_actions else "fold"}