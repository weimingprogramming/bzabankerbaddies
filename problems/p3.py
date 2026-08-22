from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Log the previous hand's showdown to deduce the secret rules
    recent = state.get("recent_hands", [])
    if recent:
        last_hand = recent[-1]
        
        # Only log if it went to a showdown (we need to see their cards)
        if last_hand.get("showdown"):
            rule = state.get("table_rule", "unknown")
            leg = state.get("leg_number", "?")
            
            # Extract hands and winner
            my_cards = last_hand["showdown"].get("your_cards", [])
            opp_cards = last_hand["showdown"].get("opponent_cards", [])
            chip_delta = last_hand.get("chip_delta", 0)
            
            winner = "TIE"
            if chip_delta > 0: winner = "WE WON"
            elif chip_delta < 0: winner = "OPPONENT WON"
            
            # flush=True forces Render to show the logs immediately
            print(f"--- LEG {leg} | RULE: {rule} ---", flush=True)
            print(f"My Cards: {my_cards}", flush=True)
            print(f"Op Cards: {opp_cards}", flush=True)
            print(f"Result:   {winner}\n", flush=True)

    # 2. Force the showdown by always Checking or Calling
    valid_actions = state.get("valid_actions", [])
    
    if "check" in valid_actions:
        return {"action": "check"}
    elif "call" in valid_actions:
        return {"action": "call"}
    else:
        return {"action": valid_actions[0] if valid_actions else "fold"}