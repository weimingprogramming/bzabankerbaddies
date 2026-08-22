from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    
    # 1. Safely extract action strings from the dictionaries
    raw_actions = state.get("valid_actions", [])
    action_names = []
    for a in raw_actions:
        if isinstance(a, dict):
            action_names.append(a.get("action", ""))
        else:
            action_names.append(a)
            
    # 2. Locate our hole card to evaluate our hand strength
    my_card = None
    for key in ["your_cards", "hand", "cards", "pocket_cards", "hole_cards"]:
        val = state.get(key)
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], int):
            my_card = val[0]
            break
        elif isinstance(val, int):
            my_card = val
            break
            
    rule = state.get("table_rule", "unknown")
    
    # 3. THE OBSIDIAN COUNTER-STRATEGY (Lowest Card Wins)
    if rule == "obsidian" and my_card is not None:
        if my_card <= 6:
            # We have a strong low card (1-6). Call their value bets!
            if "call" in action_names: return {"action": "call"}
            elif "check" in action_names: return {"action": "check"}
        else:
            # We have a high card (bad). Fold if they bet.
            if "check" in action_names: return {"action": "check"}
            return {"action": "fold"}
            
    # 4. DEFAULT STRATEGY (The 300-point bluff catcher)
    # For all other rules, the opponent bluffs wildly. We just check/call to farm chips.
    if "check" in action_names:
        return {"action": "check"}
    elif "call" in action_names:
        return {"action": "call"}
        
    # Fallback
    return {"action": "fold"}