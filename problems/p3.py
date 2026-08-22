from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # 1. Safely extract action strings, whether they are dicts or raw strings
        raw_actions = state.get("valid_actions", [])
        action_names = []
        for a in raw_actions:
            if isinstance(a, dict):
                action_names.append(a.get("action", ""))
            else:
                action_names.append(str(a))
                
        # Define our safe fallback (Check if we can, otherwise Call, otherwise Fold)
        fallback = "check" if "check" in action_names else ("call" if "call" in action_names else "fold")
                
        # 2. Locate our hole card
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
        
        # 3. OBSIDIAN STRATEGY: Lowest Card Wins
        if rule == "obsidian" and my_card is not None:
            if my_card <= 6:
                # Strong low card: Call their value bets
                return {"action": "call" if "call" in action_names else "check"}
            else:
                # Weak high card: Fold if they bet
                return {"action": "check" if "check" in action_names else "fold"}
                
        # 4. DEFAULT STRATEGY: Bluff Catching (Legs 1, 2, 3)
        return {"action": fallback}
        
    except Exception:
        # If absolutely anything goes wrong, default to a safe passive action
        return {"action": "fold"}