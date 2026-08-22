from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Extract legal actions correctly based on the payload structure
    legal_actions = state.get("legal_actions", [])
    
    # 2. Extract your card number
    my_card = state.get("your_number")
    rule = state.get("table_rule", "unknown")
    
    # 3. OBSIDIAN RULE STRATEGY (Lowest card wins)
    if rule == "obsidian" and my_card is not None:
        if my_card <= 6:
            # Strong low card: prefer to call or check to see the showdown
            if "call" in legal_actions:
                return {"action": "call"}
            elif "check" in legal_actions:
                return {"action": "check"}
        else:
            # High card: fold if facing a bet, otherwise check
            if "check" in legal_actions:
                return {"action": "check"}
            elif "fold" in legal_actions:
                return {"action": "fold"}
                
    # 4. DEFAULT STRATEGY FOR OTHER LEGS (Bluff farming)
    # Check or call to extract chips from the opponent's aggressive betting
    if "check" in legal_actions:
        return {"action": "check"}
    elif "call" in legal_actions:
        return {"action": "call"}
    elif "fold" in legal_actions:
        return {"action": "fold"}
        
    return {"action": legal_actions[0] if legal_actions else "fold"}