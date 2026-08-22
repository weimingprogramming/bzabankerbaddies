from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

# Global storage to look at the exact raw payload from the judge
last_payload = {}

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    global last_payload
    last_payload = state
    
    # Extract valid actions safely no matter how they are formatted
    raw_actions = state.get("valid_actions", [])
    action_names = []
    for a in raw_actions:
        if isinstance(a, dict):
            # Check common keys where action names might hide
            for k in ["action", "type", "name"]:
                if k in a:
                    action_names.append(str(a[k]))
        elif isinstance(a, str):
            action_names.append(a)
            
    # Fallback to whatever action is available
    if "check" in action_names:
        return {"action": "check"}
    elif "call" in action_names:
        return {"action": "call"}
    elif action_names:
        return {"action": action_names[0]}
    
    return {"action": "fold"}

@router.get("/inspect")
def inspect_payload():
    return last_payload