from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

# Global memory to store the scout's recon data
scout_logs = []

@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    global scout_logs
    
    rule = state.get("table_rule", "unknown")
    leg = state.get("leg_number", "?")
    
    recent = state.get("recent_hands", [])
    if recent:
        last_hand = recent[-1]
        
        # Save the raw dictionary so we can inspect exactly what the judge sends
        log_entry = {
            "leg": leg,
            "rule": rule,
            "hand_data": last_hand
        }
        scout_logs.append(log_entry)
        
        # Keep the last 100 hands so the server doesn't run out of memory
        if len(scout_logs) > 100:
            scout_logs.pop(0)

    # Force the showdown
    valid_actions = state.get("valid_actions", [])
    
    if "check" in valid_actions:
        return {"action": "check"}
    elif "call" in valid_actions:
        return {"action": "call"}
    else:
        return {"action": valid_actions[0] if valid_actions else "fold"}

# --- NEW: The Intel Extraction Endpoint ---
@router.get("/intel")
def get_intel():
    return {"recon_data": scout_logs}