from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

def evaluate_strength(my_card: int, comm_card: int | None, rule: str) -> float:
    """
    Returns hand strength from 0.0 (worst) to 1.0 (best).
    Replace the rules below with what you discover from your logs!
    """
    # ---------------------------------------------------------
    # RULE 1: REPLACE THIS WITH YOUR LOGGED DISCOVERIES
    # ---------------------------------------------------------
    if rule == "obsidian":
        # Example guess: Maybe obsidian means Lowest Card Wins, ignoring pairs?
        return (13 - my_card) / 13.0
        
    # ---------------------------------------------------------
    # RULE 2: REPLACE THIS WITH YOUR LOGGED DISCOVERIES
    # ---------------------------------------------------------
    elif rule == "chalcedony":
        # Example guess: Maybe evens beat odds?
        if my_card % 2 == 0:
            return 0.8  # Strong
        return 0.2      # Weak
        
    # ---------------------------------------------------------
    # DEFAULT STRATEGY (Assumes standard rules: Pairs > High Cards)
    # ---------------------------------------------------------
    else:
        # If we have a pair, we have a monster hand (1.0)
        if comm_card is not None and my_card == comm_card:
            return 1.0
            
        # Otherwise, higher is better
        return my_card / 13.0


@router.post("/move")
def play_showdown(state: Dict[str, Any]) -> Dict[str, Any]:
    legal_actions = state.get("legal_actions", [])
    my_card = state.get("your_number")
    comm_card = state.get("community_number")
    rule = state.get("table_rule", "unknown")
    
    # =====================================================================
    # INTELLIGENCE GATHERING: UNCOMMENT TO LOG SHOWDOWNS & FIND THE RULES
    # =====================================================================
    # recent = state.get("recent_hands", [])
    # if recent and recent[-1].get("shown_numbers"):
    #     print(f"RULE: {rule}")
    #     print(f"Community: {recent[-1].get('community_number')}")
    #     print(f"Shown: {recent[-1].get('shown_numbers')} | Winners: {recent[-1].get('winners')}")
    #     print("-" * 40)
    
    # 1. Calculate how good our hand is right now
    if my_card is None:
        return {"action": "fold"} # Failsafe
        
    strength = evaluate_strength(my_card, comm_card, rule)
    
    # 2. Betting Logic
    
    # MONSTER HAND: Bet or Raise to extract chips!
    if strength >= 0.85:
        if "raise" in legal_actions:
            # You can extract min_raise_to from state to size your bets perfectly, 
            # but for now, just choosing 'raise' is enough to build the pot.
            return {"action": "raise", "amount": state.get("min_raise_to", 0)}
        elif "bet" in legal_actions:
            return {"action": "bet", "amount": state.get("min_raise_to", 0)}
        elif "call" in legal_actions:
            return {"action": "call"}
            
    # DECENT HAND: See cards cheaply
    elif strength >= 0.50:
        if "check" in legal_actions:
            return {"action": "check"}
        elif "call" in legal_actions:
            return {"action": "call"}
            
    # TRASH HAND: Get out for free, or fold if faced with a bet
    else:
        if "check" in legal_actions:
            return {"action": "check"}
        elif "fold" in legal_actions:
            return {"action": "fold"}
            
    # Failsafe
    return {"action": "fold" if "fold" in legal_actions else "check"}