from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

router = APIRouter()


class ShowdownPayload(BaseModel):
  model_config = ConfigDict(extra="ignore")

  round: str
  your_number: int
  community_number: Optional[int] = None
  pot: int
  to_call: int
  min_raise_to: Optional[int] = None
  max_raise_to: Optional[int] = None
  legal_actions: List[str]
  your_stack: int


def calculate_move(data: ShowdownPayload) -> Dict[str, Any]:
  legal = set(data.legal_actions)
  is_post = data.round == "post_reveal"
  to_call = data.to_call
  pot = data.pot
  your_num = data.your_number

  # A pair is an automatic win unless the opponent has the exact same pair (a tie)
  has_pair = is_post and (data.community_number is not None) and (your_num == data.community_number)

  # --- Helper Actions ---
  def do_raise(target_amt: int) -> Dict[str, Any]:
    if data.min_raise_to is not None and data.max_raise_to is not None:
      amt = max(data.min_raise_to, min(target_amt, data.max_raise_to))
      if "raise" in legal:
        return {"action": "raise", "amount": amt}
      if "bet" in legal:
        return {"action": "bet", "amount": amt}
    return do_call()

  def do_call() -> Dict[str, Any]:
    if "call" in legal: return {"action": "call"}
    if "check" in legal: return {"action": "check"}
    return {"action": "fold"}

  def do_check_fold() -> Dict[str, Any]:
    if "check" in legal: return {"action": "check"}
    if "fold" in legal: return {"action": "fold"}
    return {"action": "call"}

  # --- POST-REVEAL (COMMUNITY CARD IS OUT) ---
  if is_post:
    if has_pair:
      # We have the nuts. Go all-in or max raise to extract everything.
      return do_raise(data.max_raise_to or 200)
    
    if your_num == 13:
      # Almost guaranteed win (only loses if Gaston randomly paired the board). 
      if to_call == 0:
        return do_raise(pot) # Bet the pot for value
      return do_call()       # Just call to keep them bluffing
      
    if your_num >= 11:
      # Very strong high card.
      if to_call == 0:
        return do_raise(max(data.min_raise_to or 4, int(pot * 0.5)))
      if to_call <= 50:
        return do_call()
      return do_check_fold()
      
    if your_num >= 8:
      # Bluff catchers. Do not fold to tiny bets anymore.
      if to_call == 0:
        return {"action": "check"}
      if to_call <= 15:
        return do_call()
      return do_check_fold()
      
    # Trash (1-7). Give up immediately if faced with a bet.
    return do_check_fold()

  # --- PRE-REVEAL (NO COMMUNITY CARD YET) ---
  else:
    if your_num == 13:
      # Raise big with the best card, just call if they shoved
      if to_call <= 30:
        return do_raise(max(data.min_raise_to or 6, 12))
      return do_call()

    if your_num >= 11:
      # Value raise small bets, call medium bets
      if to_call <= 5:
        return do_raise(max(data.min_raise_to or 6, 10))
      if to_call <= 15:
        return do_call()
      return do_check_fold()

    if your_num >= 8:
      # Keep Gaston honest by calling small pre-reveal raises
      if to_call <= 8:
        return do_call()
      return do_check_fold()

    # Trash (1-7). Fold immediately to avoid the slow bleed.
    return do_check_fold()


@router.post("/move")
def handle_move(payload: ShowdownPayload):
  return calculate_move(payload)