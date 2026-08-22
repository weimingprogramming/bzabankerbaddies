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
  your_num = data.your_number
  comm_num = data.community_number
  is_post = data.round == "post_reveal"
  to_call = data.to_call
  pot = data.pot

  has_pair = is_post and (comm_num is not None) and (your_num == comm_num)

  # Helper to safely construct bet/raise action
  def make_raise_or_bet(target_amt: int) -> Dict[str, Any]:
    if data.min_raise_to is not None and data.max_raise_to is not None:
      clamped_amt = max(
          data.min_raise_to, min(target_amt, data.max_raise_to)
      )
      if "raise" in legal:
        return {"action": "raise", "amount": clamped_amt}
      if "bet" in legal:
        return {"action": "bet", "amount": clamped_amt}
    if "call" in legal:
      return {"action": "call"}
    if "check" in legal:
      return {"action": "check"}
    return {"action": "fold"}

  # --- TIER 1: MONSTER HAND (PAIR POST-REVEAL) ---
  if has_pair:
    # Value bet / raise aggressively
    target_amt = (
        data.min_raise_to + max(4, pot // 2)
        if data.min_raise_to
        else pot + to_call
    )
    if "raise" in legal or "bet" in legal:
      return make_raise_or_bet(target_amt)
    if "call" in legal:
      return {"action": "call"}
    if "check" in legal:
      return {"action": "check"}

  # --- TIER 2: STRONG HIGH CARDS (11, 12, 13) ---
  if your_num >= 11:
    if to_call == 0:
      # Pre-reveal: raise high card for value
      if not is_post and ("raise" in legal or "bet" in legal) and your_num >= 12:
        return make_raise_or_bet(data.min_raise_to or 4)
      if "check" in legal:
        return {"action": "check"}
    else:
      # Facing a bet: call moderate bets (pot odds)
      if to_call <= max(8, pot * 0.45) and "call" in legal:
        return {"action": "call"}
      if "fold" in legal:
        return {"action": "fold"}

  # --- TIER 3: MEDIUM CARDS (8, 9, 10) ---
  if your_num >= 8:
    if to_call == 0:
      if "check" in legal:
        return {"action": "check"}
    else:
      # Only call cheap forced bets / mini-bets
      if to_call <= 3 and "call" in legal:
        return {"action": "call"}
      if "fold" in legal:
        return {"action": "fold"}

  # --- TIER 4: WEAK CARDS (1 - 7) ---
  if to_call == 0:
    if "check" in legal:
      return {"action": "check"}
    if "call" in legal:
      return {"action": "call"}

  if "fold" in legal:
    return {"action": "fold"}
  if "check" in legal:
    return {"action": "check"}
  return {"action": "call"}


@router.post("/move")
def handle_move(payload: ShowdownPayload):
  return calculate_move(payload)