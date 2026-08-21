from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ProblemInput(BaseModel):
  payload: dict = {}


@router.post("/solve")
def solve_challenge(data: ProblemInput):
  # Logic will be implemented per phase tomorrow
  return {"result": "ok", "received": data.payload}