import base64
import json
import math
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Priority string to integer mapping
PRIORITY_MAP = {
    "LOW": 1,
    "NORMAL": 2,
    "MED": 2,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
    "URGENT": 4,
}


class SolveRequest(BaseModel):
  payload: str


@router.post("/solve")
def solve_problem_2(request: SolveRequest) -> Dict[str, Any]:
  try:
    # Handle base64 padding issues if present
    b64_str = request.payload.strip()
    missing_padding = len(b64_str) % 4
    if missing_padding:
      b64_str += "=" * (4 - missing_padding)

    decoded_bytes = base64.b64decode(b64_str)
    data = json.loads(decoded_bytes.decode("utf-8"))
  except Exception as e:
    raise HTTPException(
        status_code=400, detail=f"Invalid base64 or JSON: {str(e)}"
    )

  adapt_input = data.get("adaptInput", {})
  user = adapt_input.get("user", {})
  metadata = adapt_input.get("metadata", {})

  # 1. Action mapping (convert to lowercase)
  action = str(adapt_input.get("action", "")).lower()

  # 2. Priority mapping (string -> int or fallback to int)
  raw_priority = metadata.get("priority", 1)
  if isinstance(raw_priority, str):
    priority = PRIORITY_MAP.get(raw_priority.upper(), 1)
  elif isinstance(raw_priority, (int, float)):
    priority = int(raw_priority)
  else:
    priority = 1

  result: Dict[str, Any] = {
      "adaptOutput": {
          "id": user.get("id", ""),
          "name": user.get("fullName", ""),
          "action": action,
          "priority": priority,
      }
  }

  # SLO computation from heartbeats
  heartbeats: List[Dict] = data.get("heartbeats", [])
  slo_query = data.get("sloQuery")

  if slo_query and heartbeats:
    target_service = slo_query.get("service", "")
    since = slo_query.get("since", 0)

    filtered = [
        hb for hb in heartbeats
        if hb.get("service") == target_service and hb.get("timestamp", 0) >= since
    ]

    if filtered:
      ok_count = sum(1 for hb in filtered if hb.get("status") == "OK")
      availability = ok_count / len(filtered)

      latencies = sorted(hb.get("latencyMs", 0) for hb in filtered)
      idx = math.ceil(len(latencies) * 0.95) - 1
      p95 = latencies[max(idx, 0)]

      result["sloOutput"] = {
          "availability": availability,
          "p95LatencyMs": p95,
      }

  return result