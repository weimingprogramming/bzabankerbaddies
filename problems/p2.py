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
    # Handle base64: strip whitespace/newlines, handle URL-safe encoding, fix padding
    b64_str = request.payload.strip().replace('\n', '').replace('\r', '').replace(' ', '')
    b64_str = b64_str.replace('-', '+').replace('_', '/')
    missing_padding = len(b64_str) % 4
    if missing_padding:
      b64_str += "=" * (4 - missing_padding)

    decoded_bytes = base64.b64decode(b64_str)
    data = json.loads(decoded_bytes.decode("utf-8"))
  except Exception as e:
    raise HTTPException(
        status_code=400, detail=f"Invalid base64 or JSON: {str(e)}"
    )

  adapt_input = data.get("adaptInput") or {}
  user = adapt_input.get("user") or {}
  metadata = adapt_input.get("metadata") or {}

  # 1. Action mapping (convert to lowercase, strip whitespace)
  action = str(adapt_input.get("action", "")).strip().lower()

  # 2. Priority mapping (string -> int or fallback to int)
  raw_priority = metadata.get("priority", 1)
  if isinstance(raw_priority, str):
    priority = PRIORITY_MAP.get(raw_priority.strip().upper(), 1)
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
  heartbeats: List[Dict] = data.get("heartbeats") or []
  slo_query = data.get("sloQuery")

  if slo_query is not None:
    target_service = slo_query.get("service", "")
    since = slo_query.get("since", 0)

    filtered = [
        hb for hb in heartbeats
        if hb.get("service") == target_service and hb.get("timestamp", 0) >= since
    ]

    if not filtered:
      result["sloOutput"] = {
          "availability": 0.0,
          "p95LatencyMs": 0,
      }
    else:
      ok_count = sum(
          1 for hb in filtered
          if str(hb.get("status", "")).strip().upper() == "OK"
      )
      availability = round(ok_count / len(filtered), 4)

      latencies = sorted(hb.get("latencyMs", 0) for hb in filtered)
      n = len(latencies)
      idx = math.ceil(n * 0.95) - 1
      p95 = latencies[max(idx, 0)]

      result["sloOutput"] = {
          "availability": availability,
          "p95LatencyMs": p95,
      }

  return result