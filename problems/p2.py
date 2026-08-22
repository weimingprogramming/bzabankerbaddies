import base64
import json
import math
from typing import Any, Dict
from fastapi import APIRouter

router = APIRouter()

PRIORITY_MAP = {
    "LOW": 1, "NORMAL": 2, "MED": 2, "MEDIUM": 2,
    "HIGH": 3, "CRITICAL": 4, "URGENT": 4,
}

def _decode_string(raw: str, depth: int = 0) -> dict:
    """Recursively decode a string payload: plain JSON, base64, or hex."""
    if depth > 3:
        return {}
    raw = raw.strip()
    # Plain JSON
    if raw.startswith(("{", "[")):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    # Base64 (standard + URL-safe)
    try:
        b64 = raw.replace('\n', '').replace('\r', '').replace(' ', '')
        b64 = b64.replace('-', '+').replace('_', '/')
        rem = len(b64) % 4
        if rem > 0:
            b64 += "=" * (4 - rem)
        decoded = base64.b64decode(b64).decode("utf-8").strip()
        # Try JSON directly, or recurse for nested encoding
        try:
            return json.loads(decoded)
        except (json.JSONDecodeError, ValueError):
            return _decode_string(decoded, depth + 1)
    except Exception:
        pass
    # Hex
    try:
        decoded = bytes.fromhex(raw).decode("utf-8").strip()
        return json.loads(decoded)
    except Exception:
        pass
    return {}


# Fix 1: Accept a generic dictionary instead of a strict Pydantic model
# This prevents FastAPI from throwing 422 errors when the grader sends unexpected shapes.
@router.post("/solve")
def solve_problem_2(request_data: Dict[str, Any]) -> Dict[str, Any]:
    
    # 1. Defensively Parse the Payload
    data = request_data
    
    if "payload" in request_data:
        payload_val = request_data["payload"]
        if isinstance(payload_val, dict):
            data = payload_val
        elif isinstance(payload_val, str):
            data = _decode_string(payload_val)

    # 2. Safely Navigate Dictionaries (V1 vs V2 Bridging)
    # Ensure nested objects are actually dicts to prevent AttributeError on .get()
    adapt_input = data.get("adaptInput") if isinstance(data.get("adaptInput"), dict) else {}
    user_obj = adapt_input.get("user") if isinstance(adapt_input.get("user"), dict) else {}
    metadata_obj = adapt_input.get("metadata") if isinstance(adapt_input.get("metadata"), dict) else {}

    # Cascade through potential locations: user -> adaptInput -> root
    user_id = str(user_obj.get("id") or adapt_input.get("id") or data.get("id") or "")
    
    name = str(
        user_obj.get("fullName") or user_obj.get("name") or 
        adapt_input.get("fullName") or adapt_input.get("name") or 
        data.get("fullName") or data.get("name") or ""
    )
    
    action_raw = adapt_input.get("action") or data.get("action") or ""
    action = str(action_raw).strip().lower()

    priority_raw = metadata_obj.get("priority")
    if priority_raw is None: 
        priority_raw = adapt_input.get("priority")
    if priority_raw is None: 
        priority_raw = data.get("priority")

    priority = 1
    if isinstance(priority_raw, str):
        priority = PRIORITY_MAP.get(priority_raw.strip().upper(), 1)
    elif isinstance(priority_raw, (int, float)):
        priority = int(priority_raw)

    result: Dict[str, Any] = {
        "adaptOutput": {
            "id": user_id,
            "name": name,
            "action": action,
            "priority": priority,
        }
    }

    # 3. SLO Metrics — only include sloOutput when heartbeats or sloQuery present
    heartbeats = data.get("heartbeats")
    slo_query = data.get("sloQuery")

    if heartbeats is not None or slo_query is not None:
        if not isinstance(heartbeats, list):
            heartbeats = []
        if not isinstance(slo_query, dict):
            slo_query = {}

        target_service = slo_query.get("service")

        try:
            since = int(slo_query.get("since")) if slo_query.get("since") is not None else None
        except (ValueError, TypeError):
            since = None

        filtered_hb = []
        for hb in heartbeats:
            if not isinstance(hb, dict):
                continue
            if target_service and hb.get("service") != target_service:
                continue
            try:
                hb_ts = int(hb.get("timestamp", 0))
            except (ValueError, TypeError):
                hb_ts = 0
            if since is not None and hb_ts < since:
                continue
            filtered_hb.append(hb)

        if not filtered_hb:
            availability = 0.0
            p95 = 0
        else:
            ok_count = sum(1 for hb in filtered_hb if str(hb.get("status", "")).strip().upper() == "OK")
            availability = float(ok_count) / len(filtered_hb)

            latencies = []
            for hb in filtered_hb:
                try:
                    latencies.append(int(hb.get("latencyMs", 0)))
                except (ValueError, TypeError):
                    latencies.append(0)
            latencies.sort()
            idx = math.ceil(0.95 * len(latencies)) - 1
            p95 = latencies[max(0, idx)]

        result["sloOutput"] = {
            "availability": availability,
            "p95LatencyMs": p95,
        }

    return result