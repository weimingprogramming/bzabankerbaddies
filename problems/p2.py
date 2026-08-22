import base64
import json
import math
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

PRIORITY_MAP = {
    "LOW": 1, "NORMAL": 2, "MED": 2, "MEDIUM": 2,
    "HIGH": 3, "CRITICAL": 4, "URGENT": 4,
}


def _decode_b64(raw: str) -> str:
    """Decode a base64 string (standard or URL-safe) to UTF-8."""
    b64 = raw.replace('\n', '').replace('\r', '').replace(' ', '')
    b64 = b64.replace('-', '+').replace('_', '/')
    rem = len(b64) % 4
    if rem > 0:
        b64 += "=" * (4 - rem)
    return base64.b64decode(b64, validate=True).decode("utf-8")


def _safe_dict(val: Any) -> dict:
    """Return val if it's a dict, else empty dict."""
    return val if isinstance(val, dict) else {}


def _extract_adapt(data: dict) -> dict:
    """
    Extract adaptOutput fields from data, handling:
    - V1: {adaptInput: {user: {id, fullName}, action, metadata: {priority}}}
    - V2 flat: {adaptInput: {id, name, action, priority}}
    - Root flat: {user: {id, fullName}, action, metadata: {priority}}
    - Fully flat: {id, name, action, priority}
    """
    adapt_input = _safe_dict(data.get("adaptInput"))

    # V1 nested user object — check adaptInput.user first, then root user
    user_obj = _safe_dict(adapt_input.get("user")) or _safe_dict(data.get("user"))

    # V1 nested metadata — check adaptInput.metadata first, then root metadata
    metadata_obj = _safe_dict(adapt_input.get("metadata")) or _safe_dict(data.get("metadata"))

    # ID: user.id -> adaptInput.id -> root id
    user_id = user_obj.get("id") or adapt_input.get("id") or data.get("id") or ""

    # Name: user.fullName -> user.name -> adaptInput.fullName -> adaptInput.name -> root
    user_name = (
        user_obj.get("fullName") or user_obj.get("name") or
        adapt_input.get("fullName") or adapt_input.get("name") or
        data.get("fullName") or data.get("name") or ""
    )

    # Action: adaptInput.action -> root action
    action_raw = adapt_input.get("action") or data.get("action") or ""
    action = str(action_raw).strip().lower()

    # Priority: metadata.priority -> adaptInput.priority -> root priority
    priority_raw = metadata_obj.get("priority")
    if priority_raw is None:
        priority_raw = adapt_input.get("priority")
    if priority_raw is None:
        priority_raw = data.get("priority")

    if isinstance(priority_raw, str):
        priority = PRIORITY_MAP.get(priority_raw.strip().upper(), 1)
    elif isinstance(priority_raw, (int, float)) and not math.isnan(priority_raw) and not math.isinf(priority_raw):
        priority = int(priority_raw)
    else:
        priority = 1

    return {
        "id": str(user_id) if user_id else "",
        "name": str(user_name) if user_name else "",
        "action": action,
        "priority": priority,
    }


def _compute_slo(heartbeats: List[dict], slo_query: dict) -> dict:
    """Compute SLO metrics from heartbeats filtered by sloQuery."""
    target_service = slo_query.get("service", "")
    since_raw = slo_query.get("since")

    try:
        since = int(since_raw) if since_raw is not None else None
    except (ValueError, TypeError):
        since = None

    filtered = []
    for hb in heartbeats:
        if not isinstance(hb, dict):
            continue
        if target_service and hb.get("service") != target_service:
            continue
        if since is not None:
            try:
                hb_ts = int(hb.get("timestamp", 0))
            except (ValueError, TypeError):
                continue
            if hb_ts < since:
                continue
        filtered.append(hb)

    if not filtered:
        return {"availability": 0.0, "p95LatencyMs": 0}

    ok_count = sum(
        1 for hb in filtered
        if str(hb.get("status", "")).strip().upper() == "OK"
    )
    availability = ok_count / len(filtered)

    latencies = []
    for hb in filtered:
        try:
            lat = int(hb.get("latencyMs", 0))
            if lat >= 0:
                latencies.append(lat)
        except (ValueError, TypeError):
            continue

    if not latencies:
        return {"availability": availability, "p95LatencyMs": 0}

    latencies.sort()
    n = len(latencies)
    idx = math.ceil(n * 0.95) - 1
    p95 = latencies[max(0, idx)]

    return {"availability": availability, "p95LatencyMs": p95}


@router.post("/solve")
async def solve_problem_2(request: Request) -> Dict[str, Any]:
    # Accept any body shape — raw JSON or {payload: "..."}
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object")

    # Decode payload if present
    data = body
    if "payload" in body:
        payload_val = body["payload"]
        if isinstance(payload_val, dict):
            data = payload_val
        elif isinstance(payload_val, str):
            raw = payload_val.strip()
            try:
                if raw.startswith(("{", "[")):
                    data = json.loads(raw)
                else:
                    decoded = _decode_b64(raw)
                    # Handle potential nested base64
                    decoded = decoded.strip()
                    if decoded.startswith(("{", "[")):
                        data = json.loads(decoded)
                    else:
                        data = json.loads(_decode_b64(decoded))
            except Exception:
                raise HTTPException(status_code=400, detail="Could not decode payload")

    # Build adaptOutput
    adapt_output = _extract_adapt(data)

    # Build sloOutput — always include it per the spec's "combined response"
    heartbeats = data.get("heartbeats")
    if not isinstance(heartbeats, list):
        heartbeats = []

    slo_query = data.get("sloQuery")
    if not isinstance(slo_query, dict):
        slo_query = {}

    slo_output = _compute_slo(heartbeats, slo_query)

    return {
        "adaptOutput": adapt_output,
        "sloOutput": slo_output,
    }
