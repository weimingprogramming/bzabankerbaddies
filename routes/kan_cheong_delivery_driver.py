from __future__ import annotations

import heapq
from datetime import datetime, timezone
from itertools import count
from typing import Any

from fastapi import APIRouter

router = APIRouter()


def _parse_time(value: str) -> float:
  parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed.timestamp()


def _format_time(timestamp: float) -> str:
  value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
  return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _traverse(
  edge_id: str,
  source: tuple[int, int],
  destination: tuple[int, int],
  base_duration: float,
  departure: float,
  obstructions: dict[tuple[str, tuple[int, int], tuple[int, int]], list[tuple[float, float, float]]],
) -> float | None:
  if base_duration == 0:
    active = obstructions.get((edge_id, source, destination), ())
    if any(start <= departure < end and factor == 0 for start, end, factor in active):
      return None
    return departure

  relevant = obstructions.get((edge_id, source, destination), ())
  elapsed = 0.0
  progress = 0.0

  while progress < 1.0:
    current = departure + elapsed
    active = [factor for start, end, factor in relevant if start <= current < end]
    speed_factor = min(active, default=1.0)
    if speed_factor <= 0:
      return None

    next_change = min(
      (point for start, end, _ in relevant for point in (start, end) if point > current),
      default=float("inf"),
    )
    available_time = next_change - current
    remaining = (1.0 - progress) * base_duration / speed_factor
    if remaining <= available_time:
      return current + remaining

    progress += available_time * speed_factor / base_duration
    elapsed += available_time

  return departure + elapsed


def _solve_case(data: dict[str, Any]) -> dict[str, Any]:
  start = tuple(data["start_coordinate"])
  end = tuple(data["end_coordinate"])
  departure = _parse_time(data["start_time"])

  adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], str, float]]] = {}
  for edge in data.get("edges", []):
    node1 = tuple(edge["node1"])
    node2 = tuple(edge["node2"])
    edge_id = edge["edge_id"]
    duration = float(edge["base_duration_sec"])
    adjacency.setdefault(node1, []).append((node2, edge_id, duration))
    adjacency.setdefault(node2, []).append((node1, edge_id, duration))

  obstruction_map: dict[tuple[str, tuple[int, int], tuple[int, int]], list[tuple[float, float, float]]] = {}
  for obstruction in data.get("obstructions", []):
    edge = obstruction["edge"]
    key = (obstruction["edge_id"], tuple(edge["from"]), tuple(edge["to"]))
    obstruction_map.setdefault(key, []).append(
      (
        _parse_time(obstruction["start_time"]),
        _parse_time(obstruction["end_time"]),
        float(obstruction["speed_factor"]),
      )
    )

  if start == end:
    return {"total_duration_sec": 0, "arrival_time": _format_time(departure), "path": []}

  # Later labels are necessary because arriving later can avoid a block, and waiting is forbidden.
  queue: list[tuple[float, int, tuple[int, int], tuple[str, ...]]] = []
  sequence = count()
  heapq.heappush(queue, (departure, next(sequence), start, ()))
  labels: dict[tuple[int, int], set[float]] = {start: {departure}}
  max_labels_per_node = 4096

  while queue:
    arrival, _, node, path = heapq.heappop(queue)
    if node == end:
      duration = arrival - departure
      return {
        "total_duration_sec": round(duration),
        "arrival_time": _format_time(arrival),
        "path": list(path),
      }

    for next_node, edge_id, base_duration in adjacency.get(node, []):
      next_arrival = _traverse(edge_id, node, next_node, base_duration, arrival, obstruction_map)
      if next_arrival is None:
        continue
      known = labels.setdefault(next_node, set())
      if next_arrival in known or len(known) >= max_labels_per_node:
        continue
      known.add(next_arrival)
      heapq.heappush(queue, (next_arrival, next(sequence), next_node, path + (edge_id,)))

  return {"total_duration_sec": None, "arrival_time": None, "path": []}


@router.post("/kan-cheong-delivery-driver")
def solve_delivery_driver(batch: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {case_id: _solve_case(case) for case_id, case in batch.items()}