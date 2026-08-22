from fastmcp import FastMCP
import re
import urllib.request
import json
import heapq
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

mcp = FastMCP("NurseryServer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"

STUDY_MATERIAL_URLS = [
    f"{BASE_URL}/study-materials/1",
    f"{BASE_URL}/study-materials/2",
    f"{BASE_URL}/study-materials/3",
    f"{BASE_URL}/study-materials/4",
    f"{BASE_URL}/study-materials/5",
]

# ---------------------------------------------------------------------------
# Caches (persist across calls within a single server process)
# ---------------------------------------------------------------------------

_study_cache: List[str] = []       # cached chunks from all study materials
_graph_cache: Dict[str, dict] = {} # map_id -> graph data


def _fetch_url(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.read().decode('utf-8')
    except Exception:
        return ""


def _load_study_materials() -> List[str]:
    """Fetch all study materials in parallel, chunk them, and cache."""
    global _study_cache
    if _study_cache:
        return _study_cache

    content = ""
    with ThreadPoolExecutor(max_workers=len(STUDY_MATERIAL_URLS)) as pool:
        futures = [pool.submit(_fetch_url, u) for u in STUDY_MATERIAL_URLS]
        for f in futures:
            text = f.result()
            if text:
                content += "\n" + text

    if not content.strip():
        return []

    # Split into paragraph-level chunks for relevance matching
    chunks = [c.strip() for c in re.split(r'\n\s*\n', content) if c.strip()]
    if len(chunks) < 5:
        chunks = [c.strip() for c in content.split('\n') if c.strip() and len(c.strip()) > 10]

    _study_cache = chunks
    return _study_cache


def _fetch_graph(map_id: str) -> dict:
    """Fetch and cache graph data for a map_id."""
    if map_id in _graph_cache:
        return _graph_cache[map_id]
    url = f"{BASE_URL}/graph?map_id={map_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            _graph_cache[map_id] = data
            return data
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 1: Recall — answers questions from study materials
# ---------------------------------------------------------------------------

@mcp.tool()
def recall(question: str) -> List[str]:
    """
    Recalls relevant passages from the study materials to answer a question.
    Use this tool whenever you need to answer a factual question, recall information,
    or look up any fact such as a date, name, place, event, or detail.
    Just pass the question and the tool returns the most relevant passages.

    Args:
        question: The question you need to answer.
    """
    chunks = _load_study_materials()
    if not chunks:
        return ["Could not load study materials."]

    q_words = set(re.findall(r'\w+', question.lower()))

    def get_bigrams(s: str):
        words = re.findall(r'\w+', s.lower())
        return set(zip(words, words[1:]))

    q_bigrams = get_bigrams(question)

    scored = []
    for chunk in chunks:
        c_words = set(re.findall(r'\w+', chunk.lower()))
        c_bigrams = get_bigrams(chunk)
        score = len(q_words & c_words) + 3 * len(q_bigrams & c_bigrams)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    results = []
    total_tokens = 0

    for score, chunk in scored:
        chunk_tokens = len(enc.encode(chunk))

        if total_tokens + chunk_tokens > 880:
            allowed = 880 - total_tokens
            if allowed > 20:
                encoded = enc.encode(chunk)
                truncated = enc.decode(encoded[:allowed])
                results.append(truncated)
                total_tokens += len(enc.encode(truncated))
            break

        results.append(chunk)
        total_tokens += chunk_tokens

    return results if results else ["No relevant passages found."]


# ---------------------------------------------------------------------------
# Tool 2: Navigate — finds the next hop on a weighted directed graph
# ---------------------------------------------------------------------------

@mcp.tool()
def navigate(
    current_node: str,
    destination: str,
    map_id: str,
    visited_nodes: List[str] = None,
    hops_left: int = 999
) -> str:
    """
    Returns the next node to move to on a journey from current_node to destination.
    This tool fetches the map automatically using the map_id.
    Call this tool once per step of your journey.

    Args:
        current_node: The node you are currently at.
        destination: The node you need to reach.
        map_id: The map identifier from the question.
        visited_nodes: Nodes already visited on this journey (to avoid revisiting).
        hops_left: Remaining moves allowed (pass this if given a hop limit).
    """
    if visited_nodes is None:
        visited_nodes = []

    data = _fetch_graph(map_id)
    if "error" in data:
        return f"ERROR: Could not fetch map: {data['error']}"

    adjacency = data.get("adjacency", {})
    tolls = data.get("tolls", {})

    if not adjacency:
        return "ERROR: Map has no adjacency data."

    visited_set = set(visited_nodes)
    visited_set.add(current_node)

    pq = [(0.0, 0, current_node, [current_node])]
    best_states: Dict[str, List] = {}

    while pq:
        cost, hops, u, path = heapq.heappop(pq)

        if u == destination:
            return path[1] if len(path) > 1 else u

        if hops >= hops_left:
            continue

        dominated = False
        for prev_cost, prev_hops in best_states.get(u, []):
            if prev_cost <= cost and prev_hops <= hops:
                dominated = True
                break
        if dominated:
            continue

        best_states.setdefault(u, []).append((cost, hops))
        path_set = set(path)

        for v, weight in adjacency.get(u, {}).items():
            if v in visited_set and v != destination:
                continue
            if v in path_set:
                continue

            next_cost = cost + float(weight) + float(tolls.get(v, 0.0))
            heapq.heappush(pq, (next_cost, hops + 1, v, path + [v]))

    return "ERROR: No valid path found."
