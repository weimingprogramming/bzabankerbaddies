import base64
import numpy as np
import cv2
from fastmcp import FastMCP
import re
import urllib.request
import json
import heapq
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

# Initialize the MCP Server
mcp = FastMCP("NurseryServer")

# ---------------------------------------------------------------------------
# Constants & Caches
# ---------------------------------------------------------------------------

BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"

STUDY_MATERIAL_URLS = [
    f"{BASE_URL}/study-materials/1",
    f"{BASE_URL}/study-materials/2",
    f"{BASE_URL}/study-materials/3",
    f"{BASE_URL}/study-materials/4",
    f"{BASE_URL}/study-materials/5",
]

_study_cache: List[str] = []
_graph_cache: Dict[str, dict] = {}
_chunk_word_sets: List[set] = []
_idf: Dict[str, float] = {}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", 
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", 
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", 
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", 
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", 
    "own", "s", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", 
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", 
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "will", "with", "you", "your", "yours", "yourself", "yourselves", "many", "much", "roughly"
}

# ---------------------------------------------------------------------------
# Stage 1 Tools (REQUIRED - Do not remove)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_agent_name() -> str:
    """Returns the name of the agent. Use this if asked for your name."""
    return "Render-Baby"

@mcp.tool()
def calculate_math(expression: str) -> float:
    """Evaluates a full mathematical expression string (e.g., '2 + 3 * 5')."""
    try:
        expr = expression.replace("x", "*").replace("X", "*")
        expr = re.sub(r'[^0-9\+\-\*\/\.\(\)\ ]', '', expr)
        return float(eval(expr))
    except Exception:
        return 0.0

@mcp.tool()
def identify_shape(image_b64: str) -> str:
    """Identifies the shape from a base64 encoded PNG image string ('rectangle', 'triangle', or 'circle')."""
    try:
        img_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if len(img.shape) == 3 and img.shape[2] == 4:
            mask = img[:, :, 3] 
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "circle"
            
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        vertices = len(approx)
        if vertices == 3: return "triangle"
        elif vertices == 4: return "rectangle"
        else: return "circle"
    except Exception:
        return "circle"

# ---------------------------------------------------------------------------
# Core Fetch Helpers (Optimized for Speed)
# ---------------------------------------------------------------------------

def _fetch_url(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        # Max 7s timeout to ensure we don't cross the 10s evaluation threshold
        with urllib.request.urlopen(req, timeout=7) as resp:
            return resp.read().decode('utf-8')
    except Exception:
        return ""

def _load_study_materials() -> List[str]:
    """Fetch all study materials dynamically on first call, then cache."""
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

    raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', content) if c.strip()]
    if len(raw_chunks) < 5:
        raw_chunks = [c.strip() for c in content.split('\n') if c.strip() and len(c.strip()) > 10]

    chunks = []
    for chunk in raw_chunks:
        if chunks and re.match(r'^#{1,4}\s', chunks[-1]) and len(chunks[-1]) < 80:
            chunks[-1] = chunks[-1] + "\n" + chunk
        else:
            chunks.append(chunk)

    _study_cache = chunks
    return _study_cache

def _precompute_index():
    """Pre-compute IDF index at startup so retrieve calls are instant."""
    global _chunk_word_sets, _idf
    chunks = _load_study_materials()
    if not chunks:
        return
    N = len(chunks)
    _chunk_word_sets.clear()
    df: Dict[str, int] = {}
    for c in chunks:
        ws = set(re.findall(r'\b\w+\b', c.lower())) - STOPWORDS
        _chunk_word_sets.append(ws)
        for w in ws:
            df[w] = df.get(w, 0) + 1
    _idf.clear()
    _idf.update({w: math.log((N + 1) / (freq + 1)) + 1.0 for w, freq in df.items()})

# Pre-load at import time so first retrieve call is instant
_precompute_index()

def _fetch_graph(map_id: str) -> dict:
    if map_id in _graph_cache:
        return _graph_cache[map_id]
    url = f"{BASE_URL}/graph?map_id={map_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            _graph_cache[map_id] = data
            return data
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Stage 2 & 3 Tools: Recall and Navigate
# ---------------------------------------------------------------------------

def _retrieve_impl(query: str) -> List[str]:
    chunks = _study_cache
    if not chunks:
        return ["Could not load study materials."]

    # Extract query keywords
    raw_q_words = re.findall(r'\b\w+\b', query.lower())
    q_words = {w for w in raw_q_words if w not in STOPWORDS}
    q_kw_list = [w for w in raw_q_words if w not in STOPWORDS]
    q_bigrams = set(zip(q_kw_list, q_kw_list[1:]))

    # Score chunks using pre-computed IDF
    scored = []
    for i, chunk in enumerate(chunks):
        matching = q_words & _chunk_word_sets[i]
        score = sum(_idf.get(w, 0) for w in matching)

        c_kw_list = [w for w in re.findall(r'\b\w+\b', chunk.lower()) if w not in STOPWORDS]
        c_bigrams = set(zip(c_kw_list, c_kw_list[1:]))
        bigram_matches = q_bigrams & c_bigrams
        score += sum(_idf.get(a, 0) + _idf.get(b, 0) for a, b in bigram_matches) * 2

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 4. Truncate using Character Math (1 Token ≈ 4 Chars). Prevents tiktoken timeout.
    results = []
    total_chars = 0
    MAX_CHARS = 3200 # Ensures we stay strictly under the 900 token limit safely

    for score, chunk in scored:
        if score == 0 and len(results) > 0:
            break

        if total_chars + len(chunk) > MAX_CHARS:
            allowed = MAX_CHARS - total_chars
            if allowed > 100:
                results.append(chunk[:allowed] + "...")
                total_chars += allowed
            break

        results.append(chunk)
        total_chars += len(chunk)

    return results if results else ["No relevant passages found."]


@mcp.tool()
def retrieve(query: str) -> List[str]:
    """
    Recalls relevant passages from the study materials to answer a question.
    Use this tool whenever you need to answer a factual question, recall information,
    or look up any fact such as a date, name, place, event, or detail.
    Just pass the question and the tool returns the most relevant passages.

    Args:
        query: The exact question you need to answer.
    """
    return _retrieve_impl(query)


@mcp.tool()
def get_relevant_study_pass(question: str) -> List[str]:
    """
    Recalls relevant passages from the study materials to answer a question.
    Use this tool whenever you need to answer a factual question, recall information,
    or look up any fact such as a date, name, place, event, or detail.
    Just pass the question and the tool returns the most relevant passages.

    Args:
        question: The exact question you need to answer.
    """
    return _retrieve_impl(question)


@mcp.tool()
def navigate(
    current_node: str,
    destination: str,
    map_id: str,
    visited_nodes: List[str] = None,
    hops_remaining: int = 999
) -> str:
    """
    Returns the next node to move to on a journey from current_node to destination.
    CRITICAL: Use this tool IMMEDIATELY when asked a question like "How can I get from X to Y? map_id: Z".
    Call this tool once per step of your journey until you reach the destination.
    If the question says "in N moves" or "within N steps", pass hops_remaining=N on first call,
    then decrement by 1 on each subsequent call.

    Args:
        current_node: The node you are currently at (e.g., 'A').
        destination: The node you need to reach (e.g., 'D').
        map_id: The map identifier string provided in the question.
        visited_nodes: Nodes already visited on this journey (to avoid revisiting).
        hops_remaining: Maximum number of moves remaining. If the question specifies a move limit, use it. Decrement by 1 each call.
    """
    try:
        if visited_nodes is None:
            visited_nodes = []
        elif isinstance(visited_nodes, str):
            visited_nodes = [x.strip() for x in visited_nodes.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",") if x.strip()]

        hops_left = 999
        try:
            hops_left = int(hops_remaining)
        except Exception:
            pass

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
    except Exception as e:
        return f"ERROR: {str(e)}"