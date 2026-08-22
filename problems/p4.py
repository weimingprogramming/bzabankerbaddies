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

# Lazy-load: defer to first retrieve call so server starts fast
_index_ready = False

def _ensure_index():
    global _index_ready
    if not _index_ready:
        _precompute_index()
        _index_ready = True

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
    """RAG retrieval: fetch the most relevant study-material chunks for a query.

    Scoring layers (applied per chunk):
      1. Exact substring match bonus  – the query appears verbatim in the chunk
      2. Bigram TF-IDF overlap        – rewards phrase-level matches
      3. Unigram TF-IDF overlap       – rewards individual keyword matches
      4. Fuzzy / stem-prefix matching  – catches morphological variants
      5. Neighbor boost               – adjacent chunks inherit partial scores
         so topically related passages (e.g. overview → leadership) stay together
    """
    _ensure_index()
    chunks = _study_cache
    if not chunks:
        return ["Could not load study materials."]

    n_chunks = len(chunks)

    # ── query keywords ──────────────────────────────────────────────
    raw_q_words = re.findall(r'\b\w+\b', query.lower())
    q_words = {w for w in raw_q_words if w not in STOPWORDS}
    q_kw_list = [w for w in raw_q_words if w not in STOPWORDS]
    q_bigrams = set(zip(q_kw_list, q_kw_list[1:]))
    query_lower = query.lower()

    # Stem-prefix set: first 4+ chars of each query word for fuzzy matching
    q_prefixes = {w[:max(4, len(w) * 2 // 3)] for w in q_words if len(w) >= 4}

    # Semantic synonyms: map common query terms to related corpus terms
    _synonyms = {
        "leader": {"director", "commander", "chief", "head", "chair", "president", "lead"},
        "leadership": {"director", "commander", "chief", "head", "chair", "president", "lead", "governance"},
        "boss": {"director", "commander", "chief", "head", "chair", "president"},
        "outpost": {"station", "facility", "habitat", "base", "post"},
        "top": {"primary", "chief", "head", "lead", "senior", "first"},
        "post": {"position", "role", "title", "office"},
    }
    expanded_words = set(q_words)
    for qw in q_words:
        if qw in _synonyms:
            expanded_words |= _synonyms[qw]

    # ── score each chunk ────────────────────────────────────────────
    raw_scores = []
    for i, chunk in enumerate(chunks):
        score = 0.0
        chunk_lower = chunk.lower()

        # Layer 1: exact substring bonus (whole query or multi-word spans)
        if query_lower in chunk_lower:
            score += 20.0
        else:
            for start in range(len(q_kw_list) - 2):
                span = " ".join(q_kw_list[start:start + 3])
                if span in chunk_lower:
                    score += 8.0
                    break

        # Layer 2: unigram TF-IDF (with synonym expansion)
        matching = expanded_words & _chunk_word_sets[i]
        score += sum(_idf.get(w, 0) for w in matching)

        # Layer 3: bigram TF-IDF (phrase matches worth 2x)
        c_kw_list = [w for w in re.findall(r'\b\w+\b', chunk_lower) if w not in STOPWORDS]
        c_bigrams = set(zip(c_kw_list, c_kw_list[1:]))
        bigram_matches = q_bigrams & c_bigrams
        score += sum(_idf.get(a, 0) + _idf.get(b, 0) for a, b in bigram_matches) * 2

        # Layer 4: fuzzy stem-prefix matching for words not already matched
        unmatched = q_words - matching
        if unmatched and q_prefixes:
            for cw in _chunk_word_sets[i]:
                if len(cw) < 4:
                    continue
                cw_prefix = cw[:max(4, len(cw) * 2 // 3)]
                if cw_prefix in q_prefixes:
                    score += _idf.get(cw, 1.0) * 0.5

        raw_scores.append(score)

    # ── Layer 5: neighbor boost ─────────────────────────────────────
    # Chunks adjacent to high-scoring chunks get a fraction of their
    # neighbor's score, so topically-related passages stick together.
    boosted_scores = list(raw_scores)
    NEIGHBOR_FRACTION = 0.35
    for i in range(n_chunks):
        if raw_scores[i] > 0:
            if i > 0:
                boosted_scores[i - 1] = max(boosted_scores[i - 1],
                    raw_scores[i - 1] + raw_scores[i] * NEIGHBOR_FRACTION)
            if i < n_chunks - 1:
                boosted_scores[i + 1] = max(boosted_scores[i + 1],
                    raw_scores[i + 1] + raw_scores[i] * NEIGHBOR_FRACTION)

    scored = list(zip(boosted_scores, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── assemble results under token budget ─────────────────────────
    results = []
    total_chars = 0
    MAX_CHARS = 4000  # ~1000 tokens budget for retrieved context

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

# ---------------------------------------------------------------------------
# Stage 3 Tools: Working Life (Venues, Schedules, Meetings, Outings)
# ---------------------------------------------------------------------------

_inbox_cache: list = []


def _fetch_json(url: str):
    """Fetch a URL and return parsed JSON, or None on failure."""
    raw = _fetch_url(url)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _fetch_inbox() -> list:
    """Fetch and cache the android's inbox."""
    global _inbox_cache
    if _inbox_cache:
        return _inbox_cache
    data = _fetch_json(f"{BASE_URL}/inbox")
    if isinstance(data, list):
        _inbox_cache = data
    return _inbox_cache


def _parse_inbox_for_day(day: str) -> dict:
    """Parse inbox emails, return {'busy': [...], 'tentative': [...]} intervals for a given day."""
    inbox = _fetch_inbox()
    busy = []
    tentative = []
    for email in inbox:
        body = email.get("body", "")
        response = None
        when_day = None
        when_time = None
        for line in body.split("\n"):
            line = line.strip()
            if line.lower().startswith("response:"):
                response = line.split(":", 1)[1].strip().upper()
            if line.lower().startswith("when:"):
                when_part = line.split(":", 1)[1].strip()
                parts = when_part.split()
                if len(parts) >= 2:
                    when_day = parts[0]
                    when_time = parts[1] if len(parts) >= 2 else None
                    # Handle "When: Monday 09:00-10:00" format
                    # Also handle "When: Monday 09:00 - 10:00" with spaces
                    time_str = " ".join(parts[1:])
                    when_time = time_str.replace(" ", "")
        if response and when_day and when_time and when_day.lower() == day.lower():
            if response == "ACCEPTED":
                busy.append(when_time)
            elif response == "TENTATIVE":
                tentative.append(when_time)
    return {"busy": busy, "tentative": tentative}


def _time_to_min(t: str) -> int:
    """Convert HH:MM to minutes from midnight."""
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _min_to_time(m: int) -> str:
    """Convert minutes from midnight to HH:MM."""
    return f"{m // 60:02d}:{m % 60:02d}"


def _parse_interval(interval_str: str):
    """Parse 'HH:MM-HH:MM' into (start_min, end_min)."""
    parts = interval_str.split("-")
    if len(parts) == 2:
        return _time_to_min(parts[0]), _time_to_min(parts[1])
    return None


def _merge_intervals(intervals):
    """Merge overlapping (start, end) intervals."""
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _find_free_windows(busy_intervals, duration_minutes, day_start=0, day_end=1440):
    """Find all free windows of at least duration_minutes between busy intervals."""
    merged = _merge_intervals(busy_intervals)
    windows = []
    prev_end = day_start
    for s, e in merged:
        if s - prev_end >= duration_minutes:
            windows.append((prev_end, s))
        prev_end = max(prev_end, e)
    if day_end - prev_end >= duration_minutes:
        windows.append((prev_end, day_end))
    return windows


@mcp.tool()
def find_open_venues(day: str, time: str) -> str:
    """
    Find all venues that are open on a given day at a specific time.
    Use this when asked which venues/places are open or available.

    Args:
        day: The day of the week (e.g., 'Monday').
        time: The time to check in HH:MM format (e.g., '12:00').
    """
    data = _fetch_json(f"{BASE_URL}/venues/{day}")
    if not data:
        return "Could not fetch venues."
    # API returns {"day": ..., "venues": [...]} with available: [[start, end], ...]
    venues = data.get("venues", data) if isinstance(data, dict) else data
    check = _time_to_min(time)
    open_venues = []
    for v in venues:
        # Support both formats: {open, close} and {available: [[start, end], ...]}
        if "available" in v:
            for window in v["available"]:
                o = _time_to_min(window[0])
                c = _time_to_min(window[1])
                if o <= check < c:
                    open_venues.append(v["name"])
                    break
        else:
            o = _time_to_min(v.get("open", "00:00"))
            c = _time_to_min(v.get("close", "00:00"))
            if o <= check < c:
                open_venues.append(v["name"])
    if not open_venues:
        return "No venues are open at that time."
    return ", ".join(open_venues)


def _parse_schedule_response(raw: str) -> list:
    """Parse schedule API response into list of (start_min, end_min) tuples.
    Handles both formats:
      - {"busy": [["HH:MM","HH:MM"], ...]}
      - [{"start":"HH:MM","end":"HH:MM"}, ...]
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    intervals = []
    # Dict with "busy" key containing list of [start, end] arrays
    if isinstance(data, dict):
        busy = data.get("busy", [])
        for block in busy:
            if isinstance(block, list) and len(block) == 2:
                intervals.append((_time_to_min(block[0]), _time_to_min(block[1])))
            elif isinstance(block, dict):
                intervals.append((_time_to_min(block["start"]), _time_to_min(block["end"])))
    # List of objects
    elif isinstance(data, list):
        for block in data:
            if isinstance(block, list) and len(block) == 2:
                intervals.append((_time_to_min(block[0]), _time_to_min(block[1])))
            elif isinstance(block, dict):
                intervals.append((_time_to_min(block.get("start", "00:00")), _time_to_min(block.get("end", "00:00"))))
    return intervals


@mcp.tool()
def find_meeting_time(day: str, friends: str, duration_minutes: int, earliest: str = "00:00", latest: str = "23:59") -> str:
    """
    Find the earliest available meeting time for the android and a group of friends on a given day.
    Checks everyone's schedule and the android's inbox commitments to find a free window.
    Use this when asked to find a time to meet, schedule a meeting, or find when everyone is free.

    IMPORTANT: If the question specifies a time range (e.g., "between 18:00 and 23:00"),
    you MUST pass the earliest and latest parameters to constrain the search.

    Args:
        day: The day of the week (e.g., 'Monday').
        friends: Comma-separated list of friend names (e.g., 'Alice,Bob').
        duration_minutes: Required meeting duration in minutes (e.g., 60).
        earliest: Start of allowed time range in HH:MM format. MUST be set when the question specifies a range. Default '00:00'.
        latest: End of allowed time range in HH:MM format. MUST be set when the question specifies a range. Default '23:59'.
    """
    friend_list = [f.strip() for f in friends.split(",") if f.strip()]
    range_start = _time_to_min(earliest)
    range_end = _time_to_min(latest)

    # Fetch all friend schedules in parallel
    urls = [f"{BASE_URL}/schedule/{friend}/{day}" for friend in friend_list]
    all_busy = []

    with ThreadPoolExecutor(max_workers=max(len(urls), 1)) as pool:
        futures = [pool.submit(_fetch_url, u) for u in urls]
        for fut in futures:
            raw = fut.result()
            all_busy.extend(_parse_schedule_response(raw))

    # Parse android's inbox for the day
    inbox_data = _parse_inbox_for_day(day)
    tentative_intervals = []

    for t_str in inbox_data["busy"]:
        interval = _parse_interval(t_str)
        if interval:
            all_busy.append(interval)

    for t_str in inbox_data["tentative"]:
        interval = _parse_interval(t_str)
        if interval:
            tentative_intervals.append(interval)

    # Find free windows within the specified range
    free_windows = _find_free_windows(all_busy, duration_minutes, day_start=range_start, day_end=range_end)

    if not free_windows:
        return "No available meeting time found."

    # Prefer windows with no tentative conflicts
    merged_tentative = _merge_intervals(tentative_intervals) if tentative_intervals else []

    def _overlaps_tentative(window_start, window_end):
        for ts, te in merged_tentative:
            if ts < window_end and te > window_start:
                return True
        return False

    # Try to find earliest window that doesn't conflict with tentative
    for ws, we in free_windows:
        slot_end = ws + duration_minutes
        if slot_end <= we and not _overlaps_tentative(ws, slot_end):
            return f"{_min_to_time(ws)}-{_min_to_time(slot_end)}"

    # Fall back to earliest window (even if tentative conflict)
    ws, we = free_windows[0]
    slot_end = ws + duration_minutes
    return f"{_min_to_time(ws)}-{_min_to_time(slot_end)}"


@mcp.tool()
def find_meeting_point(day: str, friends: str, my_x: int, my_y: int) -> str:
    """
    Find the optimal meeting point on a 10x10 grid that minimizes total Manhattan distance
    for the android and a group of friends.
    Use this when asked where to meet, find a meeting point, or find a central location.

    Args:
        day: The day of the week (e.g., 'Monday').
        friends: Comma-separated list of friend names (e.g., 'Alice,Bob').
        my_x: The android's x-coordinate (0-9).
        my_y: The android's y-coordinate (0-9).
    """
    friend_list = [f.strip() for f in friends.split(",") if f.strip()]
    points = [(my_x, my_y)]

    # Fetch friend locations in parallel
    urls = [f"{BASE_URL}/location/{friend}/{day}" for friend in friend_list]
    with ThreadPoolExecutor(max_workers=max(len(urls), 1)) as pool:
        futures = [pool.submit(_fetch_url, u) for u in urls]
        for fut in futures:
            raw = fut.result()
            if raw:
                try:
                    loc = json.loads(raw)
                    points.append((loc["x"], loc["y"]))
                except Exception:
                    pass

    # Optimal meeting point: median of x's, median of y's
    xs = sorted(p[0] for p in points)
    ys = sorted(p[1] for p in points)

    def _total_dist(cx, cy):
        return sum(abs(cx - px) + abs(cy - py) for px, py in points)

    n = len(xs)
    if n % 2 == 1:
        best_x = xs[n // 2]
        best_y = ys[n // 2]
    else:
        # Try both median candidates and pick the better one
        x_candidates = [xs[n // 2 - 1], xs[n // 2]]
        y_candidates = [ys[n // 2 - 1], ys[n // 2]]
        best = None
        best_x, best_y = x_candidates[0], y_candidates[0]
        for cx in x_candidates:
            for cy in y_candidates:
                d = _total_dist(cx, cy)
                if best is None or d < best:
                    best = d
                    best_x, best_y = cx, cy

    # Clamp to grid
    best_x = max(0, min(9, best_x))
    best_y = max(0, min(9, best_y))

    return f"[{best_x},{best_y}]"


@mcp.tool()
def plan_outing(day: str, friends: str, my_x: int, my_y: int, duration_minutes: int, earliest: str = "00:00", latest: str = "23:59") -> str:
    """
    Plan a complete outing: find meeting time, meeting point, and a suitable open venue.
    Use this when asked to plan an outing, get-together, or hangout with friends.

    IMPORTANT: If the question specifies a time range (e.g., "between 18:00 and 23:00"),
    you MUST pass the earliest and latest parameters to constrain the search.

    Args:
        day: The day of the week (e.g., 'Monday').
        friends: Comma-separated list of friend names (e.g., 'Alice,Bob').
        my_x: The android's x-coordinate (0-9).
        my_y: The android's y-coordinate (0-9).
        duration_minutes: Required meeting duration in minutes (e.g., 60).
        earliest: Start of allowed time range in HH:MM format. MUST be set when the question specifies a range. Default '00:00'.
        latest: End of allowed time range in HH:MM format. MUST be set when the question specifies a range. Default '23:59'.
    """
    time_window = find_meeting_time(day, friends, duration_minutes, earliest, latest)
    meeting_point = find_meeting_point(day, friends, my_x, my_y)

    # Parse meeting point coordinates for venue distance calculation
    mp_match = re.match(r'\[(\d+),(\d+)\]', meeting_point)
    mp_x = int(mp_match.group(1)) if mp_match else my_x
    mp_y = int(mp_match.group(2)) if mp_match else my_y

    # Find venues open at the start of the meeting window
    meet_start = time_window.split("-")[0] if "-" in time_window else "12:00"

    # Fetch venue data directly to pick the closest open venue
    data = _fetch_json(f"{BASE_URL}/venues/{day}")
    best_venue = None
    best_dist = None
    all_open = []

    if data:
        venues = data.get("venues", data) if isinstance(data, dict) else data
        check = _time_to_min(meet_start)
        for v in venues:
            is_open = False
            if "available" in v:
                for window in v["available"]:
                    o = _time_to_min(window[0])
                    c = _time_to_min(window[1])
                    if o <= check < c:
                        is_open = True
                        break
            else:
                o = _time_to_min(v.get("open", "00:00"))
                c = _time_to_min(v.get("close", "00:00"))
                if o <= check < c:
                    is_open = True

            if is_open:
                all_open.append(v["name"])
                vx = v.get("x", 0)
                vy = v.get("y", 0)
                dist = abs(vx - mp_x) + abs(vy - mp_y)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_venue = v["name"]

    venue_result = best_venue if best_venue else "No venues are open at that time."

    return f"{time_window}, {meeting_point}, {venue_result}"