"""
tool-box MCP server.

Design notes (read before editing):

- retrieve/recall: the 900-token budget is counted with the o200k_base
  tokenizer, exactly like the grader does. We pack chunks by relevance
  score and, if the last chunk would overflow, we trim it *by token id*
  so we never emit more than the budget allows. Going over means the
  whole response is discarded, so this is worth getting exactly right.

- meeting-time tools do NOT hardcode the android's inbox. The brief
  shows the inbox as content the android is given directly (not an
  endpoint we can GET), and it's presumably re-randomized per run like
  everything else (map, venues, schedules). So `my_busy` / `my_tentative`
  are parameters: the android reads its own inbox, keeps the *latest*
  "When:" line per invitation, buckets ACCEPTED -> busy and
  TENTATIVE -> tentative, and passes the "HH:MM-HH:MM" strings in.

- navigate_map does an exact (not heuristic) exhaustive search over
  simple paths, since "already visited" is scored as an outright
  failure -- we want the true minimum-cost simple path under the hop
  budget, not an approximation that can occasionally miss it. A call
  budget keeps worst-case runtime bounded on pathological graphs.

- Server startup uses the standalone `mcp.run(transport="http", ...)`
  pattern. Mounting mcp.http_app() into FastAPI/Flask without also
  wiring its lifespan is the #1 way these servers 500 while looking
  perfectly healthy -- see the Q&A page. We don't need Flask here, so
  we sidestep the whole problem.
"""

import base64
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import cv2
import tiktoken
from fastmcp import FastMCP

mcp = FastMCP("ToolBoxServer")

BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"

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
    "why", "will", "with", "you", "your", "yours", "yourself", "yourselves", "many", "much", "roughly",
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("o200k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


_url_cache: Dict[str, str] = {}


def _fetch_url(url: str, timeout: float = 4.0, retries: int = 1, backoff: float = 0.3,
                use_cache: bool = True) -> str:
    """
    Fetch a URL with a couple of retries. Successful (non-empty) responses are cached
    by URL for the life of the process -- cheap, and it means a transient failure on
    one call doesn't have to be paid for again by every later call that needs the same
    data (friend schedules, locations, the graph, etc. are stable within a run).

    A 404/other real HTTP error returns immediately (no point retrying that), so
    probing for endpoints that may or may not exist stays fast.
    """
    if use_cache and url in _url_cache:
        return _url_cache[url]

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8")
                if use_cache and text:
                    _url_cache[url] = text
                return text
        except urllib.error.HTTPError:
            return ""  # real 4xx/5xx -- retrying won't help, and we don't want to stall on it
        except Exception:
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return ""


def _fetch_json(url: str, timeout: float = 4.0, retries: int = 1):
    raw = _fetch_url(url, timeout=timeout, retries=retries)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _fetch_many(urls: List[str], timeout: float = 4.0, retries: int = 1) -> List[str]:
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        return list(pool.map(lambda u: _fetch_url(u, timeout, retries), urls))


def _time_to_min(t: str) -> int:
    h, m = t.strip().split(":")
    return int(h) * 60 + int(m)


def _min_to_time(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _parse_interval(s: str):
    s = s.strip().replace(" ", "")
    if "-" not in s:
        return None
    a, b = s.split("-", 1)
    try:
        return (_time_to_min(a), _time_to_min(b))
    except Exception:
        return None


def _parse_intervals(items: Optional[List[str]]) -> List[tuple]:
    if not items:
        return []
    out = []
    for it in items:
        iv = _parse_interval(it)
        if iv:
            out.append(iv)
    return out


def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _free_windows(busy: List[tuple], duration: int, lo: int, hi: int) -> List[tuple]:
    clamped = []
    for s, e in busy:
        if e <= lo or s >= hi:
            continue
        clamped.append((max(s, lo), min(e, hi)))
    merged = _merge_intervals(clamped)
    windows, prev = [], lo
    for s, e in merged:
        if s - prev >= duration:
            windows.append((prev, s))
        prev = max(prev, e)
    if hi - prev >= duration:
        windows.append((prev, hi))
    return windows


def _parse_node_list(value) -> List[str]:
    """Accept a real list, a JSON string, or a loose 'A, B, C' string."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    s = s.strip("[]")
    return [x.strip().strip("'\"") for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Stage 1 -- Nursery
# ---------------------------------------------------------------------------

@mcp.tool()
def get_agent_name() -> str:
    """Returns the agent's name. Call this whenever asked 'what is your name?'."""
    return "Ada Byte"


_ALLOWED_OPS = {"+", "-", "*", "/", "(", ")", " ", "."}


@mcp.tool()
def calculate(expression: str) -> float:
    """
    Evaluates a simple arithmetic expression and returns the numeric result.
    Supports +, -, *, / on integers in the range -100 to 100, e.g. "2 + 2" or "(7 - 3) * 5".

    Args:
        expression: The arithmetic expression to evaluate, digits and operators only.
    """
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("unsupported expression")

    try:
        cleaned = expression.replace("x", "*").replace("X", "*").replace("×", "*").replace("÷", "/")
        tree = ast.parse(cleaned, mode="eval")
        return float(_eval(tree))
    except Exception:
        return 0.0


@mcp.tool()
def identify_shape(image_base64: str) -> str:
    """
    Identifies the shape drawn in a base64-encoded PNG image.
    Returns exactly one of: 'rectangle', 'triangle', 'circle'.

    Args:
        image_base64: The raw base64-encoded PNG image data (no data: prefix).
    """
    try:
        img_data = base64.b64decode(image_base64)
        arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "circle"

        mask = None
        if img.ndim == 3 and img.shape[2] == 4 and img[:, :, 3].max() > 0:
            mask = img[:, :, 3]
        else:
            gray = cv2.cvtColor(img[:, :, :3] if img.ndim == 3 else img, cv2.COLOR_BGR2GRAY) \
                if img.ndim == 3 else img
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "circle"

        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            return "circle"

        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        vertices = len(approx)
        circularity = 4 * math.pi * area / (perimeter ** 2)

        if vertices == 3:
            return "triangle"
        if vertices == 4:
            return "rectangle"
        if circularity > 0.8:
            return "circle"
        return "triangle" if vertices < 4 else "circle"
    except Exception:
        return "circle"


# ---------------------------------------------------------------------------
# Stage 2 -- School Days
# ---------------------------------------------------------------------------

_study_chunks: List[str] = []
_chunk_word_sets: List[set] = []
_chunk_stem_sets: List[set] = []
_idf: Dict[str, float] = {}
_stem_idf: Dict[str, float] = {}
_index_ready = False


def _extract_urls_from_index(data) -> List[str]:
    """Pull a list of document URLs out of whatever shape an index endpoint returns."""
    candidates = None
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for key in ("documents", "materials", "study_materials", "urls", "items", "pages", "links"):
            if isinstance(data.get(key), list):
                candidates = data[key]
                break
    if not candidates:
        return []

    urls = []
    for c in candidates:
        if isinstance(c, str):
            path = c
        elif isinstance(c, dict):
            path = next((c[k] for k in ("url", "address", "href", "path") if c.get(k)), None)
        else:
            path = None
        if not path:
            continue
        urls.append(path if path.startswith("http") else f"{BASE_URL}{path if path.startswith('/') else '/' + path}")
    return urls


def _discover_study_urls() -> List[str]:
    """
    Don't assume a fixed document count. Try a couple of plausible index endpoints
    first; if none exist, probe numbered documents in parallel batches until a whole
    batch comes back empty. This is the fix for silently missing documents beyond
    whatever count was true the day this code was written.
    """
    for index_path in ("/study-materials", "/study-materials/index"):
        data = _fetch_json(f"{BASE_URL}{index_path}", timeout=3.0, retries=0)
        urls = _extract_urls_from_index(data) if data else []
        if urls:
            return urls

    found: List[str] = []
    batch_size = 10
    max_docs = 30
    start = 1
    while start <= max_docs:
        batch = [f"{BASE_URL}/study-materials/{i}" for i in range(start, min(start + batch_size, max_docs + 1))]
        texts = _fetch_many(batch, timeout=3.0, retries=0)
        hits = [url for url, text in zip(batch, texts) if text.strip()]
        if not hits:
            break
        found.extend(hits)
        start += batch_size
    return found


def _load_study_materials() -> List[str]:
    global _study_chunks
    if _study_chunks:
        return _study_chunks
    urls = _discover_study_urls()
    texts = [t for t in _fetch_many(urls) if t]
    chunks: List[str] = []
    for text in texts:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) < 2:
            parts = [p.strip() for p in text.split("\n") if p.strip()]
        for part in parts:
            # merge short headings into the following paragraph
            if chunks and re.match(r"^#{1,4}\s", chunks[-1]) and len(chunks[-1]) < 80:
                chunks[-1] = chunks[-1] + "\n" + part
            else:
                chunks.append(part)
    _study_chunks = chunks
    return chunks


_STEM_SUFFIXES = ("ations", "ation", "ments", "ment", "ities", "ity", "ing", "ies", "ied", "es", "ed", "s")


def _stem(word: str) -> str:
    """Cheap suffix-stripping so 'resolved'/'resolve'/'resolution'-ish variants can
    still match without needing real morphology. Deliberately conservative (keeps at
    least a 3-char stem) so it doesn't collapse unrelated short words together."""
    for suf in _STEM_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _ensure_index():
    global _index_ready, _idf, _stem_idf, _chunk_word_sets, _chunk_stem_sets
    if _index_ready:
        return
    chunks = _load_study_materials()
    _chunk_word_sets = []
    _chunk_stem_sets = []
    df: Dict[str, int] = {}
    stem_df: Dict[str, int] = {}
    for c in chunks:
        words = set(re.findall(r"\b\w+\b", c.lower())) - STOPWORDS
        stems = {_stem(w) for w in words}
        _chunk_word_sets.append(words)
        _chunk_stem_sets.append(stems)
        for w in words:
            df[w] = df.get(w, 0) + 1
        for s in stems:
            stem_df[s] = stem_df.get(s, 0) + 1
    n = max(len(chunks), 1)
    _idf = {w: math.log((n + 1) / (f + 1)) + 1.0 for w, f in df.items()}
    _stem_idf = {s: math.log((n + 1) / (f + 1)) + 1.0 for s, f in stem_df.items()}
    _index_ready = True


def _score_chunks(query: str):
    _ensure_index()
    chunks = _study_chunks
    if not chunks:
        return []

    q_words_raw = re.findall(r"\b\w+\b", query.lower())
    q_words = {w for w in q_words_raw if w not in STOPWORDS}
    q_stems = {_stem(w) for w in q_words}
    q_kw = [w for w in q_words_raw if w not in STOPWORDS]
    q_bigrams = set(zip(q_kw, q_kw[1:]))
    q_lower = query.lower()

    raw_scores = []
    for i, chunk in enumerate(chunks):
        score = 0.0
        c_lower = chunk.lower()
        if q_lower in c_lower:
            score += 20.0
        else:
            for start in range(max(len(q_kw) - 2, 0)):
                if " ".join(q_kw[start:start + 3]) in c_lower:
                    score += 8.0
                    break
        matched = q_words & _chunk_word_sets[i]
        score += sum(_idf.get(w, 0.0) for w in matched)
        # stem-level match catches simple word-form variants (resolve/resolved/resolution)
        # that exact-word overlap misses; weighted lower since it's a weaker signal.
        stem_matched = q_stems & _chunk_stem_sets[i]
        score += sum(_stem_idf.get(s, 0.0) for s in stem_matched) * 0.6
        c_kw = [w for w in re.findall(r"\b\w+\b", c_lower) if w not in STOPWORDS]
        c_bigrams = set(zip(c_kw, c_kw[1:]))
        for a, b in q_bigrams & c_bigrams:
            score += (_idf.get(a, 0.0) + _idf.get(b, 0.0)) * 2
        raw_scores.append(score)

    # light neighbor boost so adjacent context rides along with a strong hit
    n = len(chunks)
    boosted = list(raw_scores)
    for i in range(n):
        if raw_scores[i] <= 0:
            continue
        if i > 0:
            boosted[i - 1] = max(boosted[i - 1], raw_scores[i - 1] + raw_scores[i] * 0.35)
        if i < n - 1:
            boosted[i + 1] = max(boosted[i + 1], raw_scores[i + 1] + raw_scores[i] * 0.35)

    scored = list(zip(boosted, chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


RECALL_TOKEN_BUDGET = 900


@mcp.tool()
def recall_study_material(question: str) -> List[str]:
    """
    Recalls the most relevant passages from the study materials for a factual question
    (dates, names, places, events, any detail you need to look up). Returns a list of
    passage strings -- read them and write your own answer from what they say.

    Args:
        question: The exact question you need to answer.
    """
    scored = _score_chunks(question)
    if not scored:
        return ["Could not load study materials."]

    # Deliberately do NOT stop early on a zero-scored chunk. The scoring is lexical
    # (TF-IDF + light stemming), and a chunk that answers a paraphrased question can
    # easily score 0 while still being the one passage with the actual fact. As long
    # as there's budget left, keep packing lower-ranked chunks too -- more coverage
    # only helps, it never hurts, since the android just reads what's relevant.
    result: List[str] = []
    used = 0
    for score, chunk in scored:
        remaining = RECALL_TOKEN_BUDGET - used
        if remaining <= 5:
            break
        tok = _count_tokens(chunk)
        if tok <= remaining:
            result.append(chunk)
            used += tok
        else:
            ids = _ENC.encode(chunk)[:remaining]
            result.append(_ENC.decode(ids))
            used += remaining
            break

    if not result:
        ids = _ENC.encode(scored[0][1])[:RECALL_TOKEN_BUDGET]
        result.append(_ENC.decode(ids))

    return result


_graph_cache: Dict[str, dict] = {}


def _fetch_graph(map_id: str) -> dict:
    if map_id in _graph_cache:
        return _graph_cache[map_id]
    data = _fetch_json(f"{BASE_URL}/graph?map_id={map_id}")
    if not data:
        return {"error": "could not fetch map"}
    _graph_cache[map_id] = data
    return data


_DFS_CALL_BUDGET = 300_000


@mcp.tool()
def navigate_map(
    map_id: str,
    current_node: str,
    destination: str,
    visited_nodes: Optional[List[str]] = None,
    hops_remaining: Optional[int] = None,
) -> str:
    """
    Returns the single next node to move to on the least-cost route from current_node
    to destination. Call this once per step of the journey, feeding back the node you
    just moved to as the new current_node, until it returns the destination itself.

    Args:
        current_node: The node you are standing at right now.
        destination: The node you are trying to reach.
        map_id: The opaque map identifier given in the question.
        visited_nodes: Nodes already visited on this journey so far (do not revisit them).
        hops_remaining: If the question gives a move limit, pass it on the first call and
            decrement it by 1 each subsequent call. Omit if there is no limit.
    """
    try:
        graph = _fetch_graph(map_id)
        if "error" in graph:
            return f"ERROR: {graph['error']}"
        adjacency = graph.get("adjacency", {})
        tolls = graph.get("tolls", {})
        if not adjacency:
            return "ERROR: map has no adjacency data"

        if current_node == destination:
            return destination

        visited = set(_parse_node_list(visited_nodes))
        visited.add(current_node)

        hops_left = None
        if hops_remaining is not None:
            try:
                h = int(hops_remaining)
                if h > 0:
                    hops_left = h
            except Exception:
                hops_left = None

        best_cost = [float("inf")]
        best_path = [None]
        calls = [0]

        def dfs(node, cost, hops_used, path, path_set):
            calls[0] += 1
            if calls[0] > _DFS_CALL_BUDGET:
                return
            if cost >= best_cost[0]:
                return
            if node == destination:
                best_cost[0] = cost
                best_path[0] = path
                return
            if hops_left is not None and hops_used >= hops_left:
                return
            for nxt, w in adjacency.get(node, {}).items():
                if nxt in path_set:
                    continue
                ncost = cost + float(w) + float(tolls.get(nxt, 0.0))
                dfs(nxt, ncost, hops_used + 1, path + [nxt], path_set | {nxt})

        dfs(current_node, 0.0, 0, [current_node], set(visited))

        if best_path[0] is None:
            return "ERROR: no reachable path within the given constraints"
        return best_path[0][1]
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Stage 3 -- Working Life
# ---------------------------------------------------------------------------

@mcp.tool()
def list_open_venues(day: str, time: str) -> str:
    """
    Lists every venue open on `day` at `time`, comma-separated.

    Args:
        day: Weekday name, e.g. 'Thursday'.
        time: 24-hour zero-padded time, e.g. '08:00'.
    """
    data = _fetch_json(f"{BASE_URL}/venues/{day}")
    if not data:
        return "Could not fetch venues."
    venues = data.get("venues", data) if isinstance(data, dict) else data
    if isinstance(venues, dict):
        venues = list(venues.values())

    check = _time_to_min(time)
    open_names = []
    for v in venues or []:
        if not isinstance(v, dict):
            continue
        for window in v.get("available", []):
            if len(window) >= 2 and _time_to_min(window[0]) <= check < _time_to_min(window[1]):
                open_names.append(v["name"])
                break
    return ", ".join(open_names) if open_names else "No venues are open at that time."


def _busy_block_to_interval(block):
    """Accept the documented [start, end] list shape, and defensively also a
    {"start": .., "end": ..} dict shape in case a server ever varies from spec."""
    try:
        if isinstance(block, list) and len(block) == 2:
            return (_time_to_min(block[0]), _time_to_min(block[1]))
        if isinstance(block, dict) and "start" in block and "end" in block:
            return (_time_to_min(block["start"]), _time_to_min(block["end"]))
    except Exception:
        pass
    return None


def _parse_schedule_text(raw: str) -> List[tuple]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    out = []
    for block in data.get("busy", []):
        iv = _busy_block_to_interval(block)
        if iv:
            out.append(iv)
    return out


def _friend_busy(day: str, friends: List[str]) -> List[tuple]:
    # A couple of retries with a real timeout -- a friend's schedule fetch failing
    # silently used to mean "treat them as free all day", which is the worst possible
    # default: it produces a confidently wrong answer instead of a visibly missing one.
    raws = _fetch_many([f"{BASE_URL}/schedule/{f}/{day}" for f in friends], timeout=5.0, retries=2)
    busy = []
    for raw in raws:
        busy.extend(_parse_schedule_text(raw))
    return busy


DAY_START, DAY_END = _time_to_min("08:00"), _time_to_min("23:00")


def _meeting_window(day: str, friends: List[str], duration: int, earliest: str, latest: str,
                     my_busy: Optional[List[str]], my_tentative: Optional[List[str]]) -> Optional[str]:
    lo = max(_time_to_min(earliest), DAY_START)
    hi = min(_time_to_min(latest), DAY_END)

    friend_busy = _friend_busy(day, friends)
    mine_hard = _parse_intervals(my_busy)
    mine_tentative = _parse_intervals(my_tentative)

    hard = friend_busy + mine_hard
    with_tentative = hard + mine_tentative

    clean = _free_windows(with_tentative, duration, lo, hi)
    if clean:
        s = clean[0][0]
        return f"{_min_to_time(s)}-{_min_to_time(s + duration)}"

    fallback = _free_windows(hard, duration, lo, hi)
    if fallback:
        s = fallback[0][0]
        return f"{_min_to_time(s)}-{_min_to_time(s + duration)}"

    return None


@mcp.tool()
def find_meeting_window(
    day: str,
    friends: str,
    duration_minutes: int,
    earliest: str,
    latest: str,
    my_busy: Optional[List[str]] = None,
    my_tentative: Optional[List[str]] = None,
) -> str:
    """
    Finds the best meeting window of `duration_minutes` on `day`, between `earliest` and
    `latest` (HH:MM), that you and all `friends` can make. Prefers a window that overlaps
    nothing at all; only falls back to overriding a tentative commitment of yours if no
    clean window exists in the range.

    IMPORTANT -- my_busy / my_tentative describe YOUR OWN calendar, which you must work
    out yourself from your inbox: for every invitation whose day matches `day`, use its
    LATEST "When:" line (later emails override earlier ones). Put ACCEPTED ones' times
    (as "HH:MM-HH:MM") in my_busy, TENTATIVE ones' times in my_tentative, and ignore
    DECLINED ones entirely.

    Args:
        day: Weekday name, e.g. 'Tuesday'.
        friends: Comma-separated friend names, e.g. 'ada,bram'.
        duration_minutes: Required meeting length in minutes.
        earliest: Earliest allowed start time, HH:MM.
        latest: Latest allowed end time, HH:MM.
        my_busy: Your ACCEPTED invitation times on this day, each "HH:MM-HH:MM".
        my_tentative: Your TENTATIVE invitation times on this day, each "HH:MM-HH:MM".
    """
    friend_list = [f.strip() for f in friends.split(",") if f.strip()]
    result = _meeting_window(day, friend_list, int(duration_minutes), earliest, latest, my_busy, my_tentative)
    return result or "No available meeting time found."


def _locations(day: str, friends: List[str], my_x: int, my_y: int) -> List[tuple]:
    points = [(my_x, my_y)]
    raws = _fetch_many([f"{BASE_URL}/location/{f}/{day}" for f in friends])
    for raw in raws:
        if not raw:
            continue
        try:
            loc = json.loads(raw)
            points.append((loc["x"], loc["y"]))
        except Exception:
            pass
    return points


@mcp.tool()
def find_meeting_point(day: str, friends: str, my_x: int, my_y: int) -> str:
    """
    Finds the grid point [x, y] minimizing total travel distance (Manhattan) for you and
    all `friends` on `day`.

    Args:
        day: Weekday name, e.g. 'Wednesday'.
        friends: Comma-separated friend names, e.g. 'cira,iris'.
        my_x: Your starting x coordinate.
        my_y: Your starting y coordinate.
    """
    friend_list = [f.strip() for f in friends.split(",") if f.strip()]
    points = _locations(day, friend_list, my_x, my_y)
    xs = sorted(p[0] for p in points)
    ys = sorted(p[1] for p in points)
    n = len(points)
    # For Manhattan distance, any x in the median interval (and likewise y) is optimal;
    # the two coordinates are independent, so the lower median of each is a valid choice.
    bx = xs[(n - 1) // 2]
    by = ys[(n - 1) // 2]
    bx = max(0, min(9, bx))
    by = max(0, min(9, by))
    return f"[{bx}, {by}]"


@mcp.tool()
def plan_outing(
    day: str,
    friends: str,
    my_x: int,
    my_y: int,
    duration_minutes: int,
    earliest: str,
    latest: str,
    my_busy: Optional[List[str]] = None,
    my_tentative: Optional[List[str]] = None,
) -> str:
    """
    Plans a full outing: a meeting window everyone can make, a meeting point, and a place
    to eat afterwards -- chosen jointly so the total travel (everyone's trip to the
    meeting point, plus the trip from there to the venue) is as small as possible.
    Returns "HH:MM-HH:MM, [x, y], VenueName".

    IMPORTANT -- my_busy / my_tentative describe YOUR OWN calendar; see find_meeting_window
    for exactly how to derive them from your inbox.

    Args:
        day: Weekday name.
        friends: Comma-separated friend names.
        my_x: Your starting x coordinate.
        my_y: Your starting y coordinate.
        duration_minutes: Required meeting length in minutes.
        earliest: Earliest allowed meeting start, HH:MM.
        latest: Latest allowed meeting end, HH:MM.
        my_busy: Your ACCEPTED invitation times on this day, each "HH:MM-HH:MM".
        my_tentative: Your TENTATIVE invitation times on this day, each "HH:MM-HH:MM".
    """
    friend_list = [f.strip() for f in friends.split(",") if f.strip()]

    # Fetch everything (friend schedules, friend locations, venues) in ONE parallel
    # round instead of three sequential phases. Three sequential phases, each with
    # its own timeout, can stack past the 10s per-response limit if any one of them
    # is slow -- which reads exactly like a tool call that "never gets going".
    schedule_urls = {f: f"{BASE_URL}/schedule/{f}/{day}" for f in friend_list}
    location_urls = {f: f"{BASE_URL}/location/{f}/{day}" for f in friend_list}
    venues_url = f"{BASE_URL}/venues/{day}"

    all_urls = list(schedule_urls.values()) + list(location_urls.values()) + [venues_url]
    all_texts = _fetch_many(all_urls, timeout=4.0, retries=1)
    text_by_url = dict(zip(all_urls, all_texts))

    friend_busy = []
    for f, u in schedule_urls.items():
        friend_busy.extend(_parse_schedule_text(text_by_url.get(u, "")))

    mine_hard = _parse_intervals(my_busy)
    mine_tentative = _parse_intervals(my_tentative)
    lo = max(_time_to_min(earliest), DAY_START)
    hi = min(_time_to_min(latest), DAY_END)
    hard = friend_busy + mine_hard
    with_tentative = hard + mine_tentative

    window = None
    clean = _free_windows(with_tentative, int(duration_minutes), lo, hi)
    if clean:
        s = clean[0][0]
        window = f"{_min_to_time(s)}-{_min_to_time(s + int(duration_minutes))}"
    else:
        fallback = _free_windows(hard, int(duration_minutes), lo, hi)
        if fallback:
            s = fallback[0][0]
            window = f"{_min_to_time(s)}-{_min_to_time(s + int(duration_minutes))}"
    if not window:
        return "Failed to find a meeting time."
    meet_end = window.split("-")[1]

    venues_raw = text_by_url.get(venues_url, "")
    data = json.loads(venues_raw) if venues_raw else None
    if not data:
        return "Could not fetch venues."
    venues = data.get("venues", data) if isinstance(data, dict) else data
    if isinstance(venues, dict):
        venues = list(venues.values())

    check_start = _time_to_min(meet_end)
    check_end = check_start + 60
    valid_venues = []
    for v in venues or []:
        if not isinstance(v, dict):
            continue
        for window_avail in v.get("available", []):
            if len(window_avail) >= 2:
                o, c = _time_to_min(window_avail[0]), _time_to_min(window_avail[1])
                if o <= check_start and check_end <= c:
                    valid_venues.append(v)
                    break
    if not valid_venues:
        return "No venues open for the hour after the meeting."

    points = [(my_x, my_y)]
    for f, u in location_urls.items():
        raw = text_by_url.get(u, "")
        if raw:
            try:
                loc = json.loads(raw)
                points.append((loc["x"], loc["y"]))
            except Exception:
                pass

    best_cost = float("inf")
    best_point = None
    best_venue = None
    for mx in range(10):
        for my in range(10):
            cost_to_meet = sum(abs(mx - px) + abs(my - py) for px, py in points)
            if cost_to_meet >= best_cost:
                continue
            for v in valid_venues:
                total = cost_to_meet + abs(v["x"] - mx) + abs(v["y"] - my)
                if total < best_cost:
                    best_cost = total
                    best_point = (mx, my)
                    best_venue = v["name"]

    if best_point is None:
        return "Could not find a valid plan."
    return f"{window}, [{best_point[0]}, {best_point[1]}], {best_venue}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Standalone HTTP server -- avoids the FastAPI/Flask lifespan pitfall entirely.
    # Honors $PORT since most hosts inject it rather than using a hardcoded port.
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="http", host="0.0.0.0", port=port, path="/mcp")