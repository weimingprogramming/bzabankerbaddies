import base64
import numpy as np
import cv2
from fastmcp import FastMCP
import re
import urllib.request
import json
import heapq
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

# Initialize the MCP Server
mcp = FastMCP("NurseryServer")

@mcp.tool()
def get_agent_name() -> str:
    """Returns the name of the agent."""
    return "Render-Baby"

@mcp.tool()
def calculate_math(expression: str) -> float:
    """
    Evaluates a full mathematical expression string, automatically handling order of operations (PEMDAS).
    Args:
        expression: The complete math problem (e.g., "2 + 3 * 5"). Do NOT break it into steps.
    """
    try:
        expr = expression.replace("x", "*").replace("X", "*")
        expr = re.sub(r'[^0-9\+\-\*\/\.\(\)\ ]', '', expr)
        return float(eval(expr))
    except Exception:
        return 0.0

@mcp.tool()
def identify_shape(image_b64: str) -> str:
    """
    Identifies the shape from a base64 encoded PNG image string.
    Returns exactly one of these strings: 'rectangle', 'triangle', or 'circle'.
    """
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
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            return "rectangle"
        else:
            return "circle"
    except Exception:
        return "circle"

@mcp.tool()
def get_relevant_study_passages(question: str, urls: List[str] = None, text_content: str = None) -> List[str]:
    """
    Extracts the most relevant passages to answer a question or find a location from provided URLs or text.
    
    AGENT INSTRUCTIONS:
    If you need to find out a specific fact, name, or destination for a trip, use this tool.
    Pass the list of URLs provided in the prompt to the `urls` parameter.
    """
    content = text_content or ""

    if isinstance(urls, str):
        urls = [urls]

    def _fetch_url(url: str) -> str:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as response:
                return response.read().decode('utf-8')
        except Exception:
            return ""

    if urls:
        with ThreadPoolExecutor(max_workers=min(len(urls), 8)) as pool:
            futures = {pool.submit(_fetch_url, u): u for u in urls}
            for f in as_completed(futures):
                text = f.result()
                if text:
                    content += "\n" + text
                
    if not content.strip():
        return ["No study material content could be retrieved."]
        
    chunks = [c.strip() for c in re.split(r'\n\s*\n', content) if c.strip()]
    if len(chunks) < 3:
        chunks = [c.strip() for c in content.split('\n') if c.strip()]
    if len(chunks) < 3:
        chunks = [c.strip() + "." for c in content.split('.') if c.strip()]
        
    q_words = set(re.findall(r'\w+', question.lower()))
    
    def get_bigrams(s):
        words = re.findall(r'\w+', s.lower())
        return set(zip(words, words[1:]))
        
    q_bigrams = get_bigrams(question)
    
    scored = []
    for chunk in chunks:
        c_words = set(re.findall(r'\w+', chunk.lower()))
        c_bigrams = get_bigrams(chunk)
        score = len(q_words & c_words) + 2 * len(q_bigrams & c_bigrams)
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
        
    return results

# Graph cache so the tool doesn't re-fetch on every hop of the same journey
_graph_cache: Dict[str, dict] = {}

GRAPH_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"


def _fetch_graph(map_id: str) -> dict:
    """Fetch and cache graph data for a map_id."""
    if map_id in _graph_cache:
        return _graph_cache[map_id]
    url = f"{GRAPH_BASE_URL}/graph?map_id={map_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode('utf-8'))
            _graph_cache[map_id] = data
            return data
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def calculate_next_hop(
    current_node: str,
    destination: str,
    map_id: str,
    visited_nodes: List[str] = None,
    hops_left: int = 999
) -> str:
    """
    Given your current position and destination on a map, returns the next node you should move to.
    This tool fetches the map data automatically — just pass the map_id from your question.
    Call this tool once per step. Pass all nodes you have already visited in visited_nodes.
    If you have a hop/move limit, pass the remaining number in hops_left.
    Return EXACTLY the node string this tool gives you as your answer.

    Args:
        current_node: The node you are currently standing at.
        destination: The node you need to reach.
        map_id: The map_id from the question (e.g. "8f3c1e0a-...").
        visited_nodes: All nodes you have already visited on this journey so far.
        hops_left: How many moves/hops you have remaining (default 999 if unlimited).
    """
    if visited_nodes is None:
        visited_nodes = []

    data = _fetch_graph(map_id)
    if "error" in data:
        return f"ERROR: Could not fetch map data: {data['error']}"

    adjacency = data.get("adjacency", {})
    tolls = data.get("tolls", {})

    if not adjacency:
        return "ERROR: Map has no adjacency data."

    # Prevent revisiting nodes already traversed in previous turns
    visited_set = set(visited_nodes)
    visited_set.add(current_node)

    # pq stores: (cost, hops_used, node, path)
    pq = [(0.0, 0, current_node, [current_node])]

    # Pareto frontier for (cost, hops) at each node
    best_states: Dict[str, List] = {}

    while pq:
        cost, hops, u, path = heapq.heappop(pq)

        if u == destination:
            if len(path) > 1:
                return path[1]
            return u

        if hops >= hops_left:
            continue

        # Check if this state is dominated
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
            next_hops = hops + 1

            heapq.heappush(pq, (next_cost, next_hops, v, path + [v]))

    return "ERROR: No valid path found."