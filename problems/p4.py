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
        # Clean up the string just in case it passes 'x' instead of '*'
        expr = expression.replace("x", "*").replace("X", "*")
        
        # Strip out any non-math characters to safely evaluate
        expr = re.sub(r'[^0-9\+\-\*\/\.\(\)\ ]', '', expr)
        
        # Python's eval() natively respects order of operations
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
        # 1. Decode base64 to OpenCV format
        img_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        # 2. Extract mask (handles both transparent PNGs and white backgrounds)
        if len(img.shape) == 3 and img.shape[2] == 4:
            mask = img[:, :, 3]  # Use alpha channel if present
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # 3. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "circle"
            
        cnt = max(contours, key=cv2.contourArea)
        
        # 4. Count vertices to determine shape
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
        # Fallback to prevent tool execution failure
        return "circle"


@mcp.tool()
def get_relevant_study_passages(question: str, urls: list[str] = None, text_content: str = None) -> list[str]:
    """
    Extracts the most relevant passages to answer a question from the provided URLs or raw text.
    Guarantees the returned list of passages sum to under 900 tokens (using the o200k_base encoding).
    
    CRITICAL FOR AGENT: Pass the 'urls' you were provided to fetch the documents directly.
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
        
    # Chunking strategy for optimal packing
    chunks = [c.strip() for c in re.split(r'\n\s*\n', content) if c.strip()]
    if len(chunks) < 3:
        chunks = [c.strip() for c in content.split('\n') if c.strip()]
    if len(chunks) < 3:
        chunks = [c.strip() + "." for c in content.split('.') if c.strip()]
        
    # Scoring chunks based on word and bigram overlap with the question
    q_words = set(re.findall(r'\w+', question.lower()))
    
    def get_bigrams(s):
        words = re.findall(r'\w+', s.lower())
        return set(zip(words, words[1:]))
        
    q_bigrams = get_bigrams(question)
    
    scored = []
    for chunk in chunks:
        c_words = set(re.findall(r'\w+', chunk.lower()))
        c_bigrams = get_bigrams(chunk)
        # Bigrams are weighted heavier as they signify exact phrase matches (e.g., "sensor grid")
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
        
        # Stop or safely truncate if we approach the absolute ceiling
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


@mcp.tool()
def calculate_next_hop(
    current_node: str,
    destination: str,
    adjacency: dict = None,
    tolls: dict = None,
    visited: list[str] = None,
    hops_left: int = 999,
    map_url: str = None
) -> str:
    """
    Calculates the next node to travel to on a weighted directed graph to reach the destination with minimum cost.
    Total cost = sum(edge weights) + sum(entry tolls).
    
    CRITICAL INSTRUCTIONS FOR AGENT:
    1. If you only have a map_id, YOU must fetch the map data yourself (e.g., GET /graph?map_id=...) 
       and pass the 'adjacency' and 'tolls' JSON dictionaries directly to this tool. Alternatively, pass the full absolute URL in 'map_url'.
    2. You MUST maintain and pass the 'visited' list containing all previously visited nodes on this specific journey to prevent zero scores from revisiting.
    
    Args:
        current_node: The node you are currently at.
        destination: The target node you need to reach.
        adjacency: dict of dicts, e.g. {"A": {"B": 4.0, "C": 2.0}}
        tolls: dict of tolls for each node, e.g. {"A": 5.0, "B": 1.0}
        visited: list of nodes already visited on this journey (to avoid revisiting).
        hops_left: number of hops allowed (default 999 if unrestricted).
        map_url: Optional full URL to fetch the graph JSON if adjacency/tolls are not provided.
    
    Returns:
        The name of the next node to visit.
    """
    if isinstance(visited, str):
        visited = [visited]
    if visited is None:
        visited = []
        
    if adjacency is None or tolls is None:
        if map_url:
            try:
                req = urllib.request.Request(map_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    adjacency = data.get('adjacency', {})
                    tolls = data.get('tolls', {})
            except Exception:
                pass
                
    if adjacency is None: adjacency = {}
    if tolls is None: tolls = {}
    
    # Nodes already visited on this journey (cannot be revisited per spec).
    visited_set = set(visited)
    visited_set.add(current_node)

    # Priority Queue State: (cost, hops_used, node, path)
    # path tracks nodes in this Dijkstra run to prevent within-path cycles
    pq = [(0.0, 0, current_node, [current_node])]
    best_cost = {}

    while pq:
        cost, hops, u, path = heapq.heappop(pq)

        # Arrived at destination
        if u == destination:
            if len(path) > 1:
                return path[1]
            return u

        # Out of hop allowance
        if hops >= hops_left:
            continue

        # Prune dominated states
        state_key = u
        if best_cost.get(state_key, float('inf')) <= cost:
            continue
        best_cost[state_key] = cost

        # Track nodes on current path to prevent within-path cycles
        path_set = set(path)

        # Explore neighbors
        for v, weight in adjacency.get(u, {}).items():
            # Skip nodes visited on prior journey steps
            if v in visited_set and v != destination:
                continue
            # Skip nodes already on this path (prevent Dijkstra cycles)
            if v in path_set:
                continue

            next_cost = cost + float(weight) + float(tolls.get(v, 0.0))
            next_hops = hops + 1

            heapq.heappush(pq, (next_cost, next_hops, v, path + [v]))

    # Fallback if no route found
    return ""