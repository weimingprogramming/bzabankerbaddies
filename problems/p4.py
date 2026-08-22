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

@mcp.tool()
def calculate_next_hop(
    current_node: str,
    destination: str,
    map_data_json: str,
    visited_nodes: List[str] = None,
    hops_left: int = 999
) -> str:
    """
    Calculates the optimal next node to travel to on the map.
    
    AGENT INSTRUCTIONS (CRITICAL):
    1. You MUST fetch the map data yourself using an HTTP GET request to `/graph?map_id=<map_id>` BEFORE calling this tool.
    2. Pass the ENTIRE raw JSON response string you receive directly into `map_data_json`. DO NOT attempt to parse it yourself.
    3. You MUST maintain and pass the list of nodes you have already visited on this specific journey in `visited_nodes` (e.g., ["A", "B"]).
    4. If the prompt gave you a maximum number of hops/moves, pass the remaining amount in `hops_left`.
    5. Output EXACTLY the single node string returned by this tool as your next move. Do not invent steps.
    """
    if visited_nodes is None:
        visited_nodes = []
        
    try:
        # Safely handle in case the LLM parses it into a dictionary anyway
        if isinstance(map_data_json, dict):
            data = map_data_json
        else:
            data = json.loads(map_data_json)
    except Exception as e:
        return f"ERROR: Invalid map_data_json. You must fetch the map data and pass the raw JSON string. Details: {e}"
        
    adjacency = data.get("adjacency", {})
    tolls = data.get("tolls", {})
    
    if not adjacency:
        return "ERROR: The map data is empty. Make sure you fetch from /graph?map_id=<map_id>."
        
    # Prevent revisiting nodes already traversed in previous turns
    visited_set = set(visited_nodes)
    visited_set.add(current_node)
    
    # pq stores: (cost, hops_used, current_node, path)
    pq = [(0.0, 0, current_node, [current_node])]
    
    # Pareto frontier for (cost, hops) at each node to prevent incorrect pruning
    best_states = {}
    
    while pq:
        cost, hops, u, path = heapq.heappop(pq)
        
        if u == destination:
            if len(path) > 1:
                return path[1]
            return u
            
        if hops >= hops_left:
            continue
            
        # Check if this state is dominated by a previously found better path to `u`
        dominated = False
        for prev_cost, prev_hops in best_states.get(u, []):
            if prev_cost <= cost and prev_hops <= hops:
                dominated = True
                break
        if dominated:
            continue
            
        if u not in best_states:
            best_states[u] = []
        best_states[u].append((cost, hops))
        
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