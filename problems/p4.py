import base64
import numpy as np
import cv2
from fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("NurseryServer")

@mcp.tool()
def get_agent_name() -> str:
    """Returns the name of the agent."""
    return "Render-Baby"

@mcp.tool()
def calculate_math(a: int, b: int, operator: str) -> float:
    """
    Performs basic arithmetic on numbers.
    Args:
        a: integer between -100 and 100
        b: integer between -100 and 100
        operator: One of '+', '-', '*', '/'
    """
    if operator == "+": return float(a + b)
    if operator == "-": return float(a - b)
    if operator == "*": return float(a * b)
    if operator == "/": return float(a / b) if b != 0 else 0.0
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