import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from problems import p1, p2, p3, p4, p5

# 1. Extract the ASGI app from the FastMCP instance
mcp_app = p4.mcp.http_app(path="/")

# 2. CRITICAL: Pass the FastMCP lifespan to FastAPI to prevent 500 errors
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.lifespan(app):
        yield

app = FastAPI(title="Competition Service Engine", lifespan=lifespan)

# 3. CRITICAL FIX: You must mount the MCP app so the /mcp endpoint exists!
app.mount("/mcp", mcp_app)

# Mount root endpoints directly
app.include_router(p1.router, prefix="", tags=["Problem 1 - Delivery Driver"])
app.include_router(p2.router, prefix="", tags=["Problem 2 - Gateway"])
app.include_router(p3.router, prefix="", tags=["Problem 3 - Showdown"])

app.include_router(p1.router, prefix="/p1", tags=["Problem 1"])
app.include_router(p2.router, prefix="/p2", tags=["Problem 2"])
app.include_router(p3.router, prefix="/p3", tags=["Problem 3"])

app.include_router(p5.router, prefix="", tags=["Problem 5 - Ghost Chains"])

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)