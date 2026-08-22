import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from problems import p1, p2, p3, p4, p5, p6

mcp_app = p4.mcp.http_app(path="/")

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.lifespan(mcp_app):
        yield

app = FastAPI(title="Competition Service Engine", lifespan=lifespan)

app.mount("/mcp", mcp_app)

# ==========================================
# 1. Mount root endpoints (REQUIRED BY GRADER)
# ==========================================
app.include_router(p1.router, prefix="", tags=["Problem 1 - Root"])
app.include_router(p2.router, prefix="", tags=["Problem 2 - Gateway"])
app.include_router(p3.router, prefix="", tags=["Problem 3 - Showdown"])
app.include_router(p5.router, prefix="", tags=["Problem 5 - Ghost Chains"])

# ADD P6 HERE FOR THE GRADER -> Exposes /stonks
app.include_router(p6.router, prefix="", tags=["Problem 6 - Root"]) 


# ==========================================
# 2. Standard prefixes (FOR CONSISTENCY)
# ==========================================
app.include_router(p1.router, prefix="/p1", tags=["Problem 1"])
app.include_router(p2.router, prefix="/p2", tags=["Problem 2"])
app.include_router(p3.router, prefix="/p3", tags=["Problem 3"])
app.include_router(p5.router, prefix="/p5", tags=["Problem 5"])

# ADD P6 HERE FOR CONSISTENCY -> Exposes /p6/stonks
app.include_router(p6.router, prefix="/p6", tags=["Problem 6 - Time Travelling Stonks"])


@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)