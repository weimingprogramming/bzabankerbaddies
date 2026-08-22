import os
import uvicorn
from fastapi import FastAPI
from problems import p1, p2, p3, p4

app = FastAPI(title="Competition Service Engine")

# Mount root endpoints directly
app.include_router(p2.router, prefix="", tags=["Problem 2 - Gateway"])
app.include_router(p3.router, prefix="", tags=["Problem 3 - Showdown"])

# Mount problem sub-routes
app.include_router(p1.router, prefix="/p1", tags=["Problem 1"])
app.include_router(p2.router, prefix="/p2", tags=["Problem 2"])
app.include_router(p3.router, prefix="/p3", tags=["Problem 3"])
app.include_router(p4.router, prefix="/p4", tags=["Problem 4"])


@app.get("/health")
def health_check():
  return {"status": "ok"}


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8000))
  uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)