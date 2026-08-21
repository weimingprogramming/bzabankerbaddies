import os
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Competition Service")


@app.get("/")
def root():
  return {"message": "Service is live"}


@app.get("/health")
def health_check():
  return {"status": "ok"}


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8000))
  uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)