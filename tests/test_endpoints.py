from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
  response = client.get("/health")
  assert response.status_code == 200
  assert response.json() == {"status": "ok"}


def test_p1_placeholder():
  response = client.post("/p1/solve", json={"payload": {"test": 123}})
  assert response.status_code == 200
  assert response.json()["result"] == "ok"