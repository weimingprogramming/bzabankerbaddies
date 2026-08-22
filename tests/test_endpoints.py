from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_problem_2_sample_case():
  payload = {
      "payload": (
          "ewoJImFkYXB0SW5wdXQiOiB7CgkJInVzZXIiOiB7CgkJCSJpZCI6ICJVNDIiLAoJCQkiZnVsbE5hbWUiOiAiSmFuZSBEb2UiCgkJfSwKCQkiYWN0aW9uIjogIkNSRUFURSIsCgkJIm1ldGFkYXRhIjogewoJCQkicHJpb3JpdHkiOiAiSElHSCIKCQl9Cgl9Cn0="
      )
  }

  response = client.post("/solve", json=payload)
  assert response.status_code == 200

  data = response.json()
  assert "adaptOutput" in data
  assert data["adaptOutput"] == {
      "id": "U42",
      "name": "Jane Doe",
      "action": "create",
      "priority": 3,
  }