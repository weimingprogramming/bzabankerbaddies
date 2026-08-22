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
def test_showdown_pair_action():
  # Holding pair (3 == 3) -> should raise/bet or call
  payload = {
      "protocol_version": 2,
      "round": "post_reveal",
      "your_number": 3,
      "community_number": 3,
      "pot": 20,
      "to_call": 4,
      "min_raise_to": 10,
      "max_raise_to": 100,
      "legal_actions": ["fold", "call", "raise"],
      "your_stack": 180,
  }
  res = client.post("/move", json=payload)
  assert res.status_code == 200
  data = res.json()
  assert data["action"] in ["raise", "call"]
  if data["action"] == "raise":
    assert 10 <= data["amount"] <= 100


def test_showdown_trash_hand_folds():
  # Holding 2 on board 11 facing a large bet -> should fold
  payload = {
      "protocol_version": 2,
      "round": "post_reveal",
      "your_number": 2,
      "community_number": 11,
      "pot": 40,
      "to_call": 20,
      "min_raise_to": 40,
      "max_raise_to": 150,
      "legal_actions": ["fold", "call", "raise"],
      "your_stack": 150,
  }
  res = client.post("/move", json=payload)
  assert res.status_code == 200
  assert res.json() == {"action": "fold"}

def test_showdown_pair_action():
  # Holding pair (3 == 3) -> Should max raise now
  payload = {
      "protocol_version": 2,
      "round": "post_reveal",
      "your_number": 3,
      "community_number": 3,
      "pot": 20,
      "to_call": 4,
      "min_raise_to": 10,
      "max_raise_to": 100,
      "legal_actions": ["fold", "call", "raise"],
      "your_stack": 180,
  }
  res = client.post("/move", json=payload)
  assert res.status_code == 200
  data = res.json()
  assert data["action"] == "raise"
  assert data["amount"] == 100  # Verify it max-raises the nuts