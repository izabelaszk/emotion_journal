import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "entries" in body


def test_classify_returns_scores():
    r = client.post("/classify", json={"text": "I am so happy today!"})
    assert r.status_code == 200
    scores = r.json()
    assert isinstance(scores, list)
    assert any(s["label"] == "joy" for s in scores)


def test_classify_scores_sum_to_one():
    r = client.post("/classify", json={"text": "Testing."})
    total = sum(s["score"] for s in r.json())
    assert abs(total - 1.0) < 0.02


def test_analyse_text():
    r = client.post("/analyse", data={"text": "I love this project!"})
    assert r.status_code == 200
    body = r.json()
    assert body["transcription"] == "I love this project!"
    assert "dominant_emotion" in body
    assert "emotions" in body


def test_analyse_no_input_returns_400():
    r = client.post("/analyse")
    assert r.status_code == 400


def test_journal_persists_after_analyse():
    client.delete("/journal")
    client.post("/analyse", data={"text": "Feeling great today."})
    r = client.get("/journal")
    assert r.status_code == 200
    assert r.json()["total"] >= 1


def test_delete_journal():
    client.post("/analyse", data={"text": "Test entry."})
    r = client.delete("/journal")
    assert r.status_code == 200
    assert r.json()["deleted"] >= 1
    assert client.get("/journal").json()["total"] == 0
