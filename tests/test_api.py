from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from spr.api.app import create_app
from spr.engine.factory import build_default_engine


@pytest.fixture(scope="module")
def client():
    eng = build_default_engine(n_normal=300, days=7)
    return TestClient(create_app(eng))


def _event(**kw):
    d = dict(event_id="e1", user_id="tester",
             timestamp=datetime(2026, 1, 1, 12, tzinfo=timezone.utc).isoformat(),
             amount=100.0, currency="EUR", merchant_category="grocery",
             country="RS", latitude=44.82, longitude=20.46,
             device_id="d1", channel="online")
    d.update(kw)
    return d


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_returns_decision(client):
    r = client.post("/score", json=_event())
    assert r.status_code == 200
    body = r.json()
    assert body["level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["action"] in {"ALLOW", "CHALLENGE", "REVIEW", "BLOCK"}
    assert "explanation" in body


def test_score_guardrail_block(client):
    r = client.post("/score", json=_event(amount=20000.0))
    assert r.json()["action"] == "BLOCK"


def test_rules_endpoint(client):
    r = client.get("/rules")
    assert r.status_code == 200
    names = [x["name"] for x in r.json()]
    assert "impossible_travel" in names
