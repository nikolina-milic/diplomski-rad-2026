from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from spr.api.app import create_app
from spr.engine.factory import build_default_engine
from spr.persistence.db import make_session_factory


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    eng = build_default_engine(n_normal=300, days=7)
    sf = make_session_factory("sqlite://")
    art = str(tmp_path_factory.mktemp("artifacts"))
    return TestClient(create_app(eng, session_factory=sf, artifacts_dir=art))


def _event(**kw):
    d = dict(event_id="e1", user_id="u1",
             timestamp=datetime(2026, 1, 1, 12, tzinfo=timezone.utc).isoformat(),
             amount=100.0, currency="EUR", merchant_category="grocery",
             country="RS", latitude=44.82, longitude=20.46,
             device_id="d1", channel="online")
    d.update(kw)
    return d


def test_score_is_logged(client):
    client.post("/score", json=_event(event_id="a1"))
    r = client.get("/decisions")
    assert r.status_code == 200
    assert any(d["event_id"] == "a1" for d in r.json())


def test_feedback_flow(client):
    client.post("/score", json=_event(event_id="a2"))
    dec_id = client.get("/decisions").json()[0]["id"]
    r = client.post("/feedback", json={"decision_id": dec_id, "label": 1})
    assert r.status_code == 200
    assert r.json()["label"] == 1


def test_health_reports_persistence(client):
    assert client.get("/health").json()["persistence"] is True


def test_models_endpoint_exists(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
