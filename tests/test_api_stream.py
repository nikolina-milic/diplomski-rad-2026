import pytest
from fastapi.testclient import TestClient
from spr.api.app import create_app
from spr.engine.factory import build_default_engine
from spr.persistence.db import make_session_factory


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    eng = build_default_engine(n_normal=300, days=7)
    sf = make_session_factory("sqlite://")
    return TestClient(create_app(eng, session_factory=sf,
                                 artifacts_dir=str(tmp_path_factory.mktemp("a"))))


def test_ws_stream_emits_decisions(client):
    with client.websocket_connect("/ws/stream?interval=0") as ws:
        msg = ws.receive_json()
        assert msg["level"] in {"LOW", "MEDIUM", "HIGH"}
        assert "explanation" in msg


def test_stats_and_metrics(client):
    with client.websocket_connect("/ws/stream?interval=0") as ws:
        for _ in range(3):
            ws.receive_json()
    assert client.get("/stats").json()["total"] >= 1
    m = client.get("/metrics").json()
    assert set(m["confusion"].keys()) == {"tp", "fp", "tn", "fn"}
