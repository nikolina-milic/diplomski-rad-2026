import pytest
from fastapi.testclient import TestClient
from spr.api.app import create_app
from spr.engine.factory import build_default_engine


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(build_default_engine(n_normal=300, days=7)))


def test_evaluation_endpoint(client):
    r = client.get("/evaluation?n_test=200")
    assert r.status_code == 200
    body = r.json()
    from spr.evaluation.report import APPROACHES
    assert set(body["approaches"]) == set(APPROACHES)
    assert body["latency"]["mean_ms"] > 0
