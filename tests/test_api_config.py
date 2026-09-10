import pytest
from fastapi.testclient import TestClient
from spr.api.app import create_app
from spr.engine.factory import build_default_engine
from spr.persistence.db import make_session_factory


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(build_default_engine(n_normal=300, days=7)))


def test_update_policy_mode_and_thresholds(client):
    r = client.put("/config/policy", json={"mode": "enforce", "low_max": 0.2, "medium_max": 0.6})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "enforce"
    assert body["low_max"] == 0.2
    assert client.get("/policy").json()["mode"] == "enforce"


def test_invalid_mode_rejected(client):
    r = client.put("/config/policy", json={"mode": "nonsense"})
    assert r.status_code == 400


def test_update_fusion(client):
    r = client.put("/config/fusion", json={"strategy": "cascade", "rules_weight": 0.7, "ml_weight": 0.3})
    assert r.status_code == 200
    assert r.json()["strategy"] == "cascade"
    assert client.get("/fusion").json()["rules_weight"] == 0.7


def test_disable_and_reweight_rule(client):
    r = client.put("/config/rules", json=[
        {"name": "new_device", "enabled": False},
        {"name": "impossible_travel", "weight": 0.5},
    ])
    assert r.status_code == 200
    rules = {x["name"]: x for x in r.json()}
    assert rules["new_device"]["enabled"] is False
    assert rules["impossible_travel"]["weight"] == 0.5


def test_rules_expose_thresholds_and_parts(client):
    rules = {x["name"]: x for x in client.get("/rules").json()}
    rf = rules["rapid_fire"]
    assert rf["thresholds"] == [5]
    assert len(rf["condition_parts"]) == len(rf["thresholds"]) + 1
    uh = rules["unusual_hour_spending"]
    assert uh["thresholds"] == [1, 3]


def test_update_threshold_changes_condition(client):
    r = client.put("/config/rules", json=[
        {"name": "rapid_fire", "thresholds": [3]},
    ])
    assert r.status_code == 200
    rf = {x["name"]: x for x in r.json()}["rapid_fire"]
    assert rf["condition"] == "txn_count_1h >= 3"
    assert rf["thresholds"] == [3]


def test_wrong_threshold_count_rejected(client):
    r = client.put("/config/rules", json=[
        {"name": "rapid_fire", "thresholds": [3, 4]},
    ])
    assert r.status_code == 400


def test_threshold_change_persists_across_restart():
    # dijeljeni in-memory factory (StaticPool) živi kroz obje instance app-a
    sf = make_session_factory("sqlite://")
    eng1 = build_default_engine(n_normal=300, days=7)
    app1 = TestClient(create_app(eng1, session_factory=sf))
    r = app1.put("/config/rules", json=[{"name": "rapid_fire", "thresholds": [9]}])
    assert r.status_code == 200

    # "restart": nov engine (default pravila) + nov app na istoj bazi
    eng2 = build_default_engine(n_normal=300, days=7)
    app2 = TestClient(create_app(eng2, session_factory=sf))
    rf = {x["name"]: x for x in app2.get("/rules").json()}["rapid_fire"]
    assert rf["condition"] == "txn_count_1h >= 9"
    assert rf["thresholds"] == [9]
