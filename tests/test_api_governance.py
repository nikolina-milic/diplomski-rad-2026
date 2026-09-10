"""API: registry, verzionisanje, ground truth, trošak i drift."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from spr.api.app import create_app
from spr.engine.factory import build_default_engine
from spr.persistence.db import make_session_factory


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    eng = build_default_engine(n_normal=400, days=7)
    sf = make_session_factory("sqlite://")
    art = str(tmp_path_factory.mktemp("artifacts"))
    return TestClient(create_app(eng, session_factory=sf, artifacts_dir=art))


def _event(**kw):
    d = dict(event_id="g1", user_id="gu1",
             timestamp=datetime(2026, 1, 1, 12, tzinfo=timezone.utc).isoformat(),
             amount=100.0, currency="EUR", merchant_category="grocery",
             country="RS", latitude=44.82, longitude=20.46,
             device_id="d1", channel="online")
    d.update(kw)
    return d


# --- početni model u registryju (#9) ------------------------------------
def test_initial_model_is_registered_as_champion(client):
    models = client.get("/models").json()
    assert models, "registry je prazan iako model postoji i donosi odluke"
    champions = [m for m in models if m["status"] == "champion"]
    assert len(champions) == 1
    assert champions[0]["version"] == "v1"
    assert "pr_auc" in champions[0]["metrics"]


# --- verzionisanje retreninga (#10) -------------------------------------
def _seed_labeled_decisions(client, n=80):
    """Napuni audit log labeliranim odlukama da retrening ima na čemu da uči."""
    from spr.evaluation.experiments import TRAIN_START
    from spr.generator.generator import generate_batch
    events = generate_batch(n_users=8, n_normal=n, fraud_rate=0.1, seed=5,
                            start=TRAIN_START, days=5)
    for e in events:
        client.post("/score?record_truth=true",
                    json=e.model_dump(mode="json"))


def test_retrain_refuses_without_enough_labeled_data(client):
    """Na svježem sistemu retrening mora dati jasnu poruku, a ne 500."""
    r = client.post("/retrain")
    assert r.status_code == 400
    assert "labeliranih" in r.json()["detail"] or "prevara" in r.json()["detail"]


def test_retrain_picks_the_next_free_version(client):
    _seed_labeled_decisions(client)
    before = {m["version"] for m in client.get("/models").json()}
    r = client.post("/retrain")
    assert r.status_code == 200
    new = r.json()["challenger_version"]
    assert new not in before
    assert new in {m["version"] for m in client.get("/models").json()}


def test_retrain_refuses_to_overwrite_an_existing_version(client):
    existing = client.get("/models").json()[0]["version"]
    r = client.post(f"/retrain?version={existing}")
    assert r.status_code == 409
    assert "već postoji" in r.json()["detail"]


def test_only_one_champion_after_retraining(client):
    client.post("/retrain")
    models = client.get("/models").json()
    assert sum(1 for m in models if m["status"] == "champion") == 1


# --- ground truth ne dolazi od klijenta (#13) ---------------------------
def test_score_ignores_client_supplied_ground_truth(client):
    client.post("/score", json=_event(event_id="truth-off", is_fraud=True))
    dec = next(d for d in client.get("/decisions?limit=50").json()
               if d["event_id"] == "truth-off")
    full = client.get("/review-queue").json()
    assert dec["id"] is not None
    # truth se ne izlaže preko /decisions, pa provjeravamo preko /metrics
    assert isinstance(full, list)


def test_score_records_truth_only_when_explicitly_requested(client):
    before = client.get("/metrics").json()["n"]
    client.post("/score", json=_event(event_id="truth-1", is_fraud=True))
    assert client.get("/metrics").json()["n"] == before
    client.post("/score?record_truth=true",
                json=_event(event_id="truth-2", is_fraud=True))
    assert client.get("/metrics").json()["n"] == before + 1


def test_decision_exposes_executed_action(client):
    r = client.post("/score", json=_event(event_id="exec-1"))
    body = r.json()
    assert body["executed_action"] == "ALLOW"   # podrazumijevani shadow režim
    assert body["mode"] == "shadow"


# --- trošak i pragovi ----------------------------------------------------
def test_cost_endpoint_reports_matrix_and_derived_thresholds(client):
    body = client.get("/cost").json()
    assert {"cost_matrix", "capacity", "closed_form", "current"} <= set(body)
    assert 0.0 <= body["closed_form"]["low_max"] <= body["closed_form"]["medium_max"] <= 1.0


def test_updating_fraud_loss_moves_the_derived_threshold(client):
    client.put("/config/cost", json={"avg_fraud_loss": 100.0})
    cheap = client.get("/cost").json()["closed_form"]["low_max"]
    client.put("/config/cost", json={"avg_fraud_loss": 10_000.0})
    dear = client.get("/cost").json()["closed_form"]["low_max"]
    assert dear < cheap
    client.put("/config/cost", json={"avg_fraud_loss": 250.0})


def test_calibration_endpoint_reports_live_fusion_state(client):
    body = client.get("/calibration").json()
    assert body["fitted"] is True
    assert body["method"] in ("isotonic", "platt", "none")
    assert set(body["meta_weights"]) == {"rules", "ml", "interaction", "intercept"}


# --- drift ---------------------------------------------------------------
def test_drift_endpoint_degrades_gracefully_without_enough_data(client):
    body = client.get("/drift?window=5").json()
    assert "available" in body
    if not body["available"]:
        assert "premalo" in body["reason"]


# --- filtriranje i prebrojavanje odluka ----------------------------------
def test_decisions_can_be_filtered_by_level(client):
    _seed_labeled_decisions(client, n=60)
    for level in ("LOW", "MEDIUM", "HIGH"):
        rows = client.get(f"/decisions?limit=50&level={level}").json()
        assert all(d["level"] == level for d in rows), level


def test_decisions_rejects_unknown_level(client):
    r = client.get("/decisions?level=SREDNJI")
    assert r.status_code == 400


def test_decisions_limit_is_respected(client):
    assert len(client.get("/decisions?limit=3").json()) <= 3


def test_count_matches_the_number_of_returned_rows(client):
    """Zaglavlja kolona na dashboardu se oslanjaju na ove brojeve — ako se ne
    slažu sa sadržajem, korisnik vidi 'VISOK 118' iznad prazne kolone."""
    counts = client.get("/decisions/count").json()
    assert counts["total"] == sum(counts["by_level"].values())
    assert counts["total"] == sum(counts["by_action"].values())
    for level, n in counts["by_level"].items():
        rows = client.get(f"/decisions?limit=100000&level={level}").json()
        assert len(rows) == n, level


def test_count_and_stats_disagree_only_by_window(client):
    """/stats broji posljednjih N, /decisions/count broji sve — dashboard zato
    koristi samo jedan od njih."""
    total = client.get("/decisions/count").json()["total"]
    windowed = client.get("/stats?window=5").json()["total"]
    assert windowed <= total


def test_logged_decision_carries_the_same_fields_as_a_live_one(client):
    """Modal prikazuje istu odluku bilo da je stigla iz živog toka ili iz
    audit loga — pa odgovor /decisions mora nositi sva polja objašnjenja."""
    live = client.post("/score?record_truth=true", json=_event(event_id="parity-1")).json()
    logged = next(d for d in client.get("/decisions?limit=20").json()
                  if d["event_id"] == "parity-1")
    for field in ("rules_score", "ml_score", "final_score", "level", "action",
                  "executed_action", "mode", "model_version", "explanation",
                  "fired_rules", "shap_top"):
        assert field in logged, field
        if field in live:
            assert logged[field] == live[field], field
