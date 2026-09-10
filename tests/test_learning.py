from datetime import datetime, timezone
from spr.persistence.db import make_session_factory
from spr.persistence import repository as repo
from spr.learning.metrics import compute_metrics
from spr.learning.service import run_retraining_cycle
from spr.engine.factory import build_default_engine
from spr.generator.generator import generate_batch


def test_compute_metrics_perfect():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
    assert m["pr_auc"] == 1.0
    assert m["f1"] == 1.0


def test_retraining_cycle_runs_and_registers(tmp_path):
    Session = make_session_factory("sqlite://")
    eng = build_default_engine(n_normal=400, days=7)
    events = generate_batch(n_users=20, n_normal=600, fraud_rate=0.08, seed=5,
                            start=datetime(2026, 2, 1, tzinfo=timezone.utc), days=10)
    with Session() as s:
        for e in events:
            dec = eng.score(e)
            repo.save_decision(s, dec, e.timestamp, truth=1 if e.is_fraud else 0)
        result = run_retraining_cycle(s, eng, str(tmp_path), new_version="v2",
                                      test_seed=999, n_normal=400, days=7)
        assert result["challenger_version"] == "v2"
        assert "pr_auc" in result["challenger_metrics"]
        versions = [m.version for m in repo.list_model_versions(s)]
        assert "v2" in versions
        assert eng.model.version in {"v1", "v2"}
        if result["promoted"]:
            assert eng.model.version == "v2"
            assert repo.get_champion(s).version == "v2"
