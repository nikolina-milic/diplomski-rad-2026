from datetime import datetime, timezone
import numpy as np
from spr.generator.generator import generate_batch
from spr.ml.dataset import build_dataset
from spr.ml.model import RiskModel
from spr.context.features import FEATURE_NAMES

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _trained():
    events = generate_batch(n_users=20, n_normal=1500, fraud_rate=0.08, seed=1, start=START, days=21)
    X, y = build_dataset(events)
    m = RiskModel(version="v1", random_state=0)
    m.train(X, y)
    return m, X, y


def test_predict_proba_in_range():
    m, X, _ = _trained()
    p = m.predict_proba(X[0])
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_model_separates_classes():
    m, X, y = _trained()
    probs = np.array([m.predict_proba(x) for x in X])
    y = np.array(y)
    # prosječna vjerovatnoća na prevarama mora biti veća nego na legitimnim
    assert probs[y == 1].mean() > probs[y == 0].mean()


def test_explain_returns_feature_contributions():
    m, X, _ = _trained()
    contrib = m.explain(X[0])
    assert set(contrib.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in contrib.values())


def test_save_load_roundtrip(tmp_path):
    m, X, _ = _trained()
    path = str(tmp_path / "model.joblib")
    m.save(path)
    loaded = RiskModel.load(path)
    assert loaded.version == "v1"
    assert abs(loaded.predict_proba(X[0]) - m.predict_proba(X[0])) < 1e-9
