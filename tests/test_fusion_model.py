import numpy as np
import pytest

from spr.fusion.model import FusionModel
from spr.fusion.strategies import FusionConfig, fuse


def _data(n=1500, seed=0):
    """Pravila su slab signal, ML jak — meta-model treba to da nauči."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    rules = np.clip(rng.normal(0.3 + 0.15 * y, 0.25), 0, 1)
    ml = np.clip(rng.normal(0.2 + 0.6 * y, 0.15), 0, 1)
    return rules.tolist(), ml.tolist(), y.tolist()


def test_unfitted_model_falls_back_to_plain_fuse():
    cfg = FusionConfig(strategy="weighted")
    fm = FusionModel(cfg)
    assert not fm.fitted
    assert fm.combine(0.4, 0.8) == pytest.approx(fuse(0.4, 0.8, cfg))


def test_fit_learns_meta_weights():
    rules, ml, y = _data()
    fm = FusionModel(FusionConfig(strategy="stacking")).fit(rules, ml, y)
    assert fm.fitted
    w = fm.weights()
    assert set(w) == {"rules", "ml", "interaction", "intercept"}
    # ML je jači signal, pa mora dobiti veću težinu od pravila
    assert w["ml"] > w["rules"]


def test_stacking_beats_naive_weighted_when_sources_differ_in_quality():
    from spr.learning.metrics import compute_metrics
    rules, ml, y = _data()
    naive = [fuse(r, m, FusionConfig(strategy="weighted")) for r, m in zip(rules, ml)]
    fm = FusionModel(FusionConfig(strategy="stacking")).fit(rules, ml, y)
    stacked = fm.combine_batch(rules, ml)
    assert compute_metrics(y, stacked)["pr_auc"] > compute_metrics(y, naive)["pr_auc"]


def test_combine_batch_matches_combine_one_by_one():
    rules, ml, y = _data(n=400)
    fm = FusionModel(FusionConfig(strategy="stacking")).fit(rules, ml, y)
    batch = fm.combine_batch(rules[:20], ml[:20])
    one = [fm.combine(r, m) for r, m in zip(rules[:20], ml[:20])]
    assert np.allclose(batch, one)


def test_output_is_a_probability():
    rules, ml, y = _data(n=400)
    for strategy in ("weighted", "cascade", "stacking"):
        fm = FusionModel(FusionConfig(strategy=strategy)).fit(rules, ml, y)
        out = fm.combine_batch(rules, ml)
        assert all(0.0 <= v <= 1.0 for v in out)


def test_single_class_does_not_crash():
    fm = FusionModel(FusionConfig(strategy="stacking")).fit(
        [0.1, 0.2], [0.3, 0.4], [0, 0])
    assert not fm.fitted
    assert 0.0 <= fm.combine(0.1, 0.3) <= 1.0


def test_calibration_shared_config_object_reflects_strategy_change():
    """Engine mijenja strategiju preko /config/fusion — FusionModel to mora
    odmah poštovati jer dijeli isti config objekat."""
    cfg = FusionConfig(strategy="weighted")
    rules, ml, y = _data(n=400)
    fm = FusionModel(cfg).fit(rules, ml, y)
    before = fm.combine(0.5, 0.5)
    cfg.strategy = "stacking"
    assert fm.combine(0.5, 0.5) != before
