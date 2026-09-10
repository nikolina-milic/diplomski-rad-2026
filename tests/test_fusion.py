import pytest
from spr.fusion.strategies import FusionConfig, fuse


def test_weighted_average():
    cfg = FusionConfig(strategy="weighted", rules_weight=0.5, ml_weight=0.5)
    assert fuse(0.4, 0.8, cfg) == pytest.approx(0.6)


def test_weighted_respects_weights():
    cfg = FusionConfig(strategy="weighted", rules_weight=0.9, ml_weight=0.1)
    assert fuse(1.0, 0.0, cfg) == pytest.approx(0.9)


def test_cascade_prefers_rules_when_confident():
    cfg = FusionConfig(strategy="cascade", cascade_threshold=0.7)
    assert fuse(0.8, 0.2, cfg) == pytest.approx(0.8)   # pravila sigurna -> pravila
    assert fuse(0.5, 0.9, cfg) == pytest.approx(0.9)   # pravila nesigurna -> ML


def test_clamped():
    cfg = FusionConfig(strategy="weighted", rules_weight=2.0, ml_weight=0.0)
    assert 0.0 <= fuse(1.0, 1.0, cfg) <= 1.0


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        fuse(0.5, 0.5, FusionConfig(strategy="nonexistent"))
