"""Režim `shadow` vs `enforce`.

Ranije je `mode` bio samo string koji se validira i upisuje — nije mijenjao
ponašanje sistema, pa je „shadow režim" bio tvrdnja bez pokrića.
"""

from datetime import datetime, timezone

import pytest

from spr.domain.schema import Event
from spr.engine.factory import build_default_engine
from spr.policy.engine import Action, PolicyConfig, RiskLevel, execute


def _policy(mode):
    return PolicyConfig(low_max=0.3, medium_max=0.7, mode=mode, level_actions={
        RiskLevel.LOW: Action.ALLOW, RiskLevel.MEDIUM: Action.CHALLENGE,
        RiskLevel.HIGH: Action.REVIEW})


@pytest.mark.parametrize("action", list(Action))
def test_shadow_never_executes_anything(action):
    assert execute(action, _policy("shadow")) is Action.ALLOW


@pytest.mark.parametrize("action", list(Action))
def test_enforce_executes_the_proposed_action(action):
    assert execute(action, _policy("enforce")) is action


def _risky_event():
    return Event(
        event_id="e-risky", user_id="u-risky",
        timestamp=datetime(2026, 3, 1, 3, tzinfo=timezone.utc),
        amount=99_000.0, currency="EUR", merchant_category="electronics",
        country="JP", latitude=35.68, longitude=139.69,
        device_id="dev-unknown", channel="online")


@pytest.fixture(scope="module")
def engine():
    return build_default_engine(n_normal=400, days=7)


def test_decision_reports_both_proposed_and_executed(engine):
    engine.policy.mode = "shadow"
    d = engine.score(_risky_event())
    assert d.action != "ALLOW"          # guardrail za veliki iznos je okinuo
    assert d.executed_action == "ALLOW"  # ali se u shadow režimu ne izvršava
    assert d.mode == "shadow"


def test_enforce_makes_executed_match_proposed(engine):
    engine.policy.mode = "enforce"
    try:
        d = engine.score(_risky_event())
        assert d.executed_action == d.action
        assert d.action == "BLOCK"      # compliance_hard_limit
    finally:
        engine.policy.mode = "shadow"
