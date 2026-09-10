from datetime import datetime, timezone
from spr.engine.factory import build_default_engine
from spr.domain.schema import Event

START = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _event(**kw):
    d = dict(event_id="e1", user_id="tester", timestamp=START, amount=100.0,
             currency="EUR", merchant_category="grocery", country="RS",
             latitude=44.82, longitude=20.46, device_id="d1", channel="online")
    d.update(kw)
    return Event(**d)


def test_score_returns_decision_fields():
    eng = build_default_engine(n_normal=300, days=7)
    dec = eng.score(_event())
    assert dec.event_id == "e1"
    assert dec.level in {"LOW", "MEDIUM", "HIGH"}
    assert dec.action in {"ALLOW", "CHALLENGE", "REVIEW", "BLOCK"}
    assert 0.0 <= dec.final_score <= 1.0
    assert dec.mode == "shadow"
    assert dec.explanation


def test_guardrail_forces_block_on_huge_amount():
    eng = build_default_engine(n_normal=300, days=7)
    dec = eng.score(_event(amount=20000.0))
    assert dec.action == "BLOCK"
    assert dec.guardrail == "compliance_hard_limit"
