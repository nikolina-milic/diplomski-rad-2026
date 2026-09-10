from datetime import datetime, timezone
from spr.domain.schema import Event


def test_event_minimal_defaults():
    e = Event(
        event_id="e1", user_id="u1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        amount=42.0, currency="EUR", merchant_category="grocery",
        country="RS", latitude=44.82, longitude=20.46,
        device_id="d1", channel="online",
    )
    assert e.is_fraud is False
    assert e.fraud_type is None


def test_event_fraud_fields():
    e = Event(
        event_id="e2", user_id="u1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        amount=9000.0, currency="EUR", merchant_category="electronics",
        country="XX", latitude=0.0, longitude=0.0,
        device_id="d9", channel="online",
        is_fraud=True, fraud_type="impossible_travel",
    )
    assert e.is_fraud is True
    assert e.fraud_type == "impossible_travel"
