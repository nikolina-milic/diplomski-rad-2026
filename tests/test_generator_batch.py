from datetime import datetime, timezone
from spr.generator.generator import generate_batch

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_batch_deterministic():
    a = generate_batch(n_users=10, n_normal=200, fraud_rate=0.05, seed=1, start=START, days=7)
    b = generate_batch(n_users=10, n_normal=200, fraud_rate=0.05, seed=1, start=START, days=7)
    assert [e.event_id for e in a] == [e.event_id for e in b]


def test_batch_sorted_by_timestamp():
    events = generate_batch(n_users=10, n_normal=200, fraud_rate=0.05, seed=2, start=START, days=7)
    ts = [e.timestamp for e in events]
    assert ts == sorted(ts)


def test_batch_has_both_classes():
    events = generate_batch(n_users=10, n_normal=500, fraud_rate=0.1, seed=3, start=START, days=7)
    frauds = [e for e in events if e.is_fraud]
    normals = [e for e in events if not e.is_fraud]
    assert len(frauds) > 0
    assert len(normals) > 0
    # gruba provjera da fraud udio nije apsurdan
    ratio = len(frauds) / len(events)
    assert 0.02 < ratio < 0.5
