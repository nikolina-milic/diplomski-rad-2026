from datetime import datetime, timezone
from spr.generator.generator import stream_events

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_stream_yields_requested_count_min():
    events = list(stream_events(n_users=5, fraud_rate=0.1, seed=1, start=START, count=50))
    # count je minimum; obrasci sa više događaja mogu premašiti
    assert len(events) >= 50


def test_stream_is_lazy_iterator():
    gen = stream_events(n_users=5, fraud_rate=0.1, seed=1, start=START, count=50)
    assert not isinstance(gen, list)
    first = next(gen)
    assert first.user_id.startswith("u")


def test_stream_deterministic():
    a = [e.event_id for e in stream_events(n_users=5, fraud_rate=0.1, seed=7, start=START, count=30)]
    b = [e.event_id for e in stream_events(n_users=5, fraud_rate=0.1, seed=7, start=START, count=30)]
    assert a == b


def test_stream_timestamps_non_decreasing():
    events = list(stream_events(n_users=5, fraud_rate=0.1, seed=2, start=START, count=40))
    ts = [e.timestamp for e in events]
    assert ts == sorted(ts)
