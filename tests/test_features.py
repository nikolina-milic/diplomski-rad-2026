from datetime import datetime, timedelta, timezone
from spr.domain.schema import Event
from spr.context.store import ContextStore
from spr.context.features import compute_features, FEATURE_NAMES

BASE = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _event(**kw) -> Event:
    defaults = dict(
        event_id="e", user_id="u1", timestamp=BASE, amount=100.0,
        currency="EUR", merchant_category="grocery", country="RS",
        latitude=44.82, longitude=20.46, device_id="d1", channel="online",
    )
    defaults.update(kw)
    return Event(**defaults)


def test_feature_names_stable_and_complete():
    store = ContextStore()
    e = _event()
    feats = compute_features(e, store.get("u1"))
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_first_event_new_device_and_country():
    store = ContextStore()
    e = _event()
    feats = compute_features(e, store.get("u1"))
    assert feats["is_new_device"] == 1.0
    assert feats["is_new_country"] == 1.0
    assert feats["txn_count_24h"] == 0.0


def test_known_device_after_update():
    store = ContextStore()
    e1 = _event(event_id="e1")
    compute_features(e1, store.get("u1"))
    store.update(e1)
    e2 = _event(event_id="e2", timestamp=BASE + timedelta(hours=1))
    feats = compute_features(e2, store.get("u1"))
    assert feats["is_new_device"] == 0.0
    assert feats["is_new_country"] == 0.0
    assert feats["txn_count_24h"] == 1.0


def test_impossible_travel_speed_high():
    store = ContextStore()
    e1 = _event(event_id="e1", country="RS", latitude=44.82, longitude=20.46)
    compute_features(e1, store.get("u1"))
    store.update(e1)
    # 10 minuta kasnije u Tokiju -> ogromna brzina
    e2 = _event(event_id="e2", timestamp=BASE + timedelta(minutes=10),
                country="JP", latitude=35.68, longitude=139.69)
    feats = compute_features(e2, store.get("u1"))
    assert feats["impossible_travel_speed_kmh"] > 5000


def test_amount_zscore_flags_large_amount():
    store = ContextStore()
    for i in range(5):
        e = _event(event_id=f"n{i}", amount=100.0,
                   timestamp=BASE + timedelta(hours=i))
        compute_features(e, store.get("u1"))
        store.update(e)
    big = _event(event_id="big", amount=2000.0, timestamp=BASE + timedelta(hours=6))
    feats = compute_features(big, store.get("u1"))
    assert feats["amount_zscore"] > 3.0
