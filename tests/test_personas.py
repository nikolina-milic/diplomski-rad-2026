import random
from datetime import datetime, timezone
from spr.domain.banking_pack import FIRST_NAMES, LAST_NAMES
from spr.generator.personas import make_personas, make_normal_event


def test_make_personas_deterministic():
    a = make_personas(5, random.Random(42))
    b = make_personas(5, random.Random(42))
    assert [p.user_id for p in a] == [p.user_id for p in b]
    assert a[0].amount_mean == b[0].amount_mean


def test_make_personas_count_and_fields():
    ps = make_personas(3, random.Random(1))
    assert len(ps) == 3
    for p in ps:
        assert p.amount_mean > 0
        assert p.amount_std >= 0
        assert len(p.devices) >= 1
        assert 0 <= p.active_hours[0] < p.active_hours[1] <= 24


def test_make_normal_event_is_consistent_with_persona():
    p = make_personas(1, random.Random(7))[0]
    rng = random.Random(7)
    e = make_normal_event(p, datetime(2026, 1, 1, 12, tzinfo=timezone.utc), rng)
    assert e.is_fraud is False
    assert e.user_id == p.user_id
    assert e.country == p.home_country
    assert e.device_id in p.devices
    assert e.amount > 0


def test_personas_have_names_from_home_country():
    for p in make_personas(20, random.Random(3)):
        assert p.first_name in FIRST_NAMES[p.home_country]
        assert p.last_name in LAST_NAMES[p.home_country]


def test_names_are_deterministic():
    a = make_personas(5, random.Random(42))
    b = make_personas(5, random.Random(42))
    assert [(p.first_name, p.last_name) for p in a] == \
           [(p.first_name, p.last_name) for p in b]


def test_event_carries_persona_name():
    p = make_personas(1, random.Random(7))[0]
    e = make_normal_event(p, datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
                          random.Random(7))
    assert e.first_name == p.first_name
    assert e.last_name == p.last_name
