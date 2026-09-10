import random
from datetime import datetime, timezone
from spr.generator.personas import Persona, PersonaState, make_personas
from spr.generator.patterns import FRAUD_PATTERNS
from spr.util.geo import haversine_km

START = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _persona() -> Persona:
    return make_personas(1, random.Random(3))[0]


def test_all_patterns_labeled_fraud():
    p = _persona()
    for name, fn in FRAUD_PATTERNS.items():
        events = fn(p, START, random.Random(1))
        assert len(events) >= 1
        # svaki fraud događaj mora nositi ovaj fraud_type; obrasci koji
        # ubacuju i legitiman "sidro" događaj (npr. impossible_travel) imaju
        # bar jedan događaj sa is_fraud=True
        assert any(e.is_fraud and e.fraud_type == name for e in events)
        for e in events:
            assert e.user_id == p.user_id


def test_all_patterns_accept_persona_state():
    """Obrasci polaze od trenutne lokacije/uređaja persone, ne od doma."""
    p = _persona()
    st = PersonaState(persona=p, country="JP", lat=35.68, lon=139.69,
                      devices=["dev-x"])
    for name, fn in FRAUD_PATTERNS.items():
        events = fn(p, START, random.Random(1), st)
        assert all(e.user_id == p.user_id for e in events)


def test_card_testing_is_a_short_burst_of_small_amounts():
    p = _persona()
    for seed in range(20):
        events = FRAUD_PATTERNS["card_testing"](p, START, random.Random(seed))
        # namjerno kratke serije (3–8): kraće od praga pravila rapid_fire,
        # pa ih pravila ne hvataju automatski
        assert 3 <= len(events) <= 8
        assert all(e.amount <= 6.0 for e in events)


def test_card_testing_bursts_are_not_always_rule_detectable():
    """Bar neka serija ima < 5 događaja — ispod praga pravila rapid_fire."""
    p = _persona()
    sizes = {len(FRAUD_PATTERNS["card_testing"](p, START, random.Random(s)))
             for s in range(30)}
    assert any(n < 5 for n in sizes)


def test_sudden_large_amount_overlaps_legit_splurge_range():
    """Multiplikator prevare (4–10x) ulazi u opseg legitimnog splurge-a (4–12x),
    pa iznos sam po sebi nije dovoljan za odluku."""
    p = _persona()
    for seed in range(20):
        e = FRAUD_PATTERNS["sudden_large_amount"](p, START, random.Random(seed))[0]
        assert e.amount > p.amount_mean * 1.5
        assert e.is_fraud


def test_impossible_travel_speed_is_physically_impossible():
    p = _persona()
    for seed in range(20):
        here, away = FRAUD_PATTERNS["impossible_travel"](
            p, START, random.Random(seed))
        dist = haversine_km(here.latitude, here.longitude,
                            away.latitude, away.longitude)
        hours = (away.timestamp - here.timestamp).total_seconds() / 3600.0
        assert dist / hours > 1000.0


def test_new_country_device_varies_its_signature():
    """Ne pojavljuje se uvijek kao 'nova zemlja I novi uređaj' — inače bi ga
    kombinacija dva pravila uvijek uhvatila."""
    p = _persona()
    signatures = set()
    for seed in range(40):
        e = FRAUD_PATTERNS["new_country_device"](p, START, random.Random(seed))[0]
        signatures.add((e.country != p.home_country, e.device_id not in p.devices))
    assert len(signatures) >= 2


def test_account_takeover_is_not_always_from_abroad():
    p = _persona()
    countries = {FRAUD_PATTERNS["account_takeover"](p, START, random.Random(s))[0].country
                 for s in range(40)}
    assert p.home_country in countries      # napadač ponekad u istoj zemlji
    assert len(countries) > 1
