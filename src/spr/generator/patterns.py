"""Obrasci prevare.

Namjerno su *suptilni*: dijele prostor obilježja sa legitimnim ponašanjem iz
`personas.py` (putovanje, zamjena uređaja, velika kupovina). Ako bi prevara
bila jedini uzrok „nove zemlje" ili „velikog iznosa", klasifikacija bi bila
trivijalna i evaluacija bi precijenila i ML i hibrid.
"""

import random
from datetime import datetime, timedelta
from typing import Callable

from spr.domain.banking_pack import COUNTRY_COORDS
from spr.domain.schema import Event
from spr.generator.personas import (
    SPLURGE_CATEGORIES, Persona, PersonaState, draw_amount, make_normal_event,
)


def _state(persona: Persona, state: PersonaState | None) -> PersonaState:
    return state if state is not None else PersonaState.initial(persona)


def _fraud_event(persona: Persona, timestamp: datetime, rng: random.Random,
                 fraud_type: str, state: PersonaState | None = None,
                 **overrides) -> Event:
    e = make_normal_event(persona, timestamp, rng, _state(persona, state))
    data = e.model_dump()
    data.update(is_fraud=True, fraud_type=fraud_type, **overrides)
    return Event(**data)


def _foreign(state: PersonaState, rng: random.Random) -> tuple[str, float, float]:
    others = [c for c in COUNTRY_COORDS if c != state.country]
    country = rng.choice(others)
    lat, lon = COUNTRY_COORDS[country]
    return country, lat, lon


def account_takeover(persona, timestamp, rng, state=None) -> list[Event]:
    """Preuzet nalog: novi uređaj, često nova zemlja, umjereno veći iznos.

    Multiplikator (1.5–3.5) se preklapa sa gornjim dijelom legitimne
    lognormalne raspodjele — nije ga moguće odvojiti samim pragom na iznosu.
    """
    st = _state(persona, state)
    overrides = {
        "device_id": f"att-{rng.randint(0, 9999)}",
        "amount": round(draw_amount(persona, rng) * rng.uniform(1.5, 3.5), 2),
    }
    if rng.random() < 0.7:  # u 30% slučajeva napadač je u istoj zemlji
        country, lat, lon = _foreign(st, rng)
        overrides.update(country=country, latitude=lat, longitude=lon)
    return [_fraud_event(persona, timestamp, rng, "account_takeover", st,
                         **overrides)]


def card_testing(persona, timestamp, rng, state=None) -> list[Event]:
    """Provjera ukradene kartice: niz malih iznosa.

    Dužina niza (3–8) i razmak (5 s – 8 min) su takvi da kraće serije ne
    okidaju pravilo `rapid_fire` (>= 5 transakcija u satu).
    """
    st = _state(persona, state)
    n = rng.randint(3, 8)
    step = rng.randint(5, 480)
    events = []
    for k in range(n):
        ts = timestamp + timedelta(seconds=k * step)
        events.append(_fraud_event(
            persona, ts, rng, "card_testing", st,
            amount=round(rng.uniform(0.5, 6.0), 2),
            merchant_category=rng.choice(["electronics", "entertainment"]),
        ))
    return events


def impossible_travel(persona, timestamp, rng, state=None) -> list[Event]:
    """Transakcija na trenutnoj lokaciji, pa ubrzo na drugom kraju svijeta.

    Jedini obrazac koji je po konstrukciji fizički nemoguć — ostaje jasno
    odvojiv i služi kao primjer gdje determinističko pravilo nadmašuje ML.
    """
    st = _state(persona, state)
    country, lat, lon = _foreign(st, rng)
    here = make_normal_event(persona, timestamp, rng, st)
    away = _fraud_event(
        persona, timestamp + timedelta(minutes=rng.randint(5, 45)),
        rng, "impossible_travel", st,
        country=country, latitude=lat, longitude=lon,
    )
    return [here, away]


def sudden_large_amount(persona, timestamp, rng, state=None) -> list[Event]:
    """Naglo velik iznos — preklapa se sa legitimnim `splurge` ponašanjem."""
    st = _state(persona, state)
    return [_fraud_event(
        persona, timestamp, rng, "sudden_large_amount", st,
        amount=round(draw_amount(persona, rng) * rng.uniform(4.0, 10.0), 2),
        merchant_category=rng.choice(SPLURGE_CATEGORIES),
    )]


def new_country_device(persona, timestamp, rng, state=None) -> list[Event]:
    """Nova zemlja i/ili novi uređaj — isti potpis kao legitimno putovanje
    ili zamjena telefona; razlikuje se samo po kontekstu (brzina, iznos, sat).
    """
    st = _state(persona, state)
    mode = rng.choice(["country", "device", "both"])
    overrides: dict = {}
    if mode in ("country", "both"):
        country, lat, lon = _foreign(st, rng)
        overrides.update(country=country, latitude=lat, longitude=lon)
    if mode in ("device", "both"):
        overrides["device_id"] = f"new-{rng.randint(0, 9999)}"
    return [_fraud_event(persona, timestamp, rng, "new_country_device", st,
                         **overrides)]


FRAUD_PATTERNS: dict[str, Callable[..., list[Event]]] = {
    "account_takeover": account_takeover,
    "card_testing": card_testing,
    "impossible_travel": impossible_travel,
    "sudden_large_amount": sudden_large_amount,
    "new_country_device": new_country_device,
}
