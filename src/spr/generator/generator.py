import random
from datetime import datetime, timedelta
from typing import Iterator

from spr.domain.banking_pack import COUNTRY_COORDS
from spr.domain.schema import Event
from spr.generator.patterns import FRAUD_PATTERNS
from spr.generator.personas import (
    Persona, PersonaState, make_normal_event, make_personas,
)
from spr.util.geo import haversine_km

# Najveća uvjerljiva brzina putovanja (avion + transferi). Legitimno putovanje
# mora poštovati ovu granicu da ne bi okinulo pravilo impossible_travel —
# time nova zemlja prestaje da bude siguran znak prevare.
MAX_TRAVEL_KMH = 850.0

# Snimak stanja persone u jednom trenutku (za naknadno ubacivanje prevara).
_Snapshot = tuple[str, float, float, tuple[str, ...]]


def _pick_hour(persona: Persona, rng: random.Random) -> int:
    if persona.is_night_owl:
        return rng.choice([21, 22, 23, 0, 1, 2, 3, 4])
    if rng.random() < persona.night_rate:
        return rng.randint(0, 5)
    lo, hi = persona.active_hours
    return rng.randrange(lo, hi)


def _align_to_active_hour(ts: datetime, persona: Persona,
                          rng: random.Random) -> datetime:
    hour = _pick_hour(persona, rng)
    return ts.replace(hour=hour, minute=rng.randrange(60),
                      second=rng.randrange(60), microsecond=0)


def _travel_gap(from_lat: float, from_lon: float, to_lat: float,
                to_lon: float) -> timedelta:
    """Minimalno vrijeme potrebno da se relacija pređe uvjerljivom brzinom."""
    dist = haversine_km(from_lat, from_lon, to_lat, to_lon)
    return timedelta(hours=dist / MAX_TRAVEL_KMH + 1.0)


def _maybe_travel(state: PersonaState, ts: datetime,
                  rng: random.Random) -> datetime:
    """Započni ili završi legitimno putovanje; vrati (možda pomjereno) vrijeme.

    Vrijeme se pomjera unaprijed tačno toliko da brzina ostane uvjerljiva.
    """
    p = state.persona
    if state.trip_ends is not None:
        if ts >= state.trip_ends:  # povratak kući
            ts = ts + _travel_gap(state.lat, state.lon, p.home_lat, p.home_lon)
            state.country, state.lat, state.lon = (
                p.home_country, p.home_lat, p.home_lon)
            state.trip_ends = None
        return ts

    if rng.random() >= p.travel_rate:
        return ts

    destinations = [c for c in COUNTRY_COORDS if c != state.country]
    country = rng.choice(destinations)
    lat, lon = COUNTRY_COORDS[country]
    ts = ts + _travel_gap(state.lat, state.lon, lat, lon)
    state.country, state.lat, state.lon = country, lat, lon
    state.trip_ends = ts + timedelta(days=rng.randint(2, 10))
    return ts


def _maybe_change_device(state: PersonaState, rng: random.Random) -> None:
    """Legitimna zamjena uređaja — novi uređaj ostaje u upotrebi."""
    if rng.random() >= state.persona.device_change_rate:
        return
    new_device = f"{state.persona.user_id}-dev-{rng.randint(1000, 9999)}"
    if len(state.devices) >= 3:
        state.devices.pop(0)
    state.devices.append(new_device)


def _simulate_persona(persona: Persona, n_events: int, start: datetime,
                      days: int, rng: random.Random,
                      ) -> tuple[list[Event], list[tuple[datetime, _Snapshot]]]:
    """Hronološka legitimna istorija jedne persone + snimci stanja."""
    state = PersonaState.initial(persona)
    mean_gap = (days * 86400) / max(n_events, 1)
    events: list[Event] = []
    snapshots: list[tuple[datetime, _Snapshot]] = []
    ts = start
    for _ in range(n_events):
        ts = ts + timedelta(seconds=max(60.0, rng.expovariate(1.0 / mean_gap)))
        ts = _align_to_active_hour(ts, persona, rng)
        if events and ts <= events[-1].timestamp:
            ts = events[-1].timestamp + timedelta(seconds=rng.randint(60, 3600))
        ts = _maybe_travel(state, ts, rng)
        _maybe_change_device(state, rng)
        events.append(make_normal_event(persona, ts, rng, state))
        snapshots.append((ts, (state.country, state.lat, state.lon,
                               tuple(state.devices))))
    return events, snapshots


def _state_at(persona: Persona, snapshot: _Snapshot) -> PersonaState:
    country, lat, lon, devices = snapshot
    return PersonaState(persona=persona, country=country, lat=lat, lon=lon,
                        devices=list(devices))


def generate_batch(
    n_users: int,
    n_normal: int,
    fraud_rate: float,
    seed: int,
    start: datetime,
    days: int,
    patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Event]:
    """Generiši hronološki sortiran batch događaja.

    Legitimne transakcije se simuliraju po personi sa kontinuitetom stanja:
    korisnici putuju, mijenjaju uređaje, povremeno naprave veliku kupovinu i
    troše van uobičajenih sati. Prevare se ubacuju u tu istoriju, pa dijele
    prostor obilježja sa legitimnim ponašanjem umjesto da budu trivijalno
    odvojive.

    n_normal: ukupan broj legitimnih transakcija.
    fraud_rate: ciljani udio prevara u ukupnom skupu (0..1).
    patterns: koje obrasce prevare koristiti (podrazumijevano sve).
    exclude_patterns: koje izostaviti — za eksperiment „nov napad", gdje se
        model trenira bez jednog obrasca, a testira na njemu.
    """
    rng = random.Random(seed)
    personas = make_personas(n_users, rng)

    weights = [p.txns_per_day for p in personas]
    total_w = sum(weights)
    counts = [max(1, round(n_normal * w / total_w)) for w in weights]

    events: list[Event] = []
    timelines: list[list[tuple[datetime, _Snapshot]]] = []
    for persona, k in zip(personas, counts):
        legit, snaps = _simulate_persona(persona, k, start, days, rng)
        events.extend(legit)
        timelines.append(snaps)

    n_fraud_target = max(1, round(len(events) * fraud_rate / (1 - fraud_rate)))
    pattern_names = list(patterns) if patterns else list(FRAUD_PATTERNS.keys())
    if exclude_patterns:
        pattern_names = [n for n in pattern_names if n not in exclude_patterns]
    if not pattern_names:
        events.sort(key=lambda e: e.timestamp)
        return events
    n_fraud = 0
    guard = 0
    while n_fraud < n_fraud_target and guard < n_fraud_target * 20:
        guard += 1
        idx = rng.randrange(len(personas))
        snaps = timelines[idx]
        if not snaps:
            continue
        ts, snapshot = snaps[rng.randrange(len(snaps))]
        state = _state_at(personas[idx], snapshot)
        burst = FRAUD_PATTERNS[rng.choice(pattern_names)](
            personas[idx], ts + timedelta(minutes=rng.randint(1, 240)), rng, state)
        events.extend(burst)
        n_fraud += sum(1 for e in burst if e.is_fraud)

    events.sort(key=lambda e: e.timestamp)
    return events


def stream_events(
    n_users: int,
    fraud_rate: float,
    seed: int,
    start: datetime,
    count: int,
) -> Iterator[Event]:
    """Lazy tok događaja u rastućem vremenu; umeće prevare prema fraud_rate.

    Kao i `generate_batch`, održava stanje persone (lokacija, uređaji), pa se
    u toku pojavljuju i legitimna putovanja i legitimne zamjene uređaja.

    `count` je minimalan broj događaja koji će se emitovati (fraud obrasci
    koji vraćaju više događaja mogu premašiti taj broj).
    """
    rng = random.Random(seed)
    personas = make_personas(n_users, rng)
    states = {p.user_id: PersonaState.initial(p) for p in personas}
    pattern_names = list(FRAUD_PATTERNS.keys())
    current = start
    emitted = 0
    while emitted < count:
        current = current + timedelta(seconds=rng.randint(1, 120))
        persona = rng.choice(personas)
        state = states[persona.user_id]
        if rng.random() < fraud_rate:
            batch = FRAUD_PATTERNS[rng.choice(pattern_names)](
                persona, current, rng, state)
        else:
            ts = _maybe_travel(state, current, rng)
            _maybe_change_device(state, rng)
            batch = [make_normal_event(persona, ts, rng, state)]
        # Obrasci mogu emitovati događaje van redoslijeda i nekoliko minuta
        # unaprijed; sortiraj batch i pomjeri kursor na najkasniji da tok
        # ostane hronološki nerastući.
        batch.sort(key=lambda e: e.timestamp)
        for e in batch:
            yield e
            emitted += 1
        current = max(current, batch[-1].timestamp)
