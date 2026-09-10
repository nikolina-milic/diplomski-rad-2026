import math
import random
from dataclasses import dataclass, field
from datetime import datetime

from spr.domain.banking_pack import (
    COUNTRY_COORDS,
    DEFAULT_CURRENCY,
    FIRST_NAMES,
    LAST_NAMES,
    MERCHANT_CATEGORIES,
)
from spr.domain.schema import Event

# Kategorije u kojima je legitimna velika kupovina uvjerljiva.
SPLURGE_CATEGORIES = ["electronics", "travel", "clothing", "entertainment"]


@dataclass
class Persona:
    user_id: str
    first_name: str
    last_name: str
    home_country: str
    home_lat: float
    home_lon: float
    amount_mean: float
    amount_std: float
    devices: list[str]
    categories: list[str]
    active_hours: tuple[int, int]  # [start, end) sat u danu
    txns_per_day: float
    # --- osobine legitimnog "anomalnog" ponašanja -------------------------
    # Bez njih su klase trivijalno separabilne (nova zemlja => prevara), pa
    # ML dostiže nerealno visok PR-AUC, a hibrid nema šta da doprinese.
    sigma_log: float = 0.35        # disperzija lognormalne raspodjele iznosa
    travel_rate: float = 0.02      # vjerovatnoća početka putovanja po danu
    device_change_rate: float = 0.01  # vjerovatnoća zamjene uređaja po danu
    splurge_rate: float = 0.03     # udio legitimno velikih kupovina
    night_rate: float = 0.05       # udio transakcija van aktivnih sati
    is_night_owl: bool = False     # persona koja redovno radi noću


@dataclass
class PersonaState:
    """Stanje persone u toku simulacije (lokacija, uređaji, putovanje).

    Omogućava da legitimno ponašanje ima kontinuitet: kad korisnik otputuje,
    naredne transakcije dolaze iz te zemlje dok se putovanje ne završi; kad
    zamijeni telefon, novi uređaj ostaje u upotrebi.
    """
    persona: Persona
    country: str
    lat: float
    lon: float
    devices: list[str] = field(default_factory=list)
    trip_ends: datetime | None = None

    @classmethod
    def initial(cls, persona: Persona) -> "PersonaState":
        return cls(persona=persona, country=persona.home_country,
                   lat=persona.home_lat, lon=persona.home_lon,
                   devices=list(persona.devices))


def make_personas(n: int, rng: random.Random) -> list[Persona]:
    countries = list(COUNTRY_COORDS.keys())
    personas: list[Persona] = []
    for i in range(n):
        country = rng.choice(countries)
        lat, lon = COUNTRY_COORDS[country]
        first_name = rng.choice(FIRST_NAMES[country])
        last_name = rng.choice(LAST_NAMES[country])
        mean = rng.uniform(20, 150)
        n_devices = rng.randint(1, 3)
        n_cats = rng.randint(2, 5)
        start = rng.randint(6, 10)
        end = rng.randint(18, 23)
        night_owl = rng.random() < 0.15
        personas.append(Persona(
            user_id=f"u{i:04d}",
            first_name=first_name,
            last_name=last_name,
            home_country=country,
            home_lat=lat,
            home_lon=lon,
            amount_mean=mean,
            amount_std=mean * rng.uniform(0.15, 0.4),
            devices=[f"dev-{i:04d}-{k}" for k in range(n_devices)],
            categories=rng.sample(MERCHANT_CATEGORIES, n_cats),
            active_hours=(start, end),
            txns_per_day=rng.uniform(1.0, 6.0),
            sigma_log=rng.uniform(0.25, 0.6),
            travel_rate=rng.uniform(0.005, 0.05),
            device_change_rate=rng.uniform(0.002, 0.02),
            splurge_rate=rng.uniform(0.01, 0.06),
            night_rate=rng.uniform(0.02, 0.12),
            is_night_owl=night_owl,
        ))
    return personas


def draw_amount(persona: Persona, rng: random.Random, splurge: bool = False) -> float:
    """Lognormalni iznos: pozitivan, sa realističnim desnim repom.

    Gaussov iznos (prethodna verzija) nema rep, pa je svaki veliki iznos
    automatski prevara — upravo to je klase činilo separabilnim.
    """
    mu = math.log(max(persona.amount_mean, 1.0))
    amount = math.exp(rng.gauss(mu, persona.sigma_log))
    if splurge:
        amount *= rng.uniform(4.0, 12.0)
    return round(max(1.0, amount), 2)


def make_normal_event(
    persona: Persona, timestamp: datetime, rng: random.Random,
    state: PersonaState | None = None,
) -> Event:
    """Legitiman događaj. Uz `state` poštuje trenutnu lokaciju i uređaje."""
    country = state.country if state else persona.home_country
    lat = state.lat if state else persona.home_lat
    lon = state.lon if state else persona.home_lon
    devices = state.devices if state else persona.devices

    splurge = rng.random() < persona.splurge_rate
    amount = draw_amount(persona, rng, splurge=splurge)
    category = (rng.choice(SPLURGE_CATEGORIES) if splurge
                else rng.choice(persona.categories))

    return Event(
        event_id=f"{persona.user_id}-{timestamp.timestamp():.0f}-{rng.randint(0, 999999)}",
        user_id=persona.user_id,
        first_name=persona.first_name,
        last_name=persona.last_name,
        timestamp=timestamp,
        amount=amount,
        currency=DEFAULT_CURRENCY,
        merchant_category=category,
        country=country,
        latitude=lat,
        longitude=lon,
        device_id=rng.choice(devices),
        channel=rng.choice(["online", "card_present", "atm"]),
        is_fraud=False,
        fraud_type=None,
    )
