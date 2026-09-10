from dataclasses import dataclass, field
from datetime import datetime

from spr.domain.schema import Event


@dataclass
class UserState:
    amounts: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    seen_devices: set[str] = field(default_factory=set)
    seen_countries: set[str] = field(default_factory=set)
    last_event: Event | None = None
    home_lat: float | None = None
    home_lon: float | None = None
    home_country: str | None = None


class ContextStore:
    """In-memory store istorije korisnika.

    Interfejs (get/update) je namjerno minimalan da se kasnije zamijeni
    Redis/PostgreSQL implementacijom bez promjene pozivaoca.
    """

    def __init__(self) -> None:
        self._states: dict[str, UserState] = {}

    def get(self, user_id: str) -> UserState:
        return self._states.setdefault(user_id, UserState())

    def update(self, event: Event) -> None:
        s = self.get(event.user_id)
        if s.home_lat is None:  # prva viđena lokacija = "dom"
            s.home_lat, s.home_lon = event.latitude, event.longitude
            s.home_country = event.country
        s.amounts.append(event.amount)
        s.timestamps.append(event.timestamp)
        s.seen_devices.add(event.device_id)
        s.seen_countries.add(event.country)
        s.last_event = event
