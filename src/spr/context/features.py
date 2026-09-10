from statistics import mean, pstdev

from spr.context.store import UserState
from spr.domain.schema import Event
from spr.util.geo import haversine_km

FEATURE_NAMES = [
    "amount",
    "amount_zscore",
    "amount_ratio",
    "txn_count_1h",
    "txn_count_24h",
    "is_new_device",
    "is_new_country",
    "hour_of_day",
    "is_unusual_hour",
    "seconds_since_last",
    "impossible_travel_speed_kmh",
    "distance_from_home_km",
]

_EPS = 1e-9


def compute_features(event: Event, state: UserState) -> dict[str, float]:
    """Feature vektor iz stanja PRIJE dodavanja događaja (no leakage)."""
    amounts = state.amounts
    if amounts:
        mu = mean(amounts)
        sigma = pstdev(amounts) if len(amounts) > 1 else 0.0
        # pod prag na devijaciji: pri <2 uzorka ili istovjetnim iznosima
        # sigma je 0, pa bi z-score eksplodirao — spuštamo na stabilan raspon
        amount_zscore = (event.amount - mu) / max(sigma, 1.0)
        amount_ratio = event.amount / (mu + _EPS)
    else:
        amount_zscore = 0.0
        amount_ratio = 1.0

    ts = event.timestamp
    count_1h = sum(1 for t in state.timestamps if 0 <= (ts - t).total_seconds() <= 3600)
    count_24h = sum(1 for t in state.timestamps if 0 <= (ts - t).total_seconds() <= 86400)

    is_new_device = 0.0 if event.device_id in state.seen_devices else 1.0
    is_new_country = 0.0 if event.country in state.seen_countries else 1.0

    hour = float(ts.hour)
    # neuobičajen sat: 1..5 ujutru (gruba heuristika; kasnije po personi)
    is_unusual_hour = 1.0 if 1 <= ts.hour <= 5 else 0.0

    if state.last_event is not None:
        gap = (ts - state.last_event.timestamp).total_seconds()
        seconds_since_last = max(0.0, gap)
        dist = haversine_km(
            state.last_event.latitude, state.last_event.longitude,
            event.latitude, event.longitude,
        )
        hours = max(gap / 3600.0, _EPS)
        speed = dist / hours
    else:
        seconds_since_last = 0.0
        speed = 0.0

    if state.home_lat is not None:
        distance_from_home = haversine_km(
            state.home_lat, state.home_lon, event.latitude, event.longitude
        )
    else:
        distance_from_home = 0.0

    return {
        "amount": event.amount,
        "amount_zscore": amount_zscore,
        "amount_ratio": amount_ratio,
        "txn_count_1h": float(count_1h),
        "txn_count_24h": float(count_24h),
        "is_new_device": is_new_device,
        "is_new_country": is_new_country,
        "hour_of_day": hour,
        "is_unusual_hour": is_unusual_hour,
        "seconds_since_last": seconds_since_last,
        "impossible_travel_speed_kmh": speed,
        "distance_from_home_km": distance_from_home,
    }
