from spr.util.geo import haversine_km


def test_zero_distance():
    assert haversine_km(45.0, 15.0, 45.0, 15.0) == 0.0


def test_known_distance_belgrade_to_zagreb():
    # Beograd (44.82,20.46) -> Zagreb (45.81,15.98), ~370 km
    d = haversine_km(44.82, 20.46, 45.81, 15.98)
    assert 350 < d < 400
