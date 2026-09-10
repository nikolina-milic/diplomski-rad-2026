import numpy as np
import pytest

from spr.ml.calibration import (
    Calibrator, brier_score, calibration_report, expected_calibration_error,
    reliability_curve,
)


def _skewed_scores(n=2000, seed=0):
    """Skorovi koji rangiraju dobro, ali su sistematski previsoki —
    tipičan nekalibrisan klasifikator."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    raw = np.clip(rng.normal(0.55 + 0.3 * y, 0.15), 0, 1)
    return raw.tolist(), y.tolist()


def test_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError):
        Calibrator("magija")


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_improves_brier_and_ece(method):
    scores, y = _skewed_scores()
    before = brier_score(y, scores)
    cal = Calibrator(method).fit(scores, y)
    after = brier_score(y, cal.transform(scores))
    assert after < before
    assert expected_calibration_error(y, cal.transform(scores)) < \
           expected_calibration_error(y, scores)


def test_none_method_is_identity():
    scores, y = _skewed_scores(n=200)
    cal = Calibrator("none").fit(scores, y)
    assert np.allclose(cal.transform(scores), scores)


def test_calibrator_output_stays_in_unit_interval():
    scores, y = _skewed_scores(n=500)
    cal = Calibrator("isotonic").fit(scores, y)
    out = cal.transform([-5.0, 0.0, 0.5, 1.0, 7.0])
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_single_class_leaves_calibrator_as_identity():
    """Bez obje klase kalibracija nije definisana — ne smije puknuti."""
    cal = Calibrator("isotonic").fit([0.1, 0.2, 0.3], [0, 0, 0])
    assert cal.transform_one(0.2) == pytest.approx(0.2)


def test_perfect_calibration_has_near_zero_ece():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.05


def test_reliability_curve_bins_are_ordered_and_counted():
    scores, y = _skewed_scores(n=1000)
    curve = reliability_curve(y, scores, bins=10)
    assert curve
    assert sum(b.count for b in curve) == len(y)
    assert curve == sorted(curve, key=lambda b: b.bin_start)


def test_calibration_report_shape():
    scores, y = _skewed_scores(n=400)
    rep = calibration_report(y, scores)
    assert set(rep) == {"brier", "ece", "reliability"}
    assert rep["reliability"][0].keys() >= {"mean_predicted", "fraction_positive"}
