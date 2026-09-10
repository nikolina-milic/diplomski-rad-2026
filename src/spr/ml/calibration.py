"""Kalibracija skorova rizika.

Problem: `rules_score` je noisy-OR kombinacija težina pravila, a `ml_score`
vjerovatnoća iz klasifikatora. To su veličine na različitim skalama — njihovo
usrednjavanje u fusion sloju nije opravdano. Kalibracija oba skora preslikava
u zajedničku skalu: „skor 0.8 znači da je u 80% takvih slučajeva riječ o
prevari". Tek tada je kombinovanje smisleno.

Mjere kvaliteta kalibracije: Brier score i ECE (expected calibration error),
uz podatke za reliability dijagram.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    """Preslikava sirov skor u kalibrisanu vjerovatnoću.

    method="isotonic" — monotona neparametarska regresija (fleksibilnija,
        traži više podataka).
    method="platt"    — logistička regresija nad skorom (robusnija na malim
        skupovima; klasičan Plattov postupak).
    method="none"     — identitet (za poređenje bez kalibracije).
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "platt", "none"):
            raise ValueError(f"Nepoznata metoda kalibracije: {method}")
        self.method = method
        self._model = None
        self.fitted = False

    def fit(self, scores, y) -> "Calibrator":
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        if self.method == "none" or len(set(y.tolist())) < 2:
            # bez obje klase kalibracija nije definisana — ostani identitet
            self.fitted = self.method == "none"
            return self
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip",
                                             y_min=0.0, y_max=1.0)
            self._model.fit(s.ravel(), y)
        else:
            self._model = LogisticRegression(max_iter=1000)
            self._model.fit(s, y)
        self.fitted = True
        return self

    def transform(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if self._model is None:
            return np.clip(s, 0.0, 1.0)
        if self.method == "isotonic":
            out = self._model.predict(s.ravel())
        else:
            out = self._model.predict_proba(s.reshape(-1, 1))[:, 1]
        return np.clip(out, 0.0, 1.0)

    def transform_one(self, score: float) -> float:
        return float(self.transform([score])[0])


def brier_score(y_true, y_prob) -> float:
    """Srednja kvadratna greška vjerovatnoće — niže je bolje."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


@dataclass
class ReliabilityBin:
    bin_start: float
    bin_end: float
    mean_predicted: float
    fraction_positive: float
    count: int


def reliability_curve(y_true, y_prob, bins: int = 10) -> list[ReliabilityBin]:
    """Podaci za reliability dijagram: predviđeno vs. stvarno po korpama."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(ReliabilityBin(
            bin_start=float(lo), bin_end=float(hi),
            mean_predicted=float(p[mask].mean()),
            fraction_positive=float(y[mask].mean()),
            count=n,
        ))
    return out


def expected_calibration_error(y_true, y_prob, bins: int = 10) -> float:
    """ECE: prosječno odstupanje predviđene od stvarne učestalosti, ponderisano
    brojem uzoraka po korpi. 0 = savršena kalibracija."""
    curve = reliability_curve(y_true, y_prob, bins=bins)
    total = sum(b.count for b in curve)
    if total == 0:
        return 0.0
    return float(sum(b.count * abs(b.mean_predicted - b.fraction_positive)
                     for b in curve) / total)


def calibration_report(y_true, y_prob, bins: int = 10) -> dict:
    return {
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, bins=bins),
        "reliability": [b.__dict__ for b in reliability_curve(y_true, y_prob, bins)],
    }
