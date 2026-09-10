"""Detekcija pomjeranja raspodjele (data drift).

Model se trenira na jednoj raspodjeli, a u produkciji sreće drugu: mijenja se
ponašanje korisnika, ulaze novi kanali, napadači mijenjaju taktiku. Bez
nadzora nad ulazima to se primijeti tek kad metrike padnu — a metrike traže
labele, koje kasne danima.

Dvije standardne mjere po obilježju:

  PSI (Population Stability Index) — sumarno pomjeranje po korpama:
      PSI = Σ (p_i − q_i) · ln(p_i / q_i)
      < 0.10 stabilno · 0.10–0.25 umjereno · > 0.25 značajno pomjeranje

  KS (Kolmogorov–Smirnov) — najveća razlika kumulativnih raspodjela; uz
      p-vrijednost, pa se pomjeranje može i testirati.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import ks_2samp

from spr.context.features import FEATURE_NAMES

PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI nad kvantilnim korpama referentne raspodjele."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if edges.size < 2:  # konstantno obilježje
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    # Laplace-ovo zaglađivanje: prazna korpa bi dala beskonačan logaritam
    p = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    q = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))
    return float(np.sum((q - p) * np.log(q / p)))


def classify_psi(value: float) -> str:
    if value < PSI_STABLE:
        return "stabilno"
    if value < PSI_SIGNIFICANT:
        return "umjereno"
    return "značajno"


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    severity: str
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float

    def to_dict(self) -> dict:
        return asdict(self)


def feature_drift(reference: list[dict], current: list[dict],
                  features: list[str] | None = None,
                  bins: int = 10) -> list[FeatureDrift]:
    """Pomjeranje po svakom obilježju, sortirano po PSI opadajuće."""
    names = features or FEATURE_NAMES
    out: list[FeatureDrift] = []
    for name in names:
        ref = np.array([row[name] for row in reference if name in row], dtype=float)
        cur = np.array([row[name] for row in current if name in row], dtype=float)
        if ref.size == 0 or cur.size == 0:
            continue
        psi = _psi(ref, cur, bins=bins)
        ks = ks_2samp(ref, cur)
        out.append(FeatureDrift(
            feature=name, psi=psi, severity=classify_psi(psi),
            ks_statistic=float(ks.statistic), ks_pvalue=float(ks.pvalue),
            reference_mean=float(ref.mean()), current_mean=float(cur.mean()),
        ))
    out.sort(key=lambda d: d.psi, reverse=True)
    return out


def score_drift(reference_scores, current_scores, bins: int = 10) -> dict:
    """Pomjeranje raspodjele samog skora rizika — najraniji signal problema."""
    ref = np.asarray(list(reference_scores), dtype=float)
    cur = np.asarray(list(current_scores), dtype=float)
    if ref.size == 0 or cur.size == 0:
        return {"psi": 0.0, "severity": "stabilno", "reference_mean": 0.0,
                "current_mean": 0.0}
    psi = _psi(ref, cur, bins=bins)
    return {"psi": psi, "severity": classify_psi(psi),
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean())}


def drift_report(reference: list[dict], current: list[dict],
                 reference_scores=None, current_scores=None) -> dict:
    """Zbirni izvještaj: po obilježju + po skoru + ukupna ocjena."""
    per_feature = feature_drift(reference, current)
    worst = per_feature[0] if per_feature else None
    report = {
        "n_reference": len(reference),
        "n_current": len(current),
        "features": [d.to_dict() for d in per_feature],
        "n_drifting": sum(1 for d in per_feature if d.psi >= PSI_STABLE),
        "max_psi": worst.psi if worst else 0.0,
        "worst_feature": worst.feature if worst else None,
        "overall": classify_psi(worst.psi) if worst else "stabilno",
        "thresholds": {"stable": PSI_STABLE, "significant": PSI_SIGNIFICANT},
    }
    if reference_scores is not None and current_scores is not None:
        report["score"] = score_drift(reference_scores, current_scores)
    return report
