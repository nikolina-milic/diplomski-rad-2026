"""Fusion sloj sa kalibracijom i naučenim meta-modelom (stacking).

Zašto ovaj sloj postoji:

`rules_score` (noisy-OR nad težinama pravila) i `ml_score` (vjerovatnoća iz
klasifikatora) nisu na istoj skali, pa ih fiksni ponder 50:50 kombinuje
neopravdano — dobar signal se razvodnjava lošijim. Ovdje se oba skora najprije
kalibrišu u vjerovatnoće, a zatim kombinuju:

  * `weighted`  — ponderisana sredina *kalibrisanih* skorova,
  * `cascade`   — pravila imaju prednost iznad praga,
  * `stacking`  — logistička regresija koja iz podataka nauči koliko kom
                  izvoru vjerovati, uključujući njihovu interakciju.

Meta-model namjerno vidi samo [rules_cal, ml_cal, rules_cal * ml_cal] — dakle
kombinuje dva bazna signala, a ne sirova obilježja. Time ostaje *fusion* sloj,
a ne još jedan klasifikator.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from spr.fusion.strategies import FusionConfig, fuse
from spr.ml.calibration import Calibrator


def _meta_matrix(rules_cal, ml_cal) -> np.ndarray:
    r = np.asarray(rules_cal, dtype=float).ravel()
    m = np.asarray(ml_cal, dtype=float).ravel()
    return np.column_stack([r, m, r * m])


class FusionModel:
    """Kalibracija + kombinovanje skorova pravila i ML-a.

    Dok nije `fit`-ovan, ponaša se kao obična `fuse` funkcija, pa engine radi
    i bez trening podataka (npr. u testovima i pri cold startu).
    """

    def __init__(self, cfg: FusionConfig, calibration: str = "isotonic") -> None:
        self.cfg = cfg
        self.calibration = calibration
        self.rules_cal = Calibrator(calibration)
        self.ml_cal = Calibrator(calibration)
        self.meta: LogisticRegression | None = None
        self.fitted = False

    # --- treniranje -----------------------------------------------------
    def fit(self, rules_scores, ml_scores, y) -> "FusionModel":
        """Nauči kalibratore i meta-model.

        VAŽNO: `ml_scores` moraju biti out-of-fold predikcije (vidi
        `RiskModel.oof_scores`). Skorovi baznog modela na sopstvenom trening
        skupu su preoptimistični, pa bi kalibracija na njima bila pogrešna.
        """
        y = np.asarray(y, dtype=int)
        self.rules_cal.fit(rules_scores, y)
        self.ml_cal.fit(ml_scores, y)
        if len(set(y.tolist())) < 2:
            return self
        rc = self.rules_cal.transform(rules_scores)
        mc = self.ml_cal.transform(ml_scores)
        # bez class_weight: bazni skorovi već nose neuravnoteženost, a meta-model
        # treba da vrati KALIBRISANU vjerovatnoću (balansiranje pomjera intercept)
        self.meta = LogisticRegression(max_iter=1000)
        self.meta.fit(_meta_matrix(rc, mc), y)
        self.fitted = True
        return self

    # --- primjena -------------------------------------------------------
    def calibrated(self, rules_score: float, ml_score: float) -> tuple[float, float]:
        return (self.rules_cal.transform_one(rules_score),
                self.ml_cal.transform_one(ml_score))

    def combine(self, rules_score: float, ml_score: float) -> float:
        if not self.fitted:
            return fuse(rules_score, ml_score, self.cfg)
        rc, mc = self.calibrated(rules_score, ml_score)
        if self.cfg.strategy == "stacking":
            p = float(self.meta.predict_proba(_meta_matrix([rc], [mc]))[0, 1])
            return max(0.0, min(1.0, p))
        # weighted i cascade rade nad kalibrisanim skorovima
        return fuse(rc, mc, self.cfg)

    def combine_batch(self, rules_scores, ml_scores) -> list[float]:
        if not self.fitted:
            return [fuse(r, m, self.cfg)
                    for r, m in zip(rules_scores, ml_scores)]
        rc = self.rules_cal.transform(rules_scores)
        mc = self.ml_cal.transform(ml_scores)
        if self.cfg.strategy == "stacking":
            p = self.meta.predict_proba(_meta_matrix(rc, mc))[:, 1]
            return [float(v) for v in np.clip(p, 0.0, 1.0)]
        return [fuse(float(r), float(m), self.cfg) for r, m in zip(rc, mc)]

    def weights(self) -> dict | None:
        """Naučeni koeficijenti meta-modela — za tumačenje u radu."""
        if not self.fitted or self.meta is None:
            return None
        coef = self.meta.coef_[0]
        return {"rules": float(coef[0]), "ml": float(coef[1]),
                "interaction": float(coef[2]),
                "intercept": float(self.meta.intercept_[0])}
