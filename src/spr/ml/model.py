"""ML skorer rizika.

Podržava više algoritama da bi izbor produkcionog modela bio *opravdan*
poređenjem, a ne pretpostavkom:

  random_forest    — podrazumijevani; nelinearan, radi sa SHAP-om
  logistic         — linearni baseline (referentna tačka za sve ostalo)
  hist_gb          — gradient boosting (jači nelinearni takmac)
  isolation_forest — nenadzirani detektor anomalija (radi bez labela)

Uz to nudi out-of-fold skorove, bez kojih se fusion sloj ne može ispravno
kalibrisati: skorovi baznog modela na sopstvenom trening skupu su
preoptimistični.
"""

import joblib
import numpy as np
import shap
from sklearn.ensemble import (
    HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spr.context.features import FEATURE_NAMES
from spr.ml.dataset import to_matrix

SUPERVISED_ALGORITHMS = ("random_forest", "logistic", "hist_gb")
ALGORITHMS = SUPERVISED_ALGORITHMS + ("isolation_forest",)


def _build_estimator(algorithm: str, random_state: int, n_estimators: int):
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
            class_weight="balanced")
    if algorithm == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
    if algorithm == "hist_gb":
        return HistGradientBoostingClassifier(
            random_state=random_state, class_weight="balanced")
    if algorithm == "isolation_forest":
        return IsolationForest(n_estimators=n_estimators,
                               random_state=random_state, contamination="auto")
    raise ValueError(f"Nepoznat algoritam: {algorithm}")


class RiskModel:
    """ML skorer rizika + SHAP objašnjenje.

    `class_weight="balanced"` ublažava neuravnoteženost (prevara je rijetka).
    """

    def __init__(self, version: str = "v1", n_estimators: int = 100,
                 random_state: int = 0, algorithm: str = "random_forest") -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Nepoznat algoritam: {algorithm}")
        self.version = version
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_estimators = n_estimators
        self._clf = _build_estimator(algorithm, random_state, n_estimators)
        self._explainer: shap.TreeExplainer | None = None
        # opseg anomaly skora za normalizaciju kod nenadziranog modela
        self._anomaly_lo: float = -0.5
        self._anomaly_hi: float = 0.5

    @property
    def is_unsupervised(self) -> bool:
        return self.algorithm == "isolation_forest"

    # --- treniranje -----------------------------------------------------
    def train(self, X: list[dict], y: list[int]) -> None:
        mat = to_matrix(X)
        if self.is_unsupervised:
            # uči samo iz legitimnog saobraćaja: anomalija = odstupanje od njega
            legit = mat[np.asarray(y) == 0] if len(y) else mat
            self._clf.fit(legit if len(legit) else mat)
            raw = self._clf.score_samples(mat)
            self._anomaly_lo, self._anomaly_hi = float(raw.min()), float(raw.max())
            return
        self._clf.fit(mat, y)
        self._explainer = None  # lijeno; TreeExplainer ne radi za sve algoritme

    def oof_scores(self, X: list[dict], y: list[int], n_splits: int = 5) -> list[float]:
        """Out-of-fold skorovi uz vremenski (nerandomizovan) split.

        Podaci su vremenski niz, pa se koristi TimeSeriesSplit: model uvijek
        predviđa unaprijed, nikad unazad. Redovi prije prvog fold-a nemaju
        out-of-fold predikciju i dobijaju skor bazne stope.
        """
        mat = to_matrix(X)
        y_arr = np.asarray(y, dtype=int)
        base_rate = float(y_arr.mean()) if len(y_arr) else 0.0
        out = np.full(len(X), base_rate, dtype=float)
        n_splits = max(2, min(n_splits, max(2, len(X) // 50)))
        for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(mat):
            fold = RiskModel(version=self.version, n_estimators=self.n_estimators,
                             random_state=self.random_state,
                             algorithm=self.algorithm)
            fold.train([X[i] for i in train_idx], y_arr[train_idx].tolist())
            out[test_idx] = [fold.predict_proba(X[i]) for i in test_idx]
        return [float(v) for v in out]

    # --- predikcija -----------------------------------------------------
    def predict_proba(self, features: dict) -> float:
        return float(self.predict_proba_batch([features])[0])

    def predict_proba_batch(self, rows: list[dict]) -> list[float]:
        mat = to_matrix(rows)
        if self.is_unsupervised:
            raw = self._clf.score_samples(mat)
            span = max(self._anomaly_hi - self._anomaly_lo, 1e-9)
            # niži score_samples = veća anomalija => invertuj u [0, 1]
            norm = 1.0 - (raw - self._anomaly_lo) / span
            return [float(v) for v in np.clip(norm, 0.0, 1.0)]
        proba = self._clf.predict_proba(mat)
        classes = list(getattr(self._clf, "classes_", [0, 1]))
        if 1 not in classes:
            # fold/skup bez ijedne prevare — model ne poznaje pozitivnu klasu
            return [0.0] * len(rows)
        return [float(v) for v in proba[:, classes.index(1)]]

    # --- objašnjenje ----------------------------------------------------
    def explain(self, features: dict) -> dict[str, float]:
        mat = to_matrix([features])
        if self.algorithm in ("random_forest", "hist_gb"):
            try:
                return self._explain_tree(mat)
            except Exception:
                pass
        if self.algorithm == "logistic":
            return self._explain_linear(mat)
        return {name: 0.0 for name in FEATURE_NAMES}

    def _explain_tree(self, mat) -> dict[str, float]:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._clf)
        sv = np.array(self._explainer.shap_values(mat))
        # binarna klasifikacija: oblik (1, n_features, 2) -> pozitivna klasa
        contrib = sv[0, :, 1] if sv.ndim == 3 else sv[0]
        return {name: float(v) for name, v in zip(FEATURE_NAMES, contrib)}

    def _explain_linear(self, mat) -> dict[str, float]:
        """Doprinos = koeficijent × standardizovana vrijednost obilježja."""
        scaler: StandardScaler = self._clf.named_steps["scaler"]
        lr: LogisticRegression = self._clf.named_steps["lr"]
        z = scaler.transform(mat)[0]
        contrib = lr.coef_[0] * z
        return {name: float(v) for name, v in zip(FEATURE_NAMES, contrib)}

    # --- perzistencija --------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump({"clf": self._clf, "version": self.version,
                     "algorithm": self.algorithm,
                     "anomaly_lo": self._anomaly_lo,
                     "anomaly_hi": self._anomaly_hi}, path)

    @classmethod
    def load(cls, path: str) -> "RiskModel":
        data = joblib.load(path)
        m = cls(version=data["version"],
                algorithm=data.get("algorithm", "random_forest"))
        m._clf = data["clf"]
        m._anomaly_lo = data.get("anomaly_lo", -0.5)
        m._anomaly_hi = data.get("anomaly_hi", 0.5)
        return m
