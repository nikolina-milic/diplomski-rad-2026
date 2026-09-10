import os
from datetime import datetime, timezone

from spr.engine.factory import fit_fusion
from spr.generator.generator import generate_batch
from spr.learning.metrics import compute_metrics
from spr.ml.dataset import build_dataset
from spr.ml.model import RiskModel
from spr.persistence import repository as repo


# Najmanji broj labeliranih primjera (i prevara) da retrening uopšte ima smisla.
MIN_TRAINING_ROWS = 50
MIN_TRAINING_FRAUD = 3


class RetrainError(Exception):
    """Retrening se ne može izvesti — sa razlogom koji korisnik treba da vidi."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def retrain_challenger(session, version: str, artifacts_dir: str):
    """Istreniraj novi model na labelama iz audit loga (feedback + ground truth).

    Vraća (model, putanja, X, y) — X i y su potrebni i pozivaocu, da bi se
    fusion sloj mogao ponovo kalibrisati nad istim podacima.
    """
    X, y = repo.collect_training_data(session)
    if len(X) < MIN_TRAINING_ROWS:
        raise RetrainError(
            f"Nema dovoljno labeliranih odluka za retrening: {len(X)} "
            f"(potrebno najmanje {MIN_TRAINING_ROWS}). Pusti live tok ili "
            f"unesi feedback u review queue.")
    if sum(y) < MIN_TRAINING_FRAUD:
        raise RetrainError(
            f"Nema dovoljno potvrđenih prevara: {sum(y)} "
            f"(potrebno najmanje {MIN_TRAINING_FRAUD}).")
    model = RiskModel(version=version, random_state=0)
    model.train(X, y)
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, f"model_{version}.joblib")
    model.save(path)
    return model, path, X, y


def run_retraining_cycle(session, engine, artifacts_dir: str,
                         new_version: str | None = None,
                         test_seed: int = 999, n_normal: int = 500,
                         days: int = 7) -> dict:
    """Istreniraj challenger, uporedi sa championom i promoviši ako je bolji.

    `new_version=None` znači da oznaku bira backend — tako se ne može pregaziti
    postojeća verzija ni njen artefakt.
    """
    if new_version is None:
        new_version = repo.next_model_version(session)
    elif repo.version_exists(session, new_version):
        raise RetrainError(
            f"Verzija '{new_version}' već postoji u registryju", status=409)
    challenger, path, X_train, y_train = retrain_challenger(
        session, new_version, artifacts_dir)

    # svjež labelirani test set (nezavisan od trening podataka)
    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    test_events = generate_batch(n_users=20, n_normal=n_normal, fraud_rate=0.02,
                                 seed=test_seed, start=start, days=days)
    X_test, y_test = build_dataset(test_events)

    champ = engine.model
    champ_scores = [champ.predict_proba(x) for x in X_test]
    chal_scores = [challenger.predict_proba(x) for x in X_test]
    champ_metrics = compute_metrics(y_test, champ_scores)
    chal_metrics = compute_metrics(y_test, chal_scores)

    promoted = chal_metrics["pr_auc"] > champ_metrics["pr_auc"]
    if promoted:
        repo.retire_champions(session)
        repo.save_model_version(session, new_version, "champion", chal_metrics, path)
        engine.model = challenger  # live promocija
        # Kalibracija i meta-model su naučeni nad skorovima STAROG modela. Bez
        # ponovnog fitovanja fusion sloj bi kalibrisao skorove koje više niko
        # ne proizvodi, pa bi kombinovana vjerovatnoća bila pogrešna.
        if len(set(y_train)) > 1:
            engine.fusion = fit_fusion(challenger, engine.rules,
                                       engine.fusion_config,
                                       X_train, y_train)[0]
    else:
        repo.save_model_version(session, new_version, "challenger", chal_metrics, path)

    return {
        "promoted": promoted,
        "champion_version": champ.version,
        "challenger_version": new_version,
        "champion_metrics": champ_metrics,
        "challenger_metrics": chal_metrics,
    }


def register_initial_model(session, engine, artifacts_dir: str,
                           n_eval: int = 800, seed: int = 4242) -> dict | None:
    """Upiši početni model u registry kao champion (ako registry je prazan).

    Bez ovoga je ekran „Model registry" prazan sve dok se ne pokrene prvi
    retrening, iako model postoji i donosi odluke od starta.
    """
    if repo.list_model_versions(session):
        return None
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    events = generate_batch(n_users=20, n_normal=n_eval, fraud_rate=0.02,
                            seed=seed, start=start, days=14)
    X, y = build_dataset(events)
    scores = engine.model.predict_proba_batch(X)
    metrics = compute_metrics(y, scores) if len(set(y)) > 1 else {}
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, f"model_{engine.model.version}.joblib")
    engine.model.save(path)
    repo.save_model_version(session, engine.model.version, "champion",
                            metrics, path)
    return metrics
