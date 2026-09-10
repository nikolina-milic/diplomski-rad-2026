from datetime import datetime, timezone
from spr.persistence.db import make_session_factory
from spr.persistence import repository as repo
from spr.engine.factory import build_default_engine
from spr.domain.schema import Event

START = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _sf():
    return make_session_factory("sqlite://")


def _event(**kw):
    d = dict(event_id="e1", user_id="u1", timestamp=START, amount=100.0,
             currency="EUR", merchant_category="grocery", country="RS",
             latitude=44.82, longitude=20.46, device_id="d1", channel="online")
    d.update(kw)
    return Event(**d)


def test_save_and_list_decision():
    Session = _sf()
    eng = build_default_engine(n_normal=300, days=7)
    with Session() as s:
        dec = eng.score(_event())
        rec = repo.save_decision(s, dec, START, truth=0)
        assert rec.id is not None
        assert len(repo.list_decisions(s)) == 1


def test_save_decision_persists_name():
    Session = _sf()
    eng = build_default_engine(n_normal=300, days=7)
    with Session() as s:
        dec = eng.score(_event(first_name="Marko", last_name="Jovanović"))
        rec = repo.save_decision(s, dec, START, truth=0)
        assert rec.first_name == "Marko"
        assert rec.last_name == "Jovanović"
        [loaded] = repo.list_decisions(s)
        assert loaded.first_name == "Marko"
        assert loaded.last_name == "Jovanović"


def test_feedback_and_collect_training_data():
    Session = _sf()
    eng = build_default_engine(n_normal=300, days=7)
    with Session() as s:
        dec = eng.score(_event(amount=20000.0))
        rec = repo.save_decision(s, dec, START, truth=1)
        fb = repo.add_feedback(s, rec.id, label=1, source="analyst")
        assert fb.id is not None
        X, y = repo.collect_training_data(s)
        assert len(X) == len(y) == 1
        assert y[0] == 1  # feedback nadjačava truth


def test_model_version_registry():
    Session = _sf()
    with Session() as s:
        repo.save_model_version(s, "v1", "champion", {"pr_auc": 0.8})
        assert repo.get_champion(s).version == "v1"
        repo.retire_champions(s)
        assert repo.get_champion(s) is None
        assert len(repo.list_model_versions(s)) == 1
