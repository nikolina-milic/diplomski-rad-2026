import asyncio
from collections import Counter
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from spr.domain.schema import Event
from spr.engine.decision import Decision
from spr.engine.engine import DecisionEngine
from spr.generator.generator import stream_events
from spr.learning.drift import drift_report
from spr.learning.metrics import compute_metrics
from spr.learning.service import (
    RetrainError, register_initial_model, run_retraining_cycle,
)
from spr.persistence import repository as repo
from spr.policy.cost import (
    CapacityConstraints, CostMatrix, constrained_thresholds, optimal_thresholds,
)
from spr.rules.engine import apply_thresholds, split_condition, threshold_features


class FeedbackIn(BaseModel):
    decision_id: int
    label: int
    source: str = "analyst"


class PolicyUpdate(BaseModel):
    low_max: float | None = None
    medium_max: float | None = None
    mode: str | None = None


class CostUpdate(BaseModel):
    """Parametri matrice troška iz kojih se izvode pragovi."""
    avg_fraud_loss: float | None = None
    challenge_cost: float | None = None
    review_cost: float | None = None
    block_cost: float | None = None
    challenge_stop_rate: float | None = None
    review_stop_rate: float | None = None
    max_challenge_rate: float | None = None
    max_review_rate: float | None = None
    apply: bool = False    # da li odmah primijeniti izvedene pragove


class FusionUpdate(BaseModel):
    strategy: str | None = None
    rules_weight: float | None = None
    ml_weight: float | None = None
    cascade_threshold: float | None = None


class RuleUpdate(BaseModel):
    name: str
    weight: float | None = None
    enabled: bool | None = None
    thresholds: list[float] | None = None


def create_app(engine: DecisionEngine, session_factory=None,
               artifacts_dir: str = "artifacts") -> FastAPI:
    app = FastAPI(title="SPR — Sistem za procjenu rizika", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # Primijeni trajne override-e pravila (ako postoje) na svjež engine.
    if session_factory is not None:
        with session_factory() as s:
            saved = repo.load_rule_configs(s)
            # Početni model mora biti u registryju od prvog pokretanja —
            # inače ekran "Model registry" stoji prazan iako model radi.
            register_initial_model(s, engine, artifacts_dir)
        for r in engine.rules:
            cfg = saved.get(r.name)
            if cfg is not None:
                r.weight = cfg["weight"]
                r.enabled = cfg["enabled"]
                r.condition = cfg["condition"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": engine.model.version,
                "mode": engine.policy.mode, "persistence": session_factory is not None}

    @app.post("/score", response_model=Decision)
    def score(event: Event, record_truth: bool = False) -> Decision:
        """Skoruj jedan događaj.

        `is_fraud` iz tijela zahtjeva se NAMJERNO ignoriše: u stvarnom sistemu
        ishod nije poznat u trenutku odluke, a klijent koji bi ga slao mogao bi
        da falsifikuje evaluaciju. Ground truth se upisuje samo kad se to
        izričito traži (`record_truth=true`) — za demo i testove sa sintetičkim
        događajima. Stvarna labela inače stiže kasnije, kroz /feedback.
        """
        decision = engine.score(event)
        if session_factory is not None:
            truth = (1 if event.is_fraud else 0) if record_truth else None
            with session_factory() as s:
                repo.save_decision(s, decision, event.timestamp, truth=truth)
        return decision

    def _rules_out():
        out = []
        for r in engine.rules:
            parts, thresholds = split_condition(r.condition)
            out.append({"name": r.name, "condition": r.condition, "weight": r.weight,
                        "explanation": r.explanation,
                        "enabled": getattr(r, "enabled", True),
                        "thresholds": thresholds, "condition_parts": parts,
                        "threshold_features": threshold_features(r.condition)})
        return out

    def _fusion_out():
        f = engine.fusion_config
        return {"strategy": f.strategy, "rules_weight": f.rules_weight,
                "ml_weight": f.ml_weight, "cascade_threshold": f.cascade_threshold}

    @app.get("/rules")
    def rules():
        return _rules_out()

    @app.get("/fusion")
    def fusion():
        return _fusion_out()

    @app.get("/policy")
    def policy():
        return {"low_max": engine.policy.low_max, "medium_max": engine.policy.medium_max,
                "mode": engine.policy.mode,
                "level_actions": {k.value: v.value
                                  for k, v in engine.policy.level_actions.items()}}

    @app.put("/config/policy")
    def update_policy(body: PolicyUpdate):
        p = engine.policy
        if body.mode is not None:
            if body.mode not in ("shadow", "enforce"):
                raise HTTPException(400, "mode mora biti 'shadow' ili 'enforce'")
            p.mode = body.mode
        if body.low_max is not None:
            p.low_max = body.low_max
        if body.medium_max is not None:
            p.medium_max = body.medium_max
        if p.low_max > p.medium_max:
            raise HTTPException(400, "low_max ne smije biti veći od medium_max")
        return {"low_max": p.low_max, "medium_max": p.medium_max, "mode": p.mode}

    @app.put("/config/fusion")
    def update_fusion(body: FusionUpdate):
        f = engine.fusion_config
        if body.strategy is not None:
            if body.strategy not in ("cascade", "weighted", "stacking"):
                raise HTTPException(400, "nepoznata fusion strategija")
            f.strategy = body.strategy
        if body.rules_weight is not None:
            f.rules_weight = body.rules_weight
        if body.ml_weight is not None:
            f.ml_weight = body.ml_weight
        if body.cascade_threshold is not None:
            f.cascade_threshold = body.cascade_threshold
        return _fusion_out()

    @app.put("/config/rules")
    def update_rules(body: list[RuleUpdate]):
        by_name = {r.name: r for r in engine.rules}
        for u in body:
            r = by_name.get(u.name)
            if r is None:
                continue
            if u.weight is not None:
                r.weight = max(0.0, min(1.0, u.weight))
            if u.enabled is not None:
                r.enabled = u.enabled
            if u.thresholds is not None:
                try:
                    r.condition = apply_thresholds(r.condition, u.thresholds)
                except ValueError as e:
                    raise HTTPException(400, f"pravilo '{u.name}': {e}")
        if session_factory is not None:
            with session_factory() as s:
                repo.upsert_rule_configs(s, engine.rules)
        return _rules_out()

    @app.get("/cost")
    def cost():
        """Matrica troška, kapacitet i pragovi koji iz njih slijede."""
        c = engine.cost or CostMatrix()
        cap = engine.capacity or CapacityConstraints()
        lo, hi = optimal_thresholds(c)
        return {
            "cost_matrix": c.to_dict(),
            "capacity": cap.to_dict(),
            "closed_form": {"low_max": lo, "medium_max": hi},
            "current": {"low_max": engine.policy.low_max,
                        "medium_max": engine.policy.medium_max},
        }

    @app.put("/config/cost")
    def update_cost(body: CostUpdate):
        """Promijeni ekonomske parametre; opciono odmah izvedi nove pragove.

        Pragovi se biraju nad logovanim odlukama (stvarna raspodjela skorova),
        uz poštovanje kapaciteta — bez toga optimizacija daje ekonomski tačan,
        ali operativno neupotrebljiv odgovor (izazov na skoro svaku transakciju).
        """
        c = engine.cost or CostMatrix()
        cap = engine.capacity or CapacityConstraints()
        for field in ("avg_fraud_loss", "challenge_cost", "review_cost",
                      "block_cost", "challenge_stop_rate", "review_stop_rate"):
            value = getattr(body, field)
            if value is not None:
                setattr(c, field, value)
        for field in ("max_challenge_rate", "max_review_rate"):
            value = getattr(body, field)
            if value is not None:
                setattr(cap, field, value)
        engine.cost, engine.capacity = c, cap

        applied = None
        if body.apply and session_factory is not None:
            with session_factory() as s:
                decs = [d for d in repo.list_decisions(s, 5000)
                        if d.truth is not None]
            if len(decs) >= 50:
                y = [int(d.truth) for d in decs]
                scores = [d.final_score for d in decs]
                applied = constrained_thresholds(y, scores, c, cap)
                engine.policy.low_max = applied["low_max"]
                engine.policy.medium_max = applied["medium_max"]
        return {"cost_matrix": c.to_dict(), "capacity": cap.to_dict(),
                "applied": applied,
                "current": {"low_max": engine.policy.low_max,
                            "medium_max": engine.policy.medium_max}}

    @app.get("/calibration")
    def calibration():
        """Kvalitet kalibracije živog fusion sloja + naučene težine meta-modela."""
        return {
            "fitted": engine.fusion.fitted,
            "method": engine.fusion.calibration,
            "strategy": engine.fusion_config.strategy,
            "meta_weights": engine.fusion.weights(),
        }

    @app.get("/evaluation")
    def evaluation(n_test: int = 800):
        from spr.evaluation.report import run_evaluation
        return run_evaluation(model=engine.model, rules=engine.rules,
                              fusion_cfg=engine.fusion_config, policy=engine.policy,
                              guardrails=engine.guardrails, n_test=n_test)

    # Udio prevara u DEMO toku. Namjerno je viši od trening udjela (2%) da bi
    # dashboard imao šta da pokaže — na 2% bi rizičan slučaj stizao jednom u
    # nekoliko minuta. Posljedica: /metrics nad live tokom precjenjuje bazu
    # prevara, pa se za mjerenje koristi /evaluation na nezavisnom test setu.
    DEMO_STREAM_FRAUD_RATE = 0.05

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket, interval: float = 5.0):
        await ws.accept()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        gen = stream_events(n_users=30, fraud_rate=DEMO_STREAM_FRAUD_RATE, seed=7,
                            start=start, count=10_000_000)
        try:
            for event in gen:
                decision = await asyncio.to_thread(engine.score, event)
                if session_factory is not None:
                    def _log(dec=decision, ev=event):
                        with session_factory() as s:
                            repo.save_decision(s, dec, ev.timestamp,
                                               truth=1 if ev.is_fraud else 0)
                    await asyncio.to_thread(_log)
                await ws.send_text(decision.model_dump_json())
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            return

    if session_factory is not None:
        @app.get("/decisions")
        def decisions(limit: int = 50, level: str | None = None):
            """Posljednje odluke, opciono filtrirane po nivou rizika.

            full=True: red iz istorije nosi isto objašnjenje kao red iz živog
            toka, pa se u dashboardu može otvoriti bez dodatnog zahtjeva.
            """
            if level is not None and level not in ("LOW", "MEDIUM", "HIGH"):
                raise HTTPException(400, "level mora biti LOW, MEDIUM ili HIGH")
            with session_factory() as s:
                return [_dec_dict(d, full=True)
                        for d in repo.list_decisions(s, limit, level)]

        @app.get("/decisions/count")
        def decisions_count():
            """Ukupan broj logovanih odluka i raspodjela po nivou rizika."""
            with session_factory() as s:
                return repo.count_decisions(s)

        @app.get("/review-queue")
        def review_queue():
            with session_factory() as s:
                return [_dec_dict(d, full=True) for d in repo.get_review_queue(s)]

        @app.post("/feedback")
        def feedback(body: FeedbackIn):
            with session_factory() as s:
                fb = repo.add_feedback(s, body.decision_id, body.label, body.source)
                return {"id": fb.id, "decision_id": fb.decision_id, "label": fb.label}

        @app.get("/models")
        def models():
            with session_factory() as s:
                return [{"version": m.version, "status": m.status,
                         "metrics": m.metrics} for m in repo.list_model_versions(s)]

        @app.post("/retrain")
        def retrain(version: str | None = None):
            """Retrening + poređenje sa championom. Oznaku verzije bira backend
            ako nije zadata, pa se postojeći artefakt ne može pregaziti."""
            with session_factory() as s:
                try:
                    return run_retraining_cycle(s, engine, artifacts_dir, version,
                                                n_normal=400, days=7)
                except RetrainError as e:
                    raise HTTPException(e.status, str(e))

        @app.get("/drift")
        def drift(window: int = 500):
            """Pomjeranje raspodjele: posljednjih `window` odluka u odnosu na
            stariji period iz audit loga."""
            with session_factory() as s:
                decs = repo.list_decisions(s, window * 2)
            rows = [d.features for d in decs if d.features]
            if len(rows) < 40:
                return {"available": False,
                        "reason": "premalo logovanih odluka za poređenje",
                        "n": len(rows)}
            half = len(rows) // 2
            current, reference = rows[:half], rows[half:]   # lista je najnovije-prvo
            scores_cur = [d.final_score for d in decs[:half]]
            scores_ref = [d.final_score for d in decs[half:half * 2]]
            out = drift_report(reference, current, scores_ref, scores_cur)
            out["available"] = True
            return out

        @app.get("/stats")
        def stats(window: int = 200):
            with session_factory() as s:
                decs = repo.list_decisions(s, window)
            return {"total": len(decs),
                    "actions": dict(Counter(d.action for d in decs)),
                    "levels": dict(Counter(d.level for d in decs))}

        @app.get("/metrics")
        def metrics():
            with session_factory() as s:
                decs = [d for d in repo.list_decisions(s, 100000) if d.truth is not None]
                versions = [{"version": m.version, "status": m.status,
                             "metrics": m.metrics}
                            for m in repo.list_model_versions(s)]
            tp = fp = tn = fn = 0
            scores, truths = [], []
            for d in decs:
                flagged = d.action in ("BLOCK", "REVIEW")
                t = bool(d.truth)
                scores.append(d.final_score)
                truths.append(int(d.truth))
                if flagged and t:
                    tp += 1
                elif flagged and not t:
                    fp += 1
                elif not flagged and t:
                    fn += 1
                else:
                    tn += 1
            live = compute_metrics(truths, scores) if len(set(truths)) > 1 else {}
            return {"confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                    "live_metrics": live, "versions": versions, "n": len(decs)}

    return app


def _dec_dict(d, full: bool = False) -> dict:
    out = {"id": d.id, "event_id": d.event_id, "user_id": d.user_id,
           "first_name": d.first_name, "last_name": d.last_name,
           "final_score": d.final_score, "level": d.level, "action": d.action,
           "executed_action": d.executed_action, "mode": d.mode,
           # Skorovi po izvoru se čuvaju u bazi i moraju se vratiti: bez njih
           # odluka iz istorije izgleda nepotpuno u odnosu na istu takvu iz
           # živog toka, iako podaci postoje.
           "rules_score": d.rules_score, "ml_score": d.ml_score,
           "guardrail": d.guardrail, "explanation": d.explanation,
           "model_version": d.model_version, "event": d.event_info or None}
    if full:
        out["fired_rules"] = d.fired_rules
        out["shap_top"] = d.shap_top
        out["features"] = d.features
    return out
