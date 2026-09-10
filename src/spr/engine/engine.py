from spr.context.features import compute_features
from spr.context.store import ContextStore
from spr.domain.banking_pack import COUNTRY_CITY
from spr.domain.schema import Event
from spr.engine.decision import Decision, EventInfo, FiredRuleOut, ShapContribution
from spr.fusion.guardrails import apply_guardrails
from spr.fusion.model import FusionModel
from spr.policy.engine import classify, decide_action, execute
from spr.rules.engine import evaluate_rules


class DecisionEngine:
    """Orkestrira context -> rules -> ml -> fusion -> policy -> guardrails."""

    def __init__(self, model, rules, policy, fusion_config, guardrails,
                 store=None, top_shap: int = 5, fusion: FusionModel | None = None,
                 cost=None, capacity=None):
        self.model = model
        self.rules = rules
        self.policy = policy
        self.fusion_config = fusion_config
        self.guardrails = guardrails
        self.store = store if store is not None else ContextStore()
        self.top_shap = top_shap
        # FusionModel dijeli isti FusionConfig objekat, pa izmjene strategije
        # preko /config/fusion odmah važe i za kalibrisano kombinovanje.
        self.fusion = fusion if fusion is not None else FusionModel(fusion_config)
        # ekonomski parametri iz kojih se izvode pragovi politike
        self.cost = cost
        self.capacity = capacity

    def score(self, event: Event) -> Decision:
        state = self.store.get(event.user_id)
        feats = compute_features(event, state)
        # matična zemlja = ustanovljeni dom iz PRETHODNE istorije (None na prvom događaju)
        home_country = state.home_country
        self.store.update(event)

        rules_result = evaluate_rules(feats, self.rules)
        ml_score = self.model.predict_proba(feats)
        final = self.fusion.combine(rules_result.score, ml_score)

        level = classify(final, self.policy)
        action = decide_action(level, self.policy)

        hit = apply_guardrails(feats, self.guardrails)
        guardrail_name = None
        if hit is not None:
            action = hit.force_action
            guardrail_name = hit.name

        # u shadow režimu se predložena akcija loguje, ali ne izvršava
        executed = execute(action, self.policy)

        shap = self.model.explain(feats)
        top = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:self.top_shap]
        explanation = self._explain(level.value, rules_result.fired, top, hit)

        return Decision(
            event_id=event.event_id,
            user_id=event.user_id,
            final_score=final,
            level=level.value,
            action=action.value,
            executed_action=executed.value,
            mode=self.policy.mode,
            rules_score=rules_result.score,
            ml_score=ml_score,
            fired_rules=[FiredRuleOut(name=f.name, weight=f.weight,
                                      explanation=f.explanation)
                         for f in rules_result.fired],
            shap_top=[ShapContribution(feature=k, value=v) for k, v in top],
            model_version=self.model.version,
            guardrail=guardrail_name,
            explanation=explanation,
            features=feats,
            event=EventInfo(
                first_name=event.first_name,
                last_name=event.last_name,
                amount=event.amount,
                currency=event.currency,
                merchant_category=event.merchant_category,
                country=event.country,
                city=COUNTRY_CITY.get(event.country),
                latitude=event.latitude,
                longitude=event.longitude,
                device_id=event.device_id,
                channel=event.channel,
                timestamp=event.timestamp,
                home_country=home_country,
                distance_from_home_km=feats["distance_from_home_km"],
            ),
        )

    @staticmethod
    def _explain(level, fired, top, hit) -> str:
        parts = [f"Nivo rizika: {level}."]
        if hit is not None:
            parts.append(f"Guardrail '{hit.name}': {hit.explanation}.")
        if fired:
            parts.append("Pravila: " + "; ".join(f.explanation for f in fired) + ".")
        if top:
            fs = ", ".join(f"{k} ({v:+.2f})" for k, v in top[:3])
            parts.append(f"Ključni ML faktori: {fs}.")
        return " ".join(parts)
