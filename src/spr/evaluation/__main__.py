"""CLI evaluacije.

    python -m spr.evaluation                    # brzo: 1 seed
    python -m spr.evaluation --seeds 10         # sa značajnošću razlika
    python -m spr.evaluation --seeds 10 --full  # + svi eksperimenti (sporo)
"""

import argparse

from spr.evaluation.report import METRIC_NAMES, run_evaluation, save_report


def _fmt(v, width=9, prec=3):
    return " " * width if v is None else f"{v:>{width}.{prec}f}"


def print_approaches(rep) -> None:
    agg = rep.get("aggregated")
    print(f"Test set: {rep['n_test']} događaja, {rep['n_fraud']} prevara "
          f"(udio {rep['fraud_rate']:.1%}), seedova: {len(rep['seeds'])}\n")
    if agg:
        print(f"{'Pristup':<20}" + "".join(f"{m.upper():>16}" for m in METRIC_NAMES))
        print("-" * (20 + 16 * len(METRIC_NAMES)))
        for name, metrics in agg.items():
            row = "".join(f"{metrics[m]['mean']:>10.3f}±{metrics[m]['std']:<5.3f}"
                          for m in METRIC_NAMES)
            print(f"{name:<20}{row}")
        print("\n(srednja vrijednost ± standardna devijacija po seedovima)")
    else:
        print(f"{'Pristup':<20}{'PR-AUC':>9}{'F1':>9}{'Prec':>9}{'Recall':>9}"
              f"{'Brier':>10}{'ECE':>9}")
        print("-" * 75)
        for name, d in rep["approaches"].items():
            m, c = d["metrics"], d["calibration"]
            print(f"{name:<20}{m['pr_auc']:>9.3f}{m['f1']:>9.3f}"
                  f"{m['precision']:>9.3f}{m['recall']:>9.3f}"
                  f"{c['brier']:>10.4f}{c['ece']:>9.4f}")


# Wilcoxonov test sa n seedova ne može dati dvostranu p-vrijednost manju od
# 2 / 2^n; ispod ovog broja nijedna razlika ne može ispasti značajna.
MIN_SEEDS_FOR_SIGNIFICANCE = 6


def print_significance(rep) -> None:
    tests = rep.get("significance") or []
    if not tests:
        return
    n = len(rep["seeds"])
    print("\nZnačajnost razlika (Wilcoxon, upareno po seedu, PR-AUC):")
    if n < MIN_SEEDS_FOR_SIGNIFICANCE:
        print(f"  UPOZORENJE: sa {n} seedova najmanja moguća p-vrijednost je "
              f"{2 / 2 ** n:.4f} — test nema snagu. Koristi --seeds "
              f"{MIN_SEEDS_FOR_SIGNIFICANCE} ili više.")
    print(f"{'poređenje':<40}{'razlika':>10}{'p':>10}   značajno")
    print("-" * 72)
    for t in tests:
        p = "—" if t.get("p_value") is None else f"{t['p_value']:.4f}"
        mark = "da" if t.get("significant") else "ne"
        print(f"{t['a'] + ' vs ' + t['b']:<40}{t['mean_diff']:>+10.3f}{p:>10}   {mark}")


def print_cost(rep) -> None:
    c = rep.get("cost")
    if not c:
        return
    print("\nTrošak politike (EUR, matrica troška iz banking pack-a):")
    for key, label in (("current", "trenutni pragovi"),
                       ("closed_form", "zatvorena forma"),
                       ("constrained", "uz kapacitet"),
                       ("empirical", "empirijski optimum")):
        d = c.get(key)
        if not d:
            continue
        print(f"  {label:<22} low={d['low_max']:.4f} med={d['medium_max']:.4f}"
              f"  ukupno={d['total_cost']:>10.1f}  po događaju={d['cost_per_event']:.3f}")
    print(f"  {'bez provjere':<22} "
          f"{'':>28}ukupno={c['allow_all_baseline']['total_cost']:>10.1f}")


def print_extras(rep) -> None:
    if "model_zoo" in rep:
        print("\nPoređenje algoritama (PR-AUC):")
        for algo, d in sorted(rep["model_zoo"].items(),
                              key=lambda kv: -kv[1]["metrics"]["pr_auc"]):
            tag = "" if d["supervised"] else "  (nenadzirani)"
            print(f"  {algo:<20}{d['metrics']['pr_auc']:>8.3f}{tag}")
    if "novel_attack" in rep:
        print("\nNov obrazac napada (izostavljen iz treninga), PR-AUC:")
        print(f"  {'obrazac':<22}{'pravila':>9}{'ml':>9}{'hyb_stack':>11}{'hyb_casc':>10}")
        for pat, d in rep["novel_attack"].items():
            print(f"  {pat:<22}{d['rules']:>9.3f}{d['ml']:>9.3f}"
                  f"{d['hybrid_stacking']:>11.3f}{d['hybrid_cascade']:>10.3f}")
    if "rule_ablation" in rep:
        print("\nAblacija pravila (pad PR-AUC skora pravila bez tog pravila):")
        for r in rep["rule_ablation"]["rules"][:5]:
            print(f"  {r['rule']:<26}{r['delta']:>+8.3f}")
    if "feature_ablation" in rep:
        print("\nAblacija obilježja (pad PR-AUC ML-a kad se obilježje neutrališe):")
        for r in rep["feature_ablation"]["features"][:5]:
            print(f"  {r['feature']:<26}{r['delta']:>+8.3f}")
    if "label_scarcity" in rep:
        print("\nOskudica labela (PR-AUC):")
        print(f"  {'udio':>7}{'n_labela':>10}{'prevara':>9}{'ml':>9}{'hibrid':>9}{'pravila':>9}")
        for r in rep["label_scarcity"]["levels"]:
            print(f"  {r['fraction']:>7.0%}{r['n_labels']:>10}{r['n_fraud']:>9}"
                  f"{_fmt(r['ml'])}{_fmt(r['hybrid_stacking'])}{_fmt(r['rules'])}")
    if "feedback_value" in rep:
        f = rep["feedback_value"]
        print("\nVrijednost analitičarskog feedbacka (PR-AUC):")
        print(f"  potpun ground truth       {_fmt(f['full_ground_truth'])}")
        print(f"  feedback analitičara      {_fmt(f['analyst_feedback'])}"
              f"   (n={f['n_reviewed']}, greška {f['analyst_error_rate']:.0%})")
        print(f"  nasumičan uzorak iste vel.{_fmt(f['random_sample'])}")
    if rep.get("meta_weights"):
        w = rep["meta_weights"]
        print(f"\nNaučene težine meta-modela: pravila={w['rules']:+.2f} "
              f"ml={w['ml']:+.2f} interakcija={w['interaction']:+.2f}")
    if "drift_scenarios" in rep:
        print("\nProvjera detektora drifta (PSI po scenariju):")
        for name, d in rep["drift_scenarios"].items():
            print(f"  {name:<22}{d['overall']:>12}  max PSI={d['max_psi']:.3f} "
                  f"na '{d['worst_feature']}'  ({d['n_drifting']} obilježja)")
    if "drift" in rep:
        d = rep["drift"]
        print(f"\nDrift trening→test: {d['overall']} "
              f"(max PSI {d['max_psi']:.3f} na '{d['worst_feature']}', "
              f"{d['n_drifting']}/{len(d['features'])} obilježja pomjereno)")


def main() -> None:
    ap = argparse.ArgumentParser(description="SPR evaluacija")
    ap.add_argument("--seeds", type=int, default=1,
                    help="broj nezavisnih seedova (>=3 za test značajnosti)")
    ap.add_argument("--full", action="store_true",
                    help="pokreni i sve eksperimente (sporo)")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--fraud-rate", type=float, default=0.02)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="reports/evaluation.json")
    args = ap.parse_args()

    rep = run_evaluation(n_train=args.n_train, n_test=args.n_test,
                         days=args.days, fraud_rate=args.fraud_rate,
                         n_seeds=args.seeds, full=args.full)
    print_approaches(rep)
    print_significance(rep)
    print_cost(rep)
    print_extras(rep)
    lat = rep["latency"]
    print(f"\nLatencija (pun engine + SHAP): mean {lat['mean_ms']:.2f} ms · "
          f"p95 {lat['p95_ms']:.2f} ms  (n={lat['n']})")
    save_report(rep, args.out)
    print(f"\nIzvještaj zapisan: {args.out}")


if __name__ == "__main__":
    main()
