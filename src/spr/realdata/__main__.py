"""CLI za stvarnu IEEE-CIS evaluaciju.

Primjer:
    python -m spr.realdata --data-dir data/ieee-cis --seeds 5
"""

import argparse

from spr.realdata.ieee_cis import run_ieee_cis_evaluation, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="SPR evaluacija na IEEE-CIS skupu")
    parser.add_argument("--data-dir", default="data/ieee-cis")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--oof-splits", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="samo za brzu tehničku provjeru; ne koristiti u radu")
    parser.add_argument("--out", default="reports/ieee_cis_evaluation.json")
    args = parser.parse_args()

    report = run_ieee_cis_evaluation(
        args.data_dir,
        seeds=args.seeds,
        test_fraction=args.test_fraction,
        n_estimators=args.n_estimators,
        oof_splits=args.oof_splits,
        max_rows=args.max_rows,
    )
    save_report(report, args.out)

    ds = report["dataset"]
    split = report["split"]
    print(f"IEEE-CIS: {ds['n_rows']} događaja, {ds['n_fraud']} prevara "
          f"({ds['fraud_rate']:.2%})")
    print(f"Vremenska podjela: {split['n_train']} trening / {split['n_test']} test")
    print()
    print(f"{'Pristup':<22}{'PR-AUC':>16}{'F1':>16}{'Preciznost':>16}{'Odziv':>16}")
    for name, metrics in report["aggregated"].items():
        values = []
        for metric in ("pr_auc", "f1", "precision", "recall"):
            value = metrics[metric]
            values.append(f"{value['mean']:.3f}±{value['std']:.3f}")
        print(f"{name:<22}" + "".join(f"{value:>16}" for value in values))
    print(f"\nIzvještaj zapisan: {args.out}")


if __name__ == "__main__":
    main()

