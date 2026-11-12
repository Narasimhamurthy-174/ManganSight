"""Phase 7/9: baseline models under random vs spatial cross-validation.

Run (project root):
    python scripts/09_baseline_cv.py --quick                          # ~1-2 min smoke test
    python scripts/09_baseline_cv.py                                  # full run, base features
    python scripts/09_baseline_cv.py --feature-set scale_1km          # other sets: see config/feature_sets.yaml
    python scripts/09_baseline_cv.py --positives occurrence_only      # drop producer / past-producer sites

Needs: features_<grid_version>.csv, positives_cells.csv, config/features.yaml, cv.yaml, feature_sets.yaml
       (+ features_<grid_version>_context.csv for non-base sets: run scripts/10_build_context_features.py)
Outputs: outputs/metrics/baseline_cv_<tag>.csv, baseline_cv_summary_<tag>.csv, outputs/figures/baseline_cv_<tag>.png
         where <tag> = [quick_]<feature_set>_<positives>
No geology yet.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.exploration.cv import plot_results, run_cv, summarise, summarise_importance

log = get_logger("baseline_cv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 repeats, block sizes random and 20 km only")
    ap.add_argument("--positives", choices=["all", "occurrence_only"], default="all")
    ap.add_argument("--feature-set", default="base")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these feature models, e.g. --models logreg rf (location baseline always runs)")
    ap.add_argument("--repeats", type=int, default=None, help="override the number of repeats in config/cv.yaml")
    args = ap.parse_args()

    cfg_main = load_config()
    gv = cfg_main["grid"]["version"]
    cv = yaml.safe_load(resolve("config/cv.yaml").read_text(encoding="utf-8"))
    fs = yaml.safe_load(resolve("config/features.yaml").read_text(encoding="utf-8"))
    sets = yaml.safe_load(resolve("config/feature_sets.yaml").read_text(encoding="utf-8"))["sets"]
    if args.feature_set not in sets:
        raise SystemExit(f"Unknown feature set {args.feature_set!r}. Choose from {list(sets)}")
    predictors = fs["predictors"] if args.feature_set == "base" else sets[args.feature_set]
    if args.quick:
        cv.update(repeats=2, block_km=[0.25, 20])
    if args.models:
        cv["models"] = ["knn_dist"] + [m for m in args.models if m != "knn_dist"]
    if args.repeats:
        cv["repeats"] = args.repeats
    tag = ("quick_" if args.quick else "") + f"{args.feature_set}_{args.positives}"

    df = pd.read_csv(resolve(f"data/processed/features_{gv}.csv"))
    if args.feature_set != "base":
        ctx_path = resolve(f"data/processed/features_{gv}_context.csv")
        if not ctx_path.exists():
            raise SystemExit("Run scripts/10_build_context_features.py first.")
        df = df.merge(pd.read_csv(ctx_path), on="cell_id", how="left", suffixes=("", "_ctx"))
    pos = pd.read_csv(resolve("data/processed/positives_cells.csv"))
    if args.positives == "occurrence_only":
        mines = pos[pos["dev_stat"].isin(["Producer", "Past Producer"])]
        df = df[~df.cell_id.isin(mines.cell_id)]
        pos = pos[~pos.cell_id.isin(mines.cell_id)]
        log.info("Excluded %d producer / past-producer cells", len(mines))
    df["is_pos"] = df.cell_id.isin(pos.cell_id)
    before, n_pos_before = len(df), int(df.is_pos.sum())
    df = df.dropna(subset=predictors).reset_index(drop=True)
    log.info("Feature set %s: %s | cells %d -> %d after dropping NaN | positives %d -> %d",
             args.feature_set, predictors, before, len(df), n_pos_before, int(df.is_pos.sum()))

    res = run_cv(df, predictors, cv)
    res.to_csv(resolve(f"outputs/metrics/baseline_cv_{tag}.csv"), index=False)
    summ = summarise(res)
    summ.round(3).to_csv(resolve(f"outputs/metrics/baseline_cv_summary_{tag}.csv"), index=False)
    plot_results(summ, resolve(f"outputs/figures/baseline_cv_{tag}.png"))

    show = summ[["block_km", "strategy", "model", "n_units", "auc_district_mean", "auc_belt_mean",
                 "auc_belt_sd", "lift_belt_mean", "cap10_belt_mean"]].round(2)
    print(f"\nBaseline skill, feature set '{args.feature_set}', positives '{args.positives}'.")
    print("auc_belt = ranking inside the 15 km belt around known records. block_km 0.25 = random CV.")
    print("Chance level: AUC 0.5, cap10 0.10, lift 1.\n")
    print(show.to_string(index=False))
    imp = res.attrs.get("importance")
    if imp is not None and len(imp):
        imp_sum = summarise_importance(imp)
        imp_sum.round(4).to_csv(resolve(f"outputs/metrics/feature_importance_{tag}.csv"), index=False)
        print(f"\nPermutation importance on held-out {cv['importance_block_km']:g} km blocks: mean DROP in belt-AUC when a feature is shuffled.")
        print("(bigger = the model relies on it more; ~0 = not used; spread across folds is large, read the ranking only)\n")
        print(imp_sum.round(3).to_string(index=False))
    log.info("Saved results with tag '%s'", tag)


if __name__ == "__main__":
    main()
