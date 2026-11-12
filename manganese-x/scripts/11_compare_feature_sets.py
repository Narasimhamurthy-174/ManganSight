"""Compare all finished (non-quick) cross-validation runs in one table.

Run (project root):  python scripts/11_compare_feature_sets.py
Reads outputs/metrics/baseline_cv_summary_*.csv. Shows mean belt-AUC per block size. Differences below
about 0.05 are within the fold-to-fold spread (~0.06-0.11) and should be treated as noise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import resolve


def parse(tag: str):
    if tag in ("all", "occurrence_only"):          # runs made by the first version of script 09
        tag = f"base_{tag}"
    for pos in ("occurrence_only", "all"):
        if tag.endswith("_" + pos):
            return tag[: -len(pos) - 1], pos
    return None


def main() -> None:
    frames = []
    for f in sorted(resolve("outputs/metrics").glob("baseline_cv_summary_*.csv")):
        tag = f.stem.replace("baseline_cv_summary_", "")
        if tag.startswith("quick_") or parse(tag) is None:
            continue
        fset, pos = parse(tag)
        d = pd.read_csv(f)
        d["feature_set"], d["positives"] = fset, pos
        frames.append(d)
    if not frames:
        raise SystemExit("No finished runs found. Run scripts/09_baseline_cv.py first.")
    all_ = pd.concat(frames)
    for pos, d in all_.groupby("positives"):
        d = d.copy()
        d["block"] = d["block_km"].map(lambda b: "random" if b < 1 else f"{b:g}km")
        order = ["random"] + [f"{b:g}km" for b in sorted(d.block_km.unique()) if b >= 1]
        loc = d[d.model == "knn_dist"].drop_duplicates(["block_km"]).assign(feature_set="(location only)", strategy="-")
        feat = d[d.model != "knn_dist"]
        tab = pd.concat([loc, feat]).pivot_table(index=["feature_set", "model", "strategy"], columns="block",
                                                values="auc_belt_mean")[order]
        tab["mean_10_30km"] = tab[[c for c in order if c in ("10km", "20km", "30km")]].mean(axis=1)
        print(f"\n=== Belt AUC (mean), positives = {pos} ===")
        print(tab.round(2).to_string())


if __name__ == "__main__":
    main()
