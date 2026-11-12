"""Phase 10 check: does the ensemble and its confidence tier carry information on UNSEEN areas?

Run (project root):  python scripts/13_validate_uncertainty.py        # about 5-10 min
                     python scripts/13_validate_uncertainty.py --quick
For each held-out spatial block set: train the ensemble on the other blocks, score the held-out cells, assign the same
tier rules as script 12 (all thresholds fixed in config/uncertainty.yaml, NOT tuned here), then measure ranking skill
(belt AUC) overall, per member, and inside each tier. Reading guide:
  * ENSEMBLE vs the single members: does averaging help?
  * Medium vs Low tier: a useful tier should have HIGHER lift and AUC. If not, the tiers carry no demonstrated information.
Outputs: outputs/metrics/uncertainty_validation_members.csv, uncertainty_validation_tiers.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.exploration.blocks import block_ids, make_folds
from src.exploration.metrics import eval_scores
from src.uncertainty.applicability import aoa_threshold, dissimilarity_index, standardize
from src.uncertainty.confidence import assign_tiers, data_quality
from src.uncertainty.data import load_cells, valid_mask
from src.uncertainty.ensemble import ensemble_ranks, member_means

log = get_logger("validate_uncertainty")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    size = cfg["grid"]["cell_size_m"]
    U = yaml.safe_load(resolve("config/uncertainty.yaml").read_text(encoding="utf-8"))
    sets = yaml.safe_load(resolve("config/feature_sets.yaml").read_text(encoding="utf-8"))["sets"]
    M, A, V = U["members"], U["applicability"], dict(U["validation"])
    if args.quick:
        V.update(block_km=[V["block_km"][0]], repeats=1)

    df = load_cells(cfg, U)
    d = df[valid_mask(df, U, sets)].reset_index(drop=True)
    xy = np.column_stack([d.col.to_numpy() * size, d.row.to_numpy() * size])
    is_pos = d.is_pos.to_numpy()
    Xsets = {s: d[sets[s]].to_numpy(float) for s in M["feature_sets"]}
    Xa = Xsets[A["feature_set"]]
    Q = data_quality(d["dense_veg_window"].to_numpy())
    belt = cKDTree(xy[is_pos]).query(xy)[0] <= V["belt_ring_km"] * 1000
    log.info("%d assessable cells, %d known records", len(d), is_pos.sum())

    mem_rows, tier_rows = [], []
    for block_km in V["block_km"]:
        for rep in range(V["repeats"]):
            rng = np.random.default_rng(U["seed"] + 1000 * rep + int(block_km * 100))
            B = block_km * 1000
            ids = block_ids(xy, B, offset=rng.uniform(0, B, 2))
            try:
                folds, k = make_folds(ids, is_pos, V["folds"], rng)
            except ValueError as e:
                log.warning("block %s repeat %d skipped: %s", block_km, rep, e)
                continue
            for f in range(k):
                test = folds == f
                pos_t, belt_t = is_pos[test], belt[test]
                if int(pos_t.sum()) < V["min_test_positives"]:
                    continue
                train, pred_idx = ~test, np.flatnonzero(test)
                ranks, labels, _ = ensemble_ranks(Xsets, M["models"], xy, is_pos, train, pred_idx, M["background_draws"],
                                                  M["background_ratio"], M["exclusion_buffer_km"] * 1000,
                                                  seed=U["seed"] + 1000 * rep + f, rf=M["rf"], xgb=M["xgb"])
                P_t, S_t = ranks.mean(axis=1), ranks.std(axis=1)
                base = {"block_km": block_km, "repeat": rep, "fold": f}
                for (m, fs), r in member_means(ranks, labels).items():
                    mem_rows.append({**base, "member": f"{m}@{fs}", "auc_belt": eval_scores(r[belt_t], pos_t[belt_t])["auc"]})
                mem_rows.append({**base, "member": "ENSEMBLE", "auc_belt": eval_scores(P_t[belt_t], pos_t[belt_t])["auc"]})
                tr_pos = np.flatnonzero(train & is_pos)
                Z, _, _ = standardize(Xa, Xa[train])
                thr, _ = aoa_threshold(Z[tr_pos], block_ids(xy[tr_pos], A["threshold_block_km"] * 1000), A["whisker"])
                inside = dissimilarity_index(Z[pred_idx], Z[tr_pos]) <= thr
                tier = assign_tiers(S_t, inside, Q[test], np.ones(len(S_t), bool), float(np.median(S_t)),
                                    U["quality"]["min_quality"])
                for t in ("Medium", "Low"):
                    msk = (tier == t) & belt_t
                    npos = int((pos_t & msk).sum())
                    auc = (eval_scores(P_t[msk], pos_t[msk])["auc"] if V["min_tier_positives"] <= npos < msk.sum() else np.nan)
                    tier_rows.append({**base, "tier": t, "n_cells_belt": int(msk.sum()), "n_pos_belt": npos, "auc_in_tier": auc})
        log.info("finished block size %s km", block_km)

    mem = pd.DataFrame(mem_rows)
    tr = pd.DataFrame(tier_rows)
    mem.to_csv(resolve("outputs/metrics/uncertainty_validation_members.csv"), index=False)
    tr.to_csv(resolve("outputs/metrics/uncertainty_validation_tiers.csv"), index=False)

    print("\n=== Belt AUC on held-out blocks: ensemble vs members (mean over folds x repeats; chance 0.5) ===")
    t1 = mem.groupby(["member", "block_km"])["auc_belt"].agg(["mean", "std", "count"]).round(2).reset_index()
    print(t1.pivot(index="member", columns="block_km", values="mean").to_string())
    print(f"(spread across folds is about {mem.groupby(['member', 'block_km'])['auc_belt'].std().mean():.2f}; differences below ~0.05 are noise)")

    print("\n=== Confidence tiers on held-out blocks, inside the 15 km belt ===")
    g = tr.groupby(["block_km", "tier"]).agg(cells=("n_cells_belt", "sum"), records=("n_pos_belt", "sum"),
                                              auc_in_tier=("auc_in_tier", "mean"), folds_with_auc=("auc_in_tier", "count")).reset_index()
    tot = g.groupby("block_km")[["cells", "records"]].transform("sum")
    g["share_of_cells"] = (g.cells / tot.cells).round(2)
    g["share_of_known_records"] = (g.records / tot.records).round(2)
    g["lift"] = (g.share_of_known_records / g.share_of_cells).round(2)
    g["auc_in_tier"] = g.auc_in_tier.round(2)
    print(g[["block_km", "tier", "share_of_cells", "share_of_known_records", "lift", "auc_in_tier", "folds_with_auc"]].to_string(index=False))
    print("\nReading guide: lift > 1 means the tier holds MORE known records than its share of cells. "
          "If Medium is not clearly above Low in lift AND AUC, the tiers have not shown real information (report that).")


if __name__ == "__main__":
    main()
