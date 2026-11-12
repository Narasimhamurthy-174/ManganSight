"""Spatial cross-validation driver for presence-background prospectivity models."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.exploration.blocks import block_ids, make_folds
from src.exploration.metrics import eval_domains, eval_scores
from src.exploration.models import fit_model, knn_dist_score
from src.exploration.sampling import draw_background

log = logging.getLogger("cv")


def run_cv(df: pd.DataFrame, predictors: list[str], cfg: dict, cell_m: float = 250.0) -> pd.DataFrame:
    """df needs columns row, col, is_pos (bool) and the predictors (no NaN). Returns one row per
    (block size, repeat, fold, strategy, model). If cfg['importance_block_km'] is set, permutation
    importance on the held-out blocks is returned in result.attrs['importance']."""
    X = df[predictors].to_numpy(float)
    xy = np.column_stack([df["col"].to_numpy() * cell_m, df["row"].to_numpy() * cell_m])
    is_pos = df["is_pos"].to_numpy(bool)
    d_any, _ = cKDTree(xy[is_pos]).query(xy)
    belt = d_any <= cfg["belt_eval_ring_km"] * 1000
    buffer_m, ring_m = cfg["exclusion_buffer_km"] * 1000, cfg["near_belt_ring_km"] * 1000
    imp_block = cfg.get("importance_block_km")
    imp_strat = cfg["strategies"][0]
    rows, imp_rows = [], []
    for block_km in cfg["block_km"]:
        for rep in range(cfg["repeats"]):
            rng = np.random.default_rng(cfg["seed"] + 1000 * rep + int(block_km * 100))
            B = block_km * 1000
            ids = block_ids(xy, B, offset=rng.uniform(0, B, 2))
            try:
                folds, k_eff = make_folds(ids, is_pos, cfg["folds"], rng)
            except ValueError as e:
                log.warning("block %.2f km repeat %d skipped: %s", block_km, rep, e)
                continue
            for f in range(k_eff):
                test = folds == f
                if int((is_pos & test).sum()) < cfg["min_test_positives"]:
                    continue
                train = ~test
                tr_pos = np.flatnonzero(train & is_pos)
                pos_t, belt_t, X_te = is_pos[test], belt[test], X[test]
                base = {"block_km": block_km, "repeat": rep, "fold": f, "k_eff": k_eff,
                        "n_test_pos": int(pos_t.sum())}
                s = knn_dist_score(xy[tr_pos], xy[test])
                rows.append({**base, "strategy": "n/a", "model": "knn_dist", **eval_domains(s, pos_t, belt_t)})
                for strat in cfg["strategies"]:
                    bg = draw_background(xy, train, is_pos, strat, cfg["background_ratio"] * len(tr_pos),
                                         rng, buffer_m, ring_m)
                    idx = np.concatenate([tr_pos, bg])
                    y = np.r_[np.ones(len(tr_pos)), np.zeros(len(bg))]
                    for m in cfg["models"]:
                        if m == "knn_dist":
                            continue
                        model = fit_model(m, X[idx], y, seed=cfg["seed"] + rep, rf=cfg.get("rf"), xgb=cfg.get("xgb"))
                        s = model.predict_proba(X_te)[:, 1]
                        rows.append({**base, "strategy": strat, "model": m, **eval_domains(s, pos_t, belt_t)})
                        if imp_block is not None and block_km == imp_block and strat == imp_strat:
                            ref = eval_scores(s[belt_t], pos_t[belt_t])["auc"]
                            for j, name in enumerate(predictors):
                                drops = []
                                for _ in range(2):
                                    Xp = X_te.copy()
                                    Xp[:, j] = rng.permutation(Xp[:, j])
                                    sp = model.predict_proba(Xp)[:, 1]
                                    drops.append(ref - eval_scores(sp[belt_t], pos_t[belt_t])["auc"])
                                imp_rows.append({"block_km": block_km, "repeat": rep, "fold": f, "model": m,
                                                 "feature": name, "auc_drop": float(np.mean(drops))})
        log.info("finished block size %.2f km", block_km)
    res = pd.DataFrame(rows)
    if imp_rows:
        res.attrs["importance"] = pd.DataFrame(imp_rows)
    return res


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread over (repeat, fold) units. Units are NOT independent: read the spread as a
    rough indication of instability, not as a confidence interval."""
    cols = [c for c in res.columns if c.startswith(("auc_", "lift_", "cap10_"))]
    g = res.groupby(["block_km", "strategy", "model"])
    out = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd"))
    out["n_units"] = g.size()
    return out.reset_index()


def summarise_importance(imp: pd.DataFrame) -> pd.DataFrame:
    """Mean drop in belt-AUC when one feature is shuffled in the held-out blocks (bigger = model relies on it more)."""
    g = imp.groupby(["model", "feature"])["auc_drop"]
    out = pd.DataFrame({"mean_auc_drop": g.mean(), "sd": g.std(), "n_units": g.size()}).reset_index()
    return out.sort_values(["model", "mean_auc_drop"], ascending=[True, False]).reset_index(drop=True)


def plot_results(summary: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = sorted(summary["block_km"].unique())
    labels = ["random" if b < 1 else f"{b:g} km" for b in blocks]
    strategies = [s for s in summary["strategy"].unique() if s != "n/a"]
    fig, axes = plt.subplots(2, len(strategies), figsize=(6 * len(strategies), 8), squeeze=False, sharey=True)
    for j, strat in enumerate(strategies):
        for i, dom in enumerate(["district", "belt"]):
            ax = axes[i][j]
            for model in summary["model"].unique():
                sub = summary[(summary.model == model) & (summary.strategy.isin([strat, "n/a"]))].set_index("block_km")
                sub = sub.reindex(blocks)
                ax.errorbar(range(len(blocks)), sub[f"auc_{dom}_mean"], yerr=sub[f"auc_{dom}_sd"],
                            marker="o", capsize=3, label=model)
            ax.axhline(0.5, color="grey", ls="--", lw=1)
            ax.set_xticks(range(len(blocks)), labels)
            ax.set_title(f"{dom} test area | background: {strat}")
            ax.set_xlabel("Validation block size")
            ax.set_ylabel("ROC-AUC (known records vs other cells)")
            ax.legend(fontsize=8)
    fig.suptitle("Does skill survive spatial validation? (dashed line = no skill)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
