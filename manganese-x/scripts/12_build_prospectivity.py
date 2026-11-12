"""Phase 10: relative prospectivity, uncertainty, confidence tier, exploration priority and zones for every cell.

Run (project root):  python scripts/12_build_prospectivity.py
Needs: features + context + positives tables (scripts 06, 10), config/uncertainty.yaml, config/feature_sets.yaml
Outputs: data/final/prospectivity_cells.csv, data/final/zones.csv, models/exploration_ensemble_v1.json,
         outputs/maps/prospectivity_overview.png
All outputs are MODEL OUTPUT: a relative ranking for exploration prioritisation, NOT a probability of a reserve.
Known records score high by construction (they are training data and the proximity term uses them), so the
interesting zones are the ones that do NOT contain a known record ("extension candidates").
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree

from src.common.config import load_config, resolve
from src.common.logging_setup import get_logger
from src.exploration.blocks import block_ids
from src.uncertainty.applicability import aoa_threshold, dissimilarity_index, standardize
from src.uncertainty.confidence import (COMPONENTS, assign_tiers, data_quality, normalised, percentile_rank,
                                        priority_components, priority_stability, select_stepout, top_mask)
from src.uncertainty.data import load_cells, valid_mask
from src.uncertainty.ensemble import ensemble_ranks
from src.uncertainty.explain import linear_contributions, top_driver
from src.uncertainty.zones import label_zones

log = get_logger("prospectivity")


def boundary_lines(path):
    import geopandas as gpd          # imported here so the maps still work if the boundary cannot be read
    lines = []
    for geom in gpd.read_file(path).to_crs("EPSG:4326").geometry:
        for p in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            lines.append(np.asarray(p.exterior.coords))
    return lines


def overview_map(df, zones, lines, path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    aspect = 1 / np.cos(np.radians(df.latitude.mean()))
    known = df[df.is_known_record]
    tier_cols = {"Medium": "#2E8B57", "Low": "#E8A33D", "Not assessed": "#BBBBBB"}
    panels = [("Relative prospectivity (percentile, 1 = highest)", df.prospectivity_pct, "viridis", None),
              ("Model disagreement (spread of member ranks)", df.model_spread, "magma_r", None),
              ("Confidence tier", None, None, tier_cols),
              ("Exploration priority (percentile)", df.priority_pct, "viridis", None)]
    for ax, (title, vals, cmap, cats) in zip(axes.ravel(), panels):
        if cats is None:
            lo, hi = np.nanpercentile(vals, [1, 99])
            sc = ax.scatter(df.longitude, df.latitude, c=vals, s=0.8, marker="s", linewidths=0, vmin=lo, vmax=hi,
                            cmap=cmap, rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        else:
            ax.scatter(df.longitude, df.latitude, c=df.confidence_tier.map(cats), s=0.8, marker="s", linewidths=0,
                       rasterized=True)
            for k, c in cats.items():
                ax.scatter([], [], c=c, label=k, s=30)
            ax.legend(fontsize=9, loc="lower left")
        for ln in lines:
            ax.plot(ln[:, 0], ln[:, 1], color="black", lw=0.6)
        ax.scatter(known.longitude, known.latitude, s=8, facecolor="none", edgecolor="red", linewidths=0.6)
        if title.startswith("Exploration priority") and len(zones):
            for _, z in zones.head(12).iterrows():
                ax.text(z.centroid_lon, z.centroid_lat, str(int(z.zone_rank)), fontsize=9, weight="bold",
                        ha="center", va="center", color="white", bbox=dict(boxstyle="circle,pad=0.15", fc="#14213D", ec="none"))
        ax.set_title(title, fontsize=11)
        ax.set_aspect(aspect)
        ax.tick_params(labelsize=7)
    fig.suptitle("MODEL OUTPUT (relative ranking for exploration prioritisation; red rings = known USGS MRDS records; "
                 "numbers = top priority zones). Not a reserve estimate.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def stepout_map(df, zones, lines, path):
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.scatter(df.longitude, df.latitude, c=df.prospectivity_pct, s=0.8, marker="s", linewidths=0, cmap="Greys",
               alpha=0.55, rasterized=True)
    cand = df[df.stepout_zone_id > 0]
    ax.scatter(cand.longitude, cand.latitude, c="#C0392B", s=2.0, marker="s", linewidths=0, rasterized=True)
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], color="black", lw=0.6)
    known = df[df.is_known_record]
    ax.scatter(known.longitude, known.latitude, s=14, facecolor="none", edgecolor="#1F5FBF", linewidths=0.9, label="known record")
    for _, z in zones.head(15).iterrows():
        ax.text(z.centroid_lon, z.centroid_lat, str(int(z.zone_rank)), fontsize=9, weight="bold", ha="center", va="center",
                color="white", bbox=dict(boxstyle="circle,pad=0.15", fc="#14213D", ec="none"))
    ax.set_aspect(1 / np.cos(np.radians(df.latitude.mean())))
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Step-out candidate zones (red): at least 5 km from any known record, Medium confidence, top 5% by the\n"
                 "features-only ranking (grey = relative prospectivity). MODEL OUTPUT: untested for discovery.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    size = cfg["grid"]["cell_size_m"]
    U = yaml.safe_load(resolve("config/uncertainty.yaml").read_text(encoding="utf-8"))
    sets = yaml.safe_load(resolve("config/feature_sets.yaml").read_text(encoding="utf-8"))["sets"]
    M, A, P_, W = U["members"], U["applicability"], U["priority"], U["priority"]["weights"]

    df = load_cells(cfg, U)
    valid = valid_mask(df, U, sets)
    n, is_pos = len(df), df["is_pos"].to_numpy()
    nrow, ncol = int(df.row.max()) + 1, int(df.col.max()) + 1
    log.info("%d cells; %d assessable (%.1f%%); known records usable: %d of %d", n, valid.sum(), 100 * valid.mean(),
             (is_pos & valid).sum(), is_pos.sum())
    xy = np.column_stack([df.col.to_numpy() * size, df.row.to_numpy() * size])
    Xsets = {s: df[sets[s]].to_numpy(float) for s in M["feature_sets"]}
    pred_idx = np.flatnonzero(valid)

    # 1. ensemble: relative prospectivity + model agreement
    prim = (U["primary_model"]["name"], U["primary_model"]["feature_set"])
    ranks, labels, kept = ensemble_ranks(Xsets, M["models"], xy, is_pos, valid, pred_idx, M["background_draws"],
                                         M["background_ratio"], M["exclusion_buffer_km"] * 1000, seed=U["seed"],
                                         rf=M["rf"], xgb=M["xgb"], keep=prim)
    log.info("ensemble: %d members (%d models x %d windows x %d draws)", ranks.shape[1], len(M["models"]),
             len(M["feature_sets"]), M["background_draws"])
    P, S = np.full(n, np.nan), np.full(n, np.nan)
    P[pred_idx], S[pred_idx] = ranks.mean(axis=1), ranks.std(axis=1)
    P_pct = percentile_rank(P)

    # 2. applicability against known-record conditions
    Xa = Xsets[A["feature_set"]]
    Z, _, _ = standardize(Xa, Xa[valid])
    pos_idx = np.flatnonzero(is_pos & valid)
    thr, _ = aoa_threshold(Z[pos_idx], block_ids(xy[pos_idx], A["threshold_block_km"] * 1000), A["whisker"])
    DI = np.full(n, np.nan)
    DI[pred_idx] = dissimilarity_index(Z[pred_idx], Z[pos_idx])
    inside = np.nan_to_num(DI, nan=np.inf) <= thr

    # 3. data quality, confidence tier
    Q = data_quality(df["dense_veg_window"].to_numpy())
    spread_cut = float(np.nanmedian(S))
    tier = assign_tiers(np.nan_to_num(S, nan=np.inf), inside, np.nan_to_num(Q, nan=0.0), valid, spread_cut,
                        U["quality"]["min_quality"])

    # 4. proximity to known records and priority
    d_known = cKDTree(xy[is_pos]).query(xy)[0] / 1000.0
    prox_pct = percentile_rank(np.where(valid, -d_known, np.nan))
    comp = priority_components(P_pct[valid], prox_pct[valid], tier[valid], Q[valid], P_["confidence_value"])
    score_v = comp @ normalised(W)
    score = np.full(n, np.nan)
    score[valid] = score_v
    topA = top_mask(score_v, P_["top_fraction_high"])
    topB = top_mask(score_v, P_["top_fraction_medium"])
    pclass = np.full(n, "Not assessed", dtype=object)
    pclass[valid] = np.where(topA, "Priority A (top 5%)", np.where(topB, "Priority B (next 15%)", "Priority C"))
    rng = np.random.default_rng(U["seed"])
    stab_v, jac = priority_stability(comp, W, P_["top_fraction_high"], P_["sensitivity_draws"],
                                     P_["dirichlet_concentration"], rng)
    stab = np.full(n, np.nan)
    stab[valid] = stab_v

    # 5. per-cell explanation from the primary logistic model
    names = sets[prim[1]]
    contrib = np.full((n, len(names)), np.nan)
    contrib[pred_idx] = linear_contributions(kept, Xsets[prim[1]][pred_idx])
    drv = np.full(n, "", dtype=object)
    drv[pred_idx] = top_driver(contrib[pred_idx], names)

    # 6. zones from Priority A
    zid = np.zeros(n, int)
    zid[valid] = label_zones(df.row.to_numpy()[valid], df.col.to_numpy()[valid], topA, nrow, ncol, U["zones"]["min_cells"])

    out = df[["cell_id", "row", "col", "latitude", "longitude"]].copy()
    out["prospectivity_index"], out["prospectivity_pct"] = P, P_pct
    out["model_spread"], out["aoa_dissimilarity"], out["inside_aoa"] = S, DI, inside & valid
    out["data_quality"], out["confidence_tier"] = Q, tier
    out["dist_to_known_km"], out["proximity_pct"] = d_known, prox_pct
    out["priority_score"], out["priority_pct"] = score, percentile_rank(score)
    out["priority_class"], out["priority_stability"] = pclass, stab
    out["top_driver"] = drv
    for j, nm in enumerate(names):
        out[f"contrib_{nm}"] = contrib[:, j]
    out["is_known_record"], out["zone_raw"], out["data_class"] = is_pos, zid, "MODEL_OUTPUT"

    # zone table, renumbered by mean priority
    zrows = []
    cw = comp * normalised(W)
    cw_full = np.full((n, 4), np.nan)
    cw_full[valid] = cw
    for z in np.unique(zid[zid > 0]):
        m = zid == z
        zr = {"zone_raw": int(z), "n_cells": int(m.sum()), "area_km2": round(m.sum() * (size / 1000) ** 2, 2),
              "centroid_lat": out.latitude[m].mean(), "centroid_lon": out.longitude[m].mean(),
              "mean_priority": np.nanmean(score[m]), "mean_prospectivity_pct": np.nanmean(P_pct[m]),
              "min_dist_to_known_km": d_known[m].min(), "n_known_records": int(is_pos[m].sum()),
              "share_medium_confidence": float((tier[m] == "Medium").mean()), "mean_stability": np.nanmean(stab[m]),
              "main_driver": COMPONENTS[int(np.argmax(np.nanmean(cw_full[m], axis=0)))]}
        zr["zone_type"] = "known belt" if zr["n_known_records"] else "extension candidate"
        zrows.append(zr)
    zones = pd.DataFrame(zrows)
    if len(zones):
        zones = zones.sort_values("mean_priority", ascending=False).reset_index(drop=True)
        zones.insert(0, "zone_rank", np.arange(1, len(zones) + 1))
        out["zone_id"] = out["zone_raw"].map(dict(zip(zones.zone_raw, zones.zone_rank))).fillna(0).astype(int)
    else:
        out["zone_id"] = 0
    # 7. step-out candidates: features-only ranking, far from known records, Medium confidence
    SO = U["stepout"]
    so_v = select_stepout(P[valid], d_known[valid], tier[valid], SO["min_distance_km"], SO["top_fraction"])
    so_zid = np.zeros(n, int)
    so_zid[valid] = label_zones(df.row.to_numpy()[valid], df.col.to_numpy()[valid], so_v, nrow, ncol, SO["min_cells"])
    so_rows = []
    for z in np.unique(so_zid[so_zid > 0]):
        m = so_zid == z
        so_rows.append({"zone_raw": int(z), "n_cells": int(m.sum()), "area_km2": round(m.sum() * (size / 1000) ** 2, 2),
                        "centroid_lat": out.latitude[m].mean(), "centroid_lon": out.longitude[m].mean(),
                        "mean_prospectivity_pct": np.nanmean(P_pct[m]), "mean_model_spread": np.nanmean(S[m]),
                        "min_dist_to_known_km": d_known[m].min(), "mean_dist_to_known_km": d_known[m].mean(),
                        "share_inside_aoa": float(inside[m].mean()),
                        "main_driver": pd.Series(drv[m]).value_counts().index[0],
                        "evidence": "features only (ensemble); no proximity term"})
    so_zones = pd.DataFrame(so_rows)
    if len(so_zones):
        so_zones = so_zones.sort_values("mean_prospectivity_pct", ascending=False).reset_index(drop=True)
        so_zones.insert(0, "zone_rank", np.arange(1, len(so_zones) + 1))
        out["stepout_zone_id"] = pd.Series(so_zid).map(dict(zip(so_zones.zone_raw, so_zones.zone_rank))).fillna(0).astype(int).to_numpy()
        for c_ in so_zones.columns:
            if so_zones[c_].dtype.kind == "f":
                so_zones[c_] = so_zones[c_].round(4)
    else:
        out["stepout_zone_id"] = 0
    out["stepout_candidate"] = out["stepout_zone_id"] > 0
    so_zones.drop(columns="zone_raw", errors="ignore").to_csv(resolve("data/final/stepout_zones.csv"), index=False)
    out = out.drop(columns="zone_raw")
    for c in zones.columns if len(zones) else []:
        if zones[c].dtype.kind == "f":
            zones[c] = zones[c].round(4)
    out.round(4).to_csv(resolve("data/final/prospectivity_cells.csv"), index=False)
    zones.drop(columns="zone_raw", errors="ignore").to_csv(resolve("data/final/zones.csv"), index=False)

    meta = {"model_name": "ManganSight exploration ensemble", "version": "v1", "data_class": "MODEL_OUTPUT",
            "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "features": {s: sets[s] for s in M["feature_sets"]}, "models": M["models"],
            "background_draws": M["background_draws"], "n_known_records_used": int((is_pos & valid).sum()),
            "validation_method": "spatial block cross-validation (10-40 km) + random CV for comparison; see docs/experiment_log.md",
            "metrics": "see docs/experiment_log.md and outputs/metrics/uncertainty_validation_*.csv (run scripts/13)",
            "config": U, "limitations": ["no geology layer", "59 imperfectly located records",
                                          "modest regional skill; location-only baseline is stronger",
                                          "'High' confidence is not assigned without independent evidence"]}
    resolve("models/exploration_ensemble_v1.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    try:
        lines = boundary_lines(resolve(cfg["study_area"]["boundary_file"]))
    except Exception as exc:  # noqa: BLE001
        log.warning("Boundary outline not drawn (%s)", exc.__class__.__name__)
        lines = []
    overview_map(out, zones, lines, resolve("outputs/maps/prospectivity_overview.png"))
    stepout_map(out, so_zones, lines, resolve("outputs/maps/stepout_candidates.png"))

    v = out[valid]
    print("\nConfidence tiers (assessed cells):")
    print(v.confidence_tier.value_counts().to_frame("cells").assign(share=lambda d: (100 * d.cells / d.cells.sum()).round(1)).to_string())
    print(f"\nInside the area of applicability: {100 * v.inside_aoa.mean():.1f}% (threshold DI <= {thr:.2f}); "
          f"median model spread {spread_cut:.3f}; mean data quality {np.nanmean(Q[valid]):.2f}")
    print(f"Priority A cells: {int(topA.sum())}; weight-sensitivity: mean overlap of the top group under random weights = {jac:.2f} "
          f"(1 = identical); cells in Priority A in >=90% of weight draws: {int((stab_v[topA] >= 0.9).sum())} of {int(topA.sum())}")
    if len(zones):
        print("\nTop zones (known belt = contains a known record; extension candidate = none inside):")
        print(zones.head(12)[["zone_rank", "n_cells", "area_km2", "zone_type", "n_known_records", "min_dist_to_known_km",
                              "mean_priority", "share_medium_confidence", "mean_stability", "main_driver"]].round(2).to_string(index=False))
    print(f"\nSTEP-OUT candidates (>= {SO['min_distance_km']:g} km from any known record, Medium confidence, top {SO['top_fraction']:.0%} "
          f"of eligible cells by the features-only ranking): {len(so_zones)} zones, {int(so_v.sum())} cells "
          f"({so_v.sum() * (size / 1000) ** 2:.0f} km2). Eligible cells: {int(((d_known[valid] >= SO['min_distance_km']) & (tier[valid] == 'Medium')).sum())}.")
    if len(so_zones):
        print(so_zones.head(12)[["zone_rank", "n_cells", "area_km2", "min_dist_to_known_km", "mean_prospectivity_pct",
                                 "mean_model_spread", "share_inside_aoa", "main_driver"]].round(2).to_string(index=False))
    print("These rest on the features-only ranking (belt AUC about 0.65 on held-out blocks) and have NOT been field-checked.")
    log.info("Saved data/final/prospectivity_cells.csv, zones.csv, stepout_zones.csv, models/exploration_ensemble_v1.json, outputs/maps/prospectivity_overview.png")


if __name__ == "__main__":
    main()
