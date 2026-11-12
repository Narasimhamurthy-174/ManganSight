"""Data quality, confidence tiers, exploration priority and its weight-sensitivity test."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

COMPONENTS = ("prospectivity", "proximity", "confidence", "quality")


def data_quality(dense_veg_share: np.ndarray) -> np.ndarray:
    """1 = surface fully visible, 0 = fully covered by dense vegetation (within the window)."""
    return np.clip(1.0 - dense_veg_share, 0.0, 1.0)


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1] among finite values; NaN stays NaN."""
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    out[ok] = rankdata(x[ok], method="average") / ok.sum()
    return out


def assign_tiers(spread, inside_aoa, quality, assessed, spread_cut, min_quality) -> np.ndarray:
    """Medium = members agree AND cell resembles known-record conditions AND surface mostly visible.
    Low = any of these fails. 'Not assessed' = missing features. 'High' is deliberately NOT assigned:
    it needs independent evidence (geology) that we do not have."""
    good = (spread <= spread_cut) & inside_aoa & (quality >= min_quality)
    tier = np.where(good, "Medium", "Low").astype(object)
    tier[~assessed] = "Not assessed"
    return tier


def priority_components(p_pct, prox_pct, tier, quality, conf_value: dict) -> np.ndarray:
    conf = np.array([conf_value.get(t, 0.0) for t in tier], dtype=float)
    return np.column_stack([p_pct, prox_pct, conf, quality])


def normalised(weights: dict) -> np.ndarray:
    w = np.array([weights[k] for k in COMPONENTS], dtype=float)
    return w / w.sum()


def top_mask(score: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(np.ceil(frac * len(score))))
    idx = np.argpartition(-score, k - 1)[:k]
    m = np.zeros(len(score), bool)
    m[idx] = True
    return m


def priority_stability(comp: np.ndarray, weights: dict, top_frac: float, n_draws: int, conc: float,
                       rng: np.random.Generator):
    """Re-draw the weights around the chosen ones (Dirichlet) and see how often each cell stays in the top group.
    Returns (share of draws in which each cell is in the top group, mean Jaccard overlap with the baseline top group)."""
    w = normalised(weights)
    base = top_mask(comp @ w, top_frac)
    counts, jac = np.zeros(len(comp)), []
    for _ in range(n_draws):
        m = top_mask(comp @ rng.dirichlet(conc * w), top_frac)
        counts += m
        jac.append((m & base).sum() / (m | base).sum())
    return counts / n_draws, float(np.mean(jac))


def select_stepout(index: np.ndarray, dist_km: np.ndarray, tier: np.ndarray, min_km: float, top_frac: float) -> np.ndarray:
    """'Step-out' candidates = top fraction (by the features-only prospectivity index) of the cells that are at least
    min_km from every known record AND have Medium confidence. Proximity to known records plays NO role here."""
    eligible = (dist_km >= min_km) & (tier == "Medium") & np.isfinite(index)
    mask = np.zeros(len(index), bool)
    idx = np.flatnonzero(eligible)
    if len(idx):
        mask[idx[top_mask(index[idx], top_frac)]] = True
    return mask
