"""Ensemble of prospectivity models: algorithms x feature windows x random background draws.

Each member's scores are turned into percentile ranks among the predicted cells (0-1], so members with
different score scales can be averaged. Mean rank = relative prospectivity index (NOT a probability).
The spread of ranks across members = model-agreement uncertainty (large spread = members disagree).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from src.exploration.models import fit_model
from src.exploration.sampling import draw_background


def ensemble_ranks(Xsets: dict, models, xy, is_pos, train_mask, pred_idx, n_draws, ratio, buffer_m,
                   seed=0, rf=None, xgb=None, keep=None):
    """Return (ranks [n_pred, n_members], labels [(model, set, draw)], kept_models).
    train_mask: bool over ALL cells (valid and not in the test fold). pred_idx: cell indices to score."""
    rng = np.random.default_rng(seed)
    tr_pos = np.flatnonzero(train_mask & is_pos)
    cols, labels, kept = [], [], []
    for d in range(n_draws):
        bg = draw_background(xy, train_mask, is_pos, "uniform", ratio * len(tr_pos), rng, buffer_m)
        idx = np.concatenate([tr_pos, bg])
        y = np.r_[np.ones(len(tr_pos)), np.zeros(len(bg))]
        for fs, X in Xsets.items():
            for m in models:
                model = fit_model(m, X[idx], y, seed=seed + d, rf=rf, xgb=xgb)
                s = model.predict_proba(X[pred_idx])[:, 1]
                cols.append(rankdata(s, method="average") / len(s))
                labels.append((m, fs, d))
                if keep is not None and (m, fs) == tuple(keep):
                    kept.append(model)
    return np.column_stack(cols), labels, kept


def member_means(ranks: np.ndarray, labels) -> dict:
    """Average the draws of each (model, feature set) member -> {(model, set): rank vector}."""
    out = {}
    for j, (m, fs, _) in enumerate(labels):
        out.setdefault((m, fs), []).append(ranks[:, j])
    return {k: np.mean(v, axis=0) for k, v in out.items()}
