"""Ranking metrics for presence vs background data.

Absolute AUC / PR-AUC depend on how many easy background cells are in the test area, so we
report them for two evaluation domains (district-wide and 'belt') and also give lift and the
share of known records captured in the top 10% of area. All are RELATIVE measures of ranking
skill, not probabilities that a reserve exists.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_scores(score: np.ndarray, is_pos: np.ndarray) -> dict[str, float]:
    n, n_pos = len(is_pos), int(is_pos.sum())
    if n_pos < 1 or n_pos == n:
        return {"auc": np.nan, "lift": np.nan, "cap10": np.nan}
    auc = roc_auc_score(is_pos, score)
    lift = average_precision_score(is_pos, score) / (n_pos / n)      # PR-AUC relative to prevalence
    pct = rankdata(score, method="average") / n
    return {"auc": float(auc), "lift": float(lift), "cap10": float(np.mean(pct[is_pos] >= 0.90))}


def eval_domains(score: np.ndarray, is_pos: np.ndarray, belt: np.ndarray) -> dict[str, float]:
    out = {}
    for tag, mask in (("district", np.ones(len(score), bool)), ("belt", belt)):
        for k, v in eval_scores(score[mask], is_pos[mask]).items():
            out[f"{k}_{tag}"] = v
    return out
