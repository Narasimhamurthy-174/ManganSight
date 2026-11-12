"""Per-cell explanation for the primary (logistic) model.

For a linear model the exact SHAP value of feature j is coef_j * (x_j - mean_j) / scale_j (relative to the training mean),
so we compute it directly; positive = pushes the cell's score UP. This is a MODEL explanation, not geological proof.
"""
from __future__ import annotations

import numpy as np


def linear_contributions(pipelines, X: np.ndarray) -> np.ndarray:
    """Average over fitted StandardScaler+LogisticRegression pipelines. Returns [n_cells, n_features]."""
    out = []
    for p in pipelines:
        sc, lr = p.named_steps["standardscaler"], p.named_steps["logisticregression"]
        out.append(lr.coef_[0] * (X - sc.mean_) / sc.scale_)
    return np.mean(out, axis=0)


def top_driver(contrib: np.ndarray, names: list[str]) -> np.ndarray:
    j = np.argmax(np.abs(contrib), axis=1)
    sign = np.sign(contrib[np.arange(len(j)), j])
    return np.array([f"{names[k]} ({'+' if s >= 0 else '-'})" for k, s in zip(j, sign)])
