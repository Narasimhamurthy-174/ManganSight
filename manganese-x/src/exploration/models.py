"""Baseline scorers. Higher score = more compatible with known-record conditions."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("knn_dist", "logreg", "rf", "xgb")


def knn_dist_score(xy_train_pos: np.ndarray, xy_test: np.ndarray) -> np.ndarray:
    """Location-only baseline: closeness to the nearest TRAINING known record (no features used)."""
    d, _ = cKDTree(xy_train_pos).query(xy_test)
    return -d


def fit_model(name: str, X_tr, y_tr, seed: int = 0, rf: dict | None = None, xgb: dict | None = None):
    """Fit one feature-based model and return it (it exposes predict_proba)."""
    if name == "logreg":
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    elif name == "rf":
        p = {"trees": 300, "max_depth": 6, "min_samples_leaf": 5, **(rf or {})}
        m = RandomForestClassifier(n_estimators=p["trees"], max_depth=p["max_depth"],
                                   min_samples_leaf=p["min_samples_leaf"],
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    elif name == "xgb":
        from xgboost import XGBClassifier          # imported here so the rest works without xgboost
        n_pos = max(1, int(np.sum(y_tr)))
        p = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 3, **(xgb or {})}
        m = XGBClassifier(**p, scale_pos_weight=float(len(y_tr) - n_pos) / n_pos, tree_method="hist",
                          n_jobs=-1, random_state=seed, verbosity=0)
    else:
        raise ValueError(f"unknown feature model {name!r}")
    m.fit(X_tr, y_tr)
    return m


def fit_predict(name: str, X_tr, y_tr, X_te, seed: int = 0, rf: dict | None = None,
                xgb: dict | None = None) -> np.ndarray:
    return fit_model(name, X_tr, y_tr, seed, rf, xgb).predict_proba(X_te)[:, 1]
