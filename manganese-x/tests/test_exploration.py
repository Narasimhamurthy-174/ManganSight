"""Synthetic data (fabricated for code testing only)."""
import numpy as np
import pandas as pd
import pytest

from src.exploration.blocks import block_ids, make_folds
from src.exploration.cv import plot_results, run_cv, summarise
from src.exploration.metrics import eval_scores
from src.exploration.sampling import draw_background


def grid_xy(n=120, cell=250.0):
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return r.ravel(), c.ravel(), np.column_stack([c.ravel() * cell, r.ravel() * cell])


def clustered_df(seed=0):
    rng = np.random.default_rng(seed)
    r, c, xy = grid_xy()
    f1 = rng.normal(size=len(r))                                   # unrelated noise feature
    pos = np.zeros(len(r), bool)
    for cx, cy in [(5000, 5000), (22000, 8000), (12000, 25000)]:   # 3 spatial clusters of records
        d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        pos[np.argsort(d)[:20]] = True
    return pd.DataFrame({"row": r, "col": c, "is_pos": pos, "f1": f1, "f2": rng.normal(size=len(r))})


def test_block_ids_group_nearby_points_and_offset_shifts_grid():
    xy = np.array([[10.0, 10.0], [90.0, 90.0], [150.0, 10.0]])
    ids = block_ids(xy, 100.0)
    assert ids[0] == ids[1] and ids[0] != ids[2]
    assert block_ids(xy, 100.0, offset=(60.0, 0.0))[1] != block_ids(xy, 100.0, offset=(60.0, 0.0))[0]
    with pytest.raises(ValueError):
        block_ids(xy, 0)


def test_folds_never_split_a_block_and_balance_positives():
    df = clustered_df()
    xy = np.column_stack([df.col * 250.0, df.row * 250.0])
    ids = block_ids(xy, 5000.0)
    folds, k = make_folds(ids, df.is_pos.to_numpy(), 3, np.random.default_rng(0))
    assert k == 3
    assert (pd.Series(folds).groupby(ids).nunique() == 1).all()     # one fold per block
    assert all(((folds == f) & df.is_pos.to_numpy()).sum() > 0 for f in range(k))


def test_too_few_positive_blocks_raises():
    df = clustered_df()
    xy = np.column_stack([df.col * 250.0, df.row * 250.0])
    ids = block_ids(xy, 100000.0)                                   # one giant block
    with pytest.raises(ValueError):
        make_folds(ids, df.is_pos.to_numpy(), 5, np.random.default_rng(0))


def test_background_respects_buffer_ring_and_train_only():
    df = clustered_df()
    xy = np.column_stack([df.col * 250.0, df.row * 250.0])
    is_pos = df.is_pos.to_numpy()
    train = df.col.to_numpy() < 60                                   # left half = train
    rng = np.random.default_rng(1)
    from scipy.spatial import cKDTree
    tp = np.flatnonzero(train & is_pos)
    for strat in ("uniform", "near_belt"):
        bg = draw_background(xy, train, is_pos, strat, 300, rng, buffer_m=1000, ring_max_m=6000)
        d, _ = cKDTree(xy[tp]).query(xy[bg])
        assert train[bg].all() and not is_pos[bg].any() and (d > 1000).all()
        if strat == "near_belt":
            assert (d <= 6000).all()
    with pytest.raises(ValueError):
        draw_background(xy, train, is_pos, "bogus", 10, rng)


def test_metrics_perfect_random_and_degenerate():
    pos = np.array([1] * 5 + [0] * 95, bool)
    perfect = eval_scores(np.r_[np.ones(5), np.zeros(95)] + np.random.default_rng(0).random(100) * 0.01, pos)
    assert perfect["auc"] == 1.0 and perfect["cap10"] == 1.0 and perfect["lift"] > 5
    rnd = [eval_scores(np.random.default_rng(s).random(2000), np.r_[np.ones(100), np.zeros(1900)].astype(bool))["auc"]
           for s in range(5)]
    assert 0.4 < np.mean(rnd) < 0.6
    assert np.isnan(eval_scores(np.zeros(10), np.zeros(10, bool))["auc"])


CFG = {"seed": 1, "folds": 3, "repeats": 2, "block_km": [0.25, 5], "strategies": ["uniform", "near_belt"],
       "background_ratio": 10, "exclusion_buffer_km": 1.0, "near_belt_ring_km": 8.0, "belt_eval_ring_km": 8.0,
       "min_test_positives": 3, "models": ["knn_dist", "logreg", "rf"],
       "rf": {"trees": 40, "max_depth": 4, "min_samples_leaf": 3}}


def test_random_cv_is_optimistic_for_clustered_records_and_noise_features_have_no_skill():
    res = run_cv(clustered_df(), ["f1", "f2"], CFG)
    s = summarise(res).set_index(["block_km", "strategy", "model"])
    knn_random = s.loc[(0.25, "n/a", "knn_dist"), "auc_district_mean"]
    knn_spatial = s.loc[(5, "n/a", "knn_dist"), "auc_district_mean"]
    assert knn_random > 0.9 and knn_random > knn_spatial + 0.1      # leakage inflates random CV
    noise_auc = s.loc[(5, "uniform", "logreg"), "auc_district_mean"]
    assert 0.25 < noise_auc < 0.75                                  # pure-noise features: ~chance
    assert set(res.model) == {"knn_dist", "logreg", "rf"}


def test_signal_feature_is_detected(tmp_path):
    df = clustered_df(3)
    df["f1"] = df["is_pos"] * 2.0 + np.random.default_rng(4).normal(size=len(df))   # informative feature
    res = run_cv(df, ["f1", "f2"], {**CFG, "models": ["logreg"]})
    s = summarise(res).set_index(["block_km", "strategy", "model"])
    assert s.loc[(5, "uniform", "logreg"), "auc_district_mean"] > 0.8
    plot_results(summarise(res), tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 1000


def test_permutation_importance_ranks_the_informative_feature_first():
    df = clustered_df(3)
    df["f1"] = df["is_pos"] * 2.0 + np.random.default_rng(4).normal(size=len(df))   # informative
    cfg = {**CFG, "models": ["logreg", "rf"], "block_km": [5], "importance_block_km": 5}
    res = run_cv(df, ["f1", "f2"], cfg)
    from src.exploration.cv import summarise_importance
    imp = summarise_importance(res.attrs["importance"]).set_index(["model", "feature"])["mean_auc_drop"]
    assert imp[("logreg", "f1")] > imp[("logreg", "f2")] + 0.05
    assert imp[("rf", "f1")] > imp[("rf", "f2")]


def test_xgb_model_runs_when_installed():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return                                   # not installed here: skipped (tested on the user's machine)
    from src.exploration.models import fit_predict
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)); y = (X[:, 0] + rng.normal(size=300) > 1).astype(float)
    p = fit_predict("xgb", X, y, X, seed=0)
    assert p.shape == (300,) and 0 <= p.min() and p.max() <= 1


def test_unknown_model_name_raises():
    from src.exploration.models import fit_predict
    with pytest.raises(ValueError):
        fit_predict("bogus", np.zeros((4, 2)), np.array([0, 1, 0, 1.0]), np.zeros((2, 2)))
