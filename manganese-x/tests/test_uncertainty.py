"""Synthetic data (fabricated for code testing only)."""
import numpy as np
import pytest

from src.exploration.models import fit_model
from src.uncertainty.applicability import aoa_threshold, dissimilarity_index, standardize
from src.uncertainty.confidence import (assign_tiers, data_quality, percentile_rank, priority_components,
                                        priority_stability, select_stepout, top_mask)
from src.uncertainty.ensemble import ensemble_ranks, member_means
from src.uncertainty.explain import linear_contributions, top_driver
from src.uncertainty.zones import label_zones


def test_standardize_and_zero_variance_column():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    Z, mean, std = standardize(X)
    assert np.allclose(Z[:, 0], [-1.2247, 0, 1.2247], atol=1e-3) and np.allclose(Z[:, 1], 0)


def test_di_zero_on_reference_and_large_far_away_and_threshold_uses_other_blocks():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(40, 3))
    di = dissimilarity_index(np.vstack([ref[:1], ref[:1] + 10]), ref)
    assert di[0] == 0 and di[1] > 2
    blocks = np.repeat(np.arange(4), 10)
    thr, di_ref = aoa_threshold(ref, blocks)
    assert thr >= np.median(di_ref) and len(di_ref) == 40
    one_block, _ = aoa_threshold(ref, np.zeros(40, int))      # falls back to leave-one-out
    assert np.isfinite(one_block)
    with pytest.raises(ValueError):
        dissimilarity_index(ref, ref[:1])


def test_quality_percentile_and_tiers():
    assert np.allclose(data_quality(np.array([0.0, 0.4, 1.5, -0.2])), [1.0, 0.6, 0.0, 1.0])
    p = percentile_rank(np.array([3.0, np.nan, 1.0, 2.0]))
    assert np.allclose(p[[0, 2, 3]], [1.0, 1 / 3, 2 / 3]) and np.isnan(p[1])
    spread = np.array([0.05, 0.30, 0.05, 0.05, np.nan])
    inside = np.array([True, True, False, True, True])
    quality = np.array([0.9, 0.9, 0.9, 0.5, 0.9])
    assessed = np.array([True, True, True, True, False])
    tier = assign_tiers(spread, inside, quality, assessed, spread_cut=0.1, min_quality=0.7)
    assert list(tier) == ["Medium", "Low", "Low", "Low", "Not assessed"]
    assert "High" not in set(tier)


def test_priority_components_top_mask_and_stability():
    tier = np.array(["Medium", "Low", "Not assessed"], dtype=object)
    comp = priority_components(np.array([.9, .5, .1]), np.array([.8, .4, .2]), tier, np.array([1., .5, 0.]),
                               {"Medium": 1.0, "Low": 0.5, "Not assessed": 0.0})
    assert comp.shape == (3, 4) and np.allclose(comp[:, 2], [1.0, 0.5, 0.0])
    assert top_mask(np.array([.1, .9, .5, .7]), 0.5).tolist() == [False, True, False, True]
    rng = np.random.default_rng(0)
    big = rng.random((2000, 4))
    w = {"prospectivity": .28, "proximity": .42, "confidence": .15, "quality": .15}
    stab, jac = priority_stability(big, w, 0.05, 30, 20, rng)
    assert stab.shape == (2000,) and 0 < jac <= 1 and stab.max() > 0.9
    # perfectly dominated cells (best on every component) are always in the top group
    big[:5] = 1.0
    stab2, _ = priority_stability(big, w, 0.05, 30, 20, rng)
    assert (stab2[:5] == 1.0).all()


def test_zones_two_blobs_and_small_noise_dropped():
    m = np.zeros((30, 30), bool)
    m[2:7, 2:7] = True        # 25 cells
    m[15:18, 15:20] = True    # 15 cells
    m[25, 25] = True          # 1 cell, dropped
    row, col = np.where(np.ones_like(m))
    z = label_zones(row, col, m[row, col], 30, 30, min_cells=8).reshape(30, 30)
    assert z[3, 3] == 1 and z[16, 16] == 2 and z[25, 25] == 0 and z[0, 0] == 0


def test_linear_contributions_reproduce_the_logit_and_driver_label():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3)); y = (X[:, 0] * 2 + rng.normal(size=400) > 0).astype(float)
    m = fit_model("logreg", X, y)
    c = linear_contributions([m], X)
    lr = m.named_steps["logisticregression"]
    assert np.allclose(c.sum(axis=1) + lr.intercept_[0], m.decision_function(X), atol=1e-8)
    drv = top_driver(c, ["a", "b", "c"])
    assert drv[0].startswith(("a", "b", "c")) and "(" in drv[0]
    assert (np.char.startswith(drv.astype(str), "a")).mean() > 0.5


def test_ensemble_ranks_shape_range_and_member_means():
    rng = np.random.default_rng(2)
    n = 3000
    xy = np.column_stack([rng.uniform(0, 30000, n), rng.uniform(0, 30000, n)])
    is_pos = np.zeros(n, bool); is_pos[:40] = True
    f = rng.normal(size=n); f[is_pos] += 2.5
    Xs = {"s1": np.column_stack([f, rng.normal(size=n)]), "s2": np.column_stack([f + rng.normal(size=n) * .3, rng.normal(size=n)])}
    ranks, labels, kept = ensemble_ranks(Xs, ["logreg", "rf"], xy, is_pos, np.ones(n, bool), np.arange(n),
                                         n_draws=2, ratio=10, buffer_m=500, seed=1,
                                         rf={"trees": 30, "max_depth": 4, "min_samples_leaf": 3}, keep=("logreg", "s1"))
    assert ranks.shape == (n, 2 * 2 * 2) and ranks.min() > 0 and ranks.max() <= 1 and len(kept) == 2
    mm = member_means(ranks, labels)
    assert set(mm) == {("logreg", "s1"), ("rf", "s1"), ("logreg", "s2"), ("rf", "s2")}
    p = ranks.mean(axis=1)
    assert p[is_pos].mean() > p[~is_pos].mean() + 0.2          # informative feature is picked up


def test_select_stepout_only_far_medium_cells_and_top_fraction():
    idx = np.array([.9, .8, .7, .6, .5, .4, np.nan, .3])
    dist = np.array([10, 1, 10, 10, 10, 10, 10, 10.0])
    tier = np.array(["Medium", "Medium", "Low", "Medium", "Medium", "Medium", "Medium", "Medium"], dtype=object)
    m = select_stepout(idx, dist, tier, 5.0, 0.5)
    assert m.tolist() == [True, False, False, True, True, False, False, False]   # far + Medium + finite, top 3 of 5
    assert not select_stepout(idx, dist * 0, tier, 5.0, 0.5).any()            # nothing is far enough -> empty
