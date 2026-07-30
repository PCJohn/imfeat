"""Tests for the per-level cross-cell summaries ("*_summary_i").

The contract: for every derived feature map, ``<group>_summary_i`` is that map
reduced across the level's cells into SUMMARY_STATS [min, max, mean, std], with
layout (n_feat, C, 4) (the C axis dropped for 2-D input). Tiers:

1. Exactness   -- min/max equal a numpy reduction over the *returned* per-cell map
                  bit-for-bit; mean/std match to float tolerance (C++ sums in
                  float64, numpy's oracle here does too).
2. Semantics   -- degenerate images (flat, single-hot cell) give the obvious
                  summary; only grid levels carry summaries, never "global".
3. Consistency -- the multichannel==single-channel invariance is already covered
                  generically in test_moments/test_struct; here we check the
                  blank-relevant reductions framegate will read.
"""

import numpy as np
import pytest

import imfeat

from conftest import groups

rng = np.random.default_rng(0)

GRIDS = [[(5, 5)], [(5, 5), (3, 3)], [(4, 4), (2, 2)], [(0, 0)]]
STRIDES = [(1, 1), (2, 2)]
GROUPS = ("struct", "hog", "cnt", "lbp", "desc", "mom")


def ref_summary(m: np.ndarray) -> np.ndarray:
    """Reduce a per-cell map (cy, cx[, C], F) over its cells -> (F, C, 4)
    [min, max, mean, std], population std, sums in float64 to mirror the core."""
    if m.ndim == 3:
        m = m[:, :, None, :]
    cells = m.reshape(-1, m.shape[2], m.shape[3])  # (K, C, F)
    d = cells.astype(np.float64)
    stats = np.stack([cells.min(0), cells.max(0), d.mean(0), d.std(0)], -1)  # (C,F,4)
    return stats.transpose(1, 0, 2)  # (F, C, 4)


def norm(sm: np.ndarray) -> np.ndarray:
    """Normalize a summary array to (F, C, 4) (add back the dropped single channel)."""
    return sm if sm.ndim == 3 else sm[:, None, :]


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("nch", [1, 3])
def test_summary_matches_cell_reduction(grid, stride, nch):
    shape = (64, 64) if nch == 1 else (64, 64, nch)
    img = rng.integers(0, 256, shape, np.uint8)
    f = groups(imfeat.FeatureComputer(shape, grid=grid, stride=stride), img)
    for i in range(len(grid)):
        for g in GROUPS:
            ref = ref_summary(f[f"{g}_{i}"])
            got = norm(f[f"{g}_summary_{i}"])
            assert got.shape == ref.shape, f"{g}_summary_{i} shape"
            # min/max exact in the map's own dtype; mean/std to tolerance
            assert np.array_equal(got[..., :2], ref[..., :2]), f"{g}_summary_{i} min/max"
            assert np.allclose(got[..., 2:], ref[..., 2:], rtol=1e-4, atol=1e-5), (
                f"{g}_summary_{i} mean/std"
            )


def test_only_grid_levels_have_summaries():
    f = groups(imfeat.FeatureComputer((32, 32, 3), grid=[(5, 5), (3, 3)]),
        rng.integers(0, 256, (32, 32, 3), np.uint8)
    )
    assert "mom_summary_0" in f and "mom_summary_1" in f
    assert "mom_summary_global" not in f  # the 1-cell global reduction is trivial
    assert f["mom_summary_0"].shape == (len(imfeat.MOMENTS), 3, len(imfeat.SUMMARY_STATS))


def test_flat_image_summary():
    """A constant image: every cell identical, so min==max==mean and std==0."""
    img = np.full((32, 32, 3), 7, np.uint8)
    f = groups(imfeat.FeatureComputer(img.shape, grid=[(4, 4)]), img)
    s = f["mom_summary_0"]  # (4, 3, 4)
    mean_stat = s[imfeat.MOMENTS.index("mean")]  # (C, 4): [min,max,mean,std] of cell means
    assert np.allclose(mean_stat[:, 0], 7.0) and np.allclose(mean_stat[:, 1], 7.0)
    assert np.allclose(mean_stat[:, 3], 0.0)  # zero spread across cells
    var_stat = s[imfeat.MOMENTS.index("var")]
    assert np.allclose(var_stat[:, 1], 0.0)  # max cell variance is 0 -> blank


def test_single_hot_cell_max():
    """One textured cell in an otherwise flat frame: the max stat must pick it up
    while the min stays at the flat floor -- exactly the blank discriminator."""
    img = np.zeros((32, 32), np.uint8)
    img[:8, :8] = rng.integers(0, 256, (8, 8), np.uint8)  # top-left cell only
    f = groups(imfeat.FeatureComputer(img.shape, grid=[(2, 2)]), img)  # 4x4 cells
    vmax = f["mom_summary_0"][imfeat.MOMENTS.index("var")]  # (4,) [min,max,mean,std]
    assert vmax[1] > 1.0 and vmax[0] == 0.0  # some cell varies; some cell is flat


def test_blank_relevant_reductions():
    """The two scalars framegate's blank check reads, cross-checked against a hand
    reduction over the finest maps: max V cell-variance and max V edge-energy."""
    img = rng.integers(0, 256, (64, 64, 3), np.uint8)
    fc = imfeat.FeatureComputer(img.shape, grid=[(5, 5)])
    f = groups(fc, img)
    V = 2  # HSV V channel
    max_var = f["mom_summary_0"][imfeat.MOMENTS.index("var"), V, 1]
    max_energy = f["struct_summary_0"][imfeat.FEATURES.index("energy"), V, 1]
    assert max_var == f["mom_0"][:, :, V, imfeat.MOMENTS.index("var")].max()
    assert max_energy == f["struct_0"][:, :, V, imfeat.FEATURES.index("energy")].max()
