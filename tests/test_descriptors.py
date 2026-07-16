"""Derived nonlinear descriptors (desc_i): standardized moments, structure-tensor
ratios, and HOG histogram-shape summaries.

Three tiers:
  1. exactness  -- each descriptor is a fixed function of sums imfeat already exposes
                   (mom_i, struct_i, hog_i), so recompute it in numpy and demand equality.
  2. closed form -- synthetic patterns with a known descriptor value: a uniform ramp is
                   platykurtic (excess kurtosis -1.2), single-orientation stripes have
                   HOG concentration ~1, axis-aligned vs diagonal splits cardinality.
  3. invariance -- the five dimensionless descriptors must be *exactly* invariant to an
                   integer affine map v -> 2v+15 (contrast+brightness). This is the
                   property that lets a model trained on a few pixels transfer across
                   frames and lighting; detail (energy-weighted) is the one exception.
"""

import numpy as np
import pytest

import imfeat

rng = np.random.default_rng(3)

D = {name: k for k, name in enumerate(imfeat.DESCRIPTOR_FEATURES)}


# ---------------- images -----------------------------------------------------
def noise(n=96, lo=0, hi=256):
    return rng.integers(lo, hi, (n, n), dtype=np.uint8)


def ramp(n=96):
    return np.tile(np.linspace(0, 255, n).astype(np.uint8), (n, 1))


def stripes(n=96, period=6, axis=1):
    t = (np.arange(n) // (period // 2)) % 2 * 200 + 20
    line = t.astype(np.uint8)
    return np.tile(line, (n, 1)) if axis == 1 else np.tile(line[:, None], (1, n))


def diagonal_stripes(n=96, period=6):
    y, x = np.mgrid[0:n, 0:n]
    return (((x + y) // (period // 2)) % 2 * 200 + 20).astype(np.uint8)


def ink_on_paper(n=96, ink_frac=0.12):
    """Bright paper with sparse dark ink -- a text-like bimodal, left-skewed cell."""
    img = np.full((n, n), 210, np.uint8)
    m = rng.random((n, n)) < ink_frac
    img[m] = 30
    return img


# ==========================================================================
# tier 1 -- exactness against the maps imfeat already returns
# ==========================================================================
@pytest.mark.parametrize("img", [noise(), ramp(), stripes(), ink_on_paper()])
@pytest.mark.parametrize("grid", [[(0, 0)], [(3, 3)], [(4, 4), (2, 2)]])
def test_descriptors_match_their_formula(img, grid):
    f = imfeat.FeatureComputer(img.shape, grid=grid).features(img)
    for lv in [str(i) for i in range(len(grid))] + ["global"]:
        d = f[f"desc_{lv}"]
        var = f[f"mom_{lv}"][..., 1]
        m3, m4 = f[f"mom_{lv}"][..., 2], f[f"mom_{lv}"][..., 3]
        en = f[f"struct_{lv}"][..., 0]
        coh = f[f"struct_{lv}"][..., 1]
        hog = f[f"hog_{lv}"]
        ok = var > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            skew_ref, kurt_ref = m3 / var**1.5, m4 / var**2 - 3.0
        assert np.allclose(d[..., D["std_skew"]][ok], skew_ref[ok], atol=1e-4, rtol=1e-3)
        assert np.allclose(d[..., D["excess_kurt"]][ok], kurt_ref[ok], atol=1e-4, rtol=1e-3)
        # structure-tensor ratios
        assert np.allclose(d[..., D["edge_sharpness"]], en / (var + 1.0), atol=1e-3, rtol=1e-3)
        assert np.allclose(d[..., D["detail"]], en * (1.0 - coh), atol=1e-2, rtol=1e-3)
        # HOG shape (hog is already L1-normalised)
        assert np.allclose(d[..., D["hog_concentration"]], (hog**2).sum(-1), atol=1e-5)
        card = hog[..., 0] + hog[..., 4] + hog[..., 8]
        assert np.allclose(d[..., D["hog_cardinality"]], card, atol=1e-5)


def test_flat_cell_is_finite_and_zero():
    img = np.full((64, 64), 128, np.uint8)
    d = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).features(img)["desc_global"]
    assert np.all(np.isfinite(d)) and np.allclose(d, 0.0)


# ==========================================================================
# tier 2 -- closed-form known values
# ==========================================================================
def test_uniform_ramp_is_platykurtic():
    """A linear ramp visits every level equally -> uniform distribution, whose excess
    kurtosis is -1.2 and whose skew is 0."""
    f = imfeat.FeatureComputer((96, 96), grid=[(0, 0)]).features(ramp())
    d = f["desc_global"]
    assert abs(d[D["excess_kurt"]] - (-1.2)) < 0.05
    assert abs(d[D["std_skew"]]) < 0.02


def test_symmetric_noise_has_zero_skew():
    img = noise(160)
    d = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).features(img)["desc_global"]
    assert abs(d[D["std_skew"]]) < 0.05
    assert abs(d[D["excess_kurt"]] - (-1.2)) < 0.05  # uniform noise is also platykurtic


def test_ink_on_paper_is_left_skewed():
    """Sparse dark ink on bright paper: the long tail points to dark -> negative skew,
    and the distribution is far from Gaussian (positive excess kurtosis)."""
    img = ink_on_paper(160, 0.1)
    d = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).features(img)["desc_global"]
    assert d[D["std_skew"]] < -1.0
    assert d[D["excess_kurt"]] > 1.0


def test_single_orientation_has_high_concentration():
    """Vertical stripes -> every gradient is horizontal -> one HOG bin -> concentration
    near 1 (vs ~1/9 for isotropic noise)."""
    fv = imfeat.FeatureComputer((96, 96), grid=[(0, 0)]).features(stripes(axis=1))
    fn = imfeat.FeatureComputer((96, 96), grid=[(0, 0)]).features(noise(96))
    assert fv["desc_global"][D["hog_concentration"]] > 0.8
    assert fn["desc_global"][D["hog_concentration"]] < 0.2


def test_cardinality_separates_axis_aligned_from_diagonal():
    """Vertical and horizontal stripes are axis-aligned (high cardinality); 45-degree
    stripes put all their energy in a diagonal bin (low cardinality). Concentration is
    high for all three -- only cardinality tells them apart."""
    g = [(0, 0)]
    vert = imfeat.FeatureComputer((96, 96), grid=g).features(stripes(axis=1))["desc_global"]
    horz = imfeat.FeatureComputer((96, 96), grid=g).features(stripes(axis=0))["desc_global"]
    diag = imfeat.FeatureComputer((96, 96), grid=g).features(diagonal_stripes())["desc_global"]
    assert vert[D["hog_cardinality"]] > 0.8 and horz[D["hog_cardinality"]] > 0.8
    assert diag[D["hog_cardinality"]] < 0.2
    assert min(vert[D["hog_concentration"]], diag[D["hog_concentration"]]) > 0.7  # all peaked


def test_detail_high_for_isotropic_low_for_coherent():
    """detail = energy*(1-coherence): ~0 for a single coherent orientation (stripes),
    large for isotropic edge clutter (noise), 0 for a flat field."""
    g = [(0, 0)]
    strp = imfeat.FeatureComputer((96, 96), grid=g).features(stripes())["desc_global"]
    nois = imfeat.FeatureComputer((96, 96), grid=g).features(noise(96))["desc_global"]
    flat = imfeat.FeatureComputer((96, 96), grid=g).features(np.full((96, 96), 90, np.uint8))
    assert nois[D["detail"]] > 5 * strp[D["detail"]] + 1.0
    assert flat["desc_global"][D["detail"]] == 0.0


# ==========================================================================
# tier 3 -- affine invariance (the few-shot-transfer property)
# ==========================================================================
@pytest.mark.parametrize(
    "img", [noise(96, 0, 120), ink_on_paper(96), stripes(), diagonal_stripes()]
)
def test_dimensionless_descriptors_are_affine_invariant(img):
    """v -> 2v+15 (values kept <=255): the five dimensionless descriptors are unchanged
    to float precision. This is what lets the downstream model, trained on a handful of
    pixels, keep working under a contrast/brightness change."""
    base = (img.astype(np.int16) % 120).astype(np.uint8)  # headroom so 2v+15 <= 255
    warped = (2 * base + 15).astype(np.uint8)
    fc = imfeat.FeatureComputer(base.shape, grid=[(2, 2)])
    a = fc.features(base)["desc_0"]
    b = fc.features(warped)["desc_0"]
    for name in ("std_skew", "excess_kurt", "hog_concentration", "hog_cardinality"):
        assert np.allclose(a[..., D[name]], b[..., D[name]], atol=1e-4), name
    # edge_sharpness is dimensionless too, up to the +1 epsilon in the denominator
    assert np.allclose(
        a[..., D["edge_sharpness"]], b[..., D["edge_sharpness"]], rtol=0.02, atol=0.2
    )


def test_detail_is_not_affine_invariant():
    """The one magnitude-carrying descriptor: detail scales with contrast (energy ~ a^2),
    so doubling contrast roughly quadruples it. Documented, not a bug."""
    base = (noise(96) % 120).astype(np.uint8)
    warped = (2 * base + 15).astype(np.uint8)
    fc = imfeat.FeatureComputer(base.shape, grid=[(0, 0)])
    a = fc.features(base)["desc_global"][D["detail"]]
    b = fc.features(warped)["desc_global"][D["detail"]]
    assert b > 3.0 * a  # ~4x
