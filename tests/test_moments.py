"""Tests for the moment features of imfeat, mirroring the three tiers of
test_struct.py:

1. Exactness   -- per-cell central moments vs a numpy oracle over the identical
                  sampled pixel set, plus the additivity invariants the merged
                  design relies on (raw power sums pool across scales; global is
                  the grand total).
2. Semantics   -- synthetic images with known moments (constant, bimodal, ramp),
                  and precision on the low-variance/high-mean case where the
                  textbook S2/n - mean^2 form cancels catastrophically.
3. Consistency -- SIMD (multi-channel) vs scalar (single-channel) paths agree
                  bit-for-bit; channel selection and channel_axis are no-ops on
                  the values; latency is reported.
"""

import time

import numpy as np
import pytest

import imfeat

rng = np.random.default_rng(0)

GRIDS = [[(5, 5)], [(5, 5), (4, 4), (3, 3), (2, 2)], [(0, 0)], [(6, 6)]]
STRIDES = [(1, 1), (2, 2), (4, 4)]


def sampled(h, w, nx_fine, stride):
    """The pixels imfeat visits: rows stride globally, columns restart the stride
    at every finest-cell edge (so every cell keeps the same sample count)."""
    sy, sx = stride
    cw = w // nx_fine
    rows = np.arange(0, h, sy)
    cols = np.nonzero((np.arange(w) % cw) % sx == 0)[0]
    return rows, cols


def ref_moments(x):
    """[mean, var, m3, m4] of a flat float64 sample."""
    mu = x.mean()
    d = x - mu
    return np.array([mu, (d**2).mean(), (d**3).mean(), (d**4).mean()])


def ref_levels(img, grid, stride):
    """Per-cell moments for each level, from the same sampled pixels."""
    h, w = img.shape
    rows, cols = sampled(h, w, 1 << grid[0][1], stride)
    sub = img[np.ix_(rows, cols)].astype(np.float64)
    out = []
    for ky, kx in grid:
        nr, nc = 1 << ky, 1 << kx
        rc = ((np.arange(h) * nr) // h)[rows]
        cc = ((np.arange(w) * nc) // w)[cols]
        m = np.zeros((nr, nc, 4))
        for i in range(nr):
            for j in range(nc):
                sel = sub[np.ix_(rc == i, cc == j)].ravel()
                m[i, j] = ref_moments(sel) if sel.size else 0.0
        out.append(m)
    return out


def flat(v=128, n=256):
    return np.full((n, n), v, np.uint8)


def noise(n=256):
    return rng.integers(0, 256, (n, n), dtype=np.uint8)


def ramp(n=256):
    return np.tile(np.linspace(0, 255, n, dtype=np.uint8), (n, 1))


def textured(n=256):
    a = (127 + 80 * np.sin(np.mgrid[0:n, 0:n][1] / 18.0)).astype(np.uint8)
    a[40:80, 40:200] = 255
    a[120:200, 120:140] = 0
    return np.clip(a + rng.integers(-8, 8, (n, n)), 0, 255).astype(np.uint8)


# ============================== Tier 1: exactness =========================
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("stride", STRIDES)
def test_moments_vs_reference(grid, stride):
    img = textured()
    r = imfeat.FeatureComputer(img.shape, grid=grid, stride=stride).features(img)
    for i, ref in enumerate(ref_levels(img, grid, stride)):
        np.testing.assert_allclose(r[f"mom_{i}"], ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("stride", STRIDES)
def test_global_moments_vs_reference(stride):
    img = textured()
    grid = [(5, 5)]
    r = imfeat.FeatureComputer(img.shape, grid=grid, stride=stride).features(img)
    rows, cols = sampled(256, 256, 32, stride)
    ref = ref_moments(img[np.ix_(rows, cols)].astype(np.float64).ravel())
    np.testing.assert_allclose(r["mom_global"], ref, rtol=1e-9, atol=1e-9)


def test_power_sums_are_additive():
    """The merge rests on this: raw sums pool across scales, so pyramid depth is
    free and the image is read once."""
    grid = [(5, 5), (3, 3), (1, 1)]
    r = imfeat.FeatureComputer((256, 256), grid=grid).compute(textured())
    for i in range(len(grid) - 1):
        fine, coarse = r[f"mom_{i}"], r[f"mom_{i + 1}"]
        gy, gx = coarse.shape[:2]
        fy, fx = fine.shape[0] // gy, fine.shape[1] // gx
        pooled = fine.reshape(gy, fy, gx, fx, 4).sum(axis=(1, 3))
        assert np.array_equal(pooled, coarse), f"level {i}"
    assert np.array_equal(r["mom_global"], r["mom_0"].reshape(-1, 4).sum(0))


def test_raw_power_sums_exact():
    img = noise()
    r = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)
    x = img.astype(np.int64).ravel()
    ref = np.array([x.sum(), (x**2).sum(), (x**3).sum(), (x**4).sum()])
    assert np.array_equal(r["mom_global"], ref)


def test_derived_matches_raw():
    img = textured()
    fc = imfeat.FeatureComputer(img.shape, grid=[(5, 5)])
    raw, fe = fc.compute(img)["mom_0"].astype(float), fc.features(img)["mom_0"]
    n = fc.compute(img)["struct_0"][..., 3].astype(float)
    mu = raw[..., 0] / n
    np.testing.assert_allclose(fe[..., 0], mu, rtol=1e-12)
    np.testing.assert_allclose(fe[..., 1], raw[..., 1] / n - mu**2, rtol=1e-9)


# ============================== Tier 2: semantics =========================
def test_constant_image():
    r = imfeat.FeatureComputer((256, 256), grid=[(5, 5)]).features(flat(200))
    m = r["mom_0"]
    assert np.all(m[..., 0] == 200.0)
    assert np.all(m[..., 1:] == 0.0)


def test_bimodal_variance_and_symmetry():
    """Half black, half white: mean 127.5, var 127.5^2, zero skew, kurtosis 1."""
    img = flat(0)
    img[128:] = 255
    m = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).features(img)["mom_0"][0, 0]
    np.testing.assert_allclose(m[0], 127.5, rtol=1e-12)
    np.testing.assert_allclose(m[1], 127.5**2, rtol=1e-12)
    assert abs(m[2]) < 1e-6  # symmetric -> no skew
    np.testing.assert_allclose(m[3] / m[1] ** 2, 1.0, rtol=1e-12)  # two-point kurtosis


def test_skew_sign():
    """A few bright pixels on a dark field -> positive third moment."""
    img = flat(10)
    img[::32, ::32] = 250
    m = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).features(img)["mom_0"][0, 0]
    assert m[2] > 0


def test_low_variance_high_mean_precision():
    """Where S2/n - mean^2 cancels: values in {250, 251} have variance ~0.25 against
    a mean^2 of ~62750. The shifted power sums keep full precision."""
    img = (250 + rng.integers(0, 2, (256, 256))).astype(np.uint8)
    m = imfeat.FeatureComputer(img.shape, grid=[(5, 5)]).features(img)["mom_0"]
    ref = ref_levels(img, [(5, 5)], (1, 1))[0]
    np.testing.assert_allclose(m, ref, rtol=1e-12, atol=1e-12)


def test_ramp_cell_means_increase():
    m = imfeat.FeatureComputer((256, 256), grid=[(5, 5)]).features(ramp())["mom_0"]
    row = m[0, :, 0]
    assert np.all(np.diff(row) > 0)  # horizontal ramp -> means rise left to right
    assert np.allclose(m[:, :, 0], m[0, :, 0], atol=1e-9)  # rows identical


def test_stride_preserves_moments():
    """Stride is a quality/latency knob: the moment maps should barely move."""
    img = textured()
    g = [(5, 5)]
    a = imfeat.FeatureComputer(img.shape, grid=g).features(img)["mom_0"]
    b = imfeat.FeatureComputer(img.shape, grid=g, stride=(4, 4)).features(img)["mom_0"]
    for k, lo in ((0, 0.99), (1, 0.95)):  # mean, variance
        x, y = a[..., k].ravel(), b[..., k].ravel()
        assert np.corrcoef(x, y)[0, 1] > lo, f"moment {k}"
    assert np.abs(b[..., 0] - a[..., 0]).mean() < 4.0  # mean bias, grey levels


# ============================== Tier 3: consistency =======================
def test_multichannel_matches_per_channel():
    """Channels-in-lanes SIMD (C>=2) must agree bit-for-bit with the scalar path."""
    img = np.stack([textured(), noise(), ramp()], -1)
    multi = imfeat.FeatureComputer(img.shape, grid=[(5, 5), (3, 3)]).features(img)
    for c in range(3):
        plane = np.ascontiguousarray(img[:, :, c])
        solo = imfeat.FeatureComputer(plane.shape, grid=[(5, 5), (3, 3)]).features(
            plane
        )
        for k in solo:
            assert np.array_equal(multi[k][..., c, :], solo[k]), f"{k} ch{c}"


@pytest.mark.parametrize("nch", [1, 2, 3, 5, 8])
def test_arbitrary_channel_count(nch):
    img = rng.integers(0, 256, (128, 128, nch), dtype=np.uint8)
    r = imfeat.FeatureComputer(img.shape, grid=[(4, 4)]).features(img)
    assert r["mom_0"].shape == (16, 16, nch, 4)
    for c in range(nch):
        ref = ref_moments(img[:, :, c].astype(np.float64).ravel())
        np.testing.assert_allclose(r["mom_global"][c], ref, rtol=1e-9)


def test_channel_subset():
    img = np.stack([textured(), noise(), ramp()], -1)
    full = imfeat.FeatureComputer(img.shape, grid=[(5, 5)]).features(img)
    sub = imfeat.FeatureComputer(img.shape, grid=[(5, 5)], channels=[2, 0]).features(
        img
    )
    for k in full:
        if k.startswith("xchan"):  # cross-channel maps have no per-channel axis
            continue
        assert np.array_equal(sub[k][..., 0, :], full[k][..., 2, :]), k
        assert np.array_equal(sub[k][..., 1, :], full[k][..., 0, :]), k
    # selected channels [2, 0] -> the single pair (0,1) is full's pair (0,2)
    fc = imfeat.FeatureComputer(img.shape, grid=[(5, 5)], channels=[2, 0])
    assert fc.channel_pairs == [(0, 1)]
    pairs = imfeat.FeatureComputer(img.shape, grid=[(5, 5)]).channel_pairs
    assert np.allclose(sub["xchan_0"][:, :, 0], full["xchan_0"][:, :, pairs.index((0, 2))])


def test_channel_axis_first():
    img = np.stack([textured(), noise(), ramp()], -1)
    chw = np.ascontiguousarray(np.moveaxis(img, -1, 0))
    a = imfeat.FeatureComputer(img.shape, grid=[(5, 5)]).features(img)
    b = imfeat.FeatureComputer(chw.shape, grid=[(5, 5)], channel_axis=0).features(chw)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_latency():
    img = np.stack([textured(), noise(), ramp()], -1)
    fc = imfeat.FeatureComputer(img.shape, grid=[(5, 5), (4, 4), (3, 3), (2, 2)])
    fc.features(img)
    t0 = time.perf_counter()
    for _ in range(50):
        fc.features(img)
    ms = (time.perf_counter() - t0) / 50 * 1e3
    print(f"\n256x256x3, 4 levels, all features: {ms:.3f} ms/frame")
    assert ms < 20.0
