"""LBP^riu2 and cross-channel products.

Three tiers, mirroring the existing suites:
  1. exactness   -- a numpy oracle with identical rules; bit-exact equality.
  2. closed form -- images whose LBP histogram is derivable by hand (flat, impulse,
                    step edge), so a wrong sign convention or neighbour ordering
                    cannot hide behind an oracle that shares the same bug.
  3. invariance  -- the two properties LBP^riu2 is *for*: monotonic grey-scale
                    invariance and rotation invariance. Neither holds for any other
                    feature in imfeat, which is the point of adding it.
"""

import time

import numpy as np
import pytest

import imfeat

rng = np.random.default_rng(7)

NB = 10  # LBP^riu2 bins
GRIDS = [[(0, 0)], [(3, 3)], [(5, 5), (4, 4), (2, 2)]]
STRIDES = [(1, 1), (2, 2), (3, 2)]


# ---------------- oracles ----------------------------------------------------
def ref_codes(img):
    """LBP^riu2_{8,1} code per pixel, replicate-padded. Neighbours in circular order."""
    p = np.pad(img.astype(np.int16), 1, mode="edge")
    c = p[1:-1, 1:-1]
    nb = [
        p[0:-2, 0:-2],  # NW
        p[0:-2, 1:-1],  # N
        p[0:-2, 2:],  # NE
        p[1:-1, 2:],  # E
        p[2:, 2:],  # SE
        p[2:, 1:-1],  # S
        p[2:, 0:-2],  # SW
        p[1:-1, 0:-2],  # W
    ]
    s = np.stack([(x >= c).astype(np.uint8) for x in nb])
    pc = s.sum(0)
    u = sum(s[i] ^ s[(i + 1) % 8] for i in range(8))
    return np.where(u <= 2, pc, NB - 1).astype(np.int64)


def _sample(h, w, grid, stride):
    """The pixels imfeat accumulates, and their cell index at each level."""
    sy, sx = stride
    cwf = w // (1 << grid[0][1])
    rows = np.arange(0, h, sy)
    cols = np.nonzero((np.arange(w) % cwf) % sx == 0)[0]
    return rows, cols


def ref_lbp(img, grid, stride):
    """Per-cell LBP bin counts, per level."""
    h, w = img.shape
    code = ref_codes(img)
    rows, cols = _sample(h, w, grid, stride)
    out = []
    for ky, kx in grid:
        nr, nc = 1 << ky, 1 << kx
        rr, cc = np.meshgrid((rows * nr) // h, (cols * nc) // w, indexing="ij")
        idx = (rr * nc + cc).ravel() * NB + code[np.ix_(rows, cols)].ravel()
        out.append(np.bincount(idx, minlength=nr * nc * NB).reshape(nr, nc, NB))
    return out


def ref_cross(img, grid, stride):
    """Per-cell sum(v_i * v_j) per channel pair, per level."""
    h, w, c = img.shape
    rows, cols = _sample(h, w, grid, stride)
    v = img[np.ix_(rows, cols)].astype(np.int64)
    pairs = [(i, j) for i in range(c) for j in range(i + 1, c)]
    out = []
    for ky, kx in grid:
        nr, nc = 1 << ky, 1 << kx
        rr, cc = np.meshgrid((rows * nr) // h, (cols * nc) // w, indexing="ij")
        idx = (rr * nc + cc).ravel()
        flat = np.zeros((nr * nc, len(pairs)), np.int64)
        for p, (i, j) in enumerate(pairs):
            np.add.at(flat[:, p], idx, (v[..., i] * v[..., j]).ravel())
        out.append(flat.reshape(nr, nc, len(pairs)))
    return out


# ---------------- images -----------------------------------------------------
def noise(n=64):
    return rng.integers(0, 256, (n, n), dtype=np.uint8)


def textured(n=64):
    y, x = np.mgrid[0:n, 0:n]
    im = 128 + 60 * np.sin(x / 5.0) * np.cos(y / 9.0) + rng.normal(0, 10, (n, n))
    return np.clip(im, 0, 255).astype(np.uint8)


def vstep(n=64, lo=0, hi=255):
    img = np.full((n, n), lo, np.uint8)
    img[:, n // 2 :] = hi
    return img


# ==========================================================================
# tier 1 -- exactness against the oracle
# ==========================================================================
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("img", [noise(), textured(), vstep(), np.full((64, 64), 7, np.uint8)])
def test_lbp_exact(img, grid, stride):
    r = imfeat.FeatureComputer(img.shape, grid=grid, stride=stride).compute(img)
    for i, ref in enumerate(ref_lbp(img, grid, stride)):
        assert np.array_equal(r[f"lbp_{i}"], ref)


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("stride", STRIDES)
def test_cross_exact(grid, stride):
    img = np.stack([textured(), noise(), vstep()], -1)
    r = imfeat.FeatureComputer(img.shape, grid=grid, stride=stride).compute(img)
    for i, ref in enumerate(ref_cross(img, grid, stride)):
        assert np.array_equal(r[f"xchan_{i}"], ref)


def test_bins_partition_the_pixels():
    """Every sampled pixel lands in exactly one bin, so the bins sum to the count."""
    img = np.stack([textured(), noise()], -1)
    r = imfeat.FeatureComputer(img.shape, grid=[(3, 3)], stride=2).compute(img)
    assert np.array_equal(r["lbp_0"].sum(-1), r["struct_0"][..., 3])
    f = imfeat.FeatureComputer(img.shape, grid=[(3, 3)], stride=2).features(img)
    assert np.allclose(f["lbp_0"].sum(-1), 1.0)


def test_pooling_is_additive():
    img = np.stack([textured(), noise(), vstep()], -1)
    r = imfeat.FeatureComputer(img.shape, grid=[(3, 3), (1, 1)]).compute(img)
    for k in ("lbp", "xchan"):
        fine = r[f"{k}_0"].reshape(2, 4, 2, 4, *r[f"{k}_0"].shape[2:])
        assert np.array_equal(fine.sum(axis=(1, 3)), r[f"{k}_1"])
        assert np.array_equal(r[f"{k}_1"].sum(axis=(0, 1)), r[f"{k}_global"])


# ==========================================================================
# tier 2 -- closed-form histograms
# ==========================================================================
def test_flat_is_all_ones_code():
    """Every neighbour equals the centre, so (p >= c) is 1 eightfold: code 255,
    zero transitions -> uniform, popcount 8. The whole image is bin 8."""
    img = np.full((64, 64), 200, np.uint8)
    r = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)
    assert r["lbp_global"][8] == 64 * 64
    assert r["lbp_global"].sum() == 64 * 64


def test_bright_impulse():
    """A lone bright pixel: all 8 neighbours are below it -> code 0 -> bin 0. Its
    neighbours still see it as >= themselves, so they stay bin 8."""
    img = np.full((64, 64), 100, np.uint8)
    img[32, 40] = 255
    b = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)["lbp_global"]
    assert b[0] == 1
    assert b[8] == 64 * 64 - 1


def test_dark_impulse():
    """The dark pixel itself is bin 8 (every neighbour is >= it). Each of its 8
    neighbours has exactly one zero bit -> 2 transitions -> uniform, popcount 7."""
    img = np.full((64, 64), 100, np.uint8)
    img[32, 40] = 0
    b = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)["lbp_global"]
    assert b[7] == 8
    assert b[8] == 64 * 64 - 8


def test_step_edge():
    """A vertical step. Only the first bright column sees dark pixels: its NW/W/SW
    bits are 0, three contiguous zeros -> 2 transitions -> uniform, popcount 5."""
    img = vstep(64)
    b = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)["lbp_global"]
    assert b[5] == 64
    assert b[8] == 64 * 64 - 64
    assert b.sum() == 64 * 64


def test_checkerboard_splits_uniform_and_not():
    """A checkerboard. A dark centre sees light orthogonals (>) and dark diagonals
    (==): all eight bits set -> bin 8. A light centre sees dark orthogonals: the ring
    is 10101010, eight transitions -> non-uniform. Exactly half the pixels each."""
    img = (np.indices((64, 64)).sum(0) % 2 * 255).astype(np.uint8)
    b = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)["lbp_global"]
    assert b[NB - 1] == 64 * 64 // 2
    assert b[8] == 64 * 64 // 2


# ==========================================================================
# tier 3 -- the invariances LBP^riu2 exists for
# ==========================================================================
def test_monotonic_greyscale_invariance():
    """Any strictly increasing intensity map leaves every sign test unchanged. No
    other imfeat feature has this property."""
    img = textured() // 2  # values in [0, 128) so the map below can be non-affine
    lut = np.sort(rng.choice(256, 128, replace=False)).astype(np.uint8)  # strictly increasing
    warped = lut[img]
    fc = imfeat.FeatureComputer(img.shape, grid=[(3, 3)])
    assert np.array_equal(fc.compute(img)["lbp_0"], fc.compute(warped)["lbp_0"])
    # the moments, by contrast, must move
    assert not np.allclose(fc.features(img)["mom_0"], fc.features(warped)["mom_0"])


def test_rotation_invariance():
    """riu2 keys on the popcount and the circular transition count, both invariant
    to rotating the neighbour ring -- so to 90-degree rotations and reflections."""
    img = textured()
    fc = imfeat.FeatureComputer(img.shape, grid=[(0, 0)])
    b = fc.compute(img)["lbp_global"]
    for t in (np.rot90(img, 1), np.rot90(img, 2), np.fliplr(img), img.T):
        t = np.ascontiguousarray(t)
        assert np.array_equal(fc.compute(t)["lbp_global"], b)


# ==========================================================================
# cross-channel semantics
# ==========================================================================
def test_cross_identical_and_negated_channels():
    v = textured()
    img = np.stack([v, v, 255 - v], -1)
    fc = imfeat.FeatureComputer(img.shape, grid=[(2, 2)])
    f = fc.features(img)
    assert fc.channel_pairs == [(0, 1), (0, 2), (1, 2)]
    cov, corr = f["xchan_0"][..., 0], f["xchan_0"][..., 1]
    assert np.allclose(corr[..., 0], 1.0, atol=1e-5)  # v vs v
    assert np.allclose(corr[..., 1], -1.0, atol=1e-5)  # v vs 255 - v
    assert np.allclose(cov[..., 0], f["mom_0"][..., 0, 1], rtol=1e-4)  # cov(v,v) == var


def test_cross_independent_channels_decorrelate():
    img = np.stack([noise(256), noise(256)], -1)
    f = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).features(img)
    assert abs(f["xchan_global"][0, 1]) < 0.02


def test_cross_matches_numpy_corrcoef():
    v = textured()
    img = np.stack([v, noise(), np.clip(v // 2 + 40, 0, 255).astype(np.uint8)], -1)
    f = imfeat.FeatureComputer(img.shape, grid=[(1, 1)]).features(img)
    for cy in range(2):
        for cx in range(2):
            blk = img[cy * 32 : cy * 32 + 32, cx * 32 : cx * 32 + 32].astype(float)
            for p, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
                c = np.corrcoef(blk[..., i].ravel(), blk[..., j].ravel())[0, 1]
                assert np.isclose(f["xchan_0"][cy, cx, p, 1], c, atol=1e-5)


def test_cross_constant_channel_is_zero_not_nan():
    img = np.stack([textured(), np.full((64, 64), 9, np.uint8)], -1)
    f = imfeat.FeatureComputer(img.shape, grid=[(2, 2)]).features(img)
    assert np.all(np.isfinite(f["xchan_0"]))
    assert np.allclose(f["xchan_0"][..., 1], 0.0)


@pytest.mark.parametrize("c", [1, 9])
def test_cross_absent_without_pairs(c):
    """One channel has no pairs; beyond XMAX the pair count would explode, so it is
    switched off rather than silently quadratic."""
    shape = (32, 32) if c == 1 else (32, 32, c)
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    fc = imfeat.FeatureComputer(img.shape, grid=[(2, 2)])
    assert fc.channel_pairs == []
    assert not any(k.startswith("xchan") for k in fc.features(img))


# ==========================================================================
# latency
# ==========================================================================
def test_latency_regression(capsys):
    """LBP and the cross products are meant to be free. Guard the regression, and
    print enough context to diagnose a slow machine from the log alone."""
    import platform

    img = np.stack([textured(256)] * 3, -1)
    pyr = [(5, 5), (4, 4), (3, 3), (2, 2)]
    fc = imfeat.FeatureComputer(img.shape, grid=pyr, stride=2)
    for _ in range(30):
        fc.features(img)
    ts = []
    for _ in range(100):
        t0 = time.perf_counter()
        fc.features(img)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    p50, p95 = ts[50] * 1e3, ts[95] * 1e3
    with capsys.disabled():
        print(
            f"\n  [imfeat-lat] {platform.machine()} {platform.system()} "
            f"py{platform.python_version()} | 256x256x3 4-level stride=2 features(): "
            f"p50={p50:.3f} ms p95={p95:.3f} ms min={ts[0] * 1e3:.3f} ms"
        )
    assert p50 < 5.0
