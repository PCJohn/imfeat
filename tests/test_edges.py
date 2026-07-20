"""Edge cases and robustness: image borders, degenerate grids, statefulness, input
layouts, overflow headroom, and API errors. These cover the parts of the core that
the exactness suites exercise only incidentally -- above all the border handling in
prepare_row, which is the one place the hot loop trusts blindly."""

import numpy as np
import pytest

import imfeat

rng = np.random.default_rng(3)


def moments(x):
    mu = x.mean()
    d = x - mu
    return np.array([mu, (d**2).mean(), (d**3).mean(), (d**4).mean()])


# ---------------- borders ----------------------------------------------------
def test_border_pixels_are_never_extrema():
    """Replicate padding puts a border pixel into its own 8-neighbourhood, so it can
    never be a STRICT max/min. The same pixel one step inside can."""
    for y, x in [(0, 0), (0, 128), (255, 255), (17, 0), (0, 200)]:
        img = np.full((256, 256), 100, np.uint8)
        img[y, x] = 255
        r = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)
        assert r["cnt_global"][0] == 0, f"border ({y},{x}) counted as a max"
    img = np.full((256, 256), 100, np.uint8)
    img[1, 1] = 255
    r = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)
    assert r["cnt_global"][0] == 1


def test_border_gradient_matches_replicate_padding():
    """gx/gy at the border must equal a Sobel on an edge-padded image (the numpy
    oracle in test_struct pins this on random images; here on a pure border edge)."""
    img = np.zeros((64, 64), np.uint8)
    img[:, 0] = 255  # an edge that only exists in the leftmost column
    r = imfeat.FeatureComputer(img.shape, grid=[(3, 3)]).compute(img)
    p = np.pad(img.astype(np.int64), 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (
        p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2]
    )
    assert r["struct_0"][..., 0].sum() == (gx * gx).sum()


def test_extrema_envelope_vs_bruteforce():
    """The prepared max/min envelopes are a rewrite of the naive 8-neighbour scan;
    check them against it on noise, where extrema are dense."""
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    r = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)
    p = np.pad(img.astype(int), 1, mode="edge")
    nb = np.stack(
        [
            p[i : i + 64, j : j + 64]
            for i in range(3)
            for j in range(3)
            if (i, j) != (1, 1)
        ]
    )
    assert r["cnt_global"][0] == int((img > nb.max(0)).sum())
    assert r["cnt_global"][1] == int((img < nb.min(0)).sum())


# ---------------- degenerate grids and shapes --------------------------------
def test_one_pixel_cells():
    """Finest grid == image size: one pixel per cell. Moments collapse to the pixel."""
    img = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    m = imfeat.FeatureComputer(img.shape, grid=[(3, 3)]).features(img)["mom_0"]
    np.testing.assert_allclose(m[..., 0], img.astype(float))
    assert np.all(m[..., 1:] == 0.0)  # a single sample has no spread


def test_non_square_image():
    img = rng.integers(0, 256, (128, 256), dtype=np.uint8)
    r = imfeat.FeatureComputer(img.shape, grid=[(3, 5), (2, 4)]).features(img)
    assert r["mom_0"].shape == (8, 32, 4) and r["mom_1"].shape == (4, 16, 4)
    np.testing.assert_allclose(
        r["mom_global"], moments(img.astype(float).ravel()), rtol=1e-9
    )


def test_single_cell_equals_global():
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    r = imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).features(img)
    for g in ("mom", "struct", "hog", "cnt"):
        assert np.array_equal(r[f"{g}_0"][0, 0], r[f"{g}_global"])


# ---------------- state, layout, determinism ---------------------------------
def test_no_state_leaks_between_frames():
    """Buffers are reused across calls; a frame must not see the previous one."""
    fc = imfeat.FeatureComputer((64, 64, 3), grid=[(3, 3), (1, 1)])
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    b = np.zeros((64, 64, 3), np.uint8)
    fa = {k: v.copy() for k, v in fc.features(a).items()}
    fc.features(b)
    for k, v in fc.features(a).items():
        assert np.array_equal(v, fa[k]), k  # a -> b -> a reproduces a exactly
    fc.features(b)
    z = fc.features(b)
    assert np.all(z["mom_0"] == 0.0) and np.all(z["struct_0"] == 0.0)


def test_non_contiguous_input():
    """A strided view (e.g. a crop, or BGR->RGB reversal) is read via its strides."""
    big = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    view = big[::2, ::2, ::-1]
    ref = np.ascontiguousarray(view)
    fc = imfeat.FeatureComputer(view.shape, grid=[(3, 3)])
    for k, v in fc.features(view).items():
        assert np.array_equal(v, fc.features(ref)[k]), k


@pytest.mark.parametrize("nch", [4, 5, 7, 8])
def test_simd_group_boundaries(nch):
    """Channels are processed in groups of 4 lanes; check the ragged tail."""
    img = rng.integers(0, 256, (64, 64, nch), dtype=np.uint8)
    r = imfeat.FeatureComputer(img.shape, grid=[(3, 3)]).features(img)
    for c in range(nch):
        plane = np.ascontiguousarray(img[:, :, c])
        solo = imfeat.FeatureComputer(plane.shape, grid=[(3, 3)]).features(plane)
        for k in solo:
            got = r[k][c] if r[k].ndim == 1 else r[k][..., c, :]
            assert np.array_equal(got, solo[k]), f"{k} ch{c}"


# ---------------- numeric headroom -------------------------------------------
def test_power_sum_headroom_on_a_large_frame():
    """S4 is the tightest sum: 255^4 * N. At 4 megapixels that is 1.8e16, well inside
    int64 -- and the derived moments must still be exact."""
    img = np.full((2048, 2048), 255, np.uint8)
    img[::2, ::2] = 0  # keep variance non-trivial
    r = imfeat.FeatureComputer(img.shape, grid=[(0, 0)])
    raw = r.compute(img)["mom_global"]
    x = img.astype(np.int64).ravel()
    assert raw[3] == (x**4).sum() and raw[3] < np.iinfo(np.int64).max // 2
    np.testing.assert_allclose(
        r.features(img)["mom_global"], moments(img.astype(float).ravel()), rtol=1e-12
    )


# ---------------- parity with the suites imfeat replaces ---------------------
def test_stride_none_equals_one():
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    a = imfeat.FeatureComputer(img.shape, grid=[(3, 3)], stride=None).features(img)
    b = imfeat.FeatureComputer(img.shape, grid=[(3, 3)], stride=1).features(img)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_empty_cell_is_zero():
    """A stride coarser than the cell height skips whole cell-rows. Those cells have
    no samples: every feature must be a finite zero, not a NaN from a 0/0."""
    img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    fc = imfeat.FeatureComputer(img.shape, grid=[(5, 5)], stride=(16, 1))  # 8px cells
    raw, f = fc.compute(img), fc.features(img)
    assert raw["struct_0"][0, 0, 3] > 0 and raw["struct_0"][1, 0, 3] == 0
    for g in ("mom", "struct", "hog", "cnt"):
        assert np.all(f[f"{g}_0"][1::2] == 0.0), g  # the skipped cell-rows
        assert np.isfinite(f[f"{g}_0"]).all(), g


def test_pyramid_equals_separate_computers():
    """One computer with K levels == K computers with one level each."""
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    grid = [(4, 4), (3, 3), (1, 1)]
    joint = imfeat.FeatureComputer(img.shape, grid=grid).features(img)
    for i, g in enumerate(grid):
        solo = imfeat.FeatureComputer(img.shape, grid=[g]).features(img)
        for grp in ("mom", "struct", "hog", "cnt"):
            assert np.array_equal(joint[f"{grp}_{i}"], solo[f"{grp}_0"]), f"{grp} lvl{i}"


@pytest.mark.parametrize("v", [0, 255])
def test_uniform_extremes(v):
    """Saturated inputs: no gradient, no extrema, no spread -- and no division blowups."""
    img = np.full((64, 64), v, np.uint8)
    f = imfeat.FeatureComputer(img.shape, grid=[(3, 3)]).features(img)
    assert np.all(f["mom_0"][..., 0] == v) and np.all(f["mom_0"][..., 1:] == 0.0)
    assert np.all(f["struct_0"] == 0.0) and np.all(f["cnt_0"] == 0.0)
    assert np.all(f["hog_0"] == 0.0) and np.isfinite(f["struct_0"]).all()


def test_gaussian_blob_localizes():
    """A compact blob: variance and edge energy both peak on its slope (not its
    flatter apex), within a cell of the centre, and its peak is one strict local max."""
    n, cy, cx = 128, 88, 40
    y, x = np.mgrid[0:n, 0:n]
    img = (255 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * 6.0**2)))).astype(np.uint8)
    f = imfeat.FeatureComputer(img.shape, grid=[(4, 4)]).features(img)  # 8px cells
    cell = np.array([cy * 16 // n, cx * 16 // n])
    for key, band in (("mom_0", 1), ("struct_0", 0)):  # variance, edge energy
        peak = np.array(np.unravel_index(np.argmax(f[key][..., band]), (16, 16)))
        assert np.abs(peak - cell).max() <= 1, f"{key} peaked at {peak}, blob at {cell}"
    assert imfeat.FeatureComputer(img.shape, grid=[(0, 0)]).compute(img)["cnt_global"][0] == 1


# ---------------- API errors --------------------------------------------------
def test_input_validation():
    fc = imfeat.FeatureComputer((32, 32), grid=[(2, 2)])
    with pytest.raises(ValueError):
        fc.features(np.zeros((16, 16), np.uint8))
    with pytest.raises(TypeError):
        fc.features(np.zeros((32, 32), np.float32))
    with pytest.raises(ValueError):
        imfeat.FeatureComputer((30, 30), grid=[(2, 2)])  # finest grid must divide
    with pytest.raises(ValueError):
        imfeat.FeatureComputer((32, 32), grid=[(2, 2), (3, 3)])  # not nested
    with pytest.raises(ValueError):
        imfeat.FeatureComputer((32, 32, 3), grid=[(2, 2)], channels=[3])
