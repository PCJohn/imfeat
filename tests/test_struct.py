"""Tests for imfeat structure features, layered in three tiers (see README):

1. Exactness   -- bit-exact integer equality vs a numpy reference using the
                  identical Sobel kernel, plus structural invariants
                  (multi-scale consistency, global = grand total).
2. Semantics   -- synthetic images with known answers (flat / edge / corner /
                  noise / oriented line) checked through the derived features,
                  so we test that a channel *means* what we claim, not merely
                  that two implementations agree.
3. Quality/latency -- knob sweeps (stride) scored against the stride-1 output
                  with vector-aware metrics (energy correlation, circular
                  orientation error), an OpenCV sanity correlation, and a
                  reported per-call latency.
"""

import math
import time

import numpy as np
import pytest

import imfeat

import imfeat as ss

from conftest import groups

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------
# numpy reference derivations (independent oracle for the C++ features()).
# Input t is (..., 4) = [Sxx, Syy, Sxy, count].
# --------------------------------------------------------------------------
def ref_energy(t):  # raw trace
    return t[..., 0].astype(float) + t[..., 1]


def ref_coherence(t):
    sxx, syy, sxy = (t[..., i].astype(float) for i in range(3))
    tr = sxx + syy
    disc = np.sqrt((sxx - syy) ** 2 + 4.0 * sxy * sxy)
    return np.where(tr > 1e-12, disc / (tr + 1e-12), 0.0)


def ref_orientation(t):  # dominant gradient angle in (-pi/2, pi/2]
    sxx, syy, sxy = (t[..., i].astype(float) for i in range(3))
    return 0.5 * np.arctan2(2.0 * sxy, sxx - syy)


def ref_cornerness(t):  # Shi-Tomasi lambda_min (raw)
    sxx, syy, sxy = (t[..., i].astype(float) for i in range(3))
    tr = sxx + syy
    disc = np.sqrt(((sxx - syy) * 0.5) ** 2 + sxy * sxy)
    return tr * 0.5 - disc


# --------------------------------------------------------------------------
# numpy reference (identical integer Sobel, replicate border, floor cells)
# --------------------------------------------------------------------------
def sobel_int(img):
    p = np.pad(img.astype(np.int64), 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (
        p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2]
    )
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (
        p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]
    )
    return gx, gy


def ref_levels(img, grid, stride):
    h, w = img.shape
    gx, gy = sobel_int(img)
    sxx, syy, sxy = gx * gx, gy * gy, gx * gy
    sy, sx = stride
    # columns reset the stride at each *finest* cell edge; rows stride globally.
    cwf = w // (1 << grid[0][1])
    rows = np.arange(0, h, sy)
    cols = np.nonzero((np.arange(w) % cwf) % sx == 0)[0]
    out = []
    for ky, kx in grid:
        nr, nc = 1 << ky, 1 << kx
        rc = (np.arange(h) * nr) // h
        cc = (np.arange(w) * nc) // w
        rr, ccp = np.meshgrid(rc[rows], cc[cols], indexing="ij")
        idx = (rr * nc + ccp).ravel()
        flat = np.zeros((nr * nc, 4), np.int64)
        sel = np.ix_(rows, cols)
        np.add.at(flat[:, 0], idx, sxx[sel].ravel())
        np.add.at(flat[:, 1], idx, syy[sel].ravel())
        np.add.at(flat[:, 2], idx, sxy[sel].ravel())
        np.add.at(flat[:, 3], idx, np.ones(idx.size, np.int64))
        out.append(flat.reshape(nr, nc, 4))
    return out


# --------------------------------------------------------------------------
# synthetic image builders (uint8)
# --------------------------------------------------------------------------
def flat(v=128, n=256):
    return np.full((n, n), v, np.uint8)


def hedge(n=256):  # horizontal edge -> vertical gradient
    a = np.zeros((n, n), np.uint8)
    a[n // 2 :] = 255
    return a


def vedge(n=256):
    return hedge(n).T.copy()


def diag(n=256):  # lower triangle white -> 45-deg edge
    a = np.zeros((n, n), np.uint8)
    yy, xx = np.mgrid[0:n, 0:n]
    a[yy > xx] = 255
    return a


def checker(n=256, c=16):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.where(((yy // c) + (xx // c)) % 2 == 0, 0, 255).astype(np.uint8)


def noise(n=256):
    return rng.integers(0, 256, (n, n), dtype=np.uint8)


def textured(n=256):  # mixed content for correlation tests
    a = (127 + 80 * np.sin(np.mgrid[0:n, 0:n][1] / 18.0)).astype(np.uint8)
    a[40:80, 40:200] = 255
    a[120:200, 120:140] = 0
    return np.clip(a + rng.integers(-8, 8, (n, n)), 0, 255).astype(np.uint8)


GRIDS = [[(5, 5)], [(5, 5), (4, 4), (3, 3), (2, 2)], [(0, 0)], [(6, 6)]]
STRIDES = [(1, 1), (2, 2), (4, 4)]


# ============================== Tier 1: exactness =========================
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("img", [flat(), hedge(), vedge(), diag(), checker(), noise()])
def test_exact_vs_reference(img, grid, stride):
    sc = ss.FeatureComputer(img.shape, grid=grid, stride=stride)
    r = sc.compute(img)
    ref = ref_levels(img, grid, stride)
    for i, t in enumerate(ref):
        assert np.array_equal(r[f"struct_{i}"], t), f"level {i} mismatch"


def test_flat_is_zero():
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5)])
    t = sc.compute(flat())["struct_0"]
    assert t[..., :3].sum() == 0  # no gradient anywhere
    assert (t[..., 3] > 0).all()  # counts populated


def test_multiscale_consistency():
    # coarse cell sums == block-sum of finer cells (dyadic, divisible thumb)
    grid = [(5, 5), (4, 4), (3, 3)]
    sc = ss.FeatureComputer((256, 256), grid=grid)
    r = sc.compute(textured())
    for i in range(len(grid) - 1):
        fine, coarse = r[f"struct_{i}"], r[f"struct_{i + 1}"]
        gy, gx = coarse.shape[:2]
        fy, fx = fine.shape[0] // gy, fine.shape[1] // gx
        pooled = fine.reshape(gy, fy, gx, fx, 4).sum(axis=(1, 3))
        assert np.array_equal(pooled, coarse)


def test_global_equals_total():
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5)])
    r = sc.compute(textured())
    assert np.array_equal(r["struct_global"], r["struct_0"].reshape(-1, 4).sum(0))


# ============================== Tier 2: semantics =========================
def _interior(m):  # drop border cells (clamped border distorts them)
    return m[1:-1, 1:-1]


def test_horizontal_edge():
    t = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(hedge())["struct_0"]
    ori = ref_orientation(t)
    coh = ref_coherence(t)
    edge = _interior(t[..., 1]) > 0  # cells straddling the edge: strong Syy
    assert edge.any()
    # gradient is vertical -> |orientation| ~ pi/2
    assert np.allclose(np.abs(ori[1:-1, 1:-1][edge]), math.pi / 2, atol=0.05)
    assert (coh[1:-1, 1:-1][edge] > 0.9).all()


def test_vertical_edge():
    t = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(vedge())["struct_0"]
    ori = ref_orientation(t)
    coh = ref_coherence(t)
    edge = _interior(t[..., 0]) > 0
    assert edge.any()
    assert np.allclose(ori[1:-1, 1:-1][edge], 0.0, atol=0.05)  # horizontal gradient
    assert (coh[1:-1, 1:-1][edge] > 0.9).all()


def test_diagonal_edge():
    t = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(diag())["struct_0"]
    ori = ref_orientation(t)
    energy = t[..., 0] + t[..., 1]
    on = _interior(energy) > np.percentile(energy, 90)
    vals = np.abs(ori[1:-1, 1:-1][on])
    assert np.allclose(vals, math.pi / 4, atol=0.1)


def test_corner_vs_edge():
    # checkerboard corners: both eigenvalues large -> low coherence, high min-eig
    t = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(checker())["struct_0"]
    lmin = ref_cornerness(t)
    coh = ref_coherence(t)
    he = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(hedge())["struct_0"]
    assert _interior(lmin).max() > 0
    # a straight edge has near-zero min eigenvalue; corners do not
    assert _interior(ref_cornerness(he)).max() < 0.05 * _interior(lmin).max()
    assert _interior(coh).mean() < 0.9  # corners reduce coherence


def test_noise_is_isotropic():
    t = ss.FeatureComputer((256, 256), grid=[(4, 4)]).compute(noise())["struct_0"]
    assert _interior(ref_coherence(t)).mean() < 0.3  # no dominant orientation
    assert _interior(t[..., 0] + t[..., 1]).mean() > 0  # but high energy


def test_rotation_equivariance():
    # A 90-deg rotation swaps Gx<->Gy and flips the sign of Gx*Gy, so on the
    # structure tensor: Sxx<->Syy, Sxy->-Sxy (exact, integer). Hence energy,
    # coherence and cornerness are invariant, and the orientation vector negates
    # (theta -> theta +/- pi/2, i.e. 2theta -> 2theta +/- pi).
    n = 256
    img = (127 + 100 * np.sin(np.mgrid[0:n, 0:n][1] / 5.0)).astype(np.uint8)  # vertical
    sc = ss.FeatureComputer((n, n), grid=[(5, 5)])
    a = sc.compute(img)["struct_global"]
    b = sc.compute(np.ascontiguousarray(np.rot90(img)))["struct_global"]
    assert b[0] == a[1] and b[1] == a[0] and b[2] == -a[2] and b[3] == a[3]

    fa = groups(sc, img)["struct_global"]
    fb = groups(sc, np.ascontiguousarray(np.rot90(img)))["struct_global"]
    assert fa[1] > 0.5  # strongly anisotropic, so the checks below are meaningful
    assert np.allclose(fa[0], fb[0])  # energy invariant
    assert np.allclose(fa[1], fb[1])  # coherence invariant
    assert np.allclose(fa[4], fb[4])  # cornerness invariant
    assert np.allclose(fa[2], -fb[2], atol=1e-6)  # ori_cos negates
    assert np.allclose(fa[3], -fb[3], atol=1e-6)  # ori_sin negates


# ============================== Tier 3: quality + latency =================
def _circ_orient_err(a, b):  # orientation is mod pi
    d = 2 * (a - b)
    return np.abs(np.arctan2(np.sin(d), np.cos(d))) / 2.0


@pytest.mark.parametrize("stride", [(2, 2), (4, 4)])
def test_stride_quality(stride):
    # noise-free so this measures stride degradation, not noise decorrelation
    # (raw per-cell energy is noise-sensitive at small cells; orientation is not)
    n = 256
    img = (127 + 80 * np.sin(np.mgrid[0:n, 0:n][1] / 12.0)).astype(np.uint8)
    img[40:90, 40:210] = 255
    img[130:210, 110:140] = 0
    g = [(5, 5)]
    t1 = ss.FeatureComputer(img.shape, grid=g).compute(img)["struct_0"]
    ts = ss.FeatureComputer(img.shape, grid=g, stride=stride).compute(img)["struct_0"]
    e1 = (t1[..., 0] + t1[..., 1]).ravel().astype(float)
    es = (ts[..., 0] + ts[..., 1]).ravel().astype(float)
    corr = np.corrcoef(e1, es)[0, 1]
    strong = e1 > np.percentile(e1, 75)  # orientation only meaningful where energy is
    oerr = _circ_orient_err(
        ref_orientation(t1).ravel()[strong], ref_orientation(ts).ravel()[strong]
    )
    med_deg = math.degrees(np.median(oerr))
    print(
        f"\n  stride {stride}: energy corr={corr:.3f}  median orient err={med_deg:.2f} deg"
    )
    # orientation is the robust channel and gates tightly; raw energy aliases on
    # sharp edges under aggressive stride (would need INTER_AREA prefiltering),
    # so its floor loosens with stride -- documented, not silently tolerated.
    corr_min = 0.85 if stride == (2, 2) else 0.3
    assert corr > corr_min
    assert med_deg < 3.0


def test_opencv_sanity():
    cv2 = pytest.importorskip("cv2")
    img = textured()
    g = 32
    t = ss.FeatureComputer(img.shape, grid=[(5, 5)]).compute(img)["struct_0"]
    # cv2 per-pixel structure-tensor components, box-pooled to the same grid
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    comp = np.stack([gx * gx, gy * gy, gx * gy], -1)
    pooled = comp.reshape(g, img.shape[0] // g, g, img.shape[1] // g, 3).sum((1, 3))
    ours = t[..., :3].astype(float)
    for k in range(3):
        c = np.corrcoef(
            _interior(ours[..., k]).ravel(), _interior(pooled[..., k]).ravel()
        )[0, 1]
        assert c > 0.99, f"component {k} corr={c:.3f}"


def test_zero_copy_strided_channel():
    # framegate passes hsv[:, :, 2]: a non-contiguous 2-D view. Result must match
    # the contiguous copy exactly, with no ascontiguousarray in the path.
    img = textured()
    hsv = np.zeros((256, 256, 3), np.uint8)
    hsv[:, :, 2] = img
    v = hsv[:, :, 2]
    assert not v.flags["C_CONTIGUOUS"]
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5), (4, 4)])
    a, b = sc.compute(img), sc.compute(v)
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_cpp_features_match_python():
    img = textured()
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5)])
    raw = sc.compute(img)["struct_0"]
    fe = groups(sc, img)["struct_0"]
    n = np.maximum(raw[..., 3].astype(float), 1.0)
    coh, ori = ref_coherence(raw), ref_orientation(raw)
    assert ss.FEATURES == ("energy", "coherence", "ori_cos", "ori_sin", "cornerness")
    assert np.allclose(fe[..., 0], ref_energy(raw) / n, rtol=1e-5)
    assert np.allclose(fe[..., 1], coh, atol=1e-5)
    assert np.allclose(fe[..., 2], coh * np.cos(2 * ori), atol=1e-5)
    assert np.allclose(fe[..., 3], coh * np.sin(2 * ori), atol=1e-5)
    assert np.allclose(fe[..., 4], ref_cornerness(raw) / n, atol=1e-4)


def test_latency_end_to_end(capsys):
    # full path as framegate would call it: strided V channel in, float maps out
    img = textured()
    hsv = np.zeros((256, 256, 3), np.uint8)
    hsv[:, :, 2] = img
    v = hsv[:, :, 2]
    pyr = [(5, 5), (4, 4), (3, 3), (2, 2)]
    sc = ss.FeatureComputer(img.shape, grid=pyr)

    def feats():
        return groups(sc, v)

    for _ in range(20):
        feats()
    best = []
    for _ in range(50):
        t0 = time.perf_counter()
        feats()
        best.append(time.perf_counter() - t0)
    best.sort()
    p50 = best[len(best) // 2] * 1e3
    with capsys.disabled():
        print(
            f"\n  end-to-end (strided V -> 4-level float feature maps): "
            f"p50={p50:.3f} ms, min={best[0] * 1e3:.3f} ms"
        )
    assert p50 < 5.0
    img = textured()
    pyr = [(5, 5), (4, 4), (3, 3), (2, 2)]
    sc = ss.FeatureComputer(img.shape, grid=pyr)
    for _ in range(20):
        sc.compute(img)  # warmup
    best = []
    for _ in range(50):
        t0 = time.perf_counter()
        sc.compute(img)
        best.append(time.perf_counter() - t0)
    best.sort()
    p50 = best[len(best) // 2] * 1e3
    with capsys.disabled():
        print(
            f"\n  latency 256x256 4-level pyramid: p50={p50:.3f} ms, min={best[0] * 1e3:.3f} ms"
        )
    assert p50 < 5.0  # generous; real target is sub-ms


# ==========================================================================
# HOG + extrema (new features).  References mirror the C++ exactly so tier-1
# is bit-exact; semantics/cv2 tiers follow the existing pattern.
# ==========================================================================
HB = 9
RAYSCALE = 1 << 14
RAYS = [
    (
        round(RAYSCALE * math.cos(math.pi * k / HB)),
        round(RAYSCALE * math.sin(math.pi * k / HB)),
    )
    for k in range(HB)
]


def ref_hog_extrema(img, grid, stride):
    """numpy oracle for the HOG bins and extrema counts (identical rules)."""
    h, w = img.shape
    gx, gy = sobel_int(img)
    gxx, gyy = gx.copy(), gy.copy()
    neg = (gyy < 0) | ((gyy == 0) & (gxx < 0))  # fold to unsigned orientation [0,pi)
    gxx[neg] = -gxx[neg]
    gyy[neg] = -gyy[neg]
    b = np.zeros((h, w), np.int64)
    for k in range(1, HB):
        cx, cy = RAYS[k]
        b += (cx * gyy - cy * gxx >= 0).astype(np.int64)
    wgt = gx**2 + gy**2  # squared-magnitude weight
    # strict 8-neighbour extrema, replicate border
    p = np.pad(img.astype(np.int64), 1, mode="edge")
    nbrs = [
        p[dy : dy + h, dx : dx + w]
        for dy in range(3)
        for dx in range(3)
        if (dy, dx) != (1, 1)
    ]
    mx, mn = np.maximum.reduce(nbrs), np.minimum.reduce(nbrs)
    lmax, lmin = (img.astype(np.int64) > mx), (img.astype(np.int64) < mn)
    sy, sx = stride
    cwf = w // (1 << grid[0][1])  # per-cell column stride reset (see ref_levels)
    rows = np.arange(0, h, sy)
    cols = np.nonzero((np.arange(w) % cwf) % sx == 0)[0]
    sel = np.ix_(rows, cols)
    outh, outc = [], []
    for ky, kx in grid:
        nr, nc = 1 << ky, 1 << kx
        rc, cc = (np.arange(h) * nr) // h, (np.arange(w) * nc) // w
        rr, ccp = np.meshgrid(rc[rows], cc[cols], indexing="ij")
        idx = (rr * nc + ccp).ravel()
        hog = np.zeros((nr * nc, HB), np.int64)
        np.add.at(hog, (idx, b[sel].ravel()), wgt[sel].ravel())
        cnt = np.zeros((nr * nc, 2), np.int64)
        np.add.at(cnt[:, 0], idx, lmax[sel].ravel().astype(np.int64))
        np.add.at(cnt[:, 1], idx, lmin[sel].ravel().astype(np.int64))
        outh.append(hog.reshape(nr, nc, HB))
        outc.append(cnt.reshape(nr, nc, 2))
    return outh, outc


def ring(n=256, r=90, t=6):
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2)
    a = np.zeros((n, n), np.uint8)
    a[(d > r - t) & (d < r + t)] = 255
    return a


def impulses(n=256, step=16, bg=128):
    a = np.full((n, n), bg, np.uint8)
    pts = [
        (y, x) for y in range(step, n - step, step) for x in range(step, n - step, step)
    ]
    half = len(pts) // 2
    for y, x in pts[:half]:
        a[y, x] = 255  # strict local max
    for y, x in pts[half:]:
        a[y, x] = 0  # strict local min
    return a, half, len(pts) - half


# ------------------------------ tier 1: exact ------------------------------
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize(
    "img", [flat(), hedge(), vedge(), diag(), checker(), noise(), ring()]
)
def test_hog_extrema_exact(img, grid, stride):
    sc = ss.FeatureComputer(img.shape, grid=grid, stride=stride)
    r = sc.compute(img)
    refh, refc = ref_hog_extrema(img, grid, stride)
    for i in range(len(grid)):
        assert np.array_equal(r[f"hog_{i}"], refh[i]), f"hog level {i}"
        assert np.array_equal(r[f"cnt_{i}"], refc[i]), f"cnt level {i}"


def test_hog_extrema_multiscale_pool():
    grid = [(5, 5), (4, 4), (3, 3)]
    r = ss.FeatureComputer((256, 256), grid=grid).compute(textured())
    for i in range(len(grid) - 1):
        for key, k in (("hog", HB), ("cnt", 2)):
            fine, coarse = r[f"{key}_{i}"], r[f"{key}_{i + 1}"]
            gy, gx = coarse.shape[:2]
            fy, fx = fine.shape[0] // gy, fine.shape[1] // gx
            pooled = fine.reshape(gy, fy, gx, fx, k).sum(axis=(1, 3))
            assert np.array_equal(pooled, coarse)


def test_hog_extrema_global_total():
    r = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(textured())
    assert np.array_equal(r["hog_global"], r["hog_0"].reshape(-1, HB).sum(0))
    assert np.array_equal(r["cnt_global"], r["cnt_0"].reshape(-1, 2).sum(0))


# ------------------------------ tier 2: semantics --------------------------
def test_hog_axis_edges():
    hg = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(hedge())["hog_global"]
    vg = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(vedge())["hog_global"]
    assert hg.argmax() == 4  # horizontal edge -> vertical gradient -> ~90 deg (bin 4)
    assert vg.argmax() == 0  # vertical edge -> horizontal gradient -> ~0 deg (bin 0)


def test_hog_noise_isotropic():
    hg = (
        ss.FeatureComputer((256, 256), grid=[(4, 4)])
        .compute(noise())["hog_global"]
        .astype(float)
    )
    p = hg / hg.sum()
    assert (p > 0).all()
    assert p.max() / p.min() < 1.6  # no dominant orientation


def test_hog_ring_isotropic():
    hg = (
        ss.FeatureComputer((256, 256), grid=[(5, 5)])
        .compute(ring())["hog_global"]
        .astype(float)
    )
    p = hg / hg.sum()
    assert (p > 0).all()  # a circle spans every orientation
    assert p.max() / p.min() < 8  # far flatter than a single edge (min -> 0)


def test_extrema_flat_zero():
    r = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(flat())
    assert r["cnt_global"].sum() == 0


def test_extrema_impulse_count():
    img, nmax, nmin = impulses()
    r = ss.FeatureComputer(img.shape, grid=[(5, 5)]).compute(img)
    assert r["cnt_global"][0] == nmax  # isolated bright pixels
    assert r["cnt_global"][1] == nmin  # isolated dark pixels


def test_extrema_noise_positive():
    c = ss.FeatureComputer((256, 256), grid=[(4, 4)]).compute(noise())["cnt_global"]
    assert c[0] > 0 and c[1] > 0


# ------------------------------ tier 3: cv2 + features ---------------------
def test_hog_opencv_sanity():
    cv2 = pytest.importorskip("cv2")
    img = textured()
    g = 32
    hog = (
        ss.FeatureComputer(img.shape, grid=[(5, 5)]).compute(img)["hog_0"].astype(float)
    )
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    ang = np.mod(np.arctan2(gy, gx), math.pi)  # unsigned
    binf = np.minimum((ang / (math.pi / HB)).astype(int), HB - 1)
    mag2 = gx**2 + gy**2
    ref = np.zeros((img.shape[0], img.shape[1], HB))
    for bb in range(HB):
        ref[..., bb] = np.where(binf == bb, mag2, 0.0)
    pooled = ref.reshape(g, img.shape[0] // g, g, img.shape[1] // g, HB).sum((1, 3))
    c = np.corrcoef(_interior(hog).ravel(), _interior(pooled).ravel())[0, 1]
    assert c > 0.95, f"hog corr={c:.3f}"


def test_hog_extrema_features_match():
    img = textured()
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5)])
    raw = sc.compute(img)
    fe = groups(sc, img)
    hraw = raw["hog_0"].astype(float)
    hsum = np.maximum(hraw.sum(-1, keepdims=True), 1.0)
    assert np.allclose(fe["hog_0"], hraw / hsum, atol=1e-5)
    n = np.maximum(raw["struct_0"][..., 3].astype(float), 1.0)
    assert np.allclose(fe["cnt_0"][..., 0], raw["cnt_0"][..., 0] / n, atol=1e-5)
    assert np.allclose(fe["cnt_0"][..., 1], raw["cnt_0"][..., 1] / n, atol=1e-5)


# ==========================================================================
# Extended edge-case coverage: rectangular/small shapes, symmetry invariances
# (bit-exact), cross-group identities, known-output micro-images, degenerate
# configs, and config-error handling.
# ==========================================================================
def _c(x):
    return np.ascontiguousarray(x)


def ramp_h(n=256):  # value == column: constant horizontal gradient
    return np.tile(np.arange(n, dtype=np.uint8), (n, 1))


def ramp_v(n=256):
    return _c(ramp_h(n).T)


def solid_square(n=256, bg=50, fg=200):  # a flat plateau -> no strict extrema
    a = np.full((n, n), bg, np.uint8)
    a[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = fg
    return a


def single_impulse(n=256, y=128, x=140, bg=100, v=255):
    a = np.full((n, n), bg, np.uint8)
    a[y, x] = v
    return a


ALL_IMAGES = [flat(), hedge(), vedge(), diag(), checker(), noise(), ring(), textured()]


# ---------------- rectangular / non-square / small, full exactness ----------
RECT_CASES = [
    ((128, 256), [(4, 5), (2, 3)]),
    ((192, 320), [(4, 5)]),
    ((64, 128), [(5, 6), (3, 4), (1, 2)]),
    ((96, 96), [(5, 5), (0, 0)]),
    ((16, 16), [(4, 4), (2, 2), (0, 0)]),
    ((8, 8), [(3, 3), (1, 1)]),
    ((32, 96), [(4, 5), (2, 3)]),
]


@pytest.mark.parametrize("shape,grid", RECT_CASES)
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (1, 3), (3, 1), (2, 4)])
def test_exact_rectangular(shape, grid, stride):
    rng2 = np.random.default_rng(
        abs(hash((shape, tuple(map(tuple, grid)), stride))) % 2**32
    )
    img = rng2.integers(0, 256, shape, np.uint8)
    sc = ss.FeatureComputer(shape, grid=grid, stride=stride)
    r = sc.compute(img)
    lv = ref_levels(img, grid, stride)
    refh, refc = ref_hog_extrema(img, grid, stride)
    for i in range(len(grid)):
        assert np.array_equal(
            r[f"struct_{i}"], lv[i]
        ), f"struct L{i} {shape} {grid} {stride}"
        assert np.array_equal(r[f"hog_{i}"], refh[i]), f"hog L{i}"
        assert np.array_equal(r[f"cnt_{i}"], refc[i]), f"cnt L{i}"


def test_non_square_cells_pool():
    grid = [(5, 3), (4, 2), (2, 1)]  # ny != nx, still nested
    r = ss.FeatureComputer((256, 256), grid=grid).compute(textured())
    for i in range(len(grid) - 1):
        for key, k in (("struct", 4), ("hog", HB), ("cnt", 2)):
            fine, coarse = r[f"{key}_{i}"], r[f"{key}_{i + 1}"]
            gy, gx = coarse.shape[:2]
            fy, fx = fine.shape[0] // gy, fine.shape[1] // gx
            pooled = fine.reshape(gy, fy, gx, fx, k).sum(axis=(1, 3))
            assert np.array_equal(pooled, coarse), f"{key} L{i}"


# ---------------- cross-group identity: sum(HOG) == tensor energy -----------
@pytest.mark.parametrize("img", ALL_IMAGES)
@pytest.mark.parametrize("stride", STRIDES)
def test_hog_sum_equals_energy(img, stride):
    grid = [(5, 5), (3, 3)]
    r = ss.FeatureComputer(img.shape, grid=grid, stride=stride).compute(img)
    for i in range(len(grid)):
        energy = r[f"struct_{i}"][..., 0] + r[f"struct_{i}"][..., 1]
        assert np.array_equal(
            r[f"hog_{i}"].sum(-1), energy
        )  # every g^2 lands in one bin
    eg = r["struct_global"] if "struct_global" in r else r["struct_global"]
    assert np.array_equal(r["hog_global"].sum(), eg[0] + eg[1])


# ---------------- symmetry invariances (bit-exact, stride 1) ----------------
def test_rot180_full_invariance():
    img = textured()
    grid = [(5, 5), (3, 3)]
    sc = ss.FeatureComputer(img.shape, grid=grid)
    a = sc.compute(img)
    b = sc.compute(_c(np.rot90(img, 2)))
    # global: struct/HOG/extrema are all exactly invariant under a 180 deg turn
    for key in ("struct_global", "hog_global", "cnt_global"):
        assert np.array_equal(a[key], b[key]), key
    # per-cell: the map is a spatial flip of the cell grid (per-pixel values invariant)
    for i in range(len(grid)):
        for key in ("struct", "hog", "cnt"):
            assert np.array_equal(
                b[f"{key}_{i}"], a[f"{key}_{i}"][::-1, ::-1]
            ), f"{key}_{i}"


def test_extrema_flip_transpose_invariant():
    img = textured()
    sc = ss.FeatureComputer(img.shape, grid=[(5, 5)])
    base = sc.compute(img)["cnt_global"]
    for t in (img[:, ::-1], img[::-1, :], img.T):  # h-flip, v-flip, transpose
        c = sc.compute(_c(t))["cnt_global"]
        # transpose swaps max/min? no: it preserves values, so counts are identical
        assert np.array_equal(
            sorted(c), sorted(base)
        ), "extrema counts must be flip/transpose invariant"
    # h-flip and v-flip preserve which pixel is max/min exactly (no swap)
    for t in (img[:, ::-1], img[::-1, :]):
        assert np.array_equal(sc.compute(_c(t))["cnt_global"], base)


def test_transpose_swaps_struct_axes():
    img = textured()
    sc = ss.FeatureComputer(img.shape, grid=[(5, 5)])
    a = sc.compute(img)["struct_global"]
    b = sc.compute(_c(img.T))["struct_global"]
    assert a[0] == b[1] and a[1] == b[0]  # Sxx <-> Syy
    assert a[2] == b[2]  # Sxy = Gx*Gy unchanged under transpose


def test_dc_offset_invariant():
    # gradients ignore a global brightness shift (as long as nothing clips)
    rng2 = np.random.default_rng(7)
    img = rng2.integers(40, 200, (256, 256), np.uint8)
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5), (3, 3)], stride=(2, 2))
    a = sc.compute(img)
    b = sc.compute((img.astype(np.int16) + 30).astype(np.uint8))
    for k in a:
        if k.startswith("mom"):
            continue  # moments are not DC-invariant (the mean shifts); see test_moments
        assert np.array_equal(a[k], b[k]), k


# ---------------- known-output micro semantics -----------------------------
def test_ramp_single_orientation_no_extrema():
    rh = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(ramp_h())
    assert rh["hog_global"].argmax() == 0  # constant horizontal gradient
    assert rh["cnt_global"].sum() == 0  # monotonic -> no strict extrema
    assert ref_coherence(rh["struct_global"][None])[0] > 0.99
    rv = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(ramp_v())
    assert rv["hog_global"].argmax() == 4  # constant vertical gradient
    assert rv["cnt_global"].sum() == 0


def test_plateau_has_no_strict_extrema():
    assert (
        ss.FeatureComputer((256, 256), grid=[(5, 5)])
        .compute(solid_square())["cnt_global"]
        .sum()
        == 0
    )


def test_single_impulse_one_max():
    c = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(single_impulse())[
        "cnt_global"
    ]
    assert c[0] == 1 and c[1] == 0  # one bright isolated pixel, no minima
    d = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(
        single_impulse(v=0, bg=100)
    )["cnt_global"]
    assert d[0] == 0 and d[1] == 1  # one dark isolated pixel


@pytest.mark.parametrize("pos", [(0, 0), (0, 128), (128, 0), (255, 255), (255, 100)])
def test_border_impulse_suppressed(pos):
    # a peak on the replicate border is never a *strict* extremum (clamped self-neighbour)
    y, x = pos
    c = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(single_impulse(y=y, x=x))[
        "cnt_global"
    ]
    assert c[0] == 0 and c[1] == 0


def test_all_white_and_black_are_zero():
    for v in (0, 255):
        r = ss.FeatureComputer((256, 256), grid=[(5, 5)]).compute(flat(v))
        assert (
            r["struct_global"][:3].sum() == 0
            if "struct_global" in r
            else r["struct_global"][:3].sum() == 0
        )
        assert r["hog_global"].sum() == 0
        assert r["cnt_global"].sum() == 0


# ---------------- degenerate configs / property bounds ---------------------
def test_empty_cells_are_finite_zero():
    # stride coarser than a finest cell leaves some cells with zero samples
    img = noise(64)
    sc = ss.FeatureComputer((64, 64), grid=[(5, 5)], stride=(4, 4))
    raw = sc.compute(img)
    fe = groups(sc, img)
    empty = raw["struct_0"][..., 3] == 0
    assert empty.any()  # this config really does starve some cells
    for k in ("struct_0", "hog_0", "cnt_0"):
        assert np.isfinite(fe[k]).all()
    assert (fe["struct_0"][empty] == 0).all()  # no NaN/inf, just zeros
    assert (fe["hog_0"][empty] == 0).all()


@pytest.mark.parametrize("img", ALL_IMAGES)
def test_property_bounds(img):
    r = ss.FeatureComputer(img.shape, grid=[(5, 5)])
    raw = r.compute(img)["struct_0"]
    fe = groups(r, img)
    # extrema can't exceed pixel count
    cnt = r.compute(img)["cnt_0"]
    assert (cnt[..., 0] + cnt[..., 1] <= raw[..., 3]).all()
    # derived ranges
    assert (fe["struct_0"][..., 0] >= -1e-6).all()  # energy >= 0
    assert (
        (fe["struct_0"][..., 1] >= -1e-6) & (fe["struct_0"][..., 1] <= 1 + 1e-4)
    ).all()  # coherence
    hs = fe["hog_0"].sum(-1)
    active = raw[..., 0] + raw[..., 1] > 0
    assert np.allclose(
        hs[active], 1.0, atol=1e-5
    )  # HOG L1-normalised where there is energy
    assert (fe["cnt_0"] >= -1e-6).all() and (fe["cnt_0"] <= 1 + 1e-6).all()


def test_repeatable_and_no_state_leak():
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5), (3, 3)])
    a, b = textured(), noise()
    ra1 = sc.compute(a)
    _ = sc.compute(b)  # different frame in between
    ra2 = sc.compute(a)
    for k in ra1:
        assert np.array_equal(ra1[k], ra2[k]), f"non-deterministic / state leak in {k}"


def test_global_matches_reference_derive():
    # features()["*_global"] must equal deriving from compute()["*_global"]
    img = textured()
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5)])
    raw, fe = sc.compute(img), groups(sc, img)
    g = raw["struct_global"].astype(float)
    n = max(g[3], 1.0)
    assert np.allclose(fe["struct_global"][0], (g[0] + g[1]) / n, rtol=1e-5)
    assert np.allclose(fe["struct_global"][1], ref_coherence(g[None])[0], atol=1e-5)
    hg = raw["hog_global"].astype(float)
    assert np.allclose(fe["hog_global"], hg / max(hg.sum(), 1.0), atol=1e-5)
    assert np.allclose(fe["cnt_global"][0], raw["cnt_global"][0] / n, atol=1e-5)


# ---------------- config-error handling ------------------------------------
def test_config_errors():
    with pytest.raises(ValueError):  # finest must divide the shape
        ss.FeatureComputer((100, 100), grid=[(5, 5)])
    with pytest.raises(ValueError):  # pyramid must be nested finest->coarsest
        ss.FeatureComputer((256, 256), grid=[(3, 3), (4, 4)])
    with pytest.raises(ValueError):  # shape must be 2-D or 3-D
        ss.FeatureComputer((8, 256, 256, 3), grid=[(5, 5)])
    sc = ss.FeatureComputer((256, 256), grid=[(5, 5)])
    with pytest.raises(TypeError):  # uint8 only
        sc.compute(np.zeros((256, 256), np.float32))
    with pytest.raises(ValueError):  # shape must match the configured size
        sc.compute(np.zeros((128, 128), np.uint8))


def test_stride_sampling_is_per_cell_in_columns():
    # Documented quirk: under stride, ROWS subsample globally (r % sy == 0) but
    # COLUMNS restart the stride at each finest-cell edge, so column c is sampled
    # iff (c % cw_finest) % sx == 0. For dyadic grids on divisible shapes cw is a
    # multiple of the stride and the two coincide; this pins the general case.
    h, w = 8, 96
    nx = 32
    cwf = w // nx  # 3
    sx = 4
    img = np.zeros((h, w), np.uint8)
    img[:] = (np.arange(w) % 2) * 60 + 40  # every pixel has a gradient
    counts = ss.FeatureComputer((h, w), grid=[(0, 5)], stride=(1, sx)).compute(img)[
        "struct_0"
    ][..., 3]
    per_cell = sum(1 for k in range(cwf) if k % sx == 0) * h  # rows all sampled (sy=1)
    assert (counts.ravel() == per_cell).all()
    assert per_cell == 1 * h  # only the cell's left column survives sx=4 > cw=3


# =====================================================================
# multi-channel  (single spatial pass, one accumulator set per channel)
# =====================================================================
def stack_hwc(imgs):
    """Stack equal-sized 2-D uint8 images into a contiguous (H, W, C) array."""
    return _c(np.stack(imgs, axis=-1))


# distinct content per channel so cross-channel bleed would be visible
def chan_imgs(n, c):
    pool = [
        textured(n),
        checker(n),
        noise(n),
        hedge(n),
        vedge(n),
        diag(n),
        ramp_h(n),
        flat(90, n),
    ]
    return [_c(pool[i % len(pool)]) for i in range(c)]


MC_GRIDS = [[(4, 4)], [(4, 4), (3, 3), (1, 1)], [(5, 5), (2, 2)]]


@pytest.mark.parametrize("c", [1, 2, 3, 5])
@pytest.mark.parametrize("grid", MC_GRIDS)
@pytest.mark.parametrize("stride", [(1, 1), (2, 2)])
def test_multichannel_matches_numpy_oracle(c, grid, stride):
    n = 64
    imgs = chan_imgs(n, c)
    hwc = stack_hwc(imgs)
    sc = ss.FeatureComputer((n, n, c), grid=grid, stride=stride)
    raw = sc.compute(hwc)
    for ch, img in enumerate(imgs):  # oracle each channel independently, then compare
        lv = ref_levels(img, grid, stride)
        oh, oc = ref_hog_extrema(img, grid, stride)
        for i in range(len(grid)):
            assert np.array_equal(raw[f"struct_{i}"][..., ch, :], lv[i])
            assert np.array_equal(raw[f"hog_{i}"][..., ch, :], oh[i])
            assert np.array_equal(raw[f"cnt_{i}"][..., ch, :], oc[i])


@pytest.mark.parametrize("c", [1, 2, 3, 8])
def test_multichannel_equals_single_channel(c):
    # each channel of a multi-channel run == a 2-D run on that channel alone,
    # bit-exact for raw and features. This is the core single-pass guarantee.
    n, grid, stride = 96, [(4, 4), (2, 2)], (1, 1)
    imgs = chan_imgs(n, c)
    hwc = stack_hwc(imgs)
    mc = ss.FeatureComputer((n, n, c), grid=grid, stride=stride)
    rmc, fmc = mc.compute(hwc), groups(mc, hwc)
    for ch, img in enumerate(imgs):
        sc = ss.FeatureComputer((n, n), grid=grid, stride=stride)
        r1, f1 = sc.compute(img), groups(sc, img)
        for k in r1:
            assert np.array_equal(rmc[k][..., ch, :], r1[k]), f"raw {k} ch{ch}"
        for k in f1:
            # perceptual hashes are one uint64 per channel: (C,) here, scalar for 2-D.
            got = fmc[k][ch] if fmc[k].ndim == 1 else fmc[k][..., ch, :]
            assert np.array_equal(got, f1[k]), f"feat {k} ch{ch}"


def test_multichannel_output_shapes():
    n, c = 64, 3
    hwc = stack_hwc(chan_imgs(n, c))
    sc = ss.FeatureComputer((n, n, c), grid=[(4, 4), (2, 2)])
    f = groups(sc, hwc)
    assert f["struct_0"].shape == (16, 16, c, len(ss.FEATURES))
    assert f["hog_0"].shape == (16, 16, c, len(ss.HOG_FEATURES))
    assert f["cnt_1"].shape == (4, 4, c, len(ss.COUNT_FEATURES))
    assert f["struct_global"].shape == (c, len(ss.FEATURES))
    assert f["hog_global"].shape == (c, len(ss.HOG_FEATURES))
    r = sc.compute(hwc)
    assert r["struct_0"].shape == (16, 16, c, 4)
    assert r["struct_global"].shape == (c, 4)


def test_2d_input_has_no_channel_axis():
    # backward compatibility: a 2-D image yields channel-less output (unchanged API)
    n = 64
    img = textured(n)  # capture once (builder is randomised per call)
    sc2 = ss.FeatureComputer((n, n), grid=[(4, 4)])
    f2 = groups(sc2, img)
    assert f2["struct_0"].shape == (16, 16, len(ss.FEATURES))
    assert f2["struct_global"].shape == (len(ss.FEATURES),)
    # a 3-D single-channel input, by contrast, keeps an explicit size-1 channel axis
    sc3 = ss.FeatureComputer((n, n, 1), grid=[(4, 4)])
    f3 = groups(sc3, _c(img[:, :, None]))
    assert f3["struct_0"].shape == (16, 16, 1, len(ss.FEATURES))
    assert np.array_equal(f3["struct_0"][..., 0, :], f2["struct_0"])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_channel_axis_variants(axis):
    # the same data laid out along different axes gives identical per-channel output
    n, c = 64, 3
    hwc = stack_hwc(chan_imgs(n, c))
    moved = _c(np.moveaxis(hwc, -1, axis))  # place channels on `axis`
    ref = groups(ss.FeatureComputer((n, n, c), grid=[(4, 4), (2, 2)]), hwc)
    got = groups(
        ss.FeatureComputer(moved.shape, grid=[(4, 4), (2, 2)], channel_axis=axis), moved
    )
    for k in ref:
        assert np.array_equal(ref[k], got[k]), k


def test_multichannel_zero_copy_strided():
    # pass a non-contiguous multi-channel view (CHW source, HWC logical) with no
    # ascontiguousarray in the path; result must equal the contiguous version.
    n, c = 64, 3
    imgs = chan_imgs(n, c)
    chw = _c(np.stack(imgs, axis=0))  # (C, H, W) contiguous -> HWC view is strided
    view = np.moveaxis(chw, 0, -1)
    assert not view.flags["C_CONTIGUOUS"]
    sc = ss.FeatureComputer((n, n, c), grid=[(4, 4), (2, 2)])
    a = sc.compute(stack_hwc(imgs))
    b = sc.compute(view)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_multichannel_hog_sum_identity():
    # per channel, sum of HOG bins == Sxx + Syy (energy), the exact integer identity
    n, c = 96, 4
    hwc = stack_hwc(chan_imgs(n, c))
    r = ss.FeatureComputer((n, n, c), grid=[(4, 4)], stride=(2, 2)).compute(hwc)
    hog_sum = r["hog_0"].sum(-1)
    energy = r["struct_0"][..., 0] + r["struct_0"][..., 1]
    assert np.array_equal(hog_sum, energy)


def test_multichannel_global_equals_total():
    n, c = 64, 3
    hwc = stack_hwc(chan_imgs(n, c))
    r = ss.FeatureComputer((n, n, c), grid=[(4, 4), (2, 2)]).compute(hwc)
    for k, gk in [
        ("struct_0", "struct_global"),
        ("hog_0", "hog_global"),
        ("cnt_0", "cnt_global"),
    ]:
        assert np.array_equal(r[k].sum(axis=(0, 1)), r[gk])  # sum cells -> per channel


def test_multichannel_rejects_extra_spatial_axis():
    with pytest.raises(ValueError):
        ss.FeatureComputer((8, 8, 8, 3), grid=[(2, 2)])  # 3 spatial axes: unsupported
