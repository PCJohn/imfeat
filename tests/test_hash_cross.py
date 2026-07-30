"""Perceptual hashes -- aHash / wHash / pHash, imagehash-compatible.

Tiers mirroring the other suites:
  1. exactness   -- a numpy oracle with identical rules; bit-exact across sizes,
                    strides and channel counts (this is the single-pass guarantee).
  2. imagehash   -- bit-exact vs the reference library at native hash sizes (8x8 for
                    a/w, 32x32 for p), where its Lanczos resize is a no-op, so only the
                    shared core is compared. Distinct-valued grids sidestep imagehash's
                    own median-tie jitter (its wHash round-trips through pywt).
  3. closed form -- flat and single-impulse images, derivable by hand.
Plus an accuracy report (relative Hamming vs stock imagehash on full-size frames:
mean/min/p50/max) and a latency check that the in-pass accumulation stays ~free.
"""

import time

import numpy as np
import pytest
import scipy.fftpack
from PIL import Image

import imagehash

import imfeat

from conftest import groups

rng = np.random.default_rng(11)

NBITS = 64
# sizes divisible by 8 (so a (3,3) finest grid is legal) but not by 32, to exercise
# the floor-division binning of the 32x32 hash grid on non-aligned shapes.
SIZES = [(64, 64), (72, 96), (120, 56)]
STRIDES = [(1, 1), (2, 2), (3, 2)]


# ---------------- oracle -----------------------------------------------------
def _sample_cols(w, nfx, sx):
    """Columns imfeat visits: stride sx restarted at each finest-cell edge."""
    cw = w // nfx
    return np.concatenate([np.arange(e * cw, e * cw + cw, sx) for e in range(nfx)])


def _grid_sums(img, n, nfx, stride):
    """n x n grids of Sum(v) and count over exactly the sampled pixels."""
    h, w = img.shape
    sy, sx = stride
    rows = np.arange(0, h, sy)
    cols = _sample_cols(w, nfx, sx)
    ri, ci = rows * n // h, cols * n // w
    s = np.zeros((n, n))
    c = np.zeros((n, n))
    np.add.at(s, (ri[:, None], ci[None, :]), img[np.ix_(rows, cols)].astype(np.float64))
    np.add.at(c, (ri[:, None], ci[None, :]), 1.0)
    return s, c


def _pack(bits):
    """64 bits row-major, MSB first -- imfeat's packing."""
    out = 0
    for i, b in enumerate(bits.ravel()):
        if b:
            out |= 1 << (63 - i)
    return np.uint64(out)


def _unpack(u):
    u = int(u)
    return np.array([(u >> (63 - i)) & 1 for i in range(64)], bool).reshape(8, 8)


def ref_hashes(img, nfx, stride):
    s32, c32 = _grid_sums(img, 32, nfx, stride)
    m32 = np.divide(s32, c32, out=np.zeros_like(s32), where=c32 > 0)
    s8 = s32.reshape(8, 4, 8, 4).sum((1, 3))
    c8 = c32.reshape(8, 4, 8, 4).sum((1, 3))
    m8 = np.divide(s8, c8, out=np.zeros_like(s8), where=c8 > 0)
    a = _pack(m8 > m8.mean())
    w = _pack(m8 > np.median(m8))
    d = scipy.fftpack.dct(scipy.fftpack.dct(m32, axis=0), axis=1)[:8, :8]
    p = _pack(d > np.median(d))
    return {"ahash": a, "whash": w, "phash": p}


def _nfx(grid):
    return 1 << grid[0][1]


# ---------------- 1. exactness vs numpy oracle -------------------------------
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("grid", [[(0, 0)], [(3, 3)]])
def test_matches_oracle(size, stride, grid):
    h, w = size
    img = rng.integers(0, 256, (h, w), np.uint8)
    got = groups(imfeat.FeatureComputer((h, w), grid=grid, stride=stride), img)
    ref = ref_hashes(img, _nfx(grid), stride)
    for k, v in ref.items():
        assert got[k] == v, k


@pytest.mark.parametrize("stride", STRIDES)
def test_matches_oracle_deep_grid(stride):
    # a 32x32 finest grid (requires 32 | shape) shares the pass's finest cells.
    img = rng.integers(0, 256, (96, 128), np.uint8)
    grid = [(5, 5), (2, 2)]
    got = groups(imfeat.FeatureComputer(img.shape, grid=grid, stride=stride), img)
    ref = ref_hashes(img, _nfx(grid), stride)
    for k, v in ref.items():
        assert got[k] == v, k


def test_oracle_multichannel_is_per_channel():
    img = rng.integers(0, 256, (96, 96, 3), np.uint8)
    got = groups(imfeat.FeatureComputer(img.shape, grid=[(3, 3)]), img)
    for c in range(3):
        ref = ref_hashes(np.ascontiguousarray(img[:, :, c]), 8, (1, 1))
        for k, v in ref.items():
            assert got[k][c] == v, f"{k} ch{c}"


# ---------------- 2. bit-exact vs imagehash at native sizes ------------------
def _distinct(n):
    """n*n grid of distinct grey levels (no median ties -> imagehash is stable)."""
    return rng.permutation(256)[: n * n].astype(np.uint8).reshape(n, n)


def test_ahash_whash_vs_imagehash_native():
    for _ in range(300):
        g = _distinct(8)  # 8x8 input: imagehash's resize to 8x8 is identity
        f = groups(imfeat.FeatureComputer((8, 8), grid=[(3, 3)]), g)
        im = Image.fromarray(g, "L")
        assert np.array_equal(_unpack(f["ahash"]), imagehash.average_hash(im).hash)
        assert np.array_equal(_unpack(f["whash"]), imagehash.whash(im).hash)


def test_phash_vs_imagehash_native():
    for _ in range(300):
        g = rng.integers(0, 256, (32, 32), np.uint8)  # 32x32: resize is identity
        f = groups(imfeat.FeatureComputer((32, 32), grid=[(5, 5)]), g)
        im = Image.fromarray(g, "L")
        assert np.array_equal(_unpack(f["phash"]), imagehash.phash(im).hash)


def test_vs_imagehash_per_channel_native():
    imgs = [_distinct(8) for _ in range(3)]
    hwc = np.ascontiguousarray(np.stack(imgs, -1))
    f = groups(imfeat.FeatureComputer(hwc.shape, grid=[(3, 3)]), hwc)
    for c, g in enumerate(imgs):
        im = Image.fromarray(g, "L")
        assert np.array_equal(_unpack(f["ahash"][c]), imagehash.average_hash(im).hash)
        assert np.array_equal(_unpack(f["whash"][c]), imagehash.whash(im).hash)


# ---------------- 3. closed form ---------------------------------------------
def test_flat_image():
    # a constant image: every 8x8 cell equals the mean/median, so strict > is all-False
    # for aHash/wHash. pHash's off-DC coefficients are ~0 (float noise either side), but
    # the DC term is reliably positive and above the median -> bit (0,0) is set.
    f = groups(imfeat.FeatureComputer((64, 64), grid=[(0, 0)]), np.full((64, 64), 137, np.uint8))
    assert f["ahash"] == 0 and f["whash"] == 0
    assert (int(f["phash"]) >> 63) & 1 == 1


def test_bright_quadrant():
    # top-left quadrant bright, rest dark: aHash sets exactly the 4x4 top-left block.
    img = np.zeros((64, 64), np.uint8)
    img[:32, :32] = 255
    f = groups(imfeat.FeatureComputer((64, 64), grid=[(0, 0)]), img)
    exp = np.zeros((8, 8), bool)
    exp[:4, :4] = True
    assert f["ahash"] == _pack(exp)


# ---------------- accuracy report vs stock imagehash (Lanczos) ---------------
def _natural(h, w):
    """A smooth low-frequency frame + mild noise -- closer to real video than white
    noise, for which box vs Lanczos downsampling is a fair comparison."""
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    im = np.zeros((h, w))
    for _ in range(5):
        fx, fy = rng.uniform(1, 6, 2)
        ph = rng.uniform(0, 6.28)
        im += rng.uniform(0.3, 1) * np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
    im += rng.normal(0, 0.05, (h, w))
    im -= im.min()
    return (255 * im / im.max()).astype(np.uint8)


def test_accuracy_report_vs_imagehash():
    ref = {"ahash": imagehash.average_hash, "whash": imagehash.whash, "phash": imagehash.phash}
    dist = {k: [] for k in ref}
    for _ in range(60):
        g = _natural(256, 256)
        f = groups(imfeat.FeatureComputer((256, 256), grid=[(5, 5)]), g)
        im = Image.fromarray(g, "L")
        for k, fn in ref.items():
            hd = bin(int(f[k]) ^ int(_pack(fn(im).hash))).count("1")
            dist[k].append(hd / NBITS)
    print("\nrelative Hamming vs imagehash (Lanczos), 256x256 natural frames:")
    for k, d in dist.items():
        d = np.array(d)
        print(
            f"  {k}: mean={d.mean():.3f} min={d.min():.3f} p50={np.median(d):.3f} max={d.max():.3f}"
        )
        assert d.mean() < 0.20, f"{k} drift too high ({d.mean():.3f})"


# ---------------- latency: the in-pass accumulation is ~free ------------------
def test_latency_reasonable():
    img = rng.integers(0, 256, (256, 256, 3), np.uint8)
    fc = imfeat.FeatureComputer(img.shape, grid=[(5, 5), (4, 4), (3, 3), (2, 2)])
    for _ in range(5):
        groups(fc, img)
    ts = []
    for _ in range(50):
        t = time.perf_counter()
        groups(fc, img)
        ts.append((time.perf_counter() - t) * 1e3)
    print(
        f"\nfeatures()+hashes 256x256x3 4-level: p50={np.median(ts):.3f} ms, min={min(ts):.3f} ms"
    )
    assert np.median(ts) < 6.0  # generous; A/B in bench_compare shows the marginal cost
