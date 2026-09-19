"""Named views over the pyramid, for the assertions in this suite.

`features()` returns one `(H, W, C*F)` map per level, F = len(FEATURE_NAMES). These tests were written
against the older per-group dict and check values, not containers, so rather than
rewrite two hundred assertions -- and risk one of them passing vacuously -- the split
lives here, in one place, exercised by every test that uses it.

`groups(fc, img)` reproduces the old mapping exactly: verified array-for-array against
the pre-change output across six configs.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import imfeat

# (name, width) in FEATURE_NAMES order; the trailing MOMENTS block is "mom".
_W = (
    ("struct", len(imfeat.FEATURES)),
    ("hog", len(imfeat.HOG_FEATURES)),
    ("cnt", len(imfeat.COUNT_FEATURES)),
    ("lbp", len(imfeat.LBP_FEATURES)),
    ("desc", len(imfeat.DESCRIPTOR_FEATURES)),
    ("bard", len(imfeat.BARD_FEATURES)),
    ("mom", len(imfeat.MOMENTS)),
)
_F = sum(n for _, n in _W)


def _keys(n_levels: int) -> list[str]:
    return [str(i) for i in range(n_levels - 1)] + ["global"]


def groups(fc: imfeat.FeatureComputer, img: np.ndarray) -> dict[str, np.ndarray]:
    """Old-style dict: struct_i, hog_i, cnt_i, lbp_i, desc_i, mom_i, xchan_i,
    *_summary_i, and the three hashes."""
    p = fc.features(img)
    keys, out, two_d = _keys(len(p.maps)), {}, img.ndim == 2
    for i, m, mo in zip(keys, p.maps, p.moments):
        g = m.reshape(*m.shape[:-1], m.shape[-1] // _F, _F)
        o = 0
        for name, n in _W:
            v = mo if name == "mom" else g[..., o : o + n]  # float64 moments
            out[f"{name}_{i}"] = v[..., 0, :] if two_d else v
            o += n
    for i, x in zip(keys, p.cross):
        out[f"xchan_{i}"] = x
    for i, s in zip(keys[:-1], p.summary):
        o = 0
        for name, n in _W:
            v = s[o : o + n]
            out[f"{name}_summary_{i}"] = v[:, 0, :] if two_d else v
            o += n
    for name, h in zip(imfeat.HASHES, p.hashes):
        out[name] = h[0] if two_d else h
    return out


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", help="also run tests needing cv2/imagehash/PIL/scipy"
    )
    parser.addoption("--reps", type=int, default=500, help="timed calls per latency stat")


def pytest_configure(config):
    global REPS
    REPS = config.getoption("--reps")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        return
    skip = pytest.mark.skip(reason="reference-library test; run with --full")
    for item in items:
        if "full" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fgroups():
    return groups


def frame(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Deterministic textured uint8 frame: smooth wave + checker + noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    base = 90 + 60 * np.sin(y / 7.0) * np.cos(x / 11.0) + 40 * ((x // 9 + y // 5) % 2)
    a = base[..., None] if len(shape) == 3 else base
    return np.clip(a + rng.integers(-25, 26, shape), 0, 255).astype(np.uint8)


REPS = 500


def latency(fn, reps: int | None = None, warm: int = 30) -> dict[str, float]:
    """Per-call ms stats over reps (default --reps). min estimates cost; std, p90, p99 the tail."""
    reps = reps or REPS
    for _ in range(warm):
        fn()
    t = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        fn()
        t[i] = time.perf_counter() - t0
    t *= 1e3
    p50, p90, p99 = np.percentile(t, [50, 90, 99])
    return {"min": t.min(), "mean": t.mean(), "std": t.std(), "p50": p50, "p90": p90, "p99": p99}
