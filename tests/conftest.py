"""Named views over the pyramid, for the assertions in this suite.

`features()` returns one `(H, W, C*38)` map per level. These tests were written
against the older per-group dict and check values, not containers, so rather than
rewrite two hundred assertions -- and risk one of them passing vacuously -- the split
lives here, in one place, exercised by every test that uses it.

`groups(fc, img)` reproduces the old mapping exactly: verified array-for-array against
the pre-change output across six configs.
"""

from __future__ import annotations

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


@pytest.fixture
def fgroups():
    return groups
