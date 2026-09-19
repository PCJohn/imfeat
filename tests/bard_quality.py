"""Run the bard cases: invariants that must hold by definition, an oracle diff where
the mesh lines up, and a golden diff against a build known to be correct.
  python3 check.py save   -> write golden.npz from the current build
  python3 check.py check  -> assert invariants + golden (use after latency changes)"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bard_cases as cases
from bard_oracle import maps, pool

import imfeat

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
NB = len(imfeat.BARD_FEATURES)
I0 = imfeat.FEATURE_NAMES.index("bard_cover")
F = len(imfeat.FEATURE_NAMES)


def bard_of(img, grid, stride):
    C = 1 if img.ndim == 2 else img.shape[-1]
    m = imfeat.FeatureComputer(img.shape, grid, stride).features(img).maps[0]
    m = m.reshape(*m.shape[:-1], C, F)
    return m[..., I0 : I0 + NB]


fails, out, skipped = [], {}, []
for name, img, grid, stride in cases.build():
    try:
        b = bard_of(img, grid, stride)
    except ValueError as e:  # geometries imfeat rejects outright, e.g. grid not dividing
        skipped.append(f"{name}: {e}")
        continue
    out[name] = b
    cover, spec, peak, peaked, bal = (
        b[..., 0],
        b[..., 1 : 1 + NB - 4],
        b[..., NB - 3],
        b[..., NB - 2],
        b[..., NB - 1],
    )

    def bad(msg, _case=name):  # bind the case name; closing over the loop var is a trap
        fails.append(f"{_case}: {msg}")

    if not np.all(np.isfinite(b)):
        bad("non-finite output")
    if b.min() < -1.0001 or b.max() > 3.0001:
        bad(f"out of range [{b.min():.3f},{b.max():.3f}]")
    if cover.min() < -1e-6 or cover.max() > 1 + 1e-6:
        bad(f"cover outside [0,1]: {cover.min()},{cover.max()}")
    if bal.min() < -1 - 1e-6 or bal.max() > 1 + 1e-6:
        bad("bal outside [-1,1]")
    s = spec.sum(-1)
    nz = cover > 0
    if np.any(nz) and np.any(np.abs(s[nz] - 1) > 1e-5) and np.any(s[nz] > 1e-9):
        bad("spectrum does not sum to 1 where it fires")
    if np.any(peak[s > 1e-9] < 1 - 1e-5) or np.any(peak > NB - 4 + 1e-5):
        bad("peak lag outside [1,NL]")
    # flat fields, linear ramps and lone step edges contain no bar of any width
    if name in ("all_black", "all_white", "mid_grey", "ramp_x", "step_edge"):
        if cover.max() > 1e-6:
            bad(f"fires on a bar-free field: cover max {cover.max():.4f}")
        if np.abs(b).max() > 1e-6:
            bad("non-zero features on a bar-free field")
    # a stroke of width w should put its mass on the matching lag
    for wd, lag in ((1, 0), (2, 1), (4, 2)):
        if name == f"vbars_w{wd}":
            fired = cover > 0.01
            if not fired.any():
                bad("no response to its own bar width")
            else:
                dom = spec[fired].mean(0).argmax()
                if dom != lag:
                    bad(f"width {wd} peaks at lag index {dom}, expected {lag}")
                if bal[fired].mean() < 0:
                    bad("light-on-dark bars read as dark polarity")
        if name == f"vbars_w{wd}_inv" and (cover > 0.01).any() and bal[cover > 0.01].mean() > 0:
            bad("dark-on-light bars read as light polarity")
    # oracle diff where the sampling meshes coincide
    h, w = img.shape[:2]
    n = 1 << grid
    # the numpy oracle is written for the shipped lag set; skip it if they diverge
    if (
        img.ndim == 3
        and h == w
        and h % n == 0
        and (w // n) % stride == 0
        and NB - 4 == len(__import__("bard_oracle").LAGS)
    ):
        for ch in range(img.shape[-1]):
            d = np.abs(pool(maps(img[:, :, ch]), n, stride) - b[:, :, ch]).max()
            if d > 2e-5:
                bad(f"oracle mismatch ch{ch}: {d:.2e}")
if MODE == "save":
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bard_golden.npz"), **out)
    print(f"saved {len(out)} goldens")
else:
    g = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bard_golden.npz"))
    for k, got in out.items():
        if k not in g.files:
            fails.append(f"{k}: no golden")
            continue
        d = float(np.abs(got - g[k]).max())
        if d > 0:
            fails.append(f"{k}: golden drift {d:.3e}")
print(f"{len(out)} cases run, {len(skipped)} skipped, {len(fails)} failures")
for sk in skipped:
    print("  skip", sk)
for f in fails[:15]:
    print("  FAIL", f)
sys.exit(1 if fails else 0)
