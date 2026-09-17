"""Benchmarks: stride latency/accuracy sweeps, thread/phase scaling, headline latency.

Timings are printed (run with -s), never asserted beyond generous regression bounds.
Accuracy is deterministic, so it is gated. Skip all of it with -m "not bench".
"""

from __future__ import annotations

import platform

import numpy as np
import pytest

import imfeat

from conftest import frame, groups, latency

pytestmark = pytest.mark.bench

EXPS = [5, 4, 3, 2]  # pyramid: 32x32 finest -> 4x4 coarsest
GRID = [(e, e) for e in EXPS]
STRIDES = [1, 2, 4, 8]
THREADS = [1, 2, 3, 4]


def synth(n, c):
    """Wave, ramp, noise, checker + a hard rectangle, cycled over channels: mixes smooth
    and high-frequency content so stride error is not flattered by an easy image."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n]
    planes = [
        (127 + 80 * np.sin(xx / (n / 14.0))).astype(np.uint8),
        np.tile(np.linspace(0, 255, n, dtype=np.uint8), (n, 1)),
        rng.integers(0, 256, (n, n), dtype=np.uint8),
        (255 * (((yy // (n // 8)) + (xx // (n // 8))) % 2)).astype(np.uint8),
    ]
    img = np.stack([planes[k % 4] for k in range(c)], -1)
    img[n // 8 : n // 4, n // 8 : 3 * n // 4] = 255
    return np.ascontiguousarray(img[:, :, 0] if c == 1 else img)


def _corr(a, b):
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def quality(ref, out):
    """Strided vs exact, finest level. Moments: absolute grey-level error. Structure maps:
    correlations, since downstream consumes their layout, not their scale."""
    q = {}
    m, mr = out["mom_0"], ref["mom_0"]
    q["mean_err"] = float(np.abs(m[..., 0] - mr[..., 0]).mean())
    sd, sdr = np.sqrt(np.maximum(m[..., 1], 0)), np.sqrt(np.maximum(mr[..., 1], 0))
    q["std_err"] = float(np.abs(sd - sdr).mean())

    s, sr = out["struct_0"], ref["struct_0"]
    q["energy_r"] = _corr(s[..., 0], sr[..., 0])
    q["coh_err"] = float(np.abs(s[..., 1] - sr[..., 1]).mean())
    # angle between double-angle vectors, weighted by exact coherence (no edge = noise)
    dot = s[..., 2] * sr[..., 2] + s[..., 3] * sr[..., 3]
    nrm = np.hypot(s[..., 2], s[..., 3]) * np.hypot(sr[..., 2], sr[..., 3])
    dth = np.arccos(np.clip(np.where(nrm > 1e-9, dot / np.maximum(nrm, 1e-9), 1.0), -1, 1))
    w = sr[..., 1]
    q["ori_deg"] = float(np.degrees(0.5 * (dth * w).sum() / max(w.sum(), 1e-9)))

    h, hr = out["hog_0"], ref["hog_0"]
    na, nb = np.linalg.norm(h, axis=-1), np.linalg.norm(hr, axis=-1)
    ok = (na > 0) & (nb > 0)  # flat cells: all-zero histogram, no angle
    q["hog_cos"] = float(((h * hr).sum(-1)[ok] / (na[ok] * nb[ok])).mean()) if ok.any() else 1.0
    q["cnt_r"] = _corr(out["cnt_0"][..., 0], ref["cnt_0"][..., 0])
    lb, lbr = out["lbp_0"], ref["lbp_0"]  # L1-normalised, never zero
    q["lbp_cos"] = float(
        (lb * lbr).sum(-1).mean()
        / (np.linalg.norm(lb, axis=-1) * np.linalg.norm(lbr, axis=-1)).mean()
    )
    q["xchan_r"] = (
        _corr(out["xchan_0"][..., 1], ref["xchan_0"][..., 1]) if "xchan_0" in ref else 1.0
    )
    return q


COLS = [
    ("ms", 8, "{:8.3f}"),
    ("speedup", 9, "{:8.2f}x"),
    ("mean_err", 9, "{:9.3f}"),
    ("std_err", 8, "{:8.3f}"),
    ("energy_r", 9, "{:9.4f}"),
    ("coh_err", 8, "{:8.4f}"),
    ("ori_deg", 8, "{:8.2f}"),
    ("hog_cos", 8, "{:8.4f}"),
    ("cnt_r", 7, "{:7.4f}"),
    ("lbp_cos", 8, "{:8.4f}"),
    ("xchan_r", 8, "{:8.4f}"),
]


def table(rows, label=""):
    print(f"{label:>12}" + "".join(f"{c:>{w}}" for c, w, _ in COLS))
    for name, r in rows:
        print(f"{name:>12}" + "".join(f.format(r[c]) for c, _, f in COLS))


def sweep(n, c):
    """Latency + quality vs stride, relative to this config's own stride=1 run.
    * marks a stride beyond the finest cell width: one column per cell, knob saturated."""
    img = synth(n, c)
    ref, base, rows = None, None, []
    for s in STRIDES:
        fc = imfeat.FeatureComputer(img.shape, grid=GRID, stride=s)
        out = groups(fc, img)
        ref = ref or out
        ms = latency(lambda: fc.features(img), reps=50, warm=5)[0]
        base = base or ms
        tag = f"stride={s}" + ("*" if s > n >> EXPS[0] else "")
        rows.append((tag, {"ms": ms, "speedup": base / ms, **quality(ref, out)}))
    return dict(rows)


def head(title):
    print("\n" + "=" * 120 + "\n" + title + "\n" + "=" * 120)


def test_stride_sweep():
    head(
        "Stride: latency vs accuracy (256x256x3), errors vs exact stride=1.\n"
        "mean_err/std_err in grey levels; ori_deg coherence-weighted; *_r/*_cos: 1.0 = exact."
    )
    rows = sweep(256, 3)
    table(rows.items())
    for r in rows.values():
        assert all(np.isfinite(v) for v in r.values())
    r2 = rows["stride=2"]
    assert r2["mean_err"] < 8
    assert r2["energy_r"] > 0.9
    assert r2["ori_deg"] < 6
    assert r2["hog_cos"] > 0.85
    assert r2["lbp_cos"] > 0.97
    assert r2["xchan_r"] > 0.9


def test_size_sweep():
    head(f"Input size (3ch), vs own stride=1. * = stride > finest cell width (n/{1 << EXPS[0]})")
    for n in (128, 256, 512, 1024):
        table(sweep(n, 3).items(), f"{n}x{n}x3")


def test_channel_sweep():
    head("Channel count (256x256), every feature on every channel")
    for c in (1, 2, 3, 4, 8, 16):
        table(sweep(256, c).items(), f"C={c}")


def test_per_channel_cost():
    head("Per-channel cost (256x256, stride=2): SIMD packs 4 channels per vector")
    for c in (1, 2, 3, 4, 8, 16):
        im = synth(256, c)
        fc = imfeat.FeatureComputer(im.shape, grid=GRID, stride=2)
        ms = latency(lambda: fc.features(im))[0]
        print(f"  C={c:>2}: {ms:7.3f} ms  ({ms / c:6.3f} ms/channel)")


@pytest.mark.parametrize("n,k,stride", [(256, 5, 2), (512, 5, 2), (1024, 6, 2), (1024, 5, 4)])
def test_thread_phases(n, k, stride):
    """accumulate = compute pass only; tail = derive + summary + hashes + Python assembly.
    accumulate scaling but total not -> serial tail. Neither -> memory bandwidth / E-cores."""
    img = frame((n, n, 3))
    grid = [(k - i, k - i) for i in range(4)]
    head(f"Threads: {n}x{n}x3 finest-{1 << k} s{stride}  (cpu_count={imfeat.cpu_count()}, ms)")
    print(
        f"  {'':9s} {'acc min':>8} {'x':>6} {'tail':>7} {'min':>7} {'p50':>7} {'p95':>7} {'x':>6}"
    )
    a0 = t0 = None
    for nt in THREADS:
        fc = imfeat.FeatureComputer(img.shape, grid=grid, stride=stride, threads=nt)
        view = fc._view(img)
        acc = latency(lambda: fc._impl.raw(view))[0]
        mn, p50, p95 = latency(lambda: fc.features(img))
        a0, t0 = a0 or acc, t0 or mn
        print(
            f"  threads={fc.threads} {acc:8.3f} {a0 / acc:5.2f}x {mn - acc:7.3f} "
            f"{mn:7.3f} {p50:7.3f} {p95:7.3f} {t0 / mn:5.2f}x"
        )


@pytest.mark.parametrize("shape,stride", [((256, 256, 3), 1), ((256, 256, 3), 2), ((256, 256), 1)])
def test_latency(shape, stride):
    """Headline 4-level features() latency with a generous regression bound."""
    img = frame(shape)
    fc = imfeat.FeatureComputer(shape, grid=GRID, stride=stride)
    mn, p50, p95 = latency(lambda: fc.features(img))
    print(
        f"\n  {platform.machine()} {platform.system()} py{platform.python_version()} | "
        f"{shape} 4-level stride={stride}: min={mn:.3f} p50={p50:.3f} p95={p95:.3f} ms"
    )
    assert p50 < 6.0
