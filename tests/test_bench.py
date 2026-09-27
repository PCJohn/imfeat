"""Benchmarks: stride latency/accuracy sweeps, thread/phase scaling, headline latency.

Timings are printed (run with -s), never asserted beyond generous regression bounds.
Accuracy is deterministic, so it is gated. Skip all of it with -m "not bench".
The images are synthetic planes, taken as they are (``feature_space=None``); the colour
conversion's own cost is ``test_colour_conversion``.
"""

from __future__ import annotations

import platform
import time

import numpy as np
import pytest

import imfeat

from conftest import frame, groups, latency

pytestmark = pytest.mark.bench

STRIDES = [1, 2, 4, 8]
THREADS = [1, 2, 3, 4]
CHANNELS = [1, 2, 3, 4, 8, 16]
# every benchmark runs at each size with a 32x32 and a 64x64 finest grid, 4 levels deep
SIZE_GRID = pytest.mark.parametrize("n,k", [(n, k) for n in (256, 512, 1024) for k in (5, 6)])


def pyramid(k):
    return [(k - i, k - i) for i in range(4)]


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


def sweep(n, k, c):
    """Latency + quality vs stride, relative to this config's own stride=1 run.
    * marks a stride beyond the finest cell width: one column per cell, knob saturated."""
    img = synth(n, c)
    ref, base, rows = None, None, {}
    for s in STRIDES:
        fc = imfeat.FeatureComputer(img.shape, grid=pyramid(k), stride=s, feature_space=None)
        out = groups(fc, img)
        ref = ref or out
        ms = latency(lambda: fc.features(img), reps=50, warm=5)["min"]
        base = base or ms
        rows[s] = {"ms": ms, "speedup": base / ms, **quality(ref, out)}
    return rows


def table(rows, n, k, label=""):
    print(f"{label:>12}" + "".join(f"{c:>{w}}" for c, w, _ in COLS))
    for s, r in rows.items():
        name = f"stride={s}" + ("*" if s > n >> k else "")
        print(f"{name:>12}" + "".join(f.format(r[c]) for c, _, f in COLS))


STATS = ("min", "mean", "std", "p50", "p90", "p99")
STATS_HEAD = " ".join(f"{c:>7}" for c in STATS)


def stats_row(st):
    return " ".join(f"{st[c]:7.3f}" for c in STATS)


def head(title):
    print("\n" + "=" * 120 + "\n" + title + "\n" + "=" * 120)


@SIZE_GRID
def test_stride_sweep(n, k):
    head(
        f"Stride: latency vs accuracy ({n}x{n}x3, finest {1 << k}), errors vs exact stride=1.\n"
        "mean_err/std_err in grey levels; ori_deg coherence-weighted; *_r/*_cos: 1.0 = exact.\n"
        f"* = stride > finest cell width ({n >> k} px)"
    )
    rows = sweep(n, k, 3)
    table(rows, n, k)
    for r in rows.values():
        assert all(np.isfinite(v) for v in r.values())
    if (n >> k) // 2 >= 4:  # stride=2 floors only where the README stride rule holds
        r2 = rows[2]
        assert r2["mean_err"] < 8
        assert r2["energy_r"] > 0.9
        assert r2["ori_deg"] < 6
        assert r2["hog_cos"] > 0.85
        assert r2["lbp_cos"] > 0.85
        assert r2["xchan_r"] > 0.9


@SIZE_GRID
def test_channel_sweep(n, k):
    head(f"Channel count ({n}x{n}, finest {1 << k}), every feature on every channel")
    for c in CHANNELS:
        table(sweep(n, k, c), n, k, f"C={c}")


@SIZE_GRID
def test_per_channel_cost(n, k):
    head(f"Per-channel cost ({n}x{n}, finest {1 << k}, stride=2): SIMD packs 4 channels/vector")
    for c in CHANNELS:
        im = synth(n, c)
        fc = imfeat.FeatureComputer(im.shape, grid=pyramid(k), stride=2, feature_space=None)
        ms = latency(lambda: fc.features(im), reps=50, warm=5)["min"]
        print(f"  C={c:>2}: {ms:7.3f} ms  ({ms / c:6.3f} ms/channel)")


@SIZE_GRID
@pytest.mark.parametrize("stride", [2, 4])
def test_thread_phases(n, k, stride):
    """features() and raw() against the thread count. They are different passes: features()
    derives the finest level as it goes and never stores it, raw() stores and returns every
    level's sums, so neither is a part of the other."""
    img = frame((n, n, 3))
    head(f"Threads: {n}x{n}x3 finest-{1 << k} s{stride}  (cpu_count={imfeat.cpu_count()}, ms)")
    print(f"  {'':9s} {'raw min':>8} {'x':>6} {STATS_HEAD} {'x':>6}")
    a0 = t0 = None
    for nt in THREADS:
        fc = imfeat.FeatureComputer(
            img.shape, grid=pyramid(k), stride=stride, threads=nt, feature_space=None
        )
        view = fc._view(img)
        acc = latency(lambda: fc._impl.raw(view))["min"]
        st = latency(lambda: fc.features(img))
        a0, t0 = a0 or acc, t0 or st["min"]
        print(
            f"  threads={fc.threads} {acc:8.3f} {a0 / acc:5.2f}x "
            f"{stats_row(st)} {t0 / st['min']:5.2f}x"
        )


@SIZE_GRID
@pytest.mark.parametrize("c", [1, 3])
@pytest.mark.parametrize("stride", [1, 2])
def test_latency(n, k, c, stride):
    """Headline 4-level features() latency; regression bound scales with pixel count."""
    shape = (n, n) if c == 1 else (n, n, c)
    img = frame(shape)
    fc = imfeat.FeatureComputer(shape, grid=pyramid(k), stride=stride, feature_space=None)
    st = latency(lambda: fc.features(img))
    print(
        f"\n  {platform.machine()} {platform.system()} py{platform.python_version()} | "
        f"{shape} finest-{1 << k} stride={stride}\n  ms: {STATS_HEAD}\n      {stats_row(st)}"
    )
    assert st["p50"] < 6.0 * (n / 256) ** 2


def interleaved(fns, reps=100, warm=10):
    """Per-call ms stats for several callables, timed round-robin so that a change of clock
    or load during the run hits all of them alike: rows of one table can be compared."""
    for fn in fns.values():
        for _ in range(warm):
            fn()
    t = {name: np.empty(reps) for name in fns}
    for i in range(reps):
        for name, fn in fns.items():
            t0 = time.perf_counter()
            fn()
            t[name][i] = time.perf_counter() - t0
    out = {}
    for name, v in t.items():
        v *= 1e3
        p50, p90, p99 = np.percentile(v, [50, 90, 99])
        out[name] = {
            "min": v.min(), "mean": v.mean(), "std": v.std(), "p50": p50, "p90": p90, "p99": p99
        }
    return out


def test_colour_conversion_kernel():
    """The conversion on its own against cv2.cvtColor: the same bytes (asserted), and the time
    per pixel of the two kernels on one thread each, then cvtColor as a host would call it,
    on OpenCV's own thread pool."""
    cv2 = pytest.importorskip("cv2")
    n = 1024
    bgr = frame((n, n, 3))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    assert np.array_equal(imfeat.convert(bgr), hsv)  # bit-exact on this build (test_color.py
    cv_threads = cv2.getNumThreads()  # has the full check, over every colour)
    head(f"BGR -> HSV alone: {n}x{n}x3 to a new image (ms; cv2 default {cv_threads} threads)")
    try:
        cv2.setNumThreads(1)
        one = interleaved(
            {
                "cv2.cvtColor, 1 thread": lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                "imfeat.convert, 1 thread": lambda: imfeat.convert(bgr),
            }
        )
    finally:
        cv2.setNumThreads(cv_threads)
    many = interleaved(
        {f"cv2.cvtColor, {cv_threads} threads": lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)}
    )
    print(f"  {'':28s} {STATS_HEAD}   ns/px (min)")
    for name, st in {**one, **many}.items():
        print(f"  {name:28s} {stats_row(st)}   {1e6 * st['min'] / (n * n):6.3f}")
    ratio = one["imfeat.convert, 1 thread"]["min"] / one["cv2.cvtColor, 1 thread"]["min"]
    print(f"  per thread, imfeat's kernel takes {ratio:.2f}x cv2's time")


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_colour_conversion(threads):
    """A BGR frame to HSV features: the conversion fused into the pass, against cvtColor (on
    OpenCV's own threads, as a host would call it) followed by the pass on its output. The
    rows are timed round-robin, so their differences mean something: fused minus the pass on
    HSV is what the conversion costs inside the pass."""
    cv2 = pytest.importorskip("cv2")
    n, k = 1024, 6
    bgr = frame((n, n, 3))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    fused = imfeat.FeatureComputer(bgr.shape, grid=pyramid(k), threads=threads)
    asis = imfeat.FeatureComputer(bgr.shape, grid=pyramid(k), threads=threads, feature_space=None)
    cv_threads = cv2.getNumThreads()
    head(
        f"BGR -> HSV features: {n}x{n}x3 finest-{1 << k} stride=1, threads={fused.threads}"
        f" (ms; cvtColor on cv2's {cv_threads} threads)"
    )
    rows = interleaved(
        {
            "cv2.cvtColor": lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
            "pass on HSV": lambda: asis.features(hsv),
            "cvtColor + pass": lambda: asis.features(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)),
            "fused pass on BGR": lambda: fused.features(bgr),
        }
    )
    print(f"  {'':18s} {STATS_HEAD}")
    for name, st in rows.items():
        print(f"  {name:18s} {stats_row(st)}")
    inside = {c: rows["fused pass on BGR"][c] - rows["pass on HSV"][c] for c in ("min", "p50")}
    saved = {c: rows["cvtColor + pass"][c] - rows["fused pass on BGR"][c] for c in ("min", "p50")}
    print(
        f"  conversion inside the pass: {inside['min']:.3f} ms (min) {inside['p50']:.3f} (p50);"
        f" fused saves {saved['min']:.3f} ms (min) {saved['p50']:.3f} (p50) over cvtColor + pass"
    )


def test_resize_kernel():
    """The thumbnail resize on its own against cv2.resize(INTER_AREA): the same bytes
    (asserted; test_resize.py has the full check), then the time of the two on one thread,
    of imfeat's on two, and of cv2's as a host would call it, on OpenCV's own thread pool."""
    cv2 = pytest.importorskip("cv2")
    src, n = (1080, 1920), 1024
    bgr = frame((*src, 3))
    small = cv2.resize(bgr, (n, n), interpolation=cv2.INTER_AREA)
    assert np.array_equal(imfeat.resize_area(bgr, n), small)
    cv_threads = cv2.getNumThreads()
    head(
        f"INTER_AREA resize alone: {src[1]}x{src[0]}x3 -> {n}x{n}"
        f" (ms; cv2 default {cv_threads} threads)"
    )
    try:
        cv2.setNumThreads(1)
        one = interleaved(
            {
                "cv2.resize, 1 thread": lambda: cv2.resize(
                    bgr, (n, n), interpolation=cv2.INTER_AREA
                ),
                "imfeat.resize_area, 1 thread": lambda: imfeat.resize_area(bgr, n),
                "imfeat.resize_area, 2 threads": lambda: imfeat.resize_area(bgr, n, threads=2),
            }
        )
    finally:
        cv2.setNumThreads(cv_threads)
    many = interleaved(
        {
            f"cv2.resize, {cv_threads} threads": lambda: cv2.resize(
                bgr, (n, n), interpolation=cv2.INTER_AREA
            )
        }
    )
    print(f"  {'':30s} {STATS_HEAD}")
    for name, st in {**one, **many}.items():
        print(f"  {name:30s} {stats_row(st)}")
    ratio = one["imfeat.resize_area, 1 thread"]["min"] / one["cv2.resize, 1 thread"]["min"]
    print(f"  per thread, imfeat's resize takes {ratio:.2f}x cv2's time")


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_resize_in_pass(threads):
    """A 1080p BGR frame to HSV features on a 1024 px thumbnail: the resize fused into the
    pass, against cv2.resize (on OpenCV's own threads, as a host would call it) followed by
    the pass on its output. Timed round-robin, so the differences mean something: fused minus
    the pass on the thumbnail is what the resize costs inside the pass."""
    cv2 = pytest.importorskip("cv2")
    src, n, k = (1080, 1920), 1024, 6
    bgr = frame((*src, 3))
    small = cv2.resize(bgr, (n, n), interpolation=cv2.INTER_AREA)
    on_thumb = imfeat.FeatureComputer(small.shape, grid=pyramid(k), threads=threads)
    on_frame = imfeat.FeatureComputer(bgr.shape, grid=pyramid(k), threads=threads, thumb=n)
    out = np.empty((n, n, 3), np.uint8)
    cv_threads = cv2.getNumThreads()
    head(
        f"resize + HSV features: {src[1]}x{src[0]}x3 -> {n}x{n} finest-{1 << k} stride=1,"
        f" threads={on_frame.threads} (ms; cv2.resize on cv2's {cv_threads} threads)"
    )
    rows = interleaved(
        {
            "cv2.resize": lambda: cv2.resize(bgr, (n, n), interpolation=cv2.INTER_AREA),
            "pass on the thumbnail": lambda: on_thumb.features(small),
            "cv2.resize + pass": lambda: on_thumb.features(
                cv2.resize(bgr, (n, n), interpolation=cv2.INTER_AREA)
            ),
            "fused pass on the frame": lambda: on_frame.features(bgr),
            "fused + thumb_out": lambda: on_frame.features(bgr, thumb_out=out),
        }
    )
    print(f"  {'':24s} {STATS_HEAD}")
    for name, st in rows.items():
        print(f"  {name:24s} {stats_row(st)}")
    fused, alone = rows["fused pass on the frame"], rows["pass on the thumbnail"]
    both = rows["cv2.resize + pass"]
    inside = {c: fused[c] - alone[c] for c in ("min", "p50")}
    saved = {c: both[c] - fused[c] for c in ("min", "p50")}
    print(
        f"  resize inside the pass: {inside['min']:.3f} ms (min) {inside['p50']:.3f} (p50);"
        f" fused saves {saved['min']:.3f} ms (min) {saved['p50']:.3f} (p50) over cv2.resize + pass"
    )


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_resize_policy(threads):
    """The thumbnail policies against a fixed 1024 px thumbnail, frame by frame: HSV features
    on a 720p, a 1080p and a 4K frame at the size each policy picks for it, fused, against the
    fixed size on the same frame -- fused too where it downscales, and on the 720p frame cv2's
    bilinear upscale (an upscale is not INTER_AREA) followed by the pass, as a host does today.
    Timed round-robin per frame. A policy costs nothing per frame: the size is settled when
    the computer is built, so the rows differ only in the size the pass runs at ("pow2-cover"
    keeps the frame's shape at the "pow2" height, so it has more pixels; "pow2-fit" keeps it at
    the "pow2" width, so fewer) -- and in its cell width: a power of two (the square, and
    "pow2-fit" on these landscape frames) tiles the vector blocks, anything else ("pow2-cover"
    here) walks a masked block per cell and costs more per pixel. 'build' is the one-off
    construction, what a host pays when the frame shape changes (it keeps one computer per
    shape); 'Mpx' the thumbnail's pixels."""
    cv2 = pytest.importorskip("cv2")
    fixed, k = 1024, 6
    cv_threads = cv2.getNumThreads()
    head(
        f"thumbnail policies vs a fixed {fixed} px thumbnail: HSV features, finest-{1 << k}"
        f" stride=1, threads={threads} (ms; cv2.resize on cv2's {cv_threads} threads)"
    )
    print(f"  {'':52s} {STATS_HEAD}   build   Mpx")
    for src in ((720, 1280), (1080, 1920), (2160, 3840)):
        bgr = frame((*src, 3))
        kw = dict(grid=pyramid(k), threads=threads)
        if min(src) >= fixed:  # the fixed size downscales: fused, as today
            build = [latency(lambda: imfeat.FeatureComputer(bgr.shape, thumb=fixed, **kw), 5, 1)]
            on_frame = imfeat.FeatureComputer(bgr.shape, thumb=fixed, **kw)
            fns = {f"fixed {fixed}: fused pass": lambda: on_frame.features(bgr)}
        else:  # the fixed size upscales: cv2 (bilinear) then the pass on its output
            build = [latency(lambda: imfeat.FeatureComputer((fixed, fixed, 3), **kw), 5, 1)]
            on_thumb = imfeat.FeatureComputer((fixed, fixed, 3), **kw)
            fns = {
                f"fixed {fixed}: cv2.resize up (bilinear) + pass": lambda: on_thumb.features(
                    cv2.resize(bgr, (fixed, fixed))
                )
            }
        pixels = [fixed * fixed]
        for policy in imfeat.THUMB_POLICIES:
            build.append(
                latency(lambda: imfeat.FeatureComputer(bgr.shape, thumb=policy, **kw), 5, 1)
            )
            fc = imfeat.FeatureComputer(bgr.shape, thumb=policy, **kw)
            rows_, cols = fc.thumb
            assert fc.thumb == imfeat.thumb_size(bgr.shape, policy) and fc.thumb_policy == policy
            same = " (the same size)" if fc.thumb == (fixed, fixed) else ""
            fns[f"{policy} -> {cols}x{rows_}: fused pass{same}"] = lambda fc=fc: fc.features(bgr)
            pixels.append(rows_ * cols)
        rows = interleaved(fns, reps=40, warm=5)
        for (name, st), b, px in zip(rows.items(), build, pixels):
            label = f"{src[1]}x{src[0]:<5d} {name}"
            print(f"  {label:52s} {stats_row(st)}   {b['min']:5.1f}  {px / 1e6:5.2f}")
        assert fc.features(bgr).maps[0].shape == (1 << k, 1 << k, 3 * len(imfeat.FEATURE_NAMES))
