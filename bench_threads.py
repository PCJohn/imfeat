"""Thread scaling for imfeat, per-frame latency.

    python bench_threads.py

Reports min / p50 / p95 ms per frame at each thread count. p95 is the one that
matters for a real-time stream: a good median with a bad tail still drops frames.
Also asserts bit-exactness against threads=1 before timing, so a fast-but-wrong
build cannot look good.
"""

from __future__ import annotations

import time

import numpy as np

import imfeat

REPS, WARM = 200, 30
THREADS = [1, 2, 3, 4]
CONFIGS = [  # (label, size, finest grid k, stride)
    ("256x256x3  finest-32  s2", 256, 5, 2),
    ("512x512x3  finest-32  s2", 512, 5, 2),
    ("1024x1024x3 finest-64 s2", 1024, 6, 2),
    ("1024x1024x3 finest-32 s4", 1024, 5, 4),
]


def frame(n: int) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    base = 90 + 60 * np.sin(y / 7.0) * np.cos(x / 11.0) + 40 * ((x // 9 + y // 5) % 2)
    a = base[..., None].repeat(3, 2) + np.random.default_rng(0).integers(-25, 26, (n, n, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def latency(fc: imfeat.FeatureComputer, img: np.ndarray) -> tuple[float, float, float]:
    for _ in range(WARM):
        fc.features(img)
    t = []
    for _ in range(REPS):
        a = time.perf_counter()
        fc.features(img)
        t.append(time.perf_counter() - a)
    v = np.array(t) * 1e3
    return float(v.min()), float(np.percentile(v, 50)), float(np.percentile(v, 95))


print(f"imfeat.cpu_count() = {imfeat.cpu_count()}   reps={REPS} warmup={WARM}\n")
for label, n, k, stride in CONFIGS:
    img = frame(n)
    grid = [(k - i, k - i) for i in range(4)]
    ref = imfeat.FeatureComputer(img.shape, grid=grid, stride=stride).features(img)
    ref = {key: np.array(val) for key, val in ref.items()}
    print(label)
    base = None
    for nt in THREADS:
        fc = imfeat.FeatureComputer(img.shape, grid=grid, stride=stride, threads=nt)
        out = fc.features(img)
        for key in ref:  # exactness first; a wrong build must not be timed
            assert np.array_equal(ref[key], np.asarray(out[key])), f"{label} threads={nt} {key}"
        mn, p50, p95 = latency(fc, img)
        base = base or mn
        print(
            f"  threads={fc.threads}  min={mn:7.3f}  p50={p50:7.3f}  p95={p95:7.3f} ms"
            f"   {base / mn:5.2f}x"
        )
    print()
