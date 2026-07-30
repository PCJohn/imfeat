"""Where does a frame go, and which part refuses to scale?

    python profile_phases.py

Splits each frame into the band-parallel pass and everything after the join, at
each thread count. compute() runs accumulate only; features() adds derive, the
hash readout and the Python-side slicing. The gap between how well those two
scale localises the remaining serial time without a profiler.

Read it like this:
  * accumulate scales well, features does not  -> serial tail (fold_summary,
    rollup, hashes, _cut). Parallelising the fold is the next lever.
  * neither scales                             -> memory bandwidth, or bands
    landing on efficiency cores.
  * scaling collapses only at 3-4 threads      -> E-cores. Compare against the
    P-core count rather than cpu_count().
"""

from __future__ import annotations

import time

import numpy as np

import imfeat

REPS, WARM = 200, 30
THREADS = [1, 2, 3, 4]
CONFIGS = [
    ("256x256x3   finest-32 s2", 256, 5, 2),
    ("1024x1024x3 finest-64 s2", 1024, 6, 2),
    ("1024x1024x3 finest-32 s4", 1024, 5, 4),
]


def frame(n: int) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    base = 90 + 60 * np.sin(y / 7.0) * np.cos(x / 11.0) + 40 * ((x // 9 + y // 5) % 2)
    a = base[..., None].repeat(3, 2) + np.random.default_rng(0).integers(-25, 26, (n, n, 3))
    return np.clip(a, 0, 255).astype(np.uint8)


def best(fn) -> float:
    for _ in range(WARM):
        fn()
    return min(_timed(fn) for _ in range(REPS)) * 1e3


def _timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


print(f"imfeat.cpu_count() = {imfeat.cpu_count()}   min of {REPS} reps, ms\n")
for label, n, k, stride in CONFIGS:
    img = frame(n)
    grid = [(k - i, k - i) for i in range(4)]
    print(f"{label}")
    print(f"  {'':9s} {'accumulate':>11s} {'scaling':>8s} {'+tail':>8s} "
          f"{'total':>8s} {'scaling':>8s}")
    ref = {}
    for nt in THREADS:
        fc = imfeat.FeatureComputer(img.shape, grid=grid, stride=stride, threads=nt)
        view = fc._view(img)
        acc = best(lambda: fc._impl.raw(view))  # accumulate + rollup only
        tot = best(lambda: fc.features(img))  # + derive, hashes, Python slicing
        ref.setdefault("acc", acc)
        ref.setdefault("tot", tot)
        print(
            f"  threads={fc.threads}  {acc:11.3f} {ref['acc'] / acc:7.2f}x "
            f"{tot - acc:8.3f} {tot:8.3f} {ref['tot'] / tot:7.2f}x"
        )
    print()
