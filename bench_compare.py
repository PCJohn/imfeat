"""imfeat benchmark: the latency/accuracy trade-off of `stride`, swept across input
sizes and channel counts.

Quality is always measured against the stride=1 output of the SAME configuration (the
exact answer), per feature group, on the finest pyramid level -- so every row compares
like with like.

Run: python bench_compare.py
"""

import time

import numpy as np

import imfeat

EXPS = [5, 4, 3, 2]  # pyramid: 32x32 finest -> 4x4 coarsest
STRIDES = [1, 2, 4, 8]
rng = np.random.default_rng(0)


def synth(n, c):
    """Deterministic multi-channel test image: texture, ramp, noise, checker + a hard
    rectangle. Mixes smooth and high-frequency content, so stride error is not
    flattered by an easy image."""
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


def bench(fn, reps=7, budget=0.05):
    """Best of `reps` batches. The mean is useless on a laptop -- one scheduler
    preemption or a clock-throttle step inflates it; the minimum is the closest
    estimate of the real cost."""
    fn()
    best = float("inf")
    for _ in range(reps):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < budget:
            fn()
            n += 1
        best = min(best, (time.perf_counter() - t0) / n * 1e3)
    return best


def _corr(a, b):
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def quality(ref, out):
    """Error of the strided output vs the exact one, finest level. Moments get
    absolute grey-level errors; the structure maps get correlations, because only
    their spatial layout is consumed downstream, not their absolute scale."""
    q = {}
    m, mr = out["mom_0"], ref["mom_0"]
    q["mean_err"] = float(np.abs(m[..., 0] - mr[..., 0]).mean())
    sd, sdr = np.sqrt(np.maximum(m[..., 1], 0)), np.sqrt(np.maximum(mr[..., 1], 0))
    q["std_err"] = float(np.abs(sd - sdr).mean())

    s, sr = out["struct_0"], ref["struct_0"]
    q["energy_r"] = _corr(s[..., 0], sr[..., 0])
    q["coh_err"] = float(np.abs(s[..., 1] - sr[..., 1]).mean())
    # orientation: angle between the double-angle vectors, weighted by the exact
    # coherence so cells with no dominant edge (angle = noise) do not pollute it
    dot = s[..., 2] * sr[..., 2] + s[..., 3] * sr[..., 3]
    nrm = np.hypot(s[..., 2], s[..., 3]) * np.hypot(sr[..., 2], sr[..., 3])
    dth = np.arccos(
        np.clip(np.where(nrm > 1e-9, dot / np.maximum(nrm, 1e-9), 1.0), -1, 1)
    )
    w = sr[..., 1]
    q["ori_deg"] = float(np.degrees(0.5 * (dth * w).sum() / max(w.sum(), 1e-9)))

    h, hr = out["hog_0"], ref["hog_0"]
    na, nb = np.linalg.norm(h, axis=-1), np.linalg.norm(hr, axis=-1)
    ok = (na > 0) & (
        nb > 0
    )  # flat cells have an all-zero histogram: no angle to compare
    cos = (h * hr).sum(-1)[ok] / (na[ok] * nb[ok])
    q["hog_cos"] = float(cos.mean()) if ok.any() else 1.0
    q["cnt_r"] = _corr(out["cnt_0"][..., 0], ref["cnt_0"][..., 0])
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
]


def table(rows, label=""):
    print(f"{label:>12}" + "".join(f"{c:>{w}}" for c, w, _ in COLS))
    for name, r in rows:
        print(f"{name:>12}" + "".join(f.format(r[c]) for c, _, f in COLS))


def sweep(n, c, strides=STRIDES):
    """Latency + quality vs stride for one (size, channels) config. Quality is
    relative to this config's own stride=1 run, so rows compare like with like."""
    img = synth(n, c)
    grid = [(e, e) for e in EXPS]
    cw = n // (
        1 << EXPS[0]
    )  # finest cell width: stride beyond this samples 1 column/cell
    ref, base, rows = None, None, []
    for s in strides:
        fc = imfeat.FeatureComputer(img.shape, grid=grid, stride=s)
        out = fc.features(img)
        if ref is None:
            ref = {k: v.copy() for k, v in out.items()}
        ms = bench(lambda fc=fc, img=img: fc.features(img))
        base = base or ms
        tag = f"stride={s}" + ("*" if s > cw else "")
        rows.append((tag, {"ms": ms, "speedup": base / ms, **quality(ref, out)}))
    return rows


def head(title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)


def main():
    T = 256
    grid = [(e, e) for e in EXPS]

    head(
        f"1. Stride: latency vs accuracy   ({T}x{T}x3)\n"
        "   Errors vs the exact stride=1 output. mean_err/std_err in grey levels;\n"
        "   ori_deg is coherence-weighted; energy_r/cnt_r/hog_cos: 1.0 = exact."
    )
    table(sweep(T, 3))

    head(
        "2. Input size   (3ch; each block's speedup/errors are vs its own stride=1)\n"
        f"   * = stride exceeds the finest cell width (n/{1 << EXPS[0]}), so only one column\n"
        "   per cell is sampled -- the knob has saturated and accuracy collapses."
    )
    for n in (128, 256, 512, 1024):
        table(sweep(n, 3), f"{n}x{n}x3")

    head(f"3. Channel count   ({T}x{T}; every feature computed on every channel)")
    for c in (1, 2, 3, 4, 8, 16):
        table(sweep(T, c), f"C={c}")

    head(f"4. Per-channel cost   ({T}x{T}, stride=2): SIMD packs 4 channels per vector")
    for c in (1, 2, 3, 4, 8, 16):
        im = synth(T, c)
        fc = imfeat.FeatureComputer(im.shape, grid=grid, stride=2)
        ms = bench(lambda fc=fc, im=im: fc.features(im))
        print(f"  C={c:>2}: {ms:7.3f} ms  ({ms / c:6.3f} ms/channel)")


if __name__ == "__main__":
    main()
