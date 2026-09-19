"""Numpy oracle for the bard bank: fastdet's dense maps + sampled pooling.

Vendored from the reference implementation; only `maps` and `pool` are used here, so the
standalone CLI harness has been dropped.
"""

import cv2
import numpy as np

LAGS = (1, 2, 4)
SPAN = 2
TAU = 8.0
EPS = 1e-9


def maps(gray, lags=LAGS):
    g = gray.astype(np.int16)
    h, w = g.shape
    dark = np.zeros((len(lags), h, w), np.uint8)
    light = np.zeros_like(dark)
    for j, d in enumerate(lags):
        if SPAN * d >= min(h, w):
            continue
        c = g[:, d : w - d]
        l = g[:, : w - SPAN * d]
        r = g[:, SPAN * d :]
        dark[j][:, d : w - d] = np.maximum(
            dark[j][:, d : w - d], np.minimum(np.maximum(l - c, 0), np.maximum(r - c, 0))
        ).astype(np.uint8)
        light[j][:, d : w - d] = np.maximum(
            light[j][:, d : w - d], np.minimum(np.maximum(c - l, 0), np.maximum(c - r, 0))
        ).astype(np.uint8)
        c = g[d : h - d, :]
        u = g[: h - SPAN * d, :]
        v = g[SPAN * d :, :]
        dark[j][d : h - d, :] = np.maximum(
            dark[j][d : h - d, :], np.minimum(np.maximum(u - c, 0), np.maximum(v - c, 0))
        ).astype(np.uint8)
        light[j][d : h - d, :] = np.maximum(
            light[j][d : h - d, :], np.minimum(np.maximum(c - u, 0), np.maximum(c - v, 0))
        ).astype(np.uint8)
    return np.stack([dark, light]).astype(np.float32)


def pool(R, grid, stride, tau=TAU):
    D, L = R[0], R[1]
    n = D.shape[0]

    def bm(s):
        return cv2.resize(s[::stride, ::stride], (grid, grid), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )

    resp = np.maximum(D, L)
    mask = (resp.max(axis=0) >= tau).astype(np.float32)
    cover = bm(mask)
    tot = np.stack([bm(resp[j] * mask) for j in range(n)], -1)
    allt = tot.sum(-1)
    spec = np.where(allt[..., None] > EPS, tot / np.maximum(allt[..., None], EPS), 0.0)
    wts = np.arange(1, n + 1, dtype=np.float32)
    peak = np.where(allt > EPS, (tot * wts).sum(-1) / np.maximum(allt, EPS), 0.0)
    mean = allt / n
    std = np.sqrt(np.maximum(((tot - mean[..., None]) ** 2).mean(-1), 0.0))
    peaked = np.where(mean > EPS, std / np.maximum(mean, EPS), 0.0)
    dk = bm(D.max(0) * mask)
    lt = bm(L.max(0) * mask)
    bal = np.where(dk + lt > EPS, (lt - dk) / np.maximum(dk + lt, EPS), 0.0)
    return np.stack([cover] + [spec[..., j] for j in range(n)] + [peak, peaked, bal], -1).astype(
        np.float32
    )
