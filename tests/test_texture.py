"""Laplacian, Laws energies and projection profiles against numpy -- the sums exactly."""

from __future__ import annotations

import numpy as np
import pytest

import imfeat

from conftest import frame

LAP1 = 36  # raw slots: sum(L), sum(L^2), then the six Laws sums of squares in TEXTURE order
L3, E3, S3 = np.array([1, 2, 1]), np.array([-1, 0, 1]), np.array([-1, 2, -1])
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
LAWS = [np.outer(y, x) for y, x in ((E3, E3), (S3, S3), (L3, S3), (S3, L3), (E3, S3), (S3, E3))]
TEX0 = imfeat.FEATURE_NAMES.index("lap_var")

CASES = [
    ((256, 256), (5, 5), 1),  # blocks span several cells
    ((256, 256), (5, 5), 2),  # ... of two column phases
    ((256, 256), (3, 3), 8),  # eight phases
    ((256, 256), (6, 6), 2),  # 4 px cells: masked mesh
    ((64, 128), (0, 0), 2),  # one cell, wider than any block
    ((96, 160), (3, 5), (2, 1)),  # 5 px cells: one masked block per cell
    ((240, 320), (4, 4), 4),  # 20 px cells
    ((64, 240), (2, 4), 1),  # the row's last block overruns it
    ((64, 64), (3, 3), 3),  # stride does not divide the cell
    ((4, 16), (0, 2), 1),  # a few pixels
    ((512, 64), (0, 3), (5, 2)),  # one cell row taller than a fold chunk
]


def respond(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """3x3 correlation over replicated borders, exact."""
    h, w = gray.shape
    p = np.pad(gray.astype(np.int64), 1, mode="edge")
    return sum(mask[i, j] * p[i : i + h, j : j + w] for i in range(3) for j in range(3))


def mesh(shape, nx, sy, sx) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[::sy] = (np.arange(shape[1]) % (shape[1] // nx)) % sx == 0  # columns restart per cell
    return m


def oracle(gray: np.ndarray, ny: int, nx: int, sy: int, sx: int) -> np.ndarray:
    """(ny, nx, 8) int64: sum(L), sum(L^2) and the six sums of squared Laws responses."""
    h, w = gray.shape
    lap = respond(gray, LAPLACIAN)
    vals = [lap, lap * lap, *(respond(gray, k) ** 2 for k in LAWS)]
    m = mesh(gray.shape, nx, sy, sx)
    return np.stack([(v * m).reshape(ny, h // ny, nx, w // nx).sum((1, 3)) for v in vals], -1)


def strides(stride) -> tuple[int, int]:
    return (stride, stride) if isinstance(stride, int) else stride


@pytest.mark.parametrize("shape,grid,stride", CASES)
@pytest.mark.parametrize("channels,threads", [(1, 1), (3, 1), (3, 2), (3, 3)])
def test_texture_sums_are_exact(shape, grid, stride, channels, threads):
    planes = [frame(shape, seed) for seed in range(channels)]
    img = planes[0] if channels == 1 else np.stack(planes, -1)
    fc = imfeat.FeatureComputer(img.shape, grid=[grid], stride=stride, threads=threads)
    raw = fc._impl.raw(fc._view(img))[0][0][..., LAP1 : LAP1 + 8]
    for k, gray in enumerate(planes):
        want = oracle(gray, 1 << grid[0], 1 << grid[1], *strides(stride))
        np.testing.assert_array_equal(raw[:, :, k], want)


@pytest.mark.parametrize("shape,grid,stride", CASES[:6])
def test_texture_features(shape, grid, stride):
    gray = frame(shape, 5)
    ny, nx = 1 << grid[0], 1 << grid[1]
    sy, sx = strides(stride)
    fc = imfeat.FeatureComputer(shape, grid=[grid], stride=stride)
    p = fc.features(gray)
    got = p.maps[0][..., TEX0 : TEX0 + len(imfeat.TEXTURE_FEATURES)].astype(np.float64)
    s = oracle(gray, ny, nx, sy, sx).astype(np.float64)
    n = mesh(shape, nx, sy, sx).reshape(ny, shape[0] // ny, nx, shape[1] // nx).sum((1, 3))
    lap_var = np.maximum(s[..., 1] / n - (s[..., 0] / n) ** 2, 0.0)
    ls, sl = s[..., 4], s[..., 5]
    energy = p.maps[0][..., imfeat.FEATURE_NAMES.index("energy")].astype(np.float64)
    want = np.stack(
        [
            lap_var,
            np.where(energy > 0, lap_var / np.maximum(energy, 1e-300), 0.0),
            *(s[..., 2 + t] / n for t in range(6)),
            np.where(ls + sl > 0, (ls - sl) / np.maximum(ls + sl, 1), 0.0),
        ],
        -1,
    )
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("shape,grid,stride", CASES)
@pytest.mark.parametrize("threads", [1, 2, 3])
def test_projection_profiles(shape, grid, stride, threads):
    img = np.stack([frame(shape, seed) for seed in range(3)], -1)
    sy, sx = strides(stride)
    fc = imfeat.FeatureComputer(img.shape, grid=[grid], stride=stride, threads=threads)
    rows, cols = fc.features(img).profiles
    m = mesh(shape, 1 << grid[1], sy, sx)
    on_rows, on_cols = m.any(1), m.any(0)
    sampled = np.where(m[..., None], img, 0).astype(np.float64)
    np.testing.assert_allclose(rows, sampled.sum(1)[on_rows] / on_cols.sum(), rtol=1e-12)
    np.testing.assert_allclose(cols, sampled.sum(0)[on_cols] / on_rows.sum(), rtol=1e-12)
