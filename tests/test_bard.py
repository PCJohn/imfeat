"""Bar-detector sums against the numpy oracle's dense maps -- exact, on every kernel geometry."""

from __future__ import annotations

import numpy as np
import pytest

import imfeat

from conftest import frame

pytest.importorskip("cv2")  # bard_oracle imports it for its pooling
from bard_oracle import TAU, maps  # noqa: E402

BARD0 = 30  # first bard slot of the raw sums: cover, response mass per lag, dark, light


def bars(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Thin dark bars on a noisy bright field, so the detector fires right up to the borders."""
    y, x = np.indices(shape)
    img = np.where((x % 5 == 0) | (y % 7 == 0), 40, 200) + frame(shape, seed) // 8
    return img.astype(np.uint8)


def oracle(gray: np.ndarray, ny: int, nx: int, sy: int, sx: int) -> np.ndarray:
    """(ny, nx, 3 + lags) int64: the gated maps summed over each cell's sampling mesh."""
    dark, light = maps(gray)
    resp = np.maximum(dark, light)
    h, w = gray.shape
    mesh = np.zeros((h, w), bool)
    mesh[::sy] = (np.arange(w) % (w // nx)) % sx == 0  # columns restart at each cell edge
    gate = mesh & (resp.max(0) >= TAU)
    vals = [gate, *(resp * gate), dark.max(0) * gate, light.max(0) * gate]
    cells = [v.reshape(ny, h // ny, nx, w // nx).sum((1, 3)) for v in vals]
    return np.stack(cells, -1).astype(np.int64)


@pytest.mark.parametrize(
    "shape,grid,stride",
    [
        ((256, 256), (5, 5), 1),  # blocks span several cells
        ((256, 256), (5, 5), 2),  # ... of two column phases
        ((256, 256), (3, 3), 8),  # eight phases: every lag lands in another plane
        ((256, 256), (6, 6), 2),  # 4 px cells: masked mesh
        ((256, 256), (2, 2), 2),  # cells wider than a 128-bit block
        ((64, 128), (0, 0), 2),  # ... and than a 256-bit one
        ((96, 160), (3, 5), (2, 1)),  # 5 px cells: one masked block per cell
        ((240, 320), (4, 4), 4),  # 20 px cells
        ((64, 64), (3, 3), 3),  # stride does not divide the cell
        ((8, 8), (1, 1), 1),  # lag 4 spans the image: dropped on both axes
        ((4, 16), (0, 2), 1),  # lags 2 and 4 dropped
        ((512, 64), (0, 3), (5, 2)),  # one cell row taller than a fold chunk
    ],
)
@pytest.mark.parametrize("channels", [1, 3])
def test_bard_sums_match_oracle(shape, grid, stride, channels):
    sy, sx = (stride, stride) if isinstance(stride, int) else stride
    planes = [bars(shape, seed) for seed in range(channels)]
    img = planes[0] if channels == 1 else np.stack(planes, -1)
    fc = imfeat.FeatureComputer(img.shape, grid=[grid], stride=stride)
    raw = fc._impl.raw(fc._view(img))[0][0][..., BARD0 : BARD0 + 3 + len(imfeat.BARD_FEATURES) - 4]
    for k, gray in enumerate(planes):
        np.testing.assert_array_equal(
            raw[:, :, k], oracle(gray, 1 << grid[0], 1 << grid[1], sy, sx)
        )


def test_bard_sums_are_thread_invariant():
    img = np.stack([bars((96, 160), seed) for seed in range(3)], -1)
    runs = [
        imfeat.FeatureComputer(img.shape, grid=[(4, 4), (2, 2)], stride=2, threads=t)._impl.raw(img)
        for t in (1, 2, 3)
    ]
    for other in runs[1:]:
        for a, b in zip(runs[0][0], other[0]):
            np.testing.assert_array_equal(a, b)
