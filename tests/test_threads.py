"""Threading must not change a single bit.

Bands are disjoint runs of finest cell rows and every accumulator is an int64 sum,
so any thread count has to reproduce the serial result exactly -- including the
derived floats, since derive() still runs serially on identical integer inputs.
Covers strides that do and do not divide the cell height, grids too shallow to
split (ny < threads), the fast and slow hash paths, and C in {1, 3, 4}.
"""

from __future__ import annotations

import numpy as np
import pytest

import imfeat

CONFIGS = [
    ((128, 128, 3), [(5, 5), (4, 4), (3, 3), (2, 2)], 2),
    ((128, 128, 3), [(5, 5), (4, 4), (3, 3), (2, 2)], 3),
    ((128, 128, 3), [(4, 4)], 1),
    ((96, 64, 3), [(3, 3), (2, 2)], 2),
    ((64, 64), [(3, 3)], 1),
    ((64, 64, 1), [(1, 1)], 2),  # ny = 2, far fewer cell rows than threads
    ((64, 64, 4), [(2, 2)], 1),
    ((64, 64, 3), [(0, 0)], 1),  # single cell: unsplittable, must stay correct
]


def _img(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = shape[0], shape[1]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 90 + 60 * np.sin(y / 7.0) * np.cos(x / 11.0) + 40 * ((x // 9 + y // 5) % 2)
    a = base[..., None] if len(shape) == 3 else base
    return np.clip(a + rng.integers(-25, 26, shape), 0, 255).astype(np.uint8)


def _outputs(shape, grid, stride, threads):
    fc = imfeat.FeatureComputer(shape, grid=grid, stride=stride, threads=threads)
    img = _img(shape, 7)
    return fc, {**fc.compute(img), **fc.features(img)}


@pytest.mark.parametrize("shape,grid,stride", CONFIGS)
@pytest.mark.parametrize("threads", [2, 3, 4, 8])
def test_threads_are_bit_exact(shape, grid, stride, threads):
    _, want = _outputs(shape, grid, stride, 1)
    fc, got = _outputs(shape, grid, stride, threads)
    assert fc.threads == min(threads, 2 ** grid[0][0])
    assert set(want) == set(got)
    for k in want:
        a, b = np.asarray(want[k]), np.asarray(got[k])
        assert (a.dtype, a.shape) == (b.dtype, b.shape), k
        assert np.array_equal(a, b), k


def test_repeated_frames_do_not_drift():
    """The pool is reused across frames; a stale barrier would show up as a
    difference on the second or third call, not the first."""
    fc = imfeat.FeatureComputer((128, 128, 3), grid=[(4, 4), (3, 3)], stride=2, threads=4)
    imgs = [_img((128, 128, 3), s) for s in range(3)]
    first = [{k: np.array(v) for k, v in fc.features(im).items()} for im in imgs]
    for _ in range(3):
        for im, ref in zip(imgs, first):
            out = fc.features(im)
            for k in ref:
                assert np.array_equal(ref[k], np.asarray(out[k])), k


def test_cpu_count_is_sane():
    assert imfeat.cpu_count() >= 1
