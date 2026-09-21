"""features() hands out views of a pooled block: they must behave like arrays that own their data."""

from __future__ import annotations

import gc

import numpy as np

import imfeat

from conftest import frame

SHAPE = (64, 96, 3)


def arrays(p: imfeat.Pyramid) -> list[np.ndarray]:
    return [*p.maps, *p.moments, *p.summary, *p.cross, p.hashes]


def computer() -> imfeat.FeatureComputer:
    return imfeat.FeatureComputer(SHAPE, grid=[(3, 3), (1, 1)], stride=2)


def test_results_survive_later_calls():
    fc = computer()
    held = [fc.features(frame(SHAPE, seed)) for seed in range(8)]  # more than the pool keeps idle
    for seed, p in enumerate(held):
        for a, b in zip(arrays(p), arrays(computer().features(frame(SHAPE, seed)))):
            np.testing.assert_array_equal(a, b)


def test_recycled_block_carries_nothing_over():
    fc = computer()
    for a in arrays(fc.features(frame(SHAPE, 1))):
        a.fill(7)  # results are writable; scribble over the block before it goes back
    gc.collect()
    got, want = fc.features(frame(SHAPE, 2)), computer().features(frame(SHAPE, 2))
    for a, b in zip(arrays(got), arrays(want)):
        np.testing.assert_array_equal(a, b)


def test_results_outlive_their_computer():
    fc = computer()
    p = fc.features(frame(SHAPE, 3))
    want = [a.copy() for a in arrays(p)]
    one = p.maps[0]  # a lone array keeps the whole block alive
    del fc, p
    gc.collect()
    np.testing.assert_array_equal(one, want[0])
    one += 1  # still ours to write
    np.testing.assert_array_equal(one, want[0] + 1)
