"""The thumbnail resize fused into the pass, bit for bit cv2.resize(INTER_AREA)'s.

Two halves. ``imfeat.resize_area`` is the row kernel the pass fuses, on its own: pinned to
``cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)`` on frames of many sizes,
contents, layouts and channel counts, through each of OpenCV's three paths (float taps,
whole-number ratios, the same size) and at several thread counts. Then the pass: a
``FeatureComputer(shape=frame.shape, thumb=...)`` on the frame gives, byte for byte, what
one on the thumbnail gives for cv2's thumbnail -- ``features()`` and ``compute()``, at every
stride, thread count, channel selection, with and without the colour conversion -- and hands
back that thumbnail through ``thumb_out``.

OpenCV is the reference; without cv2 the file is skipped rather than compared to a copy of
the algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

import imfeat

cv2 = pytest.importorskip("cv2")

GRID = [(6, 6), (5, 5), (4, 4)]
THUMB = (1024, 1024)


def cv_resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rows, cols = size
    return cv2.resize(np.ascontiguousarray(img), (cols, rows), interpolation=cv2.INTER_AREA)


def content(kind: str, shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Frames that exercise the rounding: noise, hard edges and gradients, the ends of the
    range, few levels (many exact halves in the whole-number path), and a constant."""
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    if kind == "random":
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if kind == "structured":
        yy, xx = np.mgrid[0:h, 0:w]
        planes = [
            (xx * 255 // max(w - 1, 1)).astype(np.uint8),
            (yy * 255 // max(h - 1, 1)).astype(np.uint8),
            (((xx // 7 + yy // 5) % 2) * 255).astype(np.uint8),
            ((xx * yy) % 251).astype(np.uint8),
        ]
        c = 1 if len(shape) == 2 else shape[2]
        img = np.stack([planes[k % 4] for k in range(c)], -1)
        return np.ascontiguousarray(img[:, :, 0] if len(shape) == 2 else img)
    if kind == "dark":
        return rng.integers(0, 8, shape, dtype=np.uint8)
    if kind == "bright":
        return rng.integers(248, 256, shape, dtype=np.uint8)
    if kind == "coarse":
        return (rng.integers(0, 16, shape, dtype=np.uint8) * 17).astype(np.uint8)
    if kind == "constant":
        return np.full(shape, 173, np.uint8)
    raise ValueError(kind)


def assert_same_outputs(a: imfeat.Pyramid, b: imfeat.Pyramid) -> None:
    for name in ("maps", "moments", "summary", "cross"):
        got, want = getattr(a, name), getattr(b, name)
        assert len(got) == len(want), name
        for i, (x, y) in enumerate(zip(got, want)):
            assert x.dtype == y.dtype and x.shape == y.shape, f"{name}[{i}]"
            assert np.array_equal(x, y), f"{name}[{i}]"
    assert np.array_equal(a.hashes, b.hashes)
    for x, y in zip(a.profiles, b.profiles):
        assert np.array_equal(x, y)


# ---------------- the kernel on its own, against cv2.resize -----------------------------------
# (source rows, cols) -> (rows, cols): OpenCV's float-tap path at the ratios of the common
# sources (1080p, 1440p, 4K, an odd size, one axis unchanged, a non-square thumbnail, a
# thumbnail the taps of 8 columns cannot share a 16-byte window at, and ratios so large that
# a single column's taps cannot either), its whole-number path, and the same size.
SIZES = [
    ((1080, 1920), THUMB),
    ((1440, 2560), THUMB),
    ((2160, 3840), THUMB),
    ((1079, 1917), THUMB),
    ((1025, 1024), THUMB),
    ((1050, 3000), (1024, 640)),
    ((1080, 1920), (512, 512)),
    ((1080, 1920), (256, 1024)),
    ((300, 500), (7, 13)),
    ((2048, 2048), THUMB),
    ((3072, 4096), THUMB),
    ((1024, 1024), THUMB),
]


@pytest.mark.parametrize("channels", [3, 1])
@pytest.mark.parametrize(
    ("src", "size"), SIZES, ids=[f"{s[1]}x{s[0]}->{d[1]}x{d[0]}" for s, d in SIZES]
)
def test_resize_area_matches_opencv(src, size, channels):
    for kind in ("random", "structured", "dark", "bright", "coarse", "constant"):
        img = content(kind, (*src, channels) if channels > 1 else src)
        want = cv_resize(img, size)
        for threads in (1, 2, 3):
            got = imfeat.resize_area(img, size, threads=threads)
            assert got.shape == want.shape and got.dtype == np.uint8
            assert np.array_equal(got, want), (kind, threads)


@pytest.mark.parametrize("channels", [2, 4])
def test_resize_area_other_channel_counts(channels):
    """2 and 4 channels, on both paths: the whole-number 2x2 rule rounds half up for 1, 3 and
    4 channels and through a float product for 2, as OpenCV's."""
    for src in ((1080, 1920), (2048, 2048)):
        img = content("random", (*src, channels))
        assert np.array_equal(imfeat.resize_area(img, THUMB), cv_resize(img, THUMB))
        img = content("coarse", (*src, channels), seed=3)
        assert np.array_equal(imfeat.resize_area(img, THUMB, threads=2), cv_resize(img, THUMB))


def test_resize_area_reads_any_layout():
    img = content("random", (1200, 2200, 3))
    want = cv_resize(img[:, 100:2020], THUMB)
    assert np.array_equal(imfeat.resize_area(img[:, 100:2020], THUMB), want)  # a window
    planar = np.moveaxis(np.ascontiguousarray(np.moveaxis(img, -1, 0)), 0, -1)  # (C, H, W) view
    assert np.array_equal(imfeat.resize_area(planar, THUMB), cv_resize(img, THUMB))
    fortran = np.asfortranarray(img)  # every stride unusual: the scalar gather
    assert np.array_equal(imfeat.resize_area(fortran, THUMB, threads=2), cv_resize(img, THUMB))
    gray = np.ascontiguousarray(img[::2, ::3, 1])
    assert np.array_equal(
        imfeat.resize_area(img[::2, ::3, 1], (256, 512)), cv_resize(gray, (256, 512))
    )
    assert np.array_equal(imfeat.resize_area(img, 512), cv_resize(img, (512, 512)))  # an int


def test_resize_area_arguments():
    img = content("random", (720, 1280, 3))
    with pytest.raises(ValueError, match="downscales only"):
        imfeat.resize_area(img, THUMB)  # 720 rows up to 1024: bilinear in OpenCV
    with pytest.raises(ValueError, match="downscales only"):
        imfeat.resize_area(img, (720, 1281))
    with pytest.raises(ValueError, match="1 to 4 channels"):
        imfeat.resize_area(np.zeros((16, 16, 5), np.uint8), (8, 8))
    with pytest.raises(TypeError, match="uint8"):
        imfeat.resize_area(np.zeros((16, 16, 3), np.float32), (8, 8))
    with pytest.raises(ValueError, match=r"\(H, W\) or \(H, W, C\)"):
        imfeat.resize_area(np.zeros((16, 16, 3, 1), np.uint8), (8, 8))
    with pytest.raises(ValueError, match="rows, cols"):
        imfeat.resize_area(img, (8, 8, 8))
    with pytest.raises(ValueError, match="at least one"):
        imfeat.resize_area(img, (0, 8))


# ---------------- the pass on a frame, against the pass on cv2's thumbnail ---------------------
@pytest.fixture(scope="module")
def frame_1080p() -> np.ndarray:
    return content("random", (1080, 1920, 3), seed=1)


@pytest.mark.parametrize("feature_space", ["hsv", None])
@pytest.mark.parametrize("channels", [None, [2], [0, 0, 2]])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("stride", [1, 2, (1, 3)])
def test_pass_resizes_exactly(frame_1080p, stride, threads, channels, feature_space):
    frame = frame_1080p
    thumb = cv_resize(frame, THUMB)
    kw = dict(
        grid=GRID, stride=stride, channels=channels, threads=threads, feature_space=feature_space
    )
    on_thumb = imfeat.FeatureComputer(shape=thumb.shape, **kw)
    on_frame = imfeat.FeatureComputer(shape=frame.shape, thumb=THUMB, **kw)
    assert on_frame.thumb == THUMB and on_thumb.thumb is None
    out = np.empty((*THUMB, 3), np.uint8)
    assert_same_outputs(on_frame.features(frame, thumb_out=out), on_thumb.features(thumb))
    assert np.array_equal(out, thumb)
    assert_same_outputs(on_frame.features(frame), on_thumb.features(thumb))  # no thumbnail out
    raw_frame, raw_thumb = on_frame.compute(frame), on_thumb.compute(thumb)
    assert raw_frame.keys() == raw_thumb.keys()
    for k, v in raw_frame.items():
        assert np.array_equal(v, raw_thumb[k]), k
    out.fill(0)
    on_frame.compute(frame, thumb_out=out)
    assert np.array_equal(out, thumb)


@pytest.mark.parametrize(
    ("src", "thumb", "grid"),
    [
        ((1440, 2560), (512, 512), [(5, 5), (4, 4)]),  # 5 and 2.8: taps four columns to a window
        ((1080, 1920), (256, 1024), [(4, 6), (3, 5)]),  # a non-square thumbnail
        ((2048, 2048), THUMB, GRID),  # the whole-number path
        ((1024, 1024), THUMB, GRID),  # the same size
    ],
)
def test_pass_resizes_other_shapes(src, thumb, grid):
    frame = content("structured", (*src, 3))
    small = cv_resize(frame, thumb)
    on_thumb = imfeat.FeatureComputer(shape=small.shape, grid=grid, stride=2, threads=2)
    on_frame = imfeat.FeatureComputer(
        shape=frame.shape, grid=grid, stride=2, threads=2, thumb=thumb
    )
    out = np.empty((*thumb, 3), np.uint8)
    assert_same_outputs(on_frame.features(frame, thumb_out=out), on_thumb.features(small))
    assert np.array_equal(out, small)


def test_pass_resizes_grayscale_and_planar_frames():
    gray = content("random", (1080, 1920), seed=2)
    small = cv_resize(gray, THUMB)
    on_thumb = imfeat.FeatureComputer(shape=small.shape, grid=GRID, threads=2)
    on_frame = imfeat.FeatureComputer(shape=gray.shape, grid=GRID, threads=2, thumb=1024)
    out = np.empty(THUMB, np.uint8)
    assert_same_outputs(on_frame.features(gray, thumb_out=out), on_thumb.features(small))
    assert np.array_equal(out, small)

    bgr = content("random", (1080, 1920, 3), seed=4)
    planar = np.ascontiguousarray(np.moveaxis(bgr, -1, 0))  # (3, H, W)
    small = cv_resize(bgr, THUMB)
    on_thumb = imfeat.FeatureComputer(shape=small.shape, grid=GRID, threads=2)
    on_frame = imfeat.FeatureComputer(
        shape=planar.shape, grid=GRID, threads=2, channel_axis=0, thumb=THUMB
    )
    out = np.empty((*THUMB, 3), np.uint8)
    assert_same_outputs(on_frame.features(planar, thumb_out=out), on_thumb.features(small))
    assert np.array_equal(out, small)


def test_thumbnail_arguments(frame_1080p):
    frame = frame_1080p
    with pytest.raises(ValueError, match="downscales only"):
        imfeat.FeatureComputer(shape=(720, 1280, 3), grid=GRID, thumb=1024)
    with pytest.raises(ValueError, match="finest grid must divide"):
        imfeat.FeatureComputer(shape=frame.shape, grid=GRID, thumb=1000)
    with pytest.raises(ValueError, match="rows, cols"):
        imfeat.FeatureComputer(shape=frame.shape, grid=GRID, thumb=(1024,))
    with pytest.raises(ValueError, match="up to 4 channels"):
        imfeat.FeatureComputer(shape=(64, 64, 5), grid=[(2, 2)], thumb=32, feature_space=None)
    fc = imfeat.FeatureComputer(shape=frame.shape, grid=GRID, thumb=1024)
    assert fc.thumb == THUMB
    with pytest.raises(ValueError, match="shape mismatch"):
        fc.features(cv_resize(frame, THUMB))  # the frame, not the thumbnail, goes in
    with pytest.raises(ValueError, match="must have shape"):
        fc.features(frame, thumb_out=np.empty(THUMB, np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        fc.features(frame, thumb_out=np.empty((*THUMB, 3), np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        fc.features(frame, thumb_out=np.empty((*THUMB, 3), np.uint8)[:, ::-1])
    plain = imfeat.FeatureComputer(shape=(*THUMB, 3), grid=GRID)
    with pytest.raises(ValueError, match="built with thumb"):
        plain.features(cv_resize(frame, THUMB), thumb_out=np.empty((*THUMB, 3), np.uint8))
