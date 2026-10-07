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
    for name in ("maps", "moments", "summary"):
        got, want = getattr(a, name), getattr(b, name)
        assert len(got) == len(want), name
        for i, (x, y) in enumerate(zip(got, want)):
            assert x.dtype == y.dtype and x.shape == y.shape, f"{name}[{i}]"
            assert np.array_equal(x, y), f"{name}[{i}]"
    assert np.array_equal(a.hashes, b.hashes)


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
    with pytest.raises(ValueError, match="unknown thumb policy"):
        imfeat.resize_area(img, "nearest")


# ---------------- a policy in place of the size ------------------------------------------------
# (H, W) -> what the policies pick, all from the same square: the largest power of two not
# above the shorter side. "pow2": the square. "pow2-cover": the shorter side as the square's,
# the longer one scaled to match, to the nearest multiple of 64 not above the frame's.
# "pow2-fit": the longer side as the square's, the shorter one scaled to match, the same way,
# and at least one quantum: a side too thin for the grain gives the shape up (a banner).
POLICY_SIZES = [
    # frame            pow2          pow2-cover   pow2-fit
    ((720, 1280), ((512, 512), (512, 896), (320, 512))),  # 288 rows wanted: no multiple of 64
    ((1080, 1920), ((1024, 1024), (1024, 1792), (576, 1024))),
    ((1440, 2560), ((1024, 1024), (1024, 1792), (576, 1024))),
    ((2160, 3840), ((2048, 2048), (2048, 3648), (1152, 2048))),
    ((1024, 1920), ((1024, 1024), (1024, 1920), (576, 1024))),  # a power-of-two side: kept
    ((768, 1024), ((512, 512), (512, 704), (384, 512))),  # 4:3: exact from 256 up
    ((1023, 1023), ((512, 512), (512, 512), (512, 512))),
    ((600, 800), ((512, 512), (512, 704), (384, 512))),
    ((480, 640), ((256, 256), (256, 320), (192, 256))),
    ((1024, 2016), ((1024, 1024), (1024, 1984), (512, 1024))),  # 2016 rounds up to 2048: clamped
    ((100, 3000), ((64, 64), (64, 1920), (64, 64))),  # a banner: fit gives the shape up
    ((64, 64), ((64, 64), (64, 64), (64, 64))),
    ((30, 50), ((16, 16), (16, 32), (16, 16))),  # the quantum is 16 here
    ((3, 5), ((2, 2), (2, 4), (2, 2))),
    ((1, 1), ((1, 1), (1, 1), (1, 1))),
]
POLICIES = ("pow2", "pow2-cover", "pow2-fit")
# the frames the kernel and the pass are checked on, under every policy
POLICY_FRAMES = [((720, 1280), 0), ((1080, 1920), 1), ((2160, 3840), 3), ((1024, 1920), 4)]


def test_thumb_size():
    """The rules, on the shape alone, in every layout ``FeatureComputer`` takes; a portrait
    frame gets the landscape answer transposed."""
    assert imfeat.THUMB_POLICIES == POLICIES and imfeat.THUMB_QUANTUM == 64
    for (h, w), sizes in POLICY_SIZES:
        square = sizes[0]
        assert sizes[2][0] * sizes[2][1] <= square[0] * square[1] <= sizes[1][0] * sizes[1][1]
        for policy, size in zip(POLICIES, sizes):
            assert imfeat.thumb_size((h, w), policy) == size, (h, w, policy)
            assert imfeat.thumb_size((h, w, 3), policy) == size
            assert imfeat.thumb_size((w, h, 1), policy) == size[::-1]  # portrait
            assert imfeat.thumb_size((3, h, w), policy, channel_axis=0) == size
            assert imfeat.thumb_size((h, 4, w), policy, channel_axis=1) == size
            r, c = size
            assert r <= h and c <= w  # never an upscale
            assert r % min(64, r) == 0 and c % min(64, c) == 0  # a 64-cell grid divides
    # the two aspect policies keep the frame's shape to within the rounding of the scaled
    # side (half a quantum, 32 px), until that side would round to nothing
    for (h, w), (_square, cover, fit) in POLICY_SIZES:
        for size, scaled in ((cover, max(cover)), (fit, min(fit))):
            if scaled > 64:
                r, c = size
                assert abs(c / r - w / h) / (w / h) <= 32 / (scaled - 32), (h, w, size)
    # a fixed size passes through, whatever the frame
    assert imfeat.thumb_size((720, 1280, 3), 1024) == THUMB
    assert imfeat.thumb_size((720, 1280, 3), np.int64(1024)) == THUMB
    assert imfeat.thumb_size((720, 1280, 3), (256, 1024)) == (256, 1024)
    assert imfeat.thumb_size((720, 1280, 3), None) is None
    with pytest.raises(ValueError, match="unknown thumb policy 'pow4'; known: pow2, pow2-cover"):
        imfeat.thumb_size((720, 1280, 3), "pow4")
    with pytest.raises(ValueError, match="at least 2 dims"):
        imfeat.thumb_size((720,), "pow2")
    with pytest.raises(ValueError, match="exactly 2 spatial axes"):
        imfeat.thumb_size((2, 720, 1280, 3), "pow2")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("src", "i"), POLICY_FRAMES, ids=[f"{s[1]}x{s[0]}" for s, _ in POLICY_FRAMES]
)
def test_resize_area_by_policy(src, i, policy):
    size = POLICY_SIZES[i][1][POLICIES.index(policy)]
    assert imfeat.thumb_size(src, policy) == size
    img = content("structured", (*src, 3))
    assert np.array_equal(imfeat.resize_area(img, policy), cv_resize(img, size))
    gray = content("coarse", src, seed=2)
    assert np.array_equal(imfeat.resize_area(gray, policy, threads=2), cv_resize(gray, size))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("src", "i"), POLICY_FRAMES, ids=[f"{s[1]}x{s[0]}" for s, _ in POLICY_FRAMES]
)
def test_pass_resizes_by_policy(src, i, policy):
    """``thumb=<policy>`` is ``thumb=size`` for the size the rule picks: the same bytes as the
    pass on cv2's thumbnail of that size, the thumbnail itself through ``thumb_out``, and the
    size on ``.thumb`` for the host to read."""
    size = POLICY_SIZES[i][1][POLICIES.index(policy)]
    frame = content("random", (*src, 3), seed=5)
    small = cv_resize(frame, size)
    grid = [(5, 5), (4, 4)]  # 32 divides every side here
    on_thumb = imfeat.FeatureComputer(shape=small.shape, grid=grid, stride=2, threads=2)
    by_policy = imfeat.FeatureComputer(
        shape=frame.shape, grid=grid, stride=2, threads=2, thumb=policy
    )
    fixed = imfeat.FeatureComputer(shape=frame.shape, grid=grid, stride=2, threads=2, thumb=size)
    assert by_policy.thumb == size == fixed.thumb
    assert by_policy.thumb_policy == policy and fixed.thumb_policy is None
    out = np.empty((*size, 3), np.uint8)
    assert_same_outputs(by_policy.features(frame, thumb_out=out), on_thumb.features(small))
    assert np.array_equal(out, small)
    assert_same_outputs(by_policy.features(frame), fixed.features(frame))


def test_pass_by_policy_other_layouts():
    gray = content("random", (600, 800), seed=6)  # 2-D: 512
    small = cv_resize(gray, (512, 512))
    on_thumb = imfeat.FeatureComputer(shape=small.shape, grid=GRID, threads=2)
    on_frame = imfeat.FeatureComputer(shape=gray.shape, grid=GRID, threads=2, thumb="pow2")
    assert on_frame.thumb == (512, 512)
    out = np.empty((512, 512), np.uint8)
    assert_same_outputs(on_frame.features(gray, thumb_out=out), on_thumb.features(small))
    assert np.array_equal(out, small)

    bgr = content("structured", (1080, 1920, 3), seed=7)
    planar = np.ascontiguousarray(np.moveaxis(bgr, -1, 0))  # (3, H, W): the axes, not the layout
    small = cv_resize(bgr, (576, 1024))
    on_thumb = imfeat.FeatureComputer(shape=small.shape, grid=GRID, threads=2)
    on_frame = imfeat.FeatureComputer(
        shape=planar.shape, grid=GRID, threads=2, channel_axis=0, thumb="pow2-fit"
    )
    assert on_frame.thumb == (576, 1024)
    assert_same_outputs(on_frame.features(planar), on_thumb.features(small))

    portrait = content("structured", (1920, 1080, 3), seed=8)  # the landscape answer, transposed
    small = cv_resize(portrait, (1792, 1024))
    on_thumb = imfeat.FeatureComputer(shape=small.shape, grid=GRID, threads=2)
    on_frame = imfeat.FeatureComputer(
        shape=portrait.shape, grid=GRID, threads=2, thumb="pow2-cover"
    )
    assert on_frame.thumb == (1792, 1024)
    assert_same_outputs(on_frame.features(portrait), on_thumb.features(small))


def test_thumbnail_the_frames_own_size():
    """A policy can land on the frame's own size (``thumb=(H, W)`` says so outright): the pass
    then reads the frame as it is, and ``thumb_out`` is a copy of it."""
    for frame, policy in (
        (content("random", (1024, 1920, 3), seed=9), "pow2-cover"),
        (content("structured", (256, 256, 3), seed=9), "pow2-fit"),  # a square: fit keeps it
        (content("random", (512, 512), seed=9), "pow2"),
    ):
        plain = imfeat.FeatureComputer(shape=frame.shape, grid=GRID, threads=2)
        assert plain.thumb is None
        for thumb in (policy, frame.shape[:2]):
            same = imfeat.FeatureComputer(shape=frame.shape, grid=GRID, threads=2, thumb=thumb)
            assert same.thumb == frame.shape[:2]
            out = np.empty(frame.shape, np.uint8)
            assert_same_outputs(same.features(frame, thumb_out=out), plain.features(frame))
            assert np.array_equal(out, frame)
            out.fill(0)
            raw = same.compute(frame, thumb_out=out)
            assert np.array_equal(out, frame)
            for k, v in plain.compute(frame).items():
                assert np.array_equal(v, raw[k]), k
    planar = np.ascontiguousarray(np.moveaxis(content("random", (1024, 1920, 3), seed=10), -1, 0))
    fc = imfeat.FeatureComputer(shape=planar.shape, grid=GRID, channel_axis=0, thumb="pow2-cover")
    out = np.empty((1024, 1920, 3), np.uint8)
    fc.features(planar, thumb_out=out)
    assert np.array_equal(out, np.moveaxis(planar, 0, -1))  # the copy is in (H, W, C) order


def test_policy_arguments():
    with pytest.raises(ValueError, match="unknown thumb policy"):
        imfeat.FeatureComputer(shape=(720, 1280, 3), grid=GRID, thumb="POW2")
    with pytest.raises(ValueError, match="finest grid must divide"):
        imfeat.FeatureComputer(shape=(40, 50, 3), grid=GRID, thumb="pow2")  # 32 px, 64 cells
    with pytest.raises(ValueError, match="up to 4 channels"):
        imfeat.FeatureComputer(shape=(100, 100, 5), grid=[(2, 2)], thumb="pow2", feature_space=None)
    banner = imfeat.FeatureComputer(shape=(20, 1000, 3), grid=[(2, 2)], thumb="pow2-fit")
    assert banner.thumb == (16, 16)  # too thin for the grain: the shape gives way
    assert imfeat.resize_area(np.zeros((20, 1000, 3), np.uint8), "pow2-fit").shape == (16, 16, 3)
    fc = imfeat.FeatureComputer(shape=(64, 64, 3), grid=GRID, thumb="pow2")  # a power of two
    assert fc.thumb == (64, 64)  # is its own thumbnail
    with pytest.raises(ValueError, match="must have shape"):
        fc.features(np.zeros((64, 64, 3), np.uint8), thumb_out=np.empty((32, 32, 3), np.uint8))


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
    kw = {
        "grid": GRID,
        "stride": stride,
        "channels": channels,
        "threads": threads,
        "feature_space": feature_space,
    }
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
