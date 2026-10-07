"""The colour conversion fused into the pass, bit for bit OpenCV's.

Two halves. ``imfeat.convert`` runs the row kernel the pass fuses, and is pinned to
``cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`` on every one of the 2^24 BGR values, in image
widths that send every pixel through the vector path, through the scalar tail, and through
both. Then the pass itself: the features of a BGR image converted inside are byte for byte
those of the cvtColor'd image taken as it is -- ``features()`` and ``compute()``, at every
stride, thread count, channel selection and memory layout, and once more on the
all-colours image.

OpenCV is the reference; without cv2 the file is skipped rather than compared to a copy of
the formula.
"""

from __future__ import annotations

import numpy as np
import pytest

import imfeat

cv2 = pytest.importorskip("cv2")

GRID = [(4, 4), (2, 2)]


def every_colour() -> np.ndarray:
    """(2^24, 3) uint8: every BGR value once, b fastest."""
    v = np.arange(1 << 24, dtype=np.uint32)
    return np.stack([v & 255, (v >> 8) & 255, v >> 16], -1).astype(np.uint8)


def all_colours_image(width: int) -> np.ndarray:
    """every_colour() as an (H, width, 3) image, the last row padded by wrapping around."""
    px = every_colour()
    rows = -(-px.shape[0] // width)
    return np.resize(px, (rows * width, 3)).reshape(rows, width, 3)


def colour_frame(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """Random BGR with the formula's ties well represented: a third of the pixels have two
    equal channels and a sixth are grey (v == every channel, diff == 0)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    pick = rng.random(shape[:2])
    two = pick < 1 / 3
    img[two, 1] = img[two, 0]
    grey = pick > 5 / 6
    img[grey, 1] = img[grey, 2] = img[grey, 0]
    return img


def hsv_of(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)


def assert_same_outputs(a: imfeat.Pyramid, b: imfeat.Pyramid) -> None:
    for name in ("maps", "moments", "summary"):
        got, want = getattr(a, name), getattr(b, name)
        assert len(got) == len(want), name
        for i, (x, y) in enumerate(zip(got, want)):
            assert x.dtype == y.dtype and x.shape == y.shape, f"{name}[{i}]"
            assert np.array_equal(x, y), f"{name}[{i}]"
    assert np.array_equal(a.hashes, b.hashes)
    for x, y in zip(a.profiles, b.profiles):
        assert np.array_equal(x, y)


def assert_same_raw(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert a.keys() == b.keys()
    for k, x in a.items():
        assert np.array_equal(x, b[k]), k


# ---------------- the kernel, against OpenCV, exhaustively -------------------------------
@pytest.mark.parametrize("width", [4096, 8, 40, 100])
def test_bgr2hsv_matches_opencv_on_every_colour(width):
    """cv2.cvtColor's bytes for all 2^24 BGR values. 4096 wide, every pixel goes through the
    vector path at any lane count; 8 wide, every one through the scalar tail; 40 and 100
    wide, both (at 16, 32 or 64 lanes)."""
    img = all_colours_image(width)
    assert np.array_equal(imfeat.convert(img), cv2.cvtColor(img, cv2.COLOR_BGR2HSV))


def test_convert_reads_any_layout():
    """Packed pixels, planar channels and any other strides (a column-strided view, a
    reversed channel axis) all give OpenCV's bytes for the same content."""
    img = colour_frame((37, 91, 3), 1)  # odd sizes: a scalar tail on every row
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    planar = np.moveaxis(np.ascontiguousarray(np.moveaxis(img, -1, 0)), 0, -1)
    every_other = np.ascontiguousarray(np.repeat(img, 2, axis=1))[:, ::2]
    reversed_channels = np.ascontiguousarray(img[..., ::-1])[..., ::-1]
    for view in (img, planar, every_other, reversed_channels):
        assert np.array_equal(view, img)  # the same content, laid out differently
        assert np.array_equal(imfeat.convert(view), ref)


def test_convert_arguments():
    img = colour_frame((16, 24, 3))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        imfeat.convert(img[..., 0])
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        imfeat.convert(np.zeros((16, 24, 4), np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        imfeat.convert(img.astype(np.uint16))
    with pytest.raises(ValueError, match="unknown colour space 'rgb'"):
        imfeat.convert(img, "rgb", "hsv")
    with pytest.raises(ValueError, match="no hsv -> bgr conversion"):
        imfeat.convert(img, "hsv", "bgr")
    same = imfeat.convert(img, "bgr", "bgr")  # a copy
    assert np.array_equal(same, img) and same is not img and same.flags["C_CONTIGUOUS"]
    assert imfeat.COLOR_SPACES == ("bgr", "hsv")


# ---------------- the pass: converted inside == converted before ---------------------------
@pytest.mark.parametrize("channels", [None, [2], [1, 0], [0, 0, 2]])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("stride", [1, 2, (1, 3)])
def test_pass_converts_exactly(stride, threads, channels):
    """features() and compute() of a BGR image, converted in the pass, are byte for byte
    those of the cvtColor'd image taken as it is. `channels` index the HSV channels."""
    bgr = colour_frame((96, 128, 3), 2)
    hsv = hsv_of(bgr)
    fused = imfeat.FeatureComputer(
        bgr.shape, GRID, stride=stride, channels=channels, threads=threads
    )
    asis = imfeat.FeatureComputer(
        bgr.shape, GRID, stride=stride, channels=channels, threads=threads, feature_space=None
    )
    assert fused.converts and not asis.converts
    assert_same_outputs(fused.features(bgr), asis.features(hsv))
    assert_same_raw(fused.compute(bgr), asis.compute(hsv))


def test_pass_reads_any_layout():
    """The three source paths of the fused conversion: packed pixels, planar channels
    (a channel-first array through channel_axis), and anything else (a column-strided view,
    a reversed channel axis)."""
    bgr = colour_frame((64, 96, 3), 3)
    want = imfeat.FeatureComputer(bgr.shape, GRID, feature_space=None).features(hsv_of(bgr))
    chw = np.ascontiguousarray(np.moveaxis(bgr, -1, 0))
    got = imfeat.FeatureComputer(chw.shape, GRID, channel_axis=0).features(chw)
    assert_same_outputs(got, want)
    every_other = np.ascontiguousarray(np.repeat(bgr, 2, axis=1))[:, ::2]
    reversed_channels = np.ascontiguousarray(bgr[..., ::-1])[..., ::-1]
    fc = imfeat.FeatureComputer(bgr.shape, GRID)
    for view in (bgr, every_other, reversed_channels):
        assert_same_outputs(fc.features(view), want)


@pytest.mark.parametrize("stride", [1, 2])
def test_pass_converts_every_colour(stride):
    """The all-colours image through the pass: the raw integer sums of the converted BGR
    image are those of the cvtColor'd one, so no pixel of any colour differs."""
    bgr = all_colours_image(4096)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grid = [(6, 6)]
    fused = imfeat.FeatureComputer(bgr.shape, grid, stride=stride, threads=2)
    asis = imfeat.FeatureComputer(bgr.shape, grid, stride=stride, threads=2, feature_space=None)
    assert_same_raw(fused.compute(bgr), asis.compute(hsv))


# ---------------- the colour-space arguments --------------------------------------------
def test_colour_space_arguments():
    bgr = colour_frame((32, 32, 3))
    hsv = hsv_of(bgr)
    # the defaults convert a 3-channel image; the input's own space, or None, does not
    assert imfeat.FeatureComputer(bgr.shape, GRID).converts
    as_hsv = imfeat.FeatureComputer(bgr.shape, GRID, input_space="hsv")
    assert not as_hsv.converts
    as_is = imfeat.FeatureComputer(bgr.shape, GRID, feature_space=None)
    assert_same_outputs(as_hsv.features(hsv), as_is.features(hsv))
    same = imfeat.FeatureComputer(bgr.shape, GRID, input_space="bgr", feature_space="bgr")
    assert not same.converts
    # a 2-D or single-channel image has no colour space and is taken as it is
    assert not imfeat.FeatureComputer((32, 32), GRID).converts
    assert not imfeat.FeatureComputer((32, 32, 1), GRID).converts
    assert not imfeat.FeatureComputer((1, 32, 32), GRID, channel_axis=0).converts
    # any other channel count is not colour: it has to be asked for as it is
    for c in (2, 4, 8):
        with pytest.raises(ValueError, match="3-channel"):
            imfeat.FeatureComputer((32, 32, c), GRID)
        assert not imfeat.FeatureComputer((32, 32, c), GRID, feature_space=None).converts
    with pytest.raises(ValueError, match="unknown colour space 'rgb'"):
        imfeat.FeatureComputer(bgr.shape, GRID, input_space="rgb")
    with pytest.raises(ValueError, match="unknown colour space 'lab'"):
        imfeat.FeatureComputer((32, 32), GRID, feature_space="lab")  # checked even for grey
    with pytest.raises(ValueError, match="no hsv -> bgr conversion"):
        imfeat.FeatureComputer(bgr.shape, GRID, input_space="hsv", feature_space="bgr")
