"""imfeat -- fast single-pass image feature extraction on CPU.

One traversal of the image produces, per pyramid cell per channel:

  * central moments of the pixel values  [mean, var, m3, m4]
  * gradient structure-tensor features   [energy, coherence, ori_cos, ori_sin, cornerness]
  * an L1-normalised HOG                 (9 orientation bins over [0, pi))
  * extrema densities                    [local_max, local_min]
  * an LBP^riu2_{8,1} histogram          (10 rotation-invariant uniform-LBP bins)
  * derived nonlinear descriptors        (DESCRIPTOR_FEATURES, 8)

plus three whole-frame perceptual hashes (aHash / wHash / pHash) per channel, the
projection profiles, and a per-level summary of every feature over the level's cells.

Every accumulator is an additive integer sum, so coarser pyramid levels and the
frame-wide "global" reduction are exact sums of the finest cells -- depth is free
and the image is read exactly once. The nonlinear parts (eigenvalues, central
moments, histogram normalisation) are derived once per cell at the end.

Input is one uint8 image: 2-D ``(H, W)``, or multi-channel in OpenCV order
``(H, W, C)`` for any C (set ``channel_axis`` for a different layout). A 3-channel image
is colour: by default it is taken as BGR (OpenCV's order) and the features are computed
in HSV, the conversion running inside the pass (``input_space`` / ``feature_space``). With
``thumb`` the image is a frame the pass thumbnails on the way in, exactly as
``cv2.resize(..., INTER_AREA)`` would, so a host hands over the frame itself; the size is a
fixed one or a policy the frame's shape decides (``"pow2"``: the largest power of two that
fits the shorter side, square; ``"pow2-cover"`` and ``"pow2-fit"`` keep the frame's shape
around or inside that square, as closely as the grid allows).

    import imfeat

    fc = imfeat.FeatureComputer(shape=(256, 256, 3), grid=[(5, 5), (4, 4)])
    p = fc.features(bgr_u8)
    p.maps[0]      # (32, 32, 114) float32, 3 channels x FEATURE_NAMES, channel-major
    p.maps[-1]     # (114,)        global level
    p.moments[0]   # (32, 32, 3, 4) float64 MOMENTS at full precision
    p.summary[0]   # (38, 3, 4)    float64 SUMMARY_STATS of each feature across cells
    p.hashes       # (3, 3)        uint64  HASHES x channel

``compute()`` returns the raw int64 accumulators instead, as a dict per group.

Grid cells use floor division (``cell = row * n_cells // H``), so any shape works
and, because cell counts are powers of two, level k's cell index is level 0's
shifted right -- a pixel's feature vector at every scale is an O(1) lookup.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Union, cast

import numpy as np

from . import imfeat_core as _core  # type: ignore[attr-defined]

__all__ = [
    "COLOR_SPACES",
    "COUNT_FEATURES",
    "DESCRIPTOR_FEATURES",
    "FEATURES",
    "FEATURE_NAMES",
    "HASHES",
    "HOG_FEATURES",
    "LBP_FEATURES",
    "MOMENTS",
    "SUMMARY_STATS",
    "THUMB_POLICIES",
    "THUMB_QUANTUM",
    "FeatureComputer",
    "Pyramid",
    "convert",
    "cpu_count",
    "resize_area",
    "thumb_size",
]
__version__ = "0.1.0"

# Grid spec accepted by FeatureComputer: k, (ky, kx), or a list of either.
GridSpec = Union[int, Sequence[int], Sequence[Sequence[int]]]
# Thumbnail spec accepted by FeatureComputer and resize_area: (rows, cols), one int for a
# square, or the name of a policy (THUMB_POLICIES) that derives the size from the frame's.
ThumbSpec = Union[int, Sequence[int], str]

#: Colour spaces of 3-channel images, as `input_space` / `feature_space` name them.
COLOR_SPACES = ("bgr", "hsv")

#: Thumbnail policies `thumb` takes by name, each a rule from the frame's (H, W) to the
#: thumbnail's (rows, cols), so the size follows the frame rather than being fixed up front.
#: All three start from the same square, the largest power of two not above the shorter
#: side; none upscales, and a power-of-two grid of up to 64 cells per axis divides every
#: size they make (of a frame at least 64 px short). ``thumb_size`` resolves a policy.
#:
#: "pow2": that square itself: a 720p frame becomes 512x512, 1080p and 1440p 1024x1024, 4K
#: 2048x2048. The aspect ratio goes.
#: "pow2-cover": the frame's shape around the square: the shorter side as "pow2", the longer
#: side scaled by the same factor: 720p 512x896, 1080p 1024x1792, 4K 2048x3648. Up to the
#: aspect ratio times the square's pixels.
#: "pow2-fit": the frame's shape inside the square: the longer side as "pow2", the shorter
#: side scaled by the same factor: 720p 512x320, 1080p and 1440p 1024x576, 4K 2048x1152.
#: Never more pixels than the square (the aspect ratio's share of them), so never dearer.
#:
#: Neither aspect policy keeps the shape exactly: the scaled side is rounded to a multiple
#: of 64 (THUMB_QUANTUM), which the grid needs, so it is kept as closely as that allows. At
#: widths of 1024 and 2048 a 16:9 frame comes out exact and 4:3 does from 256 up, but a
#: 16:9 frame at width 512 wants 288 rows, which is no multiple of 64, and gets 320: a 1.6:1
#: picture, a 10 % squash where the square's is 78 %. The error is at most half a quantum
#: on the scaled side, and grows as the thumbnail shrinks; a side that would round to
#: nothing becomes one quantum, so a banner far thinner than the grain ends up the square
#: under "pow2-fit" (never an upscale: the quantum is at most the shorter side's power of two).
#:
#: The pass is fastest when the thumbnail's WIDTH is a power of two: its cells are then a
#: power of two wide and tile the vector blocks, sampling stride folded in. Any other cell
#: width is walked one masked block per cell, and a stride leaves most of the block's lanes
#: idle (a 28 px cell at stride 4 uses 7 lanes of 32). On a landscape frame "pow2" and
#: "pow2-fit" have such a width and "pow2-cover" does not, so "pow2-cover" costs 2-3x what
#: its pixel count says; on a portrait frame the two aspect policies swap roles.
THUMB_POLICIES = ("pow2", "pow2-cover", "pow2-fit")
#: The side a policy scales to the aspect ratio is rounded to a multiple of this (never above
#: the frame's side, never below it), so grids of up to this many cells divide it. Also the
#: unit of the aspect error those policies make: half of it at most on that side.
THUMB_QUANTUM = 64

# Raw central moments of the pixel values ("mom_i" / "mom_global"), float64.
MOMENTS = ("mean", "var", "m3", "m4")
# Channel order of the structure-tensor float32 maps ("struct_i" / "struct_global").
FEATURES = ("energy", "coherence", "ori_cos", "ori_sin", "cornerness")
# L1-normalised orientation histogram ("hog_i"); bin count comes from the C++ core
# (compile-time HB), so this stays in sync with the accumulator layout. Bin b spans
# angles [b, b+1) * pi/HB.
HOG_FEATURES = tuple(f"hog_{b}" for b in range(_core.HB))
# Extrema densities ("cnt_i"): fraction of pixels that are strict 8-nbhd max/min.
COUNT_FEATURES = ("local_max", "local_min")
# Rotation-invariant uniform LBP ("lbp_i"), L1-normalised. Bins 0..8 are the popcounts
# of the uniform codes (<=2 circular 0/1 transitions); bin 9 collects everything else.
LBP_FEATURES = tuple(f"lbp_{b}" for b in range(_core.LBPB - 1)) + ("lbp_nonuniform",)
# Model-ready nonlinear descriptors ("desc_i"), all derived in the same pass from the sums
# above (no extra accumulators). std_skew/excess_kurt are the dimensionless (illumination-
# invariant) shape of the intensity distribution; edge_sharpness/detail are structure-tensor
# ratios framegate hand-codes today; hog_concentration/hog_cardinality summarise the gradient
# orientation histogram's peakedness and axis-alignment. Nonlinear (ratios/products/squares),
# so a linear or shallow-tree model cannot cheaply reconstruct them.
DESCRIPTOR_FEATURES = (
    "std_skew",
    "excess_kurt",
    "edge_sharpness",
    "detail",
    "hog_concentration",
    "hog_cardinality",
    "grad_sparsity",
    "rms_contrast",
)

# Bar/stroke detector ("bard_i"). A centre-surround test at BARD_NL lags: a pixel
# scores only where BOTH taps at +/-d differ from it in the same direction, so step
# edges score zero while strokes of width ~d score highly. cover is the fraction of
# sampled pixels passing the noise gate; spec_* splits the gated response mass across
# lags (a stroke-width spectrum); peak is its mass-weighted mean lag; peaked is the
# profile's coefficient of variation (one dominant width vs texture); bal is dark/light
# polarity in [-1, 1]. All but cover are ratios, hence invariant to stride and cell size.
BARD_FEATURES = (
    ("bard_cover",)
    + tuple(f"bard_spec{j + 1}" for j in range(_core.BARD_NL))
    + ("bard_peak", "bard_peaked", "bard_bal")
)

# Second-order texture from the 3x3 neighbourhood Sobel already reads. lap_var is the variance
# of the 4-neighbour Laplacian, the usual focus measure, and focus divides it by `energy`:
# second-order detail per unit of first-order, which blur lowers and contrast does not.
# laws_XY is the mean squared response of the 3x3 Laws mask with X down the rows and Y along
# the columns, from L = [1 2 1], E = [-1 0 1], S = [-1 2 -1] (LE and EL are Sobel's gx and gy,
# which `energy` already covers): ee is diagonal structure, ss spots, ls / sl vertical /
# horizontal lines, es / se line ends and ripples. line_aniso is (ls - sl) / (ls + sl) in
# [-1, 1]: +1 when every line is vertical.
TEXTURE_FEATURES = (
    "lap_var",
    "focus",
    "laws_ee",
    "laws_ss",
    "laws_ls",
    "laws_sl",
    "laws_es",
    "laws_se",
    "line_aniso",
)

# Perceptual hashes ("ahash"/"whash"/"phash"), one uint64 (64 bits, row-major MSB-first)
# per channel, computed whole-frame in the same pass (no pyramid). imagehash-compatible:
# aHash/wHash threshold an 8x8 mean grid by its mean/median; pHash the low-freq 8x8 of a
# 2D DCT of a 32x32 mean grid. Box-binned means differ from imagehash's Lanczos resize,
# so expect a small Hamming drift on full-size images (bit-exact at native 8x8 / 32x32).
HASHES = ("ahash", "whash", "phash")

# Cross-cell summary stats carried on the last axis of every "*_summary_i" array,
# in this fixed order. For each channel and feature, the feature's value is reduced
# over that pyramid level's cells (min/max exact; mean/std population, cell-weighted).
SUMMARY_STATS = ("min", "max", "mean", "std")

#: One channel's slice of the features() channel axis, in order (F = 54).
FEATURE_NAMES = (
    FEATURES
    + HOG_FEATURES
    + COUNT_FEATURES
    + LBP_FEATURES
    + DESCRIPTOR_FEATURES
    + BARD_FEATURES
    + TEXTURE_FEATURES
    + MOMENTS
)


class Pyramid(NamedTuple):
    """maps    : per level, (H, W, C*F) float32, finest first, 1-cell global last
    moments : per level, (H, W, C, 4) float64 over MOMENTS -- the same numbers as the
              trailing 4 channels of `maps`, kept at full precision because m3 and m4
              span a range float32 cannot hold to the accuracy the oracle tests demand
    summary : per level except global, (F, C, 4) float64 over SUMMARY_STATS
    hashes  : (3, C) uint64 over HASHES, whole-frame
    profiles: (rows, cols), float64 (R, C) and (K, C): mean intensity along each sampled row and
              each sampled column, in order -- projection profiles, for 1-D shift estimates
              and for finding text lines"""

    maps: list[np.ndarray]
    moments: list[np.ndarray]
    summary: list[np.ndarray]
    hashes: np.ndarray
    profiles: tuple[np.ndarray, np.ndarray]


_NS_RAW = 4  # raw structure-tensor width [Sxx, Syy, Sxy, count]
_NH = len(HOG_FEATURES)  # HOG bins
_NC = len(COUNT_FEATURES)  # 2
_NM = len(MOMENTS)  # 4
_NL = len(LBP_FEATURES)  # 10


def _parse_grid(grid: GridSpec) -> list[list[int]]:
    """-> list of [exp_y, exp_x]. Accepts k, (ky, kx), or a list of those."""
    if isinstance(grid, int):
        return [[grid, grid]]
    g = list(grid)
    specs = g if isinstance(g[0], (tuple, list)) else [g]
    return [[int(s[0]), int(s[1])] for s in cast("list[Sequence[int]]", specs)]


def _parse_stride(stride: int | Sequence[int] | None) -> list[int]:
    if stride is None:
        return [1, 1]
    if isinstance(stride, int):
        return [stride, stride]
    return [int(stride[0]), int(stride[1])]


def _spatial(shape: Sequence[int], channel_axis: int) -> tuple[int, int, int, int | None]:
    """-> (H, W, C, channel axis) of a frame of `shape`; C = 1 and no axis for 2-D."""
    dims = tuple(int(s) for s in shape)
    ndim = len(dims)
    if ndim < 2:
        raise ValueError("shape must have at least 2 dims (H, W)")
    if ndim == 2:
        return dims[0], dims[1], 1, None
    cax = channel_axis % ndim
    spatial = [ax for ax in range(ndim) if ax != cax]
    if len(spatial) != 2:
        raise ValueError("multi-channel input must have exactly 2 spatial axes")
    return dims[spatial[0]], dims[spatial[1]], dims[cax], cax


def _scaled_side(side: int, n: int, ref: int, q: int) -> int:
    """``side * n / ref`` (the side scaled as the other one was, to ``n`` from ``ref``) rounded
    to the nearest multiple of ``q``, kept at or below ``side`` (a policy never upscales) and
    at least ``q``: a side too thin for the grain becomes one quantum, and the shape gives way
    (a banner under "pow2-fit" ends up the square)."""
    x = (2 * side * n + ref * q) // (2 * ref * q) * q  # round half up, in integers
    return max(q, min(x, side // q * q))


def _policy_size(policy: str, h: int, w: int) -> tuple[int, int]:
    """(rows, cols) the policy makes of an (h, w) frame; see THUMB_POLICIES."""
    short, long = min(h, w), max(h, w)
    n = 1 << (short.bit_length() - 1)  # the largest power of two <= the shorter side
    q = min(THUMB_QUANTUM, n)
    if policy == "pow2":
        return n, n
    if policy == "pow2-cover":  # the shorter side is n, the longer follows the shape
        sides = n, _scaled_side(long, n, short, q)
    elif policy == "pow2-fit":  # the longer side is n, the shorter follows the shape
        sides = _scaled_side(short, n, long, q), n
    else:
        raise ValueError(f"unknown thumb policy {policy!r}; known: {', '.join(THUMB_POLICIES)}")
    return sides if h <= w else sides[::-1]  # (short, long) in the frame's orientation


def _resolve_thumb(thumb: ThumbSpec | None, h: int, w: int) -> tuple[int, int] | None:
    """-> (rows, cols), or None: a pair as given, one int as a square, a policy name as its
    rule applied to the frame's (h, w)."""
    if thumb is None:
        return None
    if isinstance(thumb, str):
        return _policy_size(thumb, h, w)
    if isinstance(thumb, (int, np.integer)):
        rows = cols = int(thumb)
    else:
        t = list(thumb)
        if len(t) != 2:
            raise ValueError("thumb must be (rows, cols), one int, or a policy name")
        rows, cols = int(t[0]), int(t[1])
    if rows < 1 or cols < 1:
        raise ValueError("thumb must have at least one row and one column")
    return rows, cols


def thumb_size(
    shape: Sequence[int], thumb: ThumbSpec | None, channel_axis: int = -1
) -> tuple[int, int] | None:
    """The ``(rows, cols)`` that ``thumb`` means for a frame of ``shape``, i.e. what
    ``FeatureComputer(shape, ..., thumb=thumb).thumb`` will be: a pair as given, one int as a
    square, a policy name (``THUMB_POLICIES``) as its rule applied to the frame's ``(H, W)``
    (``channel_axis`` as for ``FeatureComputer``), ``None`` as ``None``. Plain integer
    arithmetic on the shape, nothing is read: a host that keeps one computer per frame shape
    can pick the computer, or size the buffer it hands to ``thumb_out``, from this alone."""
    h, w, _, _ = _spatial(shape, channel_axis)
    return _resolve_thumb(thumb, h, w)


# (input, feature) -> the core's conversion, for the pairs that differ. Each is a row kernel
# in core.cpp (see Convert there); a new space or pair is an entry here and a kernel there.
_CONVERSIONS: dict[tuple[str, str], int] = {("bgr", "hsv"): _core.CONVERT_BGR2HSV}


def _conversion(input_space: str, feature_space: str | None, channels: int) -> int:
    """The core's conversion code for `channels`-channel input in `input_space` when the
    features are wanted in `feature_space` (None: on the channels as they are)."""
    for name in (input_space, feature_space):
        if name is not None and name not in COLOR_SPACES:
            raise ValueError(f"unknown colour space {name!r}; known: {', '.join(COLOR_SPACES)}")
    if feature_space is None or feature_space == input_space or channels == 1:
        return int(_core.CONVERT_NONE)  # nothing to convert, or no colour to convert
    if channels != 3:
        raise ValueError(
            f"{input_space} -> {feature_space} needs a 3-channel image, not {channels} channels;"
            " pass feature_space=None to take the channels as they are"
        )
    try:
        return int(_CONVERSIONS[input_space, feature_space])
    except KeyError:
        pairs = ", ".join(f"{a} -> {b}" for a, b in _CONVERSIONS)
        raise ValueError(f"no {input_space} -> {feature_space} conversion; have {pairs}") from None


class FeatureComputer:
    """Stateful feature extractor for a fixed uint8 input shape.

    shape        : (H, W) single channel, or an N-D shape with a channel axis
                   (default the last, OpenCV ``(H, W, C)`` order).
    grid         : k, a (ky, kx) tuple (2^k cells/axis), or a list of those for a
                   dyadic pyramid, finest->coarsest and nested; the finest must
                   divide (H, W). Coarser levels are exact sums of finer cells, so
                   the image is read once whatever the depth.
    stride       : None/int/(sy, sx). Subsamples which pixels accumulate; the
                   gradient stencil and cell boundaries stay at full resolution.
                   Columns restart the stride at each finest-cell edge, so every
                   cell keeps the same sample count (no inter-cell bias).
    channels     : which channels to process (default: all of them). All features
                   are computed for every selected channel in the one pass.
    channel_axis : which axis of a multi-channel input holds channels (default
                   -1). Ignored for 2-D input. The other two axes, in order, are
                   (H, W). Read in place via a zero-copy transpose.
    threads      : worker threads for the accumulate pass (default 1, no threads
                   created). Rows are split into disjoint bands of finest cell
                   rows, so the output is bit-identical at any thread count. The
                   calling thread runs one band itself, so ``threads=2`` adds one
                   worker. Capped at the finest grid's row count; ``.threads``
                   reports what was actually used. See ``cpu_count()``.
    thumb        : ``(rows, cols)`` (or one int for a square) to resize every image to
                   before anything else, inside the pass: each band makes its own rows of
                   the thumbnail from the frame's rows, bit for bit what
                   ``cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)`` gives
                   (downscaling only, up to 4 channels: an upscale is not INTER_AREA in
                   OpenCV, so resize with cv2 first). ``shape`` is then the frame's, the
                   grid divides the thumbnail, and ``features(frame, thumb_out=buf)``
                   also writes the thumbnail for reuse. Or a policy name, and the size
                   follows ``shape``: ``"pow2"`` is the largest power of two not above the
                   shorter side, square (a 720p frame: 512, 1080p: 1024, 4K: 2048);
                   ``"pow2-cover"`` keeps the frame's shape around that square, the
                   shorter side the power of two and the longer one scaled to match
                   (720p: 512x896, 1080p: 1024x1792); ``"pow2-fit"`` keeps it inside the
                   square, the longer side the power of two and the shorter one scaled
                   (720p: 512x320, 1080p: 1024x576) -- never more pixels than the square.
                   Neither keeps the shape exactly: the scaled side is rounded to a
                   multiple of 64 for the grid, so the shape is kept as closely as that
                   allows. None upscales, and a power-of-two grid of up to 64 cells
                   divides them all (``THUMB_POLICIES`` has the rules, and which of them
                   the pass is fast at: those with a power-of-two width). It is
                   settled here, once, from ``shape`` (``thumb_size`` does the same sum on
                   its own); ``.thumb`` is the size in use and ``.thumb_policy`` the name.
                   A thumbnail the frame's own size (a policy can land there) is the frame:
                   nothing is resized, and ``thumb_out`` gets a copy. See ``resize_area``.
    input_space  : the colour space of the images passed in: "bgr" (OpenCV's order,
                   the default) or "hsv". See ``COLOR_SPACES``.
    feature_space: the colour space the features are computed in, "hsv" by default:
                   a BGR frame straight from OpenCV gets HSV features with no
                   ``cvtColor`` pass, the conversion running inside the extraction
                   (in every band's thread, as each row is read), bit for bit what
                   ``cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`` gives. ``None`` takes the
                   channels as they are, as does naming the input's own space.
                   ``channels`` then index the feature space (HSV: 0, 1, 2 = H, S, V).
                   Only a 3-channel image is colour: a 2-D or single-channel one is
                   taken as it is, and any other channel count needs
                   ``feature_space=None``.

    A single all-frame "global" reduction is returned alongside the grid levels.
    """

    def __init__(
        self,
        shape: Sequence[int],
        grid: GridSpec,
        stride: int | Sequence[int] | None = None,
        channels: Sequence[int] | None = None,
        channel_axis: int = -1,
        threads: int = 1,
        input_space: str = "bgr",
        feature_space: str | None = "hsv",
        thumb: ThumbSpec | None = None,
    ) -> None:
        self._shape = tuple(int(s) for s in shape)
        # single channel (2-D): no channel axis in the output
        h, w, c, self._chan_axis = _spatial(self._shape, channel_axis)
        chan = list(range(c)) if channels is None else [int(x) for x in channels]
        if not chan or any(not 0 <= x < c for x in chan):
            raise ValueError(f"channels must be a non-empty subset of range({c})")
        conversion = _conversion(input_space, feature_space, c)
        #: whether the pass converts the input's colour space (False: channels as given).
        self.converts: bool = conversion != _core.CONVERT_NONE
        #: the ``(rows, cols)`` the pass resizes every image to first, or None. Settled here
        #: for good: a policy is applied to ``shape`` once, at no cost per image.
        self.thumb: tuple[int, int] | None = _resolve_thumb(thumb, h, w)
        #: the policy ``thumb`` named, if it named one (see ``THUMB_POLICIES``), else None.
        self.thumb_policy: str | None = thumb if isinstance(thumb, str) else None
        # A thumbnail the frame's own size (a policy can land there) is the frame: the pass
        # reads the frame as it is, and thumb_out gets a copy from here.
        self._identity = self.thumb == (h, w)
        self._grid = _parse_grid(grid)
        self._keys = [str(i) for i in range(len(self._grid))] + ["global"]
        self._impl = _core._FeatureComputerImpl()
        self._impl.set_config(
            [h, w, c],
            chan,
            self._grid,
            _parse_stride(stride),
            max(1, int(threads)),
            conversion,
            [] if self.thumb is None or self._identity else list(self.thumb),
        )
        #: bands actually used, i.e. threads participating including the caller's.
        self.threads: int = self._impl.threads()

    def _view(self, img: np.ndarray) -> np.ndarray:
        """Validate and return an (H, W[, C]) axis-order view (no copy)."""
        if img.shape != self._shape:
            raise ValueError(f"shape mismatch: expected {self._shape}, got {img.shape}")
        if img.dtype != np.uint8:
            raise TypeError("input must be uint8")
        if self._chan_axis is None or self._chan_axis == img.ndim - 1:
            return img  # already (H, W) or (H, W, C)
        return np.moveaxis(img, self._chan_axis, -1)  # zero-copy transpose to (H, W, C)

    def _thumb_out(self, out: np.ndarray | None) -> np.ndarray | None:
        """Validate the array the thumbnail is written into: ``(rows, cols, C)`` uint8,
        C-contiguous, C the input's channel count (``(rows, cols)`` for 2-D input)."""
        if out is None:
            return None
        if self.thumb is None:
            raise ValueError("thumb_out needs a FeatureComputer built with thumb=")
        c = 1 if self._chan_axis is None else self._shape[self._chan_axis]
        want = self.thumb if self._chan_axis is None else (*self.thumb, c)
        if out.shape != want:
            raise ValueError(f"thumb_out must have shape {want}, got {out.shape}")
        if out.dtype != np.uint8:
            raise TypeError("thumb_out must be uint8")
        if not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("thumb_out must be a C-contiguous, writeable array")
        return out

    def _inputs(
        self, img: np.ndarray, thumb_out: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The image view and the thumbnail buffer the core takes: none when the thumbnail
        is the frame itself, which is copied into the buffer here instead."""
        view, out = self._view(img), self._thumb_out(thumb_out)
        if out is not None and self._identity:
            np.copyto(out, view)
            out = None
        return view, out

    def _cut(self, a: np.ndarray, widths: Sequence[tuple[str, int]], i: str) -> dict:
        """Slice one wide (..., C, W) array into its feature groups, dropping the
        size-1 channel axis for single-channel (2-D) input.

        The groups are views into `a`, which the binding already hands over as a copy that
        owns its data: the returned arrays stay valid across later calls, but all groups
        from one level share that one base array, so holding one group retains the level.
        """
        out, o = {}, 0
        for name, n in widths:
            v = a[..., o : o + n]
            out[f"{name}_{i}"] = v[..., 0, :] if self._chan_axis is None else v
            o += n
        return out

    def features(self, img: np.ndarray, thumb_out: np.ndarray | None = None) -> Pyramid:
        """The whole pyramid in one shot. `maps` is one `(H, W, C*F)` float32 array per
        level, finest first and the 1-cell global level last, so `(H, W, channels)` is
        NHWC and feeds an FPN-style neck or a per-cell tree model directly. The channel
        axis is C-major over `FEATURE_NAMES` (F = 38).

        With ``thumb`` set, ``thumb_out`` (a ``(rows, cols, C)`` uint8 array, C-contiguous)
        receives the thumbnail the pass made, for reuse downstream; without it nothing is
        written out.

        Every feature is per-pixel normalised, so no level carries a cell-area factor and
        one shared-weight head can read all of them. The channels do span a very wide
        dynamic range (histogram bins near 1e-2, fourth moments near 1e6), so standardise
        per channel against statistics fixed over a dataset before training -- not per
        frame, which would discard absolute brightness and contrast. `summary` is the
        cheap way to collect them.
        """
        # Every array arrives in its final layout, sharing ownership of this call's block.
        maps, mom, summary, hashes, rows, cols = self._impl.features(
            *self._inputs(img, thumb_out)
        )
        return Pyramid(
            maps=maps, moments=mom, summary=summary, hashes=hashes, profiles=(rows, cols)
        )

    def compute(
        self, img: np.ndarray, thumb_out: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Raw int64 accumulators per cell (additive; sum them freely):
            struct_i (cells_y, cells_x[, C], 4)  [Sxx, Syy, Sxy, count]
            hog_i    (cells_y, cells_x[, C], 9)  gradient-energy orientation histogram
            cnt_i    (cells_y, cells_x[, C], 2)  [local_max, local_min]
            mom_i    (cells_y, cells_x[, C], 4)  power sums [S1, S2, S3, S4]
            lbp_i    (cells_y, cells_x[, C], 10) LBP^riu2 bin counts (sum == pixel count)
        plus the "_global" variants. The C axis is present only for multi-channel input.
        ``thumb_out`` as in :meth:`features`.
        """
        widths = (("struct", _NS_RAW), ("hog", _NH), ("cnt", _NC), ("mom", _NM), ("lbp", _NL))
        out: dict[str, np.ndarray] = {}
        for i, a in zip(self._keys, self._impl.raw(*self._inputs(img, thumb_out))):
            out.update(self._cut(a, widths, i))
        return out


def convert(img: np.ndarray, input_space: str = "bgr", feature_space: str = "hsv") -> np.ndarray:
    """``img``, an ``(H, W, 3)`` uint8 image in ``input_space``, as a new ``(H, W, 3)``
    array in ``feature_space``: the conversion ``FeatureComputer`` fuses into its pass,
    on its own. For bgr -> hsv the bytes are ``cv2.cvtColor(img, cv2.COLOR_BGR2HSV)``'s.
    Any strides (a view is fine); the same space twice is a copy."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise TypeError("input must be uint8")
    conversion = _conversion(input_space, feature_space, 3)
    if conversion == _core.CONVERT_NONE:
        return np.array(img, dtype=np.uint8, order="C")
    return np.asarray(_core.convert(img, conversion))


def resize_area(img: np.ndarray, size: ThumbSpec, threads: int = 1) -> np.ndarray:
    """``img`` (an ``(H, W)`` or ``(H, W, C)`` uint8 image, up to 4 channels, any strides)
    resized to ``size`` = ``(rows, cols)`` (or one int for a square, or a policy name such
    as ``"pow2"``, see ``thumb_size``): a new array holding
    ``cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)``'s bytes exactly, as
    the pass makes its thumbnail (``FeatureComputer(thumb=...)``). Downscaling only: an
    upscale is bilinear in OpenCV, not INTER_AREA, and is refused. ``threads`` splits the
    rows over that many threads for this call; the output is the same at any count."""
    if img.dtype != np.uint8:
        raise TypeError("input must be uint8")
    if img.ndim not in (2, 3):
        raise ValueError(f"expected an (H, W) or (H, W, C) image, got shape {img.shape}")
    resolved = _resolve_thumb(size, img.shape[0], img.shape[1])
    if resolved is None:
        raise ValueError("size must be (rows, cols), one int, or a policy name")
    rows, cols = resolved
    return np.asarray(_core.resize_area(img, rows, cols, max(1, int(threads))))


def cpu_count() -> int:
    """Logical cores the runtime reports, so callers can budget `threads` against
    whatever else shares the machine. Never returns less than 1. This is the raw
    hardware count: it does not know about cgroup/container quotas, other
    processes, or the P/E-core split on hybrid CPUs."""
    return int(_core.cpu_count())
