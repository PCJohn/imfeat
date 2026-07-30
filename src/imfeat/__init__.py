"""imfeat -- fast single-pass image feature extraction (moments + structure).

One traversal of the image produces, per pyramid cell per channel:

  * central moments of the pixel values  [mean, var, m3, m4]
  * gradient structure-tensor features   [energy, coherence, ori_cos, ori_sin, cornerness]
  * an L1-normalised HOG                 (9 orientation bins over [0, pi))
  * extrema densities                    [local_max, local_min]
  * an LBP^riu2_{8,1} histogram          (10 rotation-invariant uniform-LBP bins)

and, once per cell (not per channel), for every pair of channels:

  * cross-channel                        [cov, corr]

Every accumulator is an additive integer sum, so coarser pyramid levels and the
frame-wide "global" reduction are exact sums of the finest cells -- depth is free
and the image is read exactly once. The nonlinear parts (eigenvalues, central
moments, histogram normalisation) are derived once per cell at the end.

Input is one uint8 image: 2-D ``(H, W)``, or multi-channel in OpenCV order
``(H, W, C)`` for any C (set ``channel_axis`` for a different layout). Every
channel is processed in the SAME spatial pass and appears as its own axis in the
output (that axis is dropped for 2-D input).

    import imfeat

    fc = imfeat.FeatureComputer(shape=(256, 256, 3), grid=[(5, 5), (4, 4)])
    f = fc.features(hsv_u8)
    f["mom_0"]       # (32, 32, 3, 4)  float64  [mean, var, m3, m4]
    f["struct_0"]    # (32, 32, 3, 5)  float32
    f["hog_0"]       # (32, 32, 3, 9)  float32
    f["cnt_0"]       # (32, 32, 3, 2)  float32
    f["mom_global"]  # (3, 4)          whole-frame per-channel moments

Grid cells use floor division (``cell = row * n_cells // H``), so any shape works
and, because cell counts are powers of two, level k's cell index is level 0's
shifted right -- a pixel's feature vector at every scale is an O(1) lookup.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union, cast

import numpy as np

from . import imfeat_core as _core  # type: ignore[attr-defined]

__all__ = [
    "COUNT_FEATURES",
    "CROSS_FEATURES",
    "DESCRIPTOR_FEATURES",
    "FEATURES",
    "HASHES",
    "HOG_FEATURES",
    "LBP_FEATURES",
    "MOMENTS",
    "SUMMARY_STATS",
    "FeatureComputer",
    "cpu_count",
]
__version__ = "0.1.0"

# Grid spec accepted by FeatureComputer: k, (ky, kx), or a list of either.
GridSpec = Union[int, Sequence[int], Sequence[Sequence[int]]]

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
# Cross-channel maps ("xchan_i"), one row per channel pair (see .channel_pairs).
CROSS_FEATURES = ("cov", "corr")
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

_NS_RAW = 4  # raw structure-tensor width [Sxx, Syy, Sxy, count]
_NS_FEAT = len(FEATURES)  # derived structure-tensor width = 5
_NH = len(HOG_FEATURES)  # HOG bins
_NC = len(COUNT_FEATURES)  # 2
_NM = len(MOMENTS)  # 4
_NL = len(LBP_FEATURES)  # 10
_NX = len(CROSS_FEATURES)  # 2
_ND = len(DESCRIPTOR_FEATURES)  # 8
_NST = len(SUMMARY_STATS)  # 4


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
    ) -> None:
        shape = tuple(int(s) for s in shape)
        ndim = len(shape)
        if ndim < 2:
            raise ValueError("shape must have at least 2 dims (H, W)")
        self._shape = shape
        if ndim == 2:
            self._chan_axis = None  # single channel: no channel axis in output
            h, w, c = shape[0], shape[1], 1
        else:
            cax = channel_axis % ndim
            spatial = [ax for ax in range(ndim) if ax != cax]
            if len(spatial) != 2:
                raise ValueError("multi-channel input must have exactly 2 spatial axes")
            self._chan_axis = cax
            h, w, c = shape[spatial[0]], shape[spatial[1]], shape[cax]
        chan = list(range(c)) if channels is None else [int(x) for x in channels]
        if not chan or any(not 0 <= x < c for x in chan):
            raise ValueError(f"channels must be a non-empty subset of range({c})")
        self._grid = _parse_grid(grid)
        self._keys = [str(i) for i in range(len(self._grid))] + ["global"]
        self._impl = _core._FeatureComputerImpl()
        self._impl.set_config(
            [h, w, c], chan, self._grid, _parse_stride(stride), max(1, int(threads))
        )
        #: bands actually used, i.e. threads participating including the caller's.
        self.threads: int = self._impl.threads()
        p = self._impl.pairs()
        #: pairs of *selected*-channel indices carried by the "xchan_*" maps, in order.
        #: Empty for a single channel and for C > imfeat_core.XMAX (hyperspectral).
        self.channel_pairs: list[tuple[int, int]] = list(zip(p[::2], p[1::2]))

    def _view(self, img: np.ndarray) -> np.ndarray:
        """Validate and return an (H, W[, C]) axis-order view (no copy)."""
        if img.shape != self._shape:
            raise ValueError(f"shape mismatch: expected {self._shape}, got {img.shape}")
        if img.dtype != np.uint8:
            raise TypeError("input must be uint8")
        if self._chan_axis is None or self._chan_axis == img.ndim - 1:
            return img  # already (H, W) or (H, W, C)
        return np.moveaxis(img, self._chan_axis, -1)  # zero-copy transpose to (H, W, C)

    def _cut(self, a: np.ndarray, widths: Sequence[tuple[str, int]], i: str) -> dict:
        """Slice one wide (..., C, W) array into its feature groups, dropping the
        size-1 channel axis for single-channel (2-D) input.

        One flat copy up front, then views into it. Copying group by group instead is
        a strided gather per group (stride = the full width W), which measures ~13x
        slower and, once the accumulate pass was threaded, was the entire serial tail.
        The copy means the returned arrays own their data and stay valid across later
        calls, exactly as before -- but all groups from one level now share a single
        base array, so holding one group retains the whole level.
        """
        b, out, o = a.copy(), {}, 0
        for name, n in widths:
            v = b[..., o : o + n]
            out[f"{name}_{i}"] = v[..., 0, :] if self._chan_axis is None else v
            o += n
        return out

    def _cut_summary(self, a: np.ndarray, widths: Sequence[tuple[str, int]], i: str) -> dict:
        """Slice one (C, F, NST) cross-cell summary into per-group "<name>_summary_i"
        arrays (last axis = SUMMARY_STATS), dropping the size-1 channel axis for
        single-channel (2-D) input."""
        # core buffer is (C, F, NST); expose as (F, C, NST) so the channel axis is
        # second-to-last (as in every other map), then drop it for 2-D input. Transpose
        # once for the whole array rather than per group -- and since the groups then
        # split along the leading axis, each one comes out contiguous.
        b, out, o = np.ascontiguousarray(a.transpose(1, 0, 2)), {}, 0
        for name, n in widths:
            v = b[o : o + n]
            out[f"{name}_summary_{i}"] = v[:, 0, :] if self._chan_axis is None else v
            o += n
        return out

    def compute(self, img: np.ndarray) -> dict[str, np.ndarray]:
        """Raw int64 accumulators per cell (additive; sum them freely):
            struct_i (cells_y, cells_x[, C], 4)  [Sxx, Syy, Sxy, count]
            hog_i    (cells_y, cells_x[, C], 9)  gradient-energy orientation histogram
            cnt_i    (cells_y, cells_x[, C], 2)  [local_max, local_min]
            mom_i    (cells_y, cells_x[, C], 4)  power sums [S1, S2, S3, S4]
            lbp_i    (cells_y, cells_x[, C], 10) LBP^riu2 bin counts (sum == pixel count)
            xchan_i  (cells_y, cells_x, P)       Sum(v_i * v_j) per channel pair, P = len(channel_pairs)
        plus the "_global" variants. The C axis is present only for multi-channel input;
        "xchan_*" is absent when there are no channel pairs.
        """
        widths = (("struct", _NS_RAW), ("hog", _NH), ("cnt", _NC), ("mom", _NM), ("lbp", _NL))
        lv, cross = self._impl.raw(self._view(img))
        out: dict[str, np.ndarray] = {}
        for i, a in zip(self._keys, lv):
            out.update(self._cut(a, widths, i))
        for i, x in zip(self._keys, cross):
            out[f"xchan_{i}"] = x.copy()
        return out

    def features(self, img: np.ndarray) -> dict[str, np.ndarray]:
        """Derived maps, computed in the same pass. Same keys as compute():
        struct_i (..., 5) float32   FEATURES
        hog_i    (..., 9) float32   L1-normalised histogram
        cnt_i    (..., 2) float32   extrema densities
        lbp_i    (..., 10) float32  L1-normalised LBP^riu2 histogram
        mom_i    (..., 4) float64   MOMENTS [mean, var, m3, m4]
        desc_i   (..., 8) float32   DESCRIPTOR_FEATURES (derived nonlinear summaries)
        xchan_i  (cy, cx, P, 2) float32   CROSS_FEATURES [cov, corr] per channel pair

        Plus, per grid level i (not "global"), each derived feature reduced across
        that level's cells into SUMMARY_STATS [min, max, mean, std] on the last axis.
        Shape (n_feat, C, 4); the C axis is dropped for 2-D input:
        struct_summary_i (5, C, 4)  hog_summary_i (9, C, 4)  cnt_summary_i (2, C, 4)
        lbp_summary_i (10, C, 4)  desc_summary_i (8, C, 4)  mom_summary_i (4, C, 4)
        So e.g. the max V cell-variance a caller would otherwise reduce by hand is
        mom_summary_0[1, V, 1] (feature 1 = var, channel V, stat 1 = max).

        Also three whole-frame perceptual hashes (imfeat.HASHES), one uint64 per channel,
        computed in the same pass (no pyramid): ahash, whash, phash. The C axis is dropped
        for 2-D input (a scalar uint64).
        """
        fw = (("struct", _NS_FEAT), ("hog", _NH), ("cnt", _NC), ("lbp", _NL), ("desc", _ND))
        feat, mom, cross, fsum, msum, hashes = self._impl.features(self._view(img))
        out: dict[str, np.ndarray] = {}
        for i, (a, m) in zip(self._keys, zip(feat, mom)):
            out.update(self._cut(a, fw, i))
            out.update(self._cut(m, (("mom", _NM),), i))
        for i, x in zip(self._keys, cross):
            out[f"xchan_{i}"] = x.copy()
        # Cross-cell summaries: per grid level only (the 1-cell "global" is trivial).
        for i, (fs, ms) in zip(self._keys[:-1], zip(fsum, msum)):
            out.update(self._cut_summary(fs, fw, i))
            out.update(self._cut_summary(ms, (("mom", _NM),), i))
        # Perceptual hashes: one uint64 per channel, whole-frame only (no pyramid).
        # Drop the channel axis for 2-D input, as the other maps do.
        for name, h in zip(HASHES, hashes):
            out[name] = h.copy() if self._chan_axis is not None else h[0].copy()
        return out


def cpu_count() -> int:
    """Logical cores the runtime reports, so callers can budget `threads` against
    whatever else shares the machine. Never returns less than 1. This is the raw
    hardware count: it does not know about cgroup/container quotas, other
    processes, or the P/E-core split on hybrid CPUs."""
    return int(_core.cpu_count())
