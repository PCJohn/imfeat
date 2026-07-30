# imfeat

Fast, exact, single-pass image feature extraction on CPU.

`imfeat` walks an image **once** and returns, for every channel and every cell of a dyadic
pyramid: intensity **moments**, gradient **structure-tensor** features, an **orientation
histogram**, **extrema densities** and a rotation-invariant **LBP** histogram — plus
**cross-channel** covariance between every pair of channels, and three whole-frame
**perceptual hashes** (`imagehash`-compatible aHash / wHash / pHash). It is a
C++/[Highway](https://github.com/google/highway) SIMD core behind a small
[nanobind](https://github.com/wjakob/nanobind) Python API.

It is built for the front of a real-time vision pipeline: hand-crafted, well-understood,
weakly-correlated features, cheap enough to run on every frame, that a downstream model (a
net, a tree, a classical rule) can consume directly.

```python
import cv2, imfeat

img = cv2.imread("frame.png")                      # (H, W, 3) uint8, BGR
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

fc = imfeat.FeatureComputer(shape=hsv.shape, grid=[(5, 5), (4, 4)], stride=2)
f = fc.features(hsv)                               # one pass, all channels, both levels

f["mom_0"]        # (32, 32, 3, 4)  float64  [mean, var, m3, m4]
f["struct_0"]     # (32, 32, 3, 5)  float32  [energy, coherence, ori_cos, ori_sin, cornerness]
f["hog_0"]        # (32, 32, 3, 9)  float32  L1-normalised orientation histogram
f["cnt_0"]        # (32, 32, 3, 2)  float32  [local_max, local_min] densities
f["mom_global"]   # (3, 4)                   whole-frame, per channel
f["mom_summary_0"] # (4, 3, 4)               [feat, channel, [min,max,mean,std]] across cells
```

---

## Install

```bash
pip install git+https://github.com/PCJohn/imfeat
pytest tests -q        # 357 tests
```

Needs a C++17 compiler and CMake. Tested on Linux (AVX2), macOS/arm64 (NEON) and Windows.

---

## API

Two calls. Construct once for a fixed input shape, then call per frame — every buffer is
allocated up front and reused, so steady-state extraction allocates nothing.

### `FeatureComputer(shape, grid, stride=None, channels=None, channel_axis=-1)`

| arg | meaning |
|---|---|
| `shape` | the exact shape of every image you will pass. `(H, W)` single channel, `(H, W, C)` multi-channel. |
| `grid` | `k` → a 2^k × 2^k grid; `(ky, kx)` → 2^ky × 2^kx; or a **list** of those for a pyramid, finest → coarsest. Levels must be nested; the finest must divide `(H, W)`. |
| `stride` | `None` / `int` / `(sy, sx)`. Subsamples which pixels are accumulated. See *Efficient use*. |
| `channels` | which channels to process. Default `None` = **all of them**. |
| `channel_axis` | where the channel axis lives. Default `-1` (OpenCV `(H, W, C)`). Read via a zero-copy transpose. |

### `.features(img) -> dict[str, np.ndarray]`

The derived, model-ready maps. Keys are `{group}_{level}`, for `group` in
`mom | struct | hog | cnt | lbp | desc | xchan` and `level` in `0 .. K-1` plus `global`:

| key | shape | dtype | contents |
|---|---|---|---|
| `mom_i` | `(cy, cx, C, 4)` | float64 | `imfeat.MOMENTS` — mean, var, m3, m4 |
| `struct_i` | `(cy, cx, C, 5)` | float32 | `imfeat.FEATURES` — energy, coherence, ori_cos, ori_sin, cornerness |
| `hog_i` | `(cy, cx, C, 9)` | float32 | `imfeat.HOG_FEATURES`, L1-normalised |
| `cnt_i` | `(cy, cx, C, 2)` | float32 | `imfeat.COUNT_FEATURES` — local_max, local_min |
| `lbp_i` | `(cy, cx, C, 10)` | float32 | `imfeat.LBP_FEATURES`, L1-normalised |
| `desc_i` | `(cy, cx, C, 8)` | float32 | `imfeat.DESCRIPTOR_FEATURES` — derived nonlinear summaries |
| `xchan_i` | `(cy, cx, P, 2)` | float32 | `imfeat.CROSS_FEATURES` — cov, corr, per channel pair |
| `ahash`, `whash`, `phash` | `(C,)` | uint64 | `imfeat.HASHES` — one 64-bit perceptual hash per channel, whole-frame |
| `*_global` | `(C, …)` | | the same groups, reduced over the whole frame |
| `{mom,struct,hog,cnt,lbp,desc}_summary_i` | `(F, C, 4)` | matches the group | each map reduced **across that level's cells** into `imfeat.SUMMARY_STATS` |

The `C` axis is dropped entirely for 2-D `(H, W)` input.

`*_summary_i` is a per-level cross-cell reduction: for each channel and feature `F`, the
feature's value over the level's `cy·cx` cells is summarised into `[min, max, mean, std]`
(`imfeat.SUMMARY_STATS`; min/max exact, mean/std population). So the peak V cell-variance a
caller would otherwise reduce by hand is just `f["mom_summary_0"][1, V, 1]`. It is folded
into the same derive pass for free (no extra traversal) and exists **per grid level only** —
the `global` reduction is a single cell, so its cross-cell summary would be degenerate.

`xchan_i` is the one group that is **not** per channel: it carries one row per *pair* of
channels, `P = C*(C-1)/2`, listed in `fc.channel_pairs` as index pairs into the selected
channels. It is absent when there are no pairs — a single channel, or `C > 8`, where the
quadratic pair count is not worth paying silently.

`ahash`, `whash`, `phash` are whole-frame perceptual hashes (no pyramid): one `uint64` per
channel, packed row-major MSB-first. See [§8](#8-perceptual-hashes--ahash-whash-phash) for
what they encode and how they relate to the `imagehash` package.

### `.compute(img) -> dict[str, np.ndarray]`

The **raw int64 accumulators**, same keys, before any nonlinear derivation: `struct_i` is
`[Sxx, Syy, Sxy, count]`, `mom_i` the power sums `[S1, S2, S3, S4]`, `hog_i` the
unnormalised gradient energy per bin, `cnt_i` and `lbp_i` the raw counts, and `xchan_i`
the raw products `sum(v_i * v_j)`, shape `(cy, cx, P)`.

These are **purely additive**, which makes them composable: sum them over cells, levels,
channels or *frames* and you get the exact accumulator of the union. Pool however your
application likes, without re-reading the image.

```python
raw  = fc.compute(img)
# exact moments over an arbitrary region of interest -- no second pass
ps   = raw["mom_0"][8:16, 4:12].sum(axis=(0, 1))            # (C, 4) power sums
n    = raw["struct_0"][8:16, 4:12, :, 3].sum(axis=(0, 1))   # (C,)   pixel count
mean = ps[:, 0] / n
var  = ps[:, 1] / n - mean**2
```

### Constants

`imfeat.MOMENTS`, `imfeat.FEATURES`, `imfeat.HOG_FEATURES`, `imfeat.COUNT_FEATURES`,
`imfeat.LBP_FEATURES`, `imfeat.DESCRIPTOR_FEATURES` and `imfeat.CROSS_FEATURES` name the last axis of each group, in
order, so nothing has to be indexed by magic number. `imfeat.SUMMARY_STATS` names the last
axis of the `*_summary_i` maps (`min, max, mean, std`). `fc.channel_pairs` names the pair axis
of `xchan_i`. `imfeat.HASHES` lists the perceptual-hash keys (`ahash, whash, phash`).

---

## Features computed

Everything below is computed **per channel** and **per cell**, in the same pass. Write `Ω`
for the pixels a cell samples, and `n = |Ω|`.

### 1. Intensity moments — `mom_i`, 4 values

Central moments of the raw pixel value `v ∈ [0, 255]`:

```
mean = (1/n) Σ v
var  = (1/n) Σ (v - mean)²      m3 = (1/n) Σ (v - mean)³      m4 = (1/n) Σ (v - mean)⁴
```

The standard descriptors follow: `std = √var`, `skew = m3 / std³`, `kurtosis = m4 / var²`.
Skew is the useful one for bimodality (ink on paper, specular highlights); kurtosis for
tailedness.

**How.** The textbook two-pass form (mean, then deviations) cannot share a traversal with
anything else, so imfeat accumulates the **raw power sums** `S1..S4 = Σv, Σv², Σv³, Σv⁴` as
exact int64 and converts at the end. The naive conversion `var = S2/n - mean²` cancels
catastrophically on low-variance, high-mean cells, so the sums are first shifted by the
integer `K = round(mean)` through the binomial expansion:

```
T1 = S1 - nK
T2 = S2 - 2K·S1 + nK²
T3 = S3 - 3K·S2 + 3K²·S1 - nK³
T4 = S4 - 4K·S3 + 6K²·S2 - 4K³·S1 + nK⁴
```

Each `Tᵢ` stays exact in int64, and the residual offset `d = T1/n` is `≤ 0.5`, so
`var = T2/n - d²` has nothing left to cancel. Verified to rtol 1e-12 against numpy even on
inputs drawn from `{250, 251}`, where the naive form loses ~10 digits.

### 2. Gradient structure tensor — `struct_i`, 5 values

3×3 Sobel gradients `gx, gy` at every sampled pixel (replicate-padded at the border), then
the summed second-moment matrix of the gradient field:

```
J = [ Σgx²   Σgxgy ]     tr = Σgx² + Σgy²     D = Σgx² - Σgy²     R = √(D² + 4(Σgxgy)²)
    [ Σgxgy  Σgy²  ]
```

`R` is the spread between `J`'s eigenvalues `λ± = (tr ± R)/2`. The five outputs:

| feature | formula | reads as |
|---|---|---|
| `energy` | `tr / n` | mean squared gradient magnitude — how much edge/texture is here |
| `coherence` | `R / tr` | `∈ [0,1]`. 1 = one dominant edge direction, 0 = isotropic clutter |
| `ori_cos` | `D / tr` | double-angle orientation vector, `cos 2θ` |
| `ori_sin` | `2·Σgxgy / tr` | double-angle orientation vector, `sin 2θ` |
| `cornerness` | `λ₋ / n = (tr - R) / 2n` | Shi–Tomasi: large only when *both* eigenvalues are |

Orientation is stored as a **double-angle vector**, not an angle. Edge direction is mod π —
`θ` and `θ+π` are the same edge — so averaging raw angles across cells is simply wrong (0°
and 179° would average to 90°). The double-angle vector averages correctly, and its length
*is* the coherence. Recover the angle with `θ = ½·atan2(ori_sin, ori_cos)`.

The tensor is also usefully illumination-robust: brightening a region scales all its
gradients equally, which moves `energy` but leaves `coherence` and the orientation vector
alone.

### 3. Orientation histogram — `hog_i`, 9 values

Each sampled pixel votes its **gradient energy** `gx² + gy²` into one of 9 orientation bins
spanning `[0, π)` (bin `b` covers `[b, b+1)·π/9`). The histogram is L1-normalised per cell.

Where the structure tensor reports one *dominant* orientation plus a scalar coherence, the
histogram carries the full orientation *distribution* — enough to tell "two strong
perpendicular edge families" (a grid, a fence, hatching) from "isotropic noise", which look
identical to the tensor's coherence.

**How.** Binning uses integer arithmetic, never `atan2`. The gradient is folded into the
upper half-plane, then tested against the 8 bin-boundary rays by the sign of a cross
product; the bin index is just the count of rays it lies past. Branch-free, exact, and it
vectorises across channels.

### 4. Extrema densities — `cnt_i`, 2 values

The fraction of a cell's sampled pixels that are a **strict** maximum (resp. minimum) over
their 8-neighbourhood. A blob / speckle / keypoint density: high for texture, dots, noise
and corners; ~0 for smooth gradients and flat regions. Unlike `energy` it is insensitive to
contrast — a faint blob and a hard blob each count once.

Since the border is replicate-padded, a border pixel sits inside its own neighbourhood and
therefore can never be a strict extremum. That is deliberate, and consistent across the
four sides.

### 5. Rotation-invariant uniform LBP — `lbp_i`, 10 values

The LBP^riu2_{8,1} histogram of Ojala et al. (2002). Each pixel is compared to its 8
neighbours in **circular order**; the resulting 8-bit sign pattern is called *uniform* if it
has at most two circular 0→1 transitions, and is then labelled by its popcount (0..8). Bin 9
collects everything else. The popcount and the transition count are both invariant to
rotating the ring, which is where the rotation invariance comes from.

This is the one group with properties **no other feature here has**:

* **invariant to any monotonic change of the intensity map** (only the sign of each
  comparison matters) — so it survives gamma, exposure and contrast changes that move every
  moment and every gradient statistic;
* **invariant to 90° rotations and reflections** exactly, and largely to arbitrary rotation.

Read it as a micro-texture descriptor: bins 0/8 are flat or spot-like neighbourhoods, the
middle bins are edges and corners of varying sharpness, bin 9 is high-frequency clutter.

The comparison is `neighbour >= centre`, so a flat neighbourhood is all-ones — a flat image
lands entirely in bin 8. The border is replicate-padded, consistently with `cnt_i`.

### 6. Derived descriptors — `desc_i`, 8 values

Six *nonlinear* summaries of the sums above, computed in the same pass at no extra
accumulator cost. They exist for downstream models: a fast linear or shallow-tree classifier
can form weighted sums of features for free but cannot cheaply compute a ratio, a product, a
standardized moment, or a histogram's peakedness — so those are precomputed here rather than
left for the model to rediscover. Five of the six are dimensionless (invariant to any
`v -> a*v+b`, `a>0`), which is what lets a model trained on a handful of pixels transfer
across a contrast or brightness change.

| name | definition | reads | what it separates |
|---|---|---|---|
| `std_skew` | `m3 / var^1.5` | moments | bimodal ink-on-paper text (strongly skewed) from symmetric scenes |
| `excess_kurt` | `m4 / var^2 - 3` | moments | leptokurtic text, platykurtic binary/stripes, near-Gaussian photo |
| `edge_sharpness` | `energy / (var + 1)` | struct+mom | in-focus detail per unit contrast (a focus/defocus map) |
| `detail` | `energy * (1 - coherence)` | struct | isotropic edge clutter (QR, foliage) from a single clean edge |
| `hog_concentration` | `sum p_i^2` over HOG bins | hog | one dominant orientation (barcode) from spread (QR) |
| `hog_cardinality` | axis-aligned bin fraction | hog | horizontal/vertical structure (text, tables) from diagonal |
| `grad_sparsity` | `n*sum(g^4)/(sum g^2)^2` = 1+CV^2(grad energy) | gradient | sparse strong edges (rules, glyph/infographic borders) from dense texture |
| `rms_contrast` | `sqrt(var) / (mean + 1)` | moments | coefficient of variation (Peli 1990); brightness-relative contrast |

`std_skew`, `edge_sharpness` and `detail` are exactly the hand-coded cues the `framegate`
client computes today; promoting them to first-class features means the model reads them
directly. All but `detail` are dimensionless. `desc_i` is derived only, so it appears in
`features()` but not `compute()`; `grad_sparsity` is the one that adds an accumulator
(`sum |grad|^4`) rather than being computed purely from existing sums.

### 7. Cross-channel covariance — `xchan_i`, 2 values per channel pair

For each unordered pair of channels, `[cov, corr]`: the covariance and the Pearson
correlation of the two channels' pixel values over the cell. Everything else in `imfeat`
treats each channel independently; this is the only feature that says anything about how
they *relate*. On RGB it separates neutral regions (all pairs ~+1) from saturated colour;
on HSV or opponent spaces it picks up chromatic structure a per-channel statistic cannot.

Only one new accumulator per pair is needed — the raw product `sum(v_i * v_j)` — because the
means and variances are already in `mom_i`.

### 8. Perceptual hashes — `ahash`, `whash`, `phash`

Three `imagehash`-compatible perceptual hashes, one `uint64` per channel, computed once for
the whole frame (no pyramid, no per-cell variants). Bits are packed row-major, MSB first, so
each is a fixed-width 64-bit key — the layout a multi-index-hashing store wants.

All three are readouts of a mean grid, which is just `S1 = sum(v)` box-pooled to a fixed
resolution:

| hash | grid | bit rule |
|---|---|---|
| `ahash` (average) | 8×8 | cell mean `>` mean of the 64 cells |
| `whash` (wavelet, Haar) | 8×8 | cell mean `>` median of the 64 cells |
| `phash` (perceptual) | 32×32 | low-freq 8×8 of a 2-D DCT-II `>` its median |

Because they need only `S1`, they add **no traversal**: when the finest pyramid grid already
tiles 32×32 (e.g. a `(5,5)` finest level), the hash grid is a block-sum of the finest cells'
`S1` and the hashes cost nothing beyond a tiny end-of-pass DCT and threshold. Only a finest
grid coarser than 32×32 — which doesn't retain enough spatial detail for `phash` — falls back
to a second lightweight sum over the row buffer already in cache (still one image traversal).

The wavelet hash reduces to the same 8×8 mean grid as `ahash` under a median threshold: for
Haar, `imagehash`'s `remove_max_haar_ll` subtracts a constant from every cell, which the
median absorbs, so the bits are exactly a median-thresholded block-mean grid.

**Match to `imagehash`.** `imfeat` box-pools (area averaging); `imagehash` resizes with
Lanczos, so results are **bit-exact where no resize occurs** (an 8×8 input for a/w, 32×32 for
p — the unit tests assert this) and differ only by that resampling on full-size frames. On
natural 256×256 frames the relative Hamming distance to stock `imagehash` is ~0.00 for
`whash`, ~0.01 for `phash` and ~0.12 for `ahash` (the mean threshold is the most
resize-sensitive). Hashes are per channel like everything else, so `imagehash`'s single
grayscale hash corresponds to running `imfeat` on a luma channel.

---

## The one-pass design

Every per-pixel quantity is accumulated as an **additive int64 sum** into the finest cell of
its channel — 30 sums per cell per channel:

| slots | contents |
|---|---|
| 0..3 | `Sxx Syy Sxy count` — structure tensor |
| 4..12 | HOG: gradient energy per orientation bin |
| 13,14 | strict local max / min counts |
| 15..18 | `S1 S2 S3 S4` — power sums of the pixel value |
| 19..28 | LBP^riu2 bin counts |
| 29 | `Σ|grad|⁴` — gradient sparsity |

plus, per cell, one further int64 per channel *pair* — `sum(v_i * v_j)` — the only sum that
is not per channel.

Two consequences fall out, and they are the whole design:

**Pyramid depth is free.** Because the sums are additive, a coarse cell is the *exact sum*
of the finer cells inside it, and `global` is the sum of all of them. Extra levels cost one
cheap reduction over cells — not another pass over pixels. Four levels cost ~5% more than
one.

**Everything shares one traversal.** Moments, gradients, histogram and extrema all read the
same pixel while it is in register. The nonlinear parts — eigenvalues, central moments,
histogram normalisation — are *not* additive, so they are deferred and derived once per cell
at the end. That derivation is negligible: `features()` costs the same as `compute()`.

**Perceptual hashes ride along.** `ahash`/`whash`/`phash` need only `S1` at a fixed 32×32
grid, which the same additivity provides: when the finest level tiles 32×32 they are a
block-sum of its cells' `S1` and add nothing to the pass; only a coarser finest grid needs a
second lightweight sum over the cached row buffer (never a second image traversal). Measured
overhead is ~1–3% in the tiling case and ~3–7% in the fallback, on top of one frame.

**Channels are the SIMD lane.** All channels are walked in the same spatial pass, packed
four per vector, so every channel after the first is markedly cheaper.

Cells are assigned by floor division (`cell = row * n_cells // H`), so any shape works; and
because cell counts are powers of two, level *k*'s cell index is level 0's **shifted right**.
A pixel's features at every scale are therefore an O(1) lookup — handy when assembling a
per-pixel feature vector for a model.

---

## Efficient use

Measured on a 256×256×3 frame, 4 pyramid levels, every feature on all 3 channels (Apple
M-series; an AVX2 laptop lands within ~20%). Errors are against the exact `stride=1` output:
`energy_r` is the correlation of the edge-energy map, `hog_cos` and `lbp_cos` the mean cosine
similarity of the histograms. (The error columns are properties of the algorithm, not the
machine; `bench_compare.py` prints these and `xchan_r` on any host.)

| stride | ms | speedup | mean err (grey levels) | energy_r | orientation err | hog_cos | lbp_cos |
|---|---|---|---|---|---|---|---|
| 1 | 1.35 | 1.0× | 0.0 | 1.000 | 0.0° | 1.000 | 1.000 |
| **2** | **0.63** | **2.1×** | **4.3** | **0.952** | **3.6°** | **0.923** | **0.990** |
| 4 | 0.36 | 3.8× | 10.7 | 0.824 | 4.9° | 0.832 | 0.958 |
| 8 | 0.24 | 5.6× | 23.0 | 0.583 | 5.3° | 0.755 | 0.834 |

The LBP histogram degrades far more gracefully under subsampling than the extrema density
does (`cnt_r` falls to 0.53 by stride 4), because it is a histogram over many pixels rather
than a count of rare events.

**The stride rule: keep `cell_width / stride ≥ 4`.** Stride's real currency is *samples per
cell*, not pixels per image. At 256² on a 32×32 grid the cells are 8 px wide, so `stride=2`
(16 samples/cell) is the sweet spot and `stride=4` starts to cost the structure maps. On a
1024² frame the same grid gives 32 px cells, so `stride=8` is still safe — and 11× faster.
Push stride past the cell width and it degenerates to one column per cell: the moments
survive, the structure maps do not.

**Moments degrade far more gracefully than the structure maps.** At `stride=4` the cell
means are still within ~11 grey levels while edge-energy correlation has fallen to 0.82. If
you consume only moments, you can be much more aggressive.

**Scale by resolution first, stride second.** Cost is linear in pixels walked, so halving
the working resolution is a clean 4× *and* keeps samples-per-cell intact. Resize, extract,
then use stride as the final trim.

**Take all the pyramid levels** — they are nearly free, and give multi-scale context for one
cell reduction.

**`features()` returns one format.** A `Pyramid` namedtuple, not a dict of named groups:

| field | shape | dtype |
|---|---|---|
| `maps` | per level, `(H, W, C*38)`, finest first, 1-cell global last | float32 |
| `moments` | per level, `(H, W, C, 4)` over `MOMENTS` | float64 |
| `summary` | per level except global, `(38, C, 4)` over `SUMMARY_STATS` | float64 |
| `cross` | per level, `(H, W, P, 2)` over `CROSS_FEATURES` | float32 |
| `hashes` | `(3, C)` over `HASHES`, whole-frame | uint64 |

`maps` is NHWC with the channel axis C-major over `FEATURE_NAMES`, so on a 256² thumbnail
with a 4-level grid it is `(32,32,114) (16,16,114) (8,8,114) (4,4,114)` — an FPN P3–P6 shape
family that feeds a detector neck or a per-cell tree model directly. `moments` repeats the
trailing four channels of `maps` at full precision: m3 and m4 span a range float32 cannot hold
to the accuracy the oracle tests require, so use `moments` when precision matters and `maps`
when feeding a model.

Every feature is per-pixel normalised, so no level carries a cell-area factor and one
shared-weight head can read every level. The channels do span a very wide dynamic range
(histogram bins near 1e-2, fourth moments near 1e6), so standardise per channel against
statistics fixed over a dataset before training — not per frame, which would throw away
absolute brightness and contrast. `summary` is the cheap way to collect them.

`compute()` still returns the raw int64 accumulator sums as a dict; it is the low-level
validation surface, not the modelling one, and its keys mirror the C++ slot layout.

**Threading is opt-in and bit-exact.** `threads=N` splits the accumulate pass into disjoint
bands of finest cell rows. Two bands never touch the same accumulator, so there are no atomics
on the hot path, and because every accumulator is an int64 sum that cannot overflow the output
is *identical* at any thread count — derived floats included, since the coarse-level rollup,
the summary fold and the hash readout still run serially after the join. The calling thread
takes one band, so `threads=2` creates one worker. `N` is capped at the finest grid's row count
(`grid=0` has one cell row and cannot split); `.threads` reports what was used, and
`imfeat.cpu_count()` the logical cores available.

Measured on an M-series laptop, `features()` end to end, min of 200 frames:

| frame | grid, stride | 1 thread | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 256×256×3 | finest-32, s2 | 0.62 ms | 0.41 (1.5×) | 0.36 (1.7×) | 0.30 (2.1×) |
| 512×512×3 | finest-32, s2 | 1.73 ms | 0.95 (1.8×) | 0.76 (2.3×) | 0.61 (2.8×) |
| 1024×1024×3 | finest-64, s2 | 7.11 ms | 3.69 (1.9×) | 2.90 (2.5×) | 2.31 (3.1×) |
| 1024×1024×3 | finest-32, s4 | 2.42 ms | 1.29 (1.9×) | 1.01 (2.4×) | 0.80 (3.0×) |

The accumulate pass itself scales at 3.4× on four threads; the gap to that is the serial tail,
which is a fixed cost and so hurts small frames most. Workers park on a condition variable
between frames rather than spinning, so idle bands cost no CPU — the intent is to leave the
other cores to other real-time work. Thread *placement* is left to the OS: there is no
portable core-pinning API (macOS has none), so pin from the caller if your platform can.

**Channels after the first are cheap:** 0.34 ms at C=1, 0.65 ms at C=3, then a flat ~0.21 ms
per additional channel. The full feature set on all of H, S and V costs ~1.9× one channel,
not 3×.

| input | stride=1 | stride=2 | stride=4 |
|---|---|---|---|
| 128×128×3 | 0.47 ms | 0.29 ms | 0.20 ms |
| 256×256×3 | 1.35 ms | 0.63 ms | 0.36 ms |
| 512×512×3 | 4.79 ms | 2.04 ms | 0.99 ms |
| 1024×1024×3 | 18.5 ms | 7.6 ms | 3.4 ms |

Finer grids cost more than coarser ones (each cell's accumulators are flushed once per row,
so flush traffic grows with cell count while pixel work stays fixed). 32×32 is a good
default; 128×128 works but is both slow and statistically thin unless the image is large.

`bench_compare.py` reproduces all of the above, sweeping stride × input size × channel count
and reporting latency next to per-feature-group accuracy.

---

## Scope

Input is always **uint8** — 2-D or 3-D, any channel count (1, 3, or hyperspectral). Floating
point input, N-D tensors and arbitrary axis reductions are deliberately out of scope: the
uint8 assumption is exactly what makes the exact-integer accumulators possible.

Reductions beyond the built-in cells / levels / global are not really a limitation, since
`compute()` hands back additive sums: any pooling over cells, channels or frames is one
`numpy` sum away, and per-row or per-column moments are just a `(k, 0)` or `(0, k)` grid.
