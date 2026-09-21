# imfeat features

Every feature below is computed **per channel** and **per cell**, for every grid level and for
the whole frame (`global`), in one pass over the image. Write `Ω` for the pixels a cell
samples (the stride mesh, restarting in every cell), `n = |Ω|`, and `v ∈ [0, 255]` for the
pixel value. All 3×3 neighbourhoods are replicate-padded at the image border.

`imfeat.FEATURE_NAMES` lists the 54 per-channel features in `maps` order; `maps` is
channel-major, so `maps[i].reshape(cy, cx, C, 54)` separates channel and feature axes.

| # | group | tuple | offset | count |
|---|---|---|---|---|
| 1 | structure tensor | `FEATURES` | 0 | 5 |
| 2 | orientation histogram | `HOG_FEATURES` | 5 | 9 |
| 3 | extrema densities | `COUNT_FEATURES` | 14 | 2 |
| 4 | LBP histogram | `LBP_FEATURES` | 16 | 10 |
| 5 | derived descriptors | `DESCRIPTOR_FEATURES` | 26 | 8 |
| 6 | bar / stroke detector | `BARD_FEATURES` | 34 | 7 |
| 7 | second-order texture | `TEXTURE_FEATURES` | 41 | 9 |
| 8 | intensity moments | `MOMENTS` | 50 | 4 |

Not per channel: cross-channel covariance (§9). Whole frame only: projection profiles (§10)
and perceptual hashes (§11).

Where they are useful, roughly:

| task | features that carry it |
|---|---|
| text / OCR front end | bar detector, `line_aniso`, `laws_ls` / `laws_sl`, HOG cardinality, skew and kurtosis, row profile |
| person / object detection | HOG, LBP, structure tensor; cell means across levels give Haar-like differences |
| counting, blobs | extrema densities, `lap_var`, `laws_ss` |
| texture segmentation | LBP, Laws energies, moments, HOG concentration |
| scene classification | the HOG pyramid (GIST-like), moments, cross-channel correlation |
| registration, stabilisation | projection profiles, cornerness (Shi–Tomasi trackability), hashes |
| blur / quality gating | `lap_var`, `focus`, `edge_sharpness` |

---

## 1. Structure tensor — `FEATURES`

3×3 Sobel gradients `gx, gy`, then the summed second-moment matrix of the gradient field:

```
J = [ Σgx²   Σgxgy ]     tr = Σgx² + Σgy²     D = Σgx² - Σgy²     R = √(D² + 4(Σgxgy)²)
    [ Σgxgy  Σgy²  ]
```

`R` is the spread of `J`'s eigenvalues `λ± = (tr ± R)/2`.

| feature | formula | reads as |
|---|---|---|
| `energy` | `tr / n` | mean squared gradient magnitude |
| `coherence` | `R / tr` | 1 = one dominant edge direction, 0 = isotropic |
| `ori_cos`, `ori_sin` | `D / tr`, `2Σgxgy / tr` | double-angle orientation vector `(cos 2θ, sin 2θ)` |
| `cornerness` | `λ₋ / n = (tr − R) / 2n` | Shi–Tomasi: large only when both eigenvalues are |

Orientation is a **double-angle vector** because edge direction is mod π: raw angles do not
average (0° and 179° would give 90°), the vector does, and its length is the coherence.
Recover the angle with `θ = ½·atan2(ori_sin, ori_cos)`. Brightening a region moves `energy`
but leaves coherence and orientation alone.

## 2. Orientation histogram — `HOG_FEATURES`

Each sampled pixel votes its gradient energy `gx² + gy²` into one of 9 unsigned orientation
bins over `[0, π)`; the histogram is L1-normalised per cell. Where the tensor reports one
dominant orientation, the histogram carries the distribution — enough to tell two
perpendicular edge families (a grid, hatching) from isotropic noise.

Binning is integer: the gradient is folded into the upper half-plane and tested against the 8
bin-boundary rays by the sign of a cross product. The tests are nested, so the kernel
accumulates energy under each test's mask and the bins are differences of those sums.

## 3. Extrema densities — `COUNT_FEATURES`

The fraction of sampled pixels that are a **strict** maximum (`local_max`) or minimum
(`local_min`) of their 8-neighbourhood. A blob / speckle density: high for texture, dots and
noise, ~0 for smooth regions, and insensitive to contrast — a faint blob counts like a hard
one. A border pixel sits inside its own replicated neighbourhood and so is never an extremum.

## 4. Rotation-invariant uniform LBP — `LBP_FEATURES`

The LBP^riu2_{8,1} histogram of Ojala et al. (2002): each pixel is compared (`neighbour ≥
centre`) with its 8 neighbours in circular order; patterns with at most two 0/1 transitions
are labelled by popcount (bins 0..8), the rest fall in `lbp_nonuniform`. The only group
invariant to **any monotonic change of intensity** (gamma, exposure, contrast) and to 90°
rotations and reflections. Bins 0/8 are flat or spot-like, the middle bins edges and corners,
bin 9 high-frequency clutter; a flat image lands entirely in bin 8.

## 5. Derived descriptors — `DESCRIPTOR_FEATURES`

Nonlinear summaries of the sums above. A linear or shallow model forms weighted sums for free
but cannot cheaply form a ratio, a product or a standardised moment, so they are precomputed.

| name | definition | separates |
|---|---|---|
| `std_skew` | `m3 / var^1.5` | bimodal ink-on-paper from symmetric scenes |
| `excess_kurt` | `m4 / var² − 3` | leptokurtic text, platykurtic stripes, near-Gaussian photo |
| `edge_sharpness` | `energy / (var + 1)` | in-focus detail per unit contrast |
| `detail` | `energy · (1 − coherence)` | isotropic clutter (QR, foliage) from a clean edge |
| `hog_concentration` | `Σ pᵢ²` over HOG bins | one dominant orientation from a spread |
| `hog_cardinality` | fraction in the axis-aligned bins | horizontal / vertical structure (text, tables) |
| `grad_sparsity` | `n·Σg⁴ / (Σg²)²` = 1 + CV² of gradient energy | sparse strong edges from dense texture |
| `rms_contrast` | `√var / (mean + 1)` | brightness-relative contrast |

All but `detail` (which scales with gain²), `rms_contrast` (gain only) and `edge_sharpness`
(approximately) are invariant to `v → a·v + b`, `a > 0`.

## 6. Bar / stroke detector — `BARD_FEATURES`

A centre-surround test at lags `d ∈ {1, 2, 4}`: at lag `d` a pixel is **dark** if both taps at
`±d` are brighter than it, scoring the smaller of the two margins, and **light** if both are
darker. A step edge scores zero — one side matches the centre — so this fires on strokes and
thin structures of width ~`d`, which Sobel and LBP conflate with ordinary contrast. Horizontal
and vertical responses combine with max. A pixel counts only if its best response over all
lags reaches a noise gate of 8 grey levels; that gate is what makes the lag profile a shape
descriptor rather than a contrast one. A lag contributes on an axis only where both of its
taps lie inside the image.

| feature | definition |
|---|---|
| `bard_cover` | gated pixels / `n` |
| `bard_spec1..3` | share of the gated response mass at each lag — a stroke-width spectrum |
| `bard_peak` | mass-weighted mean lag index, in [1, 3] |
| `bard_peaked` | coefficient of variation of the lag profile: one dominant width vs texture (exactly 0 for a flat profile) |
| `bard_bal` | `(light − dark) / (light + dark)` ∈ [−1, 1]; +1 all light-on-dark |

All but `bard_cover` are ratios, hence invariant to stride and cell size.

## 7. Second-order texture — `TEXTURE_FEATURES`

From the same 3×3 neighbourhood Sobel reads. With the 1-D masks `L = [1 2 1]`, `E = [−1 0 1]`,
`S = [−1 2 −1]`, the Laws mask `XY` is `X` down the rows times `Y` along the columns; `LE` and
`EL` are Sobel's `gx` and `gy`, already covered by `energy`.

| feature | definition | reads as |
|---|---|---|
| `lap_var` | variance of the 4-neighbour Laplacian `up + down + left + right − 4c` | the standard focus / blur score; blob energy |
| `focus` | `lap_var / energy` | second-order detail per unit of first-order: blur lowers it, contrast does not |
| `laws_ee` | mean squared `EE` response | diagonal structure |
| `laws_ss` | … `SS` | spots |
| `laws_ls`, `laws_sl` | … `LS`, `SL` | vertical / horizontal lines |
| `laws_es`, `laws_se` | … `ES`, `SE` | line ends, ripples |
| `line_aniso` | `(ls − sl) / (ls + sl)` ∈ [−1, 1] | +1 when every line is vertical, as most scripts' strokes are |

The Laws energies scale with contrast²; normalise by `var` or `energy` downstream if needed.

## 8. Intensity moments — `MOMENTS`

```
mean = (1/n) Σ v      var = (1/n) Σ (v − mean)²      m3 = (1/n) Σ (v − mean)³      m4 = (1/n) Σ (v − mean)⁴
```

`maps` carries float32 copies; `Pyramid.moments` the float64 values (m3 and m4 outgrow
float32). The power sums are accumulated as exact integers and re-centred on the rounded mean
`K` by the binomial expansion before any floating point:

```
T1 = S1 − nK      T2 = S2 − 2K·S1 + nK²      T3 = S3 − 3K·S2 + 3K²·S1 − nK³      T4 = S4 − 4K·S3 + 6K²·S2 − 4K³·S1 + nK⁴
```

The residual offset `d = T1/n` is at most 0.5, so `var = T2/n − d²` has nothing left to
cancel: verified to rtol 1e-12 against numpy even on inputs drawn from `{250, 251}`, where the
naive `S2/n − mean²` loses ten digits.

## 9. Cross-channel covariance — `Pyramid.cross`

For each unordered channel pair, `[cov, corr]` of the two channels' pixel values over the
cell — the only feature relating channels. On RGB it separates neutral regions (all pairs ~+1)
from saturated colour; on HSV or opponent spaces it picks up chromatic structure. One extra
sum per pair, `Σ v_i·v_j`. Pairs are listed in `fc.channel_pairs`; the group is empty for a
single channel and for `C > 8`, where the quadratic pair count is not paid silently.

## 10. Projection profiles — `Pyramid.profiles`

`(rows, cols)`: the mean intensity of every sampled row, `(R, C)`, and of every sampled column,
`(K, C)`, over the whole frame. Correlate a frame's profiles with the previous frame's for a
1-D shift estimate at the sampling pitch; valleys of the row profile are gaps between text
lines. (At cell pitch the same profiles are the `mean` feature averaged along an axis.)

## 11. Perceptual hashes — `Pyramid.hashes`

Three `imagehash`-compatible hashes, one `uint64` per channel, bits packed row-major MSB first.
All are readouts of a box-pooled mean grid, i.e. of the `S1` sums the pass already has:

| hash | grid | bit rule |
|---|---|---|
| `ahash` | 8×8 | cell mean > mean of the 64 cells |
| `whash` (Haar) | 8×8 | cell mean > median of the 64 cells |
| `phash` | 32×32 | low-frequency 8×8 of a 2-D DCT-II > its median |

When the finest grid tiles 32×32 the hash grid is a block sum of the finest cells and the
hashes cost a small end-of-pass DCT; a coarser finest grid falls back to a light second sum
over rows already in cache. `imfeat` box-pools where `imagehash` resizes with Lanczos, so the
bits are identical where no resize occurs (8×8 / 32×32 inputs, asserted by the tests) and on
natural 256×256 frames the relative Hamming distance is ~0.00 (`whash`), ~0.01 (`phash`) and
~0.12 (`ahash`).

---

## The raw sums

`compute()` and the internal pyramid hold, per cell and channel, 44 additive int64 sums:

| slots | contents |
|---|---|
| 0..3 | `Sxx Syy Sxy count` |
| 4..12 | HOG: gradient energy per orientation bin |
| 13, 14 | strict local max / min counts |
| 15..18 | `S1 S2 S3 S4` — power sums of the pixel value |
| 19..28 | LBP bin counts |
| 29 | `Σ|grad|⁴` |
| 30..35 | bar detector: gated-pixel count, response mass per lag, dark and light mass |
| 36, 37 | Laplacian: `ΣL`, `ΣL²` |
| 38..43 | squared Laws responses: EE, SS, LS, SL, ES, SE |

plus one `Σ v_i·v_j` per channel pair. `compute()` exposes the first five groups and the pair
sums by name; the rest are internal.

## Using the pyramid

The levels form a dyadic feature pyramid, finest first, the last a single global cell — an FPN
P3–P6 shape family that a neck, a shared dense head or a U-Net decoder can consume as is. On a
1024² input with `grid=[(6,6),(5,5),(4,4),(3,3),(2,2),(1,1)]` and `C=3` the levels are
`(64, 64, 162)`, `(32, 32, 162)`, … `(2, 2, 162)` and a `(162,)` global vector, at strides 16,
32, … 512 against the input.

For dense prediction the finest level sets the output resolution, and it is a cell grid, not a
pixel grid. Push `grid` finer for a denser map, but keep roughly four sampled pixels per cell
per dimension (`cell_px / stride ≥ 4`) or the moment and histogram features stop meaning
anything; upsampling is a decoder's job. The finest level is also the only one strictly
needed: every coarser level is an exact sum of it, provided because it costs almost nothing.
