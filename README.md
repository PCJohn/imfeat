# imfeat

Fast, exact, single-pass image feature extraction on CPU.

`imfeat` walks a uint8 image **once** and returns, for every channel and every cell of a dyadic
grid pyramid, 54 classical features: intensity **moments**, gradient **structure tensor**, an
**orientation histogram**, **extrema densities**, a rotation-invariant **LBP** histogram,
nonlinear **descriptors** of those, a **bar / stroke detector** and second-order **texture**
energies (Laplacian, Laws). It also returns **cross-channel** covariance per channel pair,
whole-frame **projection profiles** and three `imagehash`-compatible **perceptual hashes**.

It is a C++17 / [Highway](https://github.com/google/highway) SIMD core behind a small
[nanobind](https://github.com/wjakob/nanobind) Python API, built for the front of a real-time
pipeline: well-understood, weakly correlated features, cheap enough for every frame (about
0.4 ms for a 256×256×3 frame at stride 2 on one core), that a net, a tree or a rule can
consume directly. Every integer sum is exact and the output is bit-identical at any thread count.

```python
import cv2, imfeat

hsv = cv2.cvtColor(cv2.imread("frame.png"), cv2.COLOR_BGR2HSV)      # (H, W, 3) uint8
fc = imfeat.FeatureComputer(shape=hsv.shape, grid=[(5, 5), (4, 4)], stride=2)
p = fc.features(hsv)                       # one pass: all channels, both levels, global

p.maps[0]      # (32, 32, 162) float32   3 channels x 54 FEATURE_NAMES, channel-major
p.maps[-1]     # (162,)        float32   the whole-frame (global) level
p.moments[0]   # (32, 32, 3, 4) float64  [mean, var, m3, m4] at full precision
p.summary[0]   # (54, 3, 4)    float64   each feature's [min, max, mean, std] over the cells
p.cross[0]     # (32, 32, 3, 2) float32  [cov, corr] per channel pair
p.hashes       # (3, 3)        uint64    [ahash, whash, phash] x channel
p.profiles     # (rows, cols)  float64   mean of each sampled row / column, per channel

f = p.maps[0].reshape(32, 32, 3, 54)       # free view: (cy, cx, channel, feature)
energy_v = f[..., 2, imfeat.FEATURE_NAMES.index("energy")]
```

More detail lives in two companion documents:
[docs/FEATURES.md](docs/FEATURES.md) (every feature: definition, formula, invariances, uses)
and [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) (how the core got fast: what worked, what did
not, how it was measured and verified).

---

## Install

```bash
pip install git+https://github.com/PCJohn/imfeat      # needs a C++17 compiler and CMake
pytest -v -m "not bench"     # unit tests
pytest -v -s                 # + benchmark tables
pytest -v -s --full          # + cross-checks vs opencv-python / imagehash (install those first)
```

Tested on Linux/GCC and Windows/MSVC (x86-64). macOS/arm64 (NEON) runs the same Highway code
but has not been re-measured since the kernel rewrite described below.

---

## API

Construct once for a fixed input shape, then call per frame. Every buffer is allocated up
front; in steady state a frame allocates nothing.

### `FeatureComputer(shape, grid, stride=None, channels=None, channel_axis=-1, threads=1)`

| arg | meaning |
|---|---|
| `shape` | exact shape of every image: `(H, W)` or `(H, W, C)`, uint8 |
| `grid` | `k` → 2^k × 2^k cells; `(ky, kx)` → 2^ky × 2^kx; or a **list**, finest → coarsest, for a pyramid. Levels must nest; the finest must divide `(H, W)` |
| `stride` | `None` / `int` / `(sy, sx)`: which pixels are sampled (restarting in every cell) |
| `channels` | channels to process; default all |
| `channel_axis` | default `-1` (OpenCV layout); read through a zero-copy transpose |
| `threads` | bands of cell rows processed in parallel. Default `1`. Output is bit-identical at any count; `.threads` reports what was used, `imfeat.cpu_count()` what is available |

### `.features(img) -> Pyramid`

A namedtuple. Lists have one entry per grid level, finest first, then a 1-cell `global` level
(whose shapes drop `(cy, cx)`). `C` = selected channels, `F = len(imfeat.FEATURE_NAMES) = 54`.

| field | shape per level | dtype | contents |
|---|---|---|---|
| `maps` | `(cy, cx, C*F)` | float32 | every per-channel feature, **channel-major** over `FEATURE_NAMES` |
| `moments` | `(cy, cx, C, 4)` | float64 | `MOMENTS` at full precision (m3, m4 outgrow float32) |
| `summary` | `(F, C, 4)` | float64 | each feature over the level's cells as `SUMMARY_STATS = [min, max, mean, std]`; grid levels only |
| `cross` | `(cy, cx, P, 2)` | float32 | `CROSS_FEATURES = [cov, corr]` per channel pair (`fc.channel_pairs`); empty for one channel or `C > 8` |
| `hashes` | `(3, C)`, not a list | uint64 | `HASHES = [ahash, whash, phash]`, whole frame |
| `profiles` | `((R, C), (K, C))` | float64 | mean of each sampled row and of each sampled column, whole frame |

`FEATURE_NAMES` concatenates the group tuples in order — index features by name, never by
number:

| tuple | count | offset |
|---|---|---|
| `FEATURES` (structure tensor) | 5 | 0 |
| `HOG_FEATURES` | 9 | 5 |
| `COUNT_FEATURES` (extrema) | 2 | 14 |
| `LBP_FEATURES` | 10 | 16 |
| `DESCRIPTOR_FEATURES` | 8 | 26 |
| `BARD_FEATURES` | 7 | 34 |
| `TEXTURE_FEATURES` | 9 | 41 |
| `MOMENTS` | 4 | 50 |

The returned arrays are views of one pooled block that they keep alive: they stay valid (and
writable) across later calls and after the computer is gone, and the block is reused once the
last of them is dropped.

### `.compute(img) -> dict`

The raw **int64 sums** before any nonlinear step, keyed `{group}_{level}`: `struct_i`
`[Sxx, Syy, Sxy, count]`, `mom_i` `[S1..S4]`, `hog_i`, `cnt_i`, `lbp_i`, and `xchan_i`
(`Σ v_i·v_j` per channel pair). They are purely additive, so any pooling over cells, levels,
channels or frames is one `numpy` sum away and exact:

```python
raw  = fc.compute(img)
ps   = raw["mom_0"][8:16, 4:12].sum(axis=(0, 1))            # (C, 4) power sums of a region
n    = raw["struct_0"][8:16, 4:12, :, 3].sum(axis=(0, 1))   # (C,)   its sample count
mean = ps[:, 0] / n
```

---

## Features at a glance

Per channel and per cell ([docs/FEATURES.md](docs/FEATURES.md) has the definitions):

| group | what it measures | notable property |
|---|---|---|
| moments | mean, variance, 3rd and 4th central moments | exact to 1e-12 even on low-variance cells |
| structure tensor | edge energy, coherence, double-angle orientation, Shi–Tomasi cornerness | orientation averages correctly across cells |
| HOG | 9-bin orientation histogram of gradient energy | integer binning, no `atan2` |
| extrema | density of strict local maxima / minima | contrast-insensitive blob and speckle density |
| LBP | rotation-invariant uniform LBP histogram (10 bins) | invariant to any monotonic intensity change |
| descriptors | skew, kurtosis, edge sharpness, detail, HOG concentration / cardinality, gradient sparsity, RMS contrast | ratios and products a linear model cannot form |
| bar detector | stroke cover, stroke-width spectrum, peak width, peakedness, polarity | fires on strokes, not on step edges |
| texture | variance of the Laplacian, focus, six Laws 3×3 energies, line anisotropy | blur / sharpness, blobs, vertical-vs-horizontal strokes |
| cross-channel | covariance and correlation per channel pair | the only group relating channels |
| profiles, hashes | row / column projection profiles; aHash, wHash, pHash | whole frame: 1-D registration, near-duplicate keys |

Every feature is normalised per sample, so no level carries a cell-area factor and one shared
head can read every level. Standardise per feature over a dataset before training (the
dynamic range spans 1e-2 to 1e6); `summary` is the cheap way to collect the statistics.

---

## How it works

**Additive integer sums.** Every per-pixel quantity — gradient products, histogram votes,
counts, powers of the pixel — is accumulated as an exact int64 sum into the finest cell of its
channel: 44 sums per cell per channel, plus one per channel pair. Everything nonlinear
(eigenvalues, central moments, normalisation, ratios) is derived from those sums once per
cell. Two consequences are the whole design: every feature shares **one traversal** of the
image, and a coarser cell is the exact sum of the cells inside it, so **pyramid depth is
nearly free**.

**Columns are the SIMD lane.** Each row is de-interleaved once into a rolling window of planar
rows (split by column phase when the stride is even, so sampled columns become adjacent). One
kernel reads a vector-wide block of columns from rows `r-1, r, r+1` and computes everything
in-register. Three exact rewrites turn every sum into a plain vector add:

* **HOG without a scatter.** The eight bin-boundary tests are nested, so the kernel adds
  `|grad|²` under each test's mask and the bins fall out as differences of those cumulative
  sums. Each test is one `pmaddwd` over interleaved `(gy, gx)` — integer, no `atan2`.
* **Moments of `v − 128`.** Centred pixels keep `w² ≤ 2¹⁴` inside 16-bit multiply-adds; a
  binomial shift restores the raw power sums exactly.
* **Shared row terms.** Each 3×3 row yields three 1-D responses once; Sobel, the Laplacian
  and all six Laws masks are 1-D combinations of them.

Accumulators ride down all the sampled rows of a cell row and are folded from column lanes
into cells once per block, so memory is written once per cell row rather than once per row.

**Cells are the second SIMD lane.** A block's cell sums stay contiguous per sum in a buffer
that never leaves L1, and go three ways at once: into the 54 features, four cells per vector,
through one compiled routine that every cell of every level shares; into the moments,
re-centred on the rounded mean exactly (in doubles, which hold every integer involved below
2⁵³); and, added in neighbouring pairs, straight into the next level's row. The finest level —
most of the pyramid — is never stored or laid out as slots in `features()`; only `compute()`
materialises it.

**Nothing avoidable on the hot path.** No divides per row, block or cell (window slots wrap,
offsets are tabulated, indices are running counts, the rounded mean is a multiply corrected by
its exact remainder); no buffer zeroed between frames (first contributions store instead of
add); no output copied (arrays view a pooled block).

**Exactness.** Integer sums are bit-exact against numpy oracles. Floats are derived by one
routine for all cells, so a grid derives to the same bits as any level of any pyramid and at
any thread count. The moments' floating-point tail spells out its operations, fused
multiply-adds included, so it does not drift with the compiler.

---

## Performance

`features()` end to end, 4 pyramid levels, every feature on every channel, one thread, ms.
Two x86 machines: a 16-core 2.1 GHz Xeon (Linux, GCC) and a 22-thread laptop (Windows, MSVC).
`pytest -v -s --full tests/test_bench.py` regenerates everything here.

Finest grid 32×32 (Xeon / laptop):

| input | stride=1 | stride=2 | stride=4 | stride=8 |
|---|---|---|---|---|
| 256×256×3 | 0.90 / 0.73 | 0.43 / 0.35 | 0.36 / 0.30 | 0.33 / 0.27 |
| 512×512×3 | 2.42 / 2.06 | 0.95 / 0.77 | 0.48 / 0.40 | 0.41 / 0.35 |
| 1024×1024×3 | 8.06 / 6.97 | 2.57 / 2.20 | 1.12 / 1.00 | 0.73 / 0.57 |
| 1024×1024 (C=1) | 2.71 / 2.29 | 0.88 / 0.73 | 0.39 / 0.30 | 0.24 / 0.19 |

Cost is linear in channels (0.12–0.17 ms per channel at 256², stride 2, from C=1 to C=16) and
in sampled pixels, plus a per-cell term that no stride removes: at 256² with 8 px cells, going
from stride 2 to 8 saves little, because what is left is folding, deriving and summarising 54
features for 1024 cells × 3 channels.

Accuracy of a stride against the exact `stride=1` output (256×256×3, 32×32 grid; properties of
the algorithm, not the machine):

| stride | mean err (grey levels) | energy_r | orientation err | hog_cos | lbp_cos |
|---|---|---|---|---|---|
| **2** | **4.6** | **0.954** | **3.4°** | **0.926** | **0.990** |
| 4 | 10.8 | 0.822 | 4.5° | 0.838 | 0.958 |
| 8 | 23.9 | 0.570 | 5.0° | 0.761 | 0.834 |

Threads (`features()` min, Xeon / laptop):

| frame | grid, stride | 1 thread | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 256×256×3 | finest-32, s2 | 0.42 / 0.35 | 0.45 / 0.27 | 0.35 / 0.31 | 0.34 / 0.29 |
| 512×512×3 | finest-32, s2 | 0.93 / 0.77 | 0.75 / 0.45 | 0.57 / 0.37 | 0.60 / 0.48 |
| 1024×1024×3 | finest-64, s2 | 3.64 / 2.99 | 2.63 / 1.66 | 1.88 / 1.25 | 1.62 / 1.59 |
| 1024×1024×3 | finest-32, s4 | 1.12 / 0.92 | 0.82 / 0.53 | 0.60 / 0.42 | 0.51 / 0.53 |

### Using it well

* **Keep `cell_width / stride ≥ 4`.** Stride's currency is samples per cell. On 8 px cells
  `stride=2` is the sweet spot; on a 1024² frame with 32 px cells `stride=8` is still safe and
  11× faster than `stride=1`. Moments survive aggressive strides far better than the
  structure maps do.
* **Scale by resolution first, stride second.** Halving the resolution is a clean 4× and
  keeps samples per cell intact.
* **Take all the pyramid levels** — they are nearly free.
* **Prefer power-of-two strides and cell widths.** Anything is exact, but cells of 4, 8, 16…
  sampled columns tile a vector block; other widths are walked one masked block per cell.
* **Threads pay from about a millisecond per frame.** Workers park between frames (the cores
  are meant to be shared), so below that the wake-up plus the serial tail (summary, coarse
  levels, hashes) rivals the work saved, and whether two threads beat one is up to the
  machine: at 256² they do on the laptop above and do not on the Xeon. Measure first.

---

## Exactness and testing

710 tests. Integer sums are compared **exactly** with numpy / OpenCV oracles over block
geometries (cells narrower and wider than a vector, masked meshes, odd strides, rows that
overrun a block, tiny images); floats with the oracles' tolerances; hashes with `imagehash`.
Invariants are tested directly: multi-channel equals per-channel, a pyramid equals separate
computers bit for bit, output is bit-identical across 1, 2, 3, 4 and 8 threads, results survive
later calls and outlive their computer. Development additionally compared 14,328 output arrays
bit for bit against the previous build on AVX2, AVX-512 and SSE4 after every change, and ran
the suite under AddressSanitizer.

---

## Optimization notes

Against the original single-pass implementation this core is 2.2–3.3× faster for three
channels and up to 8× for one (13.8 M → 5.2 M instructions and 35,000 → 13 integer divisions
per 256×256×3 frame), while computing 9 more features, with every previous output unchanged
bar two bar-detector corner cases that were bugs.
The short version, with the full log in [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md):

* **What worked:** choosing the SIMD lane twice (columns for pixels, cells for the per-cell
  stage); algebraic rewrites that make every sum a vector add; keeping data in the layout the
  next stage reads; never storing the finest level; removing every per-row / per-cell divide;
  pooled zero-copy outputs.
* **What did not:** vectorising the scalar derive with two-lane divides; a SIMD transposition
  of cell sums; a generic roll-up helper with runtime group sizes; delaying reads to dodge
  store forwarding; compiler attributes to switch off `-ffast-math` locally.
* **Lessons:** count work (instructions and divider operations), not just milliseconds —
  timings hid a 10% cost in integer divides and flattered a change that removed nothing;
  bit-exactness across code paths comes from sharing one compiled body, not from flags.

---

## Scope

Input is always **uint8**, 2-D or 3-D, any channel count. Floating-point input, N-D tensors
and arbitrary axis reductions are out of scope: the uint8 assumption is what makes the exact
integer accumulators possible. Reductions beyond cells / levels / global are one `numpy` sum
over `compute()`'s output away.
