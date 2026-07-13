# imfeat

Fast single-pass image feature extraction on CPU: intensity **moments** and gradient
**structure** statistics over a dyadic pyramid, for every channel, in one traversal of
the image. Merges [`tensorstats`](https://github.com/PCJohn/tensorstats) (moments) and
[`structstats`](https://github.com/PCJohn/structstats) (structure) — which previously
each made their own pass — into a single C++/nanobind core.

```python
import imfeat

fc = imfeat.FeatureComputer(shape=(256, 256, 3), grid=[(5, 5), (4, 4), (3, 3), (2, 2)])
f = fc.features(hsv_u8)          # one pass, all channels, all levels

f["mom_0"]        # (32, 32, 3, 4)  float64  [mean, var, m3, m4]
f["struct_0"]     # (32, 32, 3, 5)  float32  [energy, coherence, ori_cos, ori_sin, cornerness]
f["hog_0"]        # (32, 32, 3, 9)  float32  L1-normalised orientation histogram
f["cnt_0"]        # (32, 32, 3, 2)  float32  [local_max, local_min] densities
f["mom_global"]   # (3, 4)                   whole-frame per-channel moments
```

## Why one pass is possible

Every per-pixel quantity is accumulated as an **additive int64 sum** into the finest
grid cell of its channel — 19 sums per cell per channel:

| slots  | contents                                            |
|--------|-----------------------------------------------------|
| 0..3   | `Sxx Syy Sxy count` — gradient structure tensor      |
| 4..12  | HOG: gradient energy per orientation bin (9 bins)   |
| 13,14  | strict 8-neighbourhood local max / min counts        |
| 15..18 | `S1 S2 S3 S4` — raw power sums of the pixel value    |

Because the sums are additive, **coarser pyramid levels and the frame-wide reduction are
exact sums of the finest cells** — pyramid depth costs no extra image reads. Central
moments, tensor eigen-features and the histogram normalisation are nonlinear, so they are
derived once per cell at the very end.

Moments come from power sums rather than the textbook two-pass (mean, then deviations):
that is what lets them share the structure pass. Cancellation in `S2/n - mean²` is avoided
by shifting the sums by the integer nearest the mean before expanding — every intermediate
stays exact in int64, and the residual offset is ≤ 0.5.

## Channels

Any channel count: 1 (2-D input), 3 (BGR/HSV/LAB), or arbitrary (hyperspectral). All
channels are processed in the same spatial pass — the channel is the SIMD lane — and
appear as their own axis in the output, which is dropped for 2-D input. `channels=[...]`
restricts which ones are computed; the default is all.

## Grid

Cells are assigned by floor division (`cell = row * n_cells // H`), so any image shape
works. Cell counts are powers of two, so level *k*'s cell index is level 0's shifted
right: a given pixel's features at every scale are an O(1) lookup — convenient for
building per-pixel feature vectors to feed a downstream model.

`stride=(sy, sx)` subsamples which pixels accumulate while keeping the gradient stencil
and the cell boundaries at full resolution. Columns restart the stride at each finest-cell
edge, so every cell keeps an identical sample count. Stride 2–8 trades accuracy for speed.

## API

```
FeatureComputer(shape, grid, stride=None, channels=None, channel_axis=-1)
  .features(img) -> derived maps   (float32 structure / float64 moments)
  .compute(img)  -> raw int64 accumulators (additive; pool them yourself)
```

Both return a flat dict keyed `{group}_{level}` for `group` in
`struct | hog | cnt | mom` and `level` in `0..K-1 | global`.

## Cost

One pass over a 256x256x3 frame, 4 pyramid levels, every feature on every channel:
~1.0 ms at stride 2, ~2.5 ms at stride 1 (AVX2 laptop class). Roughly half of that is
the accumulator maths -- HOG's orientation binning is the single largest item -- and
half is the per-row preparation and per-cell flush. Pyramid depth is free (coarser
levels are sums of finer cells) and `features()` costs the same as `compute()` (the
derivation is negligible next to the sweep). Channels are cheap after the first: the
channel is the SIMD lane, so C=3 costs ~1.7x C=1, and beyond the 4-lane group the
marginal cost is ~0.27 ms/channel at 256x256.

`bench_compare.py` reports latency against the old two-package pipeline plus the
accuracy cost of `stride`, swept over input sizes and channel counts.

## Build

```
pip install .
pytest tests -q          # 351 tests: exactness vs numpy oracles, semantics, edges
```
