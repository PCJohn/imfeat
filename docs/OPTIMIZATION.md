# How the core got fast

An engineering log of the optimisation of `imfeat`'s C++ core: where it started, the
architecture it ended with, what each step bought, what was tried and dropped, the lessons,
and how every change was measured and verified.

The constraints throughout: no approximations; every integer sum bit-exact; no float
tolerance loosened; nothing tuned to one machine (no "more threads", no per-CPU parameters).

## Starting point

"The original" below means the implementation at commit `1c35817`, the last one before this
work. It already had the design that matters most: a single pass over the image, additive
int64 sums per cell, nonlinear features derived from the sums at the end, a pyramid that is a
sum of its finest level, and a Highway SIMD core. What it did differently:

* **Channels were the SIMD lane.** Pixels were walked one column at a time with up to four
  channels packed into a vector, through separate per-row stages (prepare, 3×3 stencil,
  accumulate) that passed intermediate row buffers. A single-channel image used one lane in
  four; throughput depended on the channel count.
* Accumulators were zeroed every frame; the finest level was written out in full, then read
  back twice (to roll it up, then to derive it); the derive step was scalar; outputs were
  copied and re-joined in Python.

## Terms

| term | meaning |
|---|---|
| cell, cell row | a cell of the finest grid; one row of such cells |
| block | one vector's worth of adjacent sampled columns (32 with AVX2), the unit the kernel works on |
| lane | one element of a SIMD vector |
| sums | the 44 additive int64 quantities accumulated per cell per channel (plus one per channel pair) |
| slots | those sums laid out cell by cell, `[cell][channel][44]`: the form the pyramid levels and `compute()` use |
| fold | summing an accumulator's column lanes into per-cell sums |
| derive | turning a cell's sums into its float features |
| band | the contiguous range of cell rows one thread processes |
| Xeon, laptop | the two test machines: a 16-core 2.1 GHz Xeon (Linux, GCC) and a 22-thread laptop (Windows, MSVC) |
| development VM | the single-core shared cloud machine the work was done on; its timings wander by ±5% |

## Result

`features()` end to end, one thread, 4 pyramid levels, every feature on every channel, on the
Xeon, in ms. The laptop shows the same ratios.

| input, finest grid, stride | original | now | speed-up |
|---|---|---|---|
| 256×256×3, 32×32, stride 2 | 1.17 | 0.42 | 2.8× |
| 256×256×3, 64×64, stride 2 | 3.03 | 1.31 | 2.3× |
| 512×512×3, 32×32, stride 2 | 2.56 | 0.93 | 2.8× |
| 1024×1024×3, 32×32, stride 1 | 26.6 | 8.05 | 3.3× |
| 1024×1024×1, 32×32, stride 1 | 22.5 | 2.68 | 8.4× |
| 1024×1024×16, 32×32, stride 2 | 39.9 | 16.0 | 2.5× |

"Now" also computes nine features and two profiles that the original did not.

The same story in machine-independent terms: operations executed per `features()` call on a
256×256×3 frame, 32×32 finest grid, stride 2 (AVX2 build, counted with valgrind; "vector"
operations process four doubles each):

| build | instructions | integer divides | FP divides | FP square roots |
|---|---|---|---|---|
| original (`1c35817`) | 13.8 M | 34,800 | 53,600 scalar | 16,400 scalar |
| after the kernel and streaming rewrites (steps 1–3 below) | 5.33 M | 10,300 | 40,100 scalar | 16,400 scalar |
| … with the texture features and profiles added (step 4) | 6.18 M | 11,300 | 44,100 scalar | 16,400 scalar |
| … with a first SIMD derive (see *What did not work*) | 6.37 M | 11,300 | 7,400 scalar + 7,100 vector | 4,200 scalar + 3,200 vector |
| final (step 5) | 5.23 M | 13 | 4,400 scalar + 7,900 vector | 4,200 scalar + 3,200 vector |

## Architecture

1. **Rows → planes.** Each image row is de-interleaved once into a rolling window of planar
   rows. With an even stride the planes are also split by column phase (up to 8 phases), so
   the sampled columns are adjacent: a stride of 2 walks half as many blocks instead of
   masking half of each one.
2. **Kernel: columns as lanes.** One kernel takes a block of columns from rows `r−1, r, r+1`
   (plus the bar detector's taps) and computes every per-pixel quantity in registers, with no
   intermediate row buffers. About 76 vector accumulators per channel are updated per row of a
   block. The first row of a cell row *assigns* them, so nothing is ever zeroed.
3. **Fold: lanes → cells.** Once per block per cell row, each accumulator's lanes are summed
   into per-cell sums with widening pairwise adds. When a block holds whole cells, those sums
   live in a block-sized buffer, contiguous per kind of sum, that never leaves the L1 cache.
4. **Cells as lanes.** From that buffer, four cells per vector: an exact integer-to-double
   conversion, the moments (re-centred exactly, in doubles), the 54 features through one
   compiled routine, a transposed write into the output, the summary fold, and a pairwise add
   straight into the level-1 row being built. The finest cells are never laid out as slots.
5. **Coarser levels.** The level-1 row becomes slots when it is complete; higher levels roll
   up from slots and derive through the same routine, gathering four cell-channels at a time.
6. **Output.** Everything is derived into one pooled block of memory; the returned arrays are
   views that keep it alive, and the block is reused once they are dropped. No copies and, in
   steady state, no allocation.

## What worked

Chronologically. Times are single-thread `features()`; instruction counts are per frame at
256×256×3, 32×32 finest grid, stride 2.

| step | change | measured effect |
|---|---|---|
| 1 | Kernel rewritten with columns as the SIMD lane (details below) | Xeon: 256×256×3 stride 2, 1.18 → 0.74 ms; 1024×1024×1 stride 2, 7.08 → 0.92 ms. Cost per channel became flat in the channel count |
| 2 | 8- and 16-channel de-interleave routed through the SIMD phase splitter | removed a regression step 1 had introduced (laptop, 1024×1024×16 stride 8: 11 → 19 → 4.9 ms) |
| 3 | Finest level streamed instead of stored; outputs derived into pooled blocks; HOG / LBP normalised in SIMD; summary folded in batches | development VM: 0.55 → 0.37 ms at 256×256×3 stride 2 |
| 4 | Nine texture features and the projection profiles added | a cost: +20% |
| 5 | Per-cell stage reworked (details below) | instructions 6.37 M → 5.23 M (−18%); Xeon 0.58 → 0.42 ms (−28%); laptop 0.45 → 0.35 ms (−22%) |

**Step 1, the kernel.** Besides the change of lane, three algebraic rewrites turn every sum
into a plain vector add:

* *HOG without a scatter.* The eight bin-boundary tests are nested (a gradient past boundary
  `j` is past every earlier one), so the kernel adds `gx² + gy²` under each test's mask and
  the bins are differences of those cumulative sums. Each test is one 16-bit multiply-add
  (`pmaddwd`) over interleaved `(gy, gx)`.
* *Moments of `v − 128`.* Centred pixels keep squares within 16-bit multiply-adds; a binomial
  shift restores the raw power sums exactly.
* *Shared row responses.* Each 3×3 row yields three 1-D responses once; Sobel, the Laplacian
  and six Laws masks are 1-D combinations of them. LBP's last bin is a remainder, and the bar
  detector runs entirely in saturating uint8 arithmetic
  (`min(max(l−c, 0), max(r−c, 0)) = SatSub(min(l, r), c)`).

**Step 5, the per-cell stage.** A profile by instruction count showed that with 8 px cells the
kernel was 20% of the frame and per-cell work 70%, about 1,000 instructions per cell per
channel. What was changed, and what each change removed from the frame:

| change | effect on the frame |
|---|---|
| cell sums kept per block in L1, flushed as soon as the block is done | −3 to −5% of the time (development VM) |
| summary written by C++ in its final layout, no Python join | about −2% of the time (development VM) |
| summary's min, max, sum and sum of squares in one pass instead of three | −3.2% of the instructions |
| level 1 rolled up from the block's sums in neighbouring pairs; finest cells never become slots | −7.1% of the instructions |
| moments re-centred exactly in doubles, four cells per vector, one shared floating-point routine | −4.1% of the instructions |
| output transposition without per-lane branches | −4.6% of the instructions |
| no divides per row, block or cell | no instructions, but most of the gap between −18% instructions and −28% time on the Xeon |

The divides deserve a note. Thousands of integer divisions per frame (34,800 in the original,
still 11,300 late in the work) sat in bookkeeping: a modulo per window-row lookup, tap offsets,
cell and row indices, pyramid factors, the rounded mean. They became wrapped counters,
per-configuration tables, shifts, and a multiply by `1/n` corrected by its exact remainder. An
instruction count treats a 40-cycle divide like an add, so this was invisible until divider
operations were counted separately.

## What did not work

* **Transposing cell sums into slots with SIMD** — slower, twice. The eventual answer was not
  to build slots for the finest level at all.
* **Two-lane vector divides inside the scalar derive** (pairing two different divisions of
  one cell) — slower: assembling the vectors sits on the critical path.
* **The first SIMD derive.** Four cells per vector cut divider operations fourfold but
  *executed more instructions* than the scalar derive it replaced (6.18 M → 6.37 M): the slot
  conversion and a branchy output transposition ate the gain, and wall-clock comparisons could
  not tell. It paid only after the transposition was fixed and the slots it made redundant
  were no longer written.
* **A generic roll-up helper** with a run-time group size and `/`, `%` by the pyramid factor:
  30% of the frame. The specialised pairwise version with shifts is 4%.
* **Reading a block's sums one block late**, and **half-width loads** sized to the earlier
  stores — both meant to avoid store-forwarding stalls. No change: that was not the
  bottleneck.
* **`__attribute__((optimize("no-fast-math")))`** to make one function strict under
  `-ffast-math`. Lambdas inside it stopped inlining (−20%), and GCC still regrouped
  `(6·d²)·u₂` among the intrinsics. What does pin a product is writing it as a fused
  multiply-add with a zero addend.
* **Aligning band boundaries to level-1 rows** for threading — it broke the contract that
  `threads` is honoured up to the number of cell rows. Rows whose parents start in another
  band are instead kept as slots and added after the threads join.
* **Kernel ideas rejected on operation count:** HOG with four boundary tests plus mirror
  logic (same count); 16-bit boundary tests (the products need 32 bits); Laws energies from
  sums of row products (nine products for eight energies, worse at stride ≥ 2); caching row
  responses between image rows (≤ 3%, stride 1 only); LBP by lookup table or bit packing (no
  fewer operations). Adding two rows together before each accumulator update would save about
  5% of the kernel and was not pursued.
* **Features considered and not added:** per-cell min / max (not additive, so it breaks the
  pyramid); intensity or centre-symmetric-LBP histograms (many sums each); signed HOG
  (doubles the HOG cost); LBP concentration and extrema balance (exact functions of existing
  outputs — a linear fit from the 54 features already explains 74% and 60% of their
  variance); temporal difference energy (would make `features()` stateful).

## Lessons

1. **Choose the SIMD lane per stage.** Ask what is contiguous and independent *at that
   stage*: columns for per-pixel work, cells for per-cell work. Channels as lanes tied
   throughput to the channel count and left three quarters of each vector idle for grayscale.
2. **Use algebra to make sums plain adds** (cumulative histogram bins, centred moments,
   remainder bins) rather than trying to make a scatter or a branch fast.
3. **The layout between stages costs as much as the arithmetic.** Do not materialise a
   representation unless a consumer needs it, and consume data while it is still in L1.
4. **Count work, not just milliseconds.** Instruction counts and divider-operation counts are
   identical from run to run. Timings on a busy machine flattered one change that removed
   nothing and hid a 10% cost in integer divides.
5. **Integer divides hide in bookkeeping** — ring-buffer modulos, index arithmetic — not in
   the numerical code. Count them.
6. **Exactness comes from structure, not compiler flags.** Keep integers until the end; use
   doubles for integer arithmetic only where every value is provably below 2⁵³; send anything
   that must be bit-equal across code paths through one compiled body.
7. **Build the oracle first.** An independent numpy oracle per feature made the kernel
   rewrites safe, and a bit-identity harness across instruction sets caught what unit tests
   had not, including a bug in the original's bar detector.
8. **Re-measure threading after optimising.** As single-thread latency falls, the fixed cost
   of waking a parked worker dominates; two threads went from a gain to a loss at 256×256 on
   one of the two test machines.

## Measuring

* **Wall clock**: minimum over many frames, builds interleaved in one session. The first tool
  and the least trustworthy; ±5% on a shared machine is noise.
* **Instructions per frame**: `valgrind --tool=cachegrind` on a run of 4 frames and a run of
  24, differenced. `callgrind` with a few functions marked `noinline` gives the split by
  stage.
* **Divider operations per frame**: `callgrind --dump-instr=yes`, summing the execution
  counts of every `div`, `idiv`, `vdiv*` and `vsqrt*` instruction in the library.

At 256×256×3, stride 2, 8 px cells the kernel is about 20% of the instructions and the
per-cell stage 70%. At stride 1 with 16 px cells the kernel is about two thirds, a quarter of
it accumulator loads and stores.

## Verifying

* The repository's tests, extended from 599 to 710: oracle tests for the bar detector (numpy)
  and for the texture sums (exact), over block geometries chosen to reach every kernel path;
  output lifetime; profiles; threads 1, 2, 3, 4 and 8.
* A bit-identity harness: 14,328 output arrays (6 shapes × 4 channel counts × 5 grids ×
  6 strides × 2 kinds of image, at 1, 2 and 3 threads) hashed per build and compared with the
  previous build after every change, on AVX2, AVX-512 and SSE4 builds.
* AddressSanitizer over the suite. It caught a block-buffer stride that overran on tiny grids.
* clang-format, cppcheck and `-Wall -Wextra -Wshadow`; ruff, black and mypy for the Python.

Differences observed against the original: none on AVX2 or AVX-512 builds, apart from two
bar-detector corner cases that were bugs in the original (its results depended on the
instruction set for large cells at an even stride, and its sampling mesh did not restart per
cell as every other feature's does). On the SSE4 build, which has no fused multiply-add, the
float64 moments move by at most one unit in the last place when the code around them changes.

## Open ideas

* **A summary independent of the band split** (sum per cell row, then the rows in order), so
  that threads can share it. Nothing at one thread; an estimated 5–15% at 2–4 threads on
  small-cell grids; it would move `summary` means and standard deviations by one unit in the
  last place, once. Not done.
* **int32 cell sums where a level's sample count bounds them** (all but `Σw⁴` and `Σg⁴` at
  typical finest levels): halves the per-cell data volume; an estimated −8% at small cells.
* **Level 1 derived straight from the rolled-up sums**, and **channel-pair features four
  cells per vector** (−6,000 scalar divide / square-root operations per frame).
* **Thread dispatch.** Two wake-ups of parked workers per frame plus a serial tail mean
  threads pay only from about a millisecond per frame.
* **NEON** has not been re-measured since the kernel rewrite.
