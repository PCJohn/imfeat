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
| sums | the 44 additive int64 quantities accumulated per cell per channel |
| slots | those sums laid out cell by cell, `[cell][channel][44]`: the form the pyramid levels and `compute()` use |
| fold | summing an accumulator's column lanes into per-cell sums |
| derive | turning a cell's sums into its float features |
| band | the contiguous range of cell rows one thread processes |
| Xeon, laptop | the two test machines: a 16-core 2.1 GHz Xeon (Linux, GCC) and a 22-thread laptop (Windows, MSVC) |
| development VM | the single-core shared cloud machine the work was done on; its timings wander by ±5% |
| gate's operating point | what framegate runs per frame: a 720p frame thumbnailed to 512×320 (`"pow2-fit"`) or a 1080p one to 1024×576, HSV, stride 1, six levels from 64×64, two threads |

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

"Now" also computes nine features that the original did not (and, until step 9, two
projection profiles and the channel-pair covariances).

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
2. **Kernel: columns as lanes.** One kernel (`block()`) takes a block of columns down all the
   sampled rows of a cell row and computes every per-pixel quantity in registers, in phases
   (step 10): bytes first (extrema and LBP, then the bar detector), then, a sub-chunk of rows
   at a time, each image row's three 1-D responses once into a small L1 buffer, the Laws
   energies and the tensor (which also write the HOG inputs), the Laplacian and the moments,
   and the HOG ray tests a quarter block at a time. Each phase carries its few accumulators in
   registers for the whole chunk; the 76 vectors of per-block sums are written once per chunk.
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
| 6 | BGR → HSV conversion fused into the row gather (details below) | development VM (AVX-512), 1024×1024×3 stride 1, one thread: `cvtColor` + pass 11.45 → 10.27 ms; the conversion costs 0.6 ms inside the pass, against `cvtColor`'s 1.4 ms plus a write and a read of the image. Its kernel went from 0.96 to 0.84 ns per pixel on the AVX-512 build and 1.8 to 1.2 on the AVX2 one between the first and the second version |
| 7 | The `INTER_AREA` thumbnail resize fused into the pass (details below) | development VM, 1920×1080×3 → 1024×1024, finest-64 stride 1: `cv2.resize` + pass 17.4 → 14.4 ms on one thread and 13.0 → 8.3 ms on two; the resize costs 3.9 ms inside the pass on one thread and 2.1 on two, against `cv2.resize`'s 13.5 ms on one thread and 6.6 on its two, plus a write and a read of the thumbnail. On its own the kernel takes 0.31× `cv2.resize`'s time per thread |
| 8 | The cross-channel products dropped: nothing read them (details below) | instructions per fused frame in the core, one thread, AVX2: 720p → 512×320 52.0 M → 50.0 M (−3.8%), 720p → 256×256 42.8 M → 41.2 M (−3.7%), 1080p → 1024×576 112.6 M → 109.1 M (−3.1%) |
| 9 | The projection profiles dropped, likewise | 50.0 M → 49.3 M (−1.4%), 41.2 M → 40.9 M (−0.7%), 109.1 M → 106.5 M (−2.4%) at the same three points; −5.2%, −4.4% and −5.4% against step 7 for the two together |
| 10 | The kernel rewritten as register-resident phases down the rows, with row responses cached and the HOG sums signed; then, from a callgrind profile, the per-block cell sums laid out kind-contiguous and the fold's whole-vector reductions through `ReduceSum` / `SumsOf8` (details below) | Ryzen 7 3700X (AVX2, GCC 11), one thread, `features()`, 4 levels: 1024×1024×3 finest-32 stride 1, 7.44 → 5.42 ms (1.37×); 1024×1024 (C=1), 2.50 → 1.85 (1.35×); 1024×1024×3 finest-64, 8.93 → 6.80 (1.31×); 512×512×3 finest-32, 2.22 → 1.71 (1.30×); 256×256×3 finest-32, 0.81 → 0.71 (1.15×); the fused BGR → HSV pass at 1024², finest-64, 9.85 → 7.74 (1.27×). Every output bit-identical |
| 12 | A batch of small items chosen by the per-instruction cost table of step 11 and kept or dropped by pinned A/B timing: plane rows aligned to cache lines (the left pad half a line, the stride a whole number of lines, the buffer 64-byte aligned); the 16-bit halves of a one-cell block widened by interleaving with zero instead of `pmovzx`; the bar detector's gate as a saturating subtract and min; the one-cell fold's sign extension through promotes; the moments' fourth-power pair sums widened every two rows; HOG sign masks from a compare instead of a shift | Pinned to a core, interleaved: 1024×1024×3 finest-32 stride 1, 5.38 → 4.84 ms (−10%); stride 2, 2.48 → 2.04 (−18%); 1024×1024 (C=1), 2.42 → 2.13 (−12%); finest-64, 6.81 → 6.45 (−5%); 512×512×3, 1.68 → 1.62 (−4%); 256×256×3, −1%. Every output bit-identical |
| 14 | Cells derived and rolled up in groups of four whatever the block width (two two-cell blocks or four one-cell blocks share one kind-major sums group), so that blocks of one or two cells no longer go to slots through the scalar flush and back through scalar gathers; the two phases of a raw row split as it is gathered | Instructions per frame: 1024×1024×3 finest-32 stride 1, 63.6 M → 52.5 M (−17%); 16 px cells, −3.6%; 8 px cells at stride 2, −2.2% (the gather). Wall clock, measured only on a machine at load 20–25 with A/B/B/A ordering: within ±2% of the build before at every configuration tried. Every output bit-identical |
| 20 | The summary of a multi-band frame spread over the threads without changing any fold's order: each level its own accumulators; from three bands, band 0 folds its share of level 0 in-band as it goes and a third job after the join folds level 0's remaining cells on the first thread and the coarser levels whole on the others (details below) | The gate's 1080p HSV frame, pinned to as many idle cores as threads, A/B/B/A: 3 threads 2.68 → 2.48 ms (−7.4%), 4 threads 2.14 → 1.98 (−7.5%); 2 threads and 1 thread on the old path, unchanged. Every output identical across 1 to 6 threads on the three builds |
| 21 | The summary after the join split over the threads by lane instead of by level: a run is one level's cells, one channel, a range of eight-feature groups (or its moments), folded on one thread in cell order, the runs dealt out by cumulative size; band 0's in-band share and the two-band serial fold go, so every multi-band frame takes the one path (details below) | The gate's 1080p HSV frame, pinned to as many idle cores as threads, A/B/B/A: 2 threads 2.83 → 2.77 and 2.78 → 2.71 ms (−2.1%, −2.4%); 3 and 4 threads within ±0.5% in two sessions each; 1 thread the same code. Every output identical across 1 to 6 threads on the three builds |
| 22 | Three items from a cycle profile per loop set against the instruction profile (where cycles outrun instructions, something stalls): the three-channel gather, where the frame is first read and its loads waited, asks for the lines 768 bytes on; the resize's finish converts without Highway's overflow fix-up (the sum is a weighted mean of bytes, so in range); a multi-cell block's byte and bar-map kinds decide the block width once outside the kinds loop instead of per kind | Alone, pinned A/B/B/A on the gate's 1080p frame: the prefetch −1.1%, −0.9%, −0.9%; the finish −0.5%, −0.4%; the kinds loop −0.6%, −1.2% (headline −0.7%). Together: the gate −0.0% and −0.4% (medians −0.2%, −1.0%), two threads −0.5%, the headline −0.7%; the module's instructions per two gate frames 142.7 M → 139.5 M (−2.2%). Bit-identical on the three builds |
| 23 | The finest level's records prefetched two groups ahead of the derive's stores, and the resize's horizontal pass testing its store bound once per group instead of once per plane store (details below) | Judged by cycles per loop, not the frame: the scatter loop 506 → 282 samples of 23,171 (−1.0% of the module), the prefetches free in the derive loop; the bound test within run noise, −1.5% of the module's instructions per two gate frames with it (139.9 M → 137.8 M). Wall clock, pinned A/B/B/A: the gate −2.4%, the headline −1.1%. Bit-identical on the three builds |
| 24 | Cells derived and rolled up in groups of two vectors (eight) instead of one, with the group's kind stride a whole cache line: the roll-up writes a whole vector of parents per kind and touches each kind's line of the level-1 row (larger than L1 at the gate's sizes) half as often (details below) | By loop cycles: the roll-up loops 553 → 370 samples of 22,472 (−0.8% of the module), the folds unchanged once the stride was even (at the odd stride of 9 they had gained 100); the module's instructions per two gate frames 137.8 M → 136.6 M (−0.9%). Wall clock, pinned A/B/B/A: the gate −1.7%, 16 px cells −1.7%, the headline −0.6%. Bit-identical on the three builds |
| 19 | The level-1 roll-up two kinds per even/odd lane add (a group's one vector of children, both kinds' parents in one vector), and the finest level's summary folded once per cell row instead of once per group of four cells, the same cells in the same order (details below) | The gate's pass (1024×576, six levels from 16×9 px cells), pinned A/B/B/A, 16 rounds: 4.09 → 3.87 ms (−5.4%); 16 px cells at 1024², −3.2%; the headline, −0.2% (its roll-up and summary are under 2% of it). Instructions per two gate frames 124.0 M → 117.6 M; the roll-up loops 6.2 M → 2.7 M, the summary's 6.3 M → 4.3 M. Bit-identical on the three builds |
| 18 | The resize's finish in one pass: with three resampled rows cached, a destination row's (up to three) tap rows are summed in tap order, rounded and stored as bytes in one sweep instead of a scaled copy, a read-modify-write pass per tap row and a rounding pass (details below, after step 17's) | 1080p → 1024×576, the resize 1.98 → 1.66 ms (−16.5%); 720p → 512×320, −4.6%; the gate's 1080p frame 6.59 → 6.16 ms (−6.6%). Bit-identical on the three builds |
| 17 | The fused INTER_AREA resize's horizontal pass over all of a row's planes at once: a group's tap patterns and weights loaded once and the loop's bookkeeping shared across the (up to four) planes, each plane's products and sums the same in the same order (details below, after step 16) | The gate's 1080p → 1024×576 HSV frame, pinned A/B/B/A, one thread: 7.18 → 6.84 ms (−4.7%); the resize alone 2.21 → 1.90 ms (−13.7%) at that size and 0.96 → 0.74 (−22.7%) at 720p → 512×320. Bit-identical on the three builds; the resize tests against cv2 pass |
| 16 | The fused HSV conversion's table step as one rounding: the float guess of OpenCV's table value round(K / x) is within 2⁻²³ of it and the quotient is at least 1 / (2x) from a half-integer, with 2K 2⁻²³ = 0.25, so the floor-and-remainder correction (six dependent ops per table) goes (details below, after step 15) | HSV frames, pinned A/B/B/A: 1024×1024×3 finest-32 stride 1, 5.05 → 4.77 ms (−5.4%); 512², −4.3%; 256², −3.9%. The conversion 0.86 → 0.61 ms. Exact on every one of the 2²⁴ colours against cv2 on the three builds |
| 15 | The HOG ray tests halved by a mirror symmetry of the rays (details below), then the energy phase at stride 1 as three chunk-wide product sums per response column with corrections once per chunk: a gradient is reflected into the first quadrant and tested against the four low rays only, with the energy summed by the test's sign and by that sign times the side of the vertical the gradient came from, which the flush turns back into the energy past all eight rays; the half-plane fold of the gradient in the energy phase becomes an `abs` and a `psign`; the bar detector's sums passed through its runs by value (GCC had stored them every row), its first lag's maxima taken as is, and its phase a function of its own so the row loop has the registers to itself | Pinned, A/B/B/A, against step 14, load 2.5: 1024×1024×3 finest-32 stride 1, 4.69 → 4.45 ms (−5.1%), then −2.3% more from the bar detector's own function and −1.9% from the energy phase's product sums; 16 px cells, −5.5%; 32 px cells stride 2, −6.1%; 8 px cells stride 2, −4.1%; 1024×1024 (C=1), −4.8%; 512×512×3, −4.1%; 256×256×3, −0.9%. Against the original build on the same quiet core, after all three: 7.51 → 4.28 ms at the first (1.75×), 128 px cells 7.34 → 3.91 (1.88×), C=1 2.67 → 1.55 (1.72×), 16 px cells 8.86 → 5.78 (1.53×), 512×512×3 2.23 → 1.41 (1.58×), 32 px cells stride 2 2.40 → 1.66 (1.45×), 256×256×3 0.81 → 0.61 (1.32×), 8 px cells stride 2 6.68 → 6.41 (1.04×); with the library's exact band threading, 1 / 2 / 4 threads at the first: 7.36 / 3.74 / 1.97 → 4.21 / 2.14 / 1.15 ms, so two threads of this build do the frame in 0.29× the original's single-thread time. Instructions per frame at the first, 58.4 M → 55.5 M (−5.0%); the kernel's multiplies per block row 78 → 62. Every output bit-identical on the three builds |
| 13 | The per-cell stages, from profiles of the small-cell and coarse-grid configurations: a batched fold for one-cell blocks (four kinds reduced together through two interleave levels into one four-lane store); the hash grid's per-row byte sums through `psadbw` where the finest grid does not tile the hash grid; batched folds for blocks of several cells (a cell's lanes added in 32 bits as 128-bit vectors, four cells reduced together) for the HOG and pair kinds; level 1 derived in its band from the rolled-up sums through `derive_block`, its slots written by a vector flush four cells at a time | Pinned, interleaved, against step 12: 1024×1024×3 finest-32 stride 1, 4.72 → 4.49 ms (−5%); 64 and 128 px cells at stride 1, −8%; 8 px cells at stride 2, −6% (folds) and −7.7% of the instructions (level 1), the time of the latter within noise with three channels and −4 to −6% with one channel or at 512²; 16 px cells at stride 2, −6%; 256×256×3, −8%. Every output bit-identical |
| 11 | The plane window made a sliding contiguous buffer (the halo copied forward when it fills, edge rows physically replicated) and the response slots affine, so every row and column tap in the kernel is a constant offset from one row pointer; one phase plane makes the column taps compile-time (`block<.., UNIT>`) | 1024×1024×3 finest-32 stride 1: 67.4 M → 61.8 M instructions per frame (−8%), wall clock unchanged within noise (the scalar work ran in parallel with the vector work, see the step); 8 px cells and 256² about −2%. Every output bit-identical |

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

**Step 6, the colour conversion.** A host with a BGR frame that wants HSV features used to
run `cv2.cvtColor` first: a pass that writes a second image and reads it back. The conversion
now happens where each row is de-interleaved, in every band's thread, so it costs no memory
traffic and threads like the rest. The output is exactly OpenCV's 8-bit integer formula
(`RGB2HSV_b`, hue in `[0, 180)`), checked against `cv2.cvtColor` on all 2^24 colours on the
AVX-512, AVX2, SSE4 and SSSE3 builds, with GCC and clang. What made it cheap is not gathering:
OpenCV's formula takes two per-pixel table values, `round(255·2¹²/v)` and `round(30·2¹²/diff)`,
and a 256-entry gather per lane is slow on every x86 generation. Instead:

* *One reciprocal per pixel serves both tables.* `1/(v·diff)` is one division; `255·2¹²·diff`
  and `30·2¹²·v` are exact in float (their odd parts have 16 and 12 bits), so a multiply each
  gives both quotients to within 2^-23, or an eighth of a unit at the largest table value.
* *An exact fix-up with one compare.* `floor(guess)` is the rounded quotient or one less, and
  the remainder `N − floor(guess)·x` decides: above `x/2` means one less. The remainder is an
  fma, exact because the difference is small although the product is not; without fma it is
  computed in 32-bit integers. Neither table has a half-way case (the numerators carry more
  factors of two than any divisor below 256), so "rounded" needs no tie rule. The first version
  divided twice per pixel and corrected in both directions; this one costs about two thirds
  of it.
* *No shuffles between lane widths.* Bytes go to 16-bit lanes and 16-bit to 32-bit as even and
  odd lanes (a mask or a shift) and come back with a shift and an or, so the shuffle port,
  the busiest one on Intel cores, is left to the 3-channel de-interleave. On this VM's AVX2
  build that measured 6% faster than promoting and demoting.
* *The rest is exact by inspection.* `v` and `diff` stay bytes and the signed hue numerator
  16-bit; each fixed-point product is below 2^21, so it and its `+ 2¹¹` are exact in float;
  the final `>> 12` is the integer shift, which floors like OpenCV's; and the H and S planes
  are only computed when a selected channel asks for them.

One thing to know about the division: under `-ffast-math`, GCC and clang both replace a vector
division by a hardware estimate plus one Newton step, whatever the enclosing function's own
options say (the `optimize` attribute does not reach through Highway's target pragmas). The
CMake flags turn that off (`-mno-recip`, `-fno-reciprocal-math`), which is worth a few
percent; the result would be exact either way, since that estimate is within the fix-up's
tolerance, and the all-colours test says so on every build.

**Step 7, the thumbnail resize.** The hosts thumbnail every frame with
`cv2.resize(INTER_AREA)` before the pass, and on a 1080p frame that costs more than the pass
itself: for any ratio that is not a whole number OpenCV takes its generic path
(`ResizeArea_Invoker` in `imgproc/src/resize.cpp`), a scalar loop over a per-tap index table
with float32 weights that resamples each source row once for every destination row it feeds,
threaded over rows on OpenCV's pool. Its arithmetic is nonetheless fully specified, so it can
be reproduced bit for bit: per destination column the taps come in table order (the partial
cell on the left, the whole cells, the partial cell on the right), each product and each
running sum rounded to float32 with no fused multiply-add (OpenCV compiles the file for its
SSE3 baseline); rows are scaled by their float32 weights and summed in order; the result is
rounded half to even and clamped. Whole-number ratios take a block-sum path (half up for 2×2
on 1, 3 or 4 channels, a float32 product otherwise) and the same size is a copy; an upscale
is not `INTER_AREA` in OpenCV at all (it turns bilinear) and is refused. A numpy model of the
generic path was written first and matched `cv2.resize` on every size tried, on the Linux
wheel and on the Windows one, before any C++.

The kernel (`area_row` and around it in `core.cpp`):

* *A source row is resampled once.* A row feeds at most the destination row it ends and the
  one it starts, in order, so a two-row cache halves the horizontal work against OpenCV.
* *The horizontal pass is a fixed shuffle.* The taps of G destination columns lie in one
  16-byte window of the source row (G = 8 for 1080p's ratio, 4 for 4K's), so a group is one
  window load and, per tap slot, one byte shuffle, one widen, one convert, one multiply and
  one add, with the slot's weights in a vector; a column with fewer taps than the slots gets a
  zero byte at weight 0, which adds +0.0 and changes nothing. Ratios so large that a single
  column's taps do not fit a window fall back to a scalar loop that sums in the same order.
* *The vertical pass* is a vector multiply then multiply-adds over the row, and the rounding
  is one `NearestInt`.
* *Per band, up front.* Each band makes its rows of the thumbnail (halo included) from the
  frame before its pass, in its own thread, and the row gather then reads them as planar
  rows; nothing leaves the band unless the caller asks for the thumbnail (`thumb_out`). Making
  each row just before the pass needed it, between cell rows, measured 1.5–2 ms slower on the
  development VM (5.9 against 4.3 ms inside the pass on one thread; 2.7 against 1.2 on two):
  the resize's working set and the kernel's evicted each other, and the frame's rows were no
  longer streamed. The thumbnail is 3 MB; a band's share of it stays in its cache.

Two compiler traps, both caught by the exactness tests, which include an odd size
(1917×1079) that they turn out to be sensitive to. Under `-ffast-math` GCC and clang fuse a
multiply and an add into one rounding, and clang also regroups a chain of sums: twelve pixels
of one frame came out one off with contraction on, one pixel with clang's reassociation on.
GCC honours the per-function `optimize` attributes the kernels carry (`IF_STRICT`), clang
decides by where the inlined vector ops were written, so clang builds turn contraction and
reassociation off for the whole file (`-ffp-contract=off -fno-associative-math`), which also
made its build pass the four thread tests it used to fail; MSVC gets `float_control(precise)`
and `fp_contract(off)` around the kernels.

*The size can follow the frame.* A fixed thumbnail upscales any frame smaller than it, and an
upscale is not `INTER_AREA`, so the hosts fell back to `cv2.resize` (bilinear) for those.
`thumb="pow2"` instead takes the largest power of two the shorter side holds, square: 720p
becomes 512 px, 1080p and 1440p 1024, 4K 2048. The rule is a bit length on the shape and runs
once, when the computer is built (the whole build is a millisecond or two), because every
plan in the computer — bands, halos, the resize taps, the scratch, the output block — is
sized from that one (frame, thumbnail) pair; a host that serves several frame shapes keeps one
computer per shape, as before, and `thumb_size(shape, "pow2")` tells it the size without
building. On the laptop (October 2026 run) a 720p frame costs 3.8 ms as a fused 512 px pass
on one thread, 2.2 on two and 1.4 on four, against 9.5, 5.1 and 3.9 for the bilinear upscale
to 1024 and the pass on it (the development VM: 4.7 and 3.4 against 10.7 and 5.7 on one and
two threads); a 4K frame goes the other way, 37.1 ms at 2048 against 15.4 at a fixed 1024 on
one thread (18.8 against 8.1 on two, 12.8 against 5.0 on four), since the pass is now four
times the pixels on top of the same read of the frame. The one-off build of a computer measured 0.3–2.5 ms on the laptop,
4–5 ms for the 4K-to-2048 one.

*Keeping the frame's shape.* Two more policies keep the aspect ratio, both from the same
square as `"pow2"`, and differ in which side becomes its power of two: `"pow2-cover"` the
shorter side (1080p → 1024×1792, the square covered), `"pow2-fit"` the longer (1080p →
1024×576, the shape fitted inside the square, so never more pixels than it). The other side
follows at the same scale, rounded to a multiple of 64 and never above the frame's own
(`THUMB_QUANTUM`), and that rounding is why neither keeps the shape exactly: a 16:9 frame is
exact at widths 1024 and 2048 (576 and 1152 rows), but at width 512 it wants 288 rows, no
multiple of 64, and gets 320 — a 1.6:1 picture, a 10% squash where the square's is 78%; a
side too thin for the grain becomes one quantum, so a banner ends up the square under
`"pow2-fit"`. The shape is kept as closely as the grid allows, no closer. (`"pow2-fit"` first
took the *longer* side's own power of two, 1024×576 for 720p as well: exact 16:9, but 2.25×
the square's pixels whenever the longer side sits an octave above the shorter, so it was the
faster rule at 1080p and the slower at 720p and 1440p; anchoring both aspect policies on the
same square made it cheaper than the square everywhere, at the cost of the rounding above.)
A policy can land on the frame's own size (1024×1920 under `"pow2-cover"`, a power-of-two
square under `"pow2-fit"`), and then nothing is resized: the pass reads the frame and
`thumb_out` gets a copy from Python, which spares the kernel's copy mode its memcpy of the
frame into every band's line buffer (6 MB at 1024×1920: one to three milliseconds on one
development-VM thread, within noise at two).

The two aspect policies differ in cost far more than in pixels, and the reason is the pass,
not the resize: its blocks span whole cells only when a cell is a power of two wide (step 4
above, with the sampling stride folded into column phases); any other width is walked one
masked block per cell, and a stride then leaves most of the block's lanes idle — a 28 px cell
at stride 4 uses 7 lanes of 32. `"pow2"` and, on a landscape frame, `"pow2-fit"` keep the
width a power of two; `"pow2-cover"` cannot, since the shorter side is the power of two and
the aspect ratio is not. On the development VM with two threads, a 1080p frame at stride 4
costs 3.7 ms as the 1024 px square, 2.5 ms as 1024×576 and 9.4 ms as 1792×1024 (at stride 1:
9.2, 6.1 and 15.2); a 720p frame at stride 2, 2.3 ms as the 512 px square, 2.2 ms as 512×320
and 6.8 ms as 896×512. So `"pow2-fit"` keeps the shape and costs at most what the square
does, on landscape frames; on portrait frames the two aspect policies swap roles.

**Steps 8 and 9, the outputs nothing read.** By October 2026 the pass had two consumers,
`fastdet` (which reads `maps`) and `framegate` (which reads `maps`, `summary` and `hashes`),
and a per-function profile of the fused pass at the gate's operating point (callgrind, AVX2
build, one thread, 720p → 512×320: 52.0 M instructions per frame in the core) put the frame
at: the thumbnail resize 29%, the pixel kernel 25%, the rest of the cell-row loop (the folds
and the level roll-up) 22%, the per-cell derivation 12%, the summary fold 6%, the HSV
conversion 3%, the row gather and band scheduling 4%. Two outputs ran through most of those
stages for no reader: the channel-pair products `Σ v_i v_j` (a `pmaddwd` per pair per half
block per row, folded, rolled up and summed into the global like every other sum, then six
divides and three square roots per cell) and the projection profiles (two vector adds per
half block per row, two load-add-stores per block, a horizontal reduction per sampled row).
Both were removed as pure deletions — [FEATURES.md §10](FEATURES.md) keeps their
definitions, the way the pass made them and what they cost, so they can be put back as a
revert — and every remaining output was checked byte for byte against the previous build over
four clips at six operating points (1–3 threads, strides 1, 2 and 4, four thumbnail rules;
20,400 and 18,900 arrays). The block loop still carries up to `GMAX` (8) channels through a block
together, which is what lets a block's cells be derived on the spot; it just no longer forms
the products on the way.

**Step 10, the kernel as phases (October 2026).** The brief was 2× at stride 1 with the output
unchanged; this step got 1.3–1.4× where the kernel dominates and is written up with the
reasons the rest is out of reach, so that nobody re-derives them. Measured on an AMD Ryzen 7
3700X (AVX2, 16 vector registers, GCC 11). `perf` is locked on that machine
(`perf_event_paranoid = 4`) and valgrind was not installed, so the first half of the work
went by *ablations* — the same build with one stage deleted, timed against the whole, the
difference being that stage's share — and by reading the generated code; the second half by
callgrind, after valgrind was built from source into the user's prefix (no root needed).

Where a stride-1 frame went before the change (1024×1024×3, finest-32, 7.4 ms): the pixel
kernel 80–86%, the fold 8–12%, everything else (row gather, derive, summary, hashes) under
10%. Inside the kernel: the HOG ray tests 27%, the bar detector 9%, the moments 7%, the Laws
energies 6%, and the rest — loads, promotions, the 1-D row responses, the `(qy, qx)`
interleave, and above all the accumulator traffic — about 40%. Deleting every accumulator
update (keeping the arithmetic) saved 22% of the frame: with 76 accumulators per block and 16
registers, every one of them was a load, an add and a store per row.

What was done, each part exact by construction (every sum is still an integer sum of the same
per-pixel integers; only the order of the additions changes):

* *Phases down the rows.* `kernel()` (one call per row per channel, everything at once) became
  `block()`: one call per block per channel per chunk of rows, which walks the rows several
  times, once per phase, each phase holding only its own accumulators — extrema (2), LBP (9),
  the bar detector (6), the energies and tensor (9 per half block), the Laplacian and moments
  (6), the HOG (9 per quarter block). Those stay in registers for the whole chunk; the Acc
  block is written once. The HOG phase compiles to 34 instructions per quarter block per row
  (8 multiply-adds, 8 sign flips, 8 adds, 2 loads, the |grad|⁴ term and the loop), the
  energy phase to 77 per half block per row, with no spills in either.
* *Row responses once per image row.* At stride 1 each image row is the row above, the centre
  and the row below of three consecutive sampled rows, and the old kernel promoted it to
  16 bits and formed its three 1-D responses three times. A sub-chunk of up to 32 sampled rows
  now computes `D = r − l`, `S = l + r − 2m`, `L = S + 4m` and `m` once per distinct row into
  an 8.7 KB L1 buffer (the distinct rows are listed per chunk by `cell_row()`, so replicated
  edge rows are computed once too), and the 16-bit phases load them. At strides above 1 there
  is no reuse and the cost is the same as before, just in a different place.
* *Signed HOG sums.* The cumulative energy past ray `j` was `Σ |grad|² · [t_j ≥ 0]`, a mask
  and an AND per ray (a compare, a min and an AND as GCC compiled it). Now the kernel sums
  `+|grad|²` where `t_j > 0` and `−|grad|²` where `t_j < 0` — one sign-flip instruction
  (`IfNegativeThenNegOrUndefIfZero`, `vpsignd` on x86) — and the flush recovers the
  cumulative sum as `(P_j + T) / 2` with `T = Sxx + Syy`, exact because `P_j + T` is even.
  This is exact only if no nonzero gradient lies *on* a ray (`t_j = 0`): the rays' reduced
  directions all have a component above 1020 (the smallest is 1283, so `(cx, cy)/gcd` is out
  of the gradient's range `|gx|, |gy| ≤ 4·255`), and `set_config()` checks that, so a retuned
  bin count or ray scale cannot silently break it. The `t = 0` case then only happens for a
  zero gradient, whose energy is 0 whatever the flip makes of it. On targets without a sign
  flip Highway's fallback costs about what the mask did.
* *The bar detector's gate as a multiply-add.* Its six gated maps were each `AND`ed with the
  gate and pair-summed with a generic widening that compiled to four instructions; the gate is
  now 0 / 1 bytes and each map pairs up through one `pmaddubsw` against it. The block- and
  row-constant branches (`valid` for a block at a row end, the lags whose rows are inside the
  image) are resolved outside the row loop: the rows with every lag in the image are one run,
  walked without a test per lag.
* *Taps as offsets.* The per-row, per-channel table of 21 tap pointers became three row
  offsets and six up/down offsets per row, shared by the channels, with the column taps as
  loop-invariant registers; the energy phase's row-slot indices are precomputed as element
  offsets. Ten to twenty scalar instructions per row of address arithmetic went away.
* *`Σ w⁴` widened unsigned* (its pair sums cannot be negative): three instructions instead of
  ten. *Small trims*: the row responses `S` and `L` as `(l + r) ∓ 2m`, one op fewer; the
  half-plane sign of the gradient from the sign bit of `gy | (gx & [gy = 0])`, one op fewer.

The frame after that (1024×1024×3, finest-32, stride 1, 5.8 ms): the kernel 80%, of which
the HOG tests 22%, the energies and tensor 15%, the bar detector 16%, LBP and extrema 12%,
the row responses 8%, the Laplacian and moments 6%; the fold 12%.

*With a profiler.* Callgrind on one such frame (per-instruction costs attributed to the
innermost loop, with cache and branch simulation) counted 74 M instructions in the module:
the energy phase 20%, the HOG phase 18%, LBP 11.5%, the bar detector 10%, the Laplacian and
moments 9%, the row responses 5.5%, extrema 5%, the fold under 5%. The kernel therefore
executes about 600 instructions per block row (32 columns, one channel) at 3.2 per cycle,
some 410 of them vector arithmetic, and it is at the floor of this formulation: one
multiply-add per energy and per ray test, and each of the eight ray tests needs its own
sign-and-add into its own accumulator. What the profile did show was the cache: 1.04 M L1
data misses per frame, 586 K of them on stores, where the planes' first touch explains
250 K. The rest came from the per-block cell sums: with one cell per block (32 px cells at
stride 1) each block wrote its 39 kinds at a 33-cell stride, 117 cache lines for 936 bytes,
three channels' worth, which evicted the row-response and HOG-input buffers between phases
(a fifth of their lines missed on every block). Three changes followed, each exact:

* the one-cell-per-block case writes its kinds contiguously (`flush()` takes a kind and a
  cell stride), five lines per block; each channel's block is folded as soon as it is done,
  so one set of accumulators serves all channels (7.5 → 2.5 KB); the HOG phase recomputes
  `|grad|²` from `(qy, qx)` instead of loading it, which halves that buffer. Misses per
  frame 1.04 M → 0.41 M, the stride-1 frame −4 to −5%;
* a vector whose lanes all belong to one cell is reduced with `ReduceSum` instead of a
  `ConcatEven` / `ConcatOdd` level per halving — every accumulator at 16 px cells and up, the
  HOG and `|grad|⁴` ones at 8 px: −6% at finest-64, −4.5% at finest-32;
* the byte-count kinds (extrema, LBP) and the bar maps sum to 64-bit lanes of eight columns
  in one step (`SumsOf8`, a `psadbw`; `SumsOf4` for the 16-bit maps) wherever a cell is eight
  columns or wider, instead of a pair sum, two promotions and two folds: another −1.5 to −2%.

A third round, measured by callgrind instruction counts since the machine was by then
shared with a training job and wall-clock noise exceeded the effects: the extrema sweep
folded into the LBP ring (a strict local maximum is exactly LBP bin 0 — popcount 0 is
uniform — and the minimum rides on the ring's loads as a running min: one sweep and nine
loads per row fewer); the LBP bins counted in nibbles, five byte accumulators each holding
two bins, voted by one byte shuffle of the bin index and emptied into the byte counts every
15 rows (no compare and subtract per bin, and four accumulators fewer); the Sobel and Laws
arguments sharing two sums (`(Da + De) ± 2Db`, likewise for S); and the HOG `(qy, qx)`
interleave in block order where a cell is at least half a block wide, since the fold then
sums every lane of a quarter anyway (the loop is instantiated per case, forced inline). A
1024×1024×3 stride-1 frame went from 73.4 M to 67.4 M instructions (−8%): the energy phase
is 68 instructions per half block per row, the HOG 34 per quarter, LBP with extrema 89, the
bar detector 86, the Laplacian and moments 33, the row responses 39.

Two smaller items came out of the same profiles: the per-chunk row tables were built with a
linear search per row (1.3% of the instructions at 32 px cells, 2.6% at 4 px); a window-slot
lookup replaced it. And the two linear sums of the Laplacian phase (`Σ lap`, `Σ w`) ride in
16-bit lanes for a sub-chunk (`|lap| ≤ 1020` over at most 32 rows fits int16) and pair up
once, two instructions a row fewer. Tried and rejected on measurement: a software prefetch of
the next block's rows (−1 to −3%: the hardware prefetcher already follows the row stride) and
unrolling the HOG and energy loops by two (no change: the loops are throughput-bound, not
latency-bound). Branch simulation put the mispredictions at one per loop exit (about 30
loops per block, 85 K a frame, 5–7% of the cycles if the real predictor did as badly); the
loops cannot be fused without spilling, and whether the hardware predicts a 32-iteration exit
was not measurable here. At stride 2 on 8 px cells (the hosts' operating point) the profile
is 60% per-cell stages — derive 17%, fold 13%, summary 8%, roll-up 4%, the level-1 flush
4% — each within a few instructions of its arithmetic floor (the derive is two instructions
per feature, the summary a min, max, sum and sum of squares in double per value), so that
configuration gained little. Where a 2× would have to come from, and why it was not taken:

* **Fewer ray tests.** Mirroring the gradient about 90° halves the multiply-adds (four rays)
  but not the masked adds (each test now feeds two accumulators, one per half-plane), and
  needs a ninth accumulator; on paper it saves 4 of 35 instructions per quarter block per
  row, about 2% of the frame. A coarse 16-bit test with an exact fallback was costed and
  rejected: with the rays at 2¹⁴ the rounding leaves a test ambiguous whenever
  `|grad| · sin(angle to the ray) < 4`, which at typical gradients is a few percent of the
  pixels per ray, so almost every vector of 16 pixels would take the fallback. Exact 16-bit
  comparison of the products' halves costs more than the 32-bit test it would replace.
* **The fold.** What is left of it is the partial reductions (several cells per vector, at
  8 px cells and below). A rewrite that reduces four accumulators at once through a
  transpose-add network would roughly halve that, but its lane bookkeeping is specific to
  256-bit vectors (two 128-bit blocks); the portable reductions (`ReduceSum`, `SumsOf8`, and
  `ConcatEven`/`ConcatOdd` per level) are what Highway gives. Not done.
* **The window ring.** The bar detector's row loop carries about 20 scalar instructions of
  pointer arithmetic because the rolling window wraps, so each row's up/down taps need their
  own offsets (and the 3×3 rows three). Done in step 11: the instructions went, the time did
  not, because they had run on the integer pipes beside the vector work.
* **Small cells** are dominated by the per-cell stages: at 4 px cells the derive is 18% of
  the instructions, the summary fold 9%, the fold 14%, the level roll-up 5%, each near its
  arithmetic floor (the summary is a min, max, sum and sum of squares in double per value).
* **What was tried and reverted:** folding the extrema into the bar detector's row sweep (one
  sweep fewer, but the registers ran out: +2%); 64-byte alignment of the response buffers (no
  measurable change, kept since it is free); the extrema and LBP in one sweep (spills: split).
* **Small cells regress slightly** (256×256×3 finest-64, 4 px cells at stride 1: −3%;
  256×256×3 finest-32 at stride 2: −1%): a chunk is then 4 rows, and a dozen phase loops with
  their prologues cost more than the register residency saves. Those configurations are
  dominated by the per-cell stages anyway.

**Step 11, the sliding window (October 2026).** The rest of the step 10 profile was
bookkeeping: the energy loop carried 15 scalar instructions a row for its row-slot lookup
(three index loads, six `lea`), the Laplacian loop 10, the bar detector 21 and the LBP 10 for
the ring offsets, because the plane window was a ring of rows (image row y in slot
`y % slots`) and so every row's above, below and lag rows had their own table entry.

* *Affine response slots.* The response buffer is private to the chunk, so the slots were
  simply laid out as `a · (q − qs) + role`, `a = min(stride, 3)`: at stride 1 a row's above
  row is the centre of the one before and at stride 2 its below row, and the shared slot is
  written twice with the same offset. The energy and Laplacian loops then walk one pointer
  with constant displacements: 68 → 55 and 33 → 27 instructions a row.
* *A contiguous window.* The ring became a buffer of `max(cell rows, 32) + 2·MAXLAG` rows
  holding image rows `[wbase, filled)` in order at slots from 0; when a cell row's rows would
  run past the end, the rows it keeps (the halo the two cell rows share, eight rows) are
  moved to the front with one `memmove` per channel, a quarter of a percent of the gather.
  Rows above and below the image are gathered as copies of the edge row (the gather clamps
  the row index), so no tap is clamped either: the centre row of sampled row q is
  `row0 + q · stride · rs`, its 3×3 rows `± rs`, a lag's rows `± lag · rs`. The `r3`, `ud`,
  `r3slot` and `seen` tables went away.
* *Compile-time column taps.* With one phase plane the column taps are −1, 0, 1 and ± the
  lags, and the bar detector's lags are 1, 2, 4: `block<MASKED, UNIT>` hard-codes them, so a
  lag's four taps are `[rb ± lag]` and `[m + rs · lag]` with x86's scale factors and no
  register per tap. That mattered more than expected: with the taps runtime, GCC 11 ran out
  of general registers in the bar detector's row loop, spilled eight tap offsets and, oddly,
  stored each neighbour load to the stack and reloaded it twice (93 instructions a row);
  clang kept the vectors but spent 31 scalar instructions a row on the same addresses. With
  the constants the loop is 79 instructions, the LBP 80 (from 89), the row responses 36.

Per frame at 1024×1024×3, finest-32, stride 1: 67.4 M → 61.8 M instructions, bit-identical
on the GCC/AVX2, clang and SSE4-only builds (40,164 arrays each). The wall clock did not move
(5.26 ms before and after, interleaved; 8 px cells and 256² about −2%): the scalar
instructions had been executing on the integer pipes alongside the vector work, for free.
What bounds the kernel is the vector pipes, and a microbenchmark of this core (ten
independent register chains per instruction, inline assembly, pinned) says how: logic ops
issue 4 a cycle, adds, min/max and compares 3, saturating subtracts, `psign`, `pshufb` and
`punpck` 2, and every multiply (`pmaddwd`, `pmaddubsw`, `pmuldq`, `pmullw`, `psadbw`), the
shifts and `pblendvb` 1 — one pipe for all of them. A block row issues about 78 multiplies
(the HOG's 36 ray and energy tests, the energy phase's 18, the moments' 10, the bar
detector's 6, `|grad|⁴`'s 8) and some 20 other single-pipe ops: about 95 cycles of that pipe
out of the roughly 220 a block row takes, with the HOG loop within 20% of its own floor
there (44 cycles of multiplies per block row against 59 measured by ablation). The other
~420 vector ops a block row need at least 105 cycles over four pipes even without the
restrictions. So the remaining slack is scheduling, not instruction count, and a 2× at one
thread is not available from this instruction mix: it would need ~230 vector ops per block
row, i.e. an algorithmic change of the features. What would still pay, in cycles of the
multiply pipe rather than instructions: `|grad|⁴` in double (two converts and two FMAs, exact
below 2⁵³, on the FMA pipes instead of two `pmuldq`), about 4% of that pipe; the widening of
the row responses through `punpcklbw` (two pipes) instead of `pmovzxbw` (one), 6 a row,
which changes the lane order and so the fold's cell mapping for cells under 16 px. Neither
measurable on this shared machine, so not done.

**Step 12, a batch of small items (October 2026).** The question was whether many changes,
each too small to measure, add up. They do, but only the ones that touch what the core is
actually short of, and the way to tell is the per-instruction cost table of step 11, not the
instruction count. Each item was built as its own variant on top of the batch so far (a
script that reverts the others by text), checked bit-exact, and A/B timed pinned to the
idlest core pair (eight interleaved rounds, the minimum), which brought the noise under
±0.5% and made 1% items visible.

* *Cache-line alignment of the plane rows* (left pad 32 bytes, a phase plane's stride a
  multiple of 64, the buffer 64-byte aligned): −3.2%. Half of every centre-row load, lag-row
  load and response-build load had straddled a line and cost two load slots. The column taps
  one to four pixels off still straddle (a 40-byte span cannot sit in one 64-byte line on
  both halves of it); about seven a row remain, inherent.
* *In-lane widening for one-cell blocks*: −4.0 to −4.5%, the largest item. The row
  responses widened bytes to 16 bits by `pmovzx`, which on this core issues at 0.64 a cycle
  (two µops on the shuffle pipes), and the upper half needed a 128-bit extract (one pipe,
  one a cycle): about twelve cycles per image row. Interleaving with zero is one two-pipe
  shuffle per half. It changes which pixel lands in which 16-bit lane, which is nobody's
  business when the whole block is one cell (every fold is a full reduction), so it is used
  exactly then (`Chunk::inlane`); blocks of several cells keep the promotions. The masks for
  the masked variant widen the same way.
* *The bar detector's gate as `Min(SaturatedSub(peak, TAU − 1), mesh)`*: −0.7% alone, one
  µop and one register fewer. *The one-cell fold's sign extension through
  `PromoteLowerTo`/`PromoteUpperTo` and `ReduceSum`* (two shuffles and a reduction instead of
  Highway's `SumsOf2` of ten shifts and compares): −0.45% alone. Together with the alignment
  the three measured −2.2 to −2.5% against alignment alone, more than their sum.
* *The moments' fourth-power pair sums widened every two rows* (two rows' pair sums add in 32
  bits, one `SumsOf2` instead of two): +0.5% on its own on top of alignment, −0.5 to −1.1% on
  top of the full batch. Kept. *HOG sign masks by a compare instead of an arithmetic shift*
  (`vpcmpgtw` on three pipes, `vpsraw` on one): ±0.3%, kept as free.
* *Dropped on measurement:* the LBP bin as a max instead of `vpblendvb` (+0.6% alone, −0.4%
  on the batch: noise either way, and one µop more); `|grad|⁴` accumulated in double on the
  FMA pipes (exact below 2⁵³; the two `pmuldq` became two converts, an extract and two FMAs,
  the loop stayed 34 instructions with no spill, and it measured +0.5%: the converts and the
  extract share one pipe, and the FMAs half-share the multiply pipe); a software prefetch of
  the next block-channel's rows from the energy loop (+0.7 to +1.3%, as the step 10 attempt
  had found: the stride prefetcher already has them).

Why the ALU trims did so little and the memory ones so much came out of a microbenchmark of
the HOG inner loop alone (its 32 rows in L1, the real rays): 16.2–16.9 cycles per quarter
row for 34 µops, against a 9-cycle floor on the multiply pipe. Two rows an iteration, two
passes of four rays, and a compare-and-mask in place of `psign` all land at 16 to 18; the
same loop with its multiplies replaced by adds runs at 11, and bare multiply-and-accumulate
(9 `pmaddwd`, 8 adds, nothing else) at 12.5. Smaller probes put the practical ceiling at
2.5–3 vector µops a cycle in any accumulate loop here, not 4, with each `pmaddwd` costing
1.15–1.5 cycles in context (the one skeleton that does better shares each ray load between
two rows, 9.2 against 10.6 cycles, and in the full loop that gain goes to register
pressure). So one ALU µop is worth 0.3–0.5 cycles of a 200-cycle block row, a line-split
load about one, a `pmovzx` 1.5 and an extract 1: that is the table to cost the next item
against. With it, a block row is about 420 vector µops at ~2.5 a cycle plus the loop exits,
which is the time measured; the kernel is at the practical floor of this instruction mix,
and the whole frame at 1024×1024×3, finest-32, stride 1 is 1.60× the starting point
(7.74 → 4.84 ms pinned; 7.34 → 4.6 expected on a quiet core).

**Step 13, the per-cell stages (October 2026).** The small-cell configurations had gained
little: at 8 px cells and stride 2 (the hosts' operating point; at 1024² that is a grid of
128 cells a side) a 1024×1024×3 frame is three-quarters per-cell work, and a profile put the
summary fold at 11%, the level-0 derive at 12%, the multi-cell folds at 9.5%, the level-1
flush and its scalar re-gather at 6%, the roll-up at 4%, the feature transposes at 4%, the
row gather and phase split at 5%, the six kernel loops at 25%. (A note on the benchmark
names: the grid parameter is log2 of the cells per side, so at 1024² "k5" is 32 px cells,
"k7" 8 px cells and "k3" 128 px cells; the log's "finest-32" is 32 cells a side.)

* *A batched fold for one-cell blocks.* The 76 accumulators of a 32 px block were each
  reduced horizontally, an extract and shuffles apiece, and stored one at a time. Now the
  halves and quarters of a kind add in registers (any two lanes of an accumulator sum within
  32 bits at KROWS = 128, the four HOG quarters within 2³⁰), widen once, and four kinds
  reduce together: two interleave levels pair the lanes of four vectors into one vector of
  their four sums, stored to four consecutive slots of the kind-contiguous layout. All
  portable Highway ops (`InterleaveLower/Upper`, `ConcatLowerLower/UpperUpper`), so the
  AVX2-specific version the open ideas had in mind was not needed; other lane widths than
  four int64 lanes (eight reduce to four first) take a per-kind reduction. −5% at 1024²,
  32 px cells, stride 1; −2.6% at 256².
* *The hash grid's row sums.* Where the finest grid does not tile the 32×32 hash grid, the
  hash means come from a per-row byte sum that GCC had vectorized by widening every byte to
  int64. A `psadbw` per vector of the run: −8% at 1024² with 64 or 128 px cells.
* *Batched folds for blocks of several cells.* For 4, 8 or 16 lanes a cell, a cell's lanes
  add in 32 bits as 128-bit vectors (a HOG lane is below 2²⁸; the unsigned pair kinds' four
  lanes reach 2³² − 2²⁵, which unsigned arithmetic holds exactly), four cells reduce through
  the same two interleave levels, widen and store to four consecutive cells. The HOG kinds:
  −4.8% at 8 px cells stride 2, −3% at 16 px stride 2, −3.3% at 256²; the fourteen pair
  kinds: another −1.5% and −2.5%. The first version of the pair fold reduced the lanes of the
  4-lane case once too often and the harness caught it in one array of 40,164; the second
  stored four cells' sums into cell 0 and the harness's count line, not its tail, said so.
  Read the whole compare output.
* *Level 1 derived in its band.* The level-1 row in flight is rolled up as kernel sums,
  flushed to slots one scalar slot at a time (225 instructions a cell-channel) and, after
  the join, derived again from those slots through scalar gathers and converts. When every
  finest row of a level-1 row is the band's, the row now derives at once from its sums
  through `derive_block`, the one compiled body level 0 uses (its double re-centring of the
  moments is exact below 2²⁰ samples, which gates it); the slots, which the coarser levels'
  roll-up still needs, are written by `flush4`: each slot as a vector of four cells, the
  binomial expansion of the power sums as shifts and adds (KBIAS is 2⁷), and the slots
  transposed four at a time into the records. Rows straddling two bands keep the slot path.
  −7.7% of the instructions at 8 px cells stride 2; in time −3.7% with one channel, −6.3% at
  512² with 16 px cells, −2.6% at stride 1, and within noise with three channels at 1024²,
  where the frame writes 16 MB of features.

That last case was looked at: a write microbenchmark puts 16 MB of regular stores at
0.83 ms on this machine and streaming stores at 0.66 ms, so streaming the feature output
would save at most 2–3% of that frame and would slow the consumer's first read of it, while
for a 2 MB output (the usual sizes) regular stores are 2.6× faster than streaming ones. Not
done. An ablation of the feature stores was attempted twice and gave nothing usable: a tiny
scratch target serializes the stores on the same lines, and the second run coincided with a
rebuild on the same machine. Then the other session's training jobs filled every core (load
average 42), which ended wall-clock measurement for the day; everything after is checked
bit-exact and counted, not timed.

**Step 14, cell groups, and measuring on a busy machine (October 2026).** Blocks of one cell
(32 px cells and up at stride 1) or two (16 px at stride 1, 32 px at stride 2) had never
derived in `cell_row()`: their cells went to slots through the scalar flush (225 instructions
a cell-channel) and, after the row, through `derive_run`'s scalar gathers and `roll_row`.
A first attempt simply let every block derive and roll up alone, and cost 21 M instructions
more per frame at 16 px cells: the generic roll-up of a two-cell block is 3.7 K instructions,
and a half-empty derive vector is still a whole derive. The version kept groups cells for
the per-block stages: a block's, or as many blocks' as fill derive's four lanes, sharing one
kind-major sums group (`nxs = group | 1`; the one-cell fold stores its four-kind batches a
lane at a time into it), derived through `derive_block` after the group's last block and
rolled up through the vector pair path. At the headline configuration that removed the
whole per-cell slot path: 63.6 M → 52.5 M instructions a frame, the six kernel loops now 82%
of what is left, the folds 4%, the gather 2%.

The wall clock could only be measured on a machine running someone else's training jobs on
every core (load average 20–25), and that produced two findings about measurement worth
more than the item. First, with two builds interleaved pinned to one core, whichever runs
second in a round measures 2–3.5% slower at the minimum, consistently, even for two copies
of the same build; the earlier pinned A/B (step 12 onwards) ran at load 1–5, where this did
not show. An A/B/B/A order within each round cancels it (`pcmp3.sh`). Second, at that load
the median of the per-run minima swings by ±25% between two identical builds, so only the
minimum carries information, and it is good to about ±2%. With the order cancelled, the
grouped build is within ±2% of the build before it at 32 px stride 1 (+1.5%), 16 px stride 1
(−1.8%), 8 px stride 2 (−1.0%) and 32 px stride 2 (−0.4%): time-neutral under that
contention, for 17% fewer instructions. On the quiet machine of the morning the same removal
of per-cell work had measured as the step-13 numbers; which of the two a deployment sees
depends on what shares its cores, and the instruction count is the one number that does not
depend on that.

Tried after step 14 and reverted, by instruction count on the symbol build: for one-cell
blocks, the two halves of the energy and Laplacian phases and the four quarters of the HOG
sharing one set of accumulators in one loop each (every lane is the cell's; any two lanes
sum within 32 bits, four HOG quarters within 2³⁰), to save five loop exits per sub-chunk and
the accumulators' round trips through `Acc`. Bit-exact, but GCC interleaves the unrolled
parts and spills: the HOG loop went from 4 × 34 instructions a row to 174 with 43 spill
moves, the Laplacian from 92 per two rows to 101 with 22, the energies from 110 to 114; a
rolled inner loop over the quarters (`_Pragma("GCC unroll 1")`) has no spills but 42
instructions a quarter row for its addressing. The separate loops stay. Also measured and
not taken: clang 14 on the same source (equal at 32 px cells, +15% at 8 px cells, stride 2).


**Step 15, the HOG tests halved by symmetry (October 2026).** The hot loop since step 10 has
been the HOG ray tests: 35 instructions a quarter block per row, of which 11 on the one
multiply pipe (eight ray tests, `|grad|²`, two for `|grad|⁴`), and a microbenchmark of that
loop alone runs it at 16.7 cycles a row, well above the 12 or so its 35 ops would take on
the pipes they could use: the multiplies bind. The eight bin-boundary rays come in mirror
pairs, ray `HB − i` being ray `i` with the cosine negated (the table is built that way now,
and was bit-identical before), and reflecting a gradient across the vertical maps "past ray
`HB − i`" onto "short of ray `i`" exactly: `t_{HB−i}(q) = −t_i(q')` for the reflection `q'`.
So the kernel tests every gradient's reflection into the first quadrant — simply
`(|gx|, |gy|)`, one `pabsw` — against the four low rays only, and keeps three signed sums
instead of eight: per low ray `S_i = Σ sgn(t_i) |grad|²` and `D_i = Σ s · sgn(t_i) |grad|²`,
plus once `E = Σ s |grad|²`, where `s` is the side of the vertical the folded gradient lies
on. The side rides in the sign of the stored `y` lane (`gy · sgn(gx)`, with `sgn(0) = 1`), so
`s · |grad|²` is one `psignd` by the loaded pair, and the half-plane fold in the energy
phase, previously seven ops (compare, and, or, compare, xor, subtract, abs), is an `abs`, an
`or` and a `psign`. With the total `T = Σ_xx + Σ_yy`, the energy past ray `i` is
`(S_i + D_i + 3T − E) / 4` and past ray `HB − i` is `(T − E − S_i + D_i) / 4`, both multiples
of four computed in the flush (`hog_cum4`); the sums are one kind more (`K_NHOG = 9`), and
every fold and roll-up takes that in its stride. The loop is now 33 instructions with 7
multiplies, 14.4 cycles a row in the microbenchmark (−14%), and the energy phase 56 → 52.
Instructions per frame 58.4 M → 56.9 M; pinned A/B/B/A on a quiet core, −4.7% at the
headline configuration and −4 to −5% wherever the kernel dominates. The first version of the
flush recomputed the shared terms for every ray and cost +1.7% at 8 px cells, stride 2, where
the per-cell stages dominate; computing each mirror pair once made that −4.1%.

Reading the loops' assembly for the first time since step 11 (a script that finds the back
edges of the kernel and counts stack stores and register copies per loop) then found that
GCC stored the bar detector's six accumulators to the stack every row and copied them back:
12 of the loop's 76 instructions. They were captured by reference in a lambda called for the
three runs of a chunk, which gives them a home in memory that the `may_alias` byte loads
could alias. Passed through the runs by value in a small struct, they stay in registers;
the first lag's `Max(0, ·)` pair, which GCC cannot fold for unsigned bytes, went with it:
76 → 70 instructions a row, 55.5 M a frame. GCC still spilled the lag-2 and lag-4 row loads
of that loop and reloaded their offsets from the stack (about 8 instructions a row): the
loop shared block()'s registers with the live values of the phases around it. Moved into a
function of its own (`bard_phase`, one call per block and chunk — a call per run, with the
192-byte sums passed in and out through memory three times, cost 1 M instructions a frame
and +4% at 16 px cells and 512²), the loop is 60 instructions with no stack reference;
block()'s other loops each grew by one or two as their invariants moved register, so the
frame stays at 55.5 M, but the time is −2.3% at the headline configuration and neutral
elsewhere. The LBP loop carries five register copies and five table reloads a row — 12% of
its 80 — that the register-to-register moves of this core likely eliminate at rename; not
measurable (the LBP phase as a function of its own, and GCC's `no-tree-reassoc` and
`schedule-insns` + `sched-pressure` on it, changed nothing). The kernel's multiplies are 62
a block row now (the HOG's 28, the energies' 18, the moments' 10, the bar detector's 6), 20%
of a block row's cycles instead of 40%, and the µop count is what binds again: at about 530
vector µops per block row and the three a cycle a real loop reaches, the ~190 cycles a block
row measured are within 10% of the floor.

The energy phase then gave up its fourteen vertical-filter sums a row. At stride 1 the
response slots are consecutive image rows, so the energy of each 3-tap vertical filter over
a chunk is a combination of three sums per response column X over its rows r: the squares
P = Σ X(r)², the lag-one products Q = Σ X(r) X(r+1) and the lag-two R = Σ X(r) X(r+2), with
the products of the chunk's first two rows and of the two past its last as corrections
(with B = 2P − P₀ − P₁ + Pₙ + Pₙ₊₁, M = 4 (P − P₀ + Pₙ) and Qc = 2Q − Q₀ + Qₙ, the E, S and L
filters give B − 2R, B + M + 2R − 4Qc and B + M + 2R + 4Qc). Nine products a row instead of
nine squares of fourteen sums, the corrections once per chunk; every sum wraps in 32 bits
and every result is in range, so it is exact, and the stride-2 path (whose slots share rows
differently) keeps the old loop. The loop went from 52 instructions a half row to 43 — after
the same lesson as the bar detector's: the ten sums captured by reference in the row lambda
were stored and copied back every row (63 instructions, +4% in time) until they went through
the loop by value. Instructions per frame 55.5 M → 54.6 M; pinned A/B/B/A, −1.9% at the
headline configuration and with one channel, neutral at 16 px cells (16-row chunks, where
the corrections eat the saving) and at 512². Bit-identical on the three builds.

Cycles against instructions, per loop, from a `perf record` of the headline frame on the
symbol build attributed to the same back edges as the callgrind counts: the HOG loop 28.9%
of the module's cycles for 24.5% of its instructions, LBP 15.5% for 14.4%, the energies 13.1%
for 15.5%, the bar detector 9.1% for 10.1%, the moments 8.5% for 8.3%, the responses 5.4%
for 6.3%. So every loop runs between 2.6 and 3.4 instructions a cycle and the HOG loop, at
about 12 cycles a quarter row in context, is the slowest of them. Alone in a microbenchmark
it takes 14.5 cycles a row and two rows an iteration bring it to 12.8 (11.3 with the
`|grad|⁴` products moved out), so both were tried in the kernel: the two-row loop as the
microbenchmark had it (74 instructions a pair, four spills beside the eight ray operands)
measured +0.1% at the headline configuration and +1.5–3% elsewhere; with `|grad|⁴` summed in
the energy loop instead (same lanes, same values; the HOG loop 63 a pair) +3.3% and +5.4% at
16 px cells. In context the loop already overlaps its rows across the phase boundary, and
the unrolled forms only add instructions. Reverted.

Harvested after that and reverted, each timed pinned A/B/B/A against the build before it:

* **A transposing fold for cell groups.** One-cell blocks in groups of four fold their
  four-kind batches a lane at a time into the kind-major group (a stack round trip and
  four scalar stores per batch). Folded kind-contiguous into a temp and transposed into the
  group four kinds at a time at the group's last block: 54.57 M → 54.03 M instructions
  (−1.0%), bit-exact — and +0.9% at the headline configuration, +1.1% with one channel,
  neutral at 128 px cells, over 16 rounds. The same with the group's kinds 8 cells apart so
  that every transposed store stays inside one cache line (the forwarding to
  `derive_block`'s loads of them was the suspect): +1.2% and +1.8%. The scalar stores spread
  over four blocks' folds and overlap the kernel; the transpose is a burst right before the
  loads that need it. The instruction count is not the time here.
* **The moments' fourth-power pair sums widened every four rows** instead of two (four
  still fit unsigned 32 bits): the loop compiles no shorter (93 instructions per four rows
  against 46 per two), 54.03 M → 54.18 M, and +6% at 16 px cells. Reverted.
* The per-frame Python work is nil (cProfile: all of a frame's time is the native call), the
  trig and division samples in `perf` are the test harness making its frame once.
* **A profile-guided build** (GCC `-fprofile-generate`, trained on six configurations,
  `-fprofile-use -fprofile-correction`): −0.5% at the headline configuration, −2% at 16 px
  cells and 256², +9.6% at 8 px cells stride 2, which was not in the training set. Not
  adopted: a build tuned to a workload, and not even a clear win on it.
* **The plane window holding two cell rows** (`wrows_` doubled), so the halo slides every
  other cell row: the `memmove` is 0.6% of the frame's cycles, the change ±0.1%. Reverted.
* **`-O2` instead of `-O3`, and `-falign-loops=32`**: +0.4% and +0.9% at the headline, within
  noise at 8 px cells stride 2. The flags stay.
* **The memory side, counted** (`perf stat`, native events, per headline frame): 20.7 M
  data-cache accesses, 426 K fills from L2 (46 a block and chunk: the plane rows' first touch,
  half a line a block over 40 rows), 13 K lines from beyond L2, 11 K DTLB misses, 16 K branch
  misses (about 1% of the time). Nothing to fix there; the generic `L1-dcache-load-misses`
  event on this CPU lands evenly on compute instructions and does not mean what its name says.
* **`|grad|⁴` through FMA in doubles** (two promotes and two FMAs a quarter row, exact below
  2⁵³, converted back once a chunk) instead of `pmuldq` pairs on the multiply pipe: the same
  instruction count, bit-exact, +0.3% at the headline, +2.3% at 16 px cells, +2.7% at 8 px
  cells stride 2 (two more live registers, two conversions a chunk). Reverted. The multiply
  pipe is not what the HOG loop waits for.

**Step 16, the fused HSV conversion (October 2026).** The headline benchmark runs with
`feature_space=None`; the library's default, and the gate's operating point, is HSV, and the
conversion costs 0.86 ms on a 4.16 ms 1024×1024×3 frame (17% on top of the pass; it was
0.88 ms on top of the original's 7.37, untouched since step 6). Its row loop is 228
instructions per 32 pixels at 1.9 instructions a cycle: four groups of eight lanes, each a
chain of some forty cycles through two converts, a float division for the shared reciprocal
1 / (v · diff), two guesses, their floor-and-remainder corrections, two fixed-point products
and the converts back. Tried, each bit-exact on the harness and on every one of the 2²⁴
colours against `cv2.cvtColor` (a check worth keeping: `hsv_exhaustive.py` in the session's
notes), and timed A/B/B/A on HSV frames:

* the division replaced by the hardware's reciprocal estimate refined by two Newton steps
  (exact by `hsv_table`'s correction from any guess within half a unit): +4.3% at 1024² and
  +3.0% at 512². With one step (12-bit estimates suffice on x86): +2.0%. The divider was
  overlapping other work; the dependent multiply-adds are not free;
* two blocks an iteration, so that eight chains could overlap: +0.1% and −0.7%.

What did move it was the table step itself. OpenCV's table value is round(K / x) for x in
[1, 255], and the kernel reaches it from a float guess within 2⁻²³ (relative) of K / x through
a floor, an exact remainder (an FMA), a compare and a conditional add: six dependent
operations per table, twelve per lane group. The guess rounds to the table value directly:
the quotient is never a half-integer (2K / x odd would need x ≥ 8192) and lies at least
1 / (2x) from the nearest one (2 (K mod x) − x is a nonzero integer), while the guess is at
most K 2⁻²³ / x from it, and the ratio of the two, 2K 2⁻²³, is 0.25 for the saturation table
(K = 255 · 2¹²) and 0.03 for the hue table (K = 15 · 2¹³), on any IEEE target. So
`hsv_table` is one `Round`, the chain per lane group is a dozen operations shorter, and the
HSV frame is −5.4% at 1024² (5.05 → 4.77 ms), −4.3% at 512², −3.9% at 256²; the conversion
itself went from 0.86 to about 0.61 ms. Exact on the harness and on every one of the 2²⁴
colours against `cv2.cvtColor`, on the GCC/AVX2, clang and SSE4-only builds. The proof is a
`static_assert` on K; the colour test is what holds the targets to it.

After it, the loop is 181 instructions per 32 pixels, 51 of them (31 shifts, 20 converts)
on the one pipe this core gives both, which is likely what binds it now. The reciprocal
estimate with one Newton step, retried on top of the rounding: +2.8% and +1.0%; the signed
even-lane extraction as an and, xor and subtract on the logic pipes instead of two shifts:
±0.5%. Both reverted. The loop keeps lane order through shifts and masks because shuffles
are the scarce port on Intel cores; on this core the balance is the other way, and a
portable version would have to cut shifts and converts outright, not move them.

A note on the host: three exhaustive runs on this shared machine once reported 150, 40 and
53 differing colours with a value read back from the input array that a byte cannot hold;
four runs of each side afterwards agreed to the pixel on all 16.7 M colours. Transient memory
corruption on the host, not the library: every bit-exact result in this log was taken from
comparisons that never showed a spurious difference, and the colour check is deterministic.

**Step 17, the resize's horizontal pass across planes (October 2026).** The gate's own
frame, a 1080p image thumbnailed to 1024×576 through the fused INTER_AREA resize, then HSV,
then the pass with six levels, had not been timed as a whole since step 7: on the current
build it was 6.99 ms, of which the resize 2.26 (32%), HSV 0.28 and the pass 4.44; the
original's was 9.01, its resize 2.86. The resize's horizontal pass (`area_hpass`, 14% of the
frame's cycles for 22% of its instructions, so already at a high rate) ran once per plane:
41 instructions per group of eight destination columns, 21 of them the three tap slots'
shuffle, promote, convert, multiply and add, the other 20 the window load, the pattern and
weight loads of the slots, the store and the loop's bookkeeping. Over a row's planes at
once (a template on the plane count, one to four), a group's patterns and weights are loaded
once and the bookkeeping shared, and each plane's products and sums are the same operations
in the same order, so every float is the same: the pass's loops 18.6 M → 12.2 M
instructions a gate frame (84.7 M → 79.5 M in all, −6.1%); 1080p → 1024×576, the resize
2.21 → 1.90 ms (−13.7%) and the gate frame 7.18 → 6.84 (−4.7%); 720p → 512×320, the resize
0.96 → 0.74 (−22.7%). The vertical accumulation (`area_vmuladd`, 4.6% of the frame) and the rounding
(2.7%) are at their floors; what remains in the resize is the deinterleave of the source
rows (4%) and the resampled row's store and reload between the two passes, which fusing the
vertical accumulation into the horizontal pass would remove. Tried: the resampled values
scaled and accumulated into one or two destination rows straight from the registers (the
row the source row ends and the one it starts, as OpenCV's mul and muladd, element by element
the same), bit-exact and the tests passing, and slower: the resize +5% at 1080p → 1024×576
and +10% at 720p → 512×320, the gate frame +0.6%. The read-modify-write stores inside the
group loop cost more than the streaming pass they replaced. Reverted.

What did pay next was the tap's widening. The plan's shuffle pattern put a group's G
selected bytes at the front of a 16-byte vector, and each tap then widened them to int32
lanes (`pmovzxbd`) before the convert to float. With 4 bytes a lane in the pattern, the
tap's byte at the lane's low byte and zeros elsewhere, the shuffle lands each tap in its
int32 lane and the widening goes: four operations a tap and plane instead of five, the same
lanes and so the same floats. On the 256-bit target the 16-byte window is duplicated into
both halves of a 32-byte table once per group (`Combine`), since the shuffle looks up within
16-byte blocks. The whole-number-ratio block sums (`area_hsum`, one plane at a time, two taps
a column at 2×) were given the same layout first and measured +4% (their pattern loads
doubled for less than a widening's worth of work), so the layout is the float path's alone
(`AreaHPlan::lane4`) and that kernel is as it was. 1080p → 1024×576, the resize −8%
(1.63 → 1.54 ms; five runs between −5.6 and −8.3%); 720p → 512×320, −5 to −6%; the gate
frame −1 to −2%, within a noisy session. Bit-identical on the three builds; the resize tests
against cv2 at every size pass.

Then the finish. A destination row was a scaled copy of its first tap row (`area_vmul`), a
read-modify-write pass per further tap row (`area_vmuladd`) and a rounding pass over the
sum (`area_round`): with two or three tap rows, four or five streaming passes over 3072
floats. With three resampled rows cached instead of two, every tap row of a destination row
with up to three is at hand, and one pass (`area_finish`) reads them, forms the sum in tap
order (each product rounded, then each sum, as before: no fused multiply-add), rounds and
stores the bytes; the sum is never stored. Rows with more taps (ratios of four and up) keep
the old passes. 1080p → 1024×576, the resize 1.98 → 1.66 ms (−16.5%); 720p → 512×320, −4.6%;
the gate frame 6.59 → 6.16 (−6.6%). Bit-identical on the three builds; the resize tests
against cv2 pass. Where the fused accumulation above lost by putting read-modify-write
stores inside the horizontal pass, this wins by removing passes: what counts is the number
of times a row crosses the cache, not where the arithmetic sits.

Small and certain after it: the 256-bit horizontal pass loaded each plane's 16-byte window
and duplicated it into both halves of the shuffle table with a shuffle; a broadcast-128 load
(`LoadDup128`) does both in the load unit, one ALU operation fewer per plane and group.
1080p → 1024×576, the resize −3% in three runs; 720p → 512×320, ±0.2%; the gate frame
−0.3 to −0.7%. Nothing it could cost on any target (the 128-bit ones do not take the path).

A note on the test suite on this host: `tests/test_resize.py` fails one of its large-image
cases (3840×2160, 4096×3072, 3000×1050) in about one run in three, a different case each
time, on this branch, on the build before step 18 and on the original build alike, while
forty focused repeats of a failing size, single- and three-threaded, agree with cv2 to the
byte and every thread of the resize has its own scratch and line buffer. Settled without
the library: forty rounds of numpy filling a 64 MB array, copying it twice and comparing the
copies mismatched in eleven, by 3 to 215 bytes a round, with no machine-check or memory
error in the kernel log (the memory is not ECC). The host's RAM corrupts large allocations
silently; the all-colours HSV test (48 MB) fails the same way about one run in three. Every
bit-exact claim in this log rests on comparisons that never showed a spurious difference
across dozens of runs (the harness's arrays are small), and the large-array tests should
be read on a machine whose memory passes a memtest.

**The gate's pass at 16×9 px cells, measured (October 2026).** The pass alone on the
1024×576 thumbnail with six levels is 54.5 M instructions a frame, of which the six kernel
loops are 24.4 M (45%); the rest is per-chunk and per-cell work: the folds 4.2 M, derive
3.5 M, the roll-ups 3.1 M, the flush to slots 2.0 M, the summary 1.6 M, the sub-chunk
bodies 1.5 M (nine-row chunks amortise them over little), the HOG loops' prologues 0.9 M.
Tried: the kinds folded two at a time at two-cell blocks (the byte counts and bar maps
through one even/odd lane add, the pair and HOG kinds two per four-input reduce instead of
two with two inputs zero): 124.0 M → 119.5 M instructions per two frames (−3.6%), bit-exact,
and within ±1% in time on the gate pass, the gate frame, 16 px cells at 1024² and the
headline, so reverted. What is left there is structural: the roll-ups cost 3.25 instructions
per child cell and kind because the sums are kind-major (a cell-major layout would sum two
cells' kinds four at a time, six times cheaper, but derive reads four cells a vector), and
levels two and up still go to slots and back through scalar gathers where level one derives
in its band (step 13); each a refactor with band-boundary cases, not a harvest item.

Measured before taking the second one on: by source range over the gate pass (the one-time
resize of the profiling script excluded), the whole slot path for levels two and up is
about 2% — `flush4` 0.5%, the slot pyramid 0.4%, the per-record derives and their moments
1.1% — and an earlier 3.7% for "the flush" was the fold's byte-count loop misread. Not worth
a day's refactor. The two ranges that are large, the roll-up (5.2%) and the summary (5.3%),
each had a contained item instead (step 19).

**Step 19, the roll-up two kinds at a time and the summary once per row (October 2026).**
`pair_up` took a kind at a time: a group's one vector of four children, two shuffles, an
add, a half store, and the loop's pointer bumps, per kind, 126 times per group of four
cells. Two kinds at a time, one even/odd lane add of their two vectors lands kind r's two
parents in the low half and kind r + 1's in the high half: half the shuffles, adds and
bookkeeping for the same stores and the same integer sums. And the finest level's summary
was folded once per group of four cells, each call reloading and storing its six groups of
four double accumulators (min, max, sum, sum of squares) — about a sixth of its work;
folded once per cell row, the same cells go in in the same order, so every double rounds as
before. Together, at the gate's pass: 124.0 M → 117.6 M instructions per two frames, the
roll-up loops 6.2 M → 2.7 M and the summary's 6.3 M → 4.3 M; pinned A/B/B/A over 16 rounds
4.09 → 3.87 ms (−5.4%); 16 px cells at 1024², −3.2%; 8 px cells stride 2, about −1%; the
headline unchanged (−0.2%). Bit-identical on the three builds. The paired folds above, which
cut 3.6% of the instructions for no time, and these, which cut 5.1% for 5.4% of the time,
differ in one thing: the fold's reduce steps were hidden behind its own stores, while the
roll-up's loop bookkeeping and the summary's accumulator reloads were on the critical path.

Tried after it at the same configuration and dropped: the stride-1 product sums only for
chunks of sixteen rows and up (their once-per-chunk corrections, about three hundred
instructions, against nine a half row saved: at nine-row chunks the plain loop should win),
−0.8% of the gate pass's instructions and, timed twice, −2.0% then +3.1%; and the HOG rays
handed to the kernel as a pointer to int16 lanes instead of a vector array, which changed
nothing in the generated loop (GCC copies the four vectors to stack slots per quarter either
way). The per-chunk fixed costs at nine-row chunks — the energy phase's epilogue, the HOG
quarters' prologues, the byte-count fold — are about 1150 instructions a block and chunk
against 3970 in the loops, and each is a few hundred; the day's noise on this host is ±3%.


**A lesson from the host.** The machine this was tuned on corrupts large allocations
silently (above), and the code will run on machines like it. The pass's streaming design,
with a working set of a few hundred kilobytes and no frame-sized intermediate, is a
robustness property there as much as a speed one: what is never materialised cannot be
corrupted, and the frame-sized arrays left — the input, the thumbnail, the outputs — are
the ones a deployment should be prepared to see flip a byte in.

**Step 20, the summary across threads (October 2026).** The gate runs with threads, and at
two to four of them the frame's serial tail is 4.5–6.4% by Amdahl (5.41 → 2.83 → 1.61 ms
at 1, 2, 4 threads). Its one sizeable stage is the summary over every level's cells, folded
after the join by the main thread alone when there is more than one band: a build without
it measured −4% at two threads and −10% at four. The fold's order is the contract (cell
order, each level a chain of float64 sums), so it cannot be split within a level; it can be
split across levels, and level 0's chain can start early. Each level gets its own
accumulators, reset once a frame; band 0 folds its own share of level 0 as it goes, in
order, as a lone band always did; and a third parallel job after the derive folds level
0's remaining cells on the first thread, continuing the same sequence, and the coarser
levels whole on the other threads. At two bands this is even: band 0's in-band fold only
makes it the longer band, and the tail after the join — level 0's other half — is no
shorter than before once the wake-up is paid (+0.4%, measured). So it engages from three
bands (`parallel_summary()`), where band 0's share is a third or less and the tail spreads:
pinned to as many idle cores as threads, A/B/B/A, the gate's 1080p frame 2.68 → 2.48 ms at
three threads (−7.4%) and 2.14 → 1.98 at four (−7.5%); one and two threads take exactly the
old path. Every output is identical across 1 to 6 threads on the GCC/AVX2, clang and
SSE4-only builds, which is the property the whole design serves. For multi-threaded timing on
a shared machine: pin the process to as many idle physical cores as it has threads, or a
band stalled by someone else's process swings the frame by 10%.

Looked at after step 20 and left alone: the fixed cost of a `features()` call is about 26 µs
(a 32×32 frame, one level), of which the 21 numpy views a gate frame returns are about
20 µs in all (a build without them: 3882 → 3866 µs on the gate's pass), 0.4%: not worth an
API change. A 32×32 frame with one cell runs the per-row hash path (its one cell does not
tile the 32×32 hash grid) and the pHash DCT, 55 K instructions a frame, is a fixed cost
too; at the gate (64×64 cells) the hash grid is tiled and the DCT is 0.4%. And the HOG
rays read once per block instead of once per quarter block and sub-chunk (the prologue 47
→ 39 instructions, 32 fewer a block and chunk): −0.8% of the gate pass's instructions and
+1.8% of its time, twelve rounds, minima and medians agreeing; the four vectors held across
the whole block cost the phases around them more than the copies did. Reverted. And the
wake-ups themselves: a worker that has finished a job parks on a condition variable, and the
next job of the same frame follows within tens of microseconds, so each of the two or three
dispatches a frame pays a wake-up (at two threads, a 64×64 frame measured 56 µs on one
thread and 69 on two: the whole parallel gain and more goes to the dispatch). A bounded
spin before parking — an atomic mirror of the epoch and of the pending count, a worker
spinning up to 200 µs for the next job when the dispatcher says one follows in this frame,
the caller spinning likewise for the last worker, nothing spinning between frames — was
bit-identical across 1–6 threads and took the 64×64 frame −9% at two threads, but the gate's
1080p frame +1.6% and +2.2% on minima at two threads in two sessions, −4.5% on the minimum
and +0.2% on the median at three. The ceiling at the gate is about 1% (two wake-ups of
30–40 µs in 2.9 ms), under what the host can resolve, and a spinning worker holds a core
that on a shared machine someone else is using. Reverted.

**Step 21, the summary split by lane (October 2026).** A per-thread instruction count of
the gate's frame at four threads (callgrind with `--separate-threads`) put the four bands
within 1% of each other on the pass and showed where the rest went: the summary fold, 3.6 M
instructions a frame, had 2.5 M of them on thread 0 after step 20's split by level, because
level 0 is four fifths of the cells and its chain, in-band share and tail alike, was one
thread's. The fold's contract is the order of each lane's additions, not which thread runs
which lane: `fold_summary` already walks a cell batch one group of lanes at a time, a group a
vector of floats (two of doubles) or, for the ragged tail, a narrower one, and a group's chain
depends on nothing outside its lanes. So the tail after the join is now cut by lane: a run is
one level's cells, one channel, a range of its features aligned to eight (whole vectors at
every width the fold uses, so the groups are the same on SSE4, AVX2 and AVX-512) or its four
moments, each run folded over all the level's cells in order on one thread; the runs are
listed finest level first and dealt to the threads by the cumulative cells × lanes their
midpoints fall in, adjacent runs of one channel on one thread merged into a single call. Band
0's in-band fold and the two-band serial path go with it: a lone band folds as it goes, as
before, and every other frame takes the one path. Bit-identical across 1–6 threads on the
three builds (a sub-range of eight lanes picks the same tags and the same groups as the full
call does). Pinned to as many idle cores as threads, A/B/B/A: the gate's frame −2.1% and −2.4%
at two threads in two sessions, minima and medians agreeing; three and four threads within
±0.5% both times, which was not the 5–9% the instruction counts promised. An ablation with
the job emptied (an environment variable; outputs wrong, timing only) says why: at four
threads the job still costs 265 µs, 15% of the frame, against about 75 µs of arithmetic per
thread, and at two threads 96 µs (3.6%), close to its arithmetic. After the join every run
reads records other cores wrote, 32–200 bytes a cell at a 648-byte stride, and at four
threads on this host that is a miss per cell per run across two core complexes. A software
prefetch of the next batch's lines (`hwy::Prefetch`, one per line per cell) did not help:
+2% at four threads against the split without it, +4.8% at three, the two-thread gain gone.
Reverted. What would fix it is folding in-band on every band, which needs per-band partial
sums and so reassociates the additions: the one thing the contract forbids.

**Step 22, cycles against instructions (October 2026).** With every loop at its instruction
count, the next map was a cycle profile of the gate's frame (`perf record` on the symbol
build, samples attributed to the same loop ranges as callgrind's counts) set against the
instruction profile: a loop whose share of cycles outruns its share of instructions stalls
on something counting cannot see. The kernel loops run at 0.8–1.15 cycles per instruction
share and the resize's horizontal pass at 0.5–0.7; what stood out was the three-channel
gather at 1.96 (34% of its samples on its first load: this is where the frame is first read,
and the loads wait on memory the hardware prefetcher had not fetched), the derive's scatter
at 1.9 (its four record stores), the fold of a multi-cell block's byte kinds at 2.3 (fifteen
branches a block, each kind testing the block width again), the row responses at 1.6 (eight
stores a row: the store port's floor), and the resize's finish at 1.2. Five items came out
of it, each built alone as a variant and timed pinned A/B/B/A against the build before;
three stayed. The gather asks for the frame's lines 768 bytes on (two prefetches a step
cover every line at the 96-byte stride): −1.1%, −0.9% and −0.9% on the gate's frame in
three sessions. The finish converts its rounded sum with `ConvertInRangeTo` instead of
`NearestInt`: the sum is a weighted mean of bytes, so Highway's compare-and-blend against
overflow was two instructions per eight outputs for nothing; −0.5% and −0.4%. The kinds
loop decides the block width once, outside the loop over kinds, the bodies the same: −0.6%
and −1.2% on the gate, −0.7% on the headline. Together they measured −0.0% and −0.4% on the
gate (medians −0.2% and −1.0%), −0.5% at two threads and −0.7% on the headline, with the
module's instructions per two gate frames 142.7 M → 139.5 M (−2.2%): less than the sum of
the parts, never above zero in twelve runs. Reverted: the horizontal pass with its first
slot peeled so the slot loop has no `k == 0` to test (+2.2% and +1.5% on the gate; the
select was never the cost, the samples had skidded onto it), and the records two groups on
prefetched ahead of the scatter's stores (−0.9% on the gate, +1.4% on the headline). A
lesson relearnt on the way: the first versions of two items used lambdas capturing the
block's vectors by reference, and GCC spilled them, +8.6% and +3.3%; the same code written
out was what measured above.

Looked at after step 22 and left alone. The clang build of the same source, which the
verification runs anyway: +3.1% on the gate's frame and +6.2% on the headline against GCC
11, so GCC stays. The derive's scatter storing two four-feature groups of a record at once
(eight-float stores, `Combine` of the two transposes' rows: the loop was at the store
port's floor, fourteen 16-byte stores a record): −0.8% on the gate and −0.9% at 16 px cells,
+1.7% and +1.6% on the headline in two sessions; the headline's records are a quarter as
many, so the swing there is the kernel's loops moving with the code size inside their
function, which this host has shown before (`-falign-loops`, step 12). Mixed, so reverted.
And the centre response dropped from the stored set (the row responses loop stores eight
vectors a row and sits at the store port's floor; the moments phase, its only reader,
would widen the centre from the row bytes instead, one instruction a row each as the loads
were, and the slots shrink by a quarter): +2.7% on the gate, +3.1% on the headline and at
16 px cells. The moments loop had gone from 46 to 79 instructions per two rows: the three
row addresses are loaded from the distance table and added to the base per row, and the
half's widening became a branch, since the half is a loop variable there. A rolling centre
(a row's lower row is the next row's centre at stride 1) would cut that to one load a row,
for a net of about nothing against the two stores. Reverted.

**Step 23, cycles per loop as the judge (October 2026).** Two of step 22's rejects were
rejected by the headline's wall clock while the gate's said otherwise, and a control settled
which to believe: the committed source with 8 or 24 bytes of padding at the top of
`cell_row` -- no change in what runs -- measured −0.6% and −0.2% on the headline and −1.5%
on the gate, pinned A/B/B/A, twelve runs each. Code placement alone moves this host's wall
clock by about ±1.5%, which is the size of the items left. So the judge for an item under 2%
is now the cycle profile of the loop it changes: `perf record` on each build, samples
attributed to loops found by their backward jumps, loops matched between builds by their
length and instruction mix, the second build scaled by the loops both have unchanged
(`loopdiff.py`), with the module's instruction count as the load-independent check. By that
judge the eight-float scatter had been worse after all (its derive loop split in three, +285
samples), the three items kept in step 22 show where they should (the gather −320 samples,
the finish −100, the kinds loop −150), and the records prefetched two groups ahead of the
derive's scatter -- rejected on the headline's +1.4% -- take the scatter loop from 506 to 282
samples of 23,171 at no cost in the derive loop, 1.0% of the module, so it is in. With it the
horizontal pass's store bound, which the compiler reloaded from the plan and tested before
each of a group's three stores, is tested once per group (`gfull`): within run noise on
cycles, and the module's instructions per two gate frames 139.9 M → 137.8 M (−1.5%) for the
two together. Wall clock, for the record: the gate 5.36 → 5.23 ms (−2.4%), the headline
4.37 → 4.32 (−1.1%). The compiler had also already peeled the slot loop's first iteration,
which is why step 22's peel by hand had nothing to gain.

**Step 24, groups of eight (October 2026).** The per-loop table at HEAD put the level-1
roll-up (`pair_up`, `roll_block`) at 3.5% of the gate's cycles for 2.5% of its instructions,
stalling on its read-modify-writes: at 16 px cells the level-1 row is 132 kinds × 36 parents
of int64, 38 KB, more than L1, and every group of four finest cells touched each kind's
line once for two parents. The group is now two derive vectors, eight cells (`GROUP`), so
a kind's line is touched once per eight cells for a whole vector of parents, through the
path that two-cell blocks of four-pixel cells already took; the derive walks a group one
vector at a time as before, and the fold's cell offsets are the same arithmetic. The first
build kept the group's odd kind stride (`group | 1` = 9, 72 bytes): the roll-up loops went
553 → 453 samples but a fold loop went 392 → 496, its stores straddling two lines where the
40-byte stride of a group of four had fitted one. With the stride a whole line for this group
size (8, 64 bytes; smaller groups keep the odd stride) the roll-up loops are 370 and the
folds are back where they were: −0.8% of the module's cycles, and −0.9% of its instructions
per two gate frames (137.8 M → 136.6 M). Wall clock, pinned A/B/B/A: the gate 5.26 → 5.17 ms
(−1.7%), 16 px cells at 1024² −1.7%, the headline (32 px cells, a level-1 row that fits L1)
−0.6%. Bit-identical on the three builds and across 1–6 threads: every cell's sums are the
same integers whichever group they are derived in.

Looked at after step 24 and left alone: the one-cell fold's lane scatter (at 32 px cells a
block is one cell, and its 44 kinds go to a kind-major group a scalar store each: 3% of the
headline's cycles at 1.8 cycles per instruction share). Folding each cell's kinds contiguous
through the fold's existing vector path into a staging row and transposing the group once at
its end (four kinds of four cells through the fold's own interleave network) took the fold's
loops from 803 to 679 samples and the headline's instructions −0.7%, but the gate, whose
two-cell blocks never take the path, measured +3.6%, +3.0% out of line and +1.6% with the
fold's target chosen at run time, and the headline +1.5% in that last form: each form
recompiled the multi-cell fold loops around it larger (one +40%, a new 434-instruction
loop), since `cell_row` holds every phase inlined and a pointer kept live across its block
loop changes the allocation of all of them. Reverted. A change inside `cell_row` has to be
judged on every configuration's loops, not only the one it targets.

Where the single thread stands after step 15: 1.75× the original at the headline
configuration (7.51 → 4.28 ms pinned on a quiet core), 1.88× at 128 px cells, 1.72× with one
channel; the per-cell-bound configurations (8 px cells at stride 2) 1.04×. The six kernel
loops are 79% of the 54.6 M instructions a frame and each is at or within a few instructions
of the count its arithmetic needs; what is left single-threaded is below 1% an item. The
exact band threading scales: 4.21 / 2.14 / 1.15 ms at 1 / 2 / 4 threads against the
original's 7.36 / 3.74 / 1.97, the same 1.7–1.75× at every thread count, and two threads of
this build do the frame in 0.29× the original's single-thread time.

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
* **A planar layout for the maps** (September 2026; not in the tree). `FeatureComputer(...,
  planar=True)` laid each feature of each channel out as one contiguous `(cy, cx)` plane
  (`maps[i]` the same `(cy, cx, C·F)` array with strides `(cx, 1, plane)`), so that a
  consumer taking the maps a feature at a time — fastdet's scorer bins whole planes — could
  read them as they lie instead of packing a copy the size of the maps (3.7 MB a frame:
  0.4 ms on the development VM, 0.58 ms on the laptop). The derive stage stored each
  vector of a feature's lanes straight into a per-band scratch and wrote a cell row to its
  planes in one run per plane; the summary fold transposed eight cells of eight planes in
  registers and, as a side effect, was split over features across bands (each feature's
  cells still added by one thread in order), which gained at two threads in both layouts
  (720p, 512×320, two VM threads: 2.4 ms planar against 2.6 interleaved and 2.7 before).
  Same bits in both layouts at 1–8 threads. It was not kept: MSVC needed the fold rewritten
  twice (lambdas over `constexpr` locals, `if constexpr` branches instantiated), it was a
  second output layout to keep correct, and the hosts' problem was solved on their side —
  fastdet's scorer now reads the cell-major maps in place through a base pointer and two
  strides per column, with no change to imfeat.
* **The resize by periods** (October 2026; not in the tree). `INTER_AREA`'s cells tile the
  row, so for a ratio `sw/dw = p/q` in lowest terms the taps of destination column `qj + m`
  are column `m`'s moved `pj` source columns along, with the same weights. A kernel that cut
  a row into eight segments of whole periods, interleaved their bytes once and resampled the
  segments in lockstep — every tap one contiguous load, widen, convert, multiply and add,
  eight periods a step, no shuffle window — with one vertical sweep over a ring of source
  rows and the rounding packed through the saturating narrowings, kept every byte of
  `cv2.resize` (eighteen sizes, six kinds of content, four channel counts, 1–3 threads, GCC
  and clang, AVX2, AVX-512 and SSE4) and took the resize from 28.7 M to 17.0 M instructions
  of a 112.7 M-instruction pass at 1080p → 1024×576, from 48.9 M to 11.0 M at 1080p →
  256×256 and from 16.0 M to 6.7 M at 720p → 512×320 (development VM, AVX2, one thread);
  wall clock, the fused pass at 1080p → 256×256 went 7.3 → 4.3 ms on one development-VM
  thread, and on the laptop the resize inside a two-thread 720p → 512×320 pass measured
  0.32 ms. The maintainer reverted it: the periodicity plan is a second resize kernel to
  keep correct beside the window one, for a saving that at the gate's operating point is a
  few tenths of a millisecond. The numbers stay here because the lever is real; the commit
  (*Resize: the horizontal pass by periods in lockstep*) can be revived if the thumbnail
  ever dominates again.
* **float16 anywhere.** Asked and answered rather than tried: the pass holds integers until
  the last step, and the few float stages (derive, summary) are float64/float32 for a
  reason; numpy float16 arithmetic on the hosts' side is 5–15× *slower* than float32 (no
  hardware path on x86), and it would change every output. Not an optimisation on any axis.

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
9. **Count instructions inside the object, not the process.** A whole-process callgrind
   difference between a 2-frame and a 12-frame run is contaminated by anything that happens
   in one run and not the other — the first run of a fresh build compiles and writes the
   package's `.pyc`, the second reads it — which made step 8's −3.8%, −3.7% and −3.1% read
   as −4.2%, −6.3% and −3.6%. Sum the self cost of the library's own object instead (the
   per-`ob=` totals of the callgrind file); those are identical run to run.
10. **Ask what reads an output before optimising how it is made.** Two whole stages of the
   pass (steps 8–9) had no consumer; deleting them was worth more than most kernel work and
   carried no risk to the rest.
11. **Read the hot loops' assembly; the compiler's spills are a cost-table item.** Find the
   back edges of the kernel in the object's listing and count stack stores and
   register-to-register copies per loop. Three loops here carried 10–20 such instructions a
   row for the same reason: accumulators captured by reference in a lambda have a home in
   memory, the `may_alias` vector loads may read it, so GCC stores them every row. Passing
   the accumulators through the loop by value, and giving a phase a function of its own when
   the enclosing function's live values crowd the registers, took them out (step 15).
12. **Symmetries of the constants are algebra too.** The HOG rays' mirror symmetry halved
   the kernel's largest loop's multiplies (step 15); it had been in the table since step 1.

13. **Below two percent, judge a change by its own loop's cycles.** Code placement alone
    moved the wall clock of this host by ±1.5% (a padding of 24 bytes in a function, nothing
    else); items of that size were accepted and rejected on that noise until the judge
    became the cycle samples of the loop the item changes, matched between builds by the
    loop's signature, with the module's instruction count as the second witness.

## Measuring

* **Wall clock**: minimum over many frames, builds interleaved in one session. The first tool
  and the least trustworthy; ±5% on a shared machine is noise.
* **Ablations**, where no profiler is available: the same build with one stage deleted, timed
  in its own process against the full build in an interleaved loop, the minimum of several
  rounds; the difference is that stage's share of the frame under real scheduling, which an
  instruction count cannot give (the LBP's 60 instructions a row cost 2% of the frame, the
  HOG's 160 cost 27%). The deleted stage's outputs are wrong, so the variant is never
  installed; `objdump -d` on the result shows what the compiler made of each loop.
* **Callgrind per loop.** On a machine without root, valgrind builds from its tarball into
  `~/.local` in a few minutes; `perf` needs `kernel.perf_event_paranoid` lowered by root.
  Run on a build with `-g` and nanobind's `NOSTRIP`, with `--dump-instr=yes --cache-sim=yes
  --branch-sim=yes`, then attribute each instruction's counts to the innermost loop that
  contains it (backward jumps within a function, from `objdump -d -l`): the per-line view
  is useless for Highway code, since every intrinsic's cost lands on its header line, but
  per loop it gives the instruction, load, store, miss and mispredict counts of each phase
  exactly, and it is what found the cell-sum layout in step 10.
* **A busy machine.** When the cores are shared, interleave the builds A/B/B/A within each
  round (the second of a pair measures 2–3.5% slower otherwise), pin to the core pair with
  the least recent busy time, take the minimum over many rounds and ignore the median. Count
  instructions with callgrind on the symbol build (the module's own count: a whole-process
  count through Python varies by 5% between runs even with a fixed hash seed) as the
  load-independent number, and never time while a build runs on the same machine.
* **Variants and pinned A/B.** For items under 2%: build each as its own variant (a copy of
  the source with the other items reverted by exact text, its own build directory, the
  Python package copied around the new module), check it bit-exact, then time all the
  variants interleaved and pinned with `taskset` to the core pair with the least recent busy
  time (`/proc/stat` over half a second, SMT siblings summed), eight rounds, the minimum.
  That held the noise under ±0.5% on a machine running someone else's training job, where
  unpinned runs wandered by 3%. Interactions are real (three items measured more together
  than apart, one flipped sign), so retest the dropped ones on top of the batch before
  discarding them.
* **Pipe widths.** Instruction counts stopped predicting time in step 11; what did was the
  per-pipe issue width of the core, measured in five minutes with inline assembly: ten
  independent register chains of one instruction in a counted loop, pinned to a core, ops per
  nanosecond against the clock. It is the only way to know, without `perf`, that every
  vector multiply shares one pipe on this core and the kernel's largest phase is within 20%
  of that pipe's floor. Run it on the target core before trading instructions for anything.
* **Instructions per frame**: `valgrind --tool=cachegrind` on a run of 4 frames and a run of
  24, differenced. `callgrind` with a few functions marked `noinline` gives the split by
  stage. For a fused pass driven from Python, `callgrind` on 2 and 12 frames with the self
  cost summed per object (lesson 9), and per function within the object for the split.
* **Divider operations per frame**: `callgrind --dump-instr=yes`, summing the execution
  counts of every `div`, `idiv`, `vdiv*` and `vsqrt*` instruction in the library.

At 256×256×3, stride 2, 8 px cells the kernel is about 20% of the instructions and the
per-cell stage 70%. At stride 1 with 16 px cells the kernel is about two thirds, a quarter of
it accumulator loads and stores.

## Verifying

* The repository's tests, extended from 599 to 853 (805 after the cross-channel and
  profile tests went with their outputs): oracle tests for the bar detector (numpy)
  and for the texture sums (exact), over block geometries chosen to reach every kernel path;
  output lifetime; threads 1, 2, 3, 4 and 8; the colour conversion against
  `cv2.cvtColor` on every BGR value, through the vector path and the scalar tail, and the
  fused pass against the pass on the converted image, byte for byte; the thumbnail resize
  against `cv2.resize(INTER_AREA)` on frames of twelve sizes, six kinds of content, four
  channel counts, any layout and 1 to 3 threads, and the pass on a frame against the pass on
  cv2's thumbnail, byte for byte, at every stride, thread count and channel selection, and
  with the thumbnail size chosen by each policy (`"pow2"`, `"pow2-cover"`, `"pow2-fit"`) as
  with the size given, a thumbnail the frame's own size included.
* A bit-identity harness: 14,328 output arrays (6 shapes × 4 channel counts × 5 grids ×
  6 strides × 2 kinds of image, at 1, 2 and 3 threads) hashed per build and compared with the
  previous build after every change, on AVX2, AVX-512 and SSE4 builds.
* AddressSanitizer over the suite. It caught a block-buffer stride that overran on tiny grids.
* clang-format, cppcheck and `-Wall -Wextra -Wshadow`; ruff, black and mypy for the Python.

Steps 10 to 14 were checked the same way: 40,164 output arrays (eight shapes of 1–8 channels, five
grids, six strides, two kinds of content, 1–3 threads, `features()` and `compute()`, the
fused BGR → HSV and thumbnail paths) hashed on the AVX2 GCC build, the clang build and an
SSE4-only build, each identical to the previous code's build with the same compiler and
target; and the 740 tests of the suite, cross-checks included.

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
* **A batched fold.** Done in step 13, with portable ops after all (`fold_cell`,
  `fold_hog_multi`, `fold_pair_multi`); the remaining single kinds (`Σw⁴`, `|grad|⁴`, the
  bytes and the bar maps of multi-cell blocks) still fold one at a time.
* **Merging the 16-bit phases for small chunks.** At 4 px cells the per-phase overhead shows
  (step 10); a single-loop variant of the energy, Laplacian and moment phases for chunks under
  about 8 rows would recover it, at the cost of a second copy of that code.
* **Level 1 derived straight from the rolled-up sums.** Done in step 13 for rows complete
  within a band; the coarser levels still go through slots and scalar gathers, a few percent
  at 8 px cells. (The channel-pair features that
  shared this item are gone, step 8.)
* **Batched folds with the roll-up fused in.** At the gate's operating point the folds and
  the level roll-up are about a fifth of the instructions; folding a cell row's blocks in one
  sweep and adding straight into the level-1 row would remove most of the per-block
  bookkeeping. Exact by construction (integer sums). Not started.
* **The serial tail at two threads.** The summary fold and the two wake-ups per frame are
  serial; the laptop's scaling of the 1080p pass at 1024 px (10.3 / 5.4 / 3.3 ms on 1 / 2 / 4
  threads, October 2026) puts the serial share at 6–9% by Amdahl's law (an earlier run,
  10.5 / 5.9 / 3.5, said 12%). The band-independent summary above is the exact fix for about
  half of it; one wake-up per frame instead of two is the rest.
* **Lazy outputs.** `summary` and `hashes` are read by framegate, `maps` by both hosts, and
  nothing else is made any more; if a host ever needs only `maps`, the summary fold (6% of
  the frame) and the hash readout could be skipped per computer. Cheap to add, exact.
* **Thread dispatch.** Two wake-ups of parked workers per frame plus a serial tail mean
  threads pay only from about a millisecond per frame.
* **NEON** has not been re-measured since the kernel rewrite.
