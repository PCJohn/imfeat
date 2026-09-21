# How the core got fast

An engineering log of the optimisation work on `imfeat`'s C++ core: the architecture it ended
with, what each step bought, what was tried and dropped, and how every change was measured
and verified. The constraints throughout: no approximations, every integer sum bit-exact, no
float tolerance loosened, nothing tuned to one machine.

## Result

`features()`, one thread, 4 levels, on a 16-core 2.1 GHz Xeon (Linux, GCC), ms:

| config | original | now | |
|---|---|---|---|
| 256×256×3, 32×32 grid, stride 2 | 1.17 | 0.42 | 2.8× |
| 256×256×3, 64×64 grid, stride 2 | 3.03 | 1.31 | 2.3× |
| 512×512×3, 32×32 grid, stride 2 | 2.56 | 0.93 | 2.8× |
| 1024×1024×3, 32×32 grid, stride 1 | 26.6 | 8.05 | 3.3× |
| 1024×1024 (C=1), 32×32 grid, stride 1 | 22.5 | 2.68 | 8.4× |
| 1024×1024×16, 32×32 grid, stride 2 | 39.9 | 16.0 | 2.5× |

A Windows / MSVC laptop shows the same ratios. "Now" also computes nine features and two
profiles the original did not. In machine-independent terms, per 256×256×3 stride-2 frame:

| build | instructions | integer divides | FP divides | FP square roots |
|---|---|---|---|---|
| original | 13.8 M | 34,800 | 53,600 scalar | 16,400 scalar |
| kernel + derive rewrites | 5.33 M | 10,300 | 40,100 scalar | 16,400 scalar |
| + texture features | 6.18 M | | | |
| + first SIMD derive | 6.37 M | 11,300 | 7,400 scalar + 7,100 vector | 4,200 + 3,200 |
| final | 5.23 M | 13 | 4,400 scalar + 7,900 vector | 4,200 + 3,200 |

## Architecture

1. **Rows → planes.** Each image row is de-interleaved once into a rolling window of planar
   rows. With an even stride the planes are split by column phase (up to 8), so sampled columns
   are adjacent and a stride of 2 walks half as many blocks instead of masking half of each.
2. **Kernel: columns as lanes.** One kernel takes a vector-wide block of columns from rows
   `r−1, r, r+1` (and the bar detector's taps) and computes every per-pixel quantity
   in-register; ~76 accumulators per channel are updated per block-row. The first row of a cell
   row *assigns* them, so nothing is ever zeroed.
3. **Fold: lanes → cells.** Once per block per cell row, each accumulator's lanes are summed
   into cells (widening pair sums, never a scalar loop). When a block holds whole cells, the
   cell sums live in a block-sized buffer, contiguous per sum kind, that stays in L1.
4. **Cells as lanes.** From that buffer, four cells a vector: exact int→double conversion,
   the moments (re-centred exactly in doubles), the 54 features through one compiled routine,
   a transposed scatter into the output, the summary fold, and a pairwise add straight into the
   level-1 row in flight. The finest cells are never laid out as the 44 int64 "slots".
5. **Coarser levels.** The level-1 row becomes slots when complete; higher levels roll up from
   slots and derive through the same routine, gathering four cell-channels at a time.
6. **Output.** Everything is derived into one pooled block; the returned arrays are views that
   keep it alive, and it is reused when they are dropped. No copies, no page faults.

## What worked

In order of effect. Times are single-thread `features()`; percentages are of the frame.

* **Columns as the SIMD lane** (the original packed *channels* into lanes). 256×256×3 s2:
  1.18 → 0.74 ms; 1024×1024 C=1 s2: 7.08 → 0.92 ms. Per-channel cost became flat in C.
* **Algebraic rewrites that make every sum a vector add.** HOG bins as differences of
  cumulative masked sums (the ray tests are nested; each is one `pmaddwd`); moments of
  `v − 128` so squares fit 16-bit multiply-adds, restored by a binomial shift; LBP's last bin
  as a remainder; the bar detector in saturating uint8 arithmetic
  (`min(max(l−c,0), max(r−c,0)) = SatSub(min(l,r), c)`); Sobel, Laplacian and six Laws masks
  from three shared 1-D row responses.
* **Never storing the finest level, and deriving it while hot.** First each cell row was
  streamed through derive, hash grid and roll-up instead of being written out and read back
  twice (with the pooled outputs: 0.55 → 0.37 ms at 256×256×3 s2 on the development machine);
  later the same was done per block, from sums that never leave L1 (the items further down).
* **Removing every per-row / per-block / per-cell divide.** 35,000 → 13 integer divides per
  frame: wrapped window slots, tabulated tap offsets and mesh lanes, running counters, shifts
  for power-of-two factors, and `floor(x/n)` as a multiply by `1/n` corrected by the exact
  remainder. Invisible in instruction counts; a large part of a 28% wall-clock gain on the
  Xeon, whose divider is slow.
* **Pooled zero-copy outputs** with shared ownership: steady state allocates nothing.
* **Summary in one pass** (min, max, sum, sum of squares together, ever narrower vectors for
  the ragged tail) and written by C++ in its final layout: −5%.
* **Roll-up from sums, pair-wise** (−7%), **shared exact moments** (−4%), **scatter with the
  full-vector case hoisted** (−3%), **block-local cell sums** (−3 to −5%).
* **SIMD de-interleave through the phase splitter** for 8 and 16 channels, which fixed a
  regression the first kernel had introduced there.

## What did not

* **SIMD transposition of cell sums into slots** — slower, twice. The eventual answer was not
  to transpose the finest level at all.
* **Two-lane vector divides inside the scalar derive** (pairing different divisions of one
  cell) — slower: building the vectors sits on the latency path.
* **The first SIMD derive.** Four cells a vector cut divider operations fourfold but
  *executed more instructions* than the scalar derive (6.18 M → 6.37 M): slot conversion and an
  inefficient scatter ate the gain, and wall-clock A/Bs could not resolve it. It paid only
  once the scatter was fixed and the slots it replaced were no longer written.
* **A generic roll-up helper** with a runtime group size and `/`, `%` by the pyramid factor:
  30% of the frame. The specialised pair-up with shifts is 4%.
* **Reading a block's sums one block late**, and **half-width loads** sized to the earlier
  stores — both to dodge store-forwarding stalls. No change: not the bottleneck.
* **`__attribute__((optimize("no-fast-math")))`** to make one function strict. Lambdas inside
  it stopped inlining (−20%), and GCC still regrouped `(6·d²)·u₂` among the intrinsics. What
  pins a product under `-ffast-math` is writing it as a fused multiply-add with a zero addend.
* **Band boundaries aligned to parent rows** for threading — broke the contract that `threads`
  is honoured up to the row count. Rows whose parents start in another band are instead held
  as slots and added after the join.
* **Kernel ideas rejected on operation count:** HOG with 4 ray tests and mirror logic (same
  ops); 16-bit ray tests (the products need 32 bits); Laws energies from row-product sums
  (nine products for eight energies, worse at stride ≥ 2); caching row responses between rows
  (≤ 3%, stride 1 only); LBP by lookup or bit-packing (no fewer ops). Accumulating two rows
  before each accumulator write would save ~5% of the kernel and was not pursued.
* **Features considered and not added:** per-cell min / max (not additive — breaks the
  pyramid); intensity or centre-symmetric-LBP histograms (many sums); signed HOG (doubles the
  HOG cost); LBP concentration and extrema balance (exact functions of existing outputs; a
  linear fit from the 54 features already explains 74% and 60% of them); temporal difference
  energy (would make `features()` stateful).

## Measuring

Wall-clock A/Bs (min of N, builds interleaved) were the first tool and the least trustworthy:
on a shared machine ±5% is noise, and they flattered a change that removed no work. Two
deterministic counts settled arguments:

* **Instructions per frame**: `valgrind --tool=cachegrind` on a run of 4 frames and a run of
  24, differenced. `callgrind` with a few functions marked `noinline` gives the split by stage.
* **Divider operations per frame**: callgrind with `--dump-instr=yes`, summing the execution
  counts of every `div`, `idiv`, `vdiv*`, `vsqrt*` instruction in the library. Instruction
  counts treat a 40-cycle divide like an add; this does not.

The profile that redirected the work, at 256×256×3 stride 2 (8 px cells): kernel 20%, the
per-cell stage 70% (~1,000 instructions per cell-channel). At stride 1 with 16 px cells the
kernel is ~66%, a quarter of it accumulator loads and stores.

## Verifying

* The repository's tests, extended to 710: oracle tests for the bar detector (numpy) and the
  texture sums (exact) over block geometries chosen to hit every kernel path; output lifetime;
  profiles; threads 1, 2, 3, 4, 8.
* A bit-identity harness: 14,328 output arrays (6 shapes × 4 channel counts × 5 grids ×
  6 strides × 2 image kinds, threads 1 / 2 / 3) hashed per build and compared with the previous
  build after every change, on AVX2, AVX-512 and SSE4 builds.
* AddressSanitizer over the suite (it caught a block-buffer stride that overran on tiny grids).
* clang-format, cppcheck and `-Wall -Wextra -Wshadow`; ruff, black and mypy for the Python.

Exactness came from structure rather than flags: integers stay integers until the end;
doubles are used for integer arithmetic only where every value is provably below 2⁵³; sums
below 2⁵² convert to double by planting them in the mantissa of 2⁵²; and a float that must be
bit-equal across code paths (a grid as level 0 of one pyramid and level 1 of another, or at
two thread counts) goes through **one compiled body**, a lane per cell, rather than two
implementations kept in agreement. Observed differences: none on AVX2 / AVX-512 builds against
the original apart from two bar-detector corner cases that were bugs (a dense-path ISA
difference, and a sampling mesh that did not restart per cell); on the non-FMA SSE4 build the
float64 moments move by at most one ulp when code around them changes.

## Open ideas

* **Summary independent of the band split** (sum per cell row, then the rows in order), so the
  summary can run inside the bands. Nothing at one thread; an estimated 5–15% at 2–4 threads
  on small-cell grids; moves `summary` means and stds by an ulp once. Not done.
* **int32 cell sums where the level's sample count bounds them** (all but `Σw⁴` and `Σg⁴` at
  typical finest levels): halves the per-cell data volume; estimated −8% at small cells.
* **Level 1 derived straight from the rolled-up sums**; **channel-pair features in lockstep**
  (−6,000 scalar divide / square-root operations per frame).
* **Thread dispatch.** Two wake-ups of parked workers per frame, plus a serial tail, mean
  threads pay only from about a millisecond per frame; at 256×256 two threads win on one test
  machine and lose on the other.
* **NEON** has not been re-measured since the kernel rewrite.
