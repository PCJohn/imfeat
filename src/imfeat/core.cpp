#include "hwy/highway.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#define IF_RESTRICT __restrict
#else
#define IF_RESTRICT __restrict__
#endif
// A lambda that must be inlined (a loop body instantiated per compile-time case, whose
// accumulators would otherwise go through memory around a call).
#if defined(__GNUC__) || defined(__clang__)
#define IF_LAMBDA_INLINE __attribute__((always_inline))
#else
#define IF_LAMBDA_INLINE // (MSVC: left to its judgement; only the speed depends on it)
#endif
// fold_summary() must square and add UNFUSED, as the scalar loop it replaced compiled: a fused
// multiply-add rounds once where that rounded twice, which moves a std by an ulp. GCC would
// contract the two intrinsics under -ffast-math, and so might clang.
#if defined(__clang__)
#define IF_UNFUSED_FN
#define IF_UNFUSED_BODY _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define IF_UNFUSED_FN __attribute__((optimize("fp-contract=off")))
#define IF_UNFUSED_BODY
#else
#define IF_UNFUSED_FN
#define IF_UNFUSED_BODY
#endif

// The moments' floating-point code spells out its operations, fused multiply-adds included, and
// IF_STRICT keeps -ffast-math from rearranging them: the same bits from any compiler.
#if defined(__clang__)
#define IF_STRICT_FN
#define IF_STRICT_BODY _Pragma("clang fp contract(off) reassociate(off)")
#elif defined(__GNUC__)
#define IF_STRICT_FN                                                                               \
  __attribute__((optimize("no-associative-math", "no-reciprocal-math",                             \
                          "no-unsafe-math-optimizations", "fp-contract=off")))
#define IF_STRICT_BODY
#else
#define IF_STRICT_FN
#define IF_STRICT_BODY
#endif

namespace nb = nanobind;

// imfeat -- single-pass multi-channel image statistics. See __init__.py.
//
// Per finest cell per channel we keep NSUM int64 additive sums:
//   [0..3] Sxx Syy Sxy count | [4..12] HOG(9) | [13,14] local-max/min counts
//   | [15..18] S1 S2 S3 S4 (raw power sums of the pixel value) | [19..28] LBP^riu2(10).
// Every sum is additive, so pyramid levels and the global reduction are exact
// sums of the finest cells -- the image is read exactly once. Central moments and
// the tensor eigen-features are nonlinear, so they are derived at the end.
//
// One traversal, COLUMNS as the SIMD lane. Every image row is de-interleaved once into a
// rolling window of planar rows; block() then takes a block of columns down all the sampled
// rows of a cell row, in phases that each carry a few accumulators in registers (bytes:
// extrema, LBP, the bar detector; 16-bit: the 1-D row responses once per image row, then the
// Laws energies and the tensor, the Laplacian and the moments, and the HOG ray tests), and the
// accumulators are folded to cells once per block, so memory is touched once per cell row.

namespace
{
constexpr int SXX = 0, SYY = 1, SXY = 2, CNT = 3, NCOMP = 4;
// HB (HOG orientation bins over [0, pi)) is compile-time: it sizes the per-cell
// accumulator and the inner histogram/ray loops, so a runtime value regresses the
// hot loop (~17% measured). Everything below is DERIVED from it -- change HB here
// and rebuild to retune bin count; no other constant is independently hardcoded.
constexpr int HB = 9;
static_assert(HB % 2 == 1, "block() pairs the bin-boundary rays up as mirror images");
constexpr int HRAYS = (HB - 1) / 2; // the rays block() tests; the rest are their mirror images
constexpr int HOG0 = NCOMP;         // 4
constexpr int NMAX = HOG0 + HB, NMIN = NMAX + 1; // 13, 14
constexpr int PS0 = NMIN + 1;                    // 15: S1 S2 S3 S4
constexpr int NMOM = 4;
// LBP^riu2_{8,1} (Ojala 2002): 10 bins -- popcount 0..8 for the "uniform" codes
// (<=2 circular 0/1 transitions), plus one catch-all bin for the rest.
constexpr int LBPB = 10;
constexpr int LBP0 = PS0 + NMOM; // 19
constexpr int SG4 = LBP0 + LBPB; // 29: sum of |grad|^4 = (gx^2+gy^2)^2, for gradient sparsity
// Bar/stroke detector ("bard"). A centre-surround test at three lags: at lag d a
// pixel is DARK if BOTH taps at +/-d are brighter than it, scoring the smaller of the
// two margins, and LIGHT if both are darker. A step edge scores zero -- one side
// matches the centre -- so this fires on strokes and thin structures of a given width,
// which the Sobel/LBP banks conflate with ordinary contrast. Horizontal and vertical
// combine with max; a pixel counts only if its best response over all lags reaches
// BARD_TAU, and that gate is what makes the lag profile a shape descriptor rather
// than a contrast one.
constexpr int BARD_LAGS[] = {1, 2, 4};
constexpr int BARD_NL = (int)(sizeof(BARD_LAGS) / sizeof(BARD_LAGS[0]));
constexpr int BARD_SPAN = 2;                 // a lag-d test reaches +/- d, so it spans 2d
constexpr int BARD_TAU = 8;                  // 8/255 gate: below this the response is sensor noise
constexpr int BARD_MAXLAG = 4;               // == BARD_LAGS[BARD_NL - 1]; sizes the plane halo
constexpr int BARD0 = SG4 + 1;               // 30: gated-pixel count ("cover")
constexpr int BARD_S0 = BARD0 + 1;           // 31: per-lag gated response mass
constexpr int BARD_DARK = BARD_S0 + BARD_NL; // 34: max-over-lags dark mass
constexpr int BARD_LIGHT = BARD_DARK + 1;    // 35: max-over-lags light mass
// Second-order texture, from the same 3x3 neighbourhood as Sobel. With the 1-D Laws masks
// L = [1 2 1] (level), E = [-1 0 1] (edge), S = [-1 2 -1] (spot), a 3x3 mask XY is X down
// the rows times Y along the columns: LE and EL are Sobel's gx and gy, and the six others are
// summed here as squared responses, in the order of LAWS below. LAP is the 4-neighbour
// Laplacian (up + down + left + right - 4 * centre): its sum and its sum of squares.
constexpr int LAP1 = BARD_LIGHT + 1; // 36
constexpr int LAP2 = LAP1 + 1;       // 37
constexpr int LAWS0 = LAP2 + 1;      // 38..43
enum
{
  LAWS_EE,
  LAWS_SS,
  LAWS_LS,
  LAWS_SL,
  LAWS_ES,
  LAWS_SE,
  NLAWS
};
constexpr int NSUM = LAWS0 + NLAWS;    // 44 int64 sums per cell per channel
constexpr int BARD_NF = 4 + BARD_NL;   // cover, spec*NL, peak, peaked, bal
constexpr int BARD_NMAP = 3 + BARD_NL; // dense maps: cover, resp*NL, dark, light
// The sliding window of planar rows block() reads. Every row of the cell row in flight plus
// a MAXLAG halo each side stays resident (wrows_ rows, set at configure time; image row y
// lives in slot y - wbase, the rows in order: see accumulate_band(), which slides them, and
// the rows past an image edge are copies of the edge row). Each phase plane carries slack at
// both ends for the taps that
// reach past the row and for the last block's overrun into the next plane, whose lanes are
// masked or folded into cells nobody reads. It is not free space: the window wants to stay
// inside L1/L2. The left pad is half a cache line and a phase plane's stride a whole number
// of them, so that with the buffer 64-byte aligned a block's loads of its own columns never
// straddle a line (only the taps a few columns off do).
constexpr int PAD_L = 32;
constexpr int PAD_R = 8;
constexpr int PLANE_ALIGN = 64;
constexpr int PHASES_MAX = 8; // most column phases the planes are split into
constexpr int NF_S = 5;       // tensor-derived float channels
// Model-ready nonlinear descriptors, all DERIVED from the sums above (no new
// accumulators): standardized skew/kurtosis, two structure-tensor ratios, and two
// HOG histogram-shape summaries. Nonlinear (ratios/products/argmax-free peakedness)
// so a linear/shallow model cannot cheaply reconstruct them; dimensionless ones are
// illumination-invariant, which helps few-shot on-the-fly training.
constexpr int NDER = 8;
constexpr int BARDF0 = NF_S + HB + 2 + LBPB + NDER; // 34: start of the bard block
// Texture block: variance of the Laplacian, that over the intensity variance (a focus measure
// that does not reward contrast), the six Laws energies, and vertical-vs-horizontal line energy.
constexpr int TEXF0 = BARDF0 + BARD_NF; // 41
constexpr int NTEX = 2 + NLAWS + 1;
constexpr int NF = TEXF0 + NTEX; // 50 derived float32 per cell per channel
// Channels block() carries through a block together, so that a block's cells are complete
// for every channel at once and can be derived on the spot (cell_row). More channels than
// this (hyperspectral) go one at a time, so the accumulators stay in L1 however many there are.
constexpr int GMAX = 8;
// Per-level cross-cell summary of every derived feature, folded into the derive
// loop for free (same traversal, no extra pass). For each (channel, feature) we
// reduce that feature's value over the level's cells into NST stats, in this
// fixed order: [0]=min [1]=max [2]=mean [3]=std. Computed for the float feature
// block (NF) and the moments (NMOM), per pyramid level (not the 1-cell global).
constexpr int NST = 4;
// One channel's slice of a `maps` row: the float features, then the moments rounded to float32.
// derive writes that final layout, so Python hands the arrays over without reassembling them.
constexpr int NMAP = NF + NMOM;
// Perceptual hashes (imagehash-compatible), one 64-bit hash per channel. Read from a
// dedicated HN x HN grid of per-channel value sums accumulated in the SAME pass (one
// add/pixel/channel, no extra traversal): aHash/wHash threshold an 8x8 mean grid
// (HBLK x HBLK block-pool of it) by its mean/median; pHash takes the low-freq 8x8 of a
// 2D DCT-II of the full 32x32 mean grid. Independent of the feature pyramid grid.
constexpr int HN = 32; // hash mean-grid resolution (== pHash DCT input side)
constexpr int HS = 8;  // hash side: HS*HS = 64 bits
constexpr int HS2 = HS * HS;
constexpr int NHASH = 3;      // aHash, wHash, pHash
constexpr int HBLK = HN / HS; // 4: aHash/wHash pool HBLK x HBLK cells of the HN grid
constexpr int64_t RAYSCALE = 1 << 14;
constexpr double PI = 3.14159265358979323846;

inline int clampi(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

namespace hn = hwy::HWY_NAMESPACE;
using D8 = hn::ScalableTag<uint8_t>; // a block: one uint8 lane per column
using DI8 = hn::Repartition<int8_t, D8>;
using D16 = hn::Repartition<int16_t, D8>; // half a block
using DU16 = hn::Repartition<uint16_t, D8>;
using D32 = hn::Repartition<int32_t, D8>; // a quarter block, or half a block of column pairs
using DU32 = hn::Repartition<uint32_t, D8>;
using D64 = hn::Repartition<int64_t, D8>;
using DU64 = hn::Repartition<uint64_t, D8>;
using V8 = hn::VFromD<D8>;
using V16 = hn::VFromD<D16>;
using V32 = hn::VFromD<D32>;
using V64 = hn::VFromD<D64>;
constexpr D8 d8;
constexpr DI8 di8;
constexpr D16 d16;
constexpr DU16 du16;
constexpr D32 d32;
constexpr DU32 du32;
constexpr D64 d64;
constexpr DU64 du64;

// The sums block() produces per cell, in fold order. To add one: add an entry next to the
// sums of the same lane geometry, accumulate it in block(), and map it to its output slot in
// cell_row()'s flush -- the folds in between walk the enum.
//   PAIR: pmaddwd outputs, one int32 lane per two columns, per half block. The moments are
//         those of w = v - KBIAS (w^2 <= 2^14 keeps every product inside pmaddwd's int16
//         inputs); the flush shifts them back to raw power sums, exactly.
//   T4:   sum w^4, one int64 lane per four columns, per half block.
//   SG4:  sum |grad|^4, one int64 lane per two columns, per quarter block.
//   HOG:  the ray tests, one int32 lane per column, per quarter block. The bin-boundary rays
//         come in mirror pairs (i, HB-i) across the vertical, and a gradient left of it
//         passes ray HB-i exactly when its reflection fails ray i, so block() tests every
//         gradient's reflection against the HRAYS low rays only: per low ray i, S_i = sum of
//         |grad|^2 by the test's sign, D_i = the same by the test's sign times the side (+1
//         right of the vertical), and once E = the energy by the side alone. With the total
//         energy T = SXX + SYY, the energy past ray i is (S_i + D_i + 3T - E) / 4 and past
//         ray HB-i is (T - E - S_i + D_i) / 4 (hog_cum4), and the ray tests are nested, so
//         bin b = cum[b-1] - cum[b].
//   U8:   counts, one uint8 lane per column over the whole block.
//   BARD: the bar detector's gated maps, one uint16 lane per two columns over the whole block.
// clang-format off
enum
{
  K_SXX, K_SYY, K_T2, K_LAP2, K_LAWS, K_NUNSIGNED = K_LAWS + NLAWS, // squares first,
  K_SXY = K_NUNSIGNED, K_T1, K_T3, K_LAP1, K_NPAIR,                 // then what can go negative
  K_T4 = K_NPAIR, K_SG4, K_HOG,
  K_NHOG = 2 * HRAYS + 1, K_U8 = K_HOG + K_NHOG, K_MAX = K_U8, K_MIN, K_LBP, // LBP bins 0..LBPB-2; the last is the rest
  K_BARD = K_LBP + LBPB - 1,
  K_N = K_BARD + BARD_NMAP,
  K_NU8 = K_BARD - K_U8
};
// clang-format on
// A pair of uint8 map values is up to 510 a row, which caps a block at 128 sampled rows
// between folds; every other sum has headroom beyond that (a pair of |grad|^2 is 2^22 a row).
constexpr int KROWS = 128;
// block()'s column taps: the 3x3 neighbourhood's left, mid, right, then each lag's left, right.
constexpr int KTAP = 3 + 2 * BARD_NL;
constexpr int KBIAS = 128; // block() takes moments of v - KBIAS

struct Acc // one block of one channel
{
  V32 pair[2][K_NPAIR], hog[4][K_NHOG];
  V64 t4[2], sg4[4];
  V8 u8[K_NU8];
  hn::VFromD<DU16> bard[BARD_NMAP];
};

// The bar detector (see BARD_LAGS) on one block of one sampled row. `t` holds each lag's
// left/right/up/down tap bases. Every tap is a saturating subtract, so dark/light need no sign
// handling and nothing widens past uint8:
//   min(max(l-c, 0), max(r-c, 0)) == SatSub(min(l, r), c), and light is SatSub(c, max(l, r));
// combining the two axes with max distributes the same way, so each tap PAIR is reduced first
// and the subtract happens once per polarity. A lag contributes on an axis only where both its
// taps are inside the image: `vok` has bit j set when lag j's rows are, and `valid` (null for a
// block clear of the row ends) is 255 where its columns are. Forcing the dark base to 0 and the
// light base to 255 there zeroes that axis alone -- SatSub(0, c) == SatSub(c, 255) == 0. `hok`
// drops a lag on BOTH axes when its span exceeds the smaller image side. A pixel counts only
// if its best response over all lags reaches BARD_TAU; peak == max(dmax, lmax), so that gate
// is one compare. `mesh` is the sampling mesh as 0 / 1 bytes, 1 wherever every lane is sampled.
// Returned: the gate as 0 / 1 bytes and the maps (cover is the gate itself), for the caller to pair
// up and add. `rb` is the centre row at the block's first lane, `tap` the column taps (KTAP), `rs`
// the window's row stride (a lag's rows are lag * rs from the centre row). VALID and ALL (every
// lag's rows and columns are in the image: the usual row) are known per block and per run of
// rows, so the branches below are resolved outside the row loop.
using VU16 = hn::VFromD<DU16>;

struct BardRow
{
  V8 keep, map[BARD_NMAP - 1];
};
struct BardSums // the gated maps' sums over a block's rows (below 2^16, as uint16)
{
  V16 b[BARD_NMAP];
};

template <bool VALID, bool ALL, bool UNIT>
HWY_INLINE BardRow bard(const uint8_t *IF_RESTRICT rb, const int64_t *IF_RESTRICT tap, int64_t rs,
                        V8 c, V8 mesh, const uint8_t *IF_RESTRICT valid, size_t vstride, int hok,
                        int vok)
{
  static_assert(BARD_LAGS[0] == 1, "lag 1's column taps are the 3x3 neighbours'");
  static_assert(BARD_NL == 3 && BARD_LAGS[1] == 2 && BARD_LAGS[2] == 4,
                "the row taps index the centre column by the lag: one scaled index per lag");
  // the centre column: the row taps go up and down it. UNIT: the column taps are the lags
  // themselves (one phase plane), so every address is the row pointer plus a constant or
  // plus the row stride scaled by a lag, and nothing is looked up per row
  const uint8_t *IF_RESTRICT m = UNIT ? rb : rb + tap[1];
  V8 dmax, lmax, resp[BARD_NL];
  for (size_t j = 0; j < BARD_NL; ++j)
  {
    V8 dk = hn::Zero(d8), lt = hn::Zero(d8);
    if (ALL || ((hok >> j) & 1))
    {
      const int64_t cl = UNIT ? -(int64_t)BARD_LAGS[j] : tap[j ? 3 + 2 * j : 0],
                    cr = UNIT ? (int64_t)BARD_LAGS[j] : tap[j ? 4 + 2 * j : 2];
      const V8 l = hn::LoadU(d8, rb + cl), r = hn::LoadU(d8, rb + cr);
      V8 lo = hn::Min(l, r), hi = hn::Max(l, r);
      if (VALID)
      {
        const V8 vm = hn::LoadU(d8, valid + j * vstride);
        lo = hn::And(lo, vm);
        hi = hn::Or(hi, hn::Not(vm));
      }
      if (ALL || ((vok >> j) & 1))
      {
        const V8 u = hn::LoadU(d8, m - BARD_LAGS[j] * rs), v = hn::LoadU(d8, m + BARD_LAGS[j] * rs);
        lo = hn::Max(lo, hn::Min(u, v));
        hi = hn::Min(hi, hn::Max(u, v));
      }
      dk = hn::SaturatedSub(lo, c);
      lt = hn::SaturatedSub(c, hi);
    }
    resp[j] = hn::Max(dk, lt);
    dmax = j ? hn::Max(dmax, dk) : dk; // (the first lag's are the maxima so far)
    lmax = j ? hn::Max(lmax, lt) : lt;
  }
  const V8 peak = hn::Max(dmax, lmax);
  BardRow o;
  // peak >= TAU as 0 / 1 (a saturating subtract leaves 1 or more exactly then), and the mesh
  o.keep = hn::Min(hn::SaturatedSub(peak, hn::Set(d8, (uint8_t)(BARD_TAU - 1))), mesh);
  for (int j = 0; j < BARD_NL; ++j) o.map[j] = resp[j];
  o.map[BARD_NL] = dmax;
  o.map[BARD_NL + 1] = lmax;
  return o;
}

// A chunk of a cell row's sampled rows, as block() walks it: what is constant over the chunk
// (set up once per chunk by cell_row()) and what is per channel and block (set per call).
// Each row's three 1-D responses D = r - l (mask E), S = l + r - 2m (mask S, negated) and
// L = l + 2m + r = S + 4m, plus its centre m, are made once per distinct image row of a
// sub-chunk and kept in `resp`: at stride 1 every image row serves three sampled rows (above,
// centre, below), so the promotions and the 1-D work are done once instead of three times.
// Sampled row q's three rows are the slots a * (q - qs) + role, a = min(stride, 3): a row two
// sampled rows share is then one slot (at stride 1 a row's above row is the centre of the one
// before; at stride 2 its below row), and a row's slots are `rstep` elements from the next
// row's. `rdist` holds the slots' plane offsets.
constexpr int NRESP = 4; // D, S, L, M

struct Chunk
{
  size_t nr = 0;                    // sampled rows in the chunk
  int64_t row0 = 0;                 // plane offset of sampled row 0 (its centre row), and
  int64_t rowstep = 0, rs = 0;      // from a sampled row to the next, from a row to the next
  const int *vok = nullptr;         // [q]: bard lags whose rows are inside the image
  bool allok = false;               // every lag fits the image (hok is all ones)
  size_t rstep = 0;                 // elements of `resp` from a sampled row's slots to the next's
  const int64_t *rdist = nullptr;   // [sub * rdstride + slot]: a distinct row's plane offset
  const int *rdn = nullptr;         // [sub]: distinct rows of the sub-chunk
  size_t rdstride = 0, subrows = 0; // ... and the sampled rows a sub-chunk holds
  bool whole = true;                // a cell is narrower than half a block: the HOG quarters
                                    // must keep the pixels in order (see block())
  const int64_t *tapcol = nullptr;  // the column taps (KTAP)
  const uint8_t *base = nullptr;    // this channel's planes, at the block's first lane
  int16_t *resp = nullptr;          // [slot][NRESP][half][Lanes(d16)]
  int32_t *hbuf = nullptr;          // [q - qs][half][z0 z1][Lanes(d32)]
  bool inlane = false;              // the block is one cell: the 16-bit halves may hold the
                                    // pixels in interleave order (see block())
  bool rows1 = false;               // stride 1: consecutive slots are consecutive image rows
};
constexpr int RESP_STRIDE = NRESP * 2; // a slot's responses, in vectors of Lanes(d16)

// The bar detector phase of a block: its gated maps paired up through one multiply-add by
// the 0 / 1 gate (`mesh1` is the mesh as 0 / 1) into `out`. A function of its own, not
// inlined: in block() its row loop had to share the registers with the phases around it and
// spilled a quarter of its instructions.
template <bool UNIT>
HWY_NOINLINE void bard_phase(const Chunk &C, int64_t tm, V8 mesh1, const uint8_t *IF_RESTRICT valid,
                             size_t vstride, int hok, VU16 *IF_RESTRICT out);

// The first 64-byte-aligned element of a vector (sized with 64 bytes of slack): the vector
// loads of the 16-bit phases then never straddle a cache line.
template <class T>
T *aligned64(std::vector<T> &v)
{
  auto p = reinterpret_cast<uintptr_t>(v.data());
  return v.data() + ((64 - p % 64) % 64) / sizeof(T);
}

// One block of columns of one channel, down all the sampled rows of a chunk. Replicate padding
// is already in the planes, so there is no border logic here. MASKED zeroes the off-mesh lanes
// (mask: 255 on the sampling mesh) before anything pairs up columns; the uint8 counts are per
// column, so they are masked once, at the fold.
//
// The work goes in phases, each a loop down the rows with its own few accumulators, so that
// they stay in registers for the whole chunk instead of going through memory on every row
// (the whole set is about 76 vectors, far more than any register file): bytes first (extrema
// and LBP, then the bar detector), then, a sub-chunk of rows at a time, the row responses into
// `resp`, the Laws energies and the tensor (which also write the HOG inputs to `hbuf`), the
// Laplacian and the moments, and the HOG ray tests a quarter block at a time. Every sum is an
// integer, so the order of the additions changes nothing.
template <bool MASKED, bool UNIT>
HWY_INLINE void block(const Chunk &C, const uint8_t *IF_RESTRICT mask,
                      const uint8_t *IF_RESTRICT valid, size_t vstride, int hok,
                      const V16 *IF_RESTRICT rays, Acc &A)
{
  const size_t L16 = hn::Lanes(d16), L32 = hn::Lanes(d32);
  const V8 one = hn::Set(d8, 1), m8 = MASKED ? hn::LoadU(d8, mask) : hn::Set(d8, 255);
  // the 3x3 column taps: UNIT (one phase plane) makes them the constants -1, 0, 1
  const int64_t tl = UNIT ? -1 : C.tapcol[0], tm = UNIT ? 0 : C.tapcol[1],
                tr = UNIT ? 1 : C.tapcol[2];
  const int64_t rs = C.rs; // the rows walk down the window: row q's centre is row0 + q steps
  {
    // LBP^riu2: sign tests (0 / -1) in circular order NW N NE E SE S SW W, so their sum is
    // -popcount and the sum of neighbouring XORs is -transitions. Uniform (<= 2 transitions)
    // votes bin popcount; the rest vote nothing and are recovered at flush as count - sum.
    // The strict 8-neighbour extrema ride on the same tests: a local maximum is a pixel no
    // neighbour reaches, which is exactly bin 0 (popcount 0 is uniform); a local minimum is
    // one below the neighbours' minimum, which the ring takes on the way
    // (c < min  <=>  SatSub(min, c) != 0).
    // The bins are counted in nibbles: five byte accumulators hold bins (0, 1), (2, 3), ...
    // as low / high nibble counts, each row's vote a byte shuffle of the bin index (bin 9,
    // the non-uniform codes, votes nothing), and they empty into the byte counts every
    // LBP_NIB rows, before a nibble can overflow -- fewer accumulators and no compare per bin.
    constexpr int LBP_NIB = 15, NPAIR = (LBPB - 1 + 1) / 2;
    V8 lbp[LBPB - 1], umin = hn::Zero(d8), tbl[NPAIR];
    for (int j = 0; j < LBPB - 1; ++j) lbp[j] = hn::Zero(d8);
    for (int k = 0; k < NPAIR; ++k)
    {
      uint8_t t[16] = {};
      t[2 * k] = 1;
      if (2 * k + 1 < LBPB - 1) t[2 * k + 1] = 16;
      tbl[k] = hn::Dup128VecFromValues(d8, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8],
                                       t[9], t[10], t[11], t[12], t[13], t[14], t[15]);
    }
    const uint8_t *IF_RESTRICT rb = C.base + C.row0;
    for (size_t q0 = 0; q0 < C.nr; q0 += LBP_NIB)
    {
      V8 nib[NPAIR];
      for (int k = 0; k < NPAIR; ++k) nib[k] = hn::Zero(d8);
      for (size_t q = q0; q < std::min(C.nr, q0 + LBP_NIB); ++q, rb += C.rowstep)
      {
        const uint8_t *IF_RESTRICT ra = rb - rs, *IF_RESTRICT re = rb + rs;
        const V8 c = hn::LoadU(d8, rb + tm);
        V8 mn = hn::LoadU(d8, ra + tl);
        auto ge = [&](V8 nb) { return hn::VecFromMask(d8, hn::Eq(hn::Max(nb, c), nb)); };
        const V8 s0 = ge(mn);
        V8 prev = s0, npc = s0, ntr = hn::Zero(d8);
        auto next = [&](const uint8_t *p) // the ring, a neighbour at a time: few tests stay alive
        {
          const V8 nb = hn::LoadU(d8, p), si = ge(nb);
          mn = hn::Min(mn, nb);
          npc = hn::Add(npc, si);
          ntr = hn::Add(ntr, hn::Xor(prev, si));
          prev = si;
        };
        next(ra + tm), next(ra + tr), next(rb + tr), next(re + tr), next(re + tm), next(re + tl),
            next(rb + tl);
        ntr = hn::Add(ntr, hn::Xor(prev, s0));
        umin = hn::Add(umin, hn::Min(hn::SaturatedSub(mn, c), one));
        const auto uniform = hn::Gt(hn::BitCast(di8, ntr), hn::Set(di8, -3));
        const V8 bin = hn::IfThenElse(hn::RebindMask(d8, uniform), hn::Sub(hn::Zero(d8), npc),
                                      hn::Set(d8, (uint8_t)(LBPB - 1)));
        for (int k = 0; k < NPAIR; ++k) nib[k] = hn::Add(nib[k], hn::TableLookupBytes(tbl[k], bin));
      }
      for (int k = 0; k < NPAIR; ++k)
      {
        lbp[2 * k] = hn::Add(lbp[2 * k], hn::And(nib[k], hn::Set(d8, 0x0F)));
        if (2 * k + 1 < LBPB - 1)
          lbp[2 * k + 1] = hn::Add(lbp[2 * k + 1], hn::ShiftRight<4>(nib[k]));
      }
    }
    for (int j = 0; j < LBPB - 1; ++j) A.u8[K_LBP - K_U8 + j] = lbp[j];
    A.u8[K_MAX - K_U8] = lbp[0];
    A.u8[K_MIN - K_U8] = umin;
  }
  bard_phase<UNIT>(C, tm, hn::And(m8, one), valid, vstride, hok, A.bard);
  const V16 ones = hn::Set(d16, 1), zero = hn::Zero(d16);
  // Bytes widen to the 16-bit halves by promotion (the first 16 pixels, then the rest) or,
  // where the block is one cell and the lanes' order is nobody's business, by interleaving
  // with zero, which is one shuffle on two pipes instead of a promotion and an extract on one.
  // The masks widen the same way.
  auto wide = [&](auto kInlane, int h, V8 v) IF_LAMBDA_INLINE
  {
    if constexpr (decltype(kInlane)::value)
      return hn::BitCast(d16, h ? hn::InterleaveUpper(d8, v, hn::Zero(d8))
                                : hn::InterleaveLower(d8, v, hn::Zero(d8)));
    else
      return hn::BitCast(d16, h ? hn::PromoteUpperTo(du16, v) : hn::PromoteLowerTo(du16, v));
  };
  auto wmask = [&](int h)
  {
    if (!MASKED) return zero;
    return C.inlane ? hn::BitCast(d16, h ? hn::InterleaveUpper(d8, m8, m8)
                                         : hn::InterleaveLower(d8, m8, m8))
                    : hn::PromoteUpperTo(d16, hn::BitCast(di8, m8));
  };
  const V16 m16[2] = {MASKED ? (C.inlane ? wmask(0) : hn::PromoteLowerTo(d16, hn::BitCast(di8, m8)))
                             : zero,
                      wmask(1)};
  auto resp = [&](const int16_t *rp, int i, int t) // rp: a sampled row's slots of one half
  { return hn::LoadU(d16, rp + (i * RESP_STRIDE + t * 2) * L16); };
  auto hin = [&](size_t q, int h, int i) { return C.hbuf + (((q * 2) + h) * 2 + i) * L32; };
  struct Gram // the stride-1 energy phase's chunk-wide sums (see there)
  {
    V32 a[10];
  };
  Gram gram[2];
  V32 head[2][9];
  for (size_t sub = 0, qs = 0; qs < C.nr; ++sub, qs += C.subrows)
  {
    const size_t qe = std::min(C.nr, qs + C.subrows);
    // a sub-chunk's accumulators start from zero and continue from what the chunk has so far
    auto init = [&](auto &acc, const auto &sofar)
    { acc = sub ? sofar : hn::Zero(hn::DFromV<std::remove_reference_t<decltype(acc)>>()); };
    // HOG inputs: (|gx|, gy * sgn(gx)), sgn(0) = 1. The absolute values are the gradient
    // reflected into the first quadrant; the sign of the second lane (that of gx * gy) says
    // whether the gradient, folded to the upper half-plane, points left of the vertical.
    // Pairs, one per 32-bit lane: which pixel lands in which lane only matters when a quarter
    // block spans more than one cell; otherwise the cheaper in-block interleave, whose lanes
    // hold the pixels of both quarters out of order, does
    auto hog_in = [&](auto kWhole, size_t q, int h, V16 gx, V16 gy) IF_LAMBDA_INLINE
    {
      const V16 ax = hn::Abs(gx), sy = hn::IfNegativeThenNegOrUndefIfZero(hn::Or(gx, ones), gy);
      V16 z0, z1;
      if constexpr (decltype(kWhole)::value)
        z0 = hn::InterleaveWholeLower(d16, ax, sy), z1 = hn::InterleaveWholeUpper(d16, ax, sy);
      else
        z0 = hn::InterleaveLower(d16, ax, sy), z1 = hn::InterleaveUpper(d16, ax, sy);
      hn::StoreU(hn::BitCast(d32, z0), d32, hin(q - qs, h, 0));
      hn::StoreU(hn::BitCast(d32, z1), d32, hin(q - qs, h, 1));
    };
    const int64_t *IF_RESTRICT dist = C.rdist + sub * C.rdstride;
    auto responses = [&](auto kInlane) IF_LAMBDA_INLINE
    {
      for (int j = 0; j < C.rdn[sub]; ++j) // the distinct rows' responses, both halves
      {
        const uint8_t *IF_RESTRICT row = C.base + dist[j];
        const V8 l = hn::LoadU(d8, row + tl), m = hn::LoadU(d8, row + tm),
                 r = hn::LoadU(d8, row + tr);
        for (int h = 0; h < 2; ++h)
        {
          const V16 wl = wide(kInlane, h, l), wm = wide(kInlane, h, m), wr = wide(kInlane, h, r),
                    lr = hn::Add(wl, wr), m2 = hn::Add(wm, wm);
          int16_t *IF_RESTRICT o = C.resp + ((size_t)j * RESP_STRIDE + h) * L16;
          hn::StoreU(hn::Sub(wr, wl), d16, o);
          hn::StoreU(hn::Sub(lr, m2), d16, o + 2 * L16);
          hn::StoreU(hn::Add(lr, m2), d16, o + 4 * L16);
          hn::StoreU(wm, d16, o + 6 * L16);
        }
      }
    };
    C.inlane ? responses(std::true_type()) : responses(std::false_type());
    for (int h = 0; h < 2; ++h) // the int16 half blocks: Laws energies, the tensor, HOG inputs
    {
      V32 *IF_RESTRICT P = A.pair[h];
      auto on_mesh = [&](V16 v) { return MASKED ? hn::And(v, m16[h]) : v; };
      if constexpr (!MASKED)
        if (C.rows1)
        {
          // Stride 1: the slots are consecutive image rows, so the energy of each 3-tap
          // vertical filter over the chunk is a combination of three sums per column X of
          // responses over its rows r: the squares P = Σ X(r)², the products Q = Σ X(r) X(r+1)
          // and R = Σ X(r) X(r+2), the sums over the chunk's sampled rows, plus the products of
          // its first two rows (P0, P1, Q0) and the two past its last (Pn, Pn1, Qn): with
          // B = 2P − P0 − P1 + Pn + Pn1, M = 4 (P − P0 + Pn) and Qc = 2Q − Q0 + Qn, the E,
          // S and L filters' energies are B − 2R, B + M + 2R − 4Qc and B + M + 2R + 4Qc. Nine
          // products a row instead of nine squares of fourteen sums; the corrections cost once
          // per chunk. Every sum wraps in 32 bits and every result is in range, so it is exact.
          auto prod = [&](V16 a, V16 b) { return hn::WidenMulPairwiseAdd(d32, a, b); };
          const int16_t *IF_RESTRICT rp = C.resp + h * L16;
          if (sub == 0)
          {
            for (int t = 0; t < 3; ++t)
            {
              const V16 x0 = resp(rp, 0, t), x1 = resp(rp, 1, t);
              head[h][3 * t] = prod(x0, x0), head[h][3 * t + 1] = prod(x1, x1);
              head[h][3 * t + 2] = prod(x0, x1);
            }
            for (auto &v : gram[h].a) v = hn::Zero(d32);
          }
          // the sums (P, Q, R of D, of S, of L, then Σ gx gy) go through the loop by value: by
          // reference they would have a home in memory that the loads may alias, and GCC
          // then stored them every row
          auto grows = [&](auto kWhole, Gram g) IF_LAMBDA_INLINE
          {
            V32 *IF_RESTRICT a = g.a;
            for (size_t q = qs; q < qe; ++q, rp += C.rstep)
            {
              const V16 d0 = resp(rp, 0, 0), d1 = resp(rp, 1, 0), d2 = resp(rp, 2, 0);
              a[0] = hn::Add(a[0], prod(d0, d0)), a[1] = hn::Add(a[1], prod(d0, d1));
              a[2] = hn::Add(a[2], prod(d0, d2));
              const V16 s0 = resp(rp, 0, 1), s1 = resp(rp, 1, 1), s2 = resp(rp, 2, 1);
              a[3] = hn::Add(a[3], prod(s0, s0)), a[4] = hn::Add(a[4], prod(s0, s1));
              a[5] = hn::Add(a[5], prod(s0, s2));
              const V16 l0 = resp(rp, 0, 2), l1 = resp(rp, 1, 2), l2 = resp(rp, 2, 2);
              a[6] = hn::Add(a[6], prod(l0, l0)), a[7] = hn::Add(a[7], prod(l0, l1));
              a[8] = hn::Add(a[8], prod(l0, l2));
              const V16 gx = hn::Add(hn::Add(d0, d2), hn::Add(d1, d1)), gy = hn::Sub(l2, l0);
              a[9] = hn::Add(a[9], prod(gx, gy));
              hog_in(kWhole, q, h, gx, gy);
            }
            return g;
          };
          gram[h] = C.whole ? grows(std::true_type(), gram[h]) : grows(std::false_type(), gram[h]);
          if (qe < C.nr) continue;
          // the chunk's last sub-chunk: rp is at the row past its sampled rows
          const V32 *IF_RESTRICT a = gram[h].a;
          auto energies = [&](int t, V32 &e, V32 &s2, V32 *l)
          {
            const V16 xn = resp(rp, 0, t), xn1 = resp(rp, 1, t);
            const V32 pn = prod(xn, xn), pn1 = prod(xn1, xn1), qn = prod(xn, xn1);
            const V32 *IF_RESTRICT hd = head[h] + 3 * t;
            const V32 pf = a[3 * t], qf = a[3 * t + 1], rf2 = hn::Add(a[3 * t + 2], a[3 * t + 2]);
            const V32 b =
                hn::Add(hn::Sub(hn::Add(pf, pf), hn::Add(hd[0], hd[1])), hn::Add(pn, pn1));
            const V32 m = hn::ShiftLeft<2>(hn::Add(hn::Sub(pf, hd[0]), pn));
            const V32 qc4 = hn::ShiftLeft<2>(hn::Add(hn::Sub(hn::Add(qf, qf), hd[2]), qn));
            const V32 bm = hn::Add(hn::Add(b, m), rf2);
            e = hn::Sub(b, rf2);
            s2 = hn::Sub(bm, qc4);
            if (l) *l = hn::Add(bm, qc4);
          };
          energies(0, P[K_LAWS + LAWS_EE], P[K_LAWS + LAWS_SE], &P[K_SXX]);
          energies(1, P[K_LAWS + LAWS_ES], P[K_LAWS + LAWS_SS], &P[K_LAWS + LAWS_LS]);
          energies(2, P[K_SYY], P[K_LAWS + LAWS_SL], nullptr);
          P[K_SXY] = a[9];
          continue;
        }
      V32 ee, se, es, ls2, ss, sl, sxx, syy, sxy;
      init(ee, P[K_LAWS + LAWS_EE]), init(se, P[K_LAWS + LAWS_SE]), init(es, P[K_LAWS + LAWS_ES]);
      init(ls2, P[K_LAWS + LAWS_LS]), init(ss, P[K_LAWS + LAWS_SS]), init(sl, P[K_LAWS + LAWS_SL]);
      init(sxx, P[K_SXX]), init(syy, P[K_SYY]), init(sxy, P[K_SXY]);
      // a response is squared into its sum as soon as it exists, which keeps few of them alive
      auto energy = [&](V32 &acc, V16 v)
      {
        v = on_mesh(v);
        acc = hn::Add(acc, hn::WidenMulPairwiseAdd(d32, v, v));
      };
      auto rows = [&](auto kWhole) IF_LAMBDA_INLINE // the loop, once per pixel order
      {
        const int16_t *IF_RESTRICT rp = C.resp + h * L16;
        for (size_t q = qs; q < qe; ++q, rp += C.rstep)
        {
          // A 3x3 Laws response is one 1-D mask down a column of the row responses -- Sobel
          // included (gx = L.D, gy = E.L) -- up to a sign that the squares drop.
          const V16 Da = resp(rp, 0, 0), Db = resp(rp, 1, 0), De = resp(rp, 2, 0);
          const V16 Dae = hn::Add(Da, De), Db2 = hn::Add(Db, Db), gx0 = hn::Add(Dae, Db2);
          energy(ee, hn::Sub(De, Da));
          energy(se, hn::Sub(Dae, Db2)); // Da + De - 2 Db == gx0 - 4 Db
          const V16 Sa = resp(rp, 0, 1), Sb = resp(rp, 1, 1), Se = resp(rp, 2, 1);
          const V16 Sae = hn::Add(Sa, Se), Sb2 = hn::Add(Sb, Sb), ls = hn::Add(Sae, Sb2);
          energy(es, hn::Sub(Se, Sa));
          energy(ls2, ls);
          energy(ss, hn::Sub(Sae, Sb2)); // Sa + Se - 2 Sb == ls - 4 Sb
          const V16 La = resp(rp, 0, 2), Lb = resp(rp, 1, 2), Le = resp(rp, 2, 2);
          energy(sl, hn::Sub(hn::Add(La, Le), hn::Add(Lb, Lb)));
          const V16 gx = on_mesh(gx0), gy = on_mesh(hn::Sub(Le, La));
          sxx = hn::Add(sxx, hn::WidenMulPairwiseAdd(d32, gx, gx));
          syy = hn::Add(syy, hn::WidenMulPairwiseAdd(d32, gy, gy));
          sxy = hn::Add(sxy, hn::WidenMulPairwiseAdd(d32, gx, gy));
          hog_in(kWhole, q, h, gx, gy);
        }
      };
      C.whole ? rows(std::true_type()) : rows(std::false_type());
      P[K_LAWS + LAWS_EE] = ee, P[K_LAWS + LAWS_SE] = se, P[K_LAWS + LAWS_ES] = es;
      P[K_LAWS + LAWS_LS] = ls2, P[K_LAWS + LAWS_SS] = ss, P[K_LAWS + LAWS_SL] = sl;
      P[K_SXX] = sxx, P[K_SYY] = syy, P[K_SXY] = sxy;
    }
    for (int h = 0; h < 2; ++h) // the Laplacian and the moments of w = v - KBIAS
    {
      V32 *IF_RESTRICT P = A.pair[h];
      auto on_mesh = [&](V16 v) { return MASKED ? hn::And(v, m16[h]) : v; };
      V32 lap1, lap2, t1, t2, t3;
      V64 t4;
      init(lap1, P[K_LAP1]), init(lap2, P[K_LAP2]), init(t1, P[K_T1]), init(t2, P[K_T2]);
      // The two linear sums ride in 16-bit lanes for the sub-chunk (|lap| <= 1020 and |w| <= 128
      // over at most 32 rows stay within int16) and pair up into the int32 sums once
      V16 lap16 = zero, w16 = zero;
      static_assert(1020 * 32 <= 32767, "a sub-chunk's Laplacian sum fits int16");
      init(t3, P[K_T3]), init(t4, A.t4[h]);
      const int16_t *IF_RESTRICT rp = C.resp + h * L16;
      // a row's sums but the fourth power's pair sums, returned: those are 2^29 a pair, so two
      // rows' add in 32 bits and widen together (they cannot go negative: zero-extended)
      auto row = [&]() IF_LAMBDA_INLINE
      {
        const V16 Sb = resp(rp, 1, 1), Ma = resp(rp, 0, 3), Mb = resp(rp, 1, 3),
                  Me = resp(rp, 2, 3);
        const V16 lap = on_mesh(
            hn::Sub(hn::Add(Sb, hn::Add(Ma, Me)), hn::Add(Mb, Mb))); // up+down+left+right-4c
        lap16 = hn::Add(lap16, lap);
        lap2 = hn::Add(lap2, hn::WidenMulPairwiseAdd(d32, lap, lap));
        const V16 wc = on_mesh(hn::Sub(Mb, hn::Set(d16, KBIAS))), w2 = hn::Mul(wc, wc);
        w16 = hn::Add(w16, wc);
        t2 = hn::Add(t2, hn::WidenMulPairwiseAdd(d32, wc, wc));
        t3 = hn::Add(t3, hn::WidenMulPairwiseAdd(d32, w2, wc));
        rp += C.rstep;
        return hn::BitCast(du32, hn::WidenMulPairwiseAdd(d32, w2, w2));
      };
      auto widen = [&](hn::VFromD<DU32> p) { t4 = hn::Add(t4, hn::BitCast(d64, hn::SumsOf2(p))); };
      size_t q = qs;
      for (; q + 1 < qe; q += 2)
      {
        const auto p0 = row();
        widen(hn::Add(p0, row()));
      }
      if (q < qe) widen(row());
      lap1 = hn::Add(lap1, hn::WidenMulPairwiseAdd(d32, lap16, ones));
      t1 = hn::Add(t1, hn::WidenMulPairwiseAdd(d32, w16, ones));
      P[K_LAP1] = lap1, P[K_LAP2] = lap2, P[K_T1] = t1, P[K_T2] = t2, P[K_T3] = t3;
      A.t4[h] = t4;
    }
    for (int k = 0; k < 4; ++k) // the int32 quarter blocks: |grad|^4 and the HOG ray tests
    {
      V32 hog[K_NHOG];
      V64 sg4;
      for (int j = 0; j < K_NHOG; ++j) init(hog[j], A.hog[k][j]);
      init(sg4, A.sg4[k]);
      for (size_t q = qs; q < qe; ++q)
      {
        const V32 zl = hn::LoadU(d32, hin(q - qs, k >> 1, k & 1)); // negative: left of the vertical
        const V16 z = hn::BitCast(d16, zl), za = hn::Abs(z);       // (|gx|, |gy|)
        const V32 g2 = hn::WidenMulPairwiseAdd(d32, z, z); // |grad|^2: cheaper than a store + load
        sg4 = hn::Add(sg4, hn::BitCast(d64, hn::Add(hn::MulEven(g2, g2), hn::MulOdd(g2, g2))));
        // A gradient left of the vertical passes every low ray, and passes a low ray's mirror
        // image exactly when its reflection (|gx|, |gy|) fails the low ray. So the low rays are
        // tested against the reflection, with + |grad|^2 past the ray (t > 0) and - |grad|^2
        // short of it (t < 0), one sign flip, into S; the same by the side into D; and the
        // energy by the side alone into E, from which the flush has the energy past every ray
        // (see K_HOG). A test is 0 only for a zero gradient (set_config checks), where
        // |grad|^2 is 0 whatever the flip makes of it.
        const V32 g2s = hn::IfNegativeThenNegOrUndefIfZero(zl, g2);
        hog[2 * HRAYS] = hn::Add(hog[2 * HRAYS], g2s);
        for (int j = 0; j < HRAYS; ++j)
        {
          const V32 t = hn::WidenMulPairwiseAdd(d32, za, rays[j]);
          hog[j] = hn::Add(hog[j], hn::IfNegativeThenNegOrUndefIfZero(t, g2));
          hog[HRAYS + j] = hn::Add(hog[HRAYS + j], hn::IfNegativeThenNegOrUndefIfZero(t, g2s));
        }
      }
      for (int j = 0; j < K_NHOG; ++j) A.hog[k][j] = hog[j];
      A.sg4[k] = sg4;
    }
  }
}

template <bool UNIT>
HWY_NOINLINE void bard_phase(const Chunk &C, int64_t tm, V8 mesh1, const uint8_t *IF_RESTRICT valid,
                             size_t vstride, int hok, VU16 *IF_RESTRICT out)
{
  // The sums (below 2^16) go through the runs by value: by reference they would have a home
  // in memory that the byte loads may alias, and GCC then stored them every row
  auto run = [&](auto kValid, auto kAll, size_t q0, size_t q1, BardSums acc) IF_LAMBDA_INLINE
  {
    const uint8_t *IF_RESTRICT rb = C.base + C.row0 + (int64_t)q0 * C.rowstep;
    for (size_t q = q0; q < q1; ++q, rb += C.rowstep)
    {
      const BardRow r = bard<decltype(kValid)::value, decltype(kAll)::value, UNIT>(
          rb, C.tapcol, C.rs, hn::LoadU(d8, rb + tm), mesh1, valid, vstride, hok, C.vok[q]);
      const auto k = hn::BitCast(di8, r.keep);
      acc.b[0] = hn::Add(acc.b[0], hn::SatWidenMulPairwiseAdd(d16, r.keep, k)); // cover: 1 * 1
      for (int j = 1; j < BARD_NMAP; ++j)
        acc.b[j] = hn::Add(acc.b[j], hn::SatWidenMulPairwiseAdd(d16, r.map[j - 1], k));
    }
    return acc;
  };
  BardSums acc;
  for (int j = 0; j < BARD_NMAP; ++j) acc.b[j] = hn::Zero(d16);
  // The rows whose lags all fit the image are one run (vok is monotone in from each image
  // edge), walked without a test per lag; the rest, at the image's top and bottom, with the tests
  size_t q0 = 0, q1 = C.nr;
  if (C.allok)
  {
    while (q0 < q1 && C.vok[q0] != hok) ++q0;
    while (q1 > q0 && C.vok[q1 - 1] != hok) --q1;
  }
  else
    q1 = q0; // no run without a test
  if (valid)
  {
    acc = run(std::true_type(), std::false_type(), 0, q0, acc);
    acc = run(std::true_type(), std::true_type(), q0, q1, acc);
    acc = run(std::true_type(), std::false_type(), q1, C.nr, acc);
  }
  else
  {
    acc = run(std::false_type(), std::false_type(), 0, q0, acc);
    acc = run(std::false_type(), std::true_type(), q0, q1, acc);
    acc = run(std::false_type(), std::false_type(), q1, C.nr, acc);
  }
  for (int j = 0; j < BARD_NMAP; ++j) out[j] = hn::BitCast(du16, acc.b[j]);
}

// Four times the |grad|^2 past each bin-boundary ray of a cell, in ray order, from its ray
// sums (see K_HOG) through `sum(kind)` and its total |grad|^2: multiples of four, never
// negative, so a shift or a multiply by 0.25 recovers them exactly. For vectors of int64 or of
// doubles, and for scalars.
template <class T, class F>
HWY_INLINE void hog_cum4(F sum, T tot, T *IF_RESTRICT cum4)
{
  const T e = sum(K_HOG + 2 * HRAYS);
  if constexpr (std::is_same_v<T, int64_t>)
  {
    const T t3e = 3 * tot - e, te = tot - e;
    for (int i = 0; i < HRAYS; ++i)
    {
      const T s = sum(K_HOG + i), d = sum(K_HOG + HRAYS + i);
      cum4[i] = s + d + t3e;                // past low ray i
      cum4[2 * HRAYS - 1 - i] = d - s + te; // past its mirror image
    }
  }
  else
  {
    const T t3e = hn::Sub(hn::Add(tot, hn::Add(tot, tot)), e), te = hn::Sub(tot, e);
    for (int i = 0; i < HRAYS; ++i)
    {
      const T s = sum(K_HOG + i), d = sum(K_HOG + HRAYS + i);
      cum4[i] = hn::Add(hn::Add(s, d), t3e);
      cum4[2 * HRAYS - 1 - i] = hn::Add(hn::Sub(d, s), te);
    }
  }
}

// dst[i] (+)= sum of v's lanes [i*g, (i+1)*g), g a power of two; g >= Lanes sums all into
// dst[0]. `add` is false for a cell's first contribution, which is stored: the cell sums are
// then never zeroed, and most folds never read them.
template <class D>
HWY_INLINE void fold(D d, hn::VFromD<D> v, size_t g, bool add, int64_t *IF_RESTRICT dst)
{
  if constexpr (HWY_MAX_LANES_D(D) > 1)
  {
    if (g >= hn::Lanes(d)) // one cell: a horizontal sum, not a level per halving
    {
      const int64_t total = (int64_t)hn::ReduceSum(d, v);
      dst[0] = add ? dst[0] + total : total;
      return;
    }
    if (g > 1)
    {
      const hn::Half<D> dh;
      const auto even = hn::LowerHalf(dh, hn::ConcatEven(d, v, v)),
                 odd = hn::LowerHalf(dh, hn::ConcatOdd(d, v, v));
      return fold(dh, hn::Add(even, odd), g / 2, add, dst);
    }
  }
  hn::StoreU(add ? hn::Add(hn::LoadU(d, dst), v) : v, d, dst);
}
// ... of int32 lanes. Sums that cannot go negative widen by zero-extension, which is a mask
// and a shift where sign-extension is a compare and two shuffles.
template <bool SIGN_EXTEND>
HWY_INLINE void fold32(V32 v, size_t g, bool add, int64_t *IF_RESTRICT dst)
{
  if (SIGN_EXTEND)
  {
    if (g >= hn::Lanes(d32)) // one cell: the halves promote (two shuffles) and reduce
    {
      const int64_t total =
          hn::ReduceSum(d64, hn::Add(hn::PromoteLowerTo(d64, v), hn::PromoteUpperTo(d64, v)));
      dst[0] = add ? dst[0] + total : total;
      return;
    }
    fold(d64, hn::SumsOf2(v), g / 2, add, dst);
  }
  else
    fold(d64, hn::BitCast(d64, hn::SumsOf2(hn::BitCast(du32, v))), g / 2, add, dst);
}

// The sums of a one-cell block (every lane of every accumulator is the cell's), into its
// kind-contiguous K_N slots. The halves and quarters of a kind add in registers first (any two
// lanes of an accumulator sum within 32 bits at KROWS = 128, and the four quarters of a HOG sum
// within 2^30), widen once, and then four kinds reduce together: two interleave levels pair the
// lanes of four vectors into one vector of their four sums, a quarter of the shuffles of four
// horizontal sums, and store four consecutive slots at once. Lane widths other than four int64
// lanes (eight reduce to four first) take the generic fold.
constexpr bool FOLD4 = HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE && HWY_MAX_LANES_D(D64) % 4 == 0;
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
using D64x4 = hn::FixedTag<int64_t, 4>;
template <class V> // (a template, so that only the branch for this width is instantiated)
HWY_INLINE hn::VFromD<D64x4> to4(V v)
{
  if constexpr (HWY_MAX_LANES_D(D64) == 4)
    return v;
  else
  {
    const D64x4 d4;
    return hn::Add(hn::LowerHalf(d4, v), hn::UpperHalf(d4, v));
  }
}
#endif
HWY_INLINE void fold_cell(const Acc &A, bool msk, V8 mv, bool add, int64_t *IF_RESTRICT cb,
                          size_t kst) // the slots kst apart: one vector store where contiguous
{
  auto wide = [&](int t) -> V64 // kind t's lanes as int64 (not yet reduced)
  {
    if (t < K_NUNSIGNED)
    {
      const auto v = hn::Add(hn::BitCast(du32, A.pair[0][t]), hn::BitCast(du32, A.pair[1][t]));
      return hn::BitCast(d64, hn::SumsOf2(v));
    }
    if (t < K_NPAIR)
    {
      const V32 v = hn::Add(A.pair[0][t], A.pair[1][t]);
      return hn::Add(hn::PromoteLowerTo(d64, v), hn::PromoteUpperTo(d64, v));
    }
    if (t == K_T4) return hn::Add(A.t4[0], A.t4[1]);
    if (t == K_SG4) return hn::Add(hn::Add(A.sg4[0], A.sg4[1]), hn::Add(A.sg4[2], A.sg4[3]));
    if (t < K_U8)
    {
      const int j = t - K_HOG;
      const V32 v = hn::Add(hn::Add(A.hog[0][j], A.hog[1][j]), hn::Add(A.hog[2][j], A.hog[3][j]));
      return hn::Add(hn::PromoteLowerTo(d64, v), hn::PromoteUpperTo(d64, v));
    }
    if (t < K_BARD)
      return hn::BitCast(d64, hn::SumsOf8(msk ? hn::And(A.u8[t - K_U8], mv) : A.u8[t - K_U8]));
    if (t < K_N) return hn::BitCast(d64, hn::SumsOf4(A.bard[t - K_BARD]));
    return hn::Zero(d64);
  };
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
  if constexpr (FOLD4)
  {
    const D64x4 d4;
    for (int t = 0; t < K_N; t += 4)
    {
      const auto a = to4(wide(t)), b = to4(wide(t + 1)), c = to4(wide(t + 2)), e = to4(wide(t + 3));
      const auto ab = hn::Add(hn::InterleaveLower(d4, a, b), hn::InterleaveUpper(d4, a, b));
      const auto ce = hn::Add(hn::InterleaveLower(d4, c, e), hn::InterleaveUpper(d4, c, e));
      auto r = hn::Add(hn::ConcatLowerLower(d4, ce, ab), hn::ConcatUpperUpper(d4, ce, ab));
      int64_t *IF_RESTRICT dst = cb + t;
      if (kst != 1) // kinds apart (a group of several blocks' cells): a lane at a time
      {
        int64_t lane[4];
        hn::StoreU(r, d4, lane);
        for (int j = 0; j < 4 && t + j < K_N; ++j)
        {
          int64_t *IF_RESTRICT p = cb + (size_t)(t + j) * kst;
          *p = add ? *p + lane[j] : lane[j];
        }
      }
      else if (t + 4 <= K_N)
      {
        if (add) r = hn::Add(r, hn::LoadU(d4, dst));
        hn::StoreU(r, d4, dst);
      }
      else // the last kinds: only theirs are stored
      {
        const auto m = hn::FirstN(d4, (size_t)(K_N - t));
        if (add) r = hn::Add(r, hn::MaskedLoad(m, d4, dst));
        hn::BlendedStore(r, m, d4, dst);
      }
    }
    return;
  }
#endif
  for (int t = 0; t < K_N; ++t)
  {
    const int64_t total = hn::ReduceSum(d64, wide(t));
    cb[t * kst] = add ? cb[t * kst] + total : total;
  }
}

// The HOG sums of a block of several cells, 4, 8 or 16 lanes (columns) a cell. A quarter block
// holds two cells, one, or half of one; the lanes of a cell add in 32 bits first (a HOG lane
// is below 2^28 at KROWS = 128, so four of them fit), as 128-bit vectors of four lanes, and
// four cells' vectors reduce together through two interleave levels into one vector of their
// sums (see fold_cell), widened to int64 and stored to the four consecutive cells. Only where
// a vector is eight int32 lanes; other widths take the generic fold.
constexpr bool FOLDH = HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE && HWY_MAX_LANES_D(D32) == 8;
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
using D32x4 = hn::FixedTag<int32_t, 4>;
using V32x4 = hn::VFromD<D32x4>;
template <bool SIGNED = true> // the sums widen by sign or (unsigned kinds, up to 2^32) by zero
HWY_INLINE hn::VFromD<D64x4> sum4x4(V32x4 a, V32x4 b, V32x4 c, V32x4 e)
{
  const D32x4 d4;
  const hn::Repartition<int64_t, D32x4> d2;
  const auto ab = hn::Add(hn::InterleaveLower(d4, a, b), hn::InterleaveUpper(d4, a, b));
  const auto ce = hn::Add(hn::InterleaveLower(d4, c, e), hn::InterleaveUpper(d4, c, e));
  const auto lo = hn::InterleaveLower(d2, hn::BitCast(d2, ab), hn::BitCast(d2, ce)),
             hi = hn::InterleaveUpper(d2, hn::BitCast(d2, ab), hn::BitCast(d2, ce));
  const auto sum = hn::Add(hn::BitCast(d4, lo), hn::BitCast(d4, hi));
  if constexpr (SIGNED)
    return hn::PromoteTo(D64x4(), sum);
  else
    return hn::BitCast(D64x4(), hn::PromoteTo(hn::RebindToUnsigned<D64x4>(),
                                              hn::BitCast(hn::RebindToUnsigned<D32x4>(), sum)));
}
// ... and the pair sums (one int32 lane per two columns, a half block each): 2, 4 or 8 lanes
// a cell. Any two lanes sum within 32 bits; the unsigned kinds' four reach 2^32 - 2^25, which
// unsigned 32-bit arithmetic holds exactly.
template <bool SIGNED>
HWY_INLINE void fold_pair_multi(V32 v0, V32 v1, size_t g, const bool *IF_RESTRICT adds,
                                const size_t *IF_RESTRICT at, int64_t *IF_RESTRICT dst, size_t cst)
{
  const D64x4 dw;
  const D32x4 d4;
  const hn::Half<D64x4> dw2;
  auto store = [&](auto v, auto d, bool add, int64_t *IF_RESTRICT p)
  { hn::StoreU(add ? hn::Add(v, hn::LoadU(d, p)) : v, d, p); };
  auto halves = [&](V32 v) { return hn::Add(hn::LowerHalf(d4, v), hn::UpperHalf(d4, v)); };
  if (g == 4) // a cell a lane pair: four cells a half, the pairs added through a shuffle each,
  {           // then only widened
    const V32 v[2] = {v0, v1};
    for (int h = 0; h < 2; ++h)
    {
      const auto lo = hn::LowerHalf(d4, v[h]), hi = hn::UpperHalf(d4, v[h]);
      const auto p = hn::Add(hn::ConcatEven(d4, hi, lo), hn::ConcatOdd(d4, hi, lo));
      hn::VFromD<D64x4> w;
      if constexpr (SIGNED)
        w = hn::PromoteTo(dw, p);
      else
        w = hn::BitCast(dw, hn::PromoteTo(hn::RebindToUnsigned<D64x4>(),
                                          hn::BitCast(hn::RebindToUnsigned<D32x4>(), p)));
      store(w, dw, adds[2 * h], dst + at[2 * h] * cst);
    }
  }
  else if (g == 8) // a cell per 128-bit half: four cells
    store(sum4x4<SIGNED>(hn::LowerHalf(d4, v0), hn::UpperHalf(d4, v0), hn::LowerHalf(d4, v1),
                         hn::UpperHalf(d4, v1)),
          dw, adds[0], dst + at[0] * cst);
  else // g == 16: a cell a half: two cells
    store(hn::LowerHalf(dw2, sum4x4<SIGNED>(halves(v0), halves(v1), hn::Zero(d4), hn::Zero(d4))),
          dw2, adds[0], dst + at[0] * cst);
}
HWY_INLINE void fold_hog_multi(const Acc &A, size_t g, const bool *IF_RESTRICT adds,
                               const size_t *IF_RESTRICT at, int64_t *IF_RESTRICT cb, size_t kst,
                               size_t cst)
{
  const D64x4 dw;
  const D32x4 d4;
  const hn::Half<D64x4> dw2;
  auto store = [&](auto v, auto d, bool add, int64_t *IF_RESTRICT dst)
  { hn::StoreU(add ? hn::Add(v, hn::LoadU(d, dst)) : v, d, dst); };
  auto halves = [&](V32 v) { return hn::Add(hn::LowerHalf(d4, v), hn::UpperHalf(d4, v)); };
  for (int j = 0; j < K_NHOG; ++j)
  {
    int64_t *IF_RESTRICT dst = cb + (K_HOG + j) * kst + at[0] * cst; // cells cst == 1 apart
    if (g == 8)                                                      // a cell a quarter
      store(sum4x4(halves(A.hog[0][j]), halves(A.hog[1][j]), halves(A.hog[2][j]),
                   halves(A.hog[3][j])),
            dw, adds[0], dst);
    else if (g == 4) // two cells a quarter: the lower and upper four lanes
    {
      store(sum4x4(hn::LowerHalf(d4, A.hog[0][j]), hn::UpperHalf(d4, A.hog[0][j]),
                   hn::LowerHalf(d4, A.hog[1][j]), hn::UpperHalf(d4, A.hog[1][j])),
            dw, adds[0], dst);
      store(sum4x4(hn::LowerHalf(d4, A.hog[2][j]), hn::UpperHalf(d4, A.hog[2][j]),
                   hn::LowerHalf(d4, A.hog[3][j]), hn::UpperHalf(d4, A.hog[3][j])),
            dw, adds[2], cb + (K_HOG + j) * kst + at[2] * cst);
    }
    else // g == 16: a cell two quarters, which add first
    {
      const auto c0 = halves(hn::Add(A.hog[0][j], A.hog[1][j])),
                 c1 = halves(hn::Add(A.hog[2][j], A.hog[3][j]));
      store(hn::LowerHalf(dw2, sum4x4(c0, c1, hn::Zero(d4), hn::Zero(d4))), dw2, adds[0], dst);
    }
  }
}
#endif

// derive, for LD cell-channels in lockstep: four a vector where vectors hold four doubles.
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
constexpr int LD = 4;
#elif HWY_MAX_BYTES >= 16
constexpr int LD = 2;
#else
constexpr int LD = 1;
#endif
using DDV = hn::FixedTag<double, LD>;

// Every float feature of LD cell-channels, a lane each, from their sums S (the NSUM slots, as
// doubles), their moments M and 1/n. The scalar guards of the definitions become lane masks: a
// guarded quotient divides by 1 where the guard fails and is then zeroed. EVERY cell of every
// level goes through this one compiled body, never inlined, and lanes do not interact: that is
// what makes a grid derive to the same bits as any level of any pyramid, in any lane.
HWY_NOINLINE void derive_lanes(const hn::VFromD<DDV> *IF_RESTRICT S,
                               const hn::VFromD<DDV> *IF_RESTRICT M, hn::VFromD<DDV> invn,
                               const int *IF_RESTRICT cardinal,
                               hn::VFromD<hn::Rebind<float, DDV>> *IF_RESTRICT out)
{
  using DD = DDV;
  const DD dd;
  using V = hn::VFromD<DD>;
  using M_ = hn::MFromD<DD>;
  const hn::Rebind<float, DD> df;
  const V zero = hn::Zero(dd), one = hn::Set(dd, 1.0);
  auto f32 = [&](V v) { return hn::DemoteTo(df, v); };
  auto over = [&](V num, V den, M_ ok)
  { return hn::IfThenElseZero(ok, hn::Div(num, hn::IfThenElse(ok, den, one))); };
  auto above = [&](V v, double t) { return hn::Gt(v, hn::Set(dd, t)); };

  // structure tensor
  const V sxx = S[SXX], syy = S[SYY], sxy = S[SXY], n = S[CNT];
  const V tr = hn::Add(sxx, syy), d = hn::Sub(sxx, syy), sxy2 = hn::Add(sxy, sxy);
  const V R = hn::Sqrt(hn::Add(hn::Mul(d, d), hn::Mul(sxy2, sxy2)));
  const M_ grad = above(tr, 0.0);
  const V invt = over(one, tr, grad);
  out[0] = f32(hn::Mul(tr, invn));
  out[1] = f32(hn::Mul(R, invt));
  out[2] = f32(hn::Mul(d, invt));
  out[3] = f32(hn::Mul(sxy2, invt));
  out[4] = f32(hn::Mul(hn::Mul(hn::Sub(tr, R), hn::Set(dd, 0.5)), invn));
  // HOG (its bins partition sum|grad|^2 = tr, so invt normalizes them), extrema, LBP
  V sq = zero, card = zero;
  for (int b = 0; b < HB; ++b)
  {
    const V hb = S[HOG0 + b];
    out[NF_S + b] = f32(hn::Mul(hb, invt));
    sq = hn::Add(sq, hn::Mul(hb, hb));
    if (cardinal[b]) card = hn::Add(card, hb);
  }
  out[NF_S + HB] = f32(hn::Mul(S[NMAX], invn));
  out[NF_S + HB + 1] = f32(hn::Mul(S[NMIN], invn));
  for (int b = 0; b < LBPB; ++b) out[NF_S + HB + 2 + b] = f32(hn::Mul(S[LBP0 + b], invn));

  // descriptors
  auto *g = out + NF_S + HB + 2 + LBPB;
  const V var = M[1], energy = hn::PromoteTo(dd, out[0]), coherence = hn::PromoteTo(dd, out[1]);
  const M_ spread = above(var, 0.0);
  const V sd = hn::Sqrt(hn::IfThenElseZero(spread, var)),
          inv2 = over(one, hn::Mul(var, var), spread);
  g[0] =
      f32(hn::IfThenElseZero(spread, hn::Mul(hn::Mul(M[2], inv2), sd))); // std_skew = m3 / var^1.5
  g[1] = f32(hn::IfThenElseZero(spread, hn::Sub(hn::Mul(M[3], inv2), hn::Set(dd, 3.0))));
  g[2] = f32(hn::Div(energy, hn::Add(var, one)));       // edge_sharpness = energy/(var+eps)
  g[3] = f32(hn::Mul(energy, hn::Sub(one, coherence))); // detail = energy*(1-coherence)
  g[4] = f32(hn::IfThenElseZero(grad, hn::Mul(hn::Mul(sq, invt), invt))); // sum p_i^2
  g[5] = f32(hn::IfThenElseZero(grad, hn::Mul(card, invt)));              // cardinal fraction
  g[6] = f32(hn::Mul(hn::Mul(hn::Mul(n, S[SG4]), invt), invt));           // gradient sparsity
  g[7] = f32(hn::Div(sd, hn::Add(M[0], one)));                            // rms contrast

  // bar detector
  auto *q = out + BARDF0;
  V btot = zero, bpk = zero, bvar = zero;
  for (int j = 0; j < BARD_NL; ++j) btot = hn::Add(btot, S[BARD_S0 + j]);
  // the divider is what this function waits on, so one reciprocal serves the whole block
  const V binv = over(one, btot, above(btot, 1e-9)), nl = hn::Set(dd, BARD_NL);
  q[0] = f32(hn::Mul(S[BARD0], invn));
  for (int j = 0; j < BARD_NL; ++j)
  {
    const V m = S[BARD_S0 + j], e = hn::Sub(hn::Mul(m, nl), btot); // NL * (m - mean), an integer
    q[1 + j] = f32(hn::Mul(m, binv)); // spectrum: share of response mass at this lag
    bpk = hn::Add(bpk, hn::Mul(m, hn::Set(dd, j + 1)));
    bvar = hn::Add(bvar, hn::Mul(e, e));
  }
  q[1 + BARD_NL] = f32(hn::Mul(bpk, binv)); // mass-weighted mean lag index, in [1, NL]
  // CV of the lag profile = sqrt(sum e^2 / NL^3) / (btot / NL): one square root and no divide,
  // and exactly 0 for a flat profile because e is an integer
  q[2 + BARD_NL] =
      f32(hn::Mul(hn::Mul(hn::Sqrt(bvar), binv), hn::Set(dd, 1.0 / std::sqrt((double)BARD_NL))));
  const V bdk = S[BARD_DARK], blt = S[BARD_LIGHT], bsum = hn::Add(bdk, blt);
  q[3 + BARD_NL] = f32(over(hn::Sub(blt, bdk), bsum, above(bsum, 1e-9))); // polarity in [-1, 1]

  // texture: variance of the Laplacian, that per unit of gradient energy, the mean squared
  // Laws responses, and how far vertical lines (LS) outweigh horizontal ones (SL)
  auto *x = out + TEXF0;
  const V lmean = hn::Mul(S[LAP1], invn);
  const V lvar = hn::Max(hn::Sub(hn::Mul(S[LAP2], invn), hn::Mul(lmean, lmean)), zero);
  x[0] = f32(lvar);
  x[1] = f32(hn::Mul(hn::Mul(lvar, n), invt));
  for (int t = 0; t < NLAWS; ++t) x[2 + t] = f32(hn::Mul(S[LAWS0 + t], invn));
  const V ls = S[LAWS0 + LAWS_LS], sl = S[LAWS0 + LAWS_SL], lines = hn::Add(ls, sl);
  x[2 + NLAWS] = f32(over(hn::Sub(ls, sl), lines, above(lines, 0.0)));
  for (int t = 0; t < NMOM; ++t) out[NF + t] = f32(M[t]); // the moments, rounded to float32
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on, push)
#endif
// Central moments [mean, var, m3, m4] of LD cell-channels from their exact power sums t[0..3]
// about the rounded mean k. EVERY cell goes through this one body, so a grid's moments do not
// depend on which level computed it; and these are the very operations -O3 -ffast-math made of
//   var = u2 - d^2,  m3 = u3 - 3 d u2 + 2 d^3,  m4 = u4 - 4 d u3 + 6 d^2 u2 - 3 d^4  (u_i = t_i/n)
// when they were scalar expressions, so the bits are the ones they always were.
IF_STRICT_FN HWY_NOINLINE void moments_tail(const hn::VFromD<DDV> *IF_RESTRICT t, hn::VFromD<DDV> k,
                                            hn::VFromD<DDV> invn, hn::VFromD<DDV> *IF_RESTRICT M)
{
  IF_STRICT_BODY
  const DDV dd;
  const auto d = hn::Mul(t[0], invn), u2 = hn::Mul(t[1], invn), d2 = hn::Mul(d, d);
  const auto u3 = hn::Mul(t[2], invn);
  M[0] = hn::Add(k, d);
  M[1] = hn::Sub(u2, d2);
  M[2] = hn::NegMulAdd(hn::Mul(d, hn::Set(dd, 3.0)), u2, hn::MulAdd(hn::Add(d, d), d2, u3));
  // (6 d^2) u2 is a chain -ffast-math may regroup, whatever this function's options say: a fused
  // multiply-add with nothing to add is the same product, and is not regrouped
  const auto six_d2 = hn::MulAdd(d2, hn::Set(dd, 6.0), hn::Zero(dd));
  const auto a = hn::NegMulAdd(u3, hn::Mul(d, hn::Set(dd, 4.0)), hn::Mul(six_d2, u2));
  const auto b = hn::MulSub(invn, t[3], hn::Mul(hn::Mul(d2, d2), hn::Set(dd, 3.0)));
  M[3] = hn::Add(b, a);
}
// Power sums a[0..3] of n samples about some centre, re-centred on the rounded mean: t and the
// offset q of that mean from the centre. In doubles, which is exact while every integer here
// stays below 2^53 -- n up to 2^20 for sums about KBIAS -- because then each operation's exact
// result is representable. The rounded mean's floor needs no divide either: the quotient by 1/n
// is within one of it, and the remainder, exact, says which way.
IF_STRICT_FN HWY_NOINLINE void moments_recentre(const hn::VFromD<DDV> *IF_RESTRICT a,
                                                double samples, double inverse,
                                                hn::VFromD<DDV> *IF_RESTRICT t, hn::VFromD<DDV> &q)
{
  IF_STRICT_BODY
  const DDV dd;
  const auto n = hn::Set(dd, samples), one = hn::Set(dd, 1.0);
  const auto num = hn::Add(a[0], hn::Set(dd, std::floor(samples * 0.5)));
  q = hn::Floor(hn::Mul(num, hn::Set(dd, inverse)));
  const auto rem = hn::Sub(num, hn::Mul(q, n));
  q = hn::Add(q, hn::Sub(hn::IfThenElseZero(hn::Ge(rem, n), one),
                         hn::IfThenElseZero(hn::Lt(rem, hn::Zero(dd)), one)));
  const auto q2 = hn::Mul(q, q), q3 = hn::Mul(q2, q), two = hn::Add(q, q), nq = hn::Mul(n, q);
  t[0] = hn::Sub(a[0], nq);
  t[1] = hn::Add(hn::Sub(a[1], hn::Mul(two, a[0])), hn::Mul(n, q2));
  t[2] = hn::Sub(hn::Add(hn::Sub(a[2], hn::Mul(hn::Mul(q, hn::Set(dd, 3.0)), a[1])),
                         hn::Mul(hn::Mul(q2, hn::Set(dd, 3.0)), a[0])),
                 hn::Mul(n, q3));
  t[3] = hn::Add(hn::Sub(hn::Add(hn::Sub(a[3], hn::Mul(hn::Mul(q, hn::Set(dd, 4.0)), a[2])),
                                 hn::Mul(hn::Mul(q2, hn::Set(dd, 6.0)), a[1])),
                         hn::Mul(hn::Mul(q3, hn::Set(dd, 4.0)), a[0])),
                 hn::Mul(hn::Mul(n, q3), q));
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#endif

// Split n columns into P phase planes `phs` apart: phase f, holding columns f, f+P, ..., lands
// at out + f*phs. One even/odd pass per factor of two, through the scratch halves in `tmp`.
void split_phases(const uint8_t *IF_RESTRICT src, size_t n, int P, size_t phs,
                  uint8_t *IF_RESTRICT tmp, uint8_t *IF_RESTRICT out)
{
  const size_t N = hn::Lanes(d8), half = n / 2;
  uint8_t *IF_RESTRICT even = P == 2 ? out : tmp,
                       *IF_RESTRICT odd = P == 2 ? out + phs : tmp + half;
  size_t x = 0;
  for (; x + 2 * N <= n; x += 2 * N)
  {
    const V8 lo = hn::LoadU(d8, src + x), hi = hn::LoadU(d8, src + x + N);
    hn::StoreU(hn::ConcatEven(d8, hi, lo), d8, even + x / 2);
    hn::StoreU(hn::ConcatOdd(d8, hi, lo), d8, odd + x / 2);
  }
  for (; x < n; x += 2)
  {
    even[x / 2] = src[x];
    odd[x / 2] = src[x + 1];
  }
  if (P > 2) // the even columns hold the even phases, 2*phs apart, and the odd ones the rest
  {
    split_phases(even, half, P / 2, 2 * phs, tmp + n, out);
    split_phases(odd, half, P / 2, 2 * phs, tmp + n, out + phs);
  }
}

// --- colour conversion, fused into the row gather ---------------------------------------------
// A host that has a BGR frame and wants features in HSV would otherwise convert the whole image
// first (cvtColor: a pass that writes and re-reads it). Here the three source channels are
// converted as each row is de-interleaved, in the band's thread, so the conversion costs no
// memory traffic and threads like the rest. The outputs are cv2.cvtColor's bytes exactly: the
// tests compare every one of the 2^24 colours. Adding a conversion means a Convert code, a row
// kernel like hsv_row, a case in convert_row, and an entry in the Python table.
enum Convert : uint8_t
{
  CONV_NONE = 0,
  CONV_BGR2HSV = 1 // OpenCV's 8-bit BGR -> HSV, hue in [0, 180)
};

// OpenCV's 8-bit RGB2HSV_b, with its fixed-point tables (hsv_shift 12):
//   v = max(b, g, r), diff = v - min(b, g, r)
//   s = (diff * sdiv[v] + 2^11) >> 12,               sdiv[i] = round(255 * 2^12 / i)
//   h = (h6 * hdiv[diff] + 2^11) >> 12, + 180 if < 0, hdiv[i] = round(180 * 2^12 / (6 i))
//   h6 = g - b           if v == r
//      = b - r + 2 diff  if v == g (and v != r)
//      = r - g + 4 diff  otherwise
// (>> the arithmetic shift, i.e. floor; sdiv[0] = hdiv[0] = 0). No quotient in either table is
// a half -- 2 * 255 * 2^12 and 2 * 30 * 2^12 hold more twos than any i < 256 -- so "round" needs
// no tie rule, and (2 N + i) / (2 i) in integers is it.
constexpr int HSV_SHIFT = 12;
constexpr int32_t HSV_SNUM = 255 << HSV_SHIFT;
constexpr int32_t HSV_HNUM = (180 << HSV_SHIFT) / 6;
constexpr int32_t HSV_HALF = 1 << (HSV_SHIFT - 1);
constexpr int32_t HSV_HRANGE = 180;

struct HsvTables
{
  int32_t sdiv[256], hdiv[256];
  constexpr HsvTables() : sdiv{}, hdiv{}
  {
    for (int i = 1; i < 256; ++i)
    {
      sdiv[i] = (2 * HSV_SNUM + i) / (2 * i);
      hdiv[i] = (2 * HSV_HNUM + i) / (2 * i);
    }
  }
};
constexpr HsvTables kHsv;

inline void hsv_pixel(int b, int g, int r, uint8_t *IF_RESTRICT hsv)
{
  const int v = std::max(b, std::max(g, r)), diff = v - std::min(b, std::min(g, r));
  const int h6 = v == r ? g - b : v == g ? b - r + 2 * diff : r - g + 4 * diff;
  const int h = (h6 * kHsv.hdiv[diff] + HSV_HALF) >> HSV_SHIFT; // floors: OpenCV's own shift
  hsv[0] = (uint8_t)(h < 0 ? h + HSV_HRANGE : h);
  hsv[1] = (uint8_t)((diff * kHsv.sdiv[v] + HSV_HALF) >> HSV_SHIFT);
  hsv[2] = (uint8_t)v;
}

// Even and odd lanes of a vector as the next wider type, and back. The even lanes are the low
// halves and the odd ones the high halves, so a split is a mask or a shift and a merge a shift
// and an or: no shuffles, whose port is the scarce one on Intel cores, and the order of the
// lanes takes care of itself. A merge of signed lanes masks the even ones' sign extension.
HWY_INLINE V16 even8(V8 v) { return hn::And(hn::BitCast(d16, v), hn::Set(d16, 0x00FF)); }
HWY_INLINE V16 odd8(V8 v) { return hn::BitCast(d16, hn::ShiftRight<8>(hn::BitCast(du16, v))); }
HWY_INLINE V32 even16u(V16 v) { return hn::And(hn::BitCast(d32, v), hn::Set(d32, 0xFFFF)); }
HWY_INLINE V32 even16(V16 v) { return hn::ShiftRight<16>(hn::ShiftLeft<16>(hn::BitCast(d32, v))); }
HWY_INLINE V32 odd16(V16 v) { return hn::ShiftRight<16>(hn::BitCast(d32, v)); }
HWY_INLINE V16 merge16(V32 even, V32 odd)
{
  return hn::BitCast(d16, hn::Or(hn::And(even, hn::Set(d32, 0xFFFF)), hn::ShiftLeft<16>(odd)));
}
HWY_INLINE V8 merge8(V16 even, V16 odd) // lanes in [0, 255]
{
  return hn::BitCast(d8, hn::Or(even, hn::ShiftLeft<8>(odd)));
}

using DF = hn::Repartition<float, D8>;
using VF = hn::VFromD<DF>;

// The float arithmetic below is exact where it needs to be and within a tolerance elsewhere,
// as written: IF_STRICT keeps -ffast-math from rewriting it (see moments_tail).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on, push)
#endif
// 1 / p for p in [1, 2^16]. A division: one instruction, correctly rounded, and the divider
// it occupies for a few cycles runs beside the arithmetic ports that are the busy ones here.
// GCC and clang both replace a vector division by a hardware estimate plus one Newton step
// under -ffast-math, whatever the function's own options say; CMakeLists.txt turns that off
// (-mno-recip, -fno-reciprocal-math), but hsv_table would still be exact with it on -- that
// estimate is within 2^-22 or so, and the table only needs its guess within 1/2.
HWY_INLINE VF hsv_recip(VF p)
{
  IF_STRICT_BODY
  const DF df;
  return hn::Div(hn::Set(df, 1.0f), p);
}

// (NUM / x) rounded half up, for x in [1, 255], from a float guess g of the quotient within
// 2^-23 of it (relative): the nearest integer to g. The quotient is never a half-integer
// (2 NUM / x odd would need x >= 8192) and is at least 1 / (2x) from the nearest one (2 (NUM
// mod x) - x is a nonzero integer), while g is at most NUM 2^-23 / x from it; the ratio of
// the two, 2 NUM 2^-23, is 0.25 for the saturation table and 0.03 for the hue table, so
// rounding g lands on the table value with a margin, on any IEEE target. The exhaustive
// colour test holds every target to it. The value returns as a float, which the
// fixed-point product wants.
template <int32_t NUM>
HWY_INLINE VF hsv_table(VF g)
{
  IF_STRICT_BODY
  static_assert(2.0 * NUM * (1.0 / 8388608.0) < 0.5, "the guess rounds to the table value");
  return hn::Round(g);
}

// Eight (or more) pixels' s and h -- OpenCV's integers, in float arithmetic that is exact at
// every step where it has to be: v, diff and h6 are small integers; sdiv[v] and hdiv[diff] come
// out of hsv_table exactly; each fixed-point product is below 2^21, so it and its + 2^11 are
// exact; and the final >> 12 is the integer shift, which floors like OpenCV's. Both table
// values share one reciprocal, 1 / (v * diff): sdiv = SNUM * diff * r, hdiv = HNUM * v * r,
// where SNUM * diff and HNUM * v are exact (they are 2^12 * 255 * diff and 2^13 * 15 * v), so
// each guess carries the reciprocal's rounding and one of its own: within 2^-23 relative,
// which is what hsv_table needs. When diff is 0 (v too, or a grey)
// the products that use the tables are 0, so the tables may be anything finite: the
// reciprocal is taken of max(v * diff, 1).
template <bool kH, bool kS>
HWY_INLINE void hsv_lanes(V32 v, V32 diff, V32 h6, V32 &s, V32 &h)
{
  IF_STRICT_BODY
  const DF df;
  const VF vf = hn::ConvertTo(df, v), dfv = hn::ConvertTo(df, diff);
  const VF half_ = hn::Set(df, (float)HSV_HALF);
  const VF r = hsv_recip(hn::Max(hn::Mul(vf, dfv), hn::Set(df, 1.0f)));
  if (kS)
  {
    const VF sdiv = hsv_table<HSV_SNUM>(hn::Mul(hn::Mul(hn::Set(df, (float)HSV_SNUM), dfv), r));
    s = hn::ShiftRight<HSV_SHIFT>(hn::ConvertInRangeTo(d32, hn::MulAdd(dfv, sdiv, half_)));
  }
  if (kH)
  {
    const VF hdiv = hsv_table<HSV_HNUM>(hn::Mul(hn::Mul(hn::Set(df, (float)HSV_HNUM), vf), r));
    const VF t = hn::MulAdd(hn::ConvertTo(df, h6), hdiv, half_);
    h = hn::ShiftRight<HSV_SHIFT>(hn::ConvertInRangeTo(d32, t));
  }
}

// One block of pixels: v, and the H and S planes when asked for. v and diff are taken in
// bytes, the hue numerator h6 in 16-bit lanes, and the rest in hsv_lanes, on the even and odd
// lanes of each level in turn.
template <bool kH, bool kS>
HWY_INLINE void hsv_block(V8 b, V8 g, V8 r, V8 &h, V8 &s, V8 &v)
{
  v = hn::Max(hn::Max(b, g), r);
  const V8 diff = hn::Sub(v, hn::Min(hn::Min(b, g), r));
  if (!kH && !kS) return;
  V16 h16[2], s16[2];
  for (int half = 0; half < 2; ++half)
  {
    auto lanes = [&](V8 a) { return half ? odd8(a) : even8(a); };
    const V16 v1 = lanes(v), d1 = lanes(diff);
    V16 h6 = hn::Zero(d16);
    if (kH)
    {
      const V16 b1 = lanes(b), g1 = lanes(g), r1 = lanes(r);
      const V16 d2 = hn::Add(d1, d1);
      h6 = hn::IfThenElse(hn::Eq(v1, r1), hn::Sub(g1, b1),
                          hn::IfThenElse(hn::Eq(v1, g1), hn::Add(hn::Sub(b1, r1), d2),
                                         hn::Add(hn::Sub(r1, g1), hn::Add(d2, d2))));
    }
    V32 s32[2], h32[2];
    for (int q = 0; q < 2; ++q)
    {
      const V32 v2 = q ? odd16(v1) : even16u(v1), d2 = q ? odd16(d1) : even16u(d1);
      const V32 h2 = kH ? (q ? odd16(h6) : even16(h6)) : hn::Zero(d32);
      hsv_lanes<kH, kS>(v2, d2, h2, s32[q], h32[q]);
    }
    if (kS) s16[half] = merge16(s32[0], s32[1]);
    if (kH) // h is in [-30, 150]: + 180 where negative
    {
      const V16 hh = merge16(h32[0], h32[1]);
      h16[half] = hn::Add(hh, hn::And(hn::BroadcastSignBit(hh), hn::Set(d16, HSV_HRANGE)));
    }
  }
  if (kS) s = merge8(s16[0], s16[1]);
  if (kH) h = merge8(h16[0], h16[1]);
}

// One row: the selected HSV channels (chan[0..c), each to out + k*ostride, as gather_row lays
// them out) from the three source channels at bytes coff[0..2] of each pixel. `packed`: the
// pixels are 3 contiguous bytes in B, G, R order. `tmp` holds 3*w bytes for the strided case.
template <bool kH, bool kS>
IF_STRICT_FN HWY_NOINLINE void hsv_row(const uint8_t *IF_RESTRICT src, int64_t cs,
                                       const int64_t *IF_RESTRICT coff, const int *IF_RESTRICT chan,
                                       int c, int w, bool packed, uint8_t *IF_RESTRICT tmp,
                                       uint8_t *IF_RESTRICT out, size_t ostride)
{
  IF_STRICT_BODY
  const size_t N = hn::Lanes(d8);
  const uint8_t *p[3];
  int64_t ps = 1; // p[k] + x * ps is channel k of pixel x
  if (packed)
  {
    for (int k = 0; k < 3; ++k) p[k] = src + k;
    ps = 3;
  }
  else if (cs == 1) // planar
    for (int k = 0; k < 3; ++k) p[k] = src + coff[k];
  else // anything else: gathered a pixel at a time first
    for (int k = 0; k < 3; ++k)
    {
      uint8_t *IF_RESTRICT plane = tmp + (size_t)k * (size_t)w;
      for (int x = 0; x < w; ++x) plane[x] = src[(int64_t)x * cs + coff[k]];
      p[k] = plane;
    }
  size_t x = 0;
  for (; x + N <= (size_t)w; x += N)
  {
    V8 b, g, r, h, s, v;
    if (packed)
      hn::LoadInterleaved3(d8, src + x * 3, b, g, r);
    else
    {
      b = hn::LoadU(d8, p[0] + x);
      g = hn::LoadU(d8, p[1] + x);
      r = hn::LoadU(d8, p[2] + x);
    }
    hsv_block<kH, kS>(b, g, r, h, s, v);
    for (int k = 0; k < c; ++k)
      hn::StoreU(chan[k] == 0 ? h : chan[k] == 1 ? s : v, d8, out + (size_t)k * ostride + x);
  }
  for (; x < (size_t)w; ++x)
  {
    uint8_t px[3];
    hsv_pixel(p[0][x * ps], p[1][x * ps], p[2][x * ps], px);
    for (int k = 0; k < c; ++k) out[(size_t)k * ostride + x] = px[chan[k]];
  }
}

// The instantiations, here so that they are compiled under the options above.
#define IMFEAT_HSV_ROW_ARGS                                                                        \
  (const uint8_t *, int64_t, const int64_t *, const int *, int, int, bool, uint8_t *, uint8_t *,   \
   size_t)
template void hsv_row<true, true> IMFEAT_HSV_ROW_ARGS;
template void hsv_row<true, false> IMFEAT_HSV_ROW_ARGS;
template void hsv_row<false, true> IMFEAT_HSV_ROW_ARGS;
template void hsv_row<false, false> IMFEAT_HSV_ROW_ARGS;
#undef IMFEAT_HSV_ROW_ARGS
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#endif

// gather_row's counterpart under a conversion: the selected feature-space channels of one row,
// converted from the source channels at coff[0..2], channel k of `chan` at out + k*ostride.
void convert_row(int conv, const uint8_t *IF_RESTRICT src, int64_t cs,
                 const int64_t *IF_RESTRICT coff, const int *IF_RESTRICT chan, int c, int w,
                 bool packed, uint8_t *IF_RESTRICT tmp, uint8_t *IF_RESTRICT out, size_t ostride)
{
  bool want[3] = {false, false, false};
  for (int k = 0; k < c; ++k) want[chan[k]] = true;
  switch (conv)
  {
  case CONV_BGR2HSV:
    if (want[0] && want[1])
      hsv_row<true, true>(src, cs, coff, chan, c, w, packed, tmp, out, ostride);
    else if (want[0])
      hsv_row<true, false>(src, cs, coff, chan, c, w, packed, tmp, out, ostride);
    else if (want[1])
      hsv_row<false, true>(src, cs, coff, chan, c, w, packed, tmp, out, ostride);
    else
      hsv_row<false, false>(src, cs, coff, chan, c, w, packed, tmp, out, ostride);
    return;
  default: throw std::invalid_argument("unknown colour conversion");
  }
}

// Gather each selected channel's row into a contiguous run: channel k lands at
// out + k*ostride. Every layout imfeat accepts has a wide path here except the
// general strided one, and the phase-split below reuses all of them rather than
// carrying its own -- getting that wrong made a 2-channel or greyscale image fall to
// a scalar byte loop over the whole frame.
void gather_row(const uint8_t *IF_RESTRICT src, int64_t cs, const int64_t *IF_RESTRICT coff, int c,
                int w, bool packed, uint8_t *IF_RESTRICT tmp, uint8_t *IF_RESTRICT out,
                size_t ostride)
{
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint8_t> du;
  const size_t N = hn::Lanes(du);
  if (cs == 1) // already planar (includes 2-D input)
  {
    for (int k = 0; k < c; ++k) std::memcpy(out + (size_t)k * ostride, src + coff[k], (size_t)w);
    return;
  }
  if (packed && c > 4 && (c & (c - 1)) == 0) // channel k is byte phase k of the packed row
    return split_phases(src, (size_t)w * c, c, ostride, tmp, out);
  size_t x = 0;
  if (packed && c == 2)
    for (; x + N <= (size_t)w; x += N)
    {
      decltype(hn::Zero(du)) a, b;
      hn::LoadInterleaved2(du, src + x * 2, a, b);
      hn::StoreU(a, du, out + x);
      hn::StoreU(b, du, out + ostride + x);
    }
  else if (packed && c == 3)
    for (; x + N <= (size_t)w; x += N)
    {
      decltype(hn::Zero(du)) a, b, e;
      hn::LoadInterleaved3(du, src + x * 3, a, b, e);
      hn::StoreU(a, du, out + x);
      hn::StoreU(b, du, out + ostride + x);
      hn::StoreU(e, du, out + 2 * ostride + x);
    }
  else if (packed && c == 4)
    for (; x + N <= (size_t)w; x += N)
    {
      decltype(hn::Zero(du)) a, b, e, g;
      hn::LoadInterleaved4(du, src + x * 4, a, b, e, g);
      hn::StoreU(a, du, out + x);
      hn::StoreU(b, du, out + ostride + x);
      hn::StoreU(e, du, out + 2 * ostride + x);
      hn::StoreU(g, du, out + 3 * ostride + x);
    }
  for (; x < (size_t)w; ++x)
    for (int k = 0; k < c; ++k) out[(size_t)k * ostride + x] = src[(int64_t)x * cs + coff[k]];
}

// --- the thumbnail resize, fused into the row gather -------------------------------------------
// A host that thumbnails a frame before the pass runs cv2.resize(INTER_AREA): a pass over the
// frame on OpenCV's thread pool that writes the thumbnail, which the pass then reads back. Here
// each band makes its own rows of the thumbnail (halo included) from the frame's rows as its
// first step, in its own thread, and its row gather then reads them like any planar rows;
// nothing leaves the band unless the caller asks for the thumbnail. The bytes are cv2.resize's
// exactly -- the tests compare frames of many sizes -- because the arithmetic is OpenCV's
// (imgproc/src/resize.cpp) step for step:
//   * a destination column takes its source columns in table order (the partial cell on the
//     left, the whole cells, the partial cell on the right) with float32 weights; each product
//     and each running sum is rounded to float32, and there is no fused multiply-add anywhere
//     (OpenCV compiles this for its SSE3 baseline: see the pragmas and IF_STRICT below);
//   * a destination row is its source rows' resampled values scaled by their float32 weights
//     and summed in order; the result is rounded half to even and clamped;
//   * when both ratios are whole numbers OpenCV sums the block instead and rounds sum / area,
//     half up for 2x2 on 1, 3 or 4 channels and through a float32 product otherwise; and the
//     same size is a copy. Upscaling is not INTER_AREA at all in OpenCV (it turns bilinear),
//     so it is refused here and a host resizes with cv2 first.
// What makes it fast: a source row is resampled once (OpenCV does it once per destination row
// it feeds, about twice); the horizontal pass is a fixed byte shuffle per group of G destination
// columns, whose taps all lie in one 16-byte window of the row, one shuffle and one multiply per
// tap slot; the vertical pass is a vector multiply-add over the row; and the work splits over
// the bands like the rest of the pass, with the output identical at any thread count.
struct AreaTap
{
  int si;
  float alpha;
};

// OpenCV's computeResizeAreaTab, per destination index: the taps, in the order it sums them.
std::vector<std::vector<AreaTap>> area_taps(int ssize, int dsize)
{
  const double scale = 1.0 / ((double)dsize / ssize); // cv::resize: 1 / inv_scale
  std::vector<std::vector<AreaTap>> out((size_t)dsize);
  for (int dx = 0; dx < dsize; ++dx)
  {
    const double fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const double cell = std::min(scale, ssize - fsx1);
    int sx1 = (int)std::ceil(fsx1), sx2 = (int)std::floor(fsx2);
    sx2 = std::min(sx2, ssize - 1);
    sx1 = std::min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) out[dx].push_back({sx1 - 1, (float)((sx1 - fsx1) / cell)});
    for (int sx = sx1; sx < sx2; ++sx) out[dx].push_back({sx, (float)(1.0 / cell)});
    if (fsx2 - sx2 > 1e-3)
      out[dx].push_back({sx2, (float)(std::min(std::min(fsx2 - sx2, 1.0), cell) / cell)});
  }
  return out;
}

// The horizontal plan: groups of G destination columns, each with the start of the 16-byte
// source window its taps lie in, a shuffle pattern per tap slot and G weights per slot. The
// pattern is G bytes at the front of 16 (0x80 picks a zero byte, whose weight is 0) for the
// whole-number block sums, and for the float taps (`lane4`) 4 bytes a lane, the tap's byte at
// the lane's low byte and zeros elsewhere, so that the shuffle lands each tap in its int32
// lane with no widening after it. G is the widest of 8 (4 on 128-bit targets), 4, 2, 1
// that fits every group in a window; when even one column's taps do not (a ratio near 16 or
// more), G is 0 and the row goes through the scalar loop, which is the same sum in the same
// order.
constexpr int AREA_MAX_G = HWY_MAX_BYTES >= 32 ? 8 : 4;

struct AreaHPlan
{
  int G = 0, slots = 0, groups = 0, dsize = 0;
  bool lane4 = false;           // the pattern layout (see above)
  std::vector<int> w0;          // [group]
  std::vector<uint8_t> pattern; // [group][slot][16], or [group][slot][4 * G] where lane4
  std::vector<float> weight;    // [group][slot][G]
  std::vector<int> toff;        // scalar: index i's taps are taps[toff[i] .. toff[i+1])
  std::vector<AreaTap> taps;
};

bool area_hplan_fit(const std::vector<std::vector<AreaTap>> &t, int G, AreaHPlan &p)
{
  const int dsize = (int)t.size();
  p.G = G;
  p.dsize = dsize;
  p.groups = (dsize + G - 1) / G;
  p.slots = 0;
  for (const auto &taps : t) p.slots = std::max(p.slots, (int)taps.size());
  p.w0.assign((size_t)p.groups, 0);
  const size_t pb = p.lane4 ? 4 * (size_t)G : 16, pl = p.lane4 ? 4 : 1; // a slot's bytes, a lane's
  p.pattern.assign((size_t)p.groups * p.slots * pb, 0x80);
  p.weight.assign((size_t)p.groups * p.slots * G, 0.0f);
  for (int g = 0; g < p.groups; ++g)
  {
    int lo = 1 << 30, hi = -1;
    for (int l = 0; l < G && g * G + l < dsize; ++l)
      for (const AreaTap &tap : t[(size_t)g * G + l])
      {
        lo = std::min(lo, tap.si);
        hi = std::max(hi, tap.si);
      }
    if (hi - lo + 1 > 16) return false;
    p.w0[g] = lo;
    for (int l = 0; l < G && g * G + l < dsize; ++l)
    {
      const auto &taps = t[(size_t)g * G + l];
      for (int k = 0; k < (int)taps.size(); ++k)
      {
        p.pattern[((size_t)g * p.slots + k) * pb + pl * l] = (uint8_t)(taps[k].si - lo);
        p.weight[((size_t)g * p.slots + k) * G + l] = taps[k].alpha;
      }
    }
  }
  return true;
}

void area_hplan(const std::vector<std::vector<AreaTap>> &t, AreaHPlan &p, bool lane4)
{
  p.lane4 = lane4;
  for (int G = AREA_MAX_G; G >= 1; G /= 2)
    if (area_hplan_fit(t, G, p)) return;
  p = AreaHPlan(); // scalar
  p.dsize = (int)t.size();
  p.toff.assign((size_t)p.dsize + 1, 0);
  for (int i = 0; i < p.dsize; ++i)
  {
    p.taps.insert(p.taps.end(), t[i].begin(), t[i].end());
    p.toff[i + 1] = (int)p.taps.size();
  }
}

// The float arithmetic below is OpenCV's to the last rounding, as written: no fused
// multiply-add, no reassociation (IF_STRICT; on MSVC the precise model and, since that alone
// may leave /fp:fast's contractions on, fp_contract off -- which has no push, so it is put back
// on after the region, /fp:fast's default, which is what CMakeLists.txt sets).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif
// One row of one plane through the horizontal plan: dsize float32 sums of the taps, in tap order.
// P planes at once (plane q at src + q * sstride, its row at dst + q * dstride): a group's
// patterns and weights are loaded once for all of them, and the loop's bookkeeping is shared;
// each plane's products and sums are the same, in the same order.
template <int G, int P>
IF_STRICT_FN HWY_NOINLINE void area_hpass(const AreaHPlan &p, const uint8_t *IF_RESTRICT src,
                                          size_t sstride, float *IF_RESTRICT dst, size_t dstride)
{
  IF_STRICT_BODY
  const hn::FixedTag<uint8_t, 16> dwin;
  const hn::FixedTag<uint8_t, 4 * G> dpat; // a slot's pattern: G int32 lanes' worth of bytes
  const hn::FixedTag<int32_t, G> di;
  const hn::Rebind<float, decltype(di)> df;
  // the window as the shuffle's table: itself where a pattern is at most 16 bytes, else in
  // every 16-byte block of a pattern-sized vector (the shuffle looks up within blocks)
  constexpr bool kWide = 4 * G > 16;
  using Tab = std::conditional_t<kWide, hn::VFromD<decltype(dpat)>, hn::VFromD<decltype(dwin)>>;
  for (size_t g = 0; g < (size_t)p.groups; ++g)
  {
    Tab win[P];
    for (int q = 0; q < P; ++q)
    {
      const auto w16 = hn::LoadU(dwin, src + q * sstride + p.w0[g]);
      if constexpr (kWide)
        win[q] = hn::Combine(dpat, w16, w16);
      else
        win[q] = w16;
    }
    const uint8_t *IF_RESTRICT pat = p.pattern.data() + g * p.slots * 4 * G;
    const float *IF_RESTRICT wgt = p.weight.data() + g * p.slots * G;
    hn::VFromD<decltype(df)> acc[P];
    for (size_t k = 0; k < (size_t)p.slots; ++k)
    {
      const auto pk = hn::LoadU(dpat, pat + k * 4 * G);
      const auto wk = hn::LoadU(df, wgt + k * G);
      for (int q = 0; q < P; ++q)
      {
        const auto sel = hn::TableLookupBytesOr0(win[q], pk); // each tap's byte in its lane
        const auto v = hn::ConvertTo(df, hn::BitCast(di, sel));
        const auto prod = hn::Mul(v, wk);               // rounded, like OpenCV's S * alpha
        acc[q] = k == 0 ? prod : hn::Add(acc[q], prod); // 0 + the first product is the product
      }
    }
    for (int q = 0; q < P; ++q)
    {
      float *IF_RESTRICT o = dst + q * dstride + g * G;
      if (g * G + G <= (size_t)p.dsize)
        hn::StoreU(acc[q], df, o);
      else
        hn::StoreN(acc[q], df, o, (size_t)p.dsize - g * G);
    }
  }
}

IF_STRICT_FN HWY_NOINLINE void area_hpass_scalar(const AreaHPlan &p, const uint8_t *IF_RESTRICT src,
                                                 float *IF_RESTRICT dst)
{
  IF_STRICT_BODY
  for (int i = 0; i < p.dsize; ++i)
  {
    float acc = 0.0f;
    for (int j = p.toff[i]; j < p.toff[i + 1]; ++j)
    {
      const float prod = (float)src[p.taps[j].si] * p.taps[j].alpha;
      acc = acc + prod;
    }
    dst[i] = acc;
  }
}

// dst = beta * a, and dst = dst + beta * a: OpenCV's inter_area::mul and muladd.
IF_STRICT_FN HWY_NOINLINE void area_vmul(const float *IF_RESTRICT a, int n, float beta,
                                         float *IF_RESTRICT dst)
{
  IF_STRICT_BODY
  const hn::ScalableTag<float> df;
  const auto b = hn::Set(df, beta);
  size_t i = 0;
  for (; i + hn::Lanes(df) <= (size_t)n; i += hn::Lanes(df))
    hn::StoreU(hn::Mul(b, hn::LoadU(df, a + i)), df, dst + i);
  for (; i < (size_t)n; ++i) dst[i] = beta * a[i];
}

IF_STRICT_FN HWY_NOINLINE void area_vmuladd(const float *IF_RESTRICT a, int n, float beta,
                                            float *IF_RESTRICT dst)
{
  IF_STRICT_BODY
  const hn::ScalableTag<float> df;
  const auto b = hn::Set(df, beta);
  size_t i = 0;
  for (; i + hn::Lanes(df) <= (size_t)n; i += hn::Lanes(df))
  {
    const auto prod = hn::Mul(b, hn::LoadU(df, a + i));
    hn::StoreU(hn::Add(hn::LoadU(df, dst + i), prod), df, dst + i);
  }
  for (; i < (size_t)n; ++i)
  {
    const float prod = beta * a[i];
    dst[i] = dst[i] + prod;
  }
}

// Round half to even and clamp to a byte: OpenCV's saturate_cast<uchar>(float).
IF_STRICT_FN HWY_NOINLINE void area_round(const float *IF_RESTRICT sum, int n,
                                          uint8_t *IF_RESTRICT out)
{
  IF_STRICT_BODY
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di;
  const hn::Rebind<uint8_t, decltype(df)> du;
  size_t i = 0;
  for (; i + hn::Lanes(df) <= (size_t)n; i += hn::Lanes(df))
  {
    const auto r = hn::NearestInt(hn::LoadU(df, sum + i));
    hn::StoreU(hn::DemoteTo(du, hn::Min(hn::Max(r, hn::Zero(di)), hn::Set(di, 255))), du, out + i);
  }
  for (; i < (size_t)n; ++i)
  {
    const int v = (int)std::nearbyint(sum[i]);
    out[i] = (uint8_t)std::min(std::max(v, 0), 255);
  }
}

// The whole-number path's rounding: (sum + 2) >> 2 for 2x2, else sum * (1 / area) in float32
// rounded half to even -- OpenCV's resizeAreaFast_ on its two branches.
IF_STRICT_FN HWY_NOINLINE void area_int_round(const int32_t *IF_RESTRICT sum, int n, float scale,
                                              bool half_up_2x2, uint8_t *IF_RESTRICT out)
{
  IF_STRICT_BODY
  const hn::ScalableTag<int32_t> di;
  const hn::Rebind<float, decltype(di)> df;
  const hn::Rebind<uint8_t, decltype(di)> du;
  const auto s = hn::Set(df, scale);
  size_t i = 0;
  for (; i + hn::Lanes(di) <= (size_t)n; i += hn::Lanes(di))
  {
    const auto v = hn::LoadU(di, sum + i);
    const auto r = half_up_2x2 ? hn::ShiftRight<2>(hn::Add(v, hn::Set(di, 2)))
                               : hn::NearestInt(hn::Mul(hn::ConvertTo(df, v), s));
    hn::StoreU(hn::DemoteTo(du, hn::Min(hn::Max(r, hn::Zero(di)), hn::Set(di, 255))), du, out + i);
  }
  for (; i < (size_t)n; ++i)
  {
    const int v = half_up_2x2 ? (sum[i] + 2) >> 2 : (int)std::nearbyint((float)sum[i] * scale);
    out[i] = (uint8_t)std::min(std::max(v, 0), 255);
  }
}
#define IMFEAT_HPASS(G)                                                                            \
  template void area_hpass<G, 1>(const AreaHPlan &, const uint8_t *, size_t, float *, size_t);     \
  template void area_hpass<G, 2>(const AreaHPlan &, const uint8_t *, size_t, float *, size_t);     \
  template void area_hpass<G, 3>(const AreaHPlan &, const uint8_t *, size_t, float *, size_t);     \
  template void area_hpass<G, 4>(const AreaHPlan &, const uint8_t *, size_t, float *, size_t);
IMFEAT_HPASS(1)
IMFEAT_HPASS(2)
IMFEAT_HPASS(4)
#if HWY_MAX_BYTES >= 32
IMFEAT_HPASS(8)
#endif
#undef IMFEAT_HPASS
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#pragma fp_contract(on)
#endif

// The whole-number path's horizontal sums: the plan's patterns pick each column's block of
// source bytes, summed as integers.
template <int G>
HWY_NOINLINE void area_hsum(const AreaHPlan &p, const uint8_t *IF_RESTRICT src,
                            int32_t *IF_RESTRICT dst)
{
  const hn::FixedTag<uint8_t, 16> dwin;
  const hn::FixedTag<uint8_t, G> dg;
  const hn::Rebind<int32_t, decltype(dg)> di;
  for (size_t g = 0; g < (size_t)p.groups; ++g)
  {
    const auto win = hn::LoadU(dwin, src + p.w0[g]);
    const uint8_t *IF_RESTRICT pat = p.pattern.data() + g * p.slots * 16;
    auto acc = hn::Zero(di);
    for (size_t k = 0; k < (size_t)p.slots; ++k)
    {
      const auto sel = hn::TableLookupBytesOr0(win, hn::LoadU(dwin, pat + k * 16));
      acc = hn::Add(acc, hn::PromoteTo(di, hn::ResizeBitCast(dg, sel)));
    }
    if (g * G + G <= (size_t)p.dsize)
      hn::StoreU(acc, di, dst + g * G);
    else
      hn::StoreN(acc, di, dst + g * G, (size_t)p.dsize - g * G);
  }
}

void area_hsum_scalar(const AreaHPlan &p, const uint8_t *IF_RESTRICT src, int32_t *IF_RESTRICT dst)
{
  for (int i = 0; i < p.dsize; ++i)
  {
    int32_t acc = 0;
    for (int j = p.toff[i]; j < p.toff[i + 1]; ++j) acc += src[p.taps[j].si];
    dst[i] = acc;
  }
}

void area_iadd(const int32_t *IF_RESTRICT a, int n, int32_t *IF_RESTRICT dst)
{
  const hn::ScalableTag<int32_t> di;
  size_t i = 0;
  for (; i + hn::Lanes(di) <= (size_t)n; i += hn::Lanes(di))
    hn::StoreU(hn::Add(hn::LoadU(di, dst + i), hn::LoadU(di, a + i)), di, dst + i);
  for (; i < (size_t)n; ++i) dst[i] += a[i];
}

// One configured resize: the source and thumbnail sizes, which of OpenCV's three paths it
// takes, and the plans. Read-only once set up, so every band shares it.
struct AreaResizer
{
  enum Mode : uint8_t
  {
    COPY,    // the same size: cv::resize copies
    GENERIC, // float32 taps
    INTEGER  // whole-number ratios: block sums
  };
  int sh = 0, sw = 0, dh = 0, dw = 0;
  Mode mode = COPY;
  int ix = 1, iy = 1;   // INTEGER: the ratios
  bool half_up = false; // INTEGER: 2x2 on 1, 3 or 4 channels rounds half up
  float iscale = 1.0f;  // INTEGER: 1 / (ix * iy), as OpenCV forms it
  AreaHPlan hp;
  std::vector<int> yoff, ysi; // GENERIC: row dy's taps are ysi/ybeta[yoff[dy] .. yoff[dy+1])
  std::vector<float> ybeta;

  // Sets up the resize of (sh, sw) to (dh, dw) with `channels` channels; throws for upscaling.
  void setup(int sh_, int sw_, int dh_, int dw_, int channels)
  {
    sh = sh_, sw = sw_, dh = dh_, dw = dw_;
    if (dh < 1 || dw < 1)
      throw std::invalid_argument("the thumbnail needs at least one row and column");
    if (sh < dh || sw < dw)
      throw std::invalid_argument("resize_area downscales only, as cv2's INTER_AREA does: OpenCV "
                                  "turns bilinear for an upscale, so resize with cv2 first");
    hp = AreaHPlan();
    yoff.clear(), ysi.clear(), ybeta.clear();
    if (sh == dh && sw == dw)
    {
      mode = COPY;
      return;
    }
    const double scale_x = 1.0 / ((double)dw / sw), scale_y = 1.0 / ((double)dh / sh);
    const double rx = std::nearbyint(scale_x), ry = std::nearbyint(scale_y); // saturate_cast<int>
    const double eps = std::numeric_limits<double>::epsilon(); // DBL_EPSILON, as cv::resize
    if (std::fabs(scale_x - rx) < eps && std::fabs(scale_y - ry) < eps)
    {
      mode = INTEGER;
      ix = (int)rx, iy = (int)ry;
      half_up = ix == 2 && iy == 2 && (channels == 1 || channels == 3 || channels == 4);
      iscale = 1.0f / (float)(ix * iy);
      std::vector<std::vector<AreaTap>> t((size_t)dw);
      for (int dx = 0; dx < dw; ++dx)
        for (int k = 0; k < ix; ++k) t[dx].push_back({dx * ix + k, 1.0f});
      area_hplan(t, hp, false);
      return;
    }
    mode = GENERIC;
    area_hplan(area_taps(sw, dw), hp, true);
    const auto yt = area_taps(sh, dh);
    yoff.assign((size_t)dh + 1, 0);
    for (int dy = 0; dy < dh; ++dy)
    {
      for (const AreaTap &tap : yt[dy]) ysi.push_back(tap.si), ybeta.push_back(tap.alpha);
      yoff[dy + 1] = (int)ysi.size();
    }
  }
};

// One band's (or thread's) working set: a de-interleaved source row, the two most recently
// resampled source rows (a source row feeds at most the destination row it ends and the one it
// starts, in order, so two are enough), and the destination row's sums.
struct AreaScratch
{
  int planes = 0, plane_stride = 0, dw = 0;
  std::vector<uint8_t> src;        // [plane][sw + 32]: the taps read a 16-byte window, so slack
  std::vector<float> hrow[2];      // [plane][dw] resampled rows
  int hsy[2] = {-1, -1};           // ... and which source rows they hold
  std::vector<int32_t> isum, irow; // INTEGER: the block sums in flight, and one row's
  std::vector<float> sum;          // GENERIC: the destination row in flight

  void setup(const AreaResizer &r, int nplanes)
  {
    planes = nplanes;
    plane_stride = r.sw + 32;
    dw = r.dw;
    src.assign((size_t)planes * plane_stride, 0);
    for (auto &b : hrow) b.assign((size_t)planes * dw, 0.0f);
    isum.assign((size_t)planes * dw, 0);
    irow.assign((size_t)planes * dw, 0);
    sum.assign((size_t)planes * dw, 0.0f);
    reset();
  }
  void reset() { hsy[0] = hsy[1] = -1; }
};

// The GENERIC resample of source row sy on the planes the scratch holds, from the cache or afresh.
const float *area_resampled(const AreaResizer &r, AreaScratch &s, int sy)
{
  for (int c = 0; c < 2; ++c)
    if (s.hsy[c] == sy) return s.hrow[c].data();
  const int slot = s.hsy[0] < s.hsy[1] ? 0 : 1; // the older one goes
  float *out = s.hrow[slot].data();
  const uint8_t *IF_RESTRICT src = s.src.data();
  const size_t ss = (size_t)s.plane_stride, ds = (size_t)s.dw;
  auto planes = [&](auto kG) // the planes together (up to four: the channels a thumbnail has)
  {
    constexpr int G = decltype(kG)::value;
    switch (s.planes)
    {
    case 1: area_hpass<G, 1>(r.hp, src, ss, out, ds); break;
    case 2: area_hpass<G, 2>(r.hp, src, ss, out, ds); break;
    case 3: area_hpass<G, 3>(r.hp, src, ss, out, ds); break;
    default: area_hpass<G, 4>(r.hp, src, ss, out, ds); break;
    }
  };
  switch (r.hp.G)
  {
#if HWY_MAX_BYTES >= 32
  case 8: planes(std::integral_constant<int, 8>()); break;
#endif
  case 4: planes(std::integral_constant<int, 4>()); break;
  case 2: planes(std::integral_constant<int, 2>()); break;
  case 1: planes(std::integral_constant<int, 1>()); break;
  default:
    for (int p = 0; p < s.planes; ++p)
      area_hpass_scalar(r.hp, src + (size_t)p * ss, out + (size_t)p * ds);
    break;
  }
  s.hsy[slot] = sy;
  return out;
}

void area_hsum_row(const AreaResizer &r, const AreaScratch &s, int32_t *out)
{
  for (int p = 0; p < s.planes; ++p)
  {
    const uint8_t *IF_RESTRICT src = s.src.data() + (size_t)p * s.plane_stride;
    int32_t *IF_RESTRICT dst = out + (size_t)p * s.dw;
    switch (r.hp.G)
    {
#if HWY_MAX_BYTES >= 32
    case 8: area_hsum<8>(r.hp, src, dst); break;
#endif
    case 4: area_hsum<4>(r.hp, src, dst); break;
    case 2: area_hsum<2>(r.hp, src, dst); break;
    case 1: area_hsum<1>(r.hp, src, dst); break;
    default: area_hsum_scalar(r.hp, src, dst); break;
    }
  }
}

// Thumbnail row dy of the source image (rows rs apart, pixels cs apart, the wanted source
// channels at bytes coff[0 .. planes) of a pixel; `packed`: the pixels are `planes` contiguous
// bytes in channel order): `planes` planes of dw bytes, plane p at out + p * ostride.
void area_row(const AreaResizer &r, AreaScratch &s, const uint8_t *IF_RESTRICT img, int64_t rs,
              int64_t cs, const int64_t *IF_RESTRICT coff, bool packed, int dy,
              uint8_t *IF_RESTRICT out, size_t ostride)
{
  // the source row, one plane per wanted channel, in the scratch (gather_row's wide paths)
  auto load = [&](int sy)
  {
    gather_row(img + (int64_t)sy * rs, cs, coff, s.planes, r.sw, packed, nullptr, s.src.data(),
               (size_t)s.plane_stride);
  };
  const int n = s.planes * r.dw;
  switch (r.mode)
  {
  case AreaResizer::COPY:
    load(dy);
    for (int p = 0; p < s.planes; ++p)
      std::memcpy(out + (size_t)p * ostride, s.src.data() + (size_t)p * s.plane_stride,
                  (size_t)r.dw);
    return;
  case AreaResizer::INTEGER:
    for (int k = 0; k < r.iy; ++k)
    {
      load(dy * r.iy + k);
      area_hsum_row(r, s, k == 0 ? s.isum.data() : s.irow.data());
      if (k) area_iadd(s.irow.data(), n, s.isum.data());
    }
    for (int p = 0; p < s.planes; ++p)
      area_int_round(s.isum.data() + (size_t)p * r.dw, r.dw, r.iscale, r.half_up,
                     out + (size_t)p * ostride);
    return;
  case AreaResizer::GENERIC:
    for (int j = r.yoff[dy]; j < r.yoff[dy + 1]; ++j)
    {
      const int sy = r.ysi[j];
      bool cached = false;
      for (int c = 0; c < 2; ++c) cached = cached || s.hsy[c] == sy;
      if (!cached) load(sy);
      const float *row = area_resampled(r, s, sy);
      if (j == r.yoff[dy])
        area_vmul(row, n, r.ybeta[j], s.sum.data());
      else
        area_vmuladd(row, n, r.ybeta[j], s.sum.data());
    }
    for (int p = 0; p < s.planes; ++p)
      area_round(s.sum.data() + (size_t)p * r.dw, r.dw, out + (size_t)p * ostride);
    return;
  }
}

// `planes` planes of w bytes (plane p at src + p * sstride) interleaved into a row of pixels.
void interleave_row(const uint8_t *IF_RESTRICT src, size_t sstride, int planes, int w,
                    uint8_t *IF_RESTRICT dst)
{
  const size_t N = hn::Lanes(d8);
  size_t x = 0;
  if (planes == 1)
  {
    std::memcpy(dst, src, (size_t)w);
    return;
  }
  if (planes == 3)
    for (; x + N <= (size_t)w; x += N)
      hn::StoreInterleaved3(hn::LoadU(d8, src + x), hn::LoadU(d8, src + sstride + x),
                            hn::LoadU(d8, src + 2 * sstride + x), d8, dst + x * 3);
  else if (planes == 4)
    for (; x + N <= (size_t)w; x += N)
      hn::StoreInterleaved4(hn::LoadU(d8, src + x), hn::LoadU(d8, src + sstride + x),
                            hn::LoadU(d8, src + 2 * sstride + x),
                            hn::LoadU(d8, src + 3 * sstride + x), d8, dst + x * 4);
  else if (planes == 2)
    for (; x + N <= (size_t)w; x += N)
      hn::StoreInterleaved2(hn::LoadU(d8, src + x), hn::LoadU(d8, src + sstride + x), d8,
                            dst + x * 2);
  for (; x < (size_t)w; ++x)
    for (int p = 0; p < planes; ++p) dst[x * planes + p] = src[(size_t)p * sstride + x];
}

// One image row into the rolling window. With P == 1 the gathered rows are the planes.
// With P > 1 each row is then split by column phase: phase p holds columns p, p+P, ...
// so the sampled columns are exactly phase 0 and a tap d columns away is still a contiguous
// load, from phase (d % P). The columns just outside the image replicate the border, so
// block()'s 3x3 taps need no border logic: column -1 is lane -1 of the last phase, and
// column w (reachable only when P == 1) is lane w.
void deinterleave_row(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs,
                      const int64_t *IF_RESTRICT coff, const int *IF_RESTRICT chan, int conv, int c,
                      int w, int y, bool packed, int64_t prs, int slots, int slot, int P,
                      int64_t phs, uint8_t *IF_RESTRICT tmp, uint8_t *IF_RESTRICT dst)
{
  const uint8_t *IF_RESTRICT src = img + (int64_t)y * rs;
  uint8_t *IF_RESTRICT d0 = dst + (size_t)slot * (size_t)prs + PAD_L;
  const size_t pstride = (size_t)slots * (size_t)prs;
  uint8_t *IF_RESTRICT scratch = tmp + (size_t)c * (size_t)w;
  // Two phases of a raw row split as they are gathered: planar input straight from the
  // image, packed 2-4 channel input from the de-interleaved vectors, with no trip through
  // tmp. (More phases, or a colour conversion, gather first and split below.)
  if (P == 2 && conv == CONV_NONE && (cs == 1 || (packed && c >= 2 && c <= 4)))
  {
    const size_t N = hn::Lanes(d8);
    size_t x = 0;
    if (cs == 1)
      for (int k = 0; k < c; ++k)
        split_phases(src + coff[k], (size_t)w, 2, (size_t)phs, scratch, d0 + (size_t)k * pstride);
    else
    {
      auto put = [&](int k, V8 lo, V8 hi)
      {
        uint8_t *IF_RESTRICT b = d0 + (size_t)k * pstride + x / 2;
        hn::StoreU(hn::ConcatEven(d8, hi, lo), d8, b);
        hn::StoreU(hn::ConcatOdd(d8, hi, lo), d8, b + phs);
      };
      for (; x + 2 * N <= (size_t)w; x += 2 * N)
      {
        V8 a0, a1, b0, b1, e0, e1, g0, g1;
        if (c == 2)
        {
          hn::LoadInterleaved2(d8, src + x * 2, a0, b0);
          hn::LoadInterleaved2(d8, src + (x + N) * 2, a1, b1);
        }
        else if (c == 3)
        {
          hn::LoadInterleaved3(d8, src + x * 3, a0, b0, e0);
          hn::LoadInterleaved3(d8, src + (x + N) * 3, a1, b1, e1);
          put(2, e0, e1);
        }
        else
        {
          hn::LoadInterleaved4(d8, src + x * 4, a0, b0, e0, g0);
          hn::LoadInterleaved4(d8, src + (x + N) * 4, a1, b1, e1, g1);
          put(2, e0, e1), put(3, g0, g1);
        }
        put(0, a0, a1), put(1, b0, b1);
      }
      for (; x < (size_t)w; ++x)    // the tail, a pixel at a time: column x is lane x / 2 of
        for (int k = 0; k < c; ++k) // phase x % 2
          d0[(size_t)k * pstride + (x & 1) * (size_t)phs + x / 2] = src[(int64_t)x * cs + coff[k]];
    }
    for (int k = 0; k < c; ++k)
    {
      uint8_t *IF_RESTRICT b = d0 + (size_t)k * pstride;
      b[(int64_t)(P - 1) * phs - 1] = b[0];
    }
    return;
  }
  // straight into the window, or into tmp's first c planes for the split below
  uint8_t *IF_RESTRICT row = P == 1 ? d0 : tmp;
  const size_t rstride = P == 1 ? pstride : (size_t)w;
  if (conv == CONV_NONE)
    gather_row(src, cs, coff, c, w, packed, scratch, row, rstride);
  else // its own 3 planes of scratch, past the split's
    convert_row(conv, src, cs, coff, chan, c, w, packed, tmp + (size_t)3 * c * (size_t)w, row,
                rstride);
  for (int k = 0; k < c; ++k)
  {
    uint8_t *IF_RESTRICT b = d0 + (size_t)k * pstride;
    if (P > 1) split_phases(tmp + (size_t)k * (size_t)w, (size_t)w, P, (size_t)phs, scratch, b);
    b[(int64_t)(P - 1) * phs - 1] = b[0];
    if (P == 1) b[w] = b[w - 1];
  }
}

// Sum(v) per channel into the hash row for the current image row, read from phase 0 of the
// row's planes (`step` lanes apart there). Sampled columns are grouped (once, in set_config)
// into runs that share a hash column and lie in one finest cell, so each run is a lane range
// [beg,end) with a constant target cell: step == 1 is a plain contiguous sum that vectorizes.
void accumulate_hash_row(const uint8_t *IF_RESTRICT row, size_t pstride, int C, int step,
                         const int *IF_RESTRICT beg, const int *IF_RESTRICT end,
                         const int *IF_RESTRICT rhc, int nrun, int64_t *IF_RESTRICT hrowbase)
{
  for (int r = 0; r < nrun; ++r)
  {
    int64_t *IF_RESTRICT cell = hrowbase + (size_t)rhc[r] * C;
    const int b = beg[r], e = end[r];
    for (int k = 0; k < C; ++k)
    {
      const uint8_t *IF_RESTRICT v = row + (size_t)k * pstride;
      int64_t s = 0;
      int c = b;
      if (step == 1) // whole vectors of the run sum through psadbw; the compiler's own
      {              // vectorization widened every byte to int64
        const size_t N = hn::Lanes(d8);
        V64 acc = hn::Zero(d64);
        for (; c + (int)N <= e; c += (int)N)
          acc = hn::Add(acc, hn::BitCast(d64, hn::SumsOf8(hn::LoadU(d8, v + c))));
        s = hn::ReduceSum(d64, acc);
      }
      for (; c < e; c += step) s += v[c];
      cell[k] += s;
    }
  }
}

// features() derives straight into a block of memory that the returned arrays then share
// ownership of, so nothing is copied out. The block comes back here when the last of those
// arrays dies and the next call reuses it: in steady state a frame allocates nothing and
// touches no fresh pages. A store outlives its FeatureComputer while arrays are still out.
struct Blocks
{
  static constexpr size_t HDR = 64, KEEP = 4; // header (the owning store) and idle blocks kept
  std::mutex mu;
  std::vector<void *> idle;
  size_t bytes = 0;
  int out = 0; // blocks some array still views
  bool retired =
      false; // the FeatureComputer let go: nothing is reused, and the last give() deletes

  void *take()
  {
    const std::lock_guard<std::mutex> lock(mu);
    ++out;
    if (!idle.empty())
    {
      void *p = idle.back();
      idle.pop_back();
      return p;
    }
    auto *p = static_cast<uint8_t *>(::operator new(HDR + bytes, std::align_val_t(HDR)));
    *reinterpret_cast<Blocks **>(p) = this;
    return p + HDR;
  }
  static void *header(void *p) { return static_cast<uint8_t *>(p) - HDR; }
  static void drop(void *p) { ::operator delete(header(p), std::align_val_t(HDR)); }
  static void give(void *p) noexcept // the capsule destructor of a block's arrays
  {
    Blocks *b = *static_cast<Blocks **>(header(p));
    bool last;
    {
      const std::lock_guard<std::mutex> lock(b->mu);
      if (b->retired || b->idle.size() >= KEEP)
        drop(p);
      else
        b->idle.push_back(p);
      last = --b->out == 0 && b->retired;
    }
    if (last) delete b;
  }
  void retire()
  {
    bool last;
    {
      const std::lock_guard<std::mutex> lock(mu);
      for (void *p : idle) drop(p);
      idle.clear();
      retired = true;
      last = out == 0;
    }
    if (last) delete this;
  }
};

struct Level
{
  int ny = 0, nx = 0, fy = 0, fx = 0;
  std::vector<int64_t> buf;
  // Outputs, in the block of the features() call in flight; o* are their offsets in it.
  float *feat = nullptr;
  double *mom = nullptr;
  double *sum = nullptr; // (NMAP, c, NST) cross-cell summary of feat, then of mom
  size_t ofeat = 0, omom = 0, osum = 0;
  std::vector<size_t> rshape, fshape, mshape, sumshape;
};

class FeatureComputer
{
  int h_ = 0, w_ = 0, c_ = 1, sy_ = 1, sx_ = 1; // the thumbnail's rows and columns when resize_
  bool interleaved_ = false;  // source rows are already the (H,W,C) layout we want
  int conv_ = CONV_NONE;      // colour conversion fused into the row gather (see Convert)
  std::vector<int> chan_;     // the selected channels, of the converted image when conv_
  std::vector<int64_t> coff_; // per-selected-channel (per-source-channel when conv_) offset
                              // within a pixel
  // The thumbnail resize fused into the row gather (see AreaResizer): the frame is (sh_, sw_)
  // with csrc_ channels and every row the pass reads is made from it on the way in.
  bool resize_ = false;
  int sh_ = 0, sw_ = 0, csrc_ = 1;
  AreaResizer area_;
  std::vector<int> need_;         // the source channels a thumbnail row needs, in order
  std::vector<int64_t> coff_src_; // ... their offsets within a frame pixel
  std::vector<int64_t> coff_rz_;  // per selected channel (per source one under conv_): its
                                  // plane's offset in the resized row
  bool packed_src_ = false;       // the frame's pixels are csrc_ contiguous bytes, all wanted
  uint8_t *thumb_out_ = nullptr;  // features(): where to write the thumbnail, if asked
  int64_t thumb_rs_ = 0;
  std::vector<Level> levels_;
  int16_t ray_[HB - 1][2] = {}; // bin-boundary ray j as (cx, -cy); block() tests the low HRAYS
  int hog_card_[HB] = {};       // 1 for the axis-aligned (cardinal) orientation bins
  std::vector<uint8_t> bvalid_; // per lag: 255 where both +/-d column taps are in range
  size_t bvalid_stride_ = 0;
  int phases_ = 1;       // column phases the planes are split into (1 = not split)
  int wrows_ = 9;        // plane window rows (see the window comment)
  int bard_lagmask_ = 0; // bit j set iff bard lag j fits the image (see bard())
  int group_ = 1;        // channels a block carries together: c_ up to GMAX, else 1

  // How block()'s blocks map to cells, in LANES of phase 0 (a cell is cw_ of them, every
  // step_-th one sampled). A block either spans whole cells (multi_: cw_ divides it) or starts
  // at a cell edge with the lanes past the cell masked off; kmask_ is that cell-relative mesh.
  int cw_ = 1, cw_log2_ = 0, step_ = 1, ncols_ = 1;
  // Nothing per row, per block or per cell divides: what would is worked out once per
  // configuration, or carried along as a running count.
  int64_t tapcol_[KTAP] = {}; // each block() column tap, as an offset into a row of planes
  size_t sub_rows_ = 1;       // sampled rows per block() sub-chunk: their distinct rows' responses
  size_t rd_stride_ = 3;      // stay in L1; and the most distinct rows a sub-chunk can have
  size_t rslots_ = 3;         // response slots a sampled row advances: min(sy_, 3)
  size_t block_cells_ = 1;    // cells a block spans (multi_)
  size_t lanes_ = 0;          // lanes in a row of one phase plane: w_ / phases_
  int64_t pstride_ = 0;       // a phase plane's stride: the lanes, the pads, rounded to a line
  int up_shift_ = 0;          // log2 of level 1's fy: parents are whole powers of two of cells
  bool multi_ = false, masked_ = false;
  std::vector<uint8_t> kmask_, ktail_; // ... and the row's last block, when it overruns the row

  // Per-band state: the plane window and everything a cell row in flight touches.
  struct Scratch
  {
    Acc acc;                        // the block accumulators (one block of one channel at a time)
    std::vector<int> vok;           // [sampled row] bard lags whose rows are inside the image
    std::vector<int> rws;           // the cell row's sampled rows
    std::vector<int> rdn;           // block()'s row tables: the distinct rows of each sub-chunk
    std::vector<int64_t> rdist;     // (Chunk::rdn) and their plane offsets (Chunk::rdist)
    std::vector<int16_t> resp;      // the sub-chunk's row responses (Chunk::resp)
    std::vector<int32_t> hbuf;      // ... and HOG inputs (Chunk::hbuf)
    std::vector<int64_t> cells;     // [channel][K_N][cell]: the cell row's sums
    std::vector<int64_t> hsum;      // private hash partial; bands >0 only
    std::vector<int64_t> row, graw; // features(): the cell row in flight, and the band's share
                                    // of the global sums
    std::vector<int64_t>
        up;           // features(): the level-1 cell row in flight, as kernel sums like `cells`
    int64_t up_n = 0; // ... and its sample count per cell so far
    std::vector<int64_t> held;   // ... and its first rows, when their parents belong to the
    std::vector<int> held_cy;    // band above: rolled up after the join
    std::vector<uint8_t> planes; // padded de-interleaved planes: the rolling window
    std::vector<uint8_t> lin;    // one row, channels contiguous, then the phase split's scratch
    AreaScratch area;            // resize_: the resize's working set, and the band's rows of
    std::vector<uint8_t> line;   // the thumbnail: from image row line0, need_ planes of w_ bytes
    int line0 = 0;               // each, the row gather's input
    uint8_t *pl = nullptr;       // the window, 64-byte aligned inside `planes`
    int64_t row_stride = 0;      // padded plane row stride
    int64_t phase_stride = 0;    // stride between a row's phase planes
    int wbase = 0, filled = 0;   // rows [wbase, filled) of the window are loaded, wbase at slot 0
  };
  std::vector<Scratch> scr_;

  // A band is a contiguous run of finest cell rows. Two bands therefore never touch
  // the same accumulator, so the hot path needs no atomics and no locks -- and since
  // every accumulator is an int64 sum that cannot overflow, the totals are identical
  // to the serial order however the rows are split. Only the coarse-level rollup,
  // derive() and the hash readout run after the join, all still serial.
  struct Band
  {
    int r0;       // the band's first sampled image row
    int cy0, cy1; // the finest cell rows they land in, [cy0, cy1)
  };
  enum Job : uint8_t
  {
    JOB_ACC,
    JOB_DERIVE
  };
  Job job_ = JOB_ACC;
  bool stream_ = false; // features(): the finest level is derived row by row and never stored
  bool rolls_ = false;  // ... block by block, its sums going straight up to level 1 (roll_block)
  std::vector<Band> bands_;

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_go_, cv_done_;
  const uint8_t *task_img_ = nullptr;
  int64_t task_rs_ = 0, task_cs_ = 0;
  uint64_t epoch_ = 0;
  int pending_ = 0;
  bool quit_ = false;

  // Perceptual-hash state (see HN/HS). Filled in the accumulate pass, derived at the end.
  std::vector<int64_t> hsum_;        // HN*HN*c_ : Sum(v) per hash cell per channel
  std::vector<int64_t> hcnt_;        // HN*HN : sampled-pixel count per cell (fixed per config)
  std::vector<double> hcntd_, hden_; // ... as doubles, and with 1 for 0: what the means divide by
  std::vector<int> hrow_, hcol_;     // row -> hash row (h_), col -> hash col (w_)
  std::vector<int> hrbeg_, hrend_, hrhc_; // per-run column range [beg,end) and its hash col
  std::vector<double> dctb_;              // HS*HN : DCT-II basis rows (scipy type-2, norm=None)
  uint64_t *hash_ = nullptr;     // (NHASH, c_) in the output block: aHash, wHash, pHash rows
  std::vector<size_t> hshape_;   // {NHASH, c_}
  bool hfast_ = false;           // finest grid tiles HN -> hashes reuse its S1
  std::vector<uint8_t> up_done_; // features(): level-1 rows derived in their band (flush_up)

  std::vector<int64_t> graw_;
  float *gfeat_ = nullptr; // the global outputs, in the output block
  double *gmom_ = nullptr;
  size_t ogfeat_ = 0, ogmom_ = 0, ohash_ = 0;
  Blocks *blocks_ = nullptr; // output blocks of this configuration
  // reusable cross-cell summary accumulators (sized c_*NF / c_*NMOM), refilled per level
  std::vector<float> sfmn_, sfmx_;
  std::vector<double> sfsm_, sfsq_, smmn_, smmx_, smsm_, smsq_;
  std::vector<size_t> graw_shape_, gfeat_shape_, gmom_shape_;

  // `count` consecutive cell-channels from their accumulator slots, LD at a time. A short last
  // group repeats its last record, and scatter() drops the repeats.
  void derive_records(const int64_t *IF_RESTRICT rec, const double *IF_RESTRICT mom,
                      const double *IF_RESTRICT invn, size_t count, float *IF_RESTRICT f) const
  {
    hn::VFromD<DDV> S[NSUM], M[NMOM];
    hn::VFromD<hn::Rebind<float, DDV>> out[NMAP];
    for (size_t i = 0; i < count; i += LD)
    {
      size_t at[LD];
      double v[LD];
      for (int l = 0; l < LD; ++l) at[l] = std::min(i + l, count - 1);
      auto lanes = [&](auto value)
      {
        for (int l = 0; l < LD; ++l) v[l] = value(at[l]);
        return gather(v);
      };
      for (int t = 0; t < NSUM; ++t)
        S[t] = lanes([&](size_t r) { return (double)rec[r * NSUM + t]; });
      for (int t = 0; t < NMOM; ++t) M[t] = lanes([&](size_t r) { return mom[r * NMOM + t]; });
      derive_lanes(S, M, lanes([&](size_t r) { return invn[r]; }), hog_card_, out);
      scatter(out, std::min<size_t>(LD, count - i), NMAP, f + i * NMAP);
    }
  }

  // The finest level, LD cells a vector, straight from a block's cell sums (`nxs` apart, the
  // block's first cell at index 0), which are contiguous per sum where the slots are not. The
  // conversions are exact: a sum below 2^52 is planted in the mantissa of 2^52 and that is
  // subtracted back off; a signed one is biased by 2^51 first; sum|grad|^4 can be wider, so its
  // halves convert separately and meet in one rounding. `cell` is the first cell's index.
  void derive_block(const Level &L, const int64_t *IF_RESTRICT sums, size_t nxs, size_t cell,
                    size_t cells, int64_t n)
  {
    const DDV dd;
    const hn::RebindToSigned<DDV> di;
    const hn::Rebind<float, DDV> df;
    using V = hn::VFromD<DDV>;
    const double invn = 1.0 / (double)n;
    const V two52 = hn::Set(dd, 4503599627370496.0), two51 = hn::Set(dd, 2251799813685248.0);
    for (int k = 0; k < c_; ++k)
      for (size_t i = 0; i < cells; i += LD)
      {
        const int64_t *IF_RESTRICT cb = sums + (size_t)k * K_N * nxs + i;
        auto lanes_of = [&](int t) { return hn::LoadU(di, cb + (size_t)t * nxs); };
        auto u52 = [&](auto v) { return hn::Sub(hn::Or(hn::BitCast(dd, v), two52), two52); };
        auto sum = [&](int t) { return u52(lanes_of(t)); };
        auto sum_signed = [&](int t)
        { return hn::Sub(u52(hn::Add(lanes_of(t), hn::Set(di, int64_t{1} << 51))), two51); };
        V S[NSUM], M[NMOM];
        hn::VFromD<decltype(df)> out[NMAP];
        S[SXX] = sum(K_SXX), S[SYY] = sum(K_SYY), S[SXY] = sum_signed(K_SXY);
        S[CNT] = hn::Set(dd, (double)n);
        const V tot = hn::Add(S[SXX], S[SYY]);
        V past = tot, rest = S[CNT]; // as the flush derives these slots
        V cum4[HB - 1];
        hog_cum4(sum_signed, tot, cum4);
        for (int j = 0; j < HB - 1; ++j)
        { // a multiple of four quartered: exact
          const V cum = hn::Mul(cum4[j], hn::Set(dd, 0.25));
          S[HOG0 + j] = hn::Sub(past, cum);
          past = cum;
        }
        S[HOG0 + HB - 1] = past;
        for (int j = 0; j < LBPB - 1; ++j) rest = hn::Sub(rest, S[LBP0 + j] = sum(K_LBP + j));
        S[LBP0 + LBPB - 1] = rest;
        S[NMAX] = sum(K_MAX), S[NMIN] = sum(K_MIN);
        const auto g4 = lanes_of(K_SG4);
        S[SG4] =
            hn::Add(hn::Mul(u52(hn::ShiftRight<32>(hn::BitCast(hn::RebindToUnsigned<DDV>(), g4))),
                            hn::Set(dd, 4294967296.0)),
                    u52(hn::And(g4, hn::Set(di, 0xFFFFFFFFLL))));
        for (int t = 0; t < BARD_NMAP; ++t) S[BARD0 + t] = sum(K_BARD + t);
        S[LAP1] = sum_signed(K_LAP1), S[LAP2] = sum(K_LAP2);
        for (int t = 0; t < NLAWS; ++t) S[LAWS0 + t] = sum(K_LAWS + t);
        // the moments, from the kernel's power sums about KBIAS: a block's cells hold at most
        // KROWS rows of half a block of columns, far below the 2^20 samples that stay exact
        const V a[NMOM] = {sum_signed(K_T1), sum(K_T2), sum_signed(K_T3), sum(K_T4)};
        V t[NMOM], q;
        moments_recentre(a, (double)n, invn, t, q);
        moments_tail(t, hn::Add(q, hn::Set(dd, (double)KBIAS)), hn::Set(dd, invn), M);
        store_moments(M, std::min<size_t>(LD, cells - i), (size_t)c_ * NMOM,
                      L.mom + ((cell + i) * c_ + k) * NMOM);
        derive_lanes(S, M, hn::Set(dd, invn), hog_card_, out);
        float *IF_RESTRICT f = L.feat + ((cell + i) * c_ + k) * NMAP;
        scatter(out, std::min<size_t>(LD, cells - i), (size_t)c_ * NMAP, f);
      }
  }
  // LD doubles into one vector without a trip through memory (which would stall the load).
  static hn::VFromD<DDV> gather(const double *m)
  {
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
    const hn::FixedTag<double, 2> d2;
    return hn::Combine(DDV(), hn::Dup128VecFromValues(d2, m[2], m[3]),
                       hn::Dup128VecFromValues(d2, m[0], m[1]));
#elif HWY_MAX_BYTES >= 16
    return hn::Dup128VecFromValues(DDV(), m[0], m[1]);
#else
    return hn::Set(DDV(), m[0]);
#endif
  }
  // out[j] holds feature j of LD cells; cell l's NMAP features go to f + l * stride. Four at a
  // time, transposed; the last four overlap the ones before, which rewrites the same values.
  static void scatter(const hn::VFromD<hn::Rebind<float, DDV>> *IF_RESTRICT out, size_t cells,
                      size_t stride, float *IF_RESTRICT f)
  {
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
    const hn::Rebind<float, DDV> d;
    auto group = [&](int j, auto &&store) // features j..j+3 of the four cells, transposed
    {
      const auto lo01 = hn::InterleaveLower(d, out[j], out[j + 1]),
                 hi01 = hn::InterleaveUpper(d, out[j], out[j + 1]),
                 lo23 = hn::InterleaveLower(d, out[j + 2], out[j + 3]),
                 hi23 = hn::InterleaveUpper(d, out[j + 2], out[j + 3]);
      store(hn::ConcatLowerLower(d, lo23, lo01), hn::ConcatUpperUpper(d, lo23, lo01),
            hn::ConcatLowerLower(d, hi23, hi01), hn::ConcatUpperUpper(d, hi23, hi01), j);
    };
    static_assert(NMAP >= 4, "the last group overlaps the one before");
    if (HWY_LIKELY(cells == LD)) // a full vector of cells: nothing to decide per lane
    {
      float *IF_RESTRICT f1 = f + stride, *IF_RESTRICT f2 = f + 2 * stride,
                         *IF_RESTRICT f3 = f + 3 * stride;
      auto all = [&](auto r0, auto r1, auto r2, auto r3, int j)
      {
        hn::StoreU(r0, d, f + j);
        hn::StoreU(r1, d, f1 + j);
        hn::StoreU(r2, d, f2 + j);
        hn::StoreU(r3, d, f3 + j);
      };
      for (int j = 0; j + 4 <= NMAP; j += 4) group(j, all);
      if (NMAP % 4) group(NMAP - 4, all);
    }
    else
    {
      auto some = [&](auto r0, auto r1, auto r2, auto r3, int j)
      {
        const decltype(r0) row[4] = {r0, r1, r2, r3};
        for (size_t l = 0; l < cells; ++l) hn::StoreU(row[l], d, f + l * stride + j);
      };
      for (int j = 0; j + 4 <= NMAP; j += 4) group(j, some);
      if (NMAP % 4) group(NMAP - 4, some);
    }
#else
    for (int j = 0; j < NMAP; ++j)
    {
      float v[LD];
      hn::StoreU(out[j], hn::Rebind<float, DDV>(), v);
      for (size_t l = 0; l < cells; ++l) f[l * stride + j] = v[l];
    }
#endif
  }

  // Central moments [mean, var, m3, m4] (float64) from the raw power sums.
  // Shifting by K = round(mean) before the binomial expansion keeps every
  // intermediate exact in int64 and kills the cancellation that S2/n - mean^2
  // suffers on low-variance cells; the residual d = (S1 - nK)/n is <= 0.5.
  // Moments of `count` consecutive cell-channels from their slots, LD at a time: the power sums
  // re-centred on the rounded mean k in exact int64 (any n), then moments_tail().
  void moments_records(const int64_t *IF_RESTRICT rec, const double *IF_RESTRICT invn, size_t count,
                       double *IF_RESTRICT mom) const
  {
    hn::VFromD<DDV> t[NMOM], M[NMOM];
    for (size_t i = 0; i < count; i += LD)
    {
      double v[NMOM + 2][LD];
      for (int l = 0; l < LD; ++l)
      {
        const size_t r = std::min(i + l, count - 1);
        const int64_t *IF_RESTRICT s = rec + r * NSUM;
        const int64_t n = s[CNT], s1 = s[PS0], s2 = s[PS0 + 1], s3 = s[PS0 + 2], s4 = s[PS0 + 3];
        // k = floor((s1 + n/2) / n) without the divide: the quotient by 1/n is within one of
        // it, and the remainder says which way
        const int64_t num = s1 + (n >> 1);
        int64_t k = (int64_t)((double)num * invn[r]);
        const int64_t rem = num - k * n;
        k += (rem >= n && n > 0) - (rem < 0);
        const int64_t k2 = k * k, k3 = k2 * k;
        v[0][l] = (double)(s1 - n * k);
        v[1][l] = (double)(s2 - 2 * k * s1 + n * k2);
        v[2][l] = (double)(s3 - 3 * k * s2 + 3 * k2 * s1 - n * k3);
        v[3][l] = (double)(s4 - 4 * k * s3 + 6 * k2 * s2 - 4 * k3 * s1 + n * k3 * k);
        v[4][l] = (double)k;
        v[5][l] = invn[r];
      }
      for (int j = 0; j < NMOM; ++j) t[j] = gather(v[j]);
      moments_tail(t, gather(v[4]), gather(v[5]), M);
      store_moments(M, std::min<size_t>(LD, count - i), NMOM, mom + i * NMOM);
    }
  }
  // M[j] holds moment j of LD cell-channels; cell-channel l's four go to mom + l * stride.
  static void store_moments(const hn::VFromD<DDV> *IF_RESTRICT M, size_t lanes, size_t stride,
                            double *IF_RESTRICT mom)
  {
    const DDV dd;
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
    const auto lo01 = hn::InterleaveLower(dd, M[0], M[1]),
               hi01 = hn::InterleaveUpper(dd, M[0], M[1]),
               lo23 = hn::InterleaveLower(dd, M[2], M[3]),
               hi23 = hn::InterleaveUpper(dd, M[2], M[3]);
    const hn::VFromD<DDV> row[4] = {
        hn::ConcatLowerLower(dd, lo23, lo01), hn::ConcatLowerLower(dd, hi23, hi01),
        hn::ConcatUpperUpper(dd, lo23, lo01), hn::ConcatUpperUpper(dd, hi23, hi01)};
    for (size_t l = 0; l < lanes; ++l) hn::StoreU(row[l], dd, mom + l * stride);
#else
    for (int j = 0; j < NMOM; ++j)
    {
      double v[LD];
      hn::StoreU(M[j], dd, v);
      for (size_t l = 0; l < lanes; ++l) mom[l * stride + j] = v[l];
    }
#endif
  }

  // Stride between the per-type rows of Scratch::cells. A block overrunning the row folds its
  // surplus lanes into up to N cells nobody reads; and the stride is odd, because a power of
  // two would put the same cell of every type on the same few cache sets.
  size_t cells_stride() const
  {
    const size_t N = hn::Lanes(d8), lanes = (size_t)(w_ / phases_);
    return ((size_t)levels_[0].nx + (multi_ && lanes % N ? N : 0)) | 1;
  }

  // One cell row's sums, from its sampled rows `rws` (all resident in the plane window).
  // Blocks outside, rows inside: a block's accumulators stay in registers down the rows and
  // fold into the cell sums once, which are then written -- not added -- to the pyramid.
  // Returns whether it also derived the row (features() only): block by block, while the
  // block's sums are still laid out contiguously per sum, which is what derive_block() reads.
  bool cell_row(Scratch &s, const std::vector<int> &rws, size_t cy, bool rolls,
                int64_t *IF_RESTRICT hs, int64_t *IF_RESTRICT dst)
  {
    const size_t N = hn::Lanes(d8), nfx = (size_t)levels_[0].nx;
    // When a block holds whole cells and the cell row is one chunk of rows, a block finishes
    // its cells: they are flushed at once from a block's worth of sums that never leaves L1.
    // Otherwise the sums are kept for the whole cell row and flushed at its end.
    const bool local = rws.size() <= (size_t)KROWS;
    const int64_t n = (int64_t)rws.size() * ncols_; // samples per cell, the same for the whole row
    // every channel in one pass of the blocks: a block's cells then derive as they are folded
    // (a block of one or two cells fills a quarter or half of derive's vectors, which still
    // costs a fraction of flushing them to slots and gathering those back)
    const bool derives = stream_ && local && group_ == c_;
    // The cells folded, derived and rolled up together: a block's, or as many blocks' as fill
    // derive's vectors, so that a block of one or two cells is not derived at a quarter or half
    // of a vector and rolls up through the vector pair path. Their sums are contiguous per kind
    // (what derive_block and the roll-up read); a row of cells kept to the end (local == false,
    // cells taller than a chunk) keeps the whole-row layout, a block of one cell's kinds
    // contiguous there, so that its fold touches five cache lines per block and not one per
    // kind (which, over a cell row, evicted block()'s row buffers from L1).
    const size_t group = local ? std::max(block_cells_, (size_t)LD) : 1;
    const size_t nxs = local ? group | 1 : cells_stride(); // never above cells_stride()
    const size_t kst = multi_ || local ? nxs : 1, cst = multi_ || local ? 1 : (size_t)K_N;
    const size_t pstride = (size_t)wrows_ * (size_t)s.row_stride, lanes = lanes_;
    V16 rays[HRAYS]; // the low rays as (-cy, cx), against block()'s (|gx|, |gy|) lanes
    for (int j = 0; j < HRAYS; ++j)
      rays[j] = hn::InterleaveLower(d16, hn::Set(d16, ray_[j][1]), hn::Set(d16, ray_[j][0]));
    const int G = group_;
    for (size_t q0 = 0; q0 < rws.size(); q0 += KROWS)
    {
      const size_t nr = std::min<size_t>(KROWS, rws.size() - q0);
      for (size_t q = 0; q < nr; ++q) // (a lag's rows are read only where they are in the image)
      {
        const int r = rws[q0 + q];
        s.vok[q] = 0;
        for (int j = 0; j < BARD_NL; ++j)
          if (r >= BARD_LAGS[j] && r < h_ - BARD_LAGS[j]) s.vok[q] |= bard_lagmask_ & (1 << j);
      }
      // A tap is a row of the window plus its column offset (tapcol_): the window holds the
      // chunk's rows in order, sy_ rows from a sampled row to the next, the rows past the image
      // edge copies of the edge row, so every row tap is a constant offset from the centre row.
      Chunk C;
      C.nr = nr;
      C.row0 = (int64_t)(rws[q0] - s.wbase) * s.row_stride + PAD_L;
      C.rs = s.row_stride;
      C.rowstep = (int64_t)sy_ * s.row_stride;
      C.vok = s.vok.data();
      C.allok = bard_lagmask_ == (1 << BARD_NL) - 1;
      C.rstep = rslots_ * RESP_STRIDE * hn::Lanes(d16);
      C.rdist = s.rdist.data();
      C.rdn = s.rdn.data();
      C.rdstride = rd_stride_;
      C.subrows = sub_rows_;
      C.whole = multi_ && cw_ < (int)N / 2;
      C.tapcol = tapcol_;
      C.resp = aligned64(s.resp);
      C.hbuf = aligned64(s.hbuf);
      C.inlane = block_cells_ == 1;
      C.rows1 = rslots_ == 1;
      for (size_t sub = 0, qs = 0; qs < nr; ++sub, qs += sub_rows_)
      {
        int64_t *IF_RESTRICT dist = &s.rdist[sub * rd_stride_];
        const size_t qe = std::min(nr, qs + sub_rows_);
        // block()'s row tables: per sub-chunk, the distinct rows its sampled rows read, a row's
        // three at slots rslots_ apart (see Chunk): a slot two rows share is written twice, alike
        for (size_t q = qs; q < qe; ++q)
          for (int i = 0; i < 3; ++i)
            dist[rslots_ * (q - qs) + i] = C.row0 + (int64_t)q * C.rowstep + (i - 1) * C.rs;
        s.rdn[sub] = (int)(rslots_ * (qe - qs - 1) + 3);
      }
      for (int k0 = 0; k0 < c_; k0 += G)
        for (size_t x0 = 0, cell = 0; x0 < lanes;
             x0 += multi_ ? N : (size_t)cw_, cell += block_cells_)
          for (size_t xv = 0; xv < (multi_ ? N : (size_t)cw_); xv += N)
          {
            const size_t x = x0 + xv, e0 = local ? cell % group : cell; // cell == x0 / cw_
            // Blocks that span cells can overrun the row: the surplus lanes are masked off too.
            const bool tail = multi_ && x + N > lanes, msk = masked_ || tail;
            const uint8_t *IF_RESTRICT mask = tail ? ktail_.data() : kmask_.data() + xv;
            // only a block within MAXLAG columns of a row end can hold an invalid bard tap
            const bool edge =
                x * phases_ < (size_t)BARD_MAXLAG || (x + N) * phases_ + BARD_MAXLAG > (size_t)w_;
            const uint8_t *IF_RESTRICT valid = edge ? bvalid_.data() : nullptr;
            // Lanes -> cells, a quarter block (or a half: quarters 0 and 2) at a time. The block
            // holds g lanes per cell, so the part starting at lane l belongs to cell e0 + l / g; a
            // block no wider than a cell (g == N) is all e0. A part adds to its cell unless it is
            // the first to reach it: the first rows of the cell row, the first lanes of the cell.
            const size_t g = multi_ ? (size_t)cw_ : N, gq = std::min(g, N / 4);
            size_t at[4];
            bool adds[4];
            for (size_t q = 0, l = 0; q < 4; ++q, l += N / 4)
            {
              at[q] = e0 + (g < N ? l >> cw_log2_ : 0);
              adds[q] = q0 > 0 || (g < N ? l & (g - 1) : xv + l) > 0;
            }
            const V8 mv = hn::LoadU(d8, mask);
            for (int k = 0; k < G; ++k) // every row of the chunk, channel by channel: block()
            {                           // then its fold, one set of accumulators serving all
              Acc &A = s.acc;
              C.base = s.pl + (size_t)(k0 + k) * pstride + x;
              const uint8_t *IF_RESTRICT vx = valid ? valid + x : nullptr;
              if (msk)
                phases_ == 1
                    ? block<true, true>(C, mask, vx, bvalid_stride_, bard_lagmask_, rays, A)
                    : block<true, false>(C, mask, vx, bvalid_stride_, bard_lagmask_, rays, A);
              else
                phases_ == 1
                    ? block<false, true>(C, mask, vx, bvalid_stride_, bard_lagmask_, rays, A)
                    : block<false, false>(C, mask, vx, bvalid_stride_, bard_lagmask_, rays, A);
              int64_t *IF_RESTRICT cb = s.cells.data() + (size_t)(k0 + k) * K_N * nxs;
              if (!multi_) // one cell (at[] all the same): its kinds kst apart
              {
                fold_cell(A, msk, mv, adds[0], cb + at[0] * cst, kst); // (the first part's flag)
                continue;
              }
              for (int t = K_U8; t < K_N; ++t)
              {
                const V8 u8 = t < K_BARD ? (msk ? hn::And(A.u8[t - K_U8], mv) : A.u8[t - K_U8])
                                         : hn::Zero(d8);
                if (g >= 8) // cells of whole octets: both kinds sum to uint64 lanes of 8 columns
                {           // in one step (psadbw for the bytes), and the cells all add alike
                  const V64 w = hn::BitCast(d64, t < K_BARD ? hn::SumsOf8(u8)
                                                            : hn::SumsOf4(A.bard[t - K_BARD]));
                  fold(d64, w, g / 8, adds[0], cb + t * kst + at[0] * cst);
                  continue;
                }
                if (g == 4) // cells of quads: uint32 lanes of 4 columns, one a cell
                {
                  const auto w = t < K_BARD ? hn::SumsOf4(u8) : hn::SumsOf2(A.bard[t - K_BARD]);
                  fold(d64, hn::BitCast(d64, hn::PromoteLowerTo(du64, w)), 1, adds[0],
                       cb + t * kst + at[0] * cst);
                  fold(d64, hn::BitCast(d64, hn::PromoteUpperTo(du64, w)), 1, adds[2],
                       cb + t * kst + at[2] * cst);
                  continue;
                }
                const auto sums = t < K_BARD ? hn::SumsOf2(u8) : A.bard[t - K_BARD];
                fold32<false>(hn::BitCast(d32, hn::PromoteLowerTo(du32, sums)), g / 2, adds[0],
                              cb + t * kst + at[0] * cst);
                fold32<false>(hn::BitCast(d32, hn::PromoteUpperTo(du32, sums)), g / 2, adds[2],
                              cb + t * kst + at[2] * cst);
              }
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
              if (FOLDH)
              {
                for (int t = 0; t < K_NUNSIGNED; ++t)
                  fold_pair_multi<false>(A.pair[0][t], A.pair[1][t], g, adds, at, cb + t * kst,
                                         cst);
                for (int t = K_NUNSIGNED; t < K_NPAIR; ++t)
                  fold_pair_multi<true>(A.pair[0][t], A.pair[1][t], g, adds, at, cb + t * kst, cst);
              }
#endif
              for (size_t h = 0; h < 2; ++h)
              {
                const bool ad = adds[2 * h];
                int64_t *IF_RESTRICT ch = cb + at[2 * h] * cst;
                if (!FOLDH)
                {
                  for (int t = 0; t < K_NUNSIGNED; ++t)
                    fold32<false>(A.pair[h][t], g / 2, ad, ch + t * kst);
                  for (int t = K_NUNSIGNED; t < K_NPAIR; ++t)
                    fold32<true>(A.pair[h][t], g / 2, ad, ch + t * kst);
                }
                fold(d64, A.t4[h], g / 4, ad, ch + K_T4 * kst);
              }
              for (int q = 0; q < 4; ++q)
              {
                int64_t *IF_RESTRICT cq = cb + at[q] * cst;
                fold(d64, A.sg4[q], gq / 2, adds[q], cq + K_SG4 * kst);
                if (FOLDH) continue; // the HOG sums of all four quarters fold together below
                for (int j = 0; j < K_NHOG; ++j) // signed sums
                  fold32<true>(A.hog[q][j], gq, adds[q], cq + (K_HOG + j) * kst);
              }
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
              if (FOLDH) fold_hog_multi(A, g, adds, at, cb, kst, cst);
#endif
            }
            if (local && xv + N >= (multi_ ? N : (size_t)cw_) && // the group's last block
                (e0 + block_cells_ >= group || cell + block_cells_ >= nfx))
            {
              const size_t e = cell - e0, cells = std::min(group, nfx - e);
              if (!(derives && rolls)) flush(s.cells.data(), nxs, nxs, 1, e, cells, k0, G, n, dst);
              if (derives)
              {
                derive_block(levels_[0], s.cells.data(), nxs, cy * nfx + e, cells, n);
                if (bands_.size() == 1) summary_fold(levels_[0], cy * nfx + e, cells);
                if (rolls) roll_block(s, s.cells.data(), nxs, cy, e, cells, n, hs);
              }
            }
          }
    }
    if (!local) flush(s.cells.data(), nxs, kst, cst, 0, nfx, 0, c_, n, dst);
    if (derives && rolls) s.up_n += n * levels_[1].fx;
    return derives;
  }

  // features(): a block's cell sums straight up into the level-1 row in flight -- the finest
  // cells are never turned into slots at all. Sums add as they are (the slots are linear in
  // them and in n, so the conversion waits for flush_up()); the first child row of a parent
  // stores. The hash grid takes its S1.
  void roll_block(Scratch &s, const int64_t *IF_RESTRICT sums, size_t nxs, size_t cy, size_t e,
                  size_t cells, int64_t n, int64_t *IF_RESTRICT hs) const
  {
    const Level &fine = levels_[0];
    const Level &lv = levels_[1];
    // fx, fy and the hash block are powers of two: every index below is a shift or a mask
    auto log2 = [](size_t v)
    {
      size_t l = 0;
      while ((size_t{1} << l) < v) ++l;
      return l;
    };
    const size_t L = hn::Lanes(d64), fx = (size_t)lv.fx, fxl = log2(fx), nxu = (size_t)lv.nx + L;
    const size_t kinds = (size_t)c_ * K_N;
    const bool top = (cy & ((size_t)lv.fy - 1)) == 0; // the first child row of its parents
    auto first = [&](size_t cell) { return top && (cell & (fx - 1)) == 0; };
    if (fx == 2 && L >= 2 && cells % L == 0) // the usual pyramid: neighbours pair up
      top ? pair_up<true>(sums, nxs, s.up.data() + (e >> 1), nxu, cells, kinds)
          : pair_up<false>(sums, nxs, s.up.data() + (e >> 1), nxu, cells, kinds);
    else
      for (size_t r = 0; r < kinds; ++r)
        for (size_t i = 0; i < cells; ++i)
        {
          int64_t &up = s.up[r * nxu + ((e + i) >> fxl)];
          up = first(e + i) ? sums[r * nxs + i] : up + sums[r * nxs + i];
        }
    if (hfast_) // the finest grid tiles the hash grid: block-sum its S1 = T1 + KBIAS * n
    {
      const size_t bxl = log2((size_t)fine.nx / HN);
      int64_t *IF_RESTRICT hrow = hs + (cy >> log2((size_t)fine.ny / HN)) * HN * c_;
      for (size_t i = 0; i < cells; ++i)
        for (int k = 0; k < c_; ++k)
          hrow[((e + i) >> bxl) * c_ + k] += sums[((size_t)k * K_N + K_T1) * nxs + i] + KBIAS * n;
    }
  }
  // `kinds` rows of `cells` sums, neighbours added in pairs into (TOP: written over) their parents.
  template <bool TOP>
  static void pair_up(const int64_t *IF_RESTRICT src, size_t nxs, int64_t *IF_RESTRICT dst,
                      size_t nxu, size_t cells, size_t kinds)
  {
    const hn::Half<D64> dh;
    const size_t L = hn::Lanes(d64);
    if (cells == 2 * L) // the usual block: one vector of parents per kind, and nothing to loop over
    {
      for (size_t r = 0; r < kinds; ++r, src += nxs, dst += nxu)
      {
        const V64 a = hn::LoadU(d64, src), b = hn::LoadU(d64, src + L);
        const V64 sum2 = hn::Add(hn::ConcatEven(d64, b, a), hn::ConcatOdd(d64, b, a));
        hn::StoreU(TOP ? sum2 : hn::Add(hn::LoadU(d64, dst), sum2), d64, dst);
      }
      return;
    }
    for (size_t r = 0; r < kinds; ++r, src += nxs, dst += nxu)
    {
      size_t i = 0;
      for (; i + 2 * L <= cells; i += 2 * L) // two vectors of children, one of parents
      {
        const V64 a = hn::LoadU(d64, src + i), b = hn::LoadU(d64, src + i + L);
        const V64 sum2 = hn::Add(hn::ConcatEven(d64, b, a), hn::ConcatOdd(d64, b, a));
        hn::StoreU(TOP ? sum2 : hn::Add(hn::LoadU(d64, dst + i / 2), sum2), d64, dst + i / 2);
      }
      for (; i < cells; i += L)
      {
        const V64 v = hn::LoadU(d64, src + i);
        const auto sum2 =
            hn::LowerHalf(dh, hn::Add(hn::ConcatEven(d64, v, v), hn::ConcatOdd(d64, v, v)));
        hn::StoreU(TOP ? sum2 : hn::Add(hn::LoadU(dh, dst + i / 2), sum2), dh, dst + i / 2);
      }
    }
  }
  // The level-1 row in flight is complete (or the band ends): its sums become slots.
  // `complete`: every finest row of this level-1 row was this band's, so its cells are final
  // and derive now, from the sums while they are in cache; otherwise (and for the coarser
  // levels) derive_cells() derives them from their slots after the join.
  void flush_up(Scratch &s, size_t cy, bool complete)
  {
    Level &lv = levels_[1];
    const size_t nxu = (size_t)lv.nx + hn::Lanes(d64), row = cy >> up_shift_;
    int64_t *IF_RESTRICT dst = lv.buf.data() + row * lv.nx * c_ * NSUM;
    if (FLUSH4)
      flush4(s.up.data(), nxu, (size_t)lv.nx, s.up_n, dst);
    else
      flush(s.up.data(), nxu, nxu, 1, 0, (size_t)lv.nx, 0, c_, s.up_n, dst);
    // (the double re-centring of the moments is exact below 2^20 samples; derive_cells()'s
    // integer one takes anything)
    if (complete && s.up_n > 0 && s.up_n < (int64_t{1} << 20))
    {
      derive_block(lv, s.up.data(), nxu, row * lv.nx, (size_t)lv.nx, s.up_n);
      up_done_[row] = 1;
    }
    s.up_n = 0;
  }
  // flush() for a kind-major source (cells contiguous, kinds `kst` apart), four cells at a
  // time: each slot as a vector of four cells, the slots then transposed into the records
  // four at a time. The same integer arithmetic as flush(); KBIAS is a power of two, so the
  // binomial coefficients are shifts and adds.
  static constexpr bool FLUSH4 = HWY_MAX_LANES_D(D64) == 4 && !HWY_HAVE_SCALABLE;
  void flush4(const int64_t *IF_RESTRICT sums, size_t kst, size_t cells, int64_t n,
              int64_t *IF_RESTRICT dst) const
  {
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
    static_assert(KBIAS == 128 && NSUM % 4 == 0, "flush4 shifts by KBIAS and transposes 4x4");
    const int64_t B = KBIAS;
    size_t i = 0;
    for (; i + 4 <= cells; i += 4)
      for (int k = 0; k < c_; ++k)
      {
        const int64_t *IF_RESTRICT cb = sums + (size_t)k * K_N * kst + i;
        auto sum = [&](int t) { return hn::LoadU(d64, cb + (size_t)t * kst); };
        auto times3 = [](V64 v) { return hn::Add(v, hn::Add(v, v)); };
        V64 a[NSUM];
        a[SXX] = sum(K_SXX), a[SYY] = sum(K_SYY), a[SXY] = sum(K_SXY);
        a[CNT] = hn::Set(d64, n);
        const V64 tot = hn::Add(a[SXX], a[SYY]);
        V64 past = tot, cum4[HB - 1];
        hog_cum4(sum, tot, cum4);
        for (int j = 0; j < HB - 1; ++j)
        {
          const V64 cum = hn::ShiftRight<2>(cum4[j]); // (never negative)
          a[HOG0 + j] = hn::Sub(past, cum);
          past = cum;
        }
        a[HOG0 + HB - 1] = past;
        a[NMAX] = sum(K_MAX), a[NMIN] = sum(K_MIN);
        const V64 t1 = sum(K_T1), t2 = sum(K_T2), t3 = sum(K_T3), t4 = sum(K_T4);
        a[PS0] = hn::Add(t1, hn::Set(d64, B * n));
        a[PS0 + 1] = hn::Add(hn::Add(t2, hn::ShiftLeft<8>(t1)), hn::Set(d64, B * B * n));
        a[PS0 + 2] = hn::Add(hn::Add(t3, times3(hn::ShiftLeft<7>(t2))),
                             hn::Add(times3(hn::ShiftLeft<14>(t1)), hn::Set(d64, B * B * B * n)));
        a[PS0 + 3] =
            hn::Add(hn::Add(hn::Add(t4, hn::ShiftLeft<9>(t3)),
                            hn::Add(hn::ShiftLeft<15>(t2), hn::ShiftLeft<16>(t2))), // 6 B^2 t2
                    hn::Add(hn::ShiftLeft<23>(t1), hn::Set(d64, B * B * B * B * n)));
        V64 rest = a[CNT];
        for (int j = 0; j < LBPB - 1; ++j) rest = hn::Sub(rest, a[LBP0 + j] = sum(K_LBP + j));
        a[LBP0 + LBPB - 1] = rest;
        a[SG4] = sum(K_SG4);
        for (int t = 0; t < BARD_NMAP; ++t) a[BARD0 + t] = sum(K_BARD + t);
        a[LAP1] = sum(K_LAP1), a[LAP2] = sum(K_LAP2);
        for (int t = 0; t < NLAWS; ++t) a[LAWS0 + t] = sum(K_LAWS + t);
        int64_t *IF_RESTRICT rec[4]; // the four cells' records
        for (size_t c = 0; c < 4; ++c) rec[c] = dst + ((i + c) * (size_t)c_ + (size_t)k) * NSUM;
        for (int g = 0; g < NSUM; g += 4) // four slots of four cells -> four cells' four slots
        {
          const V64 t01 = hn::InterleaveLower(d64, a[g], a[g + 1]),
                    u01 = hn::InterleaveUpper(d64, a[g], a[g + 1]),
                    t23 = hn::InterleaveLower(d64, a[g + 2], a[g + 3]),
                    u23 = hn::InterleaveUpper(d64, a[g + 2], a[g + 3]);
          hn::StoreU(hn::ConcatLowerLower(d64, t23, t01), d64, rec[0] + g);
          hn::StoreU(hn::ConcatLowerLower(d64, u23, u01), d64, rec[1] + g);
          hn::StoreU(hn::ConcatUpperUpper(d64, t23, t01), d64, rec[2] + g);
          hn::StoreU(hn::ConcatUpperUpper(d64, u23, u01), d64, rec[3] + g);
        }
      }
    if (i < cells) flush(sums + i, kst, kst, 1, i, cells - i, 0, c_, n, dst);
#else
    flush(sums, kst, kst, 1, 0, cells, 0, c_, n, dst);
#endif
  }

  // Cell sums -> accumulator slots, for the `cells` cells from cell0 on that the sums hold (a
  // block's, or the whole row's), kinds `kst` apart and cells `cst` apart (one of them 1), a
  // channel's K_N * nxs long, and for channels [k0, k0 + G).
  void flush(const int64_t *IF_RESTRICT sums, size_t nxs, size_t kst, size_t cst, size_t cell0,
             size_t cells, int k0, int G, int64_t n, int64_t *IF_RESTRICT dst) const
  {
    const int64_t B = KBIAS;
    for (size_t i = 0; i < cells; ++i)
    {
      const size_t e = cell0 + i, src = i; // the sums start at the first cell they hold
      for (int k = k0; k < k0 + G; ++k)
      {
        int64_t *IF_RESTRICT a = dst + (e * c_ + k) * NSUM;
        const int64_t *IF_RESTRICT cb = sums + (size_t)k * K_N * nxs + src * cst;
        auto sum = [&](int t) { return cb[(size_t)t * kst]; };
        a[SXX] = sum(K_SXX);
        a[SYY] = sum(K_SYY);
        a[SXY] = sum(K_SXY);
        a[CNT] = n;
        const int64_t tot = a[SXX] + a[SYY]; // every pixel is past "ray -1": the total |grad|^2
        int64_t past = tot, cum4[HB - 1];
        hog_cum4(sum, tot, cum4);
        for (int j = 0; j < HB - 1; ++j)
        {
          const int64_t cum = cum4[j] >> 2;
          a[HOG0 + j] = past - cum;
          past = cum;
        }
        a[HOG0 + HB - 1] = past;
        a[NMAX] = sum(K_MAX);
        a[NMIN] = sum(K_MIN);
        // raw power sums of v = w + B, by the binomial expansion
        const int64_t t1 = sum(K_T1), t2 = sum(K_T2), t3 = sum(K_T3), t4 = sum(K_T4);
        a[PS0] = t1 + B * n;
        a[PS0 + 1] = t2 + 2 * B * t1 + B * B * n;
        a[PS0 + 2] = t3 + 3 * B * t2 + 3 * B * B * t1 + B * B * B * n;
        a[PS0 + 3] = t4 + 4 * B * t3 + 6 * B * B * t2 + 4 * B * B * B * t1 + B * B * B * B * n;
        int64_t rest = n;
        for (int j = 0; j < LBPB - 1; ++j) rest -= a[LBP0 + j] = sum(K_LBP + j);
        a[LBP0 + LBPB - 1] = rest;
        a[SG4] = sum(K_SG4);
        for (int t = 0; t < BARD_NMAP; ++t) // cover, per-lag response mass, dark, light
          a[BARD0 + t] = sum(K_BARD + t);
        a[LAP1] = sum(K_LAP1);
        a[LAP2] = sum(K_LAP2);
        for (int t = 0; t < NLAWS; ++t) a[LAWS0 + t] = sum(K_LAWS + t);
      }
    }
  }

  // One band's rows. Reads only its own Scratch and writes only its own cell rows.
  void accumulate_band(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int bi)
  {
    Level &fine = levels_[0];
    const size_t nfx = (size_t)fine.nx;
    Scratch &s = scr_[bi];
    int64_t *IF_RESTRICT hs = bi ? s.hsum.data() : hsum_.data();
    if (resize_)
    {
      // The band's rows of the thumbnail, its halo included, made up front from the frame's
      // rows: one streaming pass with the resize's working set to itself, which measured
      // faster than a row at a time between the kernel's cell rows.
      s.area.planes = (int)need_.size();
      s.area.reset();
      s.line0 = std::max(0, bands_[bi].r0 - BARD_MAXLAG);
      const int y1 = std::min(h_, bands_[bi].cy1 * (h_ / fine.ny) + BARD_MAXLAG);
      const size_t line_bytes = (size_t)need_.size() * (size_t)w_;
      for (int y = s.line0; y < y1; ++y)
      {
        uint8_t *line = s.line.data() + (size_t)(y - s.line0) * line_bytes;
        area_row(area_, s.area, img, rs, cs, coff_src_.data(), packed_src_, y, line, (size_t)w_);
        if (thumb_out_)
          interleave_row(line, (size_t)w_, csrc_, w_, thumb_out_ + (int64_t)y * thumb_rs_);
      }
    }
    if (stream_)
    {
      std::fill(s.graw.begin(), s.graw.end(), (int64_t)0);
      s.held.clear();
      s.held_cy.clear();
      if (bands_.size() == 1) summary_reset();
    }
    // The rolling window fills as the cell rows advance, from the first row any tap can reach.
    s.wbase = s.filled = bands_[bi].r0 - BARD_MAXLAG; // (rows above the image: copies of row 0)
    const size_t pstride = (size_t)wrows_ * (size_t)s.row_stride;
    // A cell row's sampled rows all fold into the same cells, so everything runs per cell
    // row, once the window holds those rows and their halo.
    const int chh = h_ / fine.ny;
    std::vector<int> &rows = s.rws;
    // the band's sampled rows in order: the next one and its window slot
    int next = (bands_[bi].cy0 * chh + sy_ - 1) / sy_ * sy_;
    rows.clear(); // (the last frame's last cell row)
    for (int cy = bands_[bi].cy0; cy < bands_[bi].cy1; ++cy)
    {
      rows.clear();
      for (; next < (cy + 1) * chh; next += sy_) rows.push_back(next);
      // raw(): the row goes into the pyramid. features(): into a row buffer that stays in cache
      // while stream_row() derives it and rolls it up, so the finest level is never stored.
      int64_t *IF_RESTRICT dst = stream_ ? s.row.data() : fine.buf.data() + cy * nfx * c_ * NSUM;
      // A band rolls a row up from its sums when it also has the first row of that row's parents;
      // otherwise the row becomes slots, which wait for the join (see Scratch::held).
      const int fy1 = 1 << up_shift_; // level 1's fy, or 1 without a level 1
      const bool owner = (cy & ~(fy1 - 1)) >= bands_[bi].cy0, rolls = stream_ && rolls_ && owner;
      bool derived = false;
      if (rows.empty()) // the stride skipped the whole cell row. Every other one is written, not
      {                 // added to, by cell_row(), so nothing else ever needs zeroing.
        std::fill_n(dst, nfx * c_ * NSUM, (int64_t)0);
      }
      else
      {
        // The window holds rows [wbase, filled) at slots from 0: the cell row's rows and their
        // halo among them. When the next cell row's would run past its end, the rows it keeps
        // (the halo the two share) slide to the front, so block() reads every row of a chunk at
        // a constant offset from the one before, with no wrap to look up. Rows past the image
        // are copies of its edge rows, so the taps need no clamping either.
        const int lo = rows.front() - BARD_MAXLAG, hi = rows.back() + BARD_MAXLAG + 1;
        if (hi - s.wbase > wrows_)
        {
          const int keep = s.filled - lo;
          if (keep > 0)
            for (int k = 0; k < c_; ++k)
            {
              uint8_t *IF_RESTRICT p = s.pl + (size_t)k * pstride;
              std::memmove(p, p + (size_t)(lo - s.wbase) * s.row_stride,
                           (size_t)keep * s.row_stride);
            }
          else
            s.filled = lo;
          s.wbase = lo;
        }
        for (; s.filled < hi; ++s.filled)
        {
          const int y = clampi(s.filled, h_ - 1), slot = s.filled - s.wbase;
          if (resize_) // the thumbnail row the band made, read as a planar row
            deinterleave_row(s.line.data() + (size_t)(y - s.line0) * need_.size() * w_, 0, 1,
                             coff_rz_.data(), chan_.data(), conv_, c_, w_, 0, false, s.row_stride,
                             wrows_, slot, phases_, s.phase_stride, s.lin.data(), s.pl);
          else
            deinterleave_row(img, rs, cs, coff_.data(), chan_.data(), conv_, c_, w_, y,
                             interleaved_, s.row_stride, wrows_, slot, phases_, s.phase_stride,
                             s.lin.data(), s.pl);
        }
        derived = cell_row(s, rows, (size_t)cy, rolls, hs, dst);
        if (!hfast_)
          for (size_t i = 0; i < rows.size(); ++i)
            accumulate_hash_row(s.pl + (size_t)(rows[i] - s.wbase) * s.row_stride + PAD_L, pstride,
                                c_, step_, hrbeg_.data(), hrend_.data(), hrhc_.data(),
                                (int)hrhc_.size(), hs + (size_t)hrow_[rows[i]] * HN * c_);
      }
      if (rolls)
      {
        if (rows.empty()) // nothing went up: derive the zeros, and start the parents at zero
        {
          derive_run(fine, (size_t)cy * nfx, nfx, dst, bands_.size() == 1);
          if ((cy & (fy1 - 1)) == 0) std::fill(s.up.begin(), s.up.end(), (int64_t)0);
        }
        if ((cy & (fy1 - 1)) == fy1 - 1 || cy == bands_[bi].cy1 - 1)
          flush_up(s, (size_t)cy, (cy & (fy1 - 1)) == fy1 - 1); // (rolls: the first row is ours)
      }
      else if (stream_)
        stream_row(s, cy, owner, derived, hs);
    }
    if (stream_ && bands_.size() == 1) summary_write(fine);
  }

  // features(): everything the finest cell row cy is needed for, while it is still in cache.
  // `owner`: this band also has the first row of cy's parents, so it may write them.
  void stream_row(Scratch &s, int cy, bool owner, bool derived, int64_t *IF_RESTRICT hs)
  {
    const Level &fine = levels_[0];
    const size_t nfx = (size_t)fine.nx, stripe = (size_t)c_ * NSUM;
    if (hfast_) // the finest grid tiles the hash grid: block-sum its S1
    {
      const size_t by = (size_t)fine.ny / HN, bx = nfx / HN;
      for (size_t e = 0; e < nfx; ++e)
        for (int k = 0; k < c_; ++k)
          hs[(((size_t)cy / by) * HN + e / bx) * c_ + k] +=
              s.row[e * stripe + (size_t)k * NSUM + PS0];
    }
    if (!derived) // cell_row() does it block by block when it can
      derive_run(fine, (size_t)cy * nfx, nfx, s.row.data(), bands_.size() == 1);
    if (levels_.size() == 1) // no coarser level: straight into the band's share of the global
    {
      for (size_t e = 0; e < nfx; ++e)
        for (size_t t = 0; t < stripe; ++t) s.graw[t] += s.row[e * stripe + t];
      return;
    }
    if (!owner)
    {
      s.held.insert(s.held.end(), s.row.begin(), s.row.end());
      s.held_cy.push_back(cy);
      return;
    }
    roll_row(cy, s.row.data());
  }
  // The finest row cy (sums at row) into its level-1 parents: a parent is written by its first
  // child and added to by the rest.
  void roll_row(int cy, const int64_t *IF_RESTRICT row)
  {
    Level &lv = levels_[1];
    const size_t i = (size_t)cy / lv.fy, stripe = (size_t)c_ * NSUM;
    for (size_t j = 0; j < (size_t)lv.nx; ++j)
    {
      int64_t *IF_RESTRICT dc = lv.buf.data() + (i * lv.nx + j) * stripe;
      for (size_t kx = 0; kx < (size_t)lv.fx; ++kx)
      {
        const int64_t *IF_RESTRICT sc = row + (j * lv.fx + kx) * stripe;
        if (cy % lv.fy == 0 && kx == 0)
        {
          std::copy(sc, sc + stripe, dc);
          continue;
        }
        for (size_t t = 0; t < stripe; ++t) dc[t] += sc[t];
      }
    }
  }

  void worker(int bi)
  {
    uint64_t seen = 0;
    for (;;)
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_go_.wait(lk, [&] { return quit_ || epoch_ != seen; });
      if (quit_) return;
      seen = epoch_;
      const Job j = job_;
      lk.unlock();
      if (j == JOB_ACC)
        accumulate_band(task_img_, task_rs_, task_cs_, bi);
      else
        derive_cells(bi);
      lk.lock();
      if (--pending_ == 0) cv_done_.notify_one();
    }
  }

  // Workers park on a condvar between frames, so a frame costs one broadcast and one
  // barrier rather than thread creation. Parking (not spinning) is deliberate: the
  // cores are expected to be shared with other real-time work between frames.
  void run_job(Job j)
  {
    const int nb = (int)bands_.size();
    auto one = [&](int bi)
    {
      if (j == JOB_ACC)
        accumulate_band(task_img_, task_rs_, task_cs_, bi);
      else
        derive_cells(bi);
    };
    if (workers_.empty())
    {
      for (int bi = 0; bi < nb; ++bi) one(bi);
      return;
    }
    {
      const std::lock_guard<std::mutex> lk(mu_);
      job_ = j;
      pending_ = nb - 1;
      ++epoch_;
    }
    cv_go_.notify_all();
    one(0); // the calling thread takes the first band
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return pending_ == 0; });
  }

  void pool_stop()
  {
    if (workers_.empty()) return;
    {
      const std::lock_guard<std::mutex> lk(mu_);
      quit_ = true;
    }
    cv_go_.notify_all();
    for (std::thread &t : workers_) t.join();
    workers_.clear();
    quit_ = false;
  }

  void accumulate(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs,
                  uint8_t *thumb_out)
  {
    thumb_out_ = thumb_out;
    thumb_rs_ = (int64_t)w_ * csrc_;
    const bool hashing = stream_ || !hfast_; // the hash grid sums: per pixel, or from streamed rows
    if (hashing)
    {
      std::fill(hsum_.begin(), hsum_.end(), (int64_t)0);
      for (size_t b = 1; b < scr_.size(); ++b)
        std::fill(scr_[b].hsum.begin(), scr_[b].hsum.end(), (int64_t)0);
    }

    if (resize_)
    {
      // A thumbnail row is made of the source channels the pass needs: all of them under a
      // conversion (it reads three) or for the thumbnail output, else the selected ones.
      need_.clear();
      if (conv_ != CONV_NONE || thumb_out)
        for (int k = 0; k < csrc_; ++k) need_.push_back(k);
      else
      {
        need_ = chan_;
        std::sort(need_.begin(), need_.end());
        need_.erase(std::unique(need_.begin(), need_.end()), need_.end());
      }
      for (size_t p = 0; p < need_.size(); ++p) coff_src_[p] = (int64_t)need_[p] * chs;
      packed_src_ = chs == 1 && cs == csrc_ && (int)need_.size() == csrc_;
      // the resized row holds need_'s planes w_ apart: where each channel the gather wants is
      if (conv_ != CONV_NONE)
        for (int k = 0; k < 3; ++k) coff_rz_[k] = (int64_t)k * w_;
      else
        for (int k = 0; k < c_; ++k)
        {
          const auto at = std::find(need_.begin(), need_.end(), chan_[k]) - need_.begin();
          coff_rz_[k] = (int64_t)at * w_;
        }
      interleaved_ = false;
    }
    else if (conv_ == CONV_NONE)
    {
      interleaved_ = chs == 1 && cs == c_; // ... and the selection is every channel in order
      for (int k = 0; k < c_; ++k)
      {
        coff_[k] = (int64_t)chan_[k] * chs;
        interleaved_ = interleaved_ && chan_[k] == k;
      }
    }
    else // the conversion reads all three source channels whatever is selected
    {
      for (int k = 0; k < 3; ++k) coff_[k] = (int64_t)k * chs;
      interleaved_ = chs == 1 && cs == 3;
    }
    task_img_ = img;
    task_rs_ = rs;
    task_cs_ = cs;
    std::fill(up_done_.begin(), up_done_.end(), (uint8_t)0);
    run_job(JOB_ACC);

    if (hashing) // fold the bands' private hash partials; int64 sums, so still exact
      for (size_t b = 1; b < scr_.size(); ++b)
        for (size_t t = 0; t < hsum_.size(); ++t) hsum_[t] += scr_[b].hsum[t];

    const int stripe = c_ * NSUM;
    for (const Scratch &sc : scr_) // the rows whose parents another band had to write first
      for (size_t r = 0; r < sc.held_cy.size(); ++r)
        roll_row(sc.held_cy[r], sc.held.data() + r * levels_[0].nx * stripe);
    for (size_t lvl = stream_ ? 2 : 1; lvl < levels_.size(); ++lvl) // stream_row() fed level 1
    {
      const Level &p = levels_[lvl - 1];
      Level &lv = levels_[lvl];
      for (int i = 0; i < lv.ny; ++i) // a parent is written as its first child, then added to
        for (int j = 0; j < lv.nx; ++j)
        {
          const size_t dst = (size_t)i * lv.nx + j;
          int64_t *IF_RESTRICT dc = lv.buf.data() + dst * stripe;
          for (int ky = 0, first = 1; ky < lv.fy; ++ky)
            for (int kx = 0; kx < lv.fx; ++kx, first = 0)
            {
              const size_t src = (size_t)(i * lv.fy + ky) * p.nx + (size_t)(j * lv.fx + kx);
              const int64_t *IF_RESTRICT sc = p.buf.data() + src * stripe;
              if (first)
              {
                std::copy(sc, sc + stripe, dc);
                continue;
              }
              for (int t = 0; t < stripe; ++t) dc[t] += sc[t];
            }
        }
    }

    const Level &last = levels_.back();
    std::fill(graw_.begin(), graw_.end(), (int64_t)0);
    if (stream_ && levels_.size() == 1) // the only level was never stored: the bands summed it
    {
      for (const Scratch &sc : scr_)
        for (int t = 0; t < stripe; ++t) graw_[t] += sc.graw[t];
      return;
    }
    for (size_t cell = 0, nc = (size_t)last.ny * last.nx; cell < nc; ++cell)
    {
      const int64_t *IF_RESTRICT s = last.buf.data() + cell * stripe;
      for (int t = 0; t < stripe; ++t) graw_[t] += s[t];
    }
  }

  // Reduce one just-derived per-cell vector (v[0..n)) into running summary
  // accumulators for its channel: min, max, sum, sum of squares (float64 sums
  // for precision; finalized to mean/std after the level's cells are all seen).
  // Fold `cells` consecutive cells' n values (rows `stride` apart) into the running min, max,
  // sum and sum of squares. Each accumulator rides in a register over the whole batch instead of
  // round-tripping through memory per cell. Cells are still added one by one, in order, and each
  // square is rounded before it is added (IF_UNFUSED), so the sums round as they always have.
  template <typename T>
  IF_UNFUSED_FN static void fold_summary(const T *IF_RESTRICT v, size_t stride, size_t cells, int n,
                                         T *IF_RESTRICT mn, T *IF_RESTRICT mx,
                                         double *IF_RESTRICT sm, double *IF_RESTRICT sq)
  {
    IF_UNFUSED_BODY
    // One pass over the cells per group of values: min, max, sum and sum of squares together,
    // all in registers. A group is a vector of T -- two vectors of doubles when T is float --
    // then ever narrower ones for the ragged tail, a lane's operations the same at any width.
    int j = 0;
    constexpr size_t W = 8 * sizeof(float) / sizeof(T); // <= 4 doubles, however wide T is
    for (size_t lanes = hn::Lanes(hn::CappedTag<T, W>()); lanes; lanes /= 2)
      for (; j + (int)lanes <= n; j += (int)lanes) switch (lanes)
        {
        case 8:
          fold_group(hn::CappedTag<T, 8>(), v + j, stride, cells, mn + j, mx + j, sm + j, sq + j);
          break;
        case 4:
          fold_group(hn::CappedTag<T, 4>(), v + j, stride, cells, mn + j, mx + j, sm + j, sq + j);
          break;
        case 2:
          fold_group(hn::CappedTag<T, 2>(), v + j, stride, cells, mn + j, mx + j, sm + j, sq + j);
          break;
        default:
          fold_group(hn::CappedTag<T, 1>(), v + j, stride, cells, mn + j, mx + j, sm + j, sq + j);
        }
  }
  template <class D, typename T = hn::TFromD<D>>
  IF_UNFUSED_FN static HWY_INLINE void
  fold_group(D d, const T *IF_RESTRICT v, size_t stride, size_t cells, T *IF_RESTRICT mn,
             T *IF_RESTRICT mx, double *IF_RESTRICT sm, double *IF_RESTRICT sq)
  {
    IF_UNFUSED_BODY
    const hn::Rebind<double, hn::Half<D>> dh; // half as many doubles: a float vector's two halves
    const hn::Rebind<double, D> dw; // ... or, where T is double or D a single lane, all of it
    constexpr bool split = sizeof(T) < sizeof(double) && HWY_MAX_LANES_D(D) > 1;
    auto lo = hn::LoadU(d, mn), hi = hn::LoadU(d, mx);
    if constexpr (split)
    {
      const size_t H = hn::Lanes(dh);
      auto s1 = hn::LoadU(dh, sm), s2 = hn::LoadU(dh, sq), t1 = hn::LoadU(dh, sm + H),
           t2 = hn::LoadU(dh, sq + H);
      for (size_t c = 0; c < cells; ++c)
      {
        const auto x = hn::LoadU(d, v + c * stride);
        lo = hn::Min(lo, x);
        hi = hn::Max(hi, x);
        const auto a = hn::PromoteLowerTo(dh, x), b = hn::PromoteUpperTo(dh, x);
        s1 = hn::Add(s1, a);
        s2 = hn::Add(s2, hn::Mul(a, a));
        t1 = hn::Add(t1, b);
        t2 = hn::Add(t2, hn::Mul(b, b));
      }
      hn::StoreU(s1, dh, sm), hn::StoreU(s2, dh, sq), hn::StoreU(t1, dh, sm + H),
          hn::StoreU(t2, dh, sq + H);
    }
    else
    {
      auto s1 = hn::LoadU(dw, sm), s2 = hn::LoadU(dw, sq);
      for (size_t c = 0; c < cells; ++c)
      {
        const auto x = hn::LoadU(d, v + c * stride);
        lo = hn::Min(lo, x);
        hi = hn::Max(hi, x);
        const auto a = as_double(dw, x);
        s1 = hn::Add(s1, a);
        s2 = hn::Add(s2, hn::Mul(a, a));
      }
      hn::StoreU(s1, dw, sm), hn::StoreU(s2, dw, sq);
    }
    hn::StoreU(lo, d, mn), hn::StoreU(hi, d, mx);
  }
  template <class DW, class V>
  static HWY_INLINE hn::VFromD<DW> as_double(DW dw, V x)
  {
    if constexpr (sizeof(hn::TFromV<V>) == sizeof(double))
      return x;
    else
      return hn::PromoteTo(dw, x);
  }

  template <typename T>
  static void write_summary(int n, size_t nc, const T *IF_RESTRICT mn, const T *IF_RESTRICT mx,
                            const double *IF_RESTRICT sm, const double *IF_RESTRICT sq,
                            double *IF_RESTRICT out, size_t stride)
  {
    const double invn = nc > 0 ? 1.0 / (double)nc : 0.0;
    for (int j = 0; j < n; ++j) // a float32 feature's stats are float32 values, widened exactly
    {
      const double mean = sm[j] * invn, var = sq[j] * invn - mean * mean;
      double *IF_RESTRICT o = out + (size_t)j * stride;
      o[0] = mn[j];
      o[1] = mx[j];
      o[2] = (T)mean;
      o[3] = (T)std::sqrt(var > 0.0 ? var : 0.0);
    }
  }

  // aHash/wHash/pHash from the accumulated HN x HN value-sum grid, one hash per channel.
  // A pure readout of hsum_: no image traversal. Bits are packed row-major, MSB first.
  void derive_hashes()
  {
    const int NHC = HN * HN;
    std::vector<double> mean(NHC + LD), d0((size_t)HS * HN);
    for (int k = 0; k < c_; ++k)
    {
      // the same division per cell, a vector of cells at a time (an empty cell divides by 1)
      for (int i = 0; i < NHC; i += LD)
      {
        double num[LD];
        for (size_t l = 0; l < LD; ++l) num[l] = (double)hsum_[(size_t)(i + l) * c_ + k];
        const auto den = hn::LoadU(DDV(), hden_.data() + i);
        const auto quo = hn::Div(gather(num), den);
        hn::StoreU(
            hn::IfThenElseZero(hn::Gt(hn::LoadU(DDV(), hcntd_.data() + i), hn::Zero(DDV())), quo),
            DDV(), mean.data() + i);
      }

      // aHash/wHash: pool the HN grid to 8x8, threshold by the mean / median of the 64.
      double m8[HS * HS];
      for (int bi = 0; bi < HS; ++bi)
        for (int bj = 0; bj < HS; ++bj)
        {
          int64_t s = 0, n = 0;
          for (int di = 0; di < HBLK; ++di)
            for (int dj = 0; dj < HBLK; ++dj)
            {
              const int idx = (bi * HBLK + di) * HN + (bj * HBLK + dj);
              s += hsum_[(size_t)idx * c_ + k];
              n += hcnt_[idx];
            }
          m8[bi * HS + bj] = n > 0 ? (double)s / (double)n : 0.0;
        }
      double avg = 0.0;
      for (int i = 0; i < HS * HS; ++i) avg += m8[i];
      avg /= HS2;
      double srt[HS2];
      std::copy(m8, m8 + HS2, srt);
      std::sort(srt, srt + HS2);
      const double med = 0.5 * (srt[HS2 / 2 - 1] + srt[HS2 / 2]);
      uint64_t a = 0, w = 0;
      for (int i = 0; i < HS * HS; ++i)
      {
        a |= (uint64_t)(m8[i] > avg) << (63 - i);
        w |= (uint64_t)(m8[i] > med) << (63 - i);
      }
      hash_[k] = a;
      hash_[(size_t)c_ + k] = w;

      // pHash: separable 2D DCT-II of the 32x32 means (rows 8x32, then cols 8x8), take
      // the low-freq 8x8, threshold by its median. Basis matches scipy.fftpack.dct.
      // (each d0[u][j] still sums over n in order; n is the outer loop so that the inner one
      // walks a row of means and vectorizes, where a column walk is a strided scalar chain)
      std::fill(d0.begin(), d0.end(), 0.0);
      for (int u = 0; u < HS; ++u)
        for (int n = 0; n < HN; ++n)
        {
          const double b = dctb_[(size_t)u * HN + n];
          const double *IF_RESTRICT row = mean.data() + (size_t)n * HN;
          double *IF_RESTRICT out = d0.data() + (size_t)u * HN;
          for (int j = 0; j < HN; ++j) out[j] += b * row[j];
        }
      double dl[HS * HS];
      for (int u = 0; u < HS; ++u)
        for (int vv = 0; vv < HS; ++vv)
        {
          double acc = 0.0;
          for (int n = 0; n < HN; ++n) acc += dctb_[(size_t)vv * HN + n] * d0[(size_t)u * HN + n];
          dl[u * HS + vv] = acc;
        }
      double psrt[HS2];
      std::copy(dl, dl + HS2, psrt);
      std::sort(psrt, psrt + HS2);
      const double pmed = 0.5 * (psrt[HS2 / 2 - 1] + psrt[HS2 / 2]);
      uint64_t p = 0;
      for (int i = 0; i < HS * HS; ++i) p |= (uint64_t)(dl[i] > pmed) << (63 - i);
      hash_[2 * (size_t)c_ + k] = p;
    }
  }

  // Per-cell derivation. Every cell writes only its own feat/mom slice, so this
  // splits over bands. The cross-cell summary fold is deliberately NOT here: its sum
  // and sum-of-squares are float64 reductions over cells, and splitting them would
  // reassociate the additions and make the output depend on the thread count. It stays
  // serial in derive(), below.
  // The cross-cell summary of a level: reset, fold a run of cells, write.
  void summary_reset()
  {
    // Largest finite values, NOT +/-inf: -ffast-math implies -ffinite-math-only, under
    // which some compilers (e.g. Apple clang) miscompile a min/max reduction seeded with
    // infinity and leave the accumulator at its init. FLT_MAX/-FLT_MAX are equivalent for
    // any finite feature and safe. (-FBIG == lowest(), the correct max seed.)
    const float FBIG = std::numeric_limits<float>::max();
    const double DBIG = std::numeric_limits<double>::max();
    std::fill(sfmn_.begin(), sfmn_.end(), FBIG);
    std::fill(sfmx_.begin(), sfmx_.end(), -FBIG);
    std::fill(sfsm_.begin(), sfsm_.end(), 0.0);
    std::fill(sfsq_.begin(), sfsq_.end(), 0.0);
    std::fill(smmn_.begin(), smmn_.end(), DBIG);
    std::fill(smmx_.begin(), smmx_.end(), -DBIG);
    std::fill(smsm_.begin(), smsm_.end(), 0.0);
    std::fill(smsq_.begin(), smsq_.end(), 0.0);
  }
  void summary_fold(const Level &L, size_t cell, size_t cells)
  {
    for (size_t k = 0; k < (size_t)c_; ++k)
    {
      fold_summary(L.feat + (cell * c_ + k) * NMAP, (size_t)c_ * NMAP, cells, NF, &sfmn_[k * NF],
                   &sfmx_[k * NF], &sfsm_[k * NF], &sfsq_[k * NF]);
      fold_summary(L.mom + (cell * c_ + k) * NMOM, (size_t)c_ * NMOM, cells, NMOM, &smmn_[k * NMOM],
                   &smmx_[k * NMOM], &smsm_[k * NMOM], &smsq_[k * NMOM]);
    }
  }
  void summary_write(const Level &L)
  {
    const size_t nc = (size_t)L.ny * L.nx;
    for (size_t k = 0; k < (size_t)c_; ++k)
    {
      const size_t stride = (size_t)c_ * NST; // feature axis first, as FEATURE_NAMES runs
      write_summary(NF, nc, &sfmn_[k * NF], &sfmx_[k * NF], &sfsm_[k * NF], &sfsq_[k * NF],
                    L.sum + k * NST, stride);
      write_summary(NMOM, nc, &smmn_[k * NMOM], &smmx_[k * NMOM], &smsm_[k * NMOM],
                    &smsq_[k * NMOM], L.sum + (NF * (size_t)c_ + k) * NST, stride);
    }
  }

  // Derive band bi's cells, SB at a time. The summary is a serial fold in cell order, so that
  // it rounds the same for any thread count: a lone band folds each batch while it is still in
  // L1, several bands leave it to derive().
  static constexpr size_t SB = 32;
  // Derive `cells` cells of L from cell0 on, whose sums start at buf.
  void derive_run(const Level &L, size_t cell0, size_t cells, const int64_t *IF_RESTRICT buf,
                  bool fold)
  {
    double invn[SB * GMAX * 2], inverse = 0.0; // 1/n per cell-channel of a batch
    int64_t n = -1;
    const size_t batch = std::min(SB, sizeof(invn) / sizeof(*invn) / (size_t)c_);
    for (size_t b0 = 0; b0 < cells; b0 += batch)
    {
      const size_t nb = std::min(batch, cells - b0);
      for (size_t i = 0; i < nb; ++i)
      {
        const int64_t *IF_RESTRICT s = buf + (b0 + i) * c_ * NSUM;
        if (s[CNT] != n) // 1/n changes with the sample count, which a level's cells mostly share
        {
          n = s[CNT];
          inverse = n > 0 ? 1.0 / (double)n : 0.0;
        }
        for (int k = 0; k < c_; ++k) invn[i * c_ + k] = inverse;
      }
      moments_records(buf + b0 * c_ * NSUM, invn, nb * c_, L.mom + (cell0 + b0) * c_ * NMOM);
      derive_records(buf + b0 * c_ * NSUM, L.mom + (cell0 + b0) * c_ * NMOM, invn, nb * c_,
                     L.feat + (cell0 + b0) * c_ * NMAP);
      if (fold) summary_fold(L, cell0 + b0, nb);
    }
  }
  void derive_cells(int bi)
  {
    const size_t nb = bands_.size();
    for (size_t lvl = stream_ ? 1 : 0; lvl < levels_.size(); ++lvl) // stream_row() derived level 0
    {
      const Level &L = levels_[lvl];
      const size_t nc = (size_t)L.ny * L.nx, c0 = nc * (size_t)bi / nb,
                   c1 = nc * (size_t)(bi + 1) / nb;
      if (nb == 1) summary_reset();
      if (lvl == 1 && stream_) // rows flush_up() derived in their band are done: the rest
      {                        // from their slots, then the summary over all of them
        for (size_t r = c0 / L.nx; r * L.nx < c1; ++r)
          if (!up_done_[r])
          {
            const size_t a = std::max(c0, r * L.nx), b = std::min(c1, (r + 1) * L.nx);
            derive_run(L, a, b - a, L.buf.data() + a * c_ * NSUM, false);
          }
        if (nb == 1)
          for (size_t cell = 0; cell < nc; cell += SB)
            summary_fold(L, cell, std::min(SB, nc - cell));
      }
      else
        derive_run(L, c0, c1 - c0, L.buf.data() + c0 * c_ * NSUM, nb == 1);
      if (nb == 1) summary_write(L);
    }
  }
  void derive()
  {
    run_job(JOB_DERIVE);
    if (bands_.size() > 1)
      for (const Level &L : levels_)
      {
        summary_reset();
        const size_t nc = (size_t)L.ny * L.nx;
        for (size_t cell = 0; cell < nc; cell += SB) summary_fold(L, cell, std::min(SB, nc - cell));
        summary_write(L);
      }
    const double ginvn = graw_[CNT] > 0 ? 1.0 / (double)graw_[CNT] : 0.0;
    double ginvns[2 * GMAX];
    std::fill_n(ginvns, std::size(ginvns), ginvn);
    std::vector<double> many((size_t)c_ > std::size(ginvns) ? (size_t)c_ : 0, ginvn);
    moments_records(graw_.data(), many.empty() ? ginvns : many.data(), (size_t)c_, gmom_);
    derive_records(graw_.data(), gmom_, many.empty() ? ginvns : many.data(), (size_t)c_, gfeat_);
  }

public:
  FeatureComputer() = default;

  void set_config(const std::vector<int64_t> &dims, const std::vector<int> &channels,
                  const std::vector<std::vector<int>> &grids, const std::vector<int64_t> &stride,
                  int threads, int convert, const std::vector<int64_t> &thumb)
  {
    pool_stop();
    size_t block = 0; // the output block: every features() array at a 64-byte-aligned offset
    auto place = [&](size_t bytes)
    {
      const size_t at = block;
      block += (bytes + Blocks::HDR - 1) / Blocks::HDR * Blocks::HDR;
      return at;
    };
    h_ = (int)dims[0];
    w_ = (int)dims[1];
    chan_ = channels;
    c_ = (int)chan_.size();
    resize_ = !thumb.empty();
    if (resize_)
    { // dims is the frame; the pass runs on the thumbnail it makes of it
      sh_ = h_, sw_ = w_;
      csrc_ = (int)dims[2];
      h_ = (int)thumb[0], w_ = (int)thumb[1];
      if (csrc_ > 4) throw std::invalid_argument("the thumbnail resize takes up to 4 channels");
      area_.setup(sh_, sw_, h_, w_, csrc_);
      need_.clear();
      coff_src_.assign((size_t)csrc_, 0);
      coff_rz_.assign((size_t)std::max(c_, 3), 0);
    }
    sy_ = (int)stride[0];
    sx_ = (int)stride[1];
    conv_ = convert;
    if (conv_ != CONV_NONE && conv_ != CONV_BGR2HSV)
      throw std::invalid_argument("unknown colour conversion");
    if (conv_ != CONV_NONE && dims[2] != 3)
      throw std::invalid_argument("a colour conversion needs a 3-channel image");

    group_ = c_ <= GMAX ? c_ : 1;
    // A lag whose span exceeds the smaller side is dropped on BOTH axes, matching the
    // dense reference maps rather than testing each axis independently.
    bard_lagmask_ = 0;
    for (int j = 0; j < BARD_NL; ++j)
      if (BARD_SPAN * BARD_LAGS[j] < std::min(h_, w_)) bard_lagmask_ |= 1 << j;

    for (int k = 1; k < HB; ++k)
    {
      // ray HB-k is ray k's mirror image across the vertical, to the bit (block() relies on it)
      const double a = PI * std::min(k, HB - k) / HB;
      const int64_t c = std::llround(RAYSCALE * std::cos(a));
      ray_[k - 1][0] = (int16_t)(k <= HRAYS ? c : -c);
      ray_[k - 1][1] = (int16_t)-std::llround(RAYSCALE * std::sin(a));
      // block() sums the energy past a ray minus the energy short of it, with the pixels ON the
      // ray (t == 0) going either way: that is exact only if no nonzero gradient lies on a ray,
      // i.e. the ray's reduced direction is out of the gradient's range (|gx|, |gy| <= 4 * 255).
      // True of these rays (the smallest component after the gcd is 467 and the largest 16135);
      // a retuned HB or RAYSCALE would be checked here.
      const int64_t cx = std::abs((int64_t)ray_[k - 1][0]), cy = std::abs((int64_t)ray_[k - 1][1]);
      int64_t g = cx, h = cy;
      while (h)
      {
        const int64_t t = g % h;
        g = h, h = t;
      }
      if (std::max(cx, cy) / g <= 4 * 255)
        throw std::logic_error(
            "HOG ray on a reachable gradient: the signed ray sums are not exact");
    }
    // A bin is "cardinal" if its centre orientation lies within 3/4 of a bin width of a
    // horizontal or vertical edge (gradient at 0, pi/2, or pi). Text, tables, UI and the
    // Devanagari head-line are strongly axis-aligned; natural scenes are not.
    for (int b = 0; b < HB; ++b)
    {
      const double c = (b + 0.5) * PI / HB;
      const double dist = std::min(std::min(c, PI - c), std::fabs(c - PI / 2));
      hog_card_[b] = dist < 0.75 * PI / HB ? 1 : 0;
    }

    levels_.clear();
    for (size_t k = 0; k < grids.size(); ++k)
    {
      Level L;
      L.ny = 1 << grids[k][0];
      L.nx = 1 << grids[k][1];
      if (k == 0)
      {
        if (h_ % L.ny != 0 || w_ % L.nx != 0)
          throw std::invalid_argument("finest grid must divide the image shape");
      }
      else
      {
        const Level &p = levels_[k - 1];
        if (p.ny % L.ny != 0 || p.nx % L.nx != 0)
          throw std::invalid_argument("pyramid levels must be nested");
        L.fy = p.ny / L.ny;
        L.fx = p.nx / L.nx;
      }
      L.buf.assign((size_t)L.ny * L.nx * c_ * NSUM, 0);
      const size_t nc = (size_t)L.ny * L.nx;
      L.ofeat = place(nc * c_ * NMAP * sizeof(float));
      L.omom = place(nc * c_ * NMOM * sizeof(double));
      L.osum = place((size_t)NMAP * c_ * NST * sizeof(double));
      L.rshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NSUM};
      L.fshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_ * NMAP};
      L.mshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NMOM};
      L.sumshape = {(size_t)NMAP, (size_t)c_, (size_t)NST};
      levels_.push_back(std::move(L));
    }

    // Split the planes by column phase while the stride and the cell stay even, so the sampled
    // columns end up adjacent in phase 0 and block() wastes no lanes on the rest -- but only
    // down to cells of whole lane quads, the coarsest geometry an accumulator folds from.
    {
      const int cwf = w_ / levels_[0].nx, N = (int)hn::Lanes(d8);
      for (phases_ = 1;
           phases_ < PHASES_MAX && sx_ % (2 * phases_) == 0 && cwf % (8 * phases_) == 0;)
        phases_ *= 2;
      // block() geometry, in lanes of phase 0. Blocks span whole cells when a cell divides
      // half a block and holds whole lane quads; otherwise each cell is walked alone, its
      // last block masked past the cell edge.
      cw_ = cwf / phases_;
      step_ = sx_ / phases_;
      ncols_ = (cw_ + step_ - 1) / step_;
      multi_ = cw_ % 4 == 0 && cw_ <= N / 2 && N % cw_ == 0;
      for (cw_log2_ = 0; (2 << cw_log2_) <= cw_;) ++cw_log2_; // exact when multi_: a power of two
      masked_ = step_ > 1 || (!multi_ && cw_ % N != 0);
      kmask_.assign((size_t)(std::max(cw_, N) + N - 1) / N * N, 0);
      for (size_t i = 0; i < kmask_.size(); ++i)
        kmask_[i] = ((multi_ || (int)i < cw_) && (i % cw_) % step_ == 0) ? 255 : 0;
      ktail_.assign((size_t)N, 0);
      lanes_ = (size_t)(w_ / phases_);
      block_cells_ = multi_ ? (size_t)N / cw_ : 1;
      // Column P*i + d is lane i + floor(d / P) of phase d mod P, for either sign of d.
      pstride_ = ((int64_t)lanes_ + PAD_L + PAD_R + PLANE_ALIGN - 1) / PLANE_ALIGN * PLANE_ALIGN;
      const int64_t phase_stride = pstride_;
      auto col = [&](int d)
      {
        const int f = ((d % phases_) + phases_) % phases_;
        return f * phase_stride + (d - f) / phases_;
      };
      for (int i = 0; i < 3; ++i)
        tapcol_[i] = col(i - 1); // (col(d) == d with one phase:
                                 // block<.., UNIT> then hard-codes them)
      for (int j = 0; j < BARD_NL; ++j)
      {
        tapcol_[3 + 2 * j] = col(-BARD_LAGS[j]);
        tapcol_[4 + 2 * j] = col(BARD_LAGS[j]);
      }
      // when cell_row() derives by the block, and there is a level for the sums to go up to
      rolls_ = group_ == c_ && levels_.size() > 1 && (h_ / levels_[0].ny + sy_ - 1) / sy_ <= KROWS;
      for (int i = 0; i < (w_ / phases_) % N; ++i) ktail_[i] = kmask_[i];
    }
    // A lag contributes on the horizontal axis only where BOTH its taps are inside the
    // image; that is a property of the column alone, so the mask is built once here.
    bvalid_stride_ = (size_t)(w_ / phases_) + 2 * hn::Lanes(d8); // a block may overrun the row
    bvalid_.assign(bvalid_stride_ * BARD_NL, 0);
    for (int j = 0; j < BARD_NL; ++j)
    {
      const int d = BARD_LAGS[j];
      for (int xs = 0; xs < w_ / phases_; ++xs)
      {
        const int x = xs * phases_; // original column of this sampled column
        if (x >= d && x < w_ - d) bvalid_[(size_t)j * bvalid_stride_ + (size_t)xs] = 255;
      }
    }
    // The whole cell row is processed at once, not a fixed slice of it. That measured
    // strictly faster at every size tried: each plane row's vector serves more of the
    // centres that use it as a tap while it is still hot, and the per-block reduction
    // amortizes over more rows. The window has to span the cell row plus a halo; it has room
    // for a few cell rows of small cells, so that the slide (see accumulate_band) moves a halo
    // per 32 rows at most.
    wrows_ = std::max(h_ / levels_[0].ny, 32) + 2 * BARD_MAXLAG;
    // A sub-chunk's sampled rows read at most about 34 distinct image rows (adjacent rows at
    // stride 1, every other one at stride 2, three apiece beyond), whose responses then fit L1.
    sub_rows_ = sy_ == 1 ? 32 : sy_ == 2 ? 16 : 11;
    rd_stride_ = 3 * sub_rows_;
    rslots_ = (size_t)std::min(sy_, 3);
    for (up_shift_ = 0; levels_.size() > 1 && (1 << up_shift_) < levels_[1].fy;) ++up_shift_;
    coff_.assign((size_t)std::max(c_, conv_ != CONV_NONE ? 3 : 0), 0);

    graw_.assign((size_t)c_ * NSUM, 0);
    ogfeat_ = place((size_t)c_ * NMAP * sizeof(float));
    ogmom_ = place((size_t)c_ * NMOM * sizeof(double));
    sfmn_.resize((size_t)c_ * NF);
    sfmx_.resize((size_t)c_ * NF);
    sfsm_.resize((size_t)c_ * NF);
    sfsq_.resize((size_t)c_ * NF);
    smmn_.resize((size_t)c_ * NMOM);
    smmx_.resize((size_t)c_ * NMOM);
    smsm_.resize((size_t)c_ * NMOM);
    smsq_.resize((size_t)c_ * NMOM);
    graw_shape_ = {(size_t)c_, (size_t)NSUM};
    gfeat_shape_ = {(size_t)c_ * NMAP};
    gmom_shape_ = {(size_t)c_, (size_t)NMOM};

    // --- perceptual hashes: HN x HN value-sum grid, filled in the same pass ---
    hrow_.resize(h_);
    for (int r = 0; r < h_; ++r) hrow_[r] = (int)((int64_t)r * HN / h_);
    hcol_.resize(w_);
    for (int c = 0; c < w_; ++c) hcol_[c] = (int)((int64_t)c * HN / w_);
    // Per-cell sampled-pixel counts + the hash-column run structure. Columns are
    // sampled at sx_ (restarted per finest cell) and visited in increasing order.
    std::vector<int64_t> rcnt(HN, 0), ccnt(HN, 0);
    for (int r = 0; r < h_; r += sy_) rcnt[hrow_[r]]++;
    // Group sampled columns into runs of one hash column within one finest cell.
    hrbeg_.clear();
    hrend_.clear();
    hrhc_.clear();
    {
      const int nfx = levels_[0].nx, cw = w_ / nfx;
      for (int e = 0; e < nfx; ++e)
      {
        const int e1 = e * cw + cw;
        for (int c = e * cw; c < e1;)
        {
          const int hc = hcol_[c], b = c;
          while (c < e1 && hcol_[c] == hc) ccnt[hc]++, c += sx_;
          hrbeg_.push_back(b / phases_); // runs are read from phase 0, so in its lanes
          hrend_.push_back(c / phases_);
          hrhc_.push_back(hc);
        }
      }
    }
    hcnt_.assign((size_t)HN * HN, 0);
    for (int i = 0; i < HN; ++i)
      for (int j = 0; j < HN; ++j) hcnt_[(size_t)i * HN + j] = rcnt[i] * ccnt[j];
    hcntd_.assign(hcnt_.size() + LD, 0.0); // (slack for a last vector)
    hden_.assign(hcnt_.size() + LD, 1.0);
    for (size_t i = 0; i < hcnt_.size(); ++i)
      hcntd_[i] = (double)hcnt_[i], hden_[i] = hcnt_[i] > 0 ? (double)hcnt_[i] : 1.0;
    // When the finest feature grid tiles the HN hash grid, its per-cell S1 (== Sum(v)
    // over the same sampled pixels) already is the hash grid: reuse it in derive_hashes
    // and skip the per-pixel pass entirely. Otherwise fall back to accumulate_hash_row.
    hfast_ = levels_[0].ny % HN == 0 && levels_[0].nx % HN == 0;
    dctb_.assign((size_t)HS * HN, 0.0);
    for (int u = 0; u < HS; ++u)
      for (int n = 0; n < HN; ++n)
        dctb_[(size_t)u * HN + n] = 2.0 * std::cos(PI * u * (2 * n + 1) / (2.0 * HN));
    hsum_.assign((size_t)HN * HN * c_, 0);
    ohash_ = place((size_t)NHASH * c_ * sizeof(uint64_t));
    hshape_ = {(size_t)NHASH, (size_t)c_};

    // Split the finest cell rows evenly, then map each band back to image rows: cell
    // row a starts at image row a * h_ / ny, so the ranges tile [0, h_) exactly. r0 is
    // rounded up to the sy_ grid: the band's first sampled row.
    const int ncy = levels_[0].ny, nb = std::min(threads > 1 ? threads : 1, ncy);
    bands_.clear();
    for (int t = 0; t < nb; ++t)
    {
      const int a = (int)((int64_t)t * ncy / nb), z = (int)((int64_t)(t + 1) * ncy / nb);
      int r0 = (int)(((int64_t)a * h_ + ncy - 1) / ncy);
      r0 = (r0 + sy_ - 1) / sy_ * sy_;
      bands_.push_back({r0, a, z});
    }
    const size_t N = hn::Lanes(d8),
                 nrows = (size_t)std::min(KROWS, (h_ / levels_[0].ny + sy_ - 1) / sy_);
    scr_.assign(bands_.size(), Scratch());
    for (size_t b = 0; b < scr_.size(); ++b)
    {
      Scratch &s = scr_[b];
      s.vok.assign(nrows, 0);
      s.rdn.assign((nrows + sub_rows_ - 1) / sub_rows_, 0);
      s.rdist.assign(s.rdn.size() * rd_stride_, 0);
      s.resp.assign(rd_stride_ * NRESP * 2 * hn::Lanes(d16) + 32, 0); // (+ alignment slack)
      s.hbuf.assign(sub_rows_ * 2 * 2 * hn::Lanes(d32) + 16, 0);
      // the cell row's sums, or a group of cells' (cell_row(): the wider of the two strides)
      s.cells.assign(
          (size_t)c_ * K_N * std::max(cells_stride(), std::max(block_cells_, (size_t)LD) | 1) + N,
          0);
      s.phase_stride = pstride_;
      s.row_stride = s.phase_stride * phases_;
      // a slack row and block so the final vector's overrun stays inside the allocation
      s.planes.assign((size_t)c_ * (size_t)wrows_ * (size_t)s.row_stride + (size_t)s.row_stride +
                          2 * N + PLANE_ALIGN,
                      0);
      s.pl = aligned64(s.planes);
      // a row of gathered planes, the phase split's scratch, and the conversion's source planes
      s.lin.assign((size_t)(3 * c_ + (conv_ != CONV_NONE ? 3 : 0)) * (size_t)w_ + N, 0);
      if (resize_)
      {
        s.area.setup(area_, csrc_);
        const int rows = (bands_[b].cy1 - bands_[b].cy0) * (h_ / levels_[0].ny) + 2 * BARD_MAXLAG;
        s.line.assign((size_t)std::min(rows, h_) * csrc_ * (size_t)w_ + N, 0);
      }
      if (b) s.hsum.assign((size_t)HN * HN * c_, 0);
      s.row.assign((size_t)levels_[0].nx * c_ * NSUM, 0);
      if (levels_.size() > 1) s.up.assign((size_t)c_ * K_N * ((size_t)levels_[1].nx + N), 0);
      if (levels_.size() > 1) up_done_.assign((size_t)levels_[1].ny, 0);
      s.graw.assign((size_t)c_ * NSUM, 0);
    }
    for (size_t b = 1; b < bands_.size(); ++b) workers_.emplace_back([this, b] { worker((int)b); });
    if (blocks_) blocks_->retire(); // arrays of the old configuration keep their blocks
    blocks_ = new Blocks;
    blocks_->bytes = block;
  }

  int threads() const { return (int)bands_.size(); }

  ~FeatureComputer()
  {
    pool_stop();
    if (blocks_) blocks_->retire();
  }

  nb::list raw(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs,
               uint8_t *thumb_out)
  {
    stream_ = false;
    accumulate(img, rs, cs, chs, thumb_out);
    nb::list out;
    for (Level &L : levels_)
      out.append(nb::ndarray<nb::numpy, int64_t>(L.buf.data(), L.rshape.size(), L.rshape.data(),
                                                 nb::handle()));
    out.append(nb::ndarray<nb::numpy, int64_t>(graw_.data(), graw_shape_.size(), graw_shape_.data(),
                                               nb::handle()));
    return out;
  }

  // Per level: the float32 `maps` rows, channel-major over NMAP, and the float64 moments
  // (moments span 0..255^4, where float32's 7 digits would cost precision).
  nb::list features(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs,
                    uint8_t *thumb_out)
  {
    auto *blk = static_cast<uint8_t *>(blocks_->take());
    const nb::capsule owner(blk, Blocks::give); // every array below keeps the block alive
    auto at = [&](auto *&ptr, size_t offset)
    { ptr = reinterpret_cast<std::remove_reference_t<decltype(ptr)>>(blk + offset); };
    for (Level &L : levels_)
    {
      at(L.feat, L.ofeat);
      at(L.mom, L.omom);
      at(L.sum, L.osum);
    }
    at(gfeat_, ogfeat_);
    at(gmom_, ogmom_);
    at(hash_, ohash_);
    stream_ = true;
    accumulate(img, rs, cs, chs, thumb_out);
    derive();
    derive_hashes();
    auto view = [&](auto *data, const std::vector<size_t> &shape)
    {
      using T = std::remove_pointer_t<decltype(data)>;
      return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner);
    };
    nb::list feat, mom, sums;
    for (const Level &L : levels_)
    {
      feat.append(view(L.feat, L.fshape));
      mom.append(view(L.mom, L.mshape));
      sums.append(view(L.sum, L.sumshape));
    }
    feat.append(view(gfeat_, gfeat_shape_));
    mom.append(view(gmom_, gmom_shape_));
    nb::list all;
    all.append(feat);
    all.append(mom);
    all.append(sums);
    all.append(view(hash_, hshape_));
    return all;
  }
};

using Arr = nb::ndarray<nb::numpy, const uint8_t, nb::device::cpu>;

// The conversion on its own: a new (H, W, 3) image from an (H, W, 3) one in any layout. The
// same row kernel the pass fuses, so this is also how the tests pin that kernel to OpenCV.
nb::ndarray<nb::numpy, uint8_t> convert_image(const Arr &a, int conv)
{
  if (a.ndim() != 3 || a.shape(2) != 3)
    throw std::invalid_argument("convert: expected an (H, W, 3) uint8 image");
  const size_t h = a.shape(0), w = a.shape(1), N = hn::Lanes(d8);
  const int64_t rs = a.stride(0), cs = a.stride(1), chs = a.stride(2);
  const int64_t coff[3] = {0, chs, 2 * chs};
  const int chan[3] = {0, 1, 2};
  const bool packed = chs == 1 && cs == 3;
  auto *buf = new uint8_t[h * w * 3];
  const nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
  std::vector<uint8_t> tmp(6 * w + N); // 3 source planes, then the 3 converted ones
  uint8_t *IF_RESTRICT planes = tmp.data() + 3 * w;
  for (size_t y = 0; y < h; ++y)
  {
    convert_row(conv, a.data() + (int64_t)y * rs, cs, coff, chan, 3, (int)w, packed, tmp.data(),
                planes, w);
    uint8_t *IF_RESTRICT row = buf + y * w * 3;
    size_t x = 0;
    for (; x + N <= w; x += N)
      hn::StoreInterleaved3(hn::LoadU(d8, planes + x), hn::LoadU(d8, planes + w + x),
                            hn::LoadU(d8, planes + 2 * w + x), d8, row + x * 3);
    for (; x < w; ++x)
      for (size_t k = 0; k < 3; ++k) row[x * 3 + k] = planes[k * w + x];
  }
  const size_t shape[3] = {h, w, 3};
  return nb::ndarray<nb::numpy, uint8_t>(buf, 3, shape, owner);
}
// The thumbnail output of features()/raw(): (H, W, C) or (H, W) uint8, C-contiguous, sized in
// Python; here only checked and unwrapped.
using ThumbArr = nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>;

uint8_t *thumb_ptr(std::optional<ThumbArr> &t) { return t ? t->data() : nullptr; }

// The resize on its own: cv2.resize(img, (dw, dh), INTER_AREA)'s bytes for a (H, W[, C]) uint8
// image in any layout, over `threads` bands of rows. The same rows the pass fuses.
nb::ndarray<nb::numpy, uint8_t> resize_image(Arr a, int dh, int dw, int threads)
{
  if (a.ndim() != 2 && a.ndim() != 3)
    throw std::invalid_argument("resize_area: expected an (H, W) or (H, W, C) uint8 image");
  const int sh = (int)a.shape(0), sw = (int)a.shape(1), C = a.ndim() == 3 ? (int)a.shape(2) : 1;
  if (C < 1 || C > 4) throw std::invalid_argument("resize_area takes 1 to 4 channels");
  AreaResizer r;
  r.setup(sh, sw, dh, dw, C);
  const int64_t rs = a.stride(0), cs = a.stride(1), chs = a.ndim() == 3 ? a.stride(2) : 0;
  const int64_t coff[4] = {0, chs, 2 * chs, 3 * chs};
  const bool packed = a.ndim() == 3 && chs == 1 && cs == C;
  auto *buf = new uint8_t[(size_t)dh * dw * C];
  const nb::capsule owner(buf, [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
  const int nb_ = std::max(1, std::min(threads, dh));
  auto band = [&](int t)
  {
    AreaScratch s;
    s.setup(r, C);
    std::vector<uint8_t> line((size_t)C * dw + 64);
    for (int dy = dh * t / nb_; dy < dh * (t + 1) / nb_; ++dy)
    {
      area_row(r, s, a.data(), rs, cs, coff, packed, dy, line.data(), (size_t)dw);
      interleave_row(line.data(), (size_t)dw, C, dw, buf + (size_t)dy * dw * C);
    }
  };
  std::vector<std::thread> workers;
  for (int t = 1; t < nb_; ++t) workers.emplace_back(band, t);
  band(0);
  for (std::thread &w : workers) w.join();
  if (a.ndim() == 2)
  {
    const size_t shape[2] = {(size_t)dh, (size_t)dw};
    return nb::ndarray<nb::numpy, uint8_t>(buf, 2, shape, owner);
  }
  const size_t shape[3] = {(size_t)dh, (size_t)dw, (size_t)C};
  return nb::ndarray<nb::numpy, uint8_t>(buf, 3, shape, owner);
}
} // namespace

NB_MODULE(imfeat_core, m)
{
  m.doc() = "imfeat internal C++ module. Public API: imfeat.FeatureComputer.";
  m.attr("HB") = HB;           // HOG orientation-bin count; Python derives its bin labels from this
  m.attr("LBPB") = LBPB;       // LBP^riu2 bin count (9 uniform + 1 non-uniform)
  m.attr("BARD_NL") = BARD_NL; // bar-detector lag count (sizes the spectrum block)
  m.attr("CONVERT_NONE") = (int)CONV_NONE; // colour conversions fused into the pass (Convert)
  m.attr("CONVERT_BGR2HSV") = (int)CONV_BGR2HSV;
  m.def("cpu_count", [] { //  0 when the runtime cannot tell; callers see at least 1
    const unsigned n = std::thread::hardware_concurrency();
    return (int)(n ? n : 1u);
  });
  m.def("convert", &convert_image, nb::arg("arr"), nb::arg("conversion"));
  m.def("resize_area", &resize_image, nb::arg("arr"), nb::arg("height"), nb::arg("width"),
        nb::arg("threads") = 1);
  nb::class_<FeatureComputer>(m, "_FeatureComputerImpl")
      .def(nb::init<>())
      .def("set_config", &FeatureComputer::set_config, nb::arg("dims"), nb::arg("channels"),
           nb::arg("grids"), nb::arg("stride"), nb::arg("threads"), nb::arg("convert"),
           nb::arg("thumb"))
      .def("threads", &FeatureComputer::threads)
      .def(
          "raw",
          [](FeatureComputer &self, const Arr &a, std::optional<ThumbArr> thumb)
          {
            return self.raw(a.data(), a.stride(0), a.stride(1), a.ndim() > 2 ? a.stride(2) : 0,
                            thumb_ptr(thumb));
          },
          nb::arg("arr"), nb::arg("thumb_out").none() = nb::none())
      .def(
          "features",
          [](FeatureComputer &self, const Arr &a, std::optional<ThumbArr> thumb)
          {
            return self.features(a.data(), a.stride(0), a.stride(1), a.ndim() > 2 ? a.stride(2) : 0,
                                 thumb_ptr(thumb));
          },
          nb::arg("arr"), nb::arg("thumb_out").none() = nb::none());
}
