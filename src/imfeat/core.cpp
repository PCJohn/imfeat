#include "hwy/highway.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#define IF_RESTRICT __restrict
#else
#define IF_RESTRICT __restrict__
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
// Alongside, one int64 per channel PAIR: Sum(v_i * v_j), the only cross-channel sum.
// Every sum is additive, so pyramid levels and the global reduction are exact
// sums of the finest cells -- the image is read exactly once. Central moments and
// the tensor eigen-features are nonlinear, so they are derived at the end.
//
// One traversal, COLUMNS as the SIMD lane. Every image row is de-interleaved once into a
// rolling window of planar rows; kernel() then reads a block of columns from rows r-1, r,
// r+1 and computes every per-pixel quantity in-register (Sobel, tensor products, HOG ray
// tests, extrema, LBP, moments). Its accumulators are carried in registers down the rows of
// a cell row and folded to cells once per block, so memory is touched once per cell row.

namespace
{
constexpr int SXX = 0, SYY = 1, SXY = 2, CNT = 3, NCOMP = 4;
// HB (HOG orientation bins over [0, pi)) is compile-time: it sizes the per-cell
// accumulator and the inner histogram/ray loops, so a runtime value regresses the
// hot loop (~17% measured). Everything below is DERIVED from it -- change HB here
// and rebuild to retune bin count; no other constant is independently hardcoded.
constexpr int HB = 9;
constexpr int HOG0 = NCOMP;                      // 4
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
constexpr int BARD_SPAN = 2;   // a lag-d test reaches +/- d, so it spans 2d
constexpr int BARD_TAU = 8;    // 8/255 gate: below this the response is sensor noise
constexpr int BARD_MAXLAG = 4; // == BARD_LAGS[BARD_NL - 1]; sizes the plane halo
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
// The rolling window of planar rows kernel() reads. Every row of the cell row in flight plus
// a MAXLAG halo each side stays resident (slots_ rows, set at configure time; image row y
// lives in slot y % slots). Each phase plane carries slack at both ends for the taps that
// reach past the row and for the last block's overrun into the next plane, whose lanes are
// masked or folded into cells nobody reads. It is not free space: the window wants to stay
// inside L1/L2.
constexpr int PAD_L = 8;
constexpr int PAD_R = 8;
constexpr int PHASES_MAX = 8;    // most column phases the planes are split into
constexpr int NF_S = 5;          // tensor-derived float channels
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
// Cross-channel products Sum(v_i * v_j) over the C*(C-1)/2 unordered pairs. Off for
// C=1 (no pairs) and for C > XMAX (hyperspectral: the pair count would explode).
constexpr int XMAX = 8;
constexpr int NXF = 2; // derived per pair: [cov, corr]
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
constexpr int HN = 32;        // hash mean-grid resolution (== pHash DCT input side)
constexpr int HS = 8;         // hash side: HS*HS = 64 bits
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

// The sums kernel() produces per cell, in fold order. To add one: add an entry next to the
// sums of the same lane geometry, accumulate it in kernel(), and map it to its output slot in
// cell_row()'s flush -- the folds in between walk the enum.
//   PAIR: pmaddwd outputs, one int32 lane per two columns, per half block. The moments are
//         those of w = v - KBIAS (w^2 <= 2^14 keeps every product inside pmaddwd's int16
//         inputs); the flush shifts them back to raw power sums, exactly.
//   T4:   sum w^4, one int64 lane per four columns, per half block.
//   SG4:  sum |grad|^4, one int64 lane per two columns, per quarter block.
//   HOG:  CUMULATIVE gradient energy past each bin-boundary ray, one int32 lane per column,
//         per quarter block. The ray tests are nested, so bin b = cum[b-1] - cum[b].
//   U8:   counts, one uint8 lane per column over the whole block.
//   BARD: the bar detector's gated maps, one uint16 lane per two columns over the whole block.
// clang-format off
enum
{
  K_SXX, K_SYY, K_T2, K_LAP2, K_LAWS, K_NUNSIGNED = K_LAWS + NLAWS, // squares first,
  K_SXY = K_NUNSIGNED, K_T1, K_T3, K_LAP1, K_NPAIR,                 // then what can go negative
  K_T4 = K_NPAIR, K_SG4, K_HOG,
  K_U8 = K_HOG + HB - 1, K_MAX = K_U8, K_MIN, K_LBP, // LBP bins 0..LBPB-2; the last is the rest
  K_BARD = K_LBP + LBPB - 1,
  K_N = K_BARD + BARD_NMAP,
  K_NU8 = K_BARD - K_U8
};
// clang-format on
// A pair of uint8 map values is up to 510 a row, which caps a block at 128 sampled rows
// between folds; every other sum has headroom beyond that (a pair of |grad|^2 is 2^22 a row).
constexpr int KROWS = 128;
// kernel() lane bases per sampled row per channel: the 3x3 neighbourhood, then four taps a lag.
constexpr int KPTR = 9 + 4 * BARD_NL;
constexpr int KBIAS = 128; // kernel() takes moments of v - KBIAS

struct Acc // one block of one channel
{
  V16 col[2]; // per half: the centred pixels down each column, for the column profile
  V32 pair[2][K_NPAIR], hog[4][HB - 1];
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
// is one compare. `mesh` is the sampling mesh, all-ones when every lane is sampled.
// acc += v -- or acc = v on a block's first row (ROW0), so that no accumulator is ever zeroed.
template <bool ROW0, class V>
HWY_INLINE void add(V &acc, V v)
{
  acc = ROW0 ? v : hn::Add(acc, v);
}

template <class V>
HWY_INLINE void add_to(bool adds, V &acc, V v)
{
  acc = adds ? hn::Add(acc, v) : v;
}

template <bool ROW0>
HWY_INLINE void bard(const uint8_t *const *t, size_t x, V8 c, const V8 *IF_RESTRICT near, V8 mesh,
                     const uint8_t *IF_RESTRICT valid, size_t vstride, int hok, int vok, Acc &A)
{
  static_assert(BARD_LAGS[0] == 1, "lag 1's taps are the 3x3 neighbours kernel() already holds");
  V8 dmax = hn::Zero(d8), lmax = hn::Zero(d8), resp[BARD_NL];
  for (int j = 0; j < BARD_NL; ++j)
  {
    V8 dk = hn::Zero(d8), lt = hn::Zero(d8);
    if ((hok >> j) & 1)
    {
      const V8 l = j ? hn::LoadU(d8, t[4 * j] + x) : near[0],
               r = j ? hn::LoadU(d8, t[4 * j + 1] + x) : near[1];
      V8 lo = hn::Min(l, r), hi = hn::Max(l, r);
      if (valid)
      {
        const V8 vm = hn::LoadU(d8, valid + (size_t)j * vstride + x);
        lo = hn::And(lo, vm);
        hi = hn::Or(hi, hn::Not(vm));
      }
      if ((vok >> j) & 1)
      {
        const V8 u = j ? hn::LoadU(d8, t[4 * j + 2] + x) : near[2],
                 v = j ? hn::LoadU(d8, t[4 * j + 3] + x) : near[3];
        lo = hn::Max(lo, hn::Min(u, v));
        hi = hn::Min(hi, hn::Max(u, v));
      }
      dk = hn::SaturatedSub(lo, c);
      lt = hn::SaturatedSub(c, hi);
    }
    resp[j] = hn::Max(dk, lt);
    dmax = hn::Max(dmax, dk);
    lmax = hn::Max(lmax, lt);
  }
  const V8 peak = hn::Max(dmax, lmax);
  const V8 keep = hn::And(
      mesh, hn::VecFromMask(d8, hn::Eq(hn::Max(peak, hn::Set(d8, (uint8_t)BARD_TAU)), peak)));
  auto gated = [&](int m, V8 v) { add<ROW0>(A.bard[m], hn::SumsOf2(hn::And(keep, v))); };
  gated(0, hn::Set(d8, 1));
  for (int j = 0; j < BARD_NL; ++j) gated(1 + j, resp[j]);
  gated(1 + BARD_NL, dmax);
  gated(2 + BARD_NL, lmax);
}

// One block of columns of one sampled row. `p` holds KPTR lane bases: the 3x3 neighbourhood,
// rows above/centre/below by columns left/mid/right, then bard()'s taps. Replicate padding is
// already in the planes, so there is no border logic here. MASKED zeroes the off-mesh lanes
// (mask: 255 on the sampling mesh) before anything pairs up columns; the uint8 counts are per
// column, so they are masked once, at the fold. `w` returns the centred pixels for the
// cross-channel products, and `rowsum` gathers them over the row's blocks for the row profile.
template <bool MASKED, bool ROW0>
HWY_INLINE void kernel(const uint8_t *const *p, size_t x, const uint8_t *IF_RESTRICT mask,
                       const uint8_t *IF_RESTRICT valid, size_t vstride, int hok, int vok,
                       const V16 *IF_RESTRICT rays, Acc &A, V16 *IF_RESTRICT w, V32 &rowsum)
{
  const V8 aL = hn::LoadU(d8, p[0] + x), aM = hn::LoadU(d8, p[1] + x), aR = hn::LoadU(d8, p[2] + x);
  const V8 bL = hn::LoadU(d8, p[3] + x), c = hn::LoadU(d8, p[4] + x), bR = hn::LoadU(d8, p[5] + x);
  const V8 eL = hn::LoadU(d8, p[6] + x), eM = hn::LoadU(d8, p[7] + x), eR = hn::LoadU(d8, p[8] + x);
  // strict 8-neighbour extrema: c > max  <=>  SatSub(c, max) != 0
  const V8 one = hn::Set(d8, 1);
  const V8 mx =
      hn::Max(hn::Max(hn::Max(aL, aM), hn::Max(aR, bL)), hn::Max(hn::Max(bR, eL), hn::Max(eM, eR)));
  const V8 mn =
      hn::Min(hn::Min(hn::Min(aL, aM), hn::Min(aR, bL)), hn::Min(hn::Min(bR, eL), hn::Min(eM, eR)));
  add<ROW0>(A.u8[K_MAX - K_U8], hn::Min(hn::SaturatedSub(c, mx), one));
  add<ROW0>(A.u8[K_MIN - K_U8], hn::Min(hn::SaturatedSub(mn, c), one));
  // LBP^riu2: sign tests (0 / -1) in circular order NW N NE E SE S SW W, so their sum is
  // -popcount and the sum of neighbouring XORs is -transitions. Uniform (<= 2 transitions)
  // votes bin popcount; the rest vote nothing and are recovered at flush as count - sum.
  auto ge = [&](V8 q) { return hn::VecFromMask(d8, hn::Eq(hn::Max(q, c), q)); };
  const V8 s0 = ge(aL), s1 = ge(aM), s2 = ge(aR), s3 = ge(bR), s4 = ge(eR), s5 = ge(eM),
           s6 = ge(eL), s7 = ge(bL);
  const V8 npc =
      hn::Add(hn::Add(hn::Add(s0, s1), hn::Add(s2, s3)), hn::Add(hn::Add(s4, s5), hn::Add(s6, s7)));
  const V8 ntr = hn::Add(
      hn::Add(hn::Add(hn::Xor(s0, s1), hn::Xor(s1, s2)), hn::Add(hn::Xor(s2, s3), hn::Xor(s3, s4))),
      hn::Add(hn::Add(hn::Xor(s4, s5), hn::Xor(s5, s6)),
              hn::Add(hn::Xor(s6, s7), hn::Xor(s7, s0))));
  const auto uniform = hn::Gt(hn::BitCast(di8, ntr), hn::Set(di8, -3));
  const V8 code = hn::BitCast(d8, hn::IfThenElse(uniform, hn::BitCast(di8, npc), hn::Set(di8, 1)));
  for (int j = 0; j < LBPB - 1; ++j)
    A.u8[K_LBP - K_U8 + j] = hn::Sub(ROW0 ? hn::Zero(d8) : A.u8[K_LBP - K_U8 + j],
                                     hn::VecFromMask(d8, hn::Eq(code, hn::Set(d8, (uint8_t)-j))));

  const V8 m8 = MASKED ? hn::LoadU(d8, mask) : hn::Set(d8, 255);
  const V8 near[4] = {bL, bR, aM,
                      eM}; // left, right, up, down: where lag 1's rows are in the image,
  bard<ROW0>(p + 9, x, c, near, m8, valid, vstride, hok, vok, A); // the 3x3 rows are not replicas
  for (int h = 0; h < 2; ++h) // the int16 half blocks
  {
    auto wide = [&](V8 v)
    { return hn::BitCast(d16, h ? hn::PromoteUpperTo(du16, v) : hn::PromoteLowerTo(du16, v)); };
    const V16 al = wide(aL), am = wide(aM), ar = wide(aR), bl = wide(bL), br = wide(bR),
              el = wide(eL), em = wide(eM), er = wide(eR), cc = wide(c), c2 = hn::Add(cc, cc);
    // Each row's three 1-D responses: D = r - l (mask E), S = l + r - 2m (mask S, negated)
    // and L = l + 2m + r = S + 4m. A 3x3 Laws response is then one 1-D mask down a column of
    // them -- Sobel included (gx = L.D, gy = E.L) -- up to a sign that the squares drop.
    const V16 Da = hn::Sub(ar, al), Db = hn::Sub(br, bl), De = hn::Sub(er, el);
    const V16 Sa = hn::Sub(hn::Add(al, ar), hn::Add(am, am)), Sb = hn::Sub(hn::Add(bl, br), c2),
              Se = hn::Sub(hn::Add(el, er), hn::Add(em, em));
    const V16 La = hn::Add(Sa, hn::ShiftLeft<2>(am)), Lb = hn::Add(Sb, hn::ShiftLeft<1>(c2)),
              Le = hn::Add(Se, hn::ShiftLeft<2>(em));
    V32 *IF_RESTRICT P = A.pair[h];
    const V16 m = !MASKED ? hn::Zero(d16)
                  : h     ? hn::PromoteUpperTo(d16, hn::BitCast(di8, m8))
                          : hn::PromoteLowerTo(d16, hn::BitCast(di8, m8));
    auto on_mesh = [&](V16 v) { return MASKED ? hn::And(v, m) : v; };
    // a response is squared into its sum as soon as it exists, which keeps few of them alive
    auto energy = [&](int k, V16 v)
    {
      v = on_mesh(v);
      add<ROW0>(P[k], hn::WidenMulPairwiseAdd(d32, v, v));
    };
    const V16 ones = hn::Set(d16, 1), gx0 = hn::Add(hn::Add(Da, De), hn::Add(Db, Db));
    const V16 ls = hn::Add(hn::Add(Sa, Se), hn::Add(Sb, Sb));
    energy(K_LAWS + LAWS_EE, hn::Sub(De, Da));
    energy(K_LAWS + LAWS_SE, hn::Sub(gx0, hn::ShiftLeft<2>(Db)));
    energy(K_LAWS + LAWS_ES, hn::Sub(Se, Sa));
    energy(K_LAWS + LAWS_LS, ls);
    energy(K_LAWS + LAWS_SS, hn::Sub(ls, hn::ShiftLeft<2>(Sb)));
    energy(K_LAWS + LAWS_SL, hn::Sub(hn::Add(La, Le), hn::Add(Lb, Lb)));
    const V16 lap = on_mesh(hn::Sub(hn::Add(Sb, hn::Add(am, em)), c2)); // up+down+left+right-4c
    add<ROW0>(P[K_LAP1], hn::WidenMulPairwiseAdd(d32, lap, ones));
    add<ROW0>(P[K_LAP2], hn::WidenMulPairwiseAdd(d32, lap, lap));
    const V16 gx = on_mesh(gx0), gy = on_mesh(hn::Sub(Le, La)),
              wc = on_mesh(hn::Sub(cc, hn::Set(d16, KBIAS)));
    w[h] = wc;
    add<ROW0>(A.col[h], wc);
    const V16 w2 = hn::Mul(wc, wc);
    add<ROW0>(P[K_SXX], hn::WidenMulPairwiseAdd(d32, gx, gx));
    add<ROW0>(P[K_SYY], hn::WidenMulPairwiseAdd(d32, gy, gy));
    add<ROW0>(P[K_SXY], hn::WidenMulPairwiseAdd(d32, gx, gy));
    const V32 t1 = hn::WidenMulPairwiseAdd(d32, wc, ones);
    add<ROW0>(P[K_T1], t1);
    rowsum = hn::Add(rowsum, t1);
    add<ROW0>(P[K_T2], hn::WidenMulPairwiseAdd(d32, wc, wc));
    add<ROW0>(P[K_T3], hn::WidenMulPairwiseAdd(d32, w2, wc));
    add<ROW0>(A.t4[h], hn::SumsOf2(hn::WidenMulPairwiseAdd(d32, w2, w2))); // 2^29 a pair: widen now
    // HOG: fold the gradient to the upper half-plane, interleave (qy, qx) so that one pmaddwd
    // against (cx, -cy) is a whole ray test, and add |grad|^2 past every ray it clears.
    const V16 zero = hn::Zero(d16);
    const V16 neg =
        hn::VecFromMask(d16, hn::Or(hn::Lt(gy, zero), hn::And(hn::Eq(gy, zero), hn::Lt(gx, zero))));
    const V16 qx = hn::Sub(hn::Xor(gx, neg), neg), qy = hn::Abs(gy);
    for (int q = 0; q < 2; ++q) // the int32 quarter blocks
    {
      const V16 z =
          q ? hn::InterleaveWholeUpper(d16, qy, qx) : hn::InterleaveWholeLower(d16, qy, qx);
      const V32 g2 = hn::WidenMulPairwiseAdd(d32, z, z);
      const int k = 2 * h + q;
      add<ROW0>(A.sg4[k], hn::BitCast(d64, hn::Add(hn::MulEven(g2, g2), hn::MulOdd(g2, g2))));
      for (int j = 0; j < HB - 1; ++j)
      {
        const V32 t = hn::WidenMulPairwiseAdd(d32, z, rays[j]);
        add<ROW0>(A.hog[k][j], hn::And(g2, hn::VecFromMask(d32, hn::Gt(t, hn::Set(d32, -1)))));
      }
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
    if (g > 1)
    {
      const hn::Half<D> dh;
      const auto even = hn::LowerHalf(dh, hn::ConcatEven(d, v, v)),
                 odd = hn::LowerHalf(dh, hn::ConcatOdd(d, v, v));
      return fold(dh, hn::Add(even, odd), g / 2, add, dst);
    }
  hn::StoreU(add ? hn::Add(hn::LoadU(d, dst), v) : v, d, dst);
}
// ... of int32 lanes. Sums that cannot go negative widen by zero-extension, which is a mask
// and a shift where sign-extension is a compare and two shuffles.
template <bool SIGN_EXTEND>
HWY_INLINE void fold32(V32 v, size_t g, bool add, int64_t *IF_RESTRICT dst)
{
  if (SIGN_EXTEND)
    fold(d64, hn::SumsOf2(v), g / 2, add, dst);
  else
    fold(d64, hn::BitCast(d64, hn::SumsOf2(hn::BitCast(du32, v))), g / 2, add, dst);
}

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

// One image row into the rolling window. With P == 1 the gathered rows are the planes.
// With P > 1 each row is then split by column phase: phase p holds columns p, p+P, ...
// so the sampled columns are exactly phase 0 and a tap d columns away is still a contiguous
// load, from phase (d % P). The columns just outside the image replicate the border, so
// kernel()'s 3x3 taps need no border logic: column -1 is lane -1 of the last phase, and
// column w (reachable only when P == 1) is lane w.
void deinterleave_row(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs,
                      const int64_t *IF_RESTRICT coff, int c, int w, int y, bool packed,
                      int64_t prs, int slots, int slot, int P, int64_t phs,
                      uint8_t *IF_RESTRICT tmp, uint8_t *IF_RESTRICT dst)
{
  const uint8_t *IF_RESTRICT src = img + (int64_t)y * rs;
  uint8_t *IF_RESTRICT d0 = dst + (size_t)slot * (size_t)prs + PAD_L; // slot == y % slots
  const size_t pstride = (size_t)slots * (size_t)prs;
  uint8_t *IF_RESTRICT scratch = tmp + (size_t)c * (size_t)w;
  if (P == 1)
    gather_row(src, cs, coff, c, w, packed, scratch, d0, pstride);
  else
    gather_row(src, cs, coff, c, w, packed, scratch, tmp, (size_t)w);
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
      for (int c = b; c < e; c += step) s += v[c];
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
    std::lock_guard<std::mutex> lock(mu);
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
      std::lock_guard<std::mutex> lock(b->mu);
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
      std::lock_guard<std::mutex> lock(mu);
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
  std::vector<int64_t> buf, xbuf;
  // Outputs, in the block of the features() call in flight; o* are their offsets in it.
  float *feat = nullptr, *xfeat = nullptr;
  double *mom = nullptr;
  double *sum = nullptr; // (NMAP, c, NST) cross-cell summary of feat, then of mom
  size_t ofeat = 0, oxfeat = 0, omom = 0, osum = 0;
  std::vector<size_t> rshape, fshape, mshape, xrshape, xfshape, sumshape;
};

class FeatureComputer
{
  int h_ = 0, w_ = 0, c_ = 1, sy_ = 1, sx_ = 1;
  bool interleaved_ = false; // source rows are already the (H,W,C) layout we want
  std::vector<int> chan_;
  std::vector<int64_t> coff_; // per-selected-channel offset within a pixel
  std::vector<Level> levels_;
  int16_t ray_[HB - 1][2] = {}; // bin-boundary ray j as (cx, -cy): t = cx*qy - cy*qx in one pmaddwd
  int hog_card_[HB] = {};    // 1 for the axis-aligned (cardinal) orientation bins
  std::vector<uint8_t> bvalid_; // per lag: 255 where both +/-d column taps are in range
  size_t bvalid_stride_ = 0;
  int phases_ = 1;           // column phases the planes are split into (1 = not split)
  int slots_ = 9;            // rolling plane window rows (see the window comment)
  int bard_lagmask_ = 0;     // bit j set iff bard lag j fits the image (see bard())
  int np_ = 0;               // number of channel pairs (0 if C<2 or C>XMAX)
  std::vector<int> pi_, pj_; // the pairs, in (i<j) lexicographic order

  // How kernel() blocks map to cells, in LANES of phase 0 (a cell is cw_ of them, every
  // step_-th one sampled). A block either spans whole cells (multi_: cw_ divides it) or starts
  // at a cell edge with the lanes past the cell masked off; kmask_ is that cell-relative mesh.
  int cw_ = 1, cw_log2_ = 0, step_ = 1, ncols_ = 1;
  // Nothing per row, per block or per cell divides: what would is worked out once per
  // configuration, or carried along as a running count.
  int64_t tapcol_[KPTR] = {}; // each kernel() tap's column, as an offset into a row of planes
  int symod_ = 0;             // sy_ % slots_: how far the window slot moves per sampled row
  size_t block_cells_ = 1;    // cells a block spans (multi_)
  size_t lanes_ = 0;          // lanes in a row of one phase plane: w_ / phases_
  int up_shift_ = 0;          // log2 of level 1's fy: parents are whole powers of two of cells
  std::vector<uint32_t> mesh_lanes_; // the lanes on the sampling mesh, for the column profile
  bool multi_ = false, masked_ = false;
  std::vector<uint8_t> kmask_, ktail_; // ... and the row's last block, when it overruns the row
  // Projection profiles: the sum of the centred pixels along every sampled row and column.
  std::vector<int64_t> rowprof_; // [sampled row][channel]; the column sums live in each band
  double *prow_ = nullptr, *pcol_ = nullptr; // their means, in the output block
  size_t oprow_ = 0, opcol_ = 0;
  std::vector<size_t> prow_shape_, pcol_shape_;

  // Per-band state: the plane window and everything a cell row in flight touches.
  struct Scratch
  {
    std::vector<Acc> acc;              // [channel] block accumulators
    std::vector<const uint8_t *> rows; // [sampled row][channel][KPTR] kernel() lane bases
    std::vector<V32> rowv;             // [sampled row][channel] row sums in flight, as pair lanes
    std::vector<int32_t> col;          // [channel][lane] the band's column sums
    std::vector<int> rslot; // the window slot of each sampled row of the cell row in hand
    int fslot = 0;          // ... of the next row to fill,
    size_t rowidx = 0;      // and the cell row's first sampled row, counted from the image top
    std::vector<int> vok;              // [sampled row] bard lags whose rows are inside the image
    std::vector<int> rws;              // the cell row's sampled rows
    std::vector<int64_t> cells;  // [channel][K_N][cell], then [pair][cell]: the cell row's sums
    std::vector<int64_t> hsum;   // private hash partial; bands >0 only
    std::vector<int64_t> row, rowx, graw, gxraw; // features(): the cell row in flight, and the
                                                 // band's share of the global sums
    std::vector<int64_t>
        up;           // features(): the level-1 cell row in flight, as kernel sums like `cells`
    int64_t up_n = 0; // ... and its sample count per cell so far
    std::vector<int64_t> held, heldx;            // ... and its first rows, when their parents
    std::vector<int> held_cy;    // belong to the band above: rolled up after the join
    std::vector<uint8_t> planes; // padded de-interleaved planes: the rolling window
    std::vector<uint8_t> lin;    // one row, channels contiguous, then the phase split's scratch
    int64_t row_stride = 0;      // padded plane row stride
    int64_t phase_stride = 0;    // stride between a row's phase planes
    int filled = 0;              // rows [.., filled) of the window are loaded
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
  enum Job
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
  std::vector<int64_t> hsum_;             // HN*HN*c_ : Sum(v) per hash cell per channel
  std::vector<int64_t> hcnt_;             // HN*HN : sampled-pixel count per cell (fixed per config)
  std::vector<double> hcntd_, hden_; // ... as doubles, and with 1 for 0: what the means divide by
  std::vector<int> hrow_, hcol_;          // row -> hash row (h_), col -> hash col (w_)
  std::vector<int> hrbeg_, hrend_, hrhc_; // per-run column range [beg,end) and its hash col
  std::vector<double> dctb_;              // HS*HN : DCT-II basis rows (scipy type-2, norm=None)
  uint64_t *hash_ = nullptr;   // (NHASH, c_) in the output block: aHash, wHash, pHash rows
  std::vector<size_t> hshape_; // {NHASH, c_}
  bool hfast_ = false;                          // finest grid tiles HN -> hashes reuse its S1

  std::vector<int64_t> graw_, gxraw_;
  float *gfeat_ = nullptr, *gxfeat_ = nullptr; // the global outputs, in the output block
  double *gmom_ = nullptr;
  size_t ogfeat_ = 0, ogxfeat_ = 0, ogmom_ = 0, ohash_ = 0;
  Blocks *blocks_ = nullptr; // output blocks of this configuration
  // reusable cross-cell summary accumulators (sized c_*NF / c_*NMOM), refilled per level
  std::vector<float> sfmn_, sfmx_;
  std::vector<double> sfsm_, sfsq_, smmn_, smmx_, smsm_, smsq_;
  std::vector<size_t> graw_shape_, gfeat_shape_, gmom_shape_, gxraw_shape_, gxfeat_shape_;

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
  // halves convert separately and meet in one rounding. `rec` / `recx` are the same cells'
  // flushed slots, for the moments and the channel pairs; `cell` is the first one's index.
  void derive_block(const int64_t *IF_RESTRICT sums, size_t nxs, size_t cell, size_t cells,
                    int64_t n)
  {
    const DDV dd;
    const hn::RebindToSigned<DDV> di;
    const hn::Rebind<float, DDV> df;
    using V = hn::VFromD<DDV>;
    const Level &L = levels_[0];
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
        V past = hn::Add(S[SXX], S[SYY]), rest = S[CNT]; // as the flush derives these slots
        for (int j = 0; j < HB - 1; ++j)
        {
          const V cum = sum(K_HOG + j);
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
        V a[NMOM] = {sum_signed(K_T1), sum(K_T2), sum_signed(K_T3), sum(K_T4)}, t[NMOM], q;
        moments_recentre(a, (double)n, invn, t, q);
        moments_tail(t, hn::Add(q, hn::Set(dd, (double)KBIAS)), hn::Set(dd, invn), M);
        store_moments(M, std::min<size_t>(LD, cells - i), (size_t)c_ * NMOM,
                      L.mom + ((cell + i) * c_ + k) * NMOM);
        derive_lanes(S, M, hn::Set(dd, invn), hog_card_, out);
        float *IF_RESTRICT f = L.feat + ((cell + i) * c_ + k) * NMAP;
        scatter(out, std::min<size_t>(LD, cells - i), (size_t)c_ * NMAP, f);
      }
    for (size_t i = 0; i < cells && np_; ++i)
    {
      int64_t x[XMAX * (XMAX - 1) / 2];
      for (int p = 0; p < np_; ++p) x[p] = pair_sum(sums, nxs, i, p, n);
      derive_cross(L.mom + (cell + i) * c_ * NMOM, invn, x, L.xfeat + (cell + i) * np_ * NXF);
    }
  }
  // sum(v_i * v_j) of cell `i` of the sums, from the centred products
  int64_t pair_sum(const int64_t *IF_RESTRICT sums, size_t nxs, size_t i, int p, int64_t n) const
  {
    const int64_t B = KBIAS, ti = sums[((size_t)pi_[p] * K_N + K_T1) * nxs + i],
                  tj = sums[((size_t)pj_[p] * K_N + K_T1) * nxs + i];
    return sums[((size_t)c_ * K_N + p) * nxs + i] + B * (ti + tj) + B * B * n;
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

  // Cross-channel covariance and Pearson correlation per channel pair, from the raw
  // products and the cell's already-derived per-channel [mean, var].
  void derive_cross(const double *IF_RESTRICT m, double invn, const int64_t *IF_RESTRICT x,
                    float *IF_RESTRICT f) const
  {
    for (int p = 0; p < np_; ++p)
    {
      const double *mi = m + (size_t)pi_[p] * NMOM, *mj = m + (size_t)pj_[p] * NMOM;
      const double cov = (double)x[p] * invn - mi[0] * mj[0];
      const double den = std::sqrt(mi[1] * mj[1]);
      f[p * NXF] = (float)cov;
      f[p * NXF + 1] = (float)(den > 0.0 ? cov / den : 0.0);
    }
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
                int64_t *IF_RESTRICT hs, int64_t *IF_RESTRICT dst, int64_t *IF_RESTRICT dstx)
  {
    const size_t N = hn::Lanes(d8), nfx = (size_t)levels_[0].nx;
    // When a block holds whole cells and the cell row is one chunk of rows, a block finishes
    // its cells: they are flushed at once from a block's worth of sums that never leaves L1.
    // Otherwise the sums are kept for the whole cell row and flushed at its end.
    const bool local = multi_ && rws.size() <= (size_t)KROWS;
    const int64_t n = (int64_t)rws.size() * ncols_; // samples per cell, the same for the whole row
    // every channel in one pass of the blocks, and blocks of enough cells to fill derive's vectors
    const bool derives = stream_ && local && (np_ || c_ == 1) && block_cells_ >= (size_t)LD;
    const size_t nxs = local ? block_cells_ | 1 : cells_stride(); // never above cells_stride()
    const size_t pstride = (size_t)slots_ * (size_t)s.row_stride, lanes = (size_t)lanes_;
    V16 rays[HB - 1];
    for (int j = 0; j < HB - 1; ++j)
      rays[j] = hn::InterleaveLower(d16, hn::Set(d16, ray_[j][0]), hn::Set(d16, ray_[j][1]));
    // Without channel pairs the channels are independent, so they go one at a time and the
    // accumulators stay in L1 however many there are.
    const int G = np_ ? c_ : 1;
    for (size_t q0 = 0; q0 < rws.size(); q0 += KROWS)
    {
      const size_t nr = std::min<size_t>(KROWS, rws.size() - q0);
      for (size_t q = 0; q < nr; ++q)
      {
        // A tap is a row of the window plus its column offset (tapcol_). The rows are r-1, r,
        // r+1, replicated at the image edge, and r -/+ each bard lag: all within r's halo, so
        // their slots are r's moved by as much, wrapped.
        const int r = rws[q0 + q], sr = s.rslot[q0 + q];
        auto row = [&](int y)
        {
          int v = sr + (y - r);
          v += v < 0 ? slots_ : 0;
          v -= v >= slots_ ? slots_ : 0;
          return (size_t)v * s.row_stride + PAD_L;
        };
        s.vok[q] = 0;
        for (int j = 0; j < BARD_NL; ++j)
          if (r >= BARD_LAGS[j] && r < h_ - BARD_LAGS[j]) s.vok[q] |= bard_lagmask_ & (1 << j);
        size_t rows3[3], up[BARD_NL], down[BARD_NL];
        for (int i = 0; i < 3; ++i) rows3[i] = row(clampi(r - 1 + i, h_ - 1));
        for (int j = 0; j < BARD_NL; ++j)
        {
          const int d = (s.vok[q] >> j) & 1 ? BARD_LAGS[j] : 0;
          up[j] = row(r - d), down[j] = row(r + d);
        }
        for (int k = 0; k < c_; ++k)
        {
          const uint8_t *IF_RESTRICT base = s.planes.data() + (size_t)k * pstride;
          const uint8_t **p = &s.rows[(q * c_ + k) * KPTR];
          for (int i = 0; i < 9; ++i) p[i] = base + rows3[i / 3] + tapcol_[i];
          for (int j = 0; j < BARD_NL; ++j)
          {
            p[9 + 4 * j] = base + rows3[1] + tapcol_[9 + 4 * j];
            p[10 + 4 * j] = base + rows3[1] + tapcol_[10 + 4 * j];
            p[11 + 4 * j] = base + up[j];
            p[12 + 4 * j] = base + down[j];
          }
        }
      }
      std::fill_n(s.rowv.begin(), nr * c_, hn::Zero(d32));
      for (int k0 = 0; k0 < c_; k0 += G)
        for (size_t x0 = 0, cell = 0; x0 < lanes;
             x0 += multi_ ? N : (size_t)cw_, cell += block_cells_)
          for (size_t xv = 0; xv < (multi_ ? N : (size_t)cw_); xv += N)
          {
            const size_t x = x0 + xv, e0 = local ? 0 : cell; // cell == x0 / cw_
            // Blocks that span cells can overrun the row: the surplus lanes are masked off too.
            const bool tail = multi_ && x + N > lanes, msk = masked_ || tail;
            const uint8_t *IF_RESTRICT mask = tail ? ktail_.data() : kmask_.data() + xv;
            // only a block within MAXLAG columns of a row end can hold an invalid bard tap
            const bool edge =
                x * phases_ < (size_t)BARD_MAXLAG || (x + N) * phases_ + BARD_MAXLAG > (size_t)w_;
            const uint8_t *IF_RESTRICT valid = edge ? bvalid_.data() : nullptr;
            V32 cross[XMAX * (XMAX - 1) / 2][2];
            V16 w[XMAX][2];
            for (size_t q = 0; q < nr; ++q) // row 0 of a block assigns its accumulators
            {
              for (int k = 0; k < G; ++k)
              {
                const uint8_t *const *rp = &s.rows[(q * c_ + k0 + k) * KPTR];
                auto run = [&](auto m, auto first)
                {
                  kernel<decltype(m)::value, decltype(first)::value>(
                      rp, x, mask, valid, bvalid_stride_, bard_lagmask_, s.vok[q], rays, s.acc[k],
                      w[k], s.rowv[q * c_ + k0 + k]);
                };
                if (msk)
                  q ? run(std::true_type(), std::false_type())
                    : run(std::true_type(), std::true_type());
                else
                  q ? run(std::false_type(), std::false_type())
                    : run(std::false_type(), std::true_type());
              }
              for (int p = 0; p < np_; ++p)
                for (int h = 0; h < 2; ++h)
                  add_to(q > 0, cross[p][h],
                         hn::WidenMulPairwiseAdd(d32, w[pi_[p]][h], w[pj_[p]][h]));
            }
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
            for (int k = 0; k < G; ++k)
            {
              const Acc &A = s.acc[k];
              int64_t *IF_RESTRICT cb = s.cells.data() + (size_t)(k0 + k) * K_N * nxs;
              for (int t = K_U8; t < K_N; ++t) // both kinds fold as uint16 lanes of column pairs
              {
                const auto sums =
                    t < K_BARD ? hn::SumsOf2(msk ? hn::And(A.u8[t - K_U8], mv) : A.u8[t - K_U8])
                               : A.bard[t - K_BARD];
                fold32<false>(hn::BitCast(d32, hn::PromoteLowerTo(du32, sums)), g / 2, adds[0],
                              cb + t * nxs + at[0]);
                fold32<false>(hn::BitCast(d32, hn::PromoteUpperTo(du32, sums)), g / 2, adds[2],
                              cb + t * nxs + at[2]);
              }
              for (int h = 0; h < 2; ++h)
              {
                const bool ad = adds[2 * h];
                int64_t *IF_RESTRICT ch = cb + at[2 * h];
                // the column profile: one lane per column, straight into the band's sums
                int32_t *IF_RESTRICT cp =
                    s.col.data() + (size_t)(k0 + k) * (lanes + N) + x + h * N / 2;
                const V32 lo = hn::PromoteLowerTo(d32, A.col[h]),
                          hi = hn::PromoteUpperTo(d32, A.col[h]);
                hn::StoreU(hn::Add(hn::LoadU(d32, cp), lo), d32, cp);
                hn::StoreU(hn::Add(hn::LoadU(d32, cp + N / 4), hi), d32, cp + N / 4);
                for (int t = 0; t < K_NUNSIGNED; ++t)
                  fold32<false>(A.pair[h][t], g / 2, ad, ch + t * nxs);
                for (int t = K_NUNSIGNED; t < K_NPAIR; ++t)
                  fold32<true>(A.pair[h][t], g / 2, ad, ch + t * nxs);
                fold(d64, A.t4[h], g / 4, ad, ch + K_T4 * nxs);
              }
              for (int q = 0; q < 4; ++q)
              {
                int64_t *IF_RESTRICT cq = cb + at[q];
                fold(d64, A.sg4[q], gq / 2, adds[q], cq + K_SG4 * nxs);
                for (int j = 0; j < HB - 1; ++j)
                  fold32<false>(A.hog[q][j], gq, adds[q], cq + (K_HOG + j) * nxs);
              }
            }
            for (int p = 0; p < np_; ++p)
              for (int h = 0; h < 2; ++h)
                fold32<true>(cross[p][h], g / 2, adds[2 * h],
                             s.cells.data() + (c_ * K_N + p) * nxs + at[2 * h]);
            if (local)
            {
              const size_t e = cell, cells = std::min(block_cells_, nfx - e);
              if (!(derives && rolls)) flush(s.cells.data(), nxs, e, cells, k0, G, n, dst, dstx);
              if (derives)
              {
                derive_block(s.cells.data(), nxs, cy * nfx + e, cells, n);
                if (bands_.size() == 1) summary_fold(levels_[0], cy * nfx + e, cells);
                if (rolls) roll_block(s, s.cells.data(), nxs, cy, e, cells, n, hs);
              }
            }
          }
      for (size_t q = 0; q < nr; ++q)
        for (int k = 0; k < c_; ++k)
          rowprof_[(s.rowidx + q0 + q) * c_ + k] = hn::ReduceSum(d32, s.rowv[q * c_ + k]);
    }
    if (!local) flush(s.cells.data(), nxs, 0, nfx, 0, c_, n, dst, dstx);
    if (derives && rolls) s.up_n += n * levels_[1].fx;
    return derives;
  }

  // features(): a block's cell sums straight up into the level-1 row in flight -- the finest
  // cells are never turned into slots at all. Sums add as they are (the slots are linear in
  // them and in n, so the conversion waits for flush_up()); the first child row of a parent
  // stores. The hash grid takes its S1, and the pair sums go up as the slots they already are.
  void roll_block(Scratch &s, const int64_t *IF_RESTRICT sums, size_t nxs, size_t cy, size_t e,
                  size_t cells, int64_t n, int64_t *IF_RESTRICT hs)
  {
    const Level &fine = levels_[0];
    Level &lv = levels_[1];
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
    const size_t up0 = (cy >> log2((size_t)lv.fy)) * lv.nx, bxl = log2((size_t)fine.nx / HN);
    int64_t *IF_RESTRICT hrow =
        hfast_ ? hs + (cy >> log2((size_t)fine.ny / HN)) * HN * c_ : nullptr;
    for (size_t i = 0; i < cells; ++i)
    {
      int64_t *IF_RESTRICT x = lv.xbuf.data() + (up0 + ((e + i) >> fxl)) * np_;
      for (int p = 0; p < np_; ++p)
        x[p] = first(e + i) ? pair_sum(sums, nxs, i, p, n) : x[p] + pair_sum(sums, nxs, i, p, n);
      if (hfast_) // the finest grid tiles the hash grid: block-sum its S1 = T1 + KBIAS * n
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
  void flush_up(Scratch &s, size_t cy)
  {
    Level &lv = levels_[1];
    flush(s.up.data(), (size_t)lv.nx + hn::Lanes(d64), 0, (size_t)lv.nx, 0, c_, s.up_n,
          lv.buf.data() + (cy >> up_shift_) * lv.nx * c_ * NSUM, nullptr);
    s.up_n = 0;
  }

  // Cell sums -> accumulator slots, for the `cells` cells from cell0 on that the sums hold (a
  // block's, or the whole row's), `nxs` apart, and for channels [k0, k0 + G).
  void flush(const int64_t *IF_RESTRICT sums, size_t nxs, size_t cell0, size_t cells, int k0, int G,
             int64_t n, int64_t *IF_RESTRICT dst, int64_t *IF_RESTRICT dstx) const
  {
    const int64_t B = KBIAS;
    for (size_t i = 0; i < cells; ++i)
    {
      const size_t e = cell0 + i, src = i; // the sums start at the first cell they hold
      for (int k = k0; k < k0 + G; ++k)
      {
        int64_t *IF_RESTRICT a = dst + (e * c_ + k) * NSUM;
        const int64_t *IF_RESTRICT cb = sums + (size_t)k * K_N * nxs + src;
        auto sum = [&](int t) { return cb[(size_t)t * nxs]; };
        a[SXX] = sum(K_SXX);
        a[SYY] = sum(K_SYY);
        a[SXY] = sum(K_SXY);
        a[CNT] = n;
        int64_t past = a[SXX] + a[SYY]; // every pixel is past "ray -1": the total |grad|^2
        for (int j = 0; j < HB - 1; ++j)
        {
          a[HOG0 + j] = past - sum(K_HOG + j);
          past = sum(K_HOG + j);
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
      if (G == c_ && dstx) // (the level-1 row in flight gathers its pair sums as it goes: no dstx)
        for (int p = 0; p < np_; ++p)
        {
          dstx[e * np_ + p] = pair_sum(sums, nxs, src, p, n);
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
    std::fill(s.col.begin(), s.col.end(), 0);
    if (stream_)
    {
      std::fill(s.graw.begin(), s.graw.end(), (int64_t)0);
      std::fill(s.gxraw.begin(), s.gxraw.end(), (int64_t)0);
      s.held.clear();
      s.heldx.clear();
      s.held_cy.clear();
      if (bands_.size() == 1) summary_reset();
    }
    // The rolling window fills as the cell rows advance, from the first row any tap can reach.
    s.filled = std::max(0, bands_[bi].r0 - BARD_MAXLAG);
    s.fslot = s.filled % slots_;
    // A cell row's sampled rows all fold into the same cells, so everything runs per cell
    // row, once the window holds those rows and their halo.
    const int chh = h_ / fine.ny;
    std::vector<int> &rows = s.rws;
    // the band's sampled rows in order: the next one, its window slot and its index from the top
    int next = (bands_[bi].cy0 * chh + sy_ - 1) / sy_ * sy_, next_slot = next % slots_;
    s.rowidx = (size_t)(next / sy_);
    rows.clear(); // (the last frame's last cell row)
    for (int cy = bands_[bi].cy0; cy < bands_[bi].cy1; ++cy)
    {
      s.rowidx += rows.size(); // past the previous cell row's
      rows.clear();
      s.rslot.clear();
      for (; next < (cy + 1) * chh; next += sy_)
      {
        rows.push_back(next);
        s.rslot.push_back(next_slot);
        next_slot += symod_;
        next_slot -= next_slot >= slots_ ? slots_ : 0;
      }
      // raw(): the row goes into the pyramid. features(): into a row buffer that stays in cache
      // while stream_row() derives it and rolls it up, so the finest level is never stored.
      int64_t *IF_RESTRICT dst = stream_ ? s.row.data() : fine.buf.data() + cy * nfx * c_ * NSUM;
      int64_t *IF_RESTRICT dstx = stream_ ? s.rowx.data() : fine.xbuf.data() + cy * nfx * np_;
      // A band rolls a row up from its sums when it also has the first row of that row's parents;
      // otherwise the row becomes slots, which wait for the join (see Scratch::held).
      const int fy1 = 1 << up_shift_; // level 1's fy, or 1 without a level 1
      const bool owner = (cy & ~(fy1 - 1)) >= bands_[bi].cy0, rolls = stream_ && rolls_ && owner;
      bool derived = false;
      if (rows.empty()) // the stride skipped the whole cell row. Every other one is written, not
      {                 // added to, by cell_row(), so nothing else ever needs zeroing.
        std::fill_n(dst, nfx * c_ * NSUM, (int64_t)0);
        std::fill_n(dstx, nfx * np_, (int64_t)0);
      }
      else
      {
        for (; s.filled <= std::min(h_ - 1, rows.back() + BARD_MAXLAG); ++s.filled)
        {
          deinterleave_row(img, rs, cs, coff_.data(), c_, w_, s.filled, interleaved_, s.row_stride,
                           slots_, s.fslot, phases_, s.phase_stride, s.lin.data(), s.planes.data());
          s.fslot = s.fslot + 1 == slots_ ? 0 : s.fslot + 1;
        }
        derived = cell_row(s, rows, (size_t)cy, rolls, hs, dst, dstx);
        if (!hfast_)
          for (size_t i = 0; i < rows.size(); ++i)
            accumulate_hash_row(s.planes.data() + (size_t)s.rslot[i] * s.row_stride + PAD_L,
                                (size_t)slots_ * (size_t)s.row_stride, c_, step_, hrbeg_.data(),
                                hrend_.data(), hrhc_.data(), (int)hrhc_.size(),
                                hs + (size_t)hrow_[rows[i]] * HN * c_);
      }
      if (rolls)
      {
        if (rows.empty()) // nothing went up: derive the zeros, and start the parents at zero
        {
          derive_run(fine, (size_t)cy * nfx, nfx, dst, dstx, bands_.size() == 1);
          if ((cy & (fy1 - 1)) == 0)
          {
            std::fill(s.up.begin(), s.up.end(), (int64_t)0);
            std::fill_n(levels_[1].xbuf.begin() + (cy >> up_shift_) * levels_[1].nx * np_,
                        levels_[1].nx * np_, (int64_t)0);
          }
        }
        if ((cy & (fy1 - 1)) == fy1 - 1 || cy == bands_[bi].cy1 - 1) flush_up(s, (size_t)cy);
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
      derive_run(fine, (size_t)cy * nfx, nfx, s.row.data(), s.rowx.data(), bands_.size() == 1);
    if (levels_.size() == 1) // no coarser level: straight into the band's share of the global
    {
      for (size_t e = 0; e < nfx; ++e)
        for (size_t t = 0; t < stripe; ++t) s.graw[t] += s.row[e * stripe + t];
      for (size_t e = 0; e < nfx; ++e)
        for (int t = 0; t < np_; ++t) s.gxraw[t] += s.rowx[e * np_ + t];
      return;
    }
    if (!owner)
    {
      s.held.insert(s.held.end(), s.row.begin(), s.row.end());
      s.heldx.insert(s.heldx.end(), s.rowx.begin(), s.rowx.end());
      s.held_cy.push_back(cy);
      return;
    }
    roll_row(cy, s.row.data(), s.rowx.data());
  }
  // The finest row cy (sums at row / rowx) into its level-1 parents: a parent is written by
  // its first child and added to by the rest.
  void roll_row(int cy, const int64_t *IF_RESTRICT row, const int64_t *IF_RESTRICT rowx)
  {
    Level &lv = levels_[1];
    const size_t i = (size_t)cy / lv.fy, stripe = (size_t)c_ * NSUM;
    for (size_t j = 0; j < (size_t)lv.nx; ++j)
    {
      int64_t *IF_RESTRICT dc = lv.buf.data() + (i * lv.nx + j) * stripe;
      int64_t *IF_RESTRICT dx = lv.xbuf.data() + (i * lv.nx + j) * np_;
      for (size_t kx = 0; kx < (size_t)lv.fx; ++kx)
      {
        const int64_t *IF_RESTRICT sc = row + (j * lv.fx + kx) * stripe;
        const int64_t *IF_RESTRICT sx = rowx + (j * lv.fx + kx) * np_;
        if (cy % lv.fy == 0 && kx == 0)
        {
          std::copy(sc, sc + stripe, dc);
          std::copy(sx, sx + np_, dx);
          continue;
        }
        for (size_t t = 0; t < stripe; ++t) dc[t] += sc[t];
        for (int t = 0; t < np_; ++t) dx[t] += sx[t];
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
      if (quit_)
        return;
      seen = epoch_;
      const Job j = job_;
      lk.unlock();
      if (j == JOB_ACC)
        accumulate_band(task_img_, task_rs_, task_cs_, bi);
      else
        derive_cells(bi);
      lk.lock();
      if (--pending_ == 0)
        cv_done_.notify_one();
    }
  }

  // Workers park on a condvar between frames, so a frame costs one broadcast and one
  // barrier rather than thread creation. Parking (not spinning) is deliberate: the
  // cores are expected to be shared with other real-time work between frames.
  void run_job(Job j)
  {
    const int nb = (int)bands_.size();
    auto one = [&](int bi) {
      if (j == JOB_ACC)
        accumulate_band(task_img_, task_rs_, task_cs_, bi);
      else
        derive_cells(bi);
    };
    if (workers_.empty())
    {
      for (int bi = 0; bi < nb; ++bi)
        one(bi);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
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
    if (workers_.empty())
      return;
    {
      std::lock_guard<std::mutex> lk(mu_);
      quit_ = true;
    }
    cv_go_.notify_all();
    for (std::thread &t : workers_)
      t.join();
    workers_.clear();
    quit_ = false;
  }

  void accumulate(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    const bool hashing = stream_ || !hfast_; // the hash grid sums: per pixel, or from streamed rows
    if (hashing)
    {
      std::fill(hsum_.begin(), hsum_.end(), (int64_t)0);
      for (size_t b = 1; b < scr_.size(); ++b)
        std::fill(scr_[b].hsum.begin(), scr_[b].hsum.end(), (int64_t)0);
    }

    for (int k = 0; k < c_; ++k)
      coff_[k] = (int64_t)chan_[k] * chs;
    interleaved_ = chs == 1 && cs == c_ && chan_[0] == 0 && chan_.back() == c_ - 1;
    task_img_ = img;
    task_rs_ = rs;
    task_cs_ = cs;
    run_job(JOB_ACC);

    if (hashing) // fold the bands' private hash partials; int64 sums, so still exact
      for (size_t b = 1; b < scr_.size(); ++b)
        for (size_t t = 0; t < hsum_.size(); ++t)
          hsum_[t] += scr_[b].hsum[t];

    const int stripe = c_ * NSUM;
    for (const Scratch &sc : scr_) // the rows whose parents another band had to write first
      for (size_t r = 0; r < sc.held_cy.size(); ++r)
        roll_row(sc.held_cy[r], sc.held.data() + r * levels_[0].nx * stripe,
                 sc.heldx.data() + r * levels_[0].nx * np_);
    for (size_t lvl = stream_ ? 2 : 1; lvl < levels_.size(); ++lvl) // stream_row() fed level 1
    {
      const Level &p = levels_[lvl - 1];
      Level &lv = levels_[lvl];
      for (int i = 0; i < lv.ny; ++i) // a parent is written as its first child, then added to
        for (int j = 0; j < lv.nx; ++j)
        {
          const size_t dst = (size_t)i * lv.nx + j;
          int64_t *IF_RESTRICT dc = lv.buf.data() + dst * stripe;
          int64_t *IF_RESTRICT dx = lv.xbuf.data() + dst * np_;
          for (int ky = 0, first = 1; ky < lv.fy; ++ky)
            for (int kx = 0; kx < lv.fx; ++kx, first = 0)
            {
              const size_t src = (size_t)(i * lv.fy + ky) * p.nx + j * lv.fx + kx;
              const int64_t *IF_RESTRICT sc = p.buf.data() + src * stripe;
              const int64_t *IF_RESTRICT sx = p.xbuf.data() + src * np_;
              if (first)
              {
                std::copy(sc, sc + stripe, dc);
                std::copy(sx, sx + np_, dx);
                continue;
              }
              for (int t = 0; t < stripe; ++t) dc[t] += sc[t];
              for (int t = 0; t < np_; ++t) dx[t] += sx[t];
            }
        }
    }

    const Level &last = levels_.back();
    std::fill(graw_.begin(), graw_.end(), (int64_t)0);
    std::fill(gxraw_.begin(), gxraw_.end(), (int64_t)0);
    if (stream_ && levels_.size() == 1) // the only level was never stored: the bands summed it
    {
      for (const Scratch &sc : scr_)
      {
        for (int t = 0; t < stripe; ++t) graw_[t] += sc.graw[t];
        for (int t = 0; t < np_; ++t) gxraw_[t] += sc.gxraw[t];
      }
      return;
    }
    for (size_t cell = 0, nc = (size_t)last.ny * last.nx; cell < nc; ++cell)
    {
      const int64_t *IF_RESTRICT s = last.buf.data() + cell * stripe;
      for (int t = 0; t < stripe; ++t)
        graw_[t] += s[t];
      const int64_t *IF_RESTRICT x = last.xbuf.data() + cell * np_;
      for (int t = 0; t < np_; ++t)
        gxraw_[t] += x[t];
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
  // Mean intensity along every sampled row, and along every sampled column (the lanes on the
  // sampling mesh, in order), from the sums of centred pixels the kernel kept.
  void derive_profiles()
  {
    const size_t nr = prow_shape_[0], nc = pcol_shape_[0], lanes = lanes_;
    for (size_t i = 0; i < nr * c_; ++i) prow_[i] = KBIAS + (double)rowprof_[i] / (double)nc;
    for (int k = 0; k < c_; ++k)
      for (size_t j = 0; j < mesh_lanes_.size(); ++j)
      {
        int64_t sum = 0;
        for (const Scratch &sc : scr_)
          sum += sc.col[(size_t)k * (lanes + hn::Lanes(d8)) + mesh_lanes_[j]];
        pcol_[j * c_ + k] = KBIAS + (double)sum / (double)nr;
      }
  }

  void derive_hashes()
  {
    const int NHC = HN * HN;
    std::vector<double> mean(NHC + LD), d0((size_t)HS * HN);
    for (int k = 0; k < c_; ++k)
    {
      // the same division per cell, a vector of cells at a time (an empty cell divides by 1)
      const size_t LH = hn::Lanes(DDV());
      for (int i = 0; i < NHC; i += (int)LH)
      {
        double num[LD];
        for (size_t l = 0; l < LH; ++l) num[l] = (double)hsum_[(size_t)(i + l) * c_ + k];
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
      for (int i = 0; i < HS * HS; ++i)
        avg += m8[i];
      avg /= HS * HS;
      double srt[HS * HS];
      std::copy(m8, m8 + HS * HS, srt);
      std::sort(srt, srt + HS * HS);
      const double med = 0.5 * (srt[HS * HS / 2 - 1] + srt[HS * HS / 2]);
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
          for (int n = 0; n < HN; ++n)
            acc += dctb_[(size_t)vv * HN + n] * d0[(size_t)u * HN + n];
          dl[u * HS + vv] = acc;
        }
      double psrt[HS * HS];
      std::copy(dl, dl + HS * HS, psrt);
      std::sort(psrt, psrt + HS * HS);
      const double pmed = 0.5 * (psrt[HS * HS / 2 - 1] + psrt[HS * HS / 2]);
      uint64_t p = 0;
      for (int i = 0; i < HS * HS; ++i)
        p |= (uint64_t)(dl[i] > pmed) << (63 - i);
      hash_[2 * (size_t)c_ + k] = p;
    }
  }

  // Per-cell derivation. Every cell writes only its own feat/mom/xfeat slice, so this
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
    for (int k = 0; k < c_; ++k)
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
    for (int k = 0; k < c_; ++k)
    {
      const size_t stride = (size_t)c_ * NST; // feature axis first, as FEATURE_NAMES runs
      write_summary(NF, nc, &sfmn_[k * NF], &sfmx_[k * NF], &sfsm_[k * NF], &sfsq_[k * NF],
                    L.sum + (size_t)k * NST, stride);
      write_summary(NMOM, nc, &smmn_[k * NMOM], &smmx_[k * NMOM], &smsm_[k * NMOM],
                    &smsq_[k * NMOM], L.sum + (NF * (size_t)c_ + k) * NST, stride);
    }
  }

  // Derive band bi's cells, SB at a time. The summary is a serial fold in cell order, so that
  // it rounds the same for any thread count: a lone band folds each batch while it is still in
  // L1, several bands leave it to derive().
  static constexpr size_t SB = 32;
  // Derive `cells` cells of L from cell0 on, whose sums start at buf / xbuf.
  void derive_run(const Level &L, size_t cell0, size_t cells, const int64_t *IF_RESTRICT buf,
                  const int64_t *IF_RESTRICT xbuf, bool fold)
  {
    double invn[SB * XMAX * 2], inverse = 0.0; // 1/n per cell-channel of a batch
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
      for (size_t i = 0; i < nb && np_; ++i)
        derive_cross(L.mom + (cell0 + b0 + i) * c_ * NMOM, invn[i * c_], xbuf + (b0 + i) * np_,
                     L.xfeat + (cell0 + b0 + i) * np_ * NXF);
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
      derive_run(L, c0, c1 - c0, L.buf.data() + c0 * c_ * NSUM, L.xbuf.data() + c0 * np_, nb == 1);
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
    double ginvns[XMAX * 2];
    std::vector<double> many((size_t)c_ > XMAX * 2 ? (size_t)c_ : 0, ginvn);
    std::fill(ginvns, ginvns + XMAX * 2, ginvn);
    moments_records(graw_.data(), many.empty() ? ginvns : many.data(), (size_t)c_, gmom_);
    derive_records(graw_.data(), gmom_, many.empty() ? ginvns : many.data(), (size_t)c_, gfeat_);
    if (np_) derive_cross(gmom_, ginvn, gxraw_.data(), gxfeat_);
  }

public:
  FeatureComputer() = default;

  void set_config(const std::vector<int64_t> &dims, const std::vector<int> &channels,
                  const std::vector<std::vector<int>> &grids,
                  const std::vector<int64_t> &stride, int threads)
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
    sy_ = (int)stride[0];
    sx_ = (int)stride[1];

    pi_.clear();
    pj_.clear();
    if (c_ >= 2 && c_ <= XMAX)
      for (int i = 0; i < c_; ++i)
        for (int j = i + 1; j < c_; ++j)
        {
          pi_.push_back(i);
          pj_.push_back(j);
        }
    np_ = (int)pi_.size();
    // A lag whose span exceeds the smaller side is dropped on BOTH axes, matching the
    // dense reference maps rather than testing each axis independently.
    bard_lagmask_ = 0;
    for (int j = 0; j < BARD_NL; ++j)
      if (BARD_SPAN * BARD_LAGS[j] < std::min(h_, w_)) bard_lagmask_ |= 1 << j;

    for (int k = 1; k < HB; ++k)
    {
      const double a = PI * k / HB;
      ray_[k - 1][0] = (int16_t)std::llround(RAYSCALE * std::cos(a));
      ray_[k - 1][1] = (int16_t)-std::llround(RAYSCALE * std::sin(a));
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
      L.oxfeat = place(nc * np_ * NXF * sizeof(float));
      L.xbuf.assign(nc * np_, 0);
      L.rshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NSUM};
      L.fshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_ * NMAP};
      L.mshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NMOM};
      L.xrshape = {(size_t)L.ny, (size_t)L.nx, (size_t)np_};
      L.xfshape = {(size_t)L.ny, (size_t)L.nx, (size_t)np_, (size_t)NXF};
      L.sumshape = {(size_t)NMAP, (size_t)c_, (size_t)NST};
      levels_.push_back(std::move(L));
    }

    // Split the planes by column phase while the stride and the cell stay even, so the sampled
    // columns end up adjacent in phase 0 and kernel() wastes no lanes on the rest -- but only
    // down to cells of whole lane quads, the coarsest geometry an accumulator folds from.
    {
      const int cwf = w_ / levels_[0].nx, N = (int)hn::Lanes(d8);
      for (phases_ = 1;
           phases_ < PHASES_MAX && sx_ % (2 * phases_) == 0 && cwf % (8 * phases_) == 0;)
        phases_ *= 2;
      // kernel() geometry, in lanes of phase 0. Blocks span whole cells when a cell divides
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
      const int64_t phase_stride = (int64_t)lanes_ + PAD_L + PAD_R;
      auto col = [&](int d)
      {
        const int f = ((d % phases_) + phases_) % phases_;
        return f * phase_stride + (d - f) / phases_;
      };
      for (int i = 0; i < 9; ++i) tapcol_[i] = col(i % 3 - 1);
      for (int j = 0; j < BARD_NL; ++j)
      {
        tapcol_[9 + 4 * j] = col(-BARD_LAGS[j]);
        tapcol_[10 + 4 * j] = col(BARD_LAGS[j]);
      }
      mesh_lanes_.clear();
      for (size_t l = 0; l < lanes_; ++l)
        if ((l % cw_) % step_ == 0) mesh_lanes_.push_back((uint32_t)l);
      // when cell_row() derives by the block, and there is a level for the sums to go up to
      rolls_ = multi_ && N / cw_ >= LD && (np_ || c_ == 1) && levels_.size() > 1 &&
               (h_ / levels_[0].ny + sy_ - 1) / sy_ <= KROWS;
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
    // amortizes over more rows. The window has to span the cell row plus a halo.
    slots_ = std::max(2 * BARD_MAXLAG + 1, h_ / levels_[0].ny + 2 * BARD_MAXLAG + 1);
    symod_ = sy_ % slots_;
    for (up_shift_ = 0; levels_.size() > 1 && (1 << up_shift_) < levels_[1].fy;) ++up_shift_;
    coff_.assign(c_, 0);

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
    gxraw_.assign((size_t)np_, 0);
    ogxfeat_ = place((size_t)np_ * NXF * sizeof(float));
    graw_shape_ = {(size_t)c_, (size_t)NSUM};
    gfeat_shape_ = {(size_t)c_ * NMAP};
    gmom_shape_ = {(size_t)c_, (size_t)NMOM};
    gxraw_shape_ = {(size_t)np_};
    gxfeat_shape_ = {(size_t)np_, (size_t)NXF};

    // --- perceptual hashes: HN x HN value-sum grid, filled in the same pass ---
    hrow_.resize(h_);
    for (int r = 0; r < h_; ++r)
      hrow_[r] = (int)((int64_t)r * HN / h_);
    hcol_.resize(w_);
    for (int c = 0; c < w_; ++c)
      hcol_[c] = (int)((int64_t)c * HN / w_);
    // Per-cell sampled-pixel counts + the hash-column run structure. Columns are
    // sampled at sx_ (restarted per finest cell) and visited in increasing order.
    std::vector<int64_t> rcnt(HN, 0), ccnt(HN, 0);
    for (int r = 0; r < h_; r += sy_)
      rcnt[hrow_[r]]++;
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
          while (c < e1 && hcol_[c] == hc)
            ccnt[hc]++, c += sx_;
          hrbeg_.push_back(b / phases_); // runs are read from phase 0, so in its lanes
          hrend_.push_back(c / phases_);
          hrhc_.push_back(hc);
        }
      }
    }
    hcnt_.assign((size_t)HN * HN, 0);
    for (int i = 0; i < HN; ++i)
      for (int j = 0; j < HN; ++j)
        hcnt_[(size_t)i * HN + j] = rcnt[i] * ccnt[j];
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
    prow_shape_ = {(size_t)((h_ + sy_ - 1) / sy_), (size_t)c_};
    pcol_shape_ = {(size_t)levels_[0].nx * ((w_ / levels_[0].nx + sx_ - 1) / sx_), (size_t)c_};
    oprow_ = place(prow_shape_[0] * c_ * sizeof(double));
    opcol_ = place(pcol_shape_[0] * c_ * sizeof(double));
    rowprof_.assign(prow_shape_[0] * c_, 0);
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
      s.acc.resize((size_t)(np_ ? c_ : 1));
      s.rows.assign(nrows * c_ * KPTR, nullptr);
      s.vok.assign(nrows, 0);
      s.rowv.assign(nrows * c_, hn::Zero(d32));
      s.col.assign((size_t)c_ * ((size_t)(w_ / phases_) + N), 0);
      s.cells.assign(((size_t)c_ * K_N + np_) * cells_stride() + N, 0);
      s.phase_stride = (int64_t)(w_ / phases_) + PAD_L + PAD_R;
      s.row_stride = s.phase_stride * phases_;
      // a slack row and block so the final vector's overrun stays inside the allocation
      s.planes.assign(
          (size_t)c_ * (size_t)slots_ * (size_t)s.row_stride + (size_t)s.row_stride + 2 * N, 0);
      s.lin.assign((size_t)3 * c_ * (size_t)w_ + N, 0);
      if (b) s.hsum.assign((size_t)HN * HN * c_, 0);
      s.row.assign((size_t)levels_[0].nx * c_ * NSUM, 0);
      if (levels_.size() > 1) s.up.assign((size_t)c_ * K_N * ((size_t)levels_[1].nx + N), 0);
      s.rowx.assign((size_t)levels_[0].nx * np_, 0);
      s.graw.assign((size_t)c_ * NSUM, 0);
      s.gxraw.assign((size_t)np_, 0);
    }
    for (size_t b = 1; b < bands_.size(); ++b)
      workers_.emplace_back([this, b] { worker((int)b); });
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

  // The channel pairs, as flat [i0, j0, i1, j1, ...]; empty when cross-channel is off.
  std::vector<int> pairs() const
  {
    std::vector<int> out;
    for (int p = 0; p < np_; ++p)
    {
      out.push_back(pi_[p]);
      out.push_back(pj_[p]);
    }
    return out;
  }

  nb::list raw(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    stream_ = false;
    accumulate(img, rs, cs, chs);
    nb::list out, cross;
    for (Level &L : levels_)
    {
      out.append(nb::ndarray<nb::numpy, int64_t>(L.buf.data(), L.rshape.size(), L.rshape.data(),
                                                 nb::handle()));
      if (np_)
        cross.append(nb::ndarray<nb::numpy, int64_t>(L.xbuf.data(), L.xrshape.size(),
                                                     L.xrshape.data(), nb::handle()));
    }
    out.append(nb::ndarray<nb::numpy, int64_t>(graw_.data(), graw_shape_.size(),
                                               graw_shape_.data(), nb::handle()));
    if (np_)
      cross.append(nb::ndarray<nb::numpy, int64_t>(gxraw_.data(), gxraw_shape_.size(),
                                                   gxraw_shape_.data(), nb::handle()));
    nb::list both;
    both.append(out);
    both.append(cross);
    return both;
  }

  // Per level: the float32 `maps` rows, channel-major over NMAP, and the float64 moments
  // (moments span 0..255^4, where float32's 7 digits would cost precision).
  nb::list features(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    auto *blk = static_cast<uint8_t *>(blocks_->take());
    const nb::capsule owner(blk, Blocks::give); // every array below keeps the block alive
    auto at = [&](auto *&ptr, size_t offset)
    { ptr = reinterpret_cast<std::remove_reference_t<decltype(ptr)>>(blk + offset); };
    for (Level &L : levels_)
    {
      at(L.feat, L.ofeat);
      at(L.mom, L.omom);
      at(L.xfeat, L.oxfeat);
      at(L.sum, L.osum);
    }
    at(gfeat_, ogfeat_);
    at(gmom_, ogmom_);
    at(gxfeat_, ogxfeat_);
    at(hash_, ohash_);
    at(prow_, oprow_);
    at(pcol_, opcol_);
    stream_ = true;
    accumulate(img, rs, cs, chs);
    derive();
    derive_hashes();
    derive_profiles();
    auto view = [&](auto *data, const std::vector<size_t> &shape)
    {
      using T = std::remove_pointer_t<decltype(data)>;
      return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner);
    };
    nb::list feat, mom, cross, sums;
    for (Level &L : levels_)
    {
      feat.append(view(L.feat, L.fshape));
      mom.append(view(L.mom, L.mshape));
      if (np_) cross.append(view(L.xfeat, L.xfshape));
      sums.append(view(L.sum, L.sumshape));
    }
    feat.append(view(gfeat_, gfeat_shape_));
    mom.append(view(gmom_, gmom_shape_));
    if (np_) cross.append(view(gxfeat_, gxfeat_shape_));
    nb::list all;
    all.append(feat);
    all.append(mom);
    all.append(cross);
    all.append(sums);
    all.append(view(hash_, hshape_));
    all.append(view(prow_, prow_shape_));
    all.append(view(pcol_, pcol_shape_));
    return all;
  }
};

using Arr = nb::ndarray<nb::numpy, const uint8_t, nb::device::cpu>;
} // namespace

NB_MODULE(imfeat_core, m)
{
  m.doc() = "imfeat internal C++ module. Public API: imfeat.FeatureComputer.";
  m.attr("HB") = HB;     // HOG orientation-bin count; Python derives its bin labels from this
  m.attr("LBPB") = LBPB; // LBP^riu2 bin count (9 uniform + 1 non-uniform)
  m.attr("BARD_NL") = BARD_NL; // bar-detector lag count (sizes the spectrum block)
  m.attr("XMAX") = XMAX; // cross-channel products are computed only for C <= XMAX
  m.def("cpu_count", [] { //  0 when the runtime cannot tell; callers see at least 1
    const unsigned n = std::thread::hardware_concurrency();
    return (int)(n ? n : 1u);
  });
  nb::class_<FeatureComputer>(m, "_FeatureComputerImpl")
      .def(nb::init<>())
      .def("set_config", &FeatureComputer::set_config, nb::arg("dims"), nb::arg("channels"),
           nb::arg("grids"), nb::arg("stride"), nb::arg("threads"))
      .def("threads", &FeatureComputer::threads)
      .def("pairs", &FeatureComputer::pairs)
      .def(
          "raw",
          [](FeatureComputer &self, Arr a) {
            return self.raw(a.data(), a.stride(0), a.stride(1), a.ndim() > 2 ? a.stride(2) : 0);
          },
          nb::arg("arr"))
      .def(
          "features",
          [](FeatureComputer &self, Arr a) {
            return self.features(a.data(), a.stride(0), a.stride(1),
                                 a.ndim() > 2 ? a.stride(2) : 0);
          },
          nb::arg("arr"));
}
