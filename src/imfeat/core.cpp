#include "hwy/highway.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#define IF_RESTRICT __restrict
#else
#define IF_RESTRICT __restrict__
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
// Two nested loops, one traversal:
//   prepare_row  -- flat, branch-free, auto-vectorized over W*C: Sobel gx/gy and
//                   the 8-neighbourhood max/min envelopes for every pixel of the
//                   row. Border clamping happens here, so the hot loop has none.
//   accumulate_row -- channels-in-lanes SIMD over the row's cells; reads only the
//                   prepared rows (5 loads/pixel) and scatters into cell sums.

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
constexpr int NSUM = BARD_LIGHT + 1;         // 36 int64 sums per cell per channel
constexpr int BARD_NF = 4 + BARD_NL;   // cover, spec*NL, peak, peaked, bal
constexpr int BARD_NMAP = 3 + BARD_NL; // dense maps: cover, resp*NL, dark, light
// Slack bytes at both ends of a plane row: the column loop reads x +/- MAXLAG and
// overruns its last vector, so padding buys a bounds-test-free hot loop.
// Plane rows carry only the slack the taps need: MAXLAG on the left, and on the right
// the final vector's tail, which the wide path bounds to total + MAXLAG. This is not
// free space -- the window is 3 * slots * row bytes and wants to stay inside L1, and
// 64 bytes a side alone pushes a 3-channel 512-wide window past 32 KB.
constexpr int BARD_LPAD = 8;
constexpr int BARD_RPAD = 8;
// Rows of the rolling plane window; a power of two >= 2*MAXLAG+1 so the slot for row
// y is y & (BARD_SLOTS-1). Vertical taps therefore wrap, and the kernels take an
// explicit pointer per tap row rather than a row stride.
// Rows of the rolling plane window. The fold now runs per CELL ROW, so every row of
// the cell row in flight plus a MAXLAG halo each side stays resident: chh + 2*MAXLAG,
// set at configure time. The slot for image row y is y % bard_slots_.
constexpr int NF_S = 5;          // tensor-derived float channels
// Model-ready nonlinear descriptors, all DERIVED from the sums above (no new
// accumulators): standardized skew/kurtosis, two structure-tensor ratios, and two
// HOG histogram-shape summaries. Nonlinear (ratios/products/argmax-free peakedness)
// so a linear/shallow model cannot cheaply reconstruct them; dimensionless ones are
// illumination-invariant, which helps few-shot on-the-fly training.
constexpr int NDER = 8;
constexpr int BARDF0 = NF_S + HB + 2 + LBPB + NDER; // 34: start of the bard block
constexpr int NF = BARDF0 + BARD_NF; // 41 derived float32 per cell per channel
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
// Perceptual hashes (imagehash-compatible), one 64-bit hash per channel. Read from a
// dedicated HN x HN grid of per-channel value sums accumulated in the SAME pass (one
// add/pixel/channel, no extra traversal): aHash/wHash threshold an 8x8 mean grid
// (HBLK x HBLK block-pool of it) by its mean/median; pHash takes the low-freq 8x8 of a
// 2D DCT-II of the full 32x32 mean grid. Independent of the feature pyramid grid.
constexpr int HN = 32;        // hash mean-grid resolution (== pHash DCT input side)
constexpr int HS = 8;         // hash side: HS*HS = 64 bits
constexpr int HBLK = HN / HS; // 4: aHash/wHash pool HBLK x HBLK cells of the HN grid
constexpr int64_t RAYSCALE = 1 << 14;
constexpr double PI = 3.14159265358979323846;

// The CHANNEL is the SIMD lane. SIMD_CH is the lane cap of the int32 tag (a
// 128-bit vector holds 4); channels are processed in groups of that many, so every
// scratch array is sized from it. SIMD_PAD (bytes in a 128-bit vector) pads the
// interleaved row buffers so a full-vector LoadU past the last channel of the last
// column stays in-bounds -- the surplus lanes are computed and then discarded.
constexpr int SIMD_CH = 4;
constexpr int SIMD_PAD = 16;

inline int clampi(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

// Rotation-invariant uniform LBP of one pixel. `a`/`b`/`e` are rows r-1/r/r+1 and
// om/o/op the columns left/centre/right; the 8 neighbours are read in circular
// order (NW N NE E SE S SW W), which is what makes the transition count U -- and
// hence the uniformity test -- rotation-invariant. Sign test is p >= centre, so a
// flat neighbourhood is code 255 -> U=0 -> bin 8. Branch-free; the caller's flat
// loop over W*C vectorizes.
inline uint8_t lbp_riu2(const uint8_t *IF_RESTRICT a, const uint8_t *IF_RESTRICT b,
                        const uint8_t *IF_RESTRICT e, int om, int o, int op)
{
  const uint8_t c = b[o];
  const uint8_t s0 = a[om] >= c, s1 = a[o] >= c, s2 = a[op] >= c, s3 = b[op] >= c,
                s4 = e[op] >= c, s5 = e[o] >= c, s6 = e[om] >= c, s7 = b[om] >= c;
  const uint8_t pc = (uint8_t)(s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7);
  const uint8_t u = (uint8_t)((s0 ^ s1) + (s1 ^ s2) + (s2 ^ s3) + (s3 ^ s4) + (s4 ^ s5) +
                              (s5 ^ s6) + (s6 ^ s7) + (s7 ^ s0));
  return u <= 2 ? pc : (uint8_t)(LBPB - 1);
}

// Sobel separables and the vertical min/max envelope, flat over W*C. CT is the channel
// count as a compile-time constant (0 = take it from `cdyn`): with a runtime C the
// +/-C offsets in the gradient loop leave gcc over its alias-versioning budget --
// "number of versioning for alias run-time tests exceeds 10" -- and it silently emits
// scalar code for what is ~40% of a frame. Sources are read-only, so they may alias
// each other at a clamped border row.
template <int CT>
void stencil(int cdyn, const uint8_t *IF_RESTRICT a, const uint8_t *IF_RESTRICT b,
             const uint8_t *IF_RESTRICT e, int WC, int16_t *IF_RESTRICT vs,
             int16_t *IF_RESTRICT vd, uint8_t *IF_RESTRICT cmx, uint8_t *IF_RESTRICT cmn,
             int16_t *IF_RESTRICT gx, int16_t *IF_RESTRICT gy)
{
  const int C = CT ? CT : cdyn;
  for (int o = 0; o < WC; ++o)
  {
    vs[o] = (int16_t)(a[o] + 2 * b[o] + e[o]); // vertical Sobel: smooth, difference
    vd[o] = (int16_t)(e[o] - a[o]);
  }
  for (int o = 0; o < WC; ++o) // uint8-wide, kept apart: one width per loop vectorizes
  {
    cmx[o] = a[o] > e[o] ? a[o] : e[o]; // the column's two off-centre rows; the
    cmn[o] = a[o] < e[o] ? a[o] : e[o]; //   horizontal 3-window below completes it
  }
  for (int o = C; o < WC - C; ++o)
  {
    gx[o] = (int16_t)(vs[o + C] - vs[o - C]);
    gy[o] = (int16_t)(vd[o - C] + 2 * vd[o] + vd[o + C]);
  }
}

// Sweeps one row of `ncell` finest cells (each `cw` px wide) for one group of `n`
// channels, resetting the accumulators at each cell start and flushing them at its
// end. One call per row (not per cell) amortises the vector-descriptor and
// ray-direction setup over the whole row.
void accumulate_row(const int16_t *IF_RESTRICT gxb, const int16_t *IF_RESTRICT gyb,
                    const uint8_t *IF_RESTRICT vb, const uint8_t *IF_RESTRICT mxb,
                    const uint8_t *IF_RESTRICT mnb, const uint8_t *IF_RESTRICT lbb, int strideC,
                    int ch0, int cw, int ncell, int n, int sx, const int32_t *IF_RESTRICT hcx,
                    const int32_t *IF_RESTRICT hcy, int cellstride,
                    int64_t *IF_RESTRICT rowbase)
{
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::CappedTag<int32_t, SIMD_CH> d;
  const hn::Rebind<int16_t, decltype(d)> d16;
  const hn::Rebind<uint8_t, decltype(d)> d8;
  const hn::Repartition<int64_t, decltype(d)> d64;
  const auto zero = hn::Zero(d);
  const int HL = (int)hn::Lanes(d64);

  auto i16_at = [&](const int16_t *p, int col)
  { return hn::PromoteTo(d, hn::LoadU(d16, p + (size_t)col * strideC + ch0)); };
  auto u8_at = [&](const uint8_t *p, int col)
  { return hn::PromoteTo(d, hn::LoadU(d8, p + (size_t)col * strideC + ch0)); };

  hn::VFromD<decltype(d)> hcxv[HB - 1], hcyv[HB - 1];
  for (int j = 0; j < HB - 1; ++j)
  {
    hcxv[j] = hn::Set(d, hcx[j]);
    hcyv[j] = hn::Set(d, hcy[j]);
  }
  int64_t hog[HB * SIMD_CH], lbp[LBPB * SIMD_CH];
  int32_t blane[SIMD_CH], g2lane[SIMD_CH], nmxa[SIMD_CH], nmna[SIMD_CH];
  int64_t se[SIMD_CH], so[SIMD_CH]; // |grad|^4 partial sums, even/odd channels
  int64_t xx[SIMD_CH], yy[SIMD_CH], xy[SIMD_CH], ps[NMOM][SIMD_CH];

  for (int e = 0; e < ncell; ++e)
  {
    auto axl = hn::Zero(d64), axh = hn::Zero(d64), ayl = hn::Zero(d64), ayh = hn::Zero(d64),
         axyl = hn::Zero(d64), axyh = hn::Zero(d64), sg4e = hn::Zero(d64), sg4o = hn::Zero(d64);
    hn::VFromD<decltype(d64)> pl[NMOM], ph[NMOM];
    for (int k = 0; k < NMOM; ++k)
      pl[k] = ph[k] = hn::Zero(d64);
    auto nmaxv = zero, nminv = zero;
    for (int k = 0; k < HB * n; ++k)
      hog[k] = 0;
    for (int k = 0; k < LBPB * n; ++k)
      lbp[k] = 0;
    int64_t colcount = 0;
    const int c1 = e * cw + cw;
    for (int c = e * cw; c < c1; c += sx)
    {
      const auto gx = i16_at(gxb, c), gy = i16_at(gyb, c);
      const auto gx2 = hn::Mul(gx, gx), gy2 = hn::Mul(gy, gy), gxy = hn::Mul(gx, gy);
      axl = hn::Add(axl, hn::PromoteLowerTo(d64, gx2));
      axh = hn::Add(axh, hn::PromoteUpperTo(d64, gx2));
      ayl = hn::Add(ayl, hn::PromoteLowerTo(d64, gy2));
      ayh = hn::Add(ayh, hn::PromoteUpperTo(d64, gy2));
      axyl = hn::Add(axyl, hn::PromoteLowerTo(d64, gxy));
      axyh = hn::Add(axyh, hn::PromoteUpperTo(d64, gxy));
      ++colcount;
      // fold to unsigned orientation, then bin by the integer edge-ray tests
      const auto neg = hn::Or(hn::Lt(gy, zero), hn::And(hn::Eq(gy, zero), hn::Lt(gx, zero)));
      const auto qx = hn::IfThenElse(neg, hn::Neg(gx), gx);
      const auto qy = hn::IfThenElse(neg, hn::Neg(gy), gy);
      auto b = zero;
      for (int j = 0; j < HB - 1; ++j)
      {
        const auto t = hn::Sub(hn::Mul(hcxv[j], qy), hn::Mul(hcyv[j], qx));
        b = hn::Sub(b, hn::VecFromMask(d, hn::Ge(t, zero))); // b += (t >= 0)
      }
      hn::StoreU(b, d, blane);
      const auto g2v = hn::Add(gx2, gy2);
      hn::StoreU(g2v, d, g2lane);
      for (int l = 0; l < n; ++l)
        if (g2lane[l])
          hog[l * HB + blane[l]] += (int64_t)g2lane[l];
      // |grad|^4 = (gx^2+gy^2)^2: widen-square in SIMD (32x32->64, one op each) rather than
      // a scalar multiply per channel. MulEven/MulOdd give the even/odd channels separately
      // (g2 <= 2.08e6 < 2^21, so g2^2 is exact); they are de-interleaved once at flush.
      sg4e = hn::Add(sg4e, hn::MulEven(g2v, g2v));
      sg4o = hn::Add(sg4o, hn::MulOdd(g2v, g2v));
      // power sums: v <= 255 so v*v fits int32, but v^4 (4.2e9) does not -- the
      // cubes and quartics accumulate in int64 lanes.
      const auto v = u8_at(vb, c);
      const auto v2 = hn::Mul(v, v);
      const auto v1l = hn::PromoteLowerTo(d64, v), v1h = hn::PromoteUpperTo(d64, v);
      const auto v2l = hn::PromoteLowerTo(d64, v2), v2h = hn::PromoteUpperTo(d64, v2);
      pl[0] = hn::Add(pl[0], v1l);
      ph[0] = hn::Add(ph[0], v1h);
      pl[1] = hn::Add(pl[1], v2l);
      ph[1] = hn::Add(ph[1], v2h);
      pl[2] = hn::Add(pl[2], hn::Mul(v2l, v1l));
      ph[2] = hn::Add(ph[2], hn::Mul(v2h, v1h));
      pl[3] = hn::Add(pl[3], hn::Mul(v2l, v2l));
      ph[3] = hn::Add(ph[3], hn::Mul(v2h, v2h));
      // strict 8-neighbour extrema, against the envelopes prepare_row built
      nmaxv = hn::Sub(nmaxv, hn::VecFromMask(d, hn::Gt(v, u8_at(mxb, c))));
      nminv = hn::Sub(nminv, hn::VecFromMask(d, hn::Lt(v, u8_at(mnb, c))));
      // LBP: prepare_row already reduced the neighbourhood to a bin index; just vote.
      // Read the bytes directly -- promoting them to a vector only to store them back
      // costs more than the n scalar increments.
      const uint8_t *IF_RESTRICT lp = lbb + (size_t)c * strideC + ch0;
      for (int l = 0; l < n; ++l)
        ++lbp[l * LBPB + lp[l]];
    }
    hn::StoreU(axl, d64, xx);
    hn::StoreU(axh, d64, xx + HL);
    hn::StoreU(ayl, d64, yy);
    hn::StoreU(ayh, d64, yy + HL);
    hn::StoreU(axyl, d64, xy);
    hn::StoreU(axyh, d64, xy + HL);
    for (int k = 0; k < NMOM; ++k)
    {
      hn::StoreU(pl[k], d64, ps[k]);
      hn::StoreU(ph[k], d64, ps[k] + HL);
    }
    hn::StoreU(nmaxv, d, nmxa);
    hn::StoreU(nminv, d, nmna);
    hn::StoreU(sg4e, d64, se); // se = {ch0^4-sum, ch2^4-sum, ...}
    hn::StoreU(sg4o, d64, so); // so = {ch1^4-sum, ch3^4-sum, ...}
    int64_t *IF_RESTRICT acc = rowbase + (size_t)e * cellstride;
    for (int l = 0; l < n; ++l)
    {
      int64_t *IF_RESTRICT a = acc + (size_t)l * NSUM;
      a[SXX] += xx[l];
      a[SYY] += yy[l];
      a[SXY] += xy[l];
      a[CNT] += colcount;
      for (int j = 0; j < HB; ++j)
        a[HOG0 + j] += hog[l * HB + j];
      a[SG4] += (l & 1) ? so[l >> 1] : se[l >> 1];
      a[NMAX] += nmxa[l];
      a[NMIN] += nmna[l];
      for (int k = 0; k < NMOM; ++k)
        a[PS0 + k] += ps[k][l];
      for (int j = 0; j < LBPB; ++j)
        a[LBP0 + j] += lbp[l * LBPB + j];
    }
  }
}

// Cross-channel raw products, one pass over the same already-cached prepared row.
// Channels are the SIMD lane in accumulate_row, so a pairwise product would be a
// cross-lane shuffle; this stays scalar over pairs and vectorizes over columns.
void accumulate_cross_row(const uint8_t *IF_RESTRICT vb, int strideC, int cw, int ncell, int sx,
                          int np, const int *IF_RESTRICT pi, const int *IF_RESTRICT pj,
                          int64_t *IF_RESTRICT rowbase)
{
  int64_t s[XMAX * (XMAX - 1) / 2];
  for (int e = 0; e < ncell; ++e)
  {
    const int c0 = e * cw, c1 = c0 + cw;
    for (int p = 0; p < np; ++p)
      s[p] = 0;
    for (int c = c0; c < c1; c += sx)
    {
      const uint8_t *IF_RESTRICT v = vb + (size_t)c * strideC; // load the pixel once
      for (int p = 0; p < np; ++p)
        s[p] += (int64_t)v[pi[p]] * (int64_t)v[pj[p]];
    }
    int64_t *IF_RESTRICT acc = rowbase + (size_t)e * np;
    for (int p = 0; p < np; ++p)
      acc[p] += s[p];
  }
}

// Sum(v) per channel into the hash row for the current image row. Sampled columns are
// grouped (once, in set_config) into runs that share a hash column and lie in one finest
// cell, so each run is a contiguous column range [beg,end) stepped by sx with a constant
// target cell: the sx==1 case reduces to a plain contiguous accumulation that vectorizes.
// De-interleave ONE image row of every selected channel into the rolling plane
// window, so the detector's +/-d column taps are unit-stride loads and the column
// loop vectorizes at full u8 width. The window holds BARD_SLOTS rows (row y at slot
// y & (BARD_SLOTS-1)), which is the whole point: a band's worth of planes is 1.2 MB
// at 512x512x3 and spills L2, while the nine rows actually in play are ~36 KB and
// stay hot. Every image row is de-interleaved exactly once either way.
// `packed` (plain interleaved, channel k at byte k) takes the wide path; cs == 1
// (already planar) is a memcpy; anything else walks scalar.
// Gather each selected channel's row into a contiguous run: channel k lands at
// out + k*ostride. Every layout imfeat accepts has a wide path here except the
// general strided one, and the phase-split below reuses all of them rather than
// carrying its own -- getting that wrong made a 2-channel or greyscale image fall to
// a scalar byte loop over the whole frame.
void gather_row(const uint8_t *IF_RESTRICT src, int64_t cs, const int64_t *IF_RESTRICT coff,
                int c, int w, bool packed, uint8_t *IF_RESTRICT out, size_t ostride)
{
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint8_t> du;
  const size_t N = hn::Lanes(du);
  if (cs == 1) // already planar (includes 2-D input)
  {
    for (int k = 0; k < c; ++k) std::memcpy(out + (size_t)k * ostride, src + coff[k], (size_t)w);
    return;
  }
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
// so the sampled columns are exactly phase 0 and a lag-d tap is still a contiguous
// load, from phase (d % P) at index +/- d/P.
void deinterleave_row(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs,
                      const int64_t *IF_RESTRICT coff, int c, int w, int y, bool packed,
                      int64_t prs, int slots, int P, int64_t phs, uint8_t *IF_RESTRICT tmp,
                      uint8_t *IF_RESTRICT dst)
{
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint8_t> du;
  const size_t N = hn::Lanes(du);
  const uint8_t *IF_RESTRICT src = img + (int64_t)y * rs;
  uint8_t *IF_RESTRICT d0 = dst + (size_t)(y % slots) * (size_t)prs + BARD_LPAD;
  const size_t pstride = (size_t)slots * (size_t)prs;
  if (P == 1)
  {
    gather_row(src, cs, coff, c, w, packed, d0, pstride);
    return;
  }
  gather_row(src, cs, coff, c, w, packed, tmp, (size_t)w);
  for (int k = 0; k < c; ++k)
  {
    const uint8_t *IF_RESTRICT lin = tmp + (size_t)k * (size_t)w;
    uint8_t *IF_RESTRICT b = d0 + (size_t)k * pstride;
    size_t x = 0;
    if (P == 2)
      for (; x + 2 * N <= (size_t)w; x += 2 * N)
      {
        const auto lo = hn::LoadU(du, lin + x), hi = hn::LoadU(du, lin + x + N);
        hn::StoreU(hn::ConcatEven(du, hi, lo), du, b + x / 2);
        hn::StoreU(hn::ConcatOdd(du, hi, lo), du, b + phs + x / 2);
      }
    for (; x < (size_t)w; ++x) b[(int64_t)(x % P) * phs + x / P] = lin[x];
  }
}

// One pixel's gated maps, used only for the <= MAXLAG columns at each row end where
// the vectorized body would read across the image border. This is the definition the
// dense path must reproduce: a lag contributes on an axis only where BOTH its taps are
// inside the image, and `lagmask` drops a lag on BOTH axes when its span exceeds the
// smaller image side (dropping per-axis would differ on very thin images).
inline void bard_pixel_vals(const uint8_t *IF_RESTRICT prow, const uint8_t *const *IF_RESTRICT up,
                            const uint8_t *const *IF_RESTRICT dn, int H, int W, int r, int x,
                            int lagmask, int32_t *IF_RESTRICT v)
{
  const int centre = prow[x];
  int dmax = 0, lmax = 0, resp[BARD_NL];
  for (int j = 0; j < BARD_NL; ++j)
  {
    const int d = BARD_LAGS[j];
    int dd = 0, ll = 0;
    if ((lagmask >> j) & 1)
    {
      if (x >= d && x < W - d)
      {
        const int l = prow[x - d], rr = prow[x + d];
        dd = std::min(l - centre, rr - centre);
        ll = std::min(centre - l, centre - rr);
      }
      if (r >= d && r < H - d)
      {
        const int tu = up[j][x], tv = dn[j][x];
        dd = std::max(dd, std::min(tu - centre, tv - centre));
        ll = std::max(ll, std::min(centre - tu, centre - tv));
      }
    }
    dd = dd > 0 ? dd : 0;
    ll = ll > 0 ? ll : 0;
    resp[j] = std::max(dd, ll);
    dmax = std::max(dmax, dd);
    lmax = std::max(lmax, ll);
  }
  const bool keep = std::max(dmax, lmax) >= BARD_TAU;
  v[0] = keep ? 1 : 0;
  for (int j = 0; j < BARD_NL; ++j) v[1 + j] = keep ? resp[j] : 0;
  v[1 + BARD_NL] = keep ? dmax : 0;
  v[2 + BARD_NL] = keep ? lmax : 0;
}

// Map-writing wrapper, for the fold path that materializes rows.
inline void bard_pixel(const uint8_t *IF_RESTRICT prow, const uint8_t *const *IF_RESTRICT up,
                       const uint8_t *const *IF_RESTRICT dn, int H, int W, int r, int x,
                       int lagmask, const uint8_t *IF_RESTRICT smask, uint8_t *IF_RESTRICT out,
                       size_t os)
{
  int32_t v[BARD_NMAP];
  bard_pixel_vals(prow, up, dn, H, W, r, x, lagmask, v);
  const bool on = smask[x] != 0;
  for (int t = 0; t < BARD_NMAP; ++t) out[(size_t)t * os + x] = on ? (uint8_t)v[t] : 0;
}

// One row's BARD_NMAP dense maps, already gated so the fold below is branchless.
// COLUMNS ARE THE SIMD LANE -- at 64 uint8 lanes this is the widest loop in the
// library, against accumulate_row's 4 int32 channel lanes, which is what lets a
// second stencil ride along for near-free. Every tap is a saturating subtract, so
// dark/light need no sign handling and nothing widens past uint8:
//   min(max(l-c, 0), max(r-c, 0)) == min(SatSub(l, c), SatSub(r, c)).
// peak == max_j max(dk_j, lt_j) == max(dmax, lmax), so the gate costs one compare.
// Vertical-tap validity is a property of the ROW, hoisted out of the loop; horizontal
// validity holds for every interior column by construction, and the row ends are
// overwritten afterwards by bard_pixel.
void bard_row_dense(const uint8_t *IF_RESTRICT prow, const uint8_t *const *IF_RESTRICT up,
                    const uint8_t *const *IF_RESTRICT dn, int lagmask, int total,
                    uint8_t *IF_RESTRICT out, size_t os, const bool *IF_RESTRICT vok,
                    const uint8_t *IF_RESTRICT smask)
{
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint8_t> du;
  using V = decltype(hn::Zero(du));
  const size_t N = hn::Lanes(du);
  const auto tau = hn::Set(du, (uint8_t)BARD_TAU);
  const auto one = hn::Set(du, (uint8_t)1);
  for (int x = 0; x < total; x += (int)N)
  {
    const V c = hn::LoadU(du, prow + x);
    V dmax = hn::Zero(du), lmax = hn::Zero(du), resp[BARD_NL];
    for (int j = 0; j < BARD_NL; ++j)
    {
      const int d = BARD_LAGS[j];
      V dk = hn::Zero(du), lt = hn::Zero(du);
      if ((lagmask >> j) & 1)
      {
        const V l = hn::LoadU(du, prow + x - d), rr = hn::LoadU(du, prow + x + d);
        dk = hn::Min(hn::SaturatedSub(l, c), hn::SaturatedSub(rr, c));
        lt = hn::Min(hn::SaturatedSub(c, l), hn::SaturatedSub(c, rr));
      }
      if (vok[j])
      {
        const V u = hn::LoadU(du, up[j] + x);
        const V v = hn::LoadU(du, dn[j] + x);
        dk = hn::Max(dk, hn::Min(hn::SaturatedSub(u, c), hn::SaturatedSub(v, c)));
        lt = hn::Max(lt, hn::Min(hn::SaturatedSub(c, u), hn::SaturatedSub(c, v)));
      }
      resp[j] = hn::Max(dk, lt);
      dmax = hn::Max(dmax, dk);
      lmax = hn::Max(lmax, lt);
    }
    // AND the sampling mesh into the gate: off-mesh columns come out zero, so the
    // fold below sums a contiguous run per cell instead of a strided one, and the
    // mesh costs one load and one And per vector rather than a stride in the fold.
    const auto keep = hn::And(hn::Ge(hn::Max(dmax, lmax), tau),
                              hn::Gt(hn::LoadU(du, smask + x), hn::Zero(du)));
    hn::StoreU(hn::IfThenElseZero(keep, one), du, out + x);
    for (int j = 0; j < BARD_NL; ++j)
      hn::StoreU(hn::IfThenElseZero(keep, resp[j]), du, out + (size_t)(1 + j) * os + x);
    hn::StoreU(hn::IfThenElseZero(keep, dmax), du, out + (size_t)(1 + BARD_NL) * os + x);
    hn::StoreU(hn::IfThenElseZero(keep, lmax), du, out + (size_t)(2 + BARD_NL) * os + x);
  }
}

// The fused path, run once per CELL ROW rather than once per row.
//
// Every sampled row of a cell row folds into the SAME cells, so the group sums are
// carried in registers across all of them and memory is touched once per column
// block instead of once per row -- at stride 2 and an 8-row cell that is a quarter
// of the read-modify-writes. The dense maps are never materialized either: storing
// and reloading BARD_NMAP bytes per pixel per channel cost more than the arithmetic,
// which measurement showed to be nearly free (three lags cost the same as one).
//
// SumsOf8 (psadbw) horizontally sums each 8-byte group in one instruction; with cw a
// multiple of 8 a group never straddles a cell, so groups reduce to cells at the end
// with a shift. `smask` is the sampling mesh with the <= MAXLAG columns at both row
// ends zeroed -- those would read across the image border, so the vector body scores
// them zero and the caller adds their true contribution with bard_pixel_vals.
void bard_cellrow_fused(const uint8_t *IF_RESTRICT pbase, int64_t prs, int slots,
                        const int *IF_RESTRICT rws, int nrows, int H, int lagmask, int total,
                        int cw, int gshift, int ncell, uint64_t *IF_RESTRICT cells,
                        const uint8_t *IF_RESTRICT smask, const uint8_t *IF_RESTRICT valid,
                        size_t vstride, int P, int64_t phs)
{
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint8_t> du;
  const hn::Repartition<uint64_t, decltype(du)> d64;
  using V = decltype(hn::Zero(du));
  using V64 = decltype(hn::Zero(d64));
  const size_t N = hn::Lanes(du), G = N / 8;
  const auto tau = hn::Set(du, (uint8_t)BARD_TAU);
  const auto one = hn::Set(du, (uint8_t)1);
  for (int x = 0, cell0 = 0; x < total; x += (int)N, cell0 += (int)N / cw)
  {
    V64 acc[BARD_NMAP];
    for (int t = 0; t < BARD_NMAP; ++t) acc[t] = hn::Zero(d64);
    const auto mesh = hn::Gt(hn::LoadU(du, smask + x), hn::Zero(du));
    // A column's taps can only fall outside the image within MAXLAG of either end, so
    // the validity masks are loaded for the two edge blocks and skipped everywhere else.
    const bool edge = x < BARD_MAXLAG || x + (int)N > total - BARD_MAXLAG;
    for (int q = 0; q < nrows; ++q)
    {
      const int r = rws[q];
      const uint8_t *IF_RESTRICT prow = pbase + (size_t)(r % slots) * prs;
      const V c = hn::LoadU(du, prow + x);
      V dmax = hn::Zero(du), lmax = hn::Zero(du), resp[BARD_NL];
      for (int j = 0; j < BARD_NL; ++j)
      {
        const int d = BARD_LAGS[j];
        V dk = hn::Zero(du), lt = hn::Zero(du);
        if ((lagmask >> j) & 1)
        {
          // min(max(l-c,0), max(r-c,0)) == max(min(l,r) - c, 0) == SatSub(min(l,r), c),
          // and the light side is SatSub(c, max(l,r)). Combining the two axes with max
          // distributes through the same way, so taking min/max of each tap PAIR first
          // and doing the saturating subtract once per polarity turns 14 ops a lag
          // into 8 -- and lag arithmetic is over half this kernel.
          // Columns whose +/-d taps fall outside the image must contribute nothing on
          // this axis -- but the vertical axis may still contribute, so the mask has to
          // apply BEFORE the two axes combine. Forcing the dark side's base to 0 and the
          // light side's to 255 does exactly that under saturating arithmetic: SatSub(0,
          // c) and SatSub(c, 255) are both zero for every uint8 c. That removes the
          // scalar per-pixel border strips, which measured at 39% of this kernel.
          // column (x*P) +/- d lives in phase (d % P) at index x +/- d/P; with P == 1
          // this is the plain prow + x -/+ d.
          const int qd = d / P, rd = d % P;
          const V l = hn::LoadU(du, prow + (rd ? (int64_t)(P - rd) * phs : 0) - (rd ? qd + 1 : qd)
                                    + x);
          const V rr = hn::LoadU(du, prow + (int64_t)rd * phs + qd + x);
          V lo = hn::Min(l, rr), hi = hn::Max(l, rr);
          if (edge) // only the first and last block of a row can hold invalid columns
          {
            const V vm = hn::LoadU(du, valid + (size_t)j * vstride + x);
            lo = hn::And(lo, vm);
            hi = hn::Or(hi, hn::Not(vm));
          }
          if (r >= d && r < H - d)
          {
            const V u = hn::LoadU(du, pbase + (size_t)((r - d) % slots) * prs + x);
            const V v = hn::LoadU(du, pbase + (size_t)((r + d) % slots) * prs + x);
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
      // peak == max_j max(dk_j, lt_j) == max(dmax, lmax), so the gate is one compare
      const auto keep = hn::And(hn::Ge(hn::Max(dmax, lmax), tau), mesh);
      auto add = [&](int t, V val) {
        acc[t] = hn::Add(acc[t], hn::SumsOf8(hn::IfThenElseZero(keep, val)));
      };
      add(0, one);
      for (int j = 0; j < BARD_NL; ++j) add(1 + j, resp[j]);
      add(1 + BARD_NL, dmax);
      add(2 + BARD_NL, lmax);
    }
    for (int t = 0; t < BARD_NMAP; ++t) // groups -> cells, once per block per cell row
    {
      uint64_t *IF_RESTRICT a = cells + (size_t)t * ncell + cell0;
      if (gshift == 0)
        hn::StoreU(hn::Add(hn::LoadU(d64, a), acc[t]), d64, a);
      else if (gshift == 1)
      {
        const hn::Half<decltype(d64)> dh;
        const auto pr = hn::Add(hn::ConcatEven(d64, acc[t], acc[t]),
                                hn::ConcatOdd(d64, acc[t], acc[t]));
        hn::StoreU(hn::Add(hn::LoadU(dh, a), hn::LowerHalf(dh, pr)), dh, a);
      }
      else
      {
        uint64_t g[64];
        hn::StoreU(acc[t], d64, g);
        for (size_t i = 0; i < G; ++i) a[i >> gshift] += g[i];
      }
    }
  }
}

// Fold one row's gated maps into that row's cell sums, on the sampling mesh. Gated-out
// pixels contribute zero, so there is no branch; the maps are u8 and a cell row holds
// at most a few hundred of them, so the partials stay in int32 until the cell closes.
void bard_row_reduce(const uint8_t *IF_RESTRICT out, size_t os, int cw, int ncell, int ncellbuf,
                     uint64_t *IF_RESTRICT cells, bool wide)
{
  namespace hn = hwy::HWY_NAMESPACE;
  for (int t = 0; t < BARD_NMAP; ++t)
  {
    const uint8_t *IF_RESTRICT m = out + (size_t)t * os;
    uint64_t *IF_RESTRICT a = cells + (size_t)t * ncell;
    if (wide)
    {
      // SumsOf8 (psadbw) horizontally sums each 8-byte group into one u64, and with
      // cw a multiple of 8 a group never straddles a cell, so one instruction does
      // what was eight dependent byte adds. Off-mesh and gated-out columns are
      // already zero, so no masking is needed here.
      const hn::ScalableTag<uint8_t> du;
      const hn::Repartition<uint64_t, decltype(du)> d64;
      const size_t N = hn::Lanes(du), G = N / 8;
      uint64_t g[64];
      for (int x = 0; x < ncell * cw; x += (int)N)
      {
        hn::StoreU(hn::SumsOf8(hn::LoadU(du, m + x)), d64, g);
        for (size_t i = 0; i < G; ++i)
          a[(size_t)((x + 8 * (int)i) / cw)] += g[i];
      }
    }
    else
      for (int e = 0; e < ncell; ++e)
      {
        // off-mesh columns are already zero, so this is a flat contiguous sum the
        // compiler vectorizes; u8 inputs and a cell row of a few hundred cannot
        // overflow int32.
        int32_t v = 0;
        for (int x = e * cw, xe = e * cw + cw; x < xe; ++x) v += m[x];
        a[(size_t)e] += (uint64_t)v;
      }
  }
}

void accumulate_hash_row(const uint8_t *IF_RESTRICT vb, int C, int sx,
                         const int *IF_RESTRICT beg, const int *IF_RESTRICT end,
                         const int *IF_RESTRICT rhc, int nrun, int64_t *IF_RESTRICT hrowbase)
{
  for (int r = 0; r < nrun; ++r)
  {
    int64_t *IF_RESTRICT cell = hrowbase + (size_t)rhc[r] * C;
    const int b = beg[r], e = end[r];
    for (int k = 0; k < C; ++k)
    {
      int64_t s = 0;
      for (int c = b; c < e; c += sx)
        s += vb[(size_t)c * C + k];
      cell[k] += s;
    }
  }
}

struct Level
{
  int ny = 0, nx = 0, fy = 0, fx = 0;
  std::vector<int64_t> buf, xbuf;
  std::vector<float> feat, xfeat;
  std::vector<double> mom;
  std::vector<float> fsum;  // (c, NF, NST)  cross-cell summary of feat
  std::vector<double> msum; // (c, NMOM, NST) cross-cell summary of mom
  std::vector<size_t> rshape, fshape, mshape, xrshape, xfshape, fsumshape, msumshape;
};

class FeatureComputer
{
  int h_ = 0, w_ = 0, c_ = 1, sy_ = 1, sx_ = 1;
  bool interleaved_ = false; // source rows are already the (H,W,C) layout we want
  std::vector<int> chan_;
  std::vector<int64_t> coff_; // per-selected-channel offset within a pixel
  std::vector<Level> levels_;
  std::vector<int> row_cell_;
  int32_t hcx_[HB - 1] = {}, hcy_[HB - 1] = {};
  int hog_card_[HB] = {};    // 1 for the axis-aligned (cardinal) orientation bins
  int bard_chunk_rows_ = 4; // sampled rows per kernel call; bounds the plane window
  std::vector<uint8_t> bvalid_; // per lag: 255 where both +/-d column taps are in range
  size_t bvalid_stride_ = 0;
  int bard_p_ = 1;           // column phases the planes are split into (1 = not split)
  int bard_slots_ = 9;       // rolling plane window rows (see the window comment)
  int bard_lagmask_ = 0;     // bit j set iff bard lag j fits the image (see bard_pixel)
  int np_ = 0;               // number of channel pairs (0 if C<2 or C>XMAX)
  std::vector<int> pi_, pj_; // the pairs, in (i<j) lexicographic order

  // Per-row scratch, all interleaved (c * c_ + k) and padded by SIMD_PAD. One row
  // is a few KB, so it stays hot in L1 across the whole cell sweep. prepare_row
  // overwrites all of it every row, so each band needs its own copy.
  struct Scratch
  {
    std::vector<uint8_t> r0, r1, r2, cmx, cmn, mx, mn, lb;
    std::vector<int16_t> vs, vd, gx, gy;
    std::vector<int64_t> hsum;   // private hash partial; bands >0 only, and only if !hfast_
    std::vector<uint8_t> bp;     // padded de-interleaved planes: rolling BARD_SLOTS-row window
    std::vector<uint8_t> bq;     // the row in flight: BARD_NMAP gated maps
    std::vector<uint8_t> smask;  // 255 on the sampling mesh, 0 off it
    std::vector<uint8_t> lin;    // one row, channels contiguous, for the phase split
    std::vector<uint8_t> smaskv; // smask with both border strips zeroed, for the fused path
    std::vector<uint64_t> bcell; // [c][BARD_NMAP][ncell] sums for the cell row in flight
    int64_t bp_rs = 0;           // padded plane row stride
    size_t bqs = 0;              // stride between maps in bq
    int64_t bp_phase = 0;        // stride between a row's phase planes
    int bp_filled = 0;           // rows [.., bp_filled) of the window are loaded
    const uint8_t *vrow = nullptr; // centre row prepare_row settled on (image or scratch)
  };
  std::vector<Scratch> scr_;

  // A band is a contiguous run of finest cell rows. Two bands therefore never touch
  // the same accumulator, so the hot path needs no atomics and no locks -- and since
  // every accumulator is an int64 sum that cannot overflow, the totals are identical
  // to the serial order however the rows are split. Only the coarse-level rollup,
  // derive() and the hash readout run after the join, all still serial.
  struct Band
  {
    int r0, r1;   // image rows [r0, r1); the visited ones are those on the sy_ grid
    int cy0, cy1; // the finest cell rows they land in, [cy0, cy1)
  };
  enum Job
  {
    JOB_ACC,
    JOB_DERIVE
  };
  Job job_ = JOB_ACC;
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
  std::vector<int> hrow_, hcol_;          // row -> hash row (h_), col -> hash col (w_)
  std::vector<int> hrbeg_, hrend_, hrhc_; // per-run column range [beg,end) and its hash col
  std::vector<double> dctb_;              // HS*HN : DCT-II basis rows (scipy type-2, norm=None)
  std::vector<uint64_t> ahash_, whash_, phash_; // c_ each, one 64-bit hash per channel
  std::vector<size_t> hshape_;                  // {c_}
  bool hfast_ = false;                          // finest grid tiles HN -> hashes reuse its S1

  std::vector<int64_t> graw_, gxraw_;
  std::vector<float> gfeat_, gxfeat_;
  std::vector<double> gmom_;
  // reusable cross-cell summary accumulators (sized c_*NF / c_*NMOM), refilled per level
  std::vector<float> sfmn_, sfmx_;
  std::vector<double> sfsm_, sfsq_, smmn_, smmx_, smsm_, smsq_;
  std::vector<size_t> graw_shape_, gfeat_shape_, gmom_shape_, gxraw_shape_, gxfeat_shape_;

  // Structure-tensor + HOG + extrema features (float32).
  void derive_cell(const int64_t *IF_RESTRICT s, const double *IF_RESTRICT mom,
                   float *IF_RESTRICT f) const
  {
    const double sxx = (double)s[SXX], syy = (double)s[SYY], sxy = (double)s[SXY];
    const double n = (double)s[CNT];
    const double tr = sxx + syy, d = sxx - syy;
    const double R = std::sqrt(d * d + 4.0 * sxy * sxy);
    const double invn = n > 0.0 ? 1.0 / n : 0.0;
    const double invt = tr > 0.0 ? 1.0 / tr : 0.0;
    f[0] = (float)(tr * invn);
    f[1] = (float)(R * invt);
    f[2] = (float)(d * invt);
    f[3] = (float)(2.0 * sxy * invt);
    f[4] = (float)((tr - R) * 0.5 * invn);
    int64_t hs = 0;
    for (int b = 0; b < HB; ++b)
      hs += s[HOG0 + b];
    const double invh = hs > 0 ? 1.0 / (double)hs : 0.0;
    for (int b = 0; b < HB; ++b)
      f[NF_S + b] = (float)((double)s[HOG0 + b] * invh);
    f[NF_S + HB + 0] = (float)((double)s[NMAX] * invn);
    f[NF_S + HB + 1] = (float)((double)s[NMIN] * invn);
    // the LBP bins partition the cell's pixels, so their sum is exactly the count
    for (int b = 0; b < LBPB; ++b)
      f[NF_S + HB + 2 + b] = (float)((double)s[LBP0 + b] * invn);

    // --- derived nonlinear descriptors (see NDER) ---
    float *IF_RESTRICT g = f + NF_S + HB + 2 + LBPB;
    const double var = mom[1], m3 = mom[2], m4 = mom[3];
    const double sd = std::sqrt(var > 0.0 ? var : 0.0),
                 inv2 = var > 0.0 ? 1.0 / (var * var) : 0.0;
    // A. standardized moments -- dimensionless, so invariant to any v -> a*v+b (a>0).
    // std_skew is framegate's own text/bimodality cue (|m3|/var^1.5) as a first-class feature.
    g[0] = (float)(var > 0.0 ? m3 * inv2 * sd : 0.0);  // std_skew = m3 / var^1.5
    g[1] = (float)(var > 0.0 ? m4 * inv2 - 3.0 : 0.0); // excess kurtosis = m4/var^2 - 3
    // B. structure-tensor ratios (energy = f[0], coherence = f[1]).
    g[2] = (float)((double)f[0] / (var + 1.0));          // edge_sharpness = energy/(var+eps)
    g[3] = (float)((double)f[0] * (1.0 - (double)f[1])); // detail = energy*(1-coherence)
    // C. HOG histogram shape (log-free). Concentration (Herfindahl) is high for a single
    // orientation (barcodes), low when spread (QR, foliage). Cardinality is the axis-aligned
    // energy fraction (text, tables). Both need the multimodality the tensor averages away.
    double sq = 0.0, card = 0.0;
    for (int b = 0; b < HB; ++b)
    {
      const double hb = (double)s[HOG0 + b];
      sq += hb * hb;
      card += hog_card_[b] ? hb : 0.0;
    }
    g[4] = (float)(hs > 0 ? sq * invh * invh : 0.0); // concentration = sum p_i^2
    g[5] = (float)(hs > 0 ? card * invh : 0.0);      // cardinality = cardinal energy fraction
    // D. gradient sparsity: normalized 4th moment of gradient magnitude E[|g|^4]/E[|g|^2]^2,
    // = 1 + CV^2 of per-pixel gradient energy (>=1 by Cauchy-Schwarz, dimensionless). This is
    // the non-central moment ratio, not the central kurtosis (which needs a per-pixel sqrt).
    // Near 1 for a uniform gradient field; large where a few strong edges dominate a mostly
    // flat cell (text strokes, line art, glyph/infographic borders).
    const double ge2 = (double)s[SXX] + (double)s[SYY]; // sum |grad|^2
    g[6] = (float)(ge2 > 0.0 ? n * (double)s[SG4] / (ge2 * ge2) : 0.0);
    // E. RMS contrast = coefficient of variation sd/mean (Peli 1990): invariant to intensity
    // gain, relative to brightness (dark textured cells read as high contrast). The +1 grey
    // level floors the denominator on near-black cells.
    g[7] = (float)(sd / (mom[0] + 1.0));
    // --- bar detector (see BARD_LAGS) ---
    // Every term but `cover` is a ratio of gated sums, so the sampled-pixel count
    // cancels and the block is invariant to stride and cell size.
    float *IF_RESTRICT q = f + BARDF0;
    q[0] = (float)((double)s[BARD0] * invn);
    double btot = 0.0;
    for (int j = 0; j < BARD_NL; ++j) btot += (double)s[BARD_S0 + j];
    const double binv = btot > 1e-9 ? 1.0 / btot : 0.0;
    double bpk = 0.0;
    for (int j = 0; j < BARD_NL; ++j)
    {
      const double m = (double)s[BARD_S0 + j];
      q[1 + j] = (float)(m * binv); // spectrum: share of response mass at this lag
      bpk += m * (double)(j + 1);
    }
    q[1 + BARD_NL] = (float)(bpk * binv); // mass-weighted mean lag index, in [1, NL]
    const double bmean = btot / (double)BARD_NL;
    double bvar = 0.0;
    for (int j = 0; j < BARD_NL; ++j)
    {
      const double e = (double)s[BARD_S0 + j] - bmean;
      bvar += e * e;
    }
    bvar /= (double)BARD_NL;
    // CV of the lag profile: high when one stroke width dominates, low when the
    // response spreads evenly over every lag (texture).
    q[2 + BARD_NL] = (float)(bmean > 1e-9 ? std::sqrt(bvar < 0.0 ? 0.0 : bvar) / bmean : 0.0);
    const double bdk = (double)s[BARD_DARK], blt = (double)s[BARD_LIGHT];
    // Polarity in [-1, 1]: +1 all light-on-dark, -1 all dark-on-light.
    q[3 + BARD_NL] = (float)(bdk + blt > 1e-9 ? (blt - bdk) / (bdk + blt) : 0.0);
  }

  // Cross-channel covariance and Pearson correlation per channel pair, from the raw
  // products and the cell's already-derived per-channel [mean, var].
  void derive_cross(const double *IF_RESTRICT m, int64_t n, const int64_t *IF_RESTRICT x,
                    float *IF_RESTRICT f) const
  {
    const double invn = n > 0 ? 1.0 / (double)n : 0.0;
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
  void derive_moments(const int64_t *IF_RESTRICT s, double *IF_RESTRICT m) const
  {
    const int64_t n = s[CNT];
    if (n <= 0)
    {
      m[0] = m[1] = m[2] = m[3] = 0.0;
      return;
    }
    const int64_t s1 = s[PS0], s2 = s[PS0 + 1], s3 = s[PS0 + 2], s4 = s[PS0 + 3];
    const int64_t k = (s1 + n / 2) / n, k2 = k * k, k3 = k2 * k;
    const int64_t t1 = s1 - n * k;
    const int64_t t2 = s2 - 2 * k * s1 + n * k2;
    const int64_t t3 = s3 - 3 * k * s2 + 3 * k2 * s1 - n * k3;
    const int64_t t4 = s4 - 4 * k * s3 + 6 * k2 * s2 - 4 * k3 * s1 + n * k3 * k;
    const double invn = 1.0 / (double)n;
    const double d = (double)t1 * invn, d2 = d * d;
    const double u2 = (double)t2 * invn, u3 = (double)t3 * invn, u4 = (double)t4 * invn;
    m[0] = (double)k + d;
    m[1] = u2 - d2;
    m[2] = u3 - 3.0 * d * u2 + 2.0 * d2 * d;
    m[3] = u4 - 4.0 * d * u3 + 6.0 * d2 * u2 - 3.0 * d2 * d2;
  }

  // Gather rows r-1, r, r+1 into the interleaved scratch, then compute -- for every
  // pixel of the row, flat over W*C and branch-free, so the compiler vectorizes it
  // over columns -- the Sobel gradients and the 8-neighbour max/min envelopes. All
  // border clamping is confined here (two fix-up columns), so the SIMD cell loop
  // that follows is a straight walk with no bounds logic.
  void prepare_row(Scratch &s, const uint8_t *IF_RESTRICT img, int r, int64_t rs, int64_t cs)
  {
    const int hi_r = h_ - 1, WC = w_ * c_;
    const uint8_t *rows[3] = {img + (int64_t)clampi(r - 1, hi_r) * rs, img + (int64_t)r * rs,
                              img + (int64_t)clampi(r + 1, hi_r) * rs};
    const uint8_t *IF_RESTRICT a, *IF_RESTRICT b, *IF_RESTRICT e;
    if (interleaved_)
    {
      // The source already has the layout we want, so read it in place: the copy was
      // pure traffic, and with threading it was three copies per row per band. Only
      // the centre row reaches accumulate_row, whose 4-lane LoadU may overrun a row by
      // up to SIMD_CH-1 bytes -- harmless mid-image, so copy just the final row.
      a = rows[0];
      e = rows[2];
      b = rows[1];
      if (r == hi_r)
      {
        std::memcpy(s.r1.data(), rows[1], (size_t)WC);
        b = s.r1.data();
      }
    }
    else
    {
      uint8_t *dst[3] = {s.r0.data(), s.r1.data(), s.r2.data()};
      for (int i = 0; i < 3; ++i)
        for (int c = 0; c < w_; ++c)
          for (int k = 0; k < c_; ++k)
            dst[i][(size_t)c * c_ + k] = rows[i][(int64_t)c * cs + coff_[k]];
      a = s.r0.data();
      b = s.r1.data();
      e = s.r2.data();
    }
    s.vrow = b;
    // Alias every scratch buffer through a restrict pointer: via the std::vector
    // members the compiler cannot prove the reads and writes are disjoint, and it
    // silently drops the flat loops below back to scalar.
    // Alias every scratch buffer through a restrict pointer: reached via the vectors
    // the compiler cannot prove the reads and writes are disjoint, and it silently
    // drops the flat loops below back to scalar.
    uint8_t *IF_RESTRICT cmx = s.cmx.data(), *IF_RESTRICT cmn = s.cmn.data();
    uint8_t *IF_RESTRICT mx = s.mx.data(), *IF_RESTRICT mn = s.mn.data(),
                         *IF_RESTRICT lb = s.lb.data();
    int16_t *IF_RESTRICT vs = s.vs.data(), *IF_RESTRICT vd = s.vd.data();
    int16_t *IF_RESTRICT gx = s.gx.data(), *IF_RESTRICT gy = s.gy.data();
    const int C = c_;
    switch (C) // specialise the common channel counts; see stencil()
    {
    case 1:
      stencil<1>(C, a, b, e, WC, vs, vd, cmx, cmn, gx, gy);
      break;
    case 3:
      stencil<3>(C, a, b, e, WC, vs, vd, cmx, cmn, gx, gy);
      break;
    case 4:
      stencil<4>(C, a, b, e, WC, vs, vd, cmx, cmn, gx, gy);
      break;
    default:
      stencil<0>(C, a, b, e, WC, vs, vd, cmx, cmn, gx, gy);
      break;
    }
    // 8-neighbour envelopes. The centre is excluded from its own: the middle row
    // contributes only b[o-C] and b[o+C] -- which collapse onto the centre at a border
    // column, exactly as a replicate-padded 3x3 does, so a border pixel can never be a
    // strict extremum.
    // Envelopes and LBP codes for the interior, explicitly vectorized over columns.
    // gcc will not auto-vectorize this one: eight base pointers with runtime +/-C
    // offsets blow past its alias-versioning budget even with __restrict__, and it
    // silently emits scalar code. Highway gives 16-32 uint8 lanes and is portable.
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<uint8_t> du;
    const int NL = (int)hn::Lanes(du);
    const auto one = hn::Set(du, 1), nonuni = hn::Set(du, LBPB - 1), two = hn::Set(du, 2);
    auto bit = [&](const uint8_t *p, int o, hn::VFromD<decltype(du)> c) { // p[o] >= c
      return hn::And(hn::VecFromMask(du, hn::Ge(hn::LoadU(du, p + o), c)), one);
    };
    int i = C;
    for (; i + NL <= WC - C; i += NL)
    {
      const auto cm = hn::LoadU(du, cmx + i), cn = hn::LoadU(du, cmn + i);
      const auto bm = hn::LoadU(du, b + i - C), bp = hn::LoadU(du, b + i + C);
      hn::StoreU(hn::Max(hn::Max(hn::LoadU(du, cmx + i - C), hn::LoadU(du, cmx + i + C)),
                         hn::Max(cm, hn::Max(bm, bp))),
                 du, mx + i);
      hn::StoreU(hn::Min(hn::Min(hn::LoadU(du, cmn + i - C), hn::LoadU(du, cmn + i + C)),
                         hn::Min(cn, hn::Min(bm, bp))),
                 du, mn + i);
      const auto c = hn::LoadU(du, b + i); // LBP: 8 neighbours in circular order
      const auto s0 = bit(a, i - C, c), s1 = bit(a, i, c), s2 = bit(a, i + C, c),
                 s3 = bit(b, i + C, c), s4 = bit(e, i + C, c), s5 = bit(e, i, c),
                 s6 = bit(e, i - C, c), s7 = bit(b, i - C, c);
      const auto pc = hn::Add(hn::Add(hn::Add(s0, s1), hn::Add(s2, s3)),
                              hn::Add(hn::Add(s4, s5), hn::Add(s6, s7)));
      const auto u = hn::Add(hn::Add(hn::Add(hn::Xor(s0, s1), hn::Xor(s1, s2)),
                                     hn::Add(hn::Xor(s2, s3), hn::Xor(s3, s4))),
                             hn::Add(hn::Add(hn::Xor(s4, s5), hn::Xor(s5, s6)),
                                     hn::Add(hn::Xor(s6, s7), hn::Xor(s7, s0))));
      hn::StoreU(hn::IfThenElse(hn::Le(u, two), pc, nonuni), du, lb + i);
    }
    for (; i < WC - C; ++i)
    {
      mx[i] = std::max({cmx[i - C], cmx[i], cmx[i + C], b[i - C], b[i + C]});
      mn[i] = std::min({cmn[i - C], cmn[i], cmn[i + C], b[i - C], b[i + C]});
      lb[i] = lbp_riu2(a, b, e, i - C, i, i + C);
    }
    const int last = (w_ - 1) * C, step = w_ > 1 ? C : 0;
    for (int k = 0; k < C; ++k) // the two border columns, clamped
      for (int o : {k, last + k})
      {
        const int om = o == k ? k : last - step + k, op = o == k ? k + step : last + k;
        gx[o] = (int16_t)(vs[op] - vs[om]);
        gy[o] = (int16_t)(vd[om] + 2 * vd[o] + vd[op]);
        mx[o] = std::max({cmx[om], cmx[o], cmx[op], b[om], b[op]});
        mn[o] = std::min({cmn[om], cmn[o], cmn[op], b[om], b[op]});
        lb[o] = lbp_riu2(a, b, e, om, o, op);
      }
  }

  // One band's rows. Reads only its own Scratch and writes only its own cell rows.
  void accumulate_band(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int bi)
  {
    Level &fine = levels_[0];
    const int nfx = fine.nx, cw = w_ / nfx;
    int64_t *IF_RESTRICT fb = fine.buf.data();
    Scratch &s = scr_[bi];
    // Clear this band's own slice rather than the whole buffer up front: it takes the
    // zeroing off the serial path and warms the lines the band is about to write.
    const size_t cs0 = (size_t)bands_[bi].cy0 * nfx, cs1 = (size_t)bands_[bi].cy1 * nfx;
    std::fill(fine.buf.begin() + cs0 * c_ * NSUM, fine.buf.begin() + cs1 * c_ * NSUM, (int64_t)0);
    if (np_)
      std::fill(fine.xbuf.begin() + cs0 * np_, fine.xbuf.begin() + cs1 * np_, (int64_t)0);
    int64_t *IF_RESTRICT hs = hfast_ ? nullptr : (bi ? s.hsum.data() : hsum_.data());
    namespace hn = hwy::HWY_NAMESPACE;
    const int L = (int)hn::Lanes(hn::CappedTag<int32_t, SIMD_CH>());
    // Prime the rolling window with the rows the first processed row's taps need.
    int by = std::max(0, bands_[bi].r0 - BARD_MAXLAG);
    for (int y = by; y <= std::min(h_ - 1, bands_[bi].r0 + BARD_MAXLAG); ++y)
        deinterleave_row(img, rs, cs, coff_.data(), c_, w_, y, interleaved_, s.bp_rs, bard_slots_,
                         bard_p_, s.bp_phase, s.lin.data(), s.bp.data());
    s.bp_filled = std::min(h_ - 1, bands_[bi].r0 + BARD_MAXLAG) + 1;
    // Everything below is in SAMPLED-column space: with the planes split into
    // bard_p_ phases, phase 0 holds exactly the sampled columns, so a cell of cw image
    // columns is cw / bard_p_ of them and the row is that many times nfx.
    const int cw_s = cw / bard_p_;
    const int bard_total = cw_s * nfx;
    // psadbw groups 8 columns, so it can stand in for the per-cell fold only when a
    // group never straddles a cell and the group -> cell map is a shift.
    const int bard_gpc = cw_s / 8, bard_gshift = bard_gpc ? __builtin_ctz((unsigned)bard_gpc) : 0;
    // ... and only while a vector spans a whole number of cells, so the running cell
    // index advances exactly; wider cells than a vector fall back to the fold.
    const int bard_lanes = (int)hn::Lanes(hn::ScalableTag<uint8_t>());
    // ... and only while the row is a whole number of vectors, since a partial
    // trailing vector's groups would scatter past the last cell.
    const bool bard_wide = cw_s % 8 == 0 && bard_gpc > 0 && (bard_gpc & (bard_gpc - 1)) == 0
                           && bard_total % cw_s == 0 && cw_s <= bard_lanes
                           && bard_lanes % cw_s == 0 && bard_total % bard_lanes == 0;
    int bard_cy = -1;
    std::vector<int> bard_rows;
    // Flush the open cell row's buffer into the pyramid accumulators.
    auto bard_flush = [&](int cy) {
      int64_t *IF_RESTRICT base = fb + (size_t)(cy * nfx) * c_ * NSUM;
      // Cell-major. A cell's BARD_NMAP accumulators are adjacent in the destination, so
      // this touches one cache line per cell per channel and writes it once. Walking
      // maps on the outside instead strides the destination by c*NSUM int64 on every
      // write and revisits each line BARD_NMAP times -- with nfx*c_*BARD_NMAP writes per
      // cell row, that ordering is most of what the bank costs on a fine grid.
      for (int e = 0; e < nfx; ++e)
        for (int k = 0; k < c_; ++k)
        {
          int64_t *IF_RESTRICT d = base + (size_t)e * c_ * NSUM + (size_t)k * NSUM + BARD0;
          const uint64_t *IF_RESTRICT src = s.bcell.data() + (size_t)k * BARD_NMAP * nfx + e;
          for (int t = 0; t < BARD_NMAP; ++t) d[t] += (int64_t)src[(size_t)t * nfx];
        }
      std::fill(s.bcell.begin(), s.bcell.end(), (uint64_t)0);
    };
    // One cell row's bard pass: collect its sampled rows, run the fused kernel per
    // channel, add the border columns the vector body scored zero, then flush.
    // Run the rows collected so far. The plane window must hold every row a pending
    // chunk can touch, so bounding the chunk bounds the window: tied to cell height it
    // grows with the grid and spills L1, turning every tap into an L2 hit. Sums land in
    // `cells` and are flushed to the pyramid when the cell row closes.
    auto bard_chunk = [&]() {
      if (bard_rows.empty()) return;
      const size_t pstride = (size_t)bard_slots_ * (size_t)s.bp_rs;
      for (int k = 0; k < c_; ++k)
      {
        const uint8_t *IF_RESTRICT pbase = s.bp.data() + (size_t)k * pstride + BARD_LPAD;
        uint64_t *IF_RESTRICT cells = s.bcell.data() + (size_t)k * BARD_NMAP * nfx;
        if (bard_wide)
          bard_cellrow_fused(pbase, s.bp_rs, bard_slots_, bard_rows.data(),
                             (int)bard_rows.size(), h_, bard_lagmask_,
                             bard_total, cw_s, bard_gshift, nfx, cells, s.smask.data(),
                             bvalid_.data(), bvalid_stride_, bard_p_, s.bp_phase);
        for (size_t qi = 0; qi < bard_rows.size(); ++qi)
        {
          const int r = bard_rows[qi];
          const uint8_t *IF_RESTRICT prow = pbase + (size_t)(r % bard_slots_) * s.bp_rs;
          const uint8_t *up[BARD_NL], *dn[BARD_NL];
          for (int j = 0; j < BARD_NL; ++j)
          {
            const int d = BARD_LAGS[j];
            up[j] = pbase + (size_t)(((r - d) % bard_slots_ + bard_slots_) % bard_slots_) * s.bp_rs;
            dn[j] = pbase + (size_t)((r + d) % bard_slots_) * s.bp_rs;
          }
          if (!bard_wide)
          {
            bool vok[BARD_NL];
            for (int j = 0; j < BARD_NL; ++j)
              vok[j] = ((bard_lagmask_ >> j) & 1) && r >= BARD_LAGS[j] && r < h_ - BARD_LAGS[j];
            bard_row_dense(prow, up, dn, bard_lagmask_, bard_total, s.bq.data(), s.bqs, vok,
                           s.smask.data()); // only reachable with bard_p_ == 1
            for (int x = 0; x < BARD_MAXLAG && x < bard_total; ++x)
              bard_pixel(prow, up, dn, h_, w_, r, x, bard_lagmask_, s.smask.data(), s.bq.data(),
                         s.bqs);
            for (int x = std::max(0, w_ - BARD_MAXLAG); x < bard_total; ++x)
              bard_pixel(prow, up, dn, h_, w_, r, x, bard_lagmask_, s.smask.data(), s.bq.data(),
                         s.bqs);
            bard_row_reduce(s.bq.data(), s.bqs, cw_s, nfx, BARD_NMAP, cells, false);
          }
          // the fused path needs no border pass: its validity masks handle those columns
        }
      }
      bard_rows.clear();
    };
    for (int r = bands_[bi].r0; r < bands_[bi].r1; r += sy_)
    {
      // A cell row's rows all fold into the same cells, so bard runs once the cell
      // row is complete -- before the window advances past its rows.
      if (row_cell_[r] != bard_cy)
      {
        bard_chunk();
        if (bard_cy >= 0) bard_flush(bard_cy);
        bard_cy = row_cell_[r];
      }
      prepare_row(s, img, r, rs, cs);
      int64_t *IF_RESTRICT rb = fb + (size_t)(row_cell_[r] * nfx) * c_ * NSUM;
      for (int ch0 = 0; ch0 < c_; ch0 += L)
        accumulate_row(s.gx.data(), s.gy.data(), s.vrow, s.mx.data(), s.mn.data(), s.lb.data(),
                       c_, ch0, cw, nfx, std::min(L, c_ - ch0), sx_, hcx_, hcy_, c_ * NSUM,
                       rb + (size_t)ch0 * NSUM);
      for (; s.bp_filled <= std::min(h_ - 1, r + BARD_MAXLAG); ++s.bp_filled)
          deinterleave_row(img, rs, cs, coff_.data(), c_, w_, s.bp_filled, interleaved_, s.bp_rs,
                           bard_slots_, bard_p_, s.bp_phase, s.lin.data(), s.bp.data());
      bard_rows.push_back(r);
      if ((int)bard_rows.size() == bard_chunk_rows_) bard_chunk();
      if (np_)
        accumulate_cross_row(s.vrow, c_, cw, nfx, sx_, np_, pi_.data(), pj_.data(),
                             fine.xbuf.data() + (size_t)(row_cell_[r] * nfx) * np_);
      if (!hfast_)
        accumulate_hash_row(s.vrow, c_, sx_, hrbeg_.data(), hrend_.data(), hrhc_.data(),
                            (int)hrhc_.size(), hs + (size_t)hrow_[r] * HN * c_);
    }
    bard_chunk();
    if (bard_cy >= 0) bard_flush(bard_cy);
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
    if (!hfast_)
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

    if (!hfast_) // fold the bands' private hash partials; int64 sums, so still exact
      for (size_t b = 1; b < scr_.size(); ++b)
        for (size_t t = 0; t < hsum_.size(); ++t)
          hsum_[t] += scr_[b].hsum[t];

    const int stripe = c_ * NSUM;
    for (size_t lvl = 1; lvl < levels_.size(); ++lvl)
    {
      const Level &p = levels_[lvl - 1];
      Level &lv = levels_[lvl];
      std::fill(lv.buf.begin(), lv.buf.end(), (int64_t)0);
      std::fill(lv.xbuf.begin(), lv.xbuf.end(), (int64_t)0);
      for (int i = 0; i < p.ny; ++i)
        for (int j = 0; j < p.nx; ++j)
        {
          const size_t src = (size_t)i * p.nx + j;
          const size_t dst = (size_t)(i / lv.fy) * lv.nx + j / lv.fx;
          const int64_t *IF_RESTRICT sc = p.buf.data() + src * stripe;
          int64_t *IF_RESTRICT dc = lv.buf.data() + dst * stripe;
          for (int t = 0; t < stripe; ++t)
            dc[t] += sc[t];
          const int64_t *IF_RESTRICT sx = p.xbuf.data() + src * np_;
          int64_t *IF_RESTRICT dx = lv.xbuf.data() + dst * np_;
          for (int t = 0; t < np_; ++t)
            dx[t] += sx[t];
        }
    }

    const Level &last = levels_.back();
    std::fill(graw_.begin(), graw_.end(), (int64_t)0);
    std::fill(gxraw_.begin(), gxraw_.end(), (int64_t)0);
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
  template <typename T>
  static void fold_summary(const T *IF_RESTRICT v, int n, T *IF_RESTRICT mn,
                           T *IF_RESTRICT mx, double *IF_RESTRICT sm, double *IF_RESTRICT sq)
  {
    for (int j = 0; j < n; ++j) // split by width: a mixed T/double body will not vectorize
    {
      mn[j] = std::min(mn[j], v[j]);
      mx[j] = std::max(mx[j], v[j]);
    }
    for (int j = 0; j < n; ++j)
    {
      const double x = (double)v[j];
      sm[j] += x;
      sq[j] += x * x;
    }
  }

  // Finalize (min,max,sum,sumsq) -> (c, n, NST)=[min,max,mean,std] for one channel.
  template <typename T>
  static void write_summary(int n, size_t nc, const T *IF_RESTRICT mn, const T *IF_RESTRICT mx,
                            const double *IF_RESTRICT sm, const double *IF_RESTRICT sq,
                            T *IF_RESTRICT out)
  {
    const double invn = nc > 0 ? 1.0 / (double)nc : 0.0;
    for (int j = 0; j < n; ++j)
    {
      const double mean = sm[j] * invn, var = sq[j] * invn - mean * mean;
      out[j * NST + 0] = mn[j];
      out[j * NST + 1] = mx[j];
      out[j * NST + 2] = (T)mean;
      out[j * NST + 3] = (T)std::sqrt(var > 0.0 ? var : 0.0);
    }
  }

  // aHash/wHash/pHash from the accumulated HN x HN value-sum grid, one hash per channel.
  // A pure readout of hsum_: no image traversal. Bits are packed row-major, MSB first.
  void derive_hashes()
  {
    // Fast path: build the HN value-sum grid from the finest level's S1 (block-sum when
    // the finest grid is finer than HN), instead of a dedicated per-pixel accumulation.
    if (hfast_)
    {
      const Level &F = levels_[0];
      const int by = F.ny / HN, bx = F.nx / HN;
      std::fill(hsum_.begin(), hsum_.end(), (int64_t)0);
      for (int i = 0; i < HN; ++i)
        for (int j = 0; j < HN; ++j)
        {
          int64_t *IF_RESTRICT dst = hsum_.data() + (size_t)(i * HN + j) * c_;
          for (int dy = 0; dy < by; ++dy)
            for (int dx = 0; dx < bx; ++dx)
            {
              const int64_t *IF_RESTRICT src =
                  F.buf.data() + (size_t)((i * by + dy) * F.nx + (j * bx + dx)) * c_ * NSUM;
              for (int k = 0; k < c_; ++k)
                dst[k] += src[(size_t)k * NSUM + PS0];
            }
        }
    }

    const int NHC = HN * HN;
    std::vector<double> mean(NHC), d0((size_t)HS * HN);
    for (int k = 0; k < c_; ++k)
    {
      for (int i = 0; i < NHC; ++i)
        mean[i] = hcnt_[i] > 0 ? (double)hsum_[(size_t)i * c_ + k] / (double)hcnt_[i] : 0.0;

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
      ahash_[k] = a;
      whash_[k] = w;

      // pHash: separable 2D DCT-II of the 32x32 means (rows 8x32, then cols 8x8), take
      // the low-freq 8x8, threshold by its median. Basis matches scipy.fftpack.dct.
      for (int u = 0; u < HS; ++u)
        for (int j = 0; j < HN; ++j)
        {
          double acc = 0.0;
          for (int n = 0; n < HN; ++n)
            acc += dctb_[(size_t)u * HN + n] * mean[(size_t)n * HN + j];
          d0[(size_t)u * HN + j] = acc;
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
      phash_[k] = p;
    }
  }

  // Per-cell derivation. Every cell writes only its own feat/mom/xfeat slice, so this
  // splits over bands. The cross-cell summary fold is deliberately NOT here: its sum
  // and sum-of-squares are float64 reductions over cells, and splitting them would
  // reassociate the additions and make the output depend on the thread count. It stays
  // serial in derive(), below.
  void derive_cells(int bi)
  {
    const size_t nb = bands_.size();
    for (Level &L : levels_)
    {
      const size_t nc = (size_t)L.ny * L.nx;
      for (size_t cell = nc * (size_t)bi / nb; cell < nc * (size_t)(bi + 1) / nb; ++cell)
      {
        const int64_t *IF_RESTRICT s = L.buf.data() + cell * c_ * NSUM;
        double *IF_RESTRICT m = L.mom.data() + cell * c_ * NMOM;
        for (int k = 0; k < c_; ++k)
        {
          derive_moments(s + (size_t)k * NSUM, m + (size_t)k * NMOM);
          derive_cell(s + (size_t)k * NSUM, m + (size_t)k * NMOM,
                      L.feat.data() + (cell * c_ + k) * NF);
        }
        if (np_)
          derive_cross(m, s[CNT], L.xbuf.data() + cell * np_, L.xfeat.data() + cell * np_ * NXF);
      }
    }
  }

  void derive()
  {
    run_job(JOB_DERIVE);
    // Largest finite values, NOT +/-inf: -ffast-math implies -ffinite-math-only, under
    // which some compilers (e.g. Apple clang) miscompile a min/max reduction seeded with
    // infinity and leave the accumulator at its init. FLT_MAX/-FLT_MAX are equivalent for
    // any finite feature and safe. (-FBIG == lowest(), the correct max seed.)
    const float FBIG = std::numeric_limits<float>::max();
    const double DBIG = std::numeric_limits<double>::max();
    float *const fmn = sfmn_.data(), *const fmx = sfmx_.data();
    double *const fsm = sfsm_.data(), *const fsq = sfsq_.data();
    double *const mmn = smmn_.data(), *const mmx = smmx_.data();
    double *const msm = smsm_.data(), *const msq = smsq_.data();
    for (Level &L : levels_)
    {
      // per-(channel,feature) cross-cell accumulators, reset per level
      std::fill_n(fmn, c_ * NF, FBIG);
      std::fill_n(fmx, c_ * NF, -FBIG);
      std::fill_n(fsm, c_ * NF, 0.0);
      std::fill_n(fsq, c_ * NF, 0.0);
      std::fill_n(mmn, c_ * NMOM, DBIG);
      std::fill_n(mmx, c_ * NMOM, -DBIG);
      std::fill_n(msm, c_ * NMOM, 0.0);
      std::fill_n(msq, c_ * NMOM, 0.0);
      const size_t nc = (size_t)L.ny * L.nx;
      for (size_t cell = 0; cell < nc; ++cell)
        for (int k = 0; k < c_; ++k)
        {
          const double *IF_RESTRICT mk = L.mom.data() + cell * c_ * NMOM + (size_t)k * NMOM;
          const float *IF_RESTRICT fk = L.feat.data() + (cell * c_ + k) * NF;
          fold_summary(fk, NF, &fmn[k * NF], &fmx[k * NF], &fsm[k * NF], &fsq[k * NF]);
          fold_summary(mk, NMOM, &mmn[k * NMOM], &mmx[k * NMOM], &msm[k * NMOM], &msq[k * NMOM]);
        }
      for (int k = 0; k < c_; ++k)
      {
        write_summary(NF, nc, &fmn[k * NF], &fmx[k * NF], &fsm[k * NF], &fsq[k * NF],
                      L.fsum.data() + (size_t)k * NF * NST);
        write_summary(NMOM, nc, &mmn[k * NMOM], &mmx[k * NMOM], &msm[k * NMOM], &msq[k * NMOM],
                      L.msum.data() + (size_t)k * NMOM * NST);
      }
    }
    for (int k = 0; k < c_; ++k)
    {
      derive_moments(graw_.data() + (size_t)k * NSUM, gmom_.data() + (size_t)k * NMOM);
      derive_cell(graw_.data() + (size_t)k * NSUM, gmom_.data() + (size_t)k * NMOM,
                  gfeat_.data() + (size_t)k * NF);
    }
    if (np_)
      derive_cross(gmom_.data(), graw_[CNT], gxraw_.data(), gxfeat_.data());
  }

public:
  FeatureComputer() = default;

  void set_config(const std::vector<int64_t> &dims, const std::vector<int> &channels,
                  const std::vector<std::vector<int>> &grids,
                  const std::vector<int64_t> &stride, int threads)
  {
    pool_stop();
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
      hcx_[k - 1] = (int32_t)std::llround(RAYSCALE * std::cos(a));
      hcy_[k - 1] = (int32_t)std::llround(RAYSCALE * std::sin(a));
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
      L.feat.assign((size_t)L.ny * L.nx * c_ * NF, 0.0f);
      L.mom.assign((size_t)L.ny * L.nx * c_ * NMOM, 0.0);
      L.fsum.assign((size_t)c_ * NF * NST, 0.0f);
      L.msum.assign((size_t)c_ * NMOM * NST, 0.0);
      L.xbuf.assign((size_t)L.ny * L.nx * np_, 0);
      L.xfeat.assign((size_t)L.ny * L.nx * np_ * NXF, 0.0f);
      L.rshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NSUM};
      L.fshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NF};
      L.mshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NMOM};
      L.xrshape = {(size_t)L.ny, (size_t)L.nx, (size_t)np_};
      L.xfshape = {(size_t)L.ny, (size_t)L.nx, (size_t)np_, (size_t)NXF};
      L.fsumshape = {(size_t)c_, (size_t)NF, (size_t)NST};
      L.msumshape = {(size_t)c_, (size_t)NMOM, (size_t)NST};
      levels_.push_back(std::move(L));
    }

    row_cell_.resize(h_);
    for (int r = 0; r < h_; ++r)
      row_cell_[r] = (int)((int64_t)r * levels_[0].ny / h_);

    // The fold runs per cell row, so the plane window must span one cell row's rows
    // plus a MAXLAG halo on each side.
    // Sized so the window stays inside L1: a chunk reaches (chunk-1)*sy + 2*MAXLAG + 1
    // rows, and the window is c * slots * (w + padding) bytes.
    // A lag contributes on the horizontal axis only where BOTH its taps are inside the
    // image; the mask is a property of the column alone, so it is built once here.
    // Split the planes by column phase when a cell still holds a whole number of
    // psadbw groups afterwards; otherwise leave them whole (P = 1). Sampled columns are
    // phase 0, so the kernel walks 1/P as many vectors for identical output.
    {
      const int cwf = levels_.empty() ? 0 : w_ / levels_[0].nx;
      bard_p_ = (sx_ > 1 && w_ % sx_ == 0 && cwf % (8 * sx_) == 0) ? sx_ : 1;
    }
    bvalid_stride_ = (size_t)(w_ / bard_p_) + SIMD_PAD + BARD_LPAD + BARD_RPAD;
    bvalid_.assign(bvalid_stride_ * BARD_NL, 0);
    for (int j = 0; j < BARD_NL; ++j)
    {
      const int d = BARD_LAGS[j];
      for (int xs = 0; xs < w_ / bard_p_; ++xs)
      {
        const int x = xs * bard_p_; // original column of this sampled column
        if (x >= d && x < w_ - d) bvalid_[(size_t)j * bvalid_stride_ + (size_t)xs] = 255;
      }
    }
    // Chunk the whole cell row, not a fixed slice of it. A bigger chunk measured
    // strictly faster at every size tried: each plane row's vector serves more of the
    // centres that use it as a tap while it is still hot, and the per-block reduction
    // amortizes over more rows. The window has to span the cell row plus a halo.
    bard_chunk_rows_ = INT_MAX;
    bard_slots_ = std::max(2 * BARD_MAXLAG + 1, h_ / levels_[0].ny + 2 * BARD_MAXLAG + 1);
    coff_.assign(c_, 0);

    graw_.assign((size_t)c_ * NSUM, 0);
    gfeat_.assign((size_t)c_ * NF, 0.0f);
    gmom_.assign((size_t)c_ * NMOM, 0.0);
    sfmn_.resize((size_t)c_ * NF);
    sfmx_.resize((size_t)c_ * NF);
    sfsm_.resize((size_t)c_ * NF);
    sfsq_.resize((size_t)c_ * NF);
    smmn_.resize((size_t)c_ * NMOM);
    smmx_.resize((size_t)c_ * NMOM);
    smsm_.resize((size_t)c_ * NMOM);
    smsq_.resize((size_t)c_ * NMOM);
    gxraw_.assign((size_t)np_, 0);
    gxfeat_.assign((size_t)np_ * NXF, 0.0f);
    graw_shape_ = {(size_t)c_, (size_t)NSUM};
    gfeat_shape_ = {(size_t)c_, (size_t)NF};
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
          hrbeg_.push_back(b);
          hrend_.push_back(c);
          hrhc_.push_back(hc);
        }
      }
    }
    hcnt_.assign((size_t)HN * HN, 0);
    for (int i = 0; i < HN; ++i)
      for (int j = 0; j < HN; ++j)
        hcnt_[(size_t)i * HN + j] = rcnt[i] * ccnt[j];
    // When the finest feature grid tiles the HN hash grid, its per-cell S1 (== Sum(v)
    // over the same sampled pixels) already is the hash grid: reuse it in derive_hashes
    // and skip the per-pixel pass entirely. Otherwise fall back to accumulate_hash_row.
    hfast_ = levels_[0].ny % HN == 0 && levels_[0].nx % HN == 0;
    dctb_.assign((size_t)HS * HN, 0.0);
    for (int u = 0; u < HS; ++u)
      for (int n = 0; n < HN; ++n)
        dctb_[(size_t)u * HN + n] = 2.0 * std::cos(PI * u * (2 * n + 1) / (2.0 * HN));
    hsum_.assign((size_t)HN * HN * c_, 0);
    ahash_.assign((size_t)c_, 0);
    whash_.assign((size_t)c_, 0);
    phash_.assign((size_t)c_, 0);
    hshape_ = {(size_t)c_};

    // Split the finest cell rows evenly, then map each band back to image rows:
    // row_cell_[r] >= a  <=>  r >= ceil(a * h_ / ny), so the ranges tile [0, h_)
    // exactly. Rounding r0 up to the sy_ grid keeps the visited set identical to
    // the serial walk -- every row is still touched, by exactly one band.
    const int ncy = levels_[0].ny, nb = std::min(threads > 1 ? threads : 1, ncy);
    bands_.clear();
    for (int t = 0; t < nb; ++t)
    {
      const int a = (int)((int64_t)t * ncy / nb), z = (int)((int64_t)(t + 1) * ncy / nb);
      int r0 = (int)(((int64_t)a * h_ + ncy - 1) / ncy);
      const int r1 = (int)(((int64_t)z * h_ + ncy - 1) / ncy);
      r0 = (r0 + sy_ - 1) / sy_ * sy_;
      bands_.push_back({r0, r1, a, z});
    }
    const size_t wcp = (size_t)c_ * w_ + SIMD_PAD;
    scr_.assign(bands_.size(), Scratch());
    for (size_t b = 0; b < scr_.size(); ++b)
    {
      Scratch &s = scr_[b];
      for (std::vector<uint8_t> *v : {&s.r0, &s.r1, &s.r2, &s.cmx, &s.cmn, &s.mx, &s.mn, &s.lb})
        v->assign(wcp, 0);
      for (std::vector<int16_t> *v : {&s.vs, &s.vd, &s.gx, &s.gy})
        v->assign(wcp, 0);
      s.bp_phase = (int64_t)(w_ / bard_p_) + BARD_LPAD + BARD_RPAD;
      s.bp_rs = s.bp_phase * bard_p_;
      // one slack row so the final vector's overrun stays inside the allocation
      s.bp.assign((size_t)c_ * (size_t)bard_slots_ * (size_t)s.bp_rs + (size_t)s.bp_rs, 0);
      s.bqs = (size_t)w_ + SIMD_PAD + 128; // own store slack; not tied to plane padding // a whole vector of store slack for the row maps
      s.bq.assign(s.bqs * (size_t)BARD_NMAP, 0);
      s.smask.assign(s.bqs, 0);
      for (int x = 0; x < w_ / bard_p_; x += sx_ / bard_p_) s.smask[(size_t)x] = 255;
      s.bcell.assign((size_t)c_ * BARD_NMAP * (size_t)levels_[0].nx + 8, 0u);
      s.lin.assign((size_t)c_ * (size_t)w_ + SIMD_PAD, 0);
      s.smaskv = s.smask;
      for (int x = 0; x < BARD_MAXLAG && x < w_; ++x) s.smaskv[(size_t)x] = 0;
      for (int x = std::max(0, w_ - BARD_MAXLAG); x < w_; ++x) s.smaskv[(size_t)x] = 0;
      if (b && !hfast_)
        s.hsum.assign((size_t)HN * HN * c_, 0);
    }
    for (size_t b = 1; b < bands_.size(); ++b)
      workers_.emplace_back([this, b] { worker((int)b); });
  }

  int threads() const { return (int)bands_.size(); }

  ~FeatureComputer() { pool_stop(); }

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

  // Two parallel lists: the float32 structure features and the float64 moments
  // (moments span 0..255^4, where float32's 7 digits would cost precision).
  nb::list features(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    accumulate(img, rs, cs, chs);
    derive();
    derive_hashes();
    nb::list feat, mom, cross, fsum, msum;
    for (Level &L : levels_)
    {
      feat.append(nb::ndarray<nb::numpy, float>(L.feat.data(), L.fshape.size(), L.fshape.data(),
                                                nb::handle()));
      mom.append(nb::ndarray<nb::numpy, double>(L.mom.data(), L.mshape.size(), L.mshape.data(),
                                                nb::handle()));
      if (np_)
        cross.append(nb::ndarray<nb::numpy, float>(L.xfeat.data(), L.xfshape.size(),
                                                   L.xfshape.data(), nb::handle()));
      fsum.append(nb::ndarray<nb::numpy, float>(L.fsum.data(), L.fsumshape.size(),
                                                L.fsumshape.data(), nb::handle()));
      msum.append(nb::ndarray<nb::numpy, double>(L.msum.data(), L.msumshape.size(),
                                                 L.msumshape.data(), nb::handle()));
    }
    feat.append(nb::ndarray<nb::numpy, float>(gfeat_.data(), gfeat_shape_.size(),
                                              gfeat_shape_.data(), nb::handle()));
    mom.append(nb::ndarray<nb::numpy, double>(gmom_.data(), gmom_shape_.size(),
                                              gmom_shape_.data(), nb::handle()));
    if (np_)
      cross.append(nb::ndarray<nb::numpy, float>(gxfeat_.data(), gxfeat_shape_.size(),
                                                 gxfeat_shape_.data(), nb::handle()));
    nb::list hashes;
    hashes.append(nb::ndarray<nb::numpy, uint64_t>(ahash_.data(), hshape_.size(),
                                                   hshape_.data(), nb::handle()));
    hashes.append(nb::ndarray<nb::numpy, uint64_t>(whash_.data(), hshape_.size(),
                                                   hshape_.data(), nb::handle()));
    hashes.append(nb::ndarray<nb::numpy, uint64_t>(phash_.data(), hshape_.size(),
                                                   hshape_.data(), nb::handle()));
    nb::list all;
    all.append(feat);
    all.append(mom);
    all.append(cross);
    all.append(fsum);
    all.append(msum);
    all.append(hashes);
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
