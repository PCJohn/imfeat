#include "hwy/highway.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
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
constexpr int NSUM = SG4 + 1;    // 30 int64 sums per cell per channel
constexpr int NF_S = 5;          // tensor-derived float channels
// Model-ready nonlinear descriptors, all DERIVED from the sums above (no new
// accumulators): standardized skew/kurtosis, two structure-tensor ratios, and two
// HOG histogram-shape summaries. Nonlinear (ratios/products/argmax-free peakedness)
// so a linear/shallow model cannot cheaply reconstruct them; dimensionless ones are
// illumination-invariant, which helps few-shot on-the-fly training.
constexpr int NDER = 8;
constexpr int NF = NF_S + HB + 2 + LBPB + NDER; // 33 derived float32 per cell per channel
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
  int np_ = 0;               // number of channel pairs (0 if C<2 or C>XMAX)
  std::vector<int> pi_, pj_; // the pairs, in (i<j) lexicographic order

  // Per-row scratch, all interleaved (c * c_ + k) and padded by SIMD_PAD. One row
  // is a few KB, so it stays hot in L1 across the whole cell sweep.
  std::vector<uint8_t> r0_, r1_, r2_, cmx_, cmn_, mx_, mn_, lb_;
  std::vector<int16_t> vs_, vd_, gx_, gy_;

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
  void prepare_row(const uint8_t *IF_RESTRICT img, int r, int64_t rs, int64_t cs)
  {
    const int hi_r = h_ - 1, WC = w_ * c_;
    const uint8_t *rows[3] = {img + (int64_t)clampi(r - 1, hi_r) * rs, img + (int64_t)r * rs,
                              img + (int64_t)clampi(r + 1, hi_r) * rs};
    uint8_t *dst[3] = {r0_.data(), r1_.data(), r2_.data()};
    for (int i = 0; i < 3; ++i)
    {
      if (interleaved_)
        std::memcpy(dst[i], rows[i], (size_t)WC);
      else
        for (int c = 0; c < w_; ++c)
          for (int k = 0; k < c_; ++k)
            dst[i][(size_t)c * c_ + k] = rows[i][(int64_t)c * cs + coff_[k]];
    }
    // Alias every scratch buffer through a restrict pointer: via the std::vector
    // members the compiler cannot prove the reads and writes are disjoint, and it
    // silently drops the flat loops below back to scalar.
    const uint8_t *IF_RESTRICT a = r0_.data(), *IF_RESTRICT b = r1_.data(),
                               *IF_RESTRICT e = r2_.data();
    const uint8_t *IF_RESTRICT cmx = cmx_.data(), *IF_RESTRICT cmn = cmn_.data();
    uint8_t *IF_RESTRICT mx = mx_.data(), *IF_RESTRICT mn = mn_.data(),
                         *IF_RESTRICT lb = lb_.data();
    // Three flat, branch-free loops over W*C, each auto-vectorized (the uint8 min/max
    // ones 16 lanes wide). Kept separate so no loop carries a mixed-width dependency
    // the vectorizer would refuse.
    for (int o = 0; o < WC; ++o)
    {
      vs_[o] = (int16_t)(a[o] + 2 * b[o] + e[o]); // vertical Sobel: smooth, difference
      vd_[o] = (int16_t)(e[o] - a[o]);
      cmx_[o] = a[o] > e[o] ? a[o] : e[o]; // the column's two off-centre rows; the
      cmn_[o] = a[o] < e[o] ? a[o] : e[o]; //   horizontal 3-window below completes it
    }
    const int C = c_;
    for (int o = C; o < WC - C; ++o)
    {
      gx_[o] = (int16_t)(vs_[o + C] - vs_[o - C]);
      gy_[o] = (int16_t)(vd_[o - C] + 2 * vd_[o] + vd_[o + C]);
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
        gx_[o] = (int16_t)(vs_[op] - vs_[om]);
        gy_[o] = (int16_t)(vd_[om] + 2 * vd_[o] + vd_[op]);
        mx[o] = std::max({cmx[om], cmx[o], cmx[op], b[om], b[op]});
        mn[o] = std::min({cmn[om], cmn[o], cmn[op], b[om], b[op]});
        lb[o] = lbp_riu2(a, b, e, om, o, op);
      }
  }

  void accumulate(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    Level &fine = levels_[0];
    const int nfx = fine.nx, cw = w_ / nfx;
    int64_t *IF_RESTRICT fb = fine.buf.data();
    std::fill(fine.buf.begin(), fine.buf.end(), (int64_t)0);
    std::fill(fine.xbuf.begin(), fine.xbuf.end(), (int64_t)0);

    for (int k = 0; k < c_; ++k)
      coff_[k] = (int64_t)chan_[k] * chs;
    interleaved_ = chs == 1 && cs == c_ && chan_[0] == 0 && chan_.back() == c_ - 1;

    namespace hn = hwy::HWY_NAMESPACE;
    const int L = (int)hn::Lanes(hn::CappedTag<int32_t, SIMD_CH>());
    for (int r = 0; r < h_; r += sy_)
    {
      prepare_row(img, r, rs, cs);
      int64_t *IF_RESTRICT rb = fb + (size_t)(row_cell_[r] * nfx) * c_ * NSUM;
      for (int ch0 = 0; ch0 < c_; ch0 += L)
        accumulate_row(gx_.data(), gy_.data(), r1_.data(), mx_.data(), mn_.data(), lb_.data(),
                       c_, ch0, cw, nfx, std::min(L, c_ - ch0), sx_, hcx_, hcy_, c_ * NSUM,
                       rb + (size_t)ch0 * NSUM);
      if (np_)
        accumulate_cross_row(r1_.data(), c_, cw, nfx, sx_, np_, pi_.data(), pj_.data(),
                             fine.xbuf.data() + (size_t)(row_cell_[r] * nfx) * np_);
    }

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
    for (int j = 0; j < n; ++j)
    {
      const T x = v[j];
      mn[j] = std::min(mn[j], x);
      mx[j] = std::max(mx[j], x);
      sm[j] += (double)x;
      sq[j] += (double)x * (double)x;
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

  void derive()
  {
    const float FINF = std::numeric_limits<float>::infinity();
    const double DINF = std::numeric_limits<double>::infinity();
    float *const fmn = sfmn_.data(), *const fmx = sfmx_.data();
    double *const fsm = sfsm_.data(), *const fsq = sfsq_.data();
    double *const mmn = smmn_.data(), *const mmx = smmx_.data();
    double *const msm = smsm_.data(), *const msq = smsq_.data();
    for (Level &L : levels_)
    {
      // per-(channel,feature) cross-cell accumulators, reset per level
      std::fill_n(fmn, c_ * NF, FINF);
      std::fill_n(fmx, c_ * NF, -FINF);
      std::fill_n(fsm, c_ * NF, 0.0);
      std::fill_n(fsq, c_ * NF, 0.0);
      std::fill_n(mmn, c_ * NMOM, DINF);
      std::fill_n(mmx, c_ * NMOM, -DINF);
      std::fill_n(msm, c_ * NMOM, 0.0);
      std::fill_n(msq, c_ * NMOM, 0.0);
      const size_t nc = (size_t)L.ny * L.nx;
      for (size_t cell = 0; cell < nc; ++cell)
      {
        const int64_t *IF_RESTRICT s = L.buf.data() + cell * c_ * NSUM;
        double *IF_RESTRICT m = L.mom.data() + cell * c_ * NMOM;
        for (int k = 0; k < c_; ++k)
        {
          double *IF_RESTRICT mk = m + (size_t)k * NMOM;
          float *IF_RESTRICT fk = L.feat.data() + (cell * c_ + k) * NF;
          derive_moments(s + (size_t)k * NSUM, mk);
          derive_cell(s + (size_t)k * NSUM, mk, fk);
          fold_summary(fk, NF, &fmn[k * NF], &fmx[k * NF], &fsm[k * NF], &fsq[k * NF]);
          fold_summary(mk, NMOM, &mmn[k * NMOM], &mmx[k * NMOM], &msm[k * NMOM], &msq[k * NMOM]);
        }
        if (np_)
          derive_cross(m, s[CNT], L.xbuf.data() + cell * np_,
                       L.xfeat.data() + cell * np_ * NXF);
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
                  const std::vector<int64_t> &stride)
  {
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

    coff_.assign(c_, 0);
    const size_t wcp = (size_t)c_ * w_ + SIMD_PAD;
    r0_.assign(wcp, 0);
    r1_.assign(wcp, 0);
    r2_.assign(wcp, 0);
    cmx_.assign(wcp, 0);
    cmn_.assign(wcp, 0);
    mx_.assign(wcp, 0);
    mn_.assign(wcp, 0);
    lb_.assign(wcp, 0);
    vs_.assign(wcp, 0);
    vd_.assign(wcp, 0);
    gx_.assign(wcp, 0);
    gy_.assign(wcp, 0);

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
    nb::list all;
    all.append(feat);
    all.append(mom);
    all.append(cross);
    all.append(fsum);
    all.append(msum);
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
  m.attr("XMAX") = XMAX; // cross-channel products are computed only for C <= XMAX
  nb::class_<FeatureComputer>(m, "_FeatureComputerImpl")
      .def(nb::init<>())
      .def("set_config", &FeatureComputer::set_config, nb::arg("dims"), nb::arg("channels"),
           nb::arg("grids"), nb::arg("stride"))
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
