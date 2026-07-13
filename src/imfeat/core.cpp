#include "hwy/highway.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
//   | [15..18] S1 S2 S3 S4 (raw power sums of the pixel value).
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
constexpr int NSUM = PS0 + NMOM;  // 19 int64 sums per cell per channel
constexpr int NF_S = 5;           // tensor-derived float channels
constexpr int NF = NF_S + HB + 2; // 16 derived float32 per cell per channel
constexpr int64_t RAYSCALE = 1 << 14;
constexpr double PI = 3.14159265358979323846;

// The CHANNEL is the SIMD lane. SIMD_CH is the lane cap of the int32 tag (a
// 128-bit vector holds 4); channels are processed in groups of that many, so every
// scratch array is sized from it. SIMD_PAD (bytes in a 128-bit vector) pads the
// interleaved row buffers so a full-vector LoadU past the last channel of the last
// column stays in-bounds -- the surplus lanes are computed and then discarded.
constexpr int SIMD_CH = 4;
constexpr int SIMD_PAD = 16;

inline int clampi(int v, int hi)
{
  return v < 0 ? 0 : (v > hi ? hi : v);
}

// Sweeps one row of `ncell` finest cells (each `cw` px wide) for one group of `n`
// channels, resetting the accumulators at each cell start and flushing them at its
// end. One call per row (not per cell) amortises the vector-descriptor and
// ray-direction setup over the whole row.
void accumulate_row(const int16_t *IF_RESTRICT gxb, const int16_t *IF_RESTRICT gyb,
                    const uint8_t *IF_RESTRICT vb, const uint8_t *IF_RESTRICT mxb,
                    const uint8_t *IF_RESTRICT mnb, int strideC, int ch0, int cw, int ncell,
                    int n, int sx, const int32_t *IF_RESTRICT hcx,
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

  auto i16_at = [&](const int16_t *p, int col) {
    return hn::PromoteTo(d, hn::LoadU(d16, p + (size_t)col * strideC + ch0));
  };
  auto u8_at = [&](const uint8_t *p, int col) {
    return hn::PromoteTo(d, hn::LoadU(d8, p + (size_t)col * strideC + ch0));
  };

  hn::VFromD<decltype(d)> hcxv[HB - 1], hcyv[HB - 1];
  for (int j = 0; j < HB - 1; ++j)
  {
    hcxv[j] = hn::Set(d, hcx[j]);
    hcyv[j] = hn::Set(d, hcy[j]);
  }
  int64_t hog[HB * SIMD_CH];
  int32_t blane[SIMD_CH], g2lane[SIMD_CH], nmxa[SIMD_CH], nmna[SIMD_CH];
  int64_t xx[SIMD_CH], yy[SIMD_CH], xy[SIMD_CH], ps[NMOM][SIMD_CH];

  for (int e = 0; e < ncell; ++e)
  {
    auto axl = hn::Zero(d64), axh = hn::Zero(d64), ayl = hn::Zero(d64), ayh = hn::Zero(d64),
         axyl = hn::Zero(d64), axyh = hn::Zero(d64);
    hn::VFromD<decltype(d64)> pl[NMOM], ph[NMOM];
    for (int k = 0; k < NMOM; ++k)
      pl[k] = ph[k] = hn::Zero(d64);
    auto nmaxv = zero, nminv = zero;
    for (int k = 0; k < HB * n; ++k)
      hog[k] = 0;
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
      hn::StoreU(hn::Add(gx2, gy2), d, g2lane);
      for (int l = 0; l < n; ++l)
        if (g2lane[l])
          hog[l * HB + blane[l]] += (int64_t)g2lane[l];
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
      a[NMAX] += nmxa[l];
      a[NMIN] += nmna[l];
      for (int k = 0; k < NMOM; ++k)
        a[PS0 + k] += ps[k][l];
    }
  }
}

struct Level
{
  int ny = 0, nx = 0, fy = 0, fx = 0;
  std::vector<int64_t> buf;
  std::vector<float> feat;
  std::vector<double> mom;
  std::vector<size_t> rshape, fshape, mshape;
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

  // Per-row scratch, all interleaved (c * c_ + k) and padded by SIMD_PAD. One row
  // is a few KB, so it stays hot in L1 across the whole cell sweep.
  std::vector<uint8_t> r0_, r1_, r2_, cmx_, cmn_, mx_, mn_;
  std::vector<int16_t> vs_, vd_, gx_, gy_;

  std::vector<int64_t> graw_;
  std::vector<float> gfeat_;
  std::vector<double> gmom_;
  std::vector<size_t> graw_shape_, gfeat_shape_, gmom_shape_;

  // Structure-tensor + HOG + extrema features (float32).
  void derive_cell(const int64_t *IF_RESTRICT s, float *IF_RESTRICT f) const
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
    const uint8_t *IF_RESTRICT a = r0_.data(), *IF_RESTRICT b = r1_.data(),
                               *IF_RESTRICT e = r2_.data();
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
    for (int o = C; o < WC - C; ++o)
    {
      uint8_t hi = cmx_[o - C] > cmx_[o + C] ? cmx_[o - C] : cmx_[o + C];
      uint8_t lo = cmn_[o - C] < cmn_[o + C] ? cmn_[o - C] : cmn_[o + C];
      const uint8_t bhi = b[o - C] > b[o + C] ? b[o - C] : b[o + C];
      const uint8_t blo = b[o - C] < b[o + C] ? b[o - C] : b[o + C];
      hi = hi > cmx_[o] ? hi : cmx_[o];
      lo = lo < cmn_[o] ? lo : cmn_[o];
      mx_[o] = hi > bhi ? hi : bhi;
      mn_[o] = lo < blo ? lo : blo;
    }
    const int last = (w_ - 1) * C, step = w_ > 1 ? C : 0;
    for (int k = 0; k < C; ++k) // the two border columns, clamped
      for (int o : {k, last + k})
      {
        const int om = o == k ? k : last - step + k, op = o == k ? k + step : last + k;
        gx_[o] = (int16_t)(vs_[op] - vs_[om]);
        gy_[o] = (int16_t)(vd_[om] + 2 * vd_[o] + vd_[op]);
        mx_[o] = std::max({cmx_[om], cmx_[o], cmx_[op], b[om], b[op]});
        mn_[o] = std::min({cmn_[om], cmn_[o], cmn_[op], b[om], b[op]});
      }
  }

  void accumulate(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    Level &fine = levels_[0];
    const int nfx = fine.nx, cw = w_ / nfx;
    int64_t *IF_RESTRICT fb = fine.buf.data();
    std::fill(fine.buf.begin(), fine.buf.end(), (int64_t)0);

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
        accumulate_row(gx_.data(), gy_.data(), r1_.data(), mx_.data(), mn_.data(), c_, ch0, cw,
                       nfx, std::min(L, c_ - ch0), sx_, hcx_, hcy_, c_ * NSUM,
                       rb + (size_t)ch0 * NSUM);
    }

    const int stripe = c_ * NSUM;
    for (size_t lvl = 1; lvl < levels_.size(); ++lvl)
    {
      const Level &p = levels_[lvl - 1];
      Level &lv = levels_[lvl];
      std::fill(lv.buf.begin(), lv.buf.end(), (int64_t)0);
      for (int i = 0; i < p.ny; ++i)
        for (int j = 0; j < p.nx; ++j)
        {
          const int64_t *IF_RESTRICT sc = p.buf.data() + ((size_t)i * p.nx + j) * stripe;
          int64_t *IF_RESTRICT dc =
              lv.buf.data() + ((size_t)(i / lv.fy) * lv.nx + j / lv.fx) * stripe;
          for (int t = 0; t < stripe; ++t)
            dc[t] += sc[t];
        }
    }

    const Level &last = levels_.back();
    std::fill(graw_.begin(), graw_.end(), (int64_t)0);
    for (size_t cell = 0, nc = (size_t)last.ny * last.nx; cell < nc; ++cell)
    {
      const int64_t *IF_RESTRICT s = last.buf.data() + cell * stripe;
      for (int t = 0; t < stripe; ++t)
        graw_[t] += s[t];
    }
  }

  void derive()
  {
    for (Level &L : levels_)
      for (size_t cell = 0, nc = (size_t)L.ny * L.nx * c_; cell < nc; ++cell)
      {
        derive_cell(L.buf.data() + cell * NSUM, L.feat.data() + cell * NF);
        derive_moments(L.buf.data() + cell * NSUM, L.mom.data() + cell * NMOM);
      }
    for (int k = 0; k < c_; ++k)
    {
      derive_cell(graw_.data() + (size_t)k * NSUM, gfeat_.data() + (size_t)k * NF);
      derive_moments(graw_.data() + (size_t)k * NSUM, gmom_.data() + (size_t)k * NMOM);
    }
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

    for (int k = 1; k < HB; ++k)
    {
      const double a = PI * k / HB;
      hcx_[k - 1] = (int32_t)std::llround(RAYSCALE * std::cos(a));
      hcy_[k - 1] = (int32_t)std::llround(RAYSCALE * std::sin(a));
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
      L.rshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NSUM};
      L.fshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NF};
      L.mshape = {(size_t)L.ny, (size_t)L.nx, (size_t)c_, (size_t)NMOM};
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
    vs_.assign(wcp, 0);
    vd_.assign(wcp, 0);
    gx_.assign(wcp, 0);
    gy_.assign(wcp, 0);

    graw_.assign((size_t)c_ * NSUM, 0);
    gfeat_.assign((size_t)c_ * NF, 0.0f);
    gmom_.assign((size_t)c_ * NMOM, 0.0);
    graw_shape_ = {(size_t)c_, (size_t)NSUM};
    gfeat_shape_ = {(size_t)c_, (size_t)NF};
    gmom_shape_ = {(size_t)c_, (size_t)NMOM};
  }

  nb::list raw(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    accumulate(img, rs, cs, chs);
    nb::list out;
    for (Level &L : levels_)
      out.append(nb::ndarray<nb::numpy, int64_t>(L.buf.data(), L.rshape.size(), L.rshape.data(),
                                                 nb::handle()));
    out.append(nb::ndarray<nb::numpy, int64_t>(graw_.data(), graw_shape_.size(),
                                               graw_shape_.data(), nb::handle()));
    return out;
  }

  // Two parallel lists: the float32 structure features and the float64 moments
  // (moments span 0..255^4, where float32's 7 digits would cost precision).
  nb::list features(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    accumulate(img, rs, cs, chs);
    derive();
    nb::list feat, mom;
    for (Level &L : levels_)
    {
      feat.append(nb::ndarray<nb::numpy, float>(L.feat.data(), L.fshape.size(), L.fshape.data(),
                                                nb::handle()));
      mom.append(nb::ndarray<nb::numpy, double>(L.mom.data(), L.mshape.size(), L.mshape.data(),
                                                nb::handle()));
    }
    feat.append(nb::ndarray<nb::numpy, float>(gfeat_.data(), gfeat_shape_.size(),
                                              gfeat_shape_.data(), nb::handle()));
    mom.append(nb::ndarray<nb::numpy, double>(gmom_.data(), gmom_shape_.size(),
                                              gmom_shape_.data(), nb::handle()));
    nb::list both;
    both.append(feat);
    both.append(mom);
    return both;
  }
};

using Arr = nb::ndarray<nb::numpy, const uint8_t, nb::device::cpu>;
} // namespace

NB_MODULE(imfeat_core, m)
{
  m.doc() = "imfeat internal C++ module. Public API: imfeat.FeatureComputer.";
  m.attr("HB") = HB; // HOG orientation-bin count; Python derives its bin labels from this
  nb::class_<FeatureComputer>(m, "_FeatureComputerImpl")
      .def(nb::init<>())
      .def("set_config", &FeatureComputer::set_config, nb::arg("dims"), nb::arg("channels"),
           nb::arg("grids"), nb::arg("stride"))
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
