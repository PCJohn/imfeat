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
// Per finest cell per channel we keep NSUM int64 additive sums:
//   [0..3] Sxx Syy Sxy count | [4..12] HOG(9) | [13,14] local-max/min counts
//   | [15..18] S1 S2 S3 S4 (raw power sums of the pixel value).
// Everything is additive, so pyramid levels and the global reduction are exact
// sums of the finest cells -- the image is read exactly once. Central moments
// and the tensor eigen-features are nonlinear, so they are derived at the end.
// Channels share ONE spatial traversal: each pixel updates its channel's slot.
// Input axis order is normalised to (H, W, C) by the wrapper (zero-copy).

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

// SIMD groups channels into vector lanes. SIMD_CH is the lane cap of the int32
// tag (a 128-bit vector holds 4 int32); every SIMD scratch array is sized from it.
// SIMD_PAD (bytes in a 128-bit vector) pads the interleaved row buffers so a LoadU
// reading a full vector past the last channel of the last column stays in-bounds.
constexpr int SIMD_CH = 4;
constexpr int SIMD_PAD = 16;

inline int clampi(int v, int hi)
{
  return v < 0 ? 0 : (v > hi ? hi : v);
}

// Channels-in-lanes SIMD kernel: the CHANNEL is the vector lane. Sweeps a whole
// row of `ncell` finest cells (each `cw` wide) in one call, resetting the
// accumulators at each cell start and flushing them to that cell at its end. One
// call per row (rather than one per cell) amortises the per-call setup -- the
// vector descriptors and the ray-direction broadcasts -- over the row; the pixel
// loop itself runs at the same rate either way. Reads interleaved vs/vd
// (vsb[c*strideC + ch]) and raw rows. Bit-exact to accumulate_run.
static void accumulate_row_simd(const int16_t *IF_RESTRICT vsb, const int16_t *IF_RESTRICT vdb,
                                const uint8_t *IF_RESTRICT r0, const uint8_t *IF_RESTRICT r1,
                                const uint8_t *IF_RESTRICT r2, int strideC, int ch0, int cw,
                                int ncell, int hi_c, int n, int sx,
                                const int32_t *IF_RESTRICT hcx, const int32_t *IF_RESTRICT hcy,
                                int cellstride, int64_t *IF_RESTRICT rowbase)
{
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::CappedTag<int32_t, SIMD_CH> d;
  const hn::Rebind<int16_t, decltype(d)> d16;
  const hn::Rebind<uint8_t, decltype(d)> d8;
  const hn::Repartition<int64_t, decltype(d)> d64;
  const auto zero = hn::Zero(d);
  const int HL = (int)hn::Lanes(d64);

  auto vs_at = [&](int col) {
    return hn::PromoteTo(d, hn::LoadU(d16, vsb + (size_t)col * strideC + ch0));
  };
  auto vd_at = [&](int col) {
    return hn::PromoteTo(d, hn::LoadU(d16, vdb + (size_t)col * strideC + ch0));
  };
  auto raw_at = [&](const uint8_t *row, int col) {
    return hn::PromoteTo(d, hn::LoadU(d8, row + (size_t)col * strideC + ch0));
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
    const int c0 = e * cw, c1 = c0 + cw;
    for (int c = c0; c < c1; c += sx)
    {
      const int cm = clampi(c - 1, hi_c), cp = clampi(c + 1, hi_c);
      const auto gx = hn::Sub(vs_at(cp), vs_at(cm));
      const auto vdc = vd_at(c);
      const auto gy = hn::Add(hn::Add(vd_at(cm), hn::Add(vdc, vdc)), vd_at(cp));
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
      const auto g2 = hn::Add(gx2, gy2);
      hn::StoreU(b, d, blane);
      hn::StoreU(g2, d, g2lane);
      for (int l = 0; l < n; ++l)
        if (g2lane[l])
          hog[l * HB + blane[l]] += (int64_t)g2lane[l];
      // raw power sums: v <= 255 so v*v fits int32; the cubes/quartics are
      // accumulated in int64 lanes (v^4 = 4.2e9 already overflows int32).
      const auto v = raw_at(r1, c);
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
      // strict 8-neighbour extrema; load each neighbour once, reduce as a tree
      const auto p00 = raw_at(r0, cm), p01 = raw_at(r0, c), p02 = raw_at(r0, cp);
      const auto p10 = raw_at(r1, cm), p12 = raw_at(r1, cp);
      const auto p20 = raw_at(r2, cm), p21 = raw_at(r2, c), p22 = raw_at(r2, cp);
      const auto mx = hn::Max(hn::Max(hn::Max(p00, p01), hn::Max(p02, p10)),
                              hn::Max(hn::Max(p12, p20), hn::Max(p21, p22)));
      const auto mn = hn::Min(hn::Min(hn::Min(p00, p01), hn::Min(p02, p10)),
                              hn::Min(hn::Min(p12, p20), hn::Min(p21, p22)));
      nmaxv = hn::Sub(nmaxv, hn::VecFromMask(d, hn::Gt(v, mx)));
      nminv = hn::Sub(nminv, hn::VecFromMask(d, hn::Lt(v, mn)));
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
  bool ident_ = true; // channels are 0..c_-1 (the whole image, in order)
  std::vector<int> chan_;
  std::vector<Level> levels_;
  std::vector<int> row_cell_;
  std::vector<int64_t> coff_;            // per-selected-channel byte offset
  std::vector<int16_t> vs_, vd_;         // per-row planar Sobel caches (scalar path), c_*w each
  std::vector<int16_t> vsi_, vdi_;       // interleaved padded Sobel caches (SIMD path)
  std::vector<uint8_t> ri0_, ri1_, ri2_; // interleaved padded raw rows (SIMD path)
  int32_t hcx_[HB - 1] = {}, hcy_[HB - 1] = {};
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
  // intermediate exact in int64 and kills the cancellation that S2/n - mu^2
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

  // One channel's column run within one finest cell -> its accumulator slot.
  void accumulate_run(const int16_t *IF_RESTRICT vs, const int16_t *IF_RESTRICT vd,
                      const uint8_t *IF_RESTRICT r0, const uint8_t *IF_RESTRICT r1,
                      const uint8_t *IF_RESTRICT r2, int64_t cs, int c0, int c1, int hi_c,
                      int64_t *IF_RESTRICT acc) const
  {
    int64_t axx = 0, ayy = 0, axy = 0, acnt = 0, nmax = 0, nmin = 0;
    int64_t hog[HB] = {0}, ps[NMOM] = {0};
    for (int c = c0; c < c1; c += sx_)
    {
      const int cm = clampi(c - 1, hi_c), cp = clampi(c + 1, hi_c);
      const int gx = vs[cp] - vs[cm];
      const int gy = vd[cm] + 2 * vd[c] + vd[cp];
      const int64_t g2 = (int64_t)gx * gx + (int64_t)gy * gy;
      axx += (int64_t)gx * gx;
      ayy += (int64_t)gy * gy;
      axy += (int64_t)gx * gy;
      ++acnt;
      int qx = gx, qy = gy;
      if (qy < 0 || (qy == 0 && qx < 0))
      {
        qx = -qx;
        qy = -qy;
      }
      if (g2)
      {
        int b = 0;
        for (int j = 0; j < HB - 1; ++j)
          b += (hcx_[j] * qy - hcy_[j] * qx) >= 0;
        hog[b] += g2;
      }
      const int64_t km = (int64_t)cm * cs, kc = (int64_t)c * cs, kp = (int64_t)cp * cs;
      const int v = r1[kc];
      const int64_t x = v, x2 = x * x;
      ps[0] += x;
      ps[1] += x2;
      ps[2] += x2 * x;
      ps[3] += x2 * x2;
      const int nb[8] = {r0[km], r0[kc], r0[kp], r1[km], r1[kp], r2[km], r2[kc], r2[kp]};
      int mx = nb[0], mn = nb[0];
      for (int t = 1; t < 8; ++t)
      {
        mx = nb[t] > mx ? nb[t] : mx;
        mn = nb[t] < mn ? nb[t] : mn;
      }
      nmax += v > mx;
      nmin += v < mn;
    }
    acc[SXX] += axx;
    acc[SYY] += ayy;
    acc[SXY] += axy;
    acc[CNT] += acnt;
    for (int b = 0; b < HB; ++b)
      acc[HOG0 + b] += hog[b];
    acc[NMAX] += nmax;
    acc[NMIN] += nmin;
    for (int k = 0; k < NMOM; ++k)
      acc[PS0 + k] += ps[k];
  }

  static int simd_lanes()
  {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::CappedTag<int32_t, SIMD_CH> d;
    return (int)hn::Lanes(d);
  }

  void accumulate(const uint8_t *IF_RESTRICT img, int64_t rs, int64_t cs, int64_t chs)
  {
    const int hi_r = h_ - 1, hi_c = w_ - 1;
    Level &fine = levels_[0];
    const int nfx = fine.nx, cw = w_ / nfx;
    int64_t *IF_RESTRICT fb = fine.buf.data();
    std::fill(fine.buf.begin(), fine.buf.end(), (int64_t)0);

    for (int k = 0; k < c_; ++k)
      coff_[k] = (int64_t)chan_[k] * chs;
    // channels-in-lanes needs the selected channels gathered into interleaved
    // rows first; that gather is cheap, so SIMD applies for any layout.
    const bool simd = c_ >= 2;
    const int L = simd ? simd_lanes() : 1;
    const bool memcpy_row = ident_ && chs == 1 && cs == c_;

    for (int r = 0; r < h_; r += sy_)
    {
      const uint8_t *IF_RESTRICT b0 = img + (int64_t)clampi(r - 1, hi_r) * rs;
      const uint8_t *IF_RESTRICT b1 = img + (int64_t)r * rs;
      const uint8_t *IF_RESTRICT b2 = img + (int64_t)clampi(r + 1, hi_r) * rs;
      if (simd)
      { // interleaved, padded raw rows -> flat (auto-vectorizable) vertical Sobel
        auto fill = [&](uint8_t *IF_RESTRICT dst, const uint8_t *IF_RESTRICT src) {
          if (memcpy_row)
            std::memcpy(dst, src, (size_t)w_ * c_); // contiguous HWC row: already interleaved
          else
            for (int c = 0; c < w_; ++c)
              for (int k = 0; k < c_; ++k)
                dst[(size_t)c * c_ + k] = src[(int64_t)c * cs + coff_[k]];
        };
        fill(ri0_.data(), b0);
        fill(ri1_.data(), b1);
        fill(ri2_.data(), b2);
        const int WC = w_ * c_;
        for (int o = 0; o < WC; ++o)
        {
          vsi_[o] = (int16_t)(ri0_[o] + 2 * ri1_[o] + ri2_[o]);
          vdi_[o] = (int16_t)(ri2_[o] - ri0_[o]);
        }
      }
      else
      { // planar vertical Sobel cache: vs_[k * w_ + c]
        for (int k = 0; k < c_; ++k)
        {
          const int64_t off = coff_[k];
          int16_t *IF_RESTRICT vs = vs_.data() + (size_t)k * w_;
          int16_t *IF_RESTRICT vd = vd_.data() + (size_t)k * w_;
          for (int c = 0; c < w_; ++c)
          {
            const int64_t j = (int64_t)c * cs + off;
            vs[c] = (int16_t)(b0[j] + 2 * b1[j] + b2[j]);
            vd[c] = (int16_t)(b2[j] - b0[j]);
          }
        }
      }
      int64_t *IF_RESTRICT rb = fb + (size_t)(row_cell_[r] * nfx) * c_ * NSUM;
      if (simd)
        for (int ch0 = 0; ch0 < c_; ch0 += L)
          accumulate_row_simd(vsi_.data(), vdi_.data(), ri0_.data(), ri1_.data(), ri2_.data(),
                              c_, ch0, cw, nfx, hi_c, std::min(L, c_ - ch0), sx_, hcx_, hcy_,
                              c_ * NSUM, rb + (size_t)ch0 * NSUM);
      else
        for (int e = 0; e < nfx; ++e)
        {
          const int c0 = e * cw, c1 = c0 + cw;
          for (int k = 0; k < c_; ++k)
          {
            const int64_t off = coff_[k];
            accumulate_run(vs_.data() + (size_t)k * w_, vd_.data() + (size_t)k * w_, b0 + off,
                           b1 + off, b2 + off, cs, c0, c1, hi_c,
                           rb + (size_t)(e * c_ + k) * NSUM);
          }
        }
    }

    for (size_t lvl = 1; lvl < levels_.size(); ++lvl)
    {
      const Level &p = levels_[lvl - 1];
      Level &lv = levels_[lvl];
      const int64_t *IF_RESTRICT pb = p.buf.data();
      int64_t *IF_RESTRICT lb = lv.buf.data();
      const int pnx = p.nx, stripe = c_ * NSUM;
      std::fill(lv.buf.begin(), lv.buf.end(), (int64_t)0);
      for (int i = 0; i < p.ny; ++i)
      {
        const int ci = (i / lv.fy) * lv.nx;
        for (int j = 0; j < pnx; ++j)
        {
          const int64_t *IF_RESTRICT sc = pb + ((size_t)i * pnx + j) * stripe;
          int64_t *IF_RESTRICT dc = lb + ((size_t)(ci + j / lv.fx)) * stripe;
          for (int t = 0; t < stripe; ++t)
            dc[t] += sc[t];
        }
      }
    }

    {
      const Level &last = levels_.back();
      const int64_t *IF_RESTRICT lb = last.buf.data();
      const size_t nc = (size_t)last.ny * last.nx;
      const int stripe = c_ * NSUM;
      std::fill(graw_.begin(), graw_.end(), (int64_t)0);
      for (size_t cell = 0; cell < nc; ++cell)
      {
        const int64_t *IF_RESTRICT s = lb + cell * stripe;
        for (int t = 0; t < stripe; ++t)
          graw_[t] += s[t];
      }
    }
  }

  void derive()
  {
    for (Level &L : levels_)
    {
      const int64_t *IF_RESTRICT b = L.buf.data();
      float *IF_RESTRICT f = L.feat.data();
      double *IF_RESTRICT m = L.mom.data();
      const size_t nc = (size_t)L.ny * L.nx * c_;
      for (size_t cell = 0; cell < nc; ++cell)
      {
        derive_cell(b + cell * NSUM, f + cell * NF);
        derive_moments(b + cell * NSUM, m + cell * NMOM);
      }
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
    ident_ = (int64_t)c_ == (dims.size() > 2 ? dims[2] : 1);
    for (int k = 0; k < c_ && ident_; ++k)
      ident_ = chan_[k] == k;

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

    const Level &f = levels_[0];
    row_cell_.resize(h_);
    for (int r = 0; r < h_; ++r)
      row_cell_[r] = (int)((int64_t)r * f.ny / h_);
    coff_.assign(c_, 0);
    vs_.resize((size_t)c_ * w_);
    vd_.resize((size_t)c_ * w_);
    // interleaved SIMD buffers, padded so a full-vector LoadU past the last
    // channel of the last column stays in-bounds (the extra lanes are ignored).
    const size_t wcp = (size_t)c_ * w_ + SIMD_PAD;
    vsi_.assign(wcp, 0);
    vdi_.assign(wcp, 0);
    ri0_.assign(wcp, 0);
    ri1_.assign(wcp, 0);
    ri2_.assign(wcp, 0);
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
    nb::list out, mom;
    for (Level &L : levels_)
    {
      out.append(nb::ndarray<nb::numpy, float>(L.feat.data(), L.fshape.size(), L.fshape.data(),
                                               nb::handle()));
      mom.append(nb::ndarray<nb::numpy, double>(L.mom.data(), L.mshape.size(), L.mshape.data(),
                                                nb::handle()));
    }
    out.append(nb::ndarray<nb::numpy, float>(gfeat_.data(), gfeat_shape_.size(),
                                             gfeat_shape_.data(), nb::handle()));
    mom.append(nb::ndarray<nb::numpy, double>(gmom_.data(), gmom_shape_.size(),
                                              gmom_shape_.data(), nb::handle()));
    nb::list both;
    both.append(out);
    both.append(mom);
    return both;
  }
};

using Arr = nb::ndarray<nb::numpy, const uint8_t, nb::device::cpu>;

void strides_of(const Arr &a, int64_t &rs, int64_t &cs, int64_t &chs)
{
  rs = a.stride(0);
  cs = a.stride(1);
  chs = a.ndim() > 2 ? a.stride(2) : 0;
}
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
            int64_t rs, cs, chs;
            strides_of(a, rs, cs, chs);
            return self.raw(a.data(), rs, cs, chs);
          },
          nb::arg("arr"))
      .def(
          "features",
          [](FeatureComputer &self, Arr a) {
            int64_t rs, cs, chs;
            strides_of(a, rs, cs, chs);
            return self.features(a.data(), rs, cs, chs);
          },
          nb::arg("arr"));
}
