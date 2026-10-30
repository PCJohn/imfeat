// HOG row body variants on an L1 buffer. mode: 0 as is; 1 sg4 last; 2 two rows interleaved;
// 3 software-pipelined (next row's za/g2 before this row's rays); 4 sg4 via double FMA;
// 5 no sg4 at all; 6 rays only (no E, no sg4)
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
static inline __m256i sgn(__m256i a, __m256i b) { return _mm256_sign_epi32(b, a); }
static inline __m256i sq64(__m256i g2) { return _mm256_add_epi64(_mm256_mul_epi32(g2, g2), _mm256_mul_epi32(_mm256_srli_epi64(g2, 32), _mm256_srli_epi64(g2, 32))); }
int main(int argc, char **argv)
{
  const int mode = atoi(argv[1]);
  const int rows = 16, reps = 2000000;
  alignas(64) static int16_t hb[64 * 32];
  alignas(64) static int16_t rt[4 * 16 + 32];
  for (int i = 0; i < 64 * 32; ++i) hb[i] = (int16_t)((i * 7919) % 2000 - 1000);
  for (int j = 0; j < 4; ++j) for (int i = 0; i < 16; ++i) rt[j * 16 + i] = (int16_t)((i & 1) ? 100 + 37 * j : -200 + 13 * j);
  __m256i hog[9], sg4 = _mm256_setzero_si256();
  __m256d sd0 = _mm256_setzero_pd(), sd1 = _mm256_setzero_pd();
  for (int j = 0; j < 9; ++j) hog[j] = _mm256_setzero_si256();
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
  {
    auto R = [&](int j) { return _mm256_load_si256((const __m256i *)(rt + j * 16)); };
    if (mode == 0 || mode == 1 || mode == 4 || mode == 5 || mode == 6 || mode == 7 || mode == 8)
      for (int q = 0; q < rows; ++q)
      {
        const __m256i zl = _mm256_load_si256((const __m256i *)(hb + q * 64));
        const __m256i za = _mm256_abs_epi16(zl);
        const __m256i g2 = _mm256_madd_epi16(zl, zl);
        if (mode == 0) sg4 = _mm256_add_epi64(sg4, sq64(g2));
        const __m256i g2s = sgn(zl, g2);
        if (mode < 6) hog[8] = _mm256_add_epi32(hog[8], g2s);
        for (int j = 0; j < 4; ++j)
        {
          const __m256i t = mode == 8 ? _mm256_add_epi32(za, R(j)) : _mm256_madd_epi16(za, R(j));
          hog[j] = _mm256_add_epi32(hog[j], mode == 7 ? _mm256_add_epi32(t, g2) : sgn(t, g2));
          hog[4 + j] = _mm256_add_epi32(hog[4 + j], mode == 7 ? _mm256_add_epi32(t, g2s) : sgn(t, g2s));
        }
        if (mode == 1) sg4 = _mm256_add_epi64(sg4, sq64(g2));
        if (mode == 4)
        {
          const __m256d lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(g2)), hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(g2, 1));
          sd0 = _mm256_fmadd_pd(lo, lo, sd0); sd1 = _mm256_fmadd_pd(hi, hi, sd1);
        }
      }
    else if (mode == 2)
      for (int q = 0; q < rows; q += 2)
      {
        const __m256i zla = _mm256_load_si256((const __m256i *)(hb + q * 64)), zlb = _mm256_load_si256((const __m256i *)(hb + (q + 1) * 64));
        const __m256i zaa = _mm256_abs_epi16(zla), zab = _mm256_abs_epi16(zlb);
        const __m256i g2a = _mm256_madd_epi16(zla, zla), g2b = _mm256_madd_epi16(zlb, zlb);
        sg4 = _mm256_add_epi64(sg4, _mm256_add_epi64(sq64(g2a), sq64(g2b)));
        const __m256i g2sa = sgn(zla, g2a), g2sb = sgn(zlb, g2b);
        hog[8] = _mm256_add_epi32(hog[8], _mm256_add_epi32(g2sa, g2sb));
        for (int j = 0; j < 4; ++j)
        {
          const __m256i ta = _mm256_madd_epi16(zaa, R(j)), tb = _mm256_madd_epi16(zab, R(j));
          hog[j] = _mm256_add_epi32(hog[j], _mm256_add_epi32(sgn(ta, g2a), sgn(tb, g2b)));
          hog[4 + j] = _mm256_add_epi32(hog[4 + j], _mm256_add_epi32(sgn(ta, g2sa), sgn(tb, g2sb)));
        }
      }
    else if (mode == 3)
    {
      __m256i zl = _mm256_load_si256((const __m256i *)hb), za = _mm256_abs_epi16(zl), g2 = _mm256_madd_epi16(zl, zl);
      for (int q = 0; q < rows; ++q)
      {
        const __m256i zln = _mm256_load_si256((const __m256i *)(hb + (q + 1) * 64));
        const __m256i zan = _mm256_abs_epi16(zln), g2n = _mm256_madd_epi16(zln, zln);
        sg4 = _mm256_add_epi64(sg4, sq64(g2));
        const __m256i g2s = sgn(zl, g2);
        hog[8] = _mm256_add_epi32(hog[8], g2s);
        for (int j = 0; j < 4; ++j)
        {
          const __m256i t = _mm256_madd_epi16(za, R(j));
          hog[j] = _mm256_add_epi32(hog[j], sgn(t, g2));
          hog[4 + j] = _mm256_add_epi32(hog[4 + j], sgn(t, g2s));
        }
        zl = zln; za = zan; g2 = g2n;
      }
    }
    asm volatile("" : "+x"(hog[0]), "+x"(hog[1]), "+x"(hog[2]), "+x"(hog[3]), "+x"(hog[4]), "+x"(hog[5]), "+x"(hog[6]), "+x"(hog[7]), "+x"(hog[8]), "+x"(sg4), "+x"(sd0), "+x"(sd1));
  }
  auto t1 = std::chrono::steady_clock::now();
  const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double)reps * rows);
  int64_t chk = 0; for (int j = 0; j < 9; ++j) chk += _mm256_extract_epi32(hog[j], 0);
  printf("mode %d: %.2f ns/row (%.1f cycles at 4.2 GHz) chk %lld\n", mode, ns, ns * 4.2, (long long)chk);
}
