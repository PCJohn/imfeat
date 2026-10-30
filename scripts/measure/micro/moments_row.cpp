// The moments loop body (the Laplacian and the moments of w = v - KBIAS, two rows an
// iteration) on an L1 buffer laid out as the kernel's `resp` slots (RESP_STRIDE 8 vectors
// groups of 16 lanes, rstep one row). mode: 0 as the kernel; 1 one row an iteration (widened
// every row); 2 without the fourth power (no w2*w2, no widen); 3 the moments only (no
// Laplacian); 4 the Laplacian only; 5 as 0 with the fourth power's pair sums kept 32-bit for
// four rows (NOT exact: a bound check only); 6 as the kernel but the M responses taken from
// the pixel rows (byte loads widened by an interleave with zero) instead of the resp slots
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
int main(int argc, char **argv)
{
  const int mode = atoi(argv[1]);
  const int rows = 32, reps = 1000000, RSTEP = 128; // int16 elements a row (256 B)
  alignas(64) static int16_t buf[(32 + 3) * 128];
  alignas(64) static uint8_t img[(32 + 3) * 1088];
  for (int i = 0; i < (32 + 3) * 1088; ++i) img[i] = (uint8_t)((i * 7919) % 251);
  for (int i = 0; i < (32 + 3) * 128; ++i) buf[i] = (int16_t)((i * 7919) % 511 - 255);
  const __m256i kb = _mm256_set1_epi16(128);
  __m256i lap16 = _mm256_setzero_si256(), w16 = lap16, lap2 = lap16, t2 = lap16, t3 = lap16, t4 = lap16;
  auto ld = [&](const int16_t *rp, int i, int t) { return _mm256_loadu_si256((const __m256i *)(rp + (i * 8 + t * 2) * 16)); };
  auto madd = [](__m256i a, __m256i b) { return _mm256_madd_epi16(a, b); };
  const uint8_t *pr = img + 32;
  auto wide = [&](const uint8_t *p) { return _mm256_unpacklo_epi8(_mm256_loadu_si256((const __m256i *)p), _mm256_setzero_si256()); };
  auto row = [&](const int16_t *rp) {
    const __m256i Sb = ld(rp, 1, 1);
    const __m256i Ma = mode == 6 ? wide(pr) : ld(rp, 0, 3), Mb = mode == 6 ? wide(pr + 1088) : ld(rp, 1, 3), Me = mode == 6 ? wide(pr + 2 * 1088) : ld(rp, 2, 3);
    pr += 1088;
    if (mode != 3)
    {
      const __m256i lap = _mm256_sub_epi16(_mm256_add_epi16(Sb, _mm256_add_epi16(Ma, Me)), _mm256_add_epi16(Mb, Mb));
      lap16 = _mm256_add_epi16(lap16, lap);
      lap2 = _mm256_add_epi32(lap2, madd(lap, lap));
    }
    if (mode == 4) return _mm256_setzero_si256();
    const __m256i wc = _mm256_sub_epi16(Mb, kb), w2 = _mm256_mullo_epi16(wc, wc);
    w16 = _mm256_add_epi16(w16, wc);
    t2 = _mm256_add_epi32(t2, madd(wc, wc));
    t3 = _mm256_add_epi32(t3, madd(w2, wc));
    return mode == 2 ? _mm256_setzero_si256() : madd(w2, w2);
  };
  auto widen = [&](__m256i p) {
    const __m256i lo = _mm256_and_si256(p, _mm256_set1_epi64x(0xffffffff)), hi = _mm256_srli_epi64(p, 32);
    t4 = _mm256_add_epi64(t4, _mm256_add_epi64(lo, hi));
  };
  for (int r = 0; r < reps; ++r)
  {
    const int16_t *rp = buf;
    pr = img + 32;
    if (mode == 1)
      for (int q = 0; q < rows; ++q, rp += RSTEP) widen(row(rp));
    else if (mode == 5)
      for (int q = 0; q < rows; q += 4, rp += 4 * RSTEP)
      {
        const __m256i p0 = row(rp), p1 = row(rp + RSTEP), p2 = row(rp + 2 * RSTEP), p3 = row(rp + 3 * RSTEP);
        widen(_mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3)));
      }
    else
      for (int q = 0; q < rows; q += 2, rp += 2 * RSTEP)
      {
        const __m256i p0 = row(rp);
        const __m256i p = _mm256_add_epi32(p0, row(rp + RSTEP));
        if (mode != 2) widen(p);
      }
    asm volatile("" : "+x"(lap16), "+x"(w16), "+x"(lap2), "+x"(t2), "+x"(t3), "+x"(t4));
  }
  long long chk = _mm256_extract_epi64(t4, 0) + _mm256_extract_epi32(lap2, 1) + _mm256_extract_epi32(t3, 2) + _mm256_extract_epi16(w16, 3) + _mm256_extract_epi16(lap16, 4) + _mm256_extract_epi32(t2, 5);
  printf("mode %d: chk %lld (divide cycles by %d rows)\n", mode, chk, reps * rows);
}
