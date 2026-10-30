// The LBP ring's row body on L1 rows (32 px a row, 3 rows of taps). mode: 0 as the kernel
// (nb >= c as a saturating subtract and a compare against zero); 1 the test as a signed
// compare of bias-flipped bytes (the flip on any pipe); 2 as 1 with the bin's select as an
// OR of the sign bit (pshufb returns 0 for an index with bit 7 set) instead of a blend
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
int main(int argc, char **argv)
{
  const int mode = atoi(argv[1]);
  const int rows = 16, reps = 1000000, RS = 1088;
  alignas(64) static uint8_t img[(16 + 2) * 1088];
  for (int i = 0; i < (16 + 2) * 1088; ++i) img[i] = (uint8_t)((i * 7919) % 251);
  __m256i nib[5], lbp[9], umin = _mm256_setzero_si256(), tbl[5];
  for (int k = 0; k < 5; ++k) { nib[k] = _mm256_setzero_si256(); uint8_t t[16] = {}; t[2 * k] = 1; if (2 * k + 1 < 9) t[2 * k + 1] = 16; tbl[k] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)t)); }
  for (int j = 0; j < 9; ++j) lbp[j] = _mm256_setzero_si256();
  const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi8(1), bias = _mm256_set1_epi8((char)0x80), m3 = _mm256_set1_epi8(-3), nine = _mm256_set1_epi8(9), hi = _mm256_set1_epi8((char)0x80);
  for (int r = 0; r < reps; ++r)
  {
    const uint8_t *rb = img + RS + 32;
    int left = 15;
    for (int q = 0; q < rows; ++q, rb += RS)
    {
      const uint8_t *ra = rb - RS, *re = rb + RS;
      const __m256i c = _mm256_loadu_si256((const __m256i *)rb), cb = _mm256_xor_si256(c, bias);
      __m256i mn = _mm256_loadu_si256((const __m256i *)(ra - 1));
      auto ge = [&](__m256i nb) {
        if (mode == 0) return _mm256_cmpeq_epi8(_mm256_subs_epu8(c, nb), zero);
        return _mm256_xor_si256(_mm256_cmpgt_epi8(cb, _mm256_xor_si256(nb, bias)), _mm256_set1_epi8(-1)); // placeholder, fixed below
      };
      // the biased form proper: si' = (cb > nb') is NOT(nb >= c); sums of complemented masks fix up after the ring
      auto gt = [&](__m256i nb) { return _mm256_cmpgt_epi8(cb, _mm256_xor_si256(nb, bias)); };
      __m256i s0, prev, npc, ntr = zero;
      if (mode == 0) { s0 = ge(mn); } else { s0 = gt(mn); }
      prev = s0; npc = s0;
      auto next = [&](const uint8_t *p) {
        const __m256i nb = _mm256_loadu_si256((const __m256i *)p);
        const __m256i si = mode == 0 ? ge(nb) : gt(nb);
        mn = _mm256_min_epu8(mn, nb);
        npc = _mm256_add_epi8(npc, si);
        ntr = _mm256_add_epi8(ntr, _mm256_xor_si256(prev, si));
        prev = si;
      };
      next(ra), next(ra + 1), next(rb + 1), next(re + 1), next(re), next(re - 1), next(rb - 1);
      ntr = _mm256_add_epi8(ntr, _mm256_xor_si256(prev, s0));
      umin = _mm256_add_epi8(umin, _mm256_min_epu8(_mm256_subs_epu8(mn, c), one));
      const __m256i uniform = _mm256_cmpgt_epi8(ntr, m3);
      // mode 0: npc = sum of (nb >= c) masks = -popcount. mode 1/2: npc' = sum of NOT masks = -8 - npc, so -npc = 8 + npc'
      const __m256i cnt = mode == 0 ? _mm256_sub_epi8(zero, npc) : _mm256_add_epi8(npc, _mm256_set1_epi8(8));
      const __m256i bin = mode == 2 ? _mm256_or_si256(cnt, _mm256_andnot_si256(uniform, hi)) : _mm256_blendv_epi8(nine, cnt, uniform);
      for (int k = 0; k < 5; ++k) nib[k] = _mm256_add_epi8(nib[k], _mm256_shuffle_epi8(tbl[k], bin));
      if (--left == 0) { for (int k = 0; k < 5; ++k) { lbp[2 * k] = _mm256_add_epi8(lbp[2 * k], _mm256_and_si256(nib[k], _mm256_set1_epi8(15))); if (2 * k + 1 < 9) lbp[2 * k + 1] = _mm256_add_epi8(lbp[2 * k + 1], _mm256_and_si256(_mm256_srli_epi16(nib[k], 4), _mm256_set1_epi8(15))); nib[k] = zero; } left = 15; }
    }
    asm volatile("" : "+x"(lbp[0]), "+x"(lbp[1]), "+x"(lbp[2]), "+x"(lbp[3]), "+x"(lbp[4]), "+x"(lbp[5]), "+x"(lbp[6]), "+x"(lbp[7]), "+x"(lbp[8]), "+x"(umin), "+x"(nib[0]), "+x"(nib[1]), "+x"(nib[2]), "+x"(nib[3]), "+x"(nib[4]));
  }
  long long chk = 0; for (int j = 0; j < 9; ++j) chk += _mm256_extract_epi8(lbp[j], j) + _mm256_extract_epi8(nib[j % 5], j); chk += _mm256_extract_epi8(umin, 3);
  printf("mode %d: chk %lld (divide cycles by %d rows)\n", mode, chk, reps * rows);
}
