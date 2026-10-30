// The LBP ring's row body on L1 rows (32 px a row, 64 rows of taps), the row loop kept to one
// row an iteration like the kernel's (the earlier copy, fully unrolled by GCC and spilled all
// over, ran 30 cycles a row for the kernel's 24). Build: g++ -O3 -march=native -DMODE=n.
// MODE 0: as the kernel (serial chains for the minimum, the popcount and the transitions);
// 1: the popcount and transition sums as trees; 2: as 1 with the minimum as a tree too;
// 3: two rows' bodies an iteration; 4: the four other vote tables as byte shifts of the first
// and the local-minimum count by a compare with zero; 5: as 4 with the ring serialized by
// asm barriers after each neighbour (GCC otherwise tests all eight first); 6: the kernel's
// form with the barriers only. Measured October 30, 2026 (OPTIMIZATION.md, "the pipe
// assignment, after step 36"): 0 25.2 cycles a row, 1-2 no change, 3 25.1, 5 23.6, 6 22.9;
// in the kernel the barriers (mode 6) made the ring 13% slower and the shifted tables 1%.
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#ifndef MODE
#define MODE 0
#endif
int main(int argc, char **argv)
{
  const int rows = 64, reps = 250000, RS = 1088;
  alignas(64) static uint8_t img[(64 + 2) * 1088];
  for (int i = 0; i < (64 + 2) * 1088; ++i) img[i] = (uint8_t)((i * 7919) % 251);
  __m256i nib[5], lbp[9], umin = _mm256_setzero_si256(), tbl[5];
  for (int k = 0; k < 5; ++k) { nib[k] = _mm256_setzero_si256(); uint8_t t[16] = {}; t[2 * k] = 1; if (2 * k + 1 < 9) t[2 * k + 1] = 16; tbl[k] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)t)); }
  for (int j = 0; j < 9; ++j) lbp[j] = _mm256_setzero_si256();
  const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi8(1), m3 = _mm256_set1_epi8(-3), nine = _mm256_set1_epi8(9);
  auto rowbody = [&](const uint8_t *rb) __attribute__((always_inline)) {
    const uint8_t *ra = rb - RS, *re = rb + RS;
    const __m256i c = _mm256_loadu_si256((const __m256i *)rb);
    auto ge = [&](const uint8_t *p) __attribute__((always_inline)) { return _mm256_cmpeq_epi8(_mm256_subs_epu8(c, _mm256_loadu_si256((const __m256i *)p)), zero); };
    auto ld = [&](const uint8_t *p) { return _mm256_loadu_si256((const __m256i *)p); };
    const uint8_t *taps[8] = {ra - 1, ra, ra + 1, rb + 1, re + 1, re, re - 1, rb - 1};
    __m256i s[8], npc, ntr, mn;
    if (MODE == 5 || MODE == 6) s[0] = ge(taps[0]); else for (int i = 0; i < 8; ++i) s[i] = ge(taps[i]);
    if (MODE == 0 || MODE == 4) {
      mn = ld(taps[0]); npc = s[0]; ntr = zero;
      for (int i = 1; i < 8; ++i) { mn = _mm256_min_epu8(mn, ld(taps[i])); npc = _mm256_add_epi8(npc, s[i]); ntr = _mm256_add_epi8(ntr, _mm256_xor_si256(s[i - 1], s[i])); }
      ntr = _mm256_add_epi8(ntr, _mm256_xor_si256(s[7], s[0]));
    } else if (MODE == 5 || MODE == 6) {
      mn = ld(taps[0]); npc = s[0]; ntr = zero;
      for (int i = 1; i < 8; ++i) { const __m256i nb = ld(taps[i]); const __m256i si = _mm256_cmpeq_epi8(_mm256_subs_epu8(c, nb), zero); mn = _mm256_min_epu8(mn, nb); npc = _mm256_add_epi8(npc, si); ntr = _mm256_add_epi8(ntr, _mm256_xor_si256(s[i - 1], si)); s[i] = si; asm volatile("" : "+x"(npc), "+x"(ntr), "+x"(mn)); }
      ntr = _mm256_add_epi8(ntr, _mm256_xor_si256(s[7], s[0]));
    } else {
      npc = _mm256_add_epi8(_mm256_add_epi8(_mm256_add_epi8(s[0], s[1]), _mm256_add_epi8(s[2], s[3])), _mm256_add_epi8(_mm256_add_epi8(s[4], s[5]), _mm256_add_epi8(s[6], s[7])));
      __m256i x[8]; for (int i = 0; i < 8; ++i) x[i] = _mm256_xor_si256(s[i], s[(i + 1) & 7]);
      ntr = _mm256_add_epi8(_mm256_add_epi8(_mm256_add_epi8(x[0], x[1]), _mm256_add_epi8(x[2], x[3])), _mm256_add_epi8(_mm256_add_epi8(x[4], x[5]), _mm256_add_epi8(x[6], x[7])));
      if (MODE == 1) { mn = ld(taps[0]); for (int i = 1; i < 8; ++i) mn = _mm256_min_epu8(mn, ld(taps[i])); }
      else mn = _mm256_min_epu8(_mm256_min_epu8(_mm256_min_epu8(ld(taps[0]), ld(taps[1])), _mm256_min_epu8(ld(taps[2]), ld(taps[3]))), _mm256_min_epu8(_mm256_min_epu8(ld(taps[4]), ld(taps[5])), _mm256_min_epu8(ld(taps[6]), ld(taps[7]))));
    }
    if (MODE == 4 || MODE == 5) umin = _mm256_sub_epi8(umin, _mm256_cmpeq_epi8(_mm256_subs_epu8(mn, c), zero));
    else umin = _mm256_add_epi8(umin, _mm256_min_epu8(_mm256_subs_epu8(mn, c), one));
    const __m256i uniform = _mm256_cmpgt_epi8(ntr, m3);
    const __m256i bin = _mm256_blendv_epi8(nine, _mm256_sub_epi8(zero, npc), uniform);
    if (MODE == 4 || MODE == 5) {
      nib[0] = _mm256_add_epi8(nib[0], _mm256_shuffle_epi8(tbl[0], bin));
      nib[1] = _mm256_add_epi8(nib[1], _mm256_shuffle_epi8(_mm256_slli_si256(tbl[0], 2), bin));
      nib[2] = _mm256_add_epi8(nib[2], _mm256_shuffle_epi8(_mm256_slli_si256(tbl[0], 4), bin));
      nib[3] = _mm256_add_epi8(nib[3], _mm256_shuffle_epi8(_mm256_slli_si256(tbl[0], 6), bin));
      nib[4] = _mm256_add_epi8(nib[4], _mm256_shuffle_epi8(_mm256_slli_si256(tbl[0], 8), bin));
    } else
    for (int k = 0; k < 5; ++k) nib[k] = _mm256_add_epi8(nib[k], _mm256_shuffle_epi8(tbl[k], bin));
  };
  for (int r = 0; r < reps; ++r)
  {
    const uint8_t *rb = img + RS + 32;
    int left = 15;
    if (MODE == 3)
      _Pragma("GCC unroll 1") for (int q = 0; q < rows; q += 2, rb += 2 * RS) { rowbody(rb); rowbody(rb + RS); left -= 2; if (left <= 0) { for (int k = 0; k < 5; ++k) { lbp[2 * k] = _mm256_add_epi8(lbp[2 * k], _mm256_and_si256(nib[k], _mm256_set1_epi8(15))); if (2 * k + 1 < 9) lbp[2 * k + 1] = _mm256_add_epi8(lbp[2 * k + 1], _mm256_and_si256(_mm256_srli_epi16(nib[k], 4), _mm256_set1_epi8(15))); nib[k] = zero; } left = 15; } }
    else
      _Pragma("GCC unroll 1") for (int q = 0; q < rows; ++q, rb += RS) { rowbody(rb); if (--left == 0) { for (int k = 0; k < 5; ++k) { lbp[2 * k] = _mm256_add_epi8(lbp[2 * k], _mm256_and_si256(nib[k], _mm256_set1_epi8(15))); if (2 * k + 1 < 9) lbp[2 * k + 1] = _mm256_add_epi8(lbp[2 * k + 1], _mm256_and_si256(_mm256_srli_epi16(nib[k], 4), _mm256_set1_epi8(15))); nib[k] = zero; } left = 15; } }
    asm volatile("" : "+x"(lbp[0]), "+x"(lbp[1]), "+x"(lbp[2]), "+x"(lbp[3]), "+x"(lbp[4]), "+x"(lbp[5]), "+x"(lbp[6]), "+x"(lbp[7]), "+x"(lbp[8]), "+x"(umin), "+x"(nib[0]), "+x"(nib[1]), "+x"(nib[2]), "+x"(nib[3]), "+x"(nib[4]));
  }
  long long chk = 0; for (int j = 0; j < 9; ++j) chk += _mm256_extract_epi8(lbp[j], j) + _mm256_extract_epi8(nib[j % 5], j); chk += _mm256_extract_epi8(umin, 3);
  printf("mode %d: chk %lld (divide cycles by %d rows)\n", MODE, chk, reps * rows);
}
