// The responses row body: 3 loads at taps -1/0/+1, in-lane widening, 8 stores to a resp buffer.
// mode 0: as the kernel (planes rows 1088 B apart, resp 512 B per row); 1: resp stores at
// a fixed row (no sweep); 2: no stores (sum into registers); 3: aligned taps; 4: without the
// M response's store (the moments loop widening the pixel rows itself)
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
int main(int argc, char **argv)
{
  const int mode = atoi(argv[1]), rows = 18, reps = 1000000, RS = 1088;
  static uint8_t planes[64 * RS + 256] __attribute__((aligned(64)));
  static int16_t resp[18 * 256 + 64] __attribute__((aligned(64)));
  for (int i = 0; i < 64 * RS; ++i) planes[i] = (uint8_t)(i * 31 + (i >> 7));
  const __m256i zero = _mm256_setzero_si256();
  __m256i acc = zero;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
  {
    const int x0 = (r & 31) * 32; // the block's column, sweeping as in the kernel
    for (int j = 0; j < rows; ++j)
    {
      const uint8_t *row = planes + j * RS + 64 + x0;
      const __m256i l = _mm256_loadu_si256((const __m256i *)(row + (mode == 3 ? 0 : -1)));
      const __m256i m = _mm256_loadu_si256((const __m256i *)row);
      const __m256i rr = _mm256_loadu_si256((const __m256i *)(row + (mode == 3 ? 0 : 1)));
      for (int h = 0; h < 2; ++h)
      {
        const __m256i wl = h ? _mm256_unpackhi_epi8(l, zero) : _mm256_unpacklo_epi8(l, zero);
        const __m256i wm = h ? _mm256_unpackhi_epi8(m, zero) : _mm256_unpacklo_epi8(m, zero);
        const __m256i wr = h ? _mm256_unpackhi_epi8(rr, zero) : _mm256_unpacklo_epi8(rr, zero);
        const __m256i lr = _mm256_add_epi16(wl, wr), m2 = _mm256_add_epi16(wm, wm);
        int16_t *o = resp + ((mode == 1 ? 0 : j) * 8 + h) * 16;
        if (mode == 2)
          acc = _mm256_add_epi16(acc, _mm256_add_epi16(_mm256_add_epi16(_mm256_sub_epi16(wr, wl), _mm256_sub_epi16(lr, m2)), _mm256_add_epi16(_mm256_add_epi16(lr, m2), wm)));
        else
        {
          _mm256_storeu_si256((__m256i *)o, _mm256_sub_epi16(wr, wl));
          _mm256_storeu_si256((__m256i *)(o + 32), _mm256_sub_epi16(lr, m2));
          _mm256_storeu_si256((__m256i *)(o + 64), _mm256_add_epi16(lr, m2));
          if (mode != 4) _mm256_storeu_si256((__m256i *)(o + 96), wm);
        }
      }
    }
    asm volatile("" : "+x"(acc) :: "memory");
  }
  auto t1 = std::chrono::steady_clock::now();
  const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double)reps * rows);
  printf("mode %d: %.2f ns/row (%.1f cycles at 4.2 GHz) chk %d %d\n", mode, ns, ns * 4.2, resp[5], _mm256_extract_epi16(acc, 3));
}
