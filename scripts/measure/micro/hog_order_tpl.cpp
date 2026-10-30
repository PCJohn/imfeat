// The loop hog_order.py fills in (@ASM@ the row body, @TAG@ a label); not built on its own.
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
int main(int argc, char **argv)
{
  const int rows = 16, reps = 2000000;
  alignas(64) static int16_t hb[64 * 32];
  alignas(64) static int16_t rt[4 * 16 + 32];
  alignas(64) static int64_t sg[4];
  for (int i = 0; i < 64 * 32; ++i) hb[i] = (int16_t)((i * 7919) % 2000 - 1000);
  for (int j = 0; j < 4; ++j) for (int i = 0; i < 16; ++i) rt[j * 16 + i] = (int16_t)((i & 1) ? 100 + 37 * j : -200 + 13 * j);
  asm volatile("vpxor %%ymm8,%%ymm8,%%ymm8\n\tvpxor %%ymm9,%%ymm9,%%ymm9\n\tvpxor %%ymm10,%%ymm10,%%ymm10\n\tvpxor %%ymm11,%%ymm11,%%ymm11\n\tvpxor %%ymm12,%%ymm12,%%ymm12\n\tvpxor %%ymm13,%%ymm13,%%ymm13\n\tvpxor %%ymm14,%%ymm14,%%ymm14\n\tvpxor %%ymm15,%%ymm15,%%ymm15" ::: "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
  for (int r = 0; r < reps; ++r)
    for (int q = 0; q < rows; ++q)
      asm volatile("@ASM@" :: [hb] "r"(hb + q * 64), [rt] "r"(rt), [sg] "r"(sg) : "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","memory");
  int32_t out[8]; asm volatile("vmovdqu %%ymm8, (%0)" :: "r"(out) : "memory");
  printf("@TAG@: chk %d %lld\n", out[0], (long long)sg[0]);
}
