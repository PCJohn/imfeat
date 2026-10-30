// Throughput and pipe assignment of one vector instruction on this core: 16 independent
// register chains in a counted loop; run under `perf stat -e cycles,fpu_pipe_assignment.total0..3`
// (AMD) and divide by 8e8 ops. Build: g++ -O2 -march=native -o pipes pipes.cpp; ./pipes CASE.
// How the per-pipe table of October 30, 2026 (OPTIMIZATION.md, "The kernel loops by pipe") was made.
#include <cstdio>
#include <cstdlib>
#define CH8(op, r) \
  asm volatile(op " %%" r "8,%%" r "0,%%" r "0\n\t" op " %%" r "8,%%" r "1,%%" r "1\n\t" op " %%" r "8,%%" r "2,%%" r "2\n\t" op " %%" r "8,%%" r "3,%%" r "3\n\t" \
               op " %%" r "8,%%" r "4,%%" r "4\n\t" op " %%" r "8,%%" r "5,%%" r "5\n\t" op " %%" r "8,%%" r "6,%%" r "6\n\t" op " %%" r "8,%%" r "7,%%" r "7\n\t" \
               ::: "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7")
#define CH16(op, r) CH8(op, r); \
  asm volatile(op " %%" r "8,%%" r "9,%%" r "9\n\t" op " %%" r "8,%%" r "10,%%" r "10\n\t" op " %%" r "8,%%" r "11,%%" r "11\n\t" op " %%" r "8,%%" r "12,%%" r "12\n\t" \
               op " %%" r "8,%%" r "13,%%" r "13\n\t" op " %%" r "8,%%" r "14,%%" r "14\n\t" op " %%" r "8,%%" r "15,%%" r "15\n\t" op " %%" r "8,%%" r "9,%%" r "9\n\t" \
               ::: "xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15")
#define CH2x8(op, a, b) \
  asm volatile(op " %%" a "0,%%" b "0\n\t" op " %%" a "1,%%" b "1\n\t" op " %%" a "2,%%" b "2\n\t" op " %%" a "3,%%" b "3\n\t" \
               op " %%" a "4,%%" b "4\n\t" op " %%" a "5,%%" b "5\n\t" op " %%" a "6,%%" b "6\n\t" op " %%" a "7,%%" b "7\n\t" \
               op " %%" a "9,%%" b "9\n\t" op " %%" a "10,%%" b "10\n\t" op " %%" a "11,%%" b "11\n\t" op " %%" a "12,%%" b "12\n\t" \
               op " %%" a "13,%%" b "13\n\t" op " %%" a "14,%%" b "14\n\t" op " %%" a "15,%%" b "15\n\t" op " %%" a "9,%%" b "9\n\t" \
               ::: "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15")
int main(int argc, char **argv)
{
  const int which = atoi(argv[1]); const long n = 50000000;
  asm volatile("vpxor %%ymm8,%%ymm8,%%ymm8" ::: "xmm8");
  for (long i = 0; i < n; ++i)
    switch (which)
    {
    case 0: CH16("vpaddd", "ymm"); break;
    case 1: CH16("vpaddd", "xmm"); break;
    case 2: CH16("vaddps", "ymm"); break;
    case 3: CH16("vpsignd", "ymm"); break;
    case 4: CH16("vpmaddwd", "ymm"); break;
    case 5: CH16("vpand", "ymm"); break;
    case 6: CH16("vpor", "ymm"); break;
    case 7: CH16("vmulps", "ymm"); break;
    case 8: CH16("vpaddw", "ymm"); break;
    case 9: CH16("vpunpcklbw", "ymm"); break;
    case 10: CH16("vpsubw", "xmm"); break;
    case 11: CH16("vpcmpeqb", "ymm"); break;
    case 12: CH16("vpsubusb", "ymm"); break;
    case 13: CH16("vpmaxub", "ymm"); break;
    case 14: CH16("vpsadbw", "ymm"); break;
    case 15: CH16("vpcmpgtd", "ymm"); break;
    case 17: CH16("vpavgb", "ymm"); break;
    case 18: CH16("vpalignr $4,", "ymm"); break;
    case 19: CH16("vpblendvb %%ymm8,", "ymm"); break;
    case 20: CH16("vpackuswb", "ymm"); break;
    case 21: CH16("vpmulld", "ymm"); break;
    case 22: CH16("vpmullw", "ymm"); break;
    case 23: CH2x8("vpsrlw $1,", "ymm", "ymm"); break;
    case 24: CH2x8("vpsrad $1,", "ymm", "ymm"); break;
    case 25: CH2x8("vpabsw", "ymm", "ymm"); break;
    case 26: CH16("vpshufb", "ymm"); break;
    case 27: CH16("vpblendw $0x55,", "ymm"); break;
    case 28: CH16("vpandn", "ymm"); break;
    case 29: CH16("vpminsd", "ymm"); break;
    case 30: CH2x8("vcvtdq2ps", "ymm", "ymm"); break;
    case 31: CH2x8("vpermq $0x4e,", "ymm", "ymm"); break;
    case 32: CH2x8("vpmovzxbw", "xmm", "ymm"); break;
    case 33: CH2x8("vpsllw $1,", "ymm", "ymm"); break;
    case 34: CH16("vhaddps", "ymm"); break;
    case 35: CH2x8("vcvtps2dq", "ymm", "ymm"); break;
    case 36: CH2x8("vpshufd $0xf5,", "ymm", "ymm"); break;
    case 37: CH2x8("vextracti128 $1,", "ymm", "xmm"); break;
    case 38: CH16("vpunpcklwd", "ymm"); break;
    case 39: CH16("vpmuludq", "ymm"); break;
    case 40: CH16("vpsubusw", "ymm"); break;
    case 41: CH16("vpaddq", "ymm"); break;
    case 42: CH16("vpackssdw", "ymm"); break;
    case 43: CH16("vpcmpeqw", "ymm"); break;
    case 44: CH16("vdivps", "ymm"); break;
    case 45: CH2x8("vroundps $8,", "ymm", "ymm"); break;
    case 46: CH16("vmaxps", "ymm"); break;
    case 47: CH16("vfmadd213ps", "ymm"); break;
    case 48: CH16("vaddps", "ymm"); break;
    case 49: CH16("vsubps", "ymm"); break;
    case 50: CH2x8("vcvtps2pd", "xmm", "ymm"); break;
    case 51: CH16("vaddpd", "ymm"); break;
    case 52: CH16("vmulpd", "ymm"); break;
    case 53: CH2x8("vrcpps", "ymm", "ymm"); break;
    case 54: CH2x8("vsqrtps", "ymm", "ymm"); break;
    case 55: CH16("vfnmadd213ps", "ymm"); break;
    }
  return 0;
}
