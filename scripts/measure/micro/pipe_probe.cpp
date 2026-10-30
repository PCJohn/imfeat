// How this core's FP dispatcher assigns multi-pipe uops next to pipe-0 multiplies: each case
// is one asm block of independent register chains, repeated 1e8 times; run under
// `perf stat -e cycles,fpu_pipe_assignment.total0..3` (AMD) and divide by 1e8 for the uops per
// block on each pipe. ymm0-3 are vpmaddwd chains (pipe 0 only), ymm4-11 vpaddd chains (pipes
// 0, 1, 3), ymm12-15 vpand chains (any pipe). Build: g++ -O2 -march=native; run ./pipe_probe N.
// Found October 30, 2026 (OPTIMIZATION.md, "the pipe assignment, after step 36"): the adds'
// share of pipe 0 is 50% in a stream of adds alone (case 0), 37% interleaved one multiply to
// two adds (1), 33% one to three (2), 29% one to four (6), 11% after a burst of four
// multiplies (3) and 0% after a burst of two (14): the assignment is stateful and
// order-dependent, so the same ops in another order land on other pipes. The cycle counts of
// these short blocks are bounded by their taken branches, not by the pipes.
#include <cstdlib>
#include <cstdio>
#define A(op, d) op " %%ymm" #d ",%%ymm" #d ",%%ymm" #d "\n\t"
#define C ::: "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15"
int main(int argc, char **argv)
{
  const int w = atoi(argv[1]);
  for (long i = 0; i < 100000000; ++i)
    switch (w)
    {
    case 0: asm volatile(A("vpaddd", 4) A("vpaddd", 5) A("vpaddd", 6) A("vpaddd", 7) A("vpaddd", 8) A("vpaddd", 9) C); break; // 6 adds
    case 1: asm volatile(A("vpmaddwd", 0) A("vpaddd", 4) A("vpaddd", 5) A("vpmaddwd", 1) A("vpaddd", 6) A("vpaddd", 7) C); break; // 1 madd : 2 adds
    case 2: asm volatile(A("vpmaddwd", 0) A("vpaddd", 4) A("vpaddd", 5) A("vpaddd", 6) A("vpmaddwd", 1) A("vpaddd", 7) A("vpaddd", 8) A("vpaddd", 9) C); break; // 1 : 3
    case 3: asm volatile(A("vpmaddwd", 0) A("vpmaddwd", 1) A("vpmaddwd", 2) A("vpmaddwd", 3) A("vpaddd", 4) A("vpaddd", 5) A("vpaddd", 6) A("vpaddd", 7) A("vpaddd", 8) A("vpaddd", 9) A("vpaddd", 10) A("vpaddd", 11) C); break; // 4 madds then 8 adds
    case 4: asm volatile(A("vpmaddwd", 0) A("vpand", 12) A("vpand", 13) A("vpmaddwd", 1) A("vpand", 14) A("vpand", 15) C); break; // 1 madd : 2 ands
    case 5: asm volatile(A("vpmaddwd", 0) A("vpaddd", 4) A("vpand", 12) A("vpaddd", 5) A("vpand", 13) A("vpmaddwd", 1) A("vpaddd", 6) A("vpand", 14) A("vpaddd", 7) A("vpand", 15) C); break; // 1 madd : 2 adds : 2 ands
    case 6: asm volatile(A("vpmaddwd", 0) A("vpaddd", 4) A("vpaddd", 5) A("vpaddd", 6) A("vpaddd", 7) A("vpmaddwd", 1) A("vpaddd", 8) A("vpaddd", 9) A("vpaddd", 10) A("vpaddd", 11) C); break; // 1 : 4
    case 7: asm volatile(A("vpsllvd", 4) A("vpsllvd", 5) A("vpsllvd", 6) A("vpsllvd", 7) A("vpsllvd", 8) A("vpsllvd", 9) A("vpsllvd", 10) A("vpsllvd", 11) C); break; // variable shifts
    case 8: asm volatile(A("vpsrlvd", 4) A("vpsrlvd", 5) A("vpsrlvd", 6) A("vpsrlvd", 7) A("vpsrlvd", 8) A("vpsrlvd", 9) A("vpsrlvd", 10) A("vpsrlvd", 11) C); break;
    case 9: asm volatile(A("vpminud", 4) A("vpminud", 5) A("vpminud", 6) A("vpminud", 7) A("vpminud", 8) A("vpminud", 9) A("vpminud", 10) A("vpminud", 11) C); break;
    case 10: asm volatile(A("vpmaddwd", 0) A("vpaddd", 4) A("vpaddd", 5) A("vpmaddwd", 1) A("vpaddd", 6) A("vpaddd", 7) "vpsrad $1,%%ymm8,%%ymm8\n\tvpsrad $1,%%ymm9,%%ymm9\n\t" C); break; // 1 madd : 2 adds, + shifts
    case 11: asm volatile(A("vpsubd", 4) A("vpsubd", 5) A("vpsubd", 6) A("vpsubd", 7) A("vpsubd", 8) A("vpsubd", 9) C); break;
    case 12: asm volatile(A("vpaddd", 4) A("vpaddd", 5) A("vpaddd", 6) A("vpaddd", 7) A("vpaddd", 8) A("vpaddd", 9) A("vpaddd", 10) A("vpaddd", 11) A("vpand", 12) A("vpand", 13) A("vpand", 14) A("vpand", 15) C); break; // 8 adds + 4 ands, no madd
    case 13: asm volatile(A("vpmaddwd", 0) A("vpaddd", 4) A("vpaddd", 5) A("vpaddd", 6) A("vpaddd", 7) A("vpaddd", 8) A("vpaddd", 9) A("vpaddd", 10) A("vpaddd", 11) C); break; // 1 : 8
    case 14: asm volatile(A("vpmaddwd", 0) A("vpmaddwd", 1) A("vpaddd", 4) A("vpaddd", 5) A("vpaddd", 6) A("vpaddd", 7) C); break; // 2 madds then 4 adds
    case 15: asm volatile(A("vpmaddwd", 0) A("vpaddd", 4) A("vpaddd", 5) A("vpaddd", 6) A("vpaddd", 7) A("vpmaddwd", 1) A("vpmaddwd", 2) A("vpaddd", 8) A("vpaddd", 9) A("vpaddd", 10) A("vpaddd", 11) C); break;
    }
  printf("case %d done\n", w);
}
