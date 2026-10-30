"""Random topological orders of the HOG quarter-row body (the kernel after step 36, |grad|^4
included) under the kernel's register budget, each built as an asm loop over 16 L1 rows and run
twice under `perf stat -e cycles,fpu_pipe_assignment.total0`; the kernel's own order first, then
N random ones, the best six and the worst printed as cycles a row and pipe-0 uops a row.
usage: hog_order.py N core [seed] [file-of-orders]    (needs hog_order_tpl.cpp beside it)
Registers: ymm0 the loaded pair then the |grad|^4 temporaries, 1 |pair|, 2 |grad|^2, 3 its signed
copy, 4 the |grad|^4 sum, 5/6 the ray tests in turn, 7 the and-temporary, 8-15 the eight sums.
Found October 30, 2026 (OPTIMIZATION.md, "the pipe assignment, after step 36"): the kernel's
order 12.0 cycles a row, the best of 150 random orders 11.1, the worst 19.9, pipe 0's count
9.5-11 throughout: the order decides the cycles, the pipe-0 count does not."""
import sys, random, subprocess, re, os
N, core = int(sys.argv[1]), sys.argv[2]
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
T = {0: 5, 1: 6, 2: 5, 3: 6}
nodes = {
 "ld":  ("vmovdqa (%[hb]), %%ymm0", []),
 "abs": ("vpabsw %%ymm0, %%ymm1", ["ld"]),
 "g2":  ("vpmaddwd %%ymm0, %%ymm0, %%ymm2", ["ld"]),
 "g2s": ("vpsignd %%ymm0, %%ymm2, %%ymm3", ["g2"]),
 "hi":  ("vpshufd $0xf5, %%ymm2, %%ymm0", ["g2s"]),
 "sqo": ("vpmuldq %%ymm0, %%ymm0, %%ymm0", ["hi"]),
 "sq":  ("vpmuldq %%ymm2, %%ymm2, %%ymm7\n\tvpaddq %%ymm7, %%ymm0, %%ymm0", ["sqo"]),
 "sg":  ("vpaddq %%ymm0, %%ymm4, %%ymm4", ["sq"]),
}
for j in range(4):
    t = T[j]
    nodes[f"m{j}"] = (f"vpmaddwd {j*32}(%[rt]), %%ymm1, %%ymm{t}", ["abs"] + ([f"q{j-2}"] if j >= 2 else []))
    nodes[f"s{j}"] = (f"vpsrad $31, %%ymm{t}, %%ymm{t}", [f"m{j}"])
    nodes[f"a{j}"] = (f"vpand %%ymm{t}, %%ymm2, %%ymm7\n\tvpaddd %%ymm7, %%ymm{8+j}, %%ymm{8+j}", [f"s{j}", "g2s"])
    nodes[f"b{j}"] = (f"vpand %%ymm{t}, %%ymm3, %%ymm7\n\tvpaddd %%ymm7, %%ymm{12+j}, %%ymm{12+j}", [f"a{j}"])
    nodes[f"q{j}"] = ("", [f"b{j}"])  # marker: t_j consumed
def topo():
    done, order = set(), []
    while len(done) < len(nodes):
        r = [n for n in nodes if n not in done and all(d in done for d in nodes[n][1])]
        n = random.choice(r); order.append(n); done.add(n)
    return order
template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hog_order_tpl.cpp")).read()
template = template.replace('asm volatile("vpxor %%ymm8', 'asm volatile("vpxor %%ymm4,%%ymm4,%%ymm4\\n\\tvpxor %%ymm8').replace('printf("@TAG@: chk %d %lld\\n", out[0], (long long)sg[0]);', 'int64_t o4[4]; asm volatile("vmovdqu %%ymm4, (%0)" :: "r"(o4) : "memory"); printf("@TAG@: chk %d %lld\\n", out[0], (long long)o4[0]);')
def run(order, tag):
    asm = "\\n\\t".join(nodes[n][0].replace("\n\t", "\\n\\t") for n in order if nodes[n][0])
    src = template.replace("@ASM@", asm).replace("@TAG@", tag)
    p = os.path.join(os.environ.get("IMFEAT_SCRATCH", "/tmp"), f"hog_order_{os.getpid()}")
    open(p + ".cpp", "w").write(src)
    r = subprocess.run(["g++", "-O2", "-march=native", "-o", p, p + ".cpp"], capture_output=True, text=True)
    if r.returncode: print(r.stderr[:1500]); raise SystemExit(1)
    best = None
    for _ in range(2):
        out = subprocess.run(["taskset", "-c", core, "perf", "stat", "-e", "cycles:u,fpu_pipe_assignment.total0:u", p], capture_output=True, text=True).stderr
        cyc = float(re.search(r"([\d,]+)\s+cycles", out).group(1).replace(",", "")) / 32e6
        p0 = float(re.search(r"([\d,]+)\s+fpu_pipe_assignment.total0", out).group(1).replace(",", "")) / 32e6
        if best is None or cyc < best[0]: best = (cyc, p0)
    return best
kern = ["ld", "abs", "g2", "g2s", "hi", "sqo", "sq", "sg", "m0", "s0", "a0", "b0", "q0", "m1", "s1", "a1", "b1", "q1", "m2", "m3", "s3", "s2", "a2", "a3", "b2", "q2", "b3", "q3"]
res = [(run(kern, "kernel"), "kernel", kern)]
print(f"kernel order: {res[0][0][0]:.2f} cycles, pipe0 {res[0][0][1]:.2f}", flush=True)
if len(sys.argv) > 4:
    for ln in open(sys.argv[4]):
        o = ln.split(); r = run(o, "given"); print(f"{r[0]:.2f} cycles pipe0 {r[1]:.2f}  given: {' '.join(o)}", flush=True)
for i in range(N):
    o = topo(); r = run(o, f"r{i}"); res.append((r, f"r{i}", o))
    if i % 25 == 24: print(f"{i+1} done; best so far {min(res)[0][0]:.2f}", flush=True)
res.sort()
for r, tag, o in res[:6]: print(f"{r[0]:.2f} cycles pipe0 {r[1]:.2f}  {tag}: {' '.join(o)}")
print("worst:", f"{res[-1][0][0]:.2f} cycles pipe0 {res[-1][0][1]:.2f}", ' '.join(res[-1][2]))
