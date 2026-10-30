"""Cycle samples per loop of two builds, loops matched by signature. usage: loopdiff.py A.data A.so B.data B.so"""
import re, sys, subprocess, collections
def profile(perf, so):
    asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", so], capture_output=True, text=True).stdout
    ins = []
    for ln in asm.splitlines():
        m = re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)', ln)
        if m: ins.append((int(m.group(1), 16), m.group(2), m.group(3)))
    idx = {a: i for i, (a, _, _) in enumerate(ins)}
    loops = []
    for a, op, arg in ins:
        if op.startswith('j'):
            m = re.match(r'([0-9a-f]+)', arg)
            if m:
                t = int(m.group(1), 16)
                if t < a and t in idx and idx[a] - idx[t] < 1500: loops.append((t, a))
    loops.sort(key=lambda l: l[1] - l[0])
    def loop_of(addr):
        for t, a in loops:
            if t <= addr <= a: return (t, a)
        return None
    mm = subprocess.run(["perf", "script", "-i", perf, "--show-mmap-events", "-F", "ip,dso"], capture_output=True, text=True).stdout
    base = None
    for ln in mm.splitlines():
        if "PERF_RECORD_MMAP" in ln and "imfeat_core" in ln:
            m = re.search(r"\[(0x[0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+)", ln)
            if m: base = int(m.group(1), 16) - int(m.group(3), 16); break
    cnt = collections.Counter(); tot = 0
    for ln in subprocess.run(["perf", "script", "-i", perf, "-F", "ip,dso"], capture_output=True, text=True).stdout.splitlines():
        p = ln.split()
        if len(p) < 2 or "imfeat_core" not in ln: continue
        tot += 1; cnt[loop_of(int(p[0], 16) - base)] += 1
    sig = {}
    for l, c in cnt.items():
        if l is None: sig[("outside",)] = sig.get(("outside",), 0) + c; continue
        body = ins[idx[l[0]]:idx[l[1]] + 1]
        mix = tuple(sorted(collections.Counter(b[1] for b in body).items()))
        key = (len(body), mix)
        sig[key] = sig.get(key, 0) + c
    return tot, sig
ta, A = profile(sys.argv[1], sys.argv[2]); tb, B = profile(sys.argv[3], sys.argv[4])
# normalise B to A by the loops both have whose counts agree within 10%
common = [k for k in A if k in B and A[k] > 150]
stable = [k for k in common if abs(B[k] / A[k] - 1) < 0.1]
norm = sum(A[k] for k in stable) / max(1, sum(B[k] for k in stable))
print(f"A {ta} samples, B {tb}; B scaled by {norm:.3f} over {len(stable)} stable loops of {len(common)}")
rows = []
for k in set(A) | set(B):
    a, b = A.get(k, 0), B.get(k, 0) * norm
    if max(a, b) >= 120: rows.append((a - b, k, a, b))
rows.sort(key=lambda r: -abs(r[0]))
print(f"{'loop (n, top ops)':48s} {'A':>7s} {'B':>7s} {'B-A':>7s}")
for d, k, a, b in rows[:24]:
    name = "outside" if k[0] == "outside" else f"n={k[0]:3d} " + " ".join(f"{o}:{c}" for o, c in sorted(k[1], key=lambda kv: -kv[1])[:4])
    print(f"{name[:48]:48s} {a:7.0f} {b:7.0f} {b-a:+7.0f}")
print(f"{'module total':48s} {ta:7d} {tb*norm:7.0f} {tb*norm-ta:+7.0f}")
