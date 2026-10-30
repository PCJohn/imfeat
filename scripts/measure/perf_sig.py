"""Samples per loop of a (stripped) build, each loop named by its instruction signature. usage: perf_sig.py perf.data lib.so"""
import re, sys, subprocess, collections
perf, so = sys.argv[1], sys.argv[2]
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
cnt = collections.Counter(); tot = 0; inmod = 0
for ln in subprocess.run(["perf", "script", "-i", perf, "-F", "ip,dso"], capture_output=True, text=True).stdout.splitlines():
    p = ln.split()
    if len(p) < 2: continue
    tot += 1
    if "imfeat_core" in ln: inmod += 1; cnt[loop_of(int(p[0], 16) - base)] += 1
print(f"total {tot} in module {inmod}")
for l, c in cnt.most_common(16):
    if l is None: print(f"{'outside':>22s} {c:6d}"); continue
    body = ins[idx[l[0]]:idx[l[1]] + 1]
    mix = collections.Counter(b[1] for b in body).most_common(4)
    print(f"{l[0]:>10x}-{l[1]:<10x} {c:6d}  n={len(body):3d}  " + " ".join(f"{o}:{k}" for o, k in mix))
