"""Cycle samples per loop of the module, matched with callgrind's Ir per loop.
usage: python perf_loops.py perf.data sym_ir.asm cg.out frames_perf frames_cg"""
import re, sys, subprocess, collections
perf, asm, cg, fp, fc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
# loops as cgloops finds them: backward jumps within a function
ins = []; func = {}; fname = "?"
for ln in open(asm):
    m = re.match(r"([0-9a-f]+) <(.*)>:", ln)
    if m: fname = m.group(2); continue
    m = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)", ln)
    if m: ins.append((int(m.group(1), 16), m.group(2), m.group(3))); func[int(m.group(1), 16)] = fname
idx = {a: i for i, (a, _, _) in enumerate(ins)}
loops = []
for a, op, arg in ins:
    if op.startswith("j"):
        m = re.match(r"([0-9a-f]+)", arg)
        if m:
            t = int(m.group(1), 16)
            if t < a and t in idx and func.get(t) == func.get(a) and idx[a] - idx[t] < 1500: loops.append((t, a))
loops.sort(key=lambda l: l[1] - l[0])  # innermost first
def loop_of(addr):
    for t, a in loops:
        if t <= addr <= a: return (t, a)
    return None
# perf samples: ip relative to the module
out = subprocess.run(["perf", "script", "-i", perf, "-F", "ip,dso,sym,symoff", "--no-demangle"], capture_output=True, text=True).stdout
# module base from perf's mmap: use the first sample's sym to resolve: simpler, use `perf script -F ip,dso` and the dso's load address from --show-mmap-events
mm = subprocess.run(["perf", "script", "-i", perf, "--show-mmap-events", "-F", "ip,dso"], capture_output=True, text=True).stdout
base = None
for ln in mm:
    pass
for ln in mm.splitlines():
    if "PERF_RECORD_MMAP" in ln and "imfeat_core" in ln:
        m = re.search(r"\[(0x[0-9a-f]+)\((0x[0-9a-f]+)\) @ (0x[0-9a-f]+)", ln)
        if m: base = int(m.group(1), 16) - int(m.group(3), 16); break
cyc = collections.Counter(); total = 0; inmod = 0
for ln in out.splitlines():
    p = ln.split()
    if len(p) < 2: continue
    total += 1
    if "imfeat_core" not in ln: continue
    inmod += 1
    off = int(p[0], 16) - base
    cyc[loop_of(off)] += 1
# callgrind Ir per loop, parsed as cgloops does (compressed positions)
ir = collections.Counter()
ev = None; inmod_ob = False; names = {}; addr = line = 0; skipnext = False
def pos(tok, old):
    if tok == "*": return old
    if tok[0] in "+-": return old + int(tok)
    return int(tok, 16) if tok.startswith("0x") else int(tok)
for ln in open(cg):
    if ln.startswith("events:"): ev = ln.split()[1:]; continue
    if ln.startswith(("ob=", "cob=")):
        tok = ln.split("=", 1)[1].strip(); m = re.match(r"\((\d+)\)(?: (.*))?", tok)
        if m:
            if m.group(2): names[m.group(1)] = m.group(2)
            name = names.get(m.group(1), "")
        else: name = tok
        if ln.startswith("ob="): inmod_ob = "imfeat_core" in name
        continue
    if ln.startswith("calls="): skipnext = True; continue
    if ln[:1] in "0123456789+-*" and ln.strip():
        parts = ln.split()
        addr = pos(parts[0], addr); line = pos(parts[1], line)
        if skipnext: skipnext = False; continue
        if not inmod_ob: continue
        ir[loop_of(addr)] += int(parts[2])
print(f"samples {total}, in module {inmod} ({100*inmod/max(total,1):.0f}%); perf frames {fp}, callgrind frames {fc}")
print(f"{'loop':>14s} {'cyc%':>6s} {'Ir%':>6s}  {'cyc/Ir':>6s}  src")
src = {}
for ln in open(asm):
    m = re.match(r"/.*core\.cpp:(\d+)", ln)
    if m: last = int(m.group(1))
rows = []
tc = sum(cyc.values()); ti = sum(ir.values())
for l, c in cyc.most_common(40):
    i = ir.get(l, 0)
    ratio = (c / tc) / (i / ti) if i else float('inf')
    rows.append((l, 100*c/tc, 100*i/ti, ratio))
for l, pc, pi, r in rows:
    name = f"{l[0]:x}-{l[1]:x}" if l else "outside"
    print(f"{name:>14s} {pc:6.1f} {pi:6.1f}  {r:6.2f}")
