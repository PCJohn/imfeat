"""Identify loops: loopid.py sym.asm a-b [a-b ...] -> instruction count, opcode mix, source lines in body"""
import re, sys, collections
asm = open(sys.argv[1]).read().split("\n")
ins = []; cur = None
for ln in asm:
    m = re.match(r'(/\S*core\.cpp):(\d+)', ln)
    if m: cur = int(m.group(2)); continue
    m = re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)', ln)
    if m: ins.append((int(m.group(1), 16), m.group(2), m.group(3), cur))
for rng in sys.argv[2:]:
    a, b = (int(x, 16) for x in rng.split("-"))
    body = [i for i in ins if a <= i[0] <= b]
    ops = collections.Counter(i[1] for i in body)
    lines = sorted(set(i[3] for i in body if i[3]))
    mem = sum(1 for i in body if "(" in i[2])
    st = sum(1 for i in body if i[1].startswith(("vmov", "mov")) and re.search(r",\s*[^,]*\(", i[2]))
    print(f"{rng}: n={len(body)} mem={mem} stores~{st}  lines={lines}")
    print("   " + " ".join(f"{o}:{c}" for o, c in ops.most_common(14)))
