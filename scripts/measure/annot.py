"""Cycle samples per instruction of a loop: annot.py perf.data lib.so START-END [min_samples]"""
import sys, subprocess, collections
perf, so, rng = sys.argv[1], sys.argv[2], sys.argv[3]
mins = int(sys.argv[4]) if len(sys.argv) > 4 else 0
a, b = (int(x, 16) for x in rng.split("-"))
ips = [int(x.split()[0], 16) for x in subprocess.run(["perf", "script", "-i", perf, "-F", "ip,dso"], capture_output=True, text=True).stdout.splitlines() if "imfeat_core" in x]
base = collections.Counter(((ip - a) & ~0xfff) for ip in ips).most_common(1)[0][0]
c = collections.Counter(ip - base for ip in ips if a <= ip - base <= b)
asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", f"--start-address={a}", f"--stop-address={b + 8}", so], capture_output=True, text=True).stdout
tot = sum(c.values()); print(f"loop {rng}: {tot} samples")
for ln in asm.splitlines():
    p = ln.strip().split(":", 1)
    try: off = int(p[0], 16)
    except (ValueError, IndexError): continue
    n = c.get(off, 0)
    if n >= mins: print(f"{off:x} {n:6d} {100 * n / max(tot, 1):5.1f}%  {p[1].strip()[:60]}")
