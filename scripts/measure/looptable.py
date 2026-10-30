"""perf_loops output annotated with source line and function. usage: looptable.py perf.data sym.asm cg.out fp fc"""
import re, subprocess, sys
S = sys.path[0]
out = subprocess.run(["python3", S + "/perf_loops.py"] + sys.argv[1:], capture_output=True, text=True).stdout
asm = open(sys.argv[2]).read().split('\n')
src = {}; fn = {}; cur = None; curfn = '?'
for ln in asm:
    m = re.match(r'([0-9a-f]+) <(.*)>:', ln)
    if m: curfn = m.group(2); continue
    m = re.match(r'(/\S*core\.cpp):(\d+)', ln)
    if m: cur = int(m.group(2)); continue
    m = re.match(r'\s*([0-9a-f]+):', ln)
    if m: a = int(m.group(1), 16); src[a] = cur; fn[a] = curfn
for ln in out.splitlines():
    m = re.match(r'\s*([0-9a-f]+)-([0-9a-f]+)\s', ln)
    if m:
        a = int(m.group(1), 16); f = re.sub(r'\(anonymous namespace\)::', '', fn.get(a, '?'))
        f = re.sub(r'^_ZN12_GLOBAL__N_1(?:15FeatureComputer)?\d*', '', f)
        print(f"{ln.rstrip():40s} src {src.get(a)}  {f[:40]}")
    else: print(ln.rstrip())
