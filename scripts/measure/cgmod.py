"""Self cost (Ir) of the imfeat_core object in a callgrind output. usage: cgmod.py cg.out"""
import re, sys
names = {}; inmod = False; skip = False; npos = 1; total = 0
for ln in open(sys.argv[1]):
    if ln.startswith("positions:"): npos = len(ln.split()) - 1; continue
    if ln.startswith(("ob=", "cob=")):
        tok = ln.split("=", 1)[1].strip(); m = re.match(r"\((\d+)\)(?: (.*))?", tok)
        if m:
            if m.group(2): names[m.group(1)] = m.group(2)
            name = names.get(m.group(1), "")
        else: name = tok
        if ln.startswith("ob="): inmod = "imfeat_core" in name
        continue
    if ln.startswith("calls="): skip = True; continue
    if ln[:1] in "0123456789+-*" and ln.strip():
        if skip: skip = False; continue
        if inmod: total += int(ln.split()[npos])
print(total)
