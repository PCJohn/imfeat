"""Make a source variant: variant.py NAME old1 new1 [old2 new2 ...] (exact text, each once)"""
import os, shutil, sys
M = os.environ.get("IMFEAT_SCRATCH", os.path.dirname(os.path.abspath(__file__)))  # where variants go
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository
name = sys.argv[1]; d = f"{M}/src_{name}"
shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
shutil.copy(f"{W}/CMakeLists.txt", d); shutil.copytree(f"{W}/src", f"{d}/src")
p = f"{d}/src/imfeat/core.cpp"; s = open(p).read()
for old, new in zip(sys.argv[2::2], sys.argv[3::2]):
    assert s.count(old) == 1, (s.count(old), old[:60])
    s = s.replace(old, new)
open(p, "w").write(s); print(d)
