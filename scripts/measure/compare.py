import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ka, kb = set(a.files), set(b.files)
bad = [k for k in sorted(ka & kb) if not np.array_equal(a[k], b[k])]
print("entries", len(ka), len(kb), "missing", len(ka ^ kb), "differ", len(bad))
for k in bad[:40]: print("  DIFF", k)
sys.exit(1 if bad or ka != kb else 0)
