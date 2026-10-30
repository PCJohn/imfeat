"""Every output identical across thread counts: threads_eq.py [maxthreads]"""
import sys, os, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
from conftest import frame
import imfeat
if os.environ.get("PYTHONPATH") and not os.path.abspath(imfeat.__file__).startswith(os.path.abspath(os.environ["PYTHONPATH"])):
    raise SystemExit(f"imfeat imported from {imfeat.__file__}, not from PYTHONPATH={os.environ['PYTHONPATH']}: no module built there?")
T = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cases = [((1024, 1024, 3), [(k, k) for k in range(6, -1, -1)], 1, "hsv"), ((512, 768, 3), [(5, 5), (4, 4), (3, 3)], 1, "hsv"),
         ((256, 256, 3), [(4, 4), (3, 3), (2, 2)], 2, None), ((576, 1024, 3), [(6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1)], 1, "hsv"),
         ((320, 480, 1), [(5, 5), (4, 4)], 1, None), ((192, 320, 4), [(4, 4), (2, 2)], 3, None)]
bad = 0
for shape, grid, stride, fs in cases:
    img = frame(shape)
    ref = None
    for th in range(1, T + 1):
        p = imfeat.FeatureComputer(shape, grid=grid, stride=stride, threads=th, feature_space=fs).features(img)
        cur = [np.ascontiguousarray(a).tobytes() for a in (*p.maps, *p.moments, *p.summary, p.hashes)]
        if ref is None: ref = cur
        elif cur != ref: bad += 1; print("DIFF", shape, grid, stride, fs, "threads", th)
print("thread identity:", "OK" if not bad else f"{bad} differences", "over", len(cases), "cases x", T, "threads")
