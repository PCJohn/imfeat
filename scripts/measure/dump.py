"""Hash every output over many configs -> npz. usage: PYTHONPATH=<pkg> python dump.py out.npz"""
import sys, hashlib, itertools
import numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
from conftest import frame
import imfeat
if os.environ.get("PYTHONPATH") and not os.path.abspath(imfeat.__file__).startswith(os.path.abspath(os.environ["PYTHONPATH"])):
    raise SystemExit(f"imfeat imported from {imfeat.__file__}, not from PYTHONPATH={os.environ['PYTHONPATH']}: no module built there?")

out = {}
def put(key, arrs):
    for i, a in enumerate(arrs):
        a = np.ascontiguousarray(a)
        out[f"{key}/{i}"] = np.frombuffer(hashlib.sha256(a.tobytes()).digest()[:8], np.uint64)[0]
        out[f"{key}/{i}/shape"] = np.array(a.shape)

rng = np.random.default_rng(1)
shapes = [(64, 64), (64, 96, 3), (128, 128, 3), (96, 160, 2), (256, 256, 3), (256, 256, 4), (40, 72, 8), (128, 128, 1)]
grids = [[(3, 3)], [(4, 4), (3, 3), (2, 2)], [(5, 5), (4, 4), (3, 3), (2, 2)], [(2, 3)], [(3, 3), (1, 1)]]
strides = [1, 2, 3, 4, (1, 2), 8]
for shape in shapes:
    for kind in ("tex", "noise"):
        img = frame(shape) if kind == "tex" else rng.integers(0, 256, shape, dtype=np.uint8)
        for gi, g in enumerate(grids):
            h, w = shape[:2]
            if any(h % (1 << a) or w % (1 << b) for a, b in g): continue
            for s in strides:
                for th in (1, 2, 3):
                    try:
                        fc = imfeat.FeatureComputer(shape, grid=g, stride=s, threads=th, feature_space=None)
                    except Exception as e:
                        continue
                    key = f"{shape}/{kind}/g{gi}/s{s}/t{th}"
                    p = fc.features(img)
                    put(key + "/maps", p.maps); put(key + "/mom", p.moments)
                    put(key + "/sum", p.summary); put(key + "/hash", [p.hashes])
                    if th == 1:
                        raw = fc.compute(img)
                        put(key + "/raw", [raw[k] for k in sorted(raw)])
# colour conversion and thumbnails
for shape in [(256, 256, 3), (720, 1280, 3), (240, 320, 3)]:
    img = frame(shape)
    for th in (1, 2):
        fc = imfeat.FeatureComputer(shape, grid=[(4, 4), (3, 3)], stride=1, threads=th, thumb=None if shape[0] == 256 else "pow2-fit")
        p = fc.features(img)
        key = f"{shape}/hsv/t{th}"
        put(key + "/maps", p.maps); put(key + "/mom", p.moments); put(key + "/sum", p.summary); put(key + "/hash", [p.hashes])
        if shape[0] != 256:
            fc = imfeat.FeatureComputer(shape, grid=[(4, 4), (3, 3)], stride=2, threads=th, thumb=128, channels=[0, 2])
            p = fc.features(img)
            key = f"{shape}/thumb256/t{th}"
            put(key + "/maps", p.maps); put(key + "/mom", p.moments); put(key + "/sum", p.summary); put(key + "/hash", [p.hashes])
np.savez(sys.argv[1], **out)
print(len(out), "entries")
