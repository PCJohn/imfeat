"""The target scenario, frames times: prof_target.py threads frames"""
import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
from conftest import frame
import imfeat
if os.environ.get("PYTHONPATH") and not os.path.abspath(imfeat.__file__).startswith(os.path.abspath(os.environ["PYTHONPATH"])):
    raise SystemExit(f"imfeat imported from {imfeat.__file__}, not from PYTHONPATH={os.environ['PYTHONPATH']}: no module built there?")
th, frames = int(sys.argv[1]), int(sys.argv[2])
img = frame((1024, 1024, 3))
fc = imfeat.FeatureComputer(img.shape, grid=[(k, k) for k in range(6, -1, -1)], stride=1, threads=th, feature_space="hsv")
for _ in range(frames): fc.features(img)
