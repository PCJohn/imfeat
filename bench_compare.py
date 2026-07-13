import time, numpy as np, tensorstats as ts, structstats as ss, imfeat
T, EXPS, STRIDE = 256, [5,4,3,2], 2
img = np.stack([(127+80*np.sin(np.mgrid[0:T,0:T][1]/18.)).astype(np.uint8),
                np.random.default_rng(0).integers(0,256,(T,T),dtype=np.uint8),
                np.tile(np.linspace(0,255,T,dtype=np.uint8),(T,1))], -1)

def bench(f, n=200):
    f(); t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1e3

# old pipeline, as framegate uses it: moments on 3 channels, structure on 1 (V)
sc = ts.StatsComputer(shape=(T,T,3), axes=[(0,1)], stride=(STRIDE,STRIDE,1), grid=[(e,e,2) for e in EXPS])
st = ss.StructComputer(shape=(T,T), grid=[(e,e) for e in EXPS], stride=STRIDE)
V = np.ascontiguousarray(img[:,:,2])
old_fg = bench(lambda: (sc.compute(img), st.features(V)))
# old pipeline, target functionality: structure on ALL channels too
st3 = ss.StructComputer(shape=(T,T,3), grid=[(e,e) for e in EXPS], stride=STRIDE)
old_all = bench(lambda: (sc.compute(img), st3.features(img)))
fc = imfeat.FeatureComputer(shape=(T,T,3), grid=[(e,e) for e in EXPS], stride=STRIDE)
new = bench(lambda: fc.features(img))
print(f"old (framegate today: moments x3ch + struct x1ch) : {old_fg:.3f} ms")
print(f"old (target: moments x3ch + struct x3ch)          : {old_all:.3f} ms")
print(f"imfeat (fused, moments+struct x3ch)               : {new:.3f} ms")
print(f"speedup vs framegate-today {old_fg/new:.2f}x | vs target {old_all/new:.2f}x")
for s in (1,2,4,8):
    fcs = imfeat.FeatureComputer(shape=(T,T,3), grid=[(e,e) for e in EXPS], stride=s)
    print(f"  imfeat stride={s}: {bench(lambda: fcs.features(img)):.3f} ms")
