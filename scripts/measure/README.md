# Measurement tools

The scripts the target-scenario profile of October 30, 2026 (OPTIMIZATION.md, "Profiled
after step 27") was made with; copies of the previous session's helpers, paths made
relative to the repository. They are conveniences, not a dependency of the build or tests.

* `build.sh SRC BLD PKG [cmake args]`: a Release build of SRC into BLD and an importable
  package in PKG (`PYTHONPATH=PKG`). `IMFEAT_HIGHWAY_SRC` names the Highway checkout.
* `variant.py NAME old new [old new ...]`: a copy of the source with exact text replaced,
  under `$IMFEAT_SCRATCH/src_NAME`, for ablations and A/B variants.
* `dump.py out.npz` / `compare.py a.npz b.npz`: the bit-exact harness (40,164 arrays over
  shapes, grids, strides, threads, the fused conversion and thumbnails), hashed per build.
* `cgmod.py cg.out`: the module's own instruction count from a callgrind file.
* `abba_nc.sh N R pkgA pkgB script args`: pinned to the N idlest physical cores, A/B/B/A
  over R rounds, minima and medians. `target.py threads` is the target scenario's minimum
  over 100 frames; `target_nc.py threads hsv|none` the same with or without the conversion.
* `prof_target.py threads frames`: runs the target scenario for `perf record` or callgrind.
* `perf_loops.py` / `looptable.py perf.data sym.asm cg.out frames_perf frames_cg`: cycle
  samples and instructions per loop (loops are the backward jumps of `objdump -d -l` on a
  symbol build); `loopid.py sym.asm a-b ...` names a loop by its size, opcode mix and the
  source lines inside it; `perf_sig.py` and `loopdiff.py` the same on stripped builds,
  loops matched between two builds by signature.
