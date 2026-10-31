| # | commit | target, 1 thread | target, 2 threads | target, 4 threads | 1024x1024x3 finest-64 s1, 1 thread (4 levels, no conversion) | subject |
|---|---|---|---|---|---|---|
| 00 | 3957392 | 9.947 | 5.135 | 3.015 | 9.013 | Add AGENTS.md |
| 01 | cf40d37 | 5.940 | 3.074 | 1.906 | 5.443 | LBP ring: the neighbour test as a saturating subtract compared with zero |
| 02 | bde5b6c | 5.796 | 2.987 | 1.962 | 5.197 | Two-cell blocks widen in-lane; their int32 kinds fold from the 128-bit halves |
| 03 | fc5afd0 | 5.700 | 2.940 | 1.859 | 5.151 | HSV row: the >> 12 and the lane merges folded into the float path |
| 04 | 9544713 | 5.851 | 3.557 | 2.004 | 5.316 | Per-block bookkeeping: the group modulo as a mask; the bar detector's run bounds once per chunk |
| 05 | dfe62b0 | 5.779 | 2.976 | 1.850 | 5.214 | HSV row: one reciprocal carries HNUM, so hdiv's guess is one multiply |
| 06 | b4e65e4 | 5.694 | 2.995 | 1.872 | 5.148 | LBP phase: one loop down the rows, the nibbles emptied on the way |
| 07 | a25107e | 5.733 | 2.952 | 1.893 | 5.188 | The coarser levels rolled up in the bands, a vector of the stripe at a time |
| 08 | ba4a003 | 5.685 | 2.983 | 1.931 | 5.214 | The hashes derived on the summary job's least-loaded thread |
| 09 | 77d5a97 | 5.678 | 2.923 | 1.869 | 5.114 | HOG: one set of accumulators through a sub-chunk's four quarters where the lanes allow it |
| 10 | b0e95e2 | 5.652 | 2.916 | 1.830 | 5.096 | HOG: the energy short of each ray by mask, and no side sum |
