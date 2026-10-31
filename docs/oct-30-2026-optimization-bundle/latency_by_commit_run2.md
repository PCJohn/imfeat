| # | commit | target, 1 thread | target, 2 threads | target, 4 threads | 1024x1024x3 finest-64 s1, 1 thread (4 levels, no conversion) | subject |
|---|---|---|---|---|---|---|
| 00 | 3957392 | 9.970 | 5.153 | 4.137 | 8.974 | Add AGENTS.md |
| 01 | cf40d37 | 5.939 | 3.056 | 1.901 | 5.352 | LBP ring: the neighbour test as a saturating subtract compared with zero |
| 02 | bde5b6c | 5.719 | 3.584 | 2.828 | 5.159 | Two-cell blocks widen in-lane; their int32 kinds fold from the 128-bit halves |
| 03 | fc5afd0 | 5.672 | 2.907 | 1.956 | 5.100 | HSV row: the >> 12 and the lane merges folded into the float path |
| 04 | 9544713 | 5.745 | 2.951 | 1.881 | 5.185 | Per-block bookkeeping: the group modulo as a mask; the bar detector's run bounds once per chunk |
| 05 | dfe62b0 | 5.807 | 2.998 | 1.868 | 5.236 | HSV row: one reciprocal carries HNUM, so hdiv's guess is one multiply |
| 06 | b4e65e4 | 5.657 | 2.911 | 1.838 | 5.118 | LBP phase: one loop down the rows, the nibbles emptied on the way |
| 07 | a25107e | 5.727 | 2.965 | 2.239 | 5.176 | The coarser levels rolled up in the bands, a vector of the stripe at a time |
| 08 | ba4a003 | 5.682 | 2.937 | 1.902 | 5.167 | The hashes derived on the summary job's least-loaded thread |
| 09 | 77d5a97 | 5.643 | 2.908 | 1.767 | 5.109 | HOG: one set of accumulators through a sub-chunk's four quarters where the lanes allow it |
| 10 | b0e95e2 | 5.623 | 2.902 | 1.822 | 5.061 | HOG: the energy short of each ray by mask, and no side sum |
