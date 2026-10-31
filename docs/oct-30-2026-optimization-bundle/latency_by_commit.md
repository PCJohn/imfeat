| # | commit | target, 1 thread | target, 2 threads | target, 4 threads | 1024x1024x3 finest-64 s1, 1 thread (4 levels, no conversion) | subject |
|---|---|---|---|---|---|---|
| 00 | 3957392 | 9.855 | 5.135 | 2.708 | 8.974 | Add AGENTS.md |
| 01 | cf40d37 | 5.939 | 3.049 | 1.614 | 5.352 | LBP ring: the neighbour test as a saturating subtract compared with zero |
| 02 | bde5b6c | 5.719 | 2.982 | 1.559 | 5.159 | Two-cell blocks widen in-lane; their int32 kinds fold from the 128-bit halves |
| 03 | fc5afd0 | 5.672 | 2.907 | 1.555 | 5.100 | HSV row: the >> 12 and the lane merges folded into the float path |
| 04 | 9544713 | 5.745 | 2.933 | 1.558 | 5.185 | Per-block bookkeeping: the group modulo as a mask; the bar detector's run bounds once per chunk |
| 05 | dfe62b0 | 5.745 | 2.976 | 1.572 | 5.214 | HSV row: one reciprocal carries HNUM, so hdiv's guess is one multiply |
| 06 | b4e65e4 | 5.656 | 2.911 | 1.563 | 5.118 | LBP phase: one loop down the rows, the nibbles emptied on the way |
| 07 | a25107e | 5.714 | 2.950 | 1.539 | 5.176 | The coarser levels rolled up in the bands, a vector of the stripe at a time |
| 08 | ba4a003 | 5.682 | 2.921 | 1.531 | 5.167 | The hashes derived on the summary job's least-loaded thread |
| 09 | 77d5a97 | 5.641 | 2.903 | 1.507 | 5.109 | HOG: one set of accumulators through a sub-chunk's four quarters where the lanes allow it |
| 10 | b0e95e2 | 5.616 | 2.898 | 1.514 | 5.061 | HOG: the energy short of each ray by mask, and no side sum |
