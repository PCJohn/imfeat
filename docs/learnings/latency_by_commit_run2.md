| # | commit | target, 1 thread | target, 2 threads | target, 4 threads | 1024x1024x3 finest-64 s1, 1 thread (4 levels, no conversion) | subject |
|---|---|---|---|---|---|---|
| 00 | 3957392 | 9.448 | 5.692 | 3.150 | 8.585 | Add AGENTS.md |
| 01 | 4715561 | 7.392 | 4.437 | 4.154 | 6.482 | Kernel as register-resident phases; cheaper folds; signed HOG sums |
| 02 | 3fdcbc4 | 7.234 | 4.334 | 2.336 | 6.389 | Extrema from the LBP ring, nibble-counted LBP bins, two shared sums |
| 03 | 0497dc3 | 7.294 | 3.810 | 2.359 | 6.410 | Sliding plane window: every kernel tap a constant offset from the row |
| 04 | a69e7bd | 7.005 | 4.214 | 2.243 | 6.118 | A batch of small kernel items, kept by pinned A/B timing: -10% at stride 1 |
| 05 | 455e92f | 6.974 | 4.164 | 2.273 | 6.068 | Batched fold for one-cell blocks: four kinds reduced together |
| 06 | 6c166f2 | 6.846 | 4.103 | 2.221 | 5.903 | Vector hash row sums; batched multi-cell folds for the HOG and pair kinds |
| 07 | 3ffd3cc | 6.869 | 4.136 | 2.249 | 5.942 | Level 1 derived in its band from the rolled-up sums; a vector flush |
| 08 | a3a86d2 | 6.848 | 4.186 | 2.256 | 5.968 | Split the two phases of a raw row as it is gathered |
| 09 | d29e6b1 | 6.672 | 4.095 | 2.215 | 5.814 | Derive and roll up cells in groups of four, whatever the block width |
| 10 | 1a381c0 | 6.407 | 3.975 | 2.150 | 5.519 | Halve the HOG ray tests by the rays' mirror symmetry; bar detector sums in registers |
| 11 | 20c7d4b | 6.327 | 3.941 | 2.148 | 5.450 | Give the bar detector phase a function of its own |
| 12 | b510004 | 6.393 | 3.948 | 2.146 | 5.538 | Energy phase at stride 1 as three chunk-wide product sums per response column |
| 13 | 875d14e | 6.070 | 3.907 | 2.290 | 5.506 | HSV conversion: the table step as one rounding of the float guess |
| 14 | c9c0c1c | 6.066 | 3.858 | 2.300 | 5.511 | Resize: the horizontal pass over all of a row's planes at once |
| 15 | cd2a2ea | 6.262 | 3.877 | 2.149 | 5.693 | Resize: the float taps' shuffle lands each byte in its int32 lane |
| 16 | 8301afe | 6.084 | 3.898 | 2.087 | 5.505 | Resize: a destination row finished in one pass from its cached tap rows |
| 17 | 502fe49 | 6.076 | 3.843 | 2.081 | 5.518 | Resize: the shuffle table's window by a broadcast-128 load |
| 18 | dff0dbb | 5.819 | 3.736 | 2.248 | 5.290 | Roll up two kinds per lane add; fold the summary once per cell row |
| 19 | 69db70a | 5.991 | 3.174 | 1.956 | 5.436 | Summary across threads from three bands, every fold in its cell order |
| 20 | 7189851 | 5.952 | 3.608 | 1.920 | 5.369 | Summary after the join split by lane across the threads |
| 21 | c874e2d | 5.852 | 3.665 | 1.894 | 5.294 | Prefetch where the frame is first read; the finish converts in range; the kinds loop decides the block width once |
| 22 | 2e23b8e | 5.925 | 3.671 | 2.000 | 5.356 | Records prefetched ahead of the derive's stores; the resize's store bound tested once per group |
| 23 | c80dbfb | 5.845 | 3.680 | 1.966 | 5.268 | Cells derived and rolled up eight at a time, the group's kind stride a whole line |
| 24 | 858fe52 | 5.831 | 3.626 | 2.010 | 5.270 | Resize rounding: the saturating pack is the clamp |
| 25 | bac2b23 | 5.848 | 3.676 | 2.028 | 5.345 | Resize: the periodic groups of the horizontal pass with their slot tables in registers |
| 26 | cf40d37 | 5.781 | 3.513 | 1.958 | 5.198 | LBP ring: the neighbour test as a saturating subtract compared with zero |
| 27 | bde5b6c | 5.713 | 2.955 | 1.859 | 5.143 | Two-cell blocks widen in-lane; their int32 kinds fold from the 128-bit halves |
| 28 | fc5afd0 | 5.673 | 2.952 | 1.932 | 5.160 | HSV row: the >> 12 and the lane merges folded into the float path |
| 29 | 9544713 | 5.618 | 3.513 | 1.808 | 5.091 | Per-block bookkeeping: the group modulo as a mask; the bar detector's run bounds once per chunk |
| 30 | dfe62b0 | 5.681 | 3.466 | 1.850 | 5.173 | HSV row: one reciprocal carries HNUM, so hdiv's guess is one multiply |
| 31 | b4e65e4 | 5.619 | 3.460 | 1.925 | 5.121 | LBP phase: one loop down the rows, the nibbles emptied on the way |
| 32 | a25107e | 5.644 | 3.387 | 1.846 | 5.227 | The coarser levels rolled up in the bands, a vector of the stripe at a time |
| 33 | ba4a003 | 5.699 | 2.927 | 1.757 | 5.146 | The hashes derived on the summary job's least-loaded thread |
