| # | commit | target, 1 thread | target, 2 threads | target, 4 threads | 1024x1024x3 finest-64 s1, 1 thread (4 levels, no conversion) | subject |
|---|---|---|---|---|---|---|
| 00 | 3957392 | 9.448 | 5.011 | 2.665 | 8.585 | Add AGENTS.md |
| 01 | 4715561 | 7.392 | 3.929 | 2.113 | 6.482 | Kernel as register-resident phases; cheaper folds; signed HOG sums |
| 02 | 3fdcbc4 | 7.234 | 3.804 | 2.054 | 6.389 | Extrema from the LBP ring, nibble-counted LBP bins, two shared sums |
| 03 | 0497dc3 | 7.294 | 3.810 | 2.062 | 6.410 | Sliding plane window: every kernel tap a constant offset from the row |
| 04 | a69e7bd | 7.005 | 3.648 | 1.983 | 6.118 | A batch of small kernel items, kept by pinned A/B timing: -10% at stride 1 |
| 05 | 455e92f | 6.974 | 3.670 | 1.985 | 6.068 | Batched fold for one-cell blocks: four kinds reduced together |
| 06 | 6c166f2 | 6.846 | 3.591 | 1.940 | 5.903 | Vector hash row sums; batched multi-cell folds for the HOG and pair kinds |
| 07 | 3ffd3cc | 6.830 | 3.596 | 1.950 | 5.924 | Level 1 derived in its band from the rolled-up sums; a vector flush |
| 08 | a3a86d2 | 6.848 | 3.592 | 1.942 | 5.968 | Split the two phases of a raw row as it is gathered |
| 09 | d29e6b1 | 6.672 | 3.488 | 1.892 | 5.814 | Derive and roll up cells in groups of four, whatever the block width |
| 10 | 1a381c0 | 6.407 | 3.333 | 1.809 | 5.519 | Halve the HOG ray tests by the rays' mirror symmetry; bar detector sums in registers |
| 11 | 20c7d4b | 6.327 | 3.317 | 1.804 | 5.450 | Give the bar detector phase a function of its own |
| 12 | b510004 | 6.383 | 3.326 | 1.808 | 5.534 | Energy phase at stride 1 as three chunk-wide product sums per response column |
| 13 | 875d14e | 6.070 | 3.194 | 1.742 | 5.506 | HSV conversion: the table step as one rounding of the float guess |
| 14 | c9c0c1c | 6.066 | 3.190 | 1.749 | 5.511 | Resize: the horizontal pass over all of a row's planes at once |
| 15 | cd2a2ea | 6.260 | 3.219 | 1.754 | 5.693 | Resize: the float taps' shuffle lands each byte in its int32 lane |
| 16 | 8301afe | 6.084 | 3.183 | 1.734 | 5.505 | Resize: a destination row finished in one pass from its cached tap rows |
| 17 | 502fe49 | 6.076 | 3.218 | 1.745 | 5.518 | Resize: the shuffle table's window by a broadcast-128 load |
| 18 | dff0dbb | 5.819 | 3.121 | 1.700 | 5.290 | Roll up two kinds per lane add; fold the summary once per cell row |
| 19 | 69db70a | 5.991 | 3.151 | 1.670 | 5.436 | Summary across threads from three bands, every fold in its cell order |
| 20 | 7189851 | 5.952 | 3.088 | 1.621 | 5.369 | Summary after the join split by lane across the threads |
| 21 | c874e2d | 5.852 | 3.069 | 1.614 | 5.294 | Prefetch where the frame is first read; the finish converts in range; the kinds loop decides the block width once |
| 22 | 2e23b8e | 5.925 | 3.054 | 1.614 | 5.356 | Records prefetched ahead of the derive's stores; the resize's store bound tested once per group |
| 23 | c80dbfb | 5.845 | 3.012 | 1.593 | 5.268 | Cells derived and rolled up eight at a time, the group's kind stride a whole line |
| 24 | 858fe52 | 5.824 | 3.016 | 1.591 | 5.246 | Resize rounding: the saturating pack is the clamp |
| 25 | bac2b23 | 5.848 | 3.023 | 1.607 | 5.345 | Resize: the periodic groups of the horizontal pass with their slot tables in registers |
| 26 | cf40d37 | 5.781 | 2.975 | 1.574 | 5.198 | LBP ring: the neighbour test as a saturating subtract compared with zero |
| 27 | bde5b6c | 5.701 | 2.940 | 1.571 | 5.061 | Two-cell blocks widen in-lane; their int32 kinds fold from the 128-bit halves |
| 28 | fc5afd0 | 5.665 | 2.918 | 1.542 | 5.103 | HSV row: the >> 12 and the lane merges folded into the float path |
| 29 | 9544713 | 5.614 | 2.915 | 1.557 | 5.091 | Per-block bookkeeping: the group modulo as a mask; the bar detector's run bounds once per chunk |
| 30 | dfe62b0 | 5.681 | 2.980 | 1.562 | 5.172 | HSV row: one reciprocal carries HNUM, so hdiv's guess is one multiply |
| 31 | b4e65e4 | 5.619 | 2.907 | 1.538 | 5.119 | LBP phase: one loop down the rows, the nibbles emptied on the way |
