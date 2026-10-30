| # | commit | target, 1 thread | target, 2 threads | target, 4 threads | 1024x1024x3 finest-64 s1, 1 thread (4 levels, no conversion) | subject |
|---|---|---|---|---|---|---|
| 00 | 3957392 | 9.669 | 5.595 | 3.046 | 8.826 | Add AGENTS.md |
| 01 | 4715561 | 7.553 | 4.499 | 2.425 | 6.729 | Kernel as register-resident phases; cheaper folds; signed HOG sums |
| 02 | 3fdcbc4 | 7.397 | 4.410 | 2.371 | 6.497 | Extrema from the LBP ring, nibble-counted LBP bins, two shared sums |
| 03 | 0497dc3 | 7.409 | 4.485 | 2.403 | 6.584 | Sliding plane window: every kernel tap a constant offset from the row |
| 04 | a69e7bd | 7.190 | 4.391 | 2.307 | 6.281 | A batch of small kernel items, kept by pinned A/B timing: -10% at stride 1 |
| 05 | 455e92f | 7.069 | 4.324 | 3.500 | 6.149 | Batched fold for one-cell blocks: four kinds reduced together |
| 06 | 6c166f2 | 6.952 | 4.232 | 2.498 | 5.998 | Vector hash row sums; batched multi-cell folds for the HOG and pair kinds |
| 07 | 3ffd3cc | 6.830 | 3.643 | 2.275 | 5.924 | Level 1 derived in its band from the rolled-up sums; a vector flush |
| 08 | a3a86d2 | 6.890 | 4.174 | 2.265 | 5.996 | Split the two phases of a raw row as it is gathered |
| 09 | d29e6b1 | 6.757 | 3.514 | 2.269 | 5.859 | Derive and roll up cells in groups of four, whatever the block width |
| 10 | 1a381c0 | 6.481 | 3.997 | 2.148 | 5.596 | Halve the HOG ray tests by the rays' mirror symmetry; bar detector sums in registers |
| 11 | 20c7d4b | 6.439 | 3.983 | 2.172 | 5.558 | Give the bar detector phase a function of its own |
| 12 | b510004 | 6.384 | 3.966 | 2.189 | 5.534 | Energy phase at stride 1 as three chunk-wide product sums per response column |
| 13 | 875d14e | 6.217 | 3.944 | 2.119 | 5.663 | HSV conversion: the table step as one rounding of the float guess |
| 14 | c9c0c1c | 6.243 | 3.926 | 2.113 | 5.687 | Resize: the horizontal pass over all of a row's planes at once |
| 15 | cd2a2ea | 6.260 | 3.219 | 2.127 | 5.698 | Resize: the float taps' shuffle lands each byte in its int32 lane |
| 16 | 8301afe | 6.230 | 3.936 | 2.110 | 5.632 | Resize: a destination row finished in one pass from its cached tap rows |
| 17 | 502fe49 | 6.168 | 3.826 | 2.096 | 5.577 | Resize: the shuffle table's window by a broadcast-128 load |
| 18 | dff0dbb | 5.974 | 3.761 | 2.061 | 5.423 | Roll up two kinds per lane add; fold the summary once per cell row |
| 19 | 69db70a | 6.080 | 3.902 | 1.982 | 5.459 | Summary across threads from three bands, every fold in its cell order |
| 20 | 7189851 | 5.964 | 3.707 | 1.918 | 5.410 | Summary after the join split by lane across the threads |
| 21 | c874e2d | 6.017 | 3.099 | 2.319 | 5.495 | Prefetch where the frame is first read; the finish converts in range; the kinds loop decides the block width once |
| 22 | 2e23b8e | 6.089 | 3.142 | 2.454 | 5.522 | Records prefetched ahead of the derive's stores; the resize's store bound tested once per group |
| 23 | c80dbfb | 5.886 | 3.080 | 1.919 | 5.292 | Cells derived and rolled up eight at a time, the group's kind stride a whole line |
| 24 | 858fe52 | 5.824 | 3.616 | 1.930 | 5.246 | Resize rounding: the saturating pack is the clamp |
| 25 | bac2b23 | 5.886 | 3.631 | 1.972 | 5.385 | Resize: the periodic groups of the horizontal pass with their slot tables in registers |
| 26 | cf40d37 | 5.845 | 4.552 | 2.420 | 5.280 | LBP ring: the neighbour test as a saturating subtract compared with zero |
| 27 | bde5b6c | 5.701 | 3.486 | 1.906 | 5.061 | Two-cell blocks widen in-lane; their int32 kinds fold from the 128-bit halves |
| 28 | fc5afd0 | 5.665 | 2.918 | 1.838 | 5.103 | HSV row: the >> 12 and the lane merges folded into the float path |
| 29 | 9544713 | 5.614 | 3.488 | 3.507 | 5.184 | Per-block bookkeeping: the group modulo as a mask; the bar detector's run bounds once per chunk |
| 30 | dfe62b0 | 5.738 | 3.490 | 1.920 | 5.172 | HSV row: one reciprocal carries HNUM, so hdiv's guess is one multiply |
| 31 | b4e65e4 | 5.646 | 3.545 | 1.956 | 5.119 | LBP phase: one loop down the rows, the nibbles emptied on the way |
