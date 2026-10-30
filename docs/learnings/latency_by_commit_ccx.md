| # | commit | target, 1 thread | target, 2 threads | target, 4 threads | 1024x1024x3 finest-64 s1, 1 thread (4 levels, no conversion) | subject |
|---|---|---|---|---|---|---|
| 00 | 3957392 | 9.652 | 5.011 | 2.665 |  | Add AGENTS.md |
| 01 | 4715561 | 7.528 | 3.929 | 2.113 |  | Kernel as register-resident phases; cheaper folds; signed HOG sums |
| 02 | 3fdcbc4 | 7.314 | 3.804 | 2.054 |  | Extrema from the LBP ring, nibble-counted LBP bins, two shared sums |
| 03 | 0497dc3 | 7.317 | 3.818 | 2.062 |  | Sliding plane window: every kernel tap a constant offset from the row |
| 04 | a69e7bd | 7.039 | 3.648 | 1.983 |  | A batch of small kernel items, kept by pinned A/B timing: -10% at stride 1 |
| 05 | 455e92f | 7.046 | 3.670 | 1.985 |  | Batched fold for one-cell blocks: four kinds reduced together |
| 06 | 6c166f2 | 6.865 | 3.591 | 1.940 |  | Vector hash row sums; batched multi-cell folds for the HOG and pair kinds |
| 07 | 3ffd3cc | 6.939 | 3.596 | 1.950 |  | Level 1 derived in its band from the rolled-up sums; a vector flush |
| 08 | a3a86d2 | 6.883 | 3.592 | 1.942 |  | Split the two phases of a raw row as it is gathered |
| 09 | d29e6b1 | 6.764 | 3.488 | 1.892 |  | Derive and roll up cells in groups of four, whatever the block width |
| 10 | 1a381c0 | 6.464 | 3.333 | 1.809 |  | Halve the HOG ray tests by the rays' mirror symmetry; bar detector sums in registers |
| 11 | 20c7d4b | 6.416 | 3.317 | 1.804 |  | Give the bar detector phase a function of its own |
| 12 | b510004 | 6.383 | 3.326 | 1.808 |  | Energy phase at stride 1 as three chunk-wide product sums per response column |
| 13 | 875d14e | 6.178 | 3.194 | 1.742 |  | HSV conversion: the table step as one rounding of the float guess |
| 14 | c9c0c1c | 6.150 | 3.190 | 1.749 |  | Resize: the horizontal pass over all of a row's planes at once |
| 15 | cd2a2ea | 6.271 | 3.227 | 1.754 |  | Resize: the float taps' shuffle lands each byte in its int32 lane |
| 16 | 8301afe | 6.183 | 3.183 | 1.734 |  | Resize: a destination row finished in one pass from its cached tap rows |
| 17 | 502fe49 | 6.215 | 3.218 | 1.745 |  | Resize: the shuffle table's window by a broadcast-128 load |
| 18 | dff0dbb | 5.964 | 3.121 | 1.700 |  | Roll up two kinds per lane add; fold the summary once per cell row |
| 19 | 69db70a | 6.040 | 3.151 | 1.670 |  | Summary across threads from three bands, every fold in its cell order |
| 20 | 7189851 | 5.987 | 3.088 | 1.621 |  | Summary after the join split by lane across the threads |
| 21 | c874e2d | 5.982 | 3.069 | 1.614 |  | Prefetch where the frame is first read; the finish converts in range; the kinds loop decides the block width once |
| 22 | 2e23b8e | 5.962 | 3.054 | 1.614 |  | Records prefetched ahead of the derive's stores; the resize's store bound tested once per group |
| 23 | c80dbfb | 5.847 | 3.012 | 1.593 |  | Cells derived and rolled up eight at a time, the group's kind stride a whole line |
| 24 | 858fe52 | 5.833 | 3.016 | 1.591 |  | Resize rounding: the saturating pack is the clamp |
| 25 | bac2b23 | 5.885 | 3.023 | 1.607 |  | Resize: the periodic groups of the horizontal pass with their slot tables in registers |
| 26 | cf40d37 | 5.811 | 2.975 | 1.574 |  | LBP ring: the neighbour test as a saturating subtract compared with zero |
