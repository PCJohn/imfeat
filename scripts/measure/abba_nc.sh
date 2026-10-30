#!/bin/bash
# pinned to N idle physical cores, A/B/B/A over R rounds: abba_nc.sh N R pkgA pkgB script args...
S=$(dirname $(readlink -f $0))
N=$1; R=$2; A=$3; B=$4; shift 4
cores=${ABBA_CORES:-$(python3 $S/idle_cores.py $N)}
ta=""; tb=""
for rnd in $(seq 1 $R); do
  for p in A B B A; do
    pk=$([ $p = A ] && echo $A || echo $B)
    t=$(PYTHONPATH=$pk taskset -c $cores python $S/"$@" | tail -1 | awk '{print $(NF-1)}')
    if [ $p = A ]; then ta="$ta $t"; else tb="$tb $t"; fi
  done
done
python3 - "$*" "$cores" "$(basename $A)" "$(basename $B)" "$ta" "$tb" <<'PY'
import sys, statistics
cfg, cores, na, nb, ta, tb = sys.argv[1:7]
a = [float(x) for x in ta.split()]; b = [float(x) for x in tb.split()]
for n, v in ((na, a), (nb, b)):
    print(f"{cfg} {n:>8s} cpus {cores:<8s} min {min(v):7.3f} ({100*(min(v)-min(a))/min(a):+5.1f}%)  median {statistics.median(v):7.3f} ({100*(statistics.median(v)-statistics.median(a))/statistics.median(a):+5.1f}%)  n={len(v)}")
PY
