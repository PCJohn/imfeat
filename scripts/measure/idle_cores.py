import sys, time
n = int(sys.argv[1])
def rd():
    d = {}
    for l in open('/proc/stat'):
        if l.startswith('cpu') and l[3].isdigit():
            f = l.split(); d[int(f[0][3:])] = int(f[1]) + int(f[3])
    return d
a = rd(); time.sleep(0.5); b = rd()
# physical cores 0..7 (siblings c and c+8): rank by the pair's busy time, take the n idlest
order = sorted(range(8), key=lambda c: (b[c] - a[c]) + (b[c + 8] - a[c + 8]))
print(",".join(str(c) for c in order[:n]))
