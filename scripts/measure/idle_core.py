import time
def rd():
    d={}
    for l in open('/proc/stat'):
        if l.startswith('cpu') and l[3].isdigit():
            f=l.split(); d[int(f[0][3:])]=int(f[1])+int(f[3])
    return d
a=rd(); time.sleep(0.5); b=rd()
# prefer a core whose SMT sibling is also idle: siblings are c and c+8 on this box
print(min(range(8), key=lambda c: (b[c]-a[c]) + (b[c+8]-a[c+8])))
