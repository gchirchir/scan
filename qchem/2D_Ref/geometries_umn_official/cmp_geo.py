import sys, os, glob, numpy as np
loc, umn = sys.argv[1], sys.argv[2]
def rd(p):
    L=open(p).read().split('\n'); n=int(L[0].split()[0])
    at=[l.split() for l in L[2:2+n]]
    return [a[0].capitalize() for a in at], np.array([[float(x) for x in a[1:4]] for a in at]), L[1].strip()
def sig(e,x):
    d=np.linalg.norm(x[:,None]-x[None],axis=-1); i,j=np.triu_indices(len(e),1)
    return sorted(e), np.sort(d[i,j])
def comp(e):
    return tuple(sorted(e))
umnfiles=glob.glob(os.path.join(umn,'MGAE109','*.xyz'))+glob.glob(os.path.join(umn,'HTBH38','*.xyz'))
U={}
for f in umnfiles:
    try: U[f]=rd(f)
    except Exception: pass
for f in sorted(glob.glob(os.path.join(loc,'*.xyz'))):
    e,x,_=rd(f); se,sd=sig(e,x)
    best=None
    for g,(e2,x2,c) in U.items():
        if comp(e2)!=comp(e): continue
        dev=np.abs(sig(e2,x2)[1]-sd).max()
        if best is None or dev<best[0]: best=(dev,g,c)
    if best: print(f"{os.path.basename(f):14s} -> {os.path.relpath(best[1],umn):22s} max|d dist|={best[0]:.5f} A   [{best[2][:50]}]")
    else: print(f"{os.path.basename(f):14s} no UMN file with same composition")
