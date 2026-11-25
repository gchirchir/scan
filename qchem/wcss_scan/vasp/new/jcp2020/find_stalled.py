import re, subprocess, glob
out = subprocess.run(["squeue","-h","-r","-u","gabchi5762","-t","RUNNING","-o","%i %j"],capture_output=True,text=True).stdout.split("\n")
stalled = []
for line in out:
    if not line.strip(): continue
    jid, name = line.split()
    if name not in ("jcp20surf","jcp20lat","jcp20fig1","jcp20relat"): continue
    arr, idx = jid.split("_")
    f = f"logs/jcp2020_{arr}_{idx}.out"
    try: txt = open(f).read()
    except OSError: continue
    k = txt.rfind("rhokerker stagnated")              # guarded code already handled a stall: look only after it
    if k >= 0:
        txt = txt[k + 20:]
    r = [float(x) for x in re.findall(r"rhokerker it +\d+ rms\(rho_out-rho_in\) = ([0-9.eE+-]+)", txt)]
    w = r[-8:]                                        # guarded code stops after 6: act only after 8
    if len(w) == 8 and max(w) < 1e-5 and (max(w)-min(w)) < 0.01*max(w):
        task = re.search(r"(surface|lattice) --system \S+( --facet \S+)? --functional \S+", txt)
        stalled.append((jid, name, task.group(1) if task else "?", w[-1]))
for s in stalled: print("STALLED", *s)
