#!/usr/bin/env python3
"""AE6/BH6 per entry for the modified gate constant c (q2d_scalar.GATE_C) on stored r2SCAN densities.

Columns: ref, r2scan, jana_ref (c = 6) and gate (c = C), all spin route B for open shells.  Evaluates the full
functional with q2d_scalar.set_gate_constant (not the ratio used in scan_weight_variants.py).
Run:  python3 gate_variant_results.py DENSITY_DIR C  -> ae6_gate.csv, bh6_gate.csv
"""
import glob, os, sys
import numpy as np
from pyscf.dft import libxc
import q2d_scalar as q
from reproduce_ae6_bh6 import AE6_REF, AE6_LABEL, BH6, MOLECULES, read_xyz, HA2KCAL
from scan_weight_variants import active

HERE = os.path.dirname(os.path.abspath(__file__))


def delta(d, c):
    rho, w = d["rho"], d["w"]
    n = np.maximum(rho[0][0] + rho[1][0], 0)
    gr = rho[0][1:4] + rho[1][1:4]
    g2 = np.einsum("ix,ix->x", gr, gr)
    tau = rho[0][4] + rho[1][4]
    a = active(n, g2, tau)
    if not a.any():
        return 0.0
    q.set_gate_constant(c)
    W, ex, ec = q._q2d(n[a], g2[a], tau[a])
    q.set_gate_constant(6)
    ra, rb = rho[0][:, a], rho[1][:, a]
    exr = (ra[0] + rb[0]) * libxc.eval_xc("MGGA_X_R2SCAN", (ra, rb), spin=1, deriv=0)[0]
    ecr = (ra[0] + rb[0]) * libxc.eval_xc("MGGA_C_R2SCAN", (ra, rb), spin=1, deriv=0)[0]
    return float(w[a] @ (W.value * (ex.value - exr + ec.value - ecr)))


def main(ddir, c):
    E = {"r2scan": {}, "jana": {}, "gate": {}}
    for f in sorted(glob.glob(os.path.join(HERE, ddir, "*.npz"))):
        nm, d = os.path.basename(f)[:-4], dict(np.load(f))
        e0 = float(d["e_tot"])
        E["r2scan"][nm], E["jana"][nm], E["gate"][nm] = e0, e0 + delta(d, 6), e0 + delta(d, c)
    atoms_of = lambda m: [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
    rows = []
    for m, ref in AE6_REF.items():
        r = dict(molecule=m, label=AE6_LABEL[m], ref=ref)
        for k in E:
            r[k] = (sum(E[k][a] for a in atoms_of(m)) - E[k][m]) * HA2KCAL
        rows.append(r)
    bh = []
    for lab, R, ts, P, ff, rr in BH6:
        for direction, side, ref in (("forward", R, ff), ("reverse", P, rr)):
            r = dict(reaction=lab, direction=direction, ref=ref)
            for k in E:
                r[k] = (E[k][ts] - sum(E[k][x] for x in side)) * HA2KCAL
            bh.append(r)
    import pandas as pd
    for df, out in ((pd.DataFrame(rows), "ae6_gate.csv"), (pd.DataFrame(bh), "bh6_gate.csv")):
        df.to_csv(os.path.join(HERE, out), index=False, float_format="%.4f")
        for k in E:
            e = df[k] - df["ref"]
            print(f"{out[:3].upper()} {k:7s} ME {e.mean():+7.2f} MAE {e.abs().mean():7.2f} max {e.abs().max():7.2f}")
    open(os.path.join(HERE, "gate_meta.txt"), "w").write(f"densities={ddir} c={c}\n")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
