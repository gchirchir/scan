#!/usr/bin/env python3
"""Frozen-core test of the ORIGINAL scalar functional (q2d_scalar.py unchanged): does the AE6 failure come from the
core electrons?  PBE orbitals (non-SCF, as in ../pbe_nonscf_ae6), def2-TZVP, grid level 6 (env FC_BASIS, FC_GRID).

Core orbitals per atom (per spin): H 0; C, O 1 (1s); Si, S 5 (1s 2s 2p).  For each species the lowest
N_core MOs of each spin are the core; valence density n_val = n - n_core (per spin).

  full     : E = E^r2SCAN[n] + dE_xc[n]                    (as in Table 9; dE_xc = scalar - r2SCAN, SS spin)
  FC-val   : E = E^r2SCAN[n] + dE_xc[n_val]                (correction from the valence density only)
  FC-mask  : E = E^r2SCAN[n] + dE_xc[n] with W -> W * n_val/n pointwise  (no correction where the core dominates)
Exchange: exact spin scaling; correlation: no spin scaling (W, Q2D branch from the total density).
Run: python3 run_fc.py            -> fc_species.csv, fc_ae6.csv
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from pyscf import dft, gto                                           # noqa: E402
from pyscf.dft import libxc                                          # noqa: E402

from q2d_scalar import _q2d, pyscf_baseline, CTF                     # noqa: E402
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES, AE6_REF, HA2KCAL, read_xyz   # noqa: E402

BASIS = os.environ.get("FC_BASIS", "def2-tzvp")
GRID = int(os.environ.get("FC_GRID", "6"))
NCORE = {"H": 0, "C": 1, "O": 1, "Si": 5, "S": 5}


def act_mask(n, g2, tau):
    safe = np.where(n > 0, n, 1.0)
    pauli = tau - g2 / (8 * safe)
    return (n > 1e-60) & (g2 > 0) & (pauli > 0) & (pauli < CTF * n ** (5 / 3))


def dexc(rs, w, scale=None):
    """dE_x (spin scaled) + dE_c (no spin scaling) of the original functional; W multiplied by scale_s if given."""
    dEx = 0.0
    for s, r in enumerate(rs):
        n2, g2, t2 = 2 * np.maximum(r[0], 0), 4 * np.einsum("ix,ix->x", r[1:4], r[1:4]), 2 * r[4]
        a = act_mask(n2, g2, t2)
        if a.any():
            W, qx, _ = _q2d(n2[a], g2[a], t2[a])
            fac = 1.0 if scale is None else scale[s][a]
            dEx += 0.5 * float(w[a] @ (W.value * fac * (qx.value - pyscf_baseline(n2[a], g2[a], t2[a]).exchange)))
    n = np.maximum(rs[0][0] + rs[1][0], 0)
    gr = rs[0][1:4] + rs[1][1:4]
    g2 = np.einsum("ix,ix->x", gr, gr)
    tau = rs[0][4] + rs[1][4]
    a = act_mask(n, g2, tau)
    dEc = 0.0
    if a.any():
        W, _, qc = _q2d(n[a], g2[a], tau[a])
        ec = n[a] * libxc.eval_xc("MGGA_C_R2SCAN", (rs[0][:, a], rs[1][:, a]), spin=1, deriv=0)[0]
        fac = 1.0 if scale is None else scale[2][a]
        dEc = float(w[a] @ (W.value * fac * (qc.value - ec)))
    return dEx, dEc


def species(name):
    if name in ATOM_SPIN:
        atom, spin, els = f"{name} 0 0 0", ATOM_SPIN[name], [name]
    else:
        xyz, spin = MOLECULES[name]
        atom = read_xyz(xyz)
        els = [l.split()[0] for l in atom.split("\n")]
    ncore = sum(NCORE[e] for e in els)
    mol = gto.M(atom=atom, basis=BASIS, spin=spin, verbose=0)
    mf = dft.UKS(mol)
    mf.xc, mf.grids.level, mf.conv_tol = "pbe", GRID, 1e-9
    mf.kernel()
    ni = dft.numint.NumInt()
    dm = mf.make_rdm1()
    e_r2 = mf.e_tot - float(ni.nr_uks(mol, mf.grids, "pbe", dm)[1]) + float(ni.nr_uks(mol, mf.grids, "r2scan", dm)[1])
    ao = ni.eval_ao(mol, mf.grids.coords, deriv=1)
    w = mf.grids.weights
    full, core = [], []
    for s in (0, 1):
        C, occ = mf.mo_coeff[s], mf.mo_occ[s]
        dmc = C[:, :ncore] @ C[:, :ncore].T
        full.append(ni.eval_rho(mol, ao, dm[s], xctype="MGGA", with_lapl=False))
        core.append(ni.eval_rho(mol, ao, dmc, xctype="MGGA", with_lapl=False))
    full, core = np.array(full), np.array(core)
    val = full - core
    dX, dC = dexc(full, w)
    vX, vC = dexc(val, w)
    frac = [np.clip(np.divide(val[s][0], full[s][0], out=np.zeros_like(w), where=full[s][0] > 1e-30), 0, 1)
            for s in (0, 1)]
    ntot = full[0][0] + full[1][0]
    frac.append(np.clip(np.divide(val[0][0] + val[1][0], ntot, out=np.zeros_like(w), where=ntot > 1e-30), 0, 1))
    mX, mC = dexc(full, w, scale=frac)
    d = dict(species=name, ncore_per_spin=ncore, e_r2scan=e_r2, dEx_full=dX, dEc_full=dC, dEx_val=vX, dEc_val=vC,
             dEx_mask=mX, dEc_mask=mC)
    print(f"{name:7s} ncore={ncore}  dEx full/val/mask = {dX*1e3:+9.2f} {vX*1e3:+9.2f} {mX*1e3:+9.2f} mHa   "
          f"dEc = {dC*1e3:+7.2f} {vC*1e3:+7.2f} {mC*1e3:+7.2f}", flush=True)
    return d


def main():
    names = list(ATOM_SPIN) + list(AE6_REF)
    sp = pd.DataFrame([species(n) for n in names]).set_index("species")
    sp.to_csv(os.path.join(HERE, f"fc_species_{BASIS}_g{GRID}.csv"), float_format="%.8f")
    rows = []
    for m, ref in AE6_REF.items():
        at = [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
        r = dict(molecule=m, ref=ref)
        for lab, cx, cc in (("r2scan", None, None), ("full", "dEx_full", "dEc_full"),
                            ("FC_val", "dEx_val", "dEc_val"), ("FC_mask", "dEx_mask", "dEc_mask")):
            E = sp.e_r2scan + (0 if cx is None else sp[cx] + sp[cc])
            r[lab] = (sum(E[a] for a in at) - E[m]) * HA2KCAL
        rows.append(r)
    ae = pd.DataFrame(rows)
    ae.to_csv(os.path.join(HERE, f"fc_ae6_{BASIS}_g{GRID}.csv"), index=False, float_format="%.4f")
    print(ae.round(2).to_string(index=False))
    for k in ("r2scan", "full", "FC_val", "FC_mask"):
        e = ae[k] - ae.ref
        print(f"{k:8s} ME={e.mean():+9.2f}  MAE={e.abs().mean():9.2f} kcal/mol")


if __name__ == "__main__":
    main()
