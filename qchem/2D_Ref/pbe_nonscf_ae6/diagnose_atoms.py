#!/usr/bin/env python3
"""Diagnostics of the scalar exchange correction for the AE6 atoms (energy expression unchanged).

Tests (PBE orbitals, non-SCF):
  T1 spin decomposition  dEx^SS = 1/2 dEx[2 n_up] + 1/2 dEx[2 n_dn]: which spin channel carries the correction
  T2 atomic reference    integer-occupation UKS (used for AE6) vs ROKS vs fractional (spherical) occupation
  T3 grid                PySCF grid levels 3, 6, 9
  T4 Be-like analogue    Be atom (1s2 2s2, closed shell) vs the doubled beta density of C
Usage: python3 diagnose_atoms.py  (env AE6_BASIS, default def2-tzvp)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                    # noqa: E402
from pyscf import dft, gto, scf                       # noqa: E402

from q2d_scalar import pyscf_baseline, evaluate      # noqa: E402

BASIS = os.environ.get("AE6_BASIS", "def2-tzvp")
SPIN = {"C": 2, "O": 2, "Si": 2, "S": 2, "Be": 0}


def spin_rho(mol, mf):
    ni = dft.numint.NumInt()
    dm = mf.make_rdm1()
    if np.asarray(dm).ndim == 2:
        dm = np.array([dm / 2, dm / 2])
    ao = ni.eval_ao(mol, mf.grids.coords, deriv=1)
    return [ni.eval_rho(mol, ao, dm[s], xctype="MGGA", with_lapl=False) for s in (0, 1)], mf.grids.weights


def dEx_channel(r, w):
    """1/2 (E_x^scalar - E_x^r2SCAN)[2 n_sigma] for one spin channel (Ha)."""
    n2, g2, t2 = 2 * np.maximum(r[0], 0), 4 * np.einsum("ix,ix->x", r[1:4], r[1:4]), 2 * r[4]
    return 0.5 * float(w @ (evaluate(n2, g2, t2).exchange - pyscf_baseline(n2, g2, t2).exchange))


def atom(el, kind="uks", grid=6):
    mol = gto.M(atom=f"{el} 0 0 0", basis=BASIS, spin=SPIN[el], verbose=0)
    if kind == "uks":
        mf = dft.UKS(mol) if SPIN[el] else dft.RKS(mol)
    elif kind == "roks":
        mf = dft.ROKS(mol)
    else:                                   # fractional occupation of degenerate open shell (spherical)
        mf = scf.addons.frac_occ(dft.UKS(mol))
    mf.xc, mf.grids.level, mf.conv_tol, mf.max_cycle = "pbe", grid, 1e-9, 200
    mf.kernel()
    rs, w = spin_rho(mol, mf)
    a, b = dEx_channel(rs[0], w), dEx_channel(rs[1], w)
    return mf.e_tot, a, b, float(w @ (rs[0][0] + rs[1][0]))


if __name__ == "__main__":
    print(f"basis {BASIS}; dEx in mHa (alpha + beta = total SS correction)")
    print("T1/T2: atoms, grid 6")
    for el in ("C", "O", "Si", "S"):
        for kind in ("uks", "roks", "frac"):
            try:
                e, a, b, n = atom(el, kind)
                print(f"  {el:2s} {kind:5s} E={e:.6f}  dEx(alpha)={a*1e3:+9.3f}  dEx(beta)={b*1e3:+9.3f}  "
                      f"total={(a+b)*1e3:+9.3f}  N={n:.4f}")
            except Exception as exc:               # report and continue
                print(f"  {el:2s} {kind:5s} failed: {exc}")
    print("T3: C atom UKS, grid levels")
    for g in (3, 6, 9):
        e, a, b, n = atom("C", "uks", g)
        print(f"  grid {g}: dEx(alpha)={a*1e3:+9.3f}  dEx(beta)={b*1e3:+9.3f}  total={(a+b)*1e3:+9.3f}")
    print("T4: Be atom (closed shell 1s2 2s2), grid 6")
    e, a, b, n = atom("Be", "uks")
    print(f"  Be: dEx total={(a+b)*1e3:+9.3f} mHa  (= dEx[n] of a 4-electron 1s2 2s2 density)")
