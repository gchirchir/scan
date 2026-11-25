#!/usr/bin/env python3
"""Non-self-consistent (energy-only) r2SCAN and scalar-Q2D AE6 atomization energies on PBE orbitals.

For each species: self-consistent PBE (RKS closed shell / UKS open shell), dyall-v4z, PySCF grid level 9 (same
basis, grid, geometries and atomic spin states as the r2SCAN-density AE6 table of main.tex). Then, on the fixed
PBE orbitals (no potential, no SCF for the new functionals):

    E^F = E^PBE - E_xc^PBE[n_up, n_dn] + E_xc^F[n_up, n_dn]

    F = r2SCAN : E_xc from Libxc MGGA_X_R2SCAN + MGGA_C_R2SCAN (spin-polarized)
    F = scalar : E_xc^scalar = E_xc^r2SCAN + (E_xc^scalar - E_xc^r2SCAN), the difference evaluated with the
                 unchanged delta_xc() of ../reproduce_ae6_bh6.py (q2d_scalar.py), i.e. the same energy
                 expression as in main.tex; SS and TD are its two spin extensions of E_x, E_c is common.

Run:  python3 run_pbe_nonscf.py species NAME   -> species/NAME.json
      python3 run_pbe_nonscf.py assemble       -> ae6_pbe_nonscf.csv (kcal/mol)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import numpy as np                                  # noqa: E402
from pyscf import dft, gto                          # noqa: E402

from reproduce_ae6_bh6 import (ATOM_SPIN, MOLECULES, AE6_REF, AE6_LABEL, HA2KCAL,  # noqa: E402
                               read_xyz, delta_xc)

BASIS = os.environ.get("AE6_BASIS", "dyall-v4z")
GRID = int(os.environ.get("AE6_GRID", "9"))
TAG = "" if (BASIS, GRID) == ("dyall-v4z", 9) else f"_{BASIS}_g{GRID}"   # default run keeps species/
SPC = os.path.join(HERE, "species" + TAG)
AE6_MOL = list(AE6_REF)


def exc_on(ni, mol, grids, xc, dm, spin):
    """E_xc[n] of functional xc on the fixed density matrix dm (no potential used)."""
    if spin:
        return float(ni.nr_uks(mol, grids, xc, dm)[1])
    return float(ni.nr_rks(mol, grids, xc, dm)[1])


def species(name):
    if name in ATOM_SPIN:
        atom, spin = f"{name} 0 0 0", ATOM_SPIN[name]
    else:
        xyz, spin = MOLECULES[name]
        atom = read_xyz(xyz)
    mol = gto.M(atom=atom, basis=BASIS, spin=spin, charge=0, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol)
    mf.xc = "pbe"
    mf.grids.level = GRID
    mf.conv_tol, mf.max_cycle = 1e-9, 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    if not mf.converged:
        raise SystemExit(f"PBE SCF not converged for {name}")
    dm = mf.make_rdm1()
    ni = dft.numint.NumInt()
    g = mf.grids
    exc_pbe = exc_on(ni, mol, g, "pbe", dm, spin)
    ex_r2 = exc_on(ni, mol, g, "MGGA_X_R2SCAN", dm, spin)
    ec_r2 = exc_on(ni, mol, g, "MGGA_C_R2SCAN", dm, spin)
    exc_scf = float(getattr(mf.get_veff(mol, dm), "exc"))   # PBE E_xc as used by the SCF (consistency check)
    dEx_SS, dEx_TD, dEc = delta_xc(mol, mf)                # unchanged scalar energy expression
    e_r2 = mf.e_tot - exc_pbe + ex_r2 + ec_r2
    d = dict(species=name, spin2S=spin, basis=BASIS, grid=GRID, nao=mol.nao, e_pbe=float(mf.e_tot),
             exc_pbe=exc_pbe, exc_pbe_scf=exc_scf, ex_r2scan=ex_r2, ec_r2scan=ec_r2,
             dEx_SS=dEx_SS, dEx_TD=dEx_TD, dEc=dEc, e_r2scan=e_r2,
             e_SS=e_r2 + dEx_SS + dEc, e_TD=e_r2 + dEx_TD + dEc)
    print(f"{name:7s} 2S={spin} E(PBE)={mf.e_tot:.8f} E(r2SCAN@PBE)={e_r2:.8f} "
          f"dEx(SS)={dEx_SS*1e3:+.3f} dEx(TD)={dEx_TD*1e3:+.3f} dEc={dEc*1e3:+.3f} mHa "
          f"[Exc check {abs(exc_pbe - exc_scf):.1e}]", flush=True)
    os.makedirs(SPC, exist_ok=True)
    with open(os.path.join(SPC, name + ".json"), "w") as fh:
        json.dump(d, fh, indent=1)


def assemble():
    import pandas as pd
    sp = {n: json.load(open(os.path.join(SPC, n + ".json")))
          for n in list(ATOM_SPIN) + AE6_MOL if os.path.exists(os.path.join(SPC, n + ".json"))}
    rows = []
    for m in AE6_MOL:
        at = [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
        if not set(at + [m]) <= set(sp):
            continue
        r = dict(molecule=m, label=AE6_LABEL[m], ref=AE6_REF[m])
        for k in ("pbe", "r2scan", "SS", "TD"):
            r[k] = (sum(sp[a]["e_" + k] for a in at) - sp[m]["e_" + k]) * HA2KCAL
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, f"ae6_pbe_nonscf{TAG}.csv"), index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    for k in ("pbe", "r2scan", "SS", "TD"):
        e = df[k] - df["ref"]
        print(f"{k:7s} ME={e.mean():+8.2f}  MAE={e.abs().mean():8.2f} kcal/mol")


if __name__ == "__main__":
    if sys.argv[1] == "species":
        species(sys.argv[2])
    else:
        assemble()
