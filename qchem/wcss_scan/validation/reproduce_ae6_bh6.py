#!/usr/bin/env python3
"""AE6 atomization energies and BH6 barrier heights with r2SCAN and the scalar Q2D functional (q2d_scalar.py).

Densities: self-consistent r2SCAN (PySCF Libxc), RKS for closed shells, UKS for open shells, dyall-v4z basis and
PySCF grid level 9 as in the atomic tables (env AE6_BASIS, AE6_GRID; def2-QZVPP/grid 6 results: species_def2qzvpp/).  The scalar functional is evaluated on these densities (post-SCF, like the atomic benchmark):
    E_scalar = E_r2SCAN + (E_xc^scalar - E_xc^r2SCAN)[n_up, n_down].

Spin extension (the scalar form of jana_ref is defined for spin-unpolarized densities only):
  exchange A   exact spin scaling, E_x[n_up, n_dn] = 1/2 E_x[2 n_up] + 1/2 E_x[2 n_dn], applied to the full scalar
               mixture (weight, Q2D branch and r2SCAN baseline evaluated on 2 n_sigma, 4 |grad n_sigma|^2, 2 tau_sigma);
               this is the rule jana_ref uses for the one-electron endpoint (H);
  exchange B   weight W and Q2D exchange from the total density, mixed with spin-polarized r2SCAN exchange (the
               form is validated on spin-unpolarized densities only; route A fails for Be-like 1s2 2s2 spin densities,
               e.g. the beta density of the C atom: closed-shell Be gets dEx = +1636 mHa on its HF density);
  correlation  E_c = int n [(1 - W) eps_c^r2SCAN(n_up, n_dn) + W eps_c^Q(n)], with W and the Q2D correlation branch
               from the total density (n, |grad n|^2, tau); exact for closed shells, an approximation for open shells;
               for one-electron densities alpha = 0, W = 0 and E_c = E_c^r2SCAN = 0, as in jana_ref.

Geometries (geometries/): AE6 from the Minnesota MGAE109 database, BH6 from HTBH38 (Lynch & Truhlar).
Reference values: Haunschild & Klopper, Theor. Chem. Acc. 131, 1112 (2012) (s00214-012-1112-3.pdf), Table 4,
column E_nrel (nonrelativistic, all-electron: FC-CCSD(T)(F12)/cc-pVQZ-F12 + dHE + dC/CV), kJ/mol / 4.184 -> kcal/mol.
(Previously Lynch & Truhlar 2003 values; they differ by up to 2.2 kcal/mol, S2 and SiH4.)
Run:  python3 reproduce_ae6_bh6.py species NAME  -> species/NAME.json   (one species; numpy only; cluster job)
      python3 reproduce_ae6_bh6.py assemble      -> ae6_bh6_species.csv, ae6_results.csv, bh6_results.csv
"""
import os
import sys

import json

import numpy as np
from pyscf import dft, gto
from pyscf.dft import libxc

from q2d_scalar import _q2d, pyscf_baseline, evaluate, CTF

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(HERE, "geometries")
SPC = os.path.join(HERE, os.environ.get("AE6_SPECIES_DIR", "species"))
HA2KCAL = 627.509474
BASIS = os.environ.get("AE6_BASIS", "dyall-v4z")   # same basis and grid as the atomic tables (reproduce_atomic.py)
GRID = int(os.environ.get("AE6_GRID", "9"))
DENS = os.path.join(HERE, os.environ.get("AE6_DENSITY_DIR", "densities"))

ATOM_SPIN = {"H": 1, "C": 2, "O": 2, "Si": 2, "S": 2}           # ground-state 2S of the free atoms
MOLECULES = {  # name: (xyz file, 2S)
    "SiH4": ("SiH4", 0), "SiO": ("SiO", 0), "S2": ("S2", 2), "C3H4": ("C3H4_pro", 0),
    "C2H2O2": ("HCOCOH", 0), "C4H8": ("C4H8_cyc", 0),
    "OH": ("OH", 1), "CH4": ("CH4", 0), "CH3": ("CH3", 1), "H2O": ("H2O", 0), "H2": ("H2", 0), "H2S": ("H2S", 0),
    "HS": ("HS", 1), "TS_OH_CH4": ("TS_OH_CH4", 1), "TS_H_OH": ("TS_H_OH", 2), "TS_H_H2S": ("TS_H_H2S", 1),
}
KJ2KCAL = 1 / 4.184
AE6_REF = {k: v * KJ2KCAL for k, v in {"SiH4": 1358.1, "SiO": 809.2, "S2": 434.7, "C3H4": 2950.9, "C2H2O2": 2658.2,
                                       "C4H8": 4812.1}.items()}
AE6_LABEL = {"SiH4": "SiH$_4$", "SiO": "SiO", "S2": "S$_2$", "C3H4": "C$_3$H$_4$ (propyne)",
             "C2H2O2": "C$_2$H$_2$O$_2$ (glyoxal)", "C4H8": "C$_4$H$_8$ (cyclobutane)"}
BH6 = [  # (label, reactants, TS, products, ref forward, ref reverse)
    ("OH + CH$_4$ $\\to$ CH$_3$ + H$_2$O", ("OH", "CH4"), "TS_OH_CH4", ("CH3", "H2O"), 25.7 * KJ2KCAL, 81.7 * KJ2KCAL),
    ("H + OH $\\to$ O + H$_2$", ("H", "OH"), "TS_H_OH", ("O", "H2"), 45.3 * KJ2KCAL, 54.6 * KJ2KCAL),
    ("H + H$_2$S $\\to$ H$_2$ + HS", ("H", "H2S"), "TS_H_H2S", ("H2", "HS"), 16.0 * KJ2KCAL, 71.4 * KJ2KCAL),
]


def read_xyz(name):
    lines = open(os.path.join(GEO, name + ".xyz")).read().strip().split("\n")
    return "\n".join(lines[2:2 + int(lines[0])])


def r2scan_polarized(code, rho_a, rho_b):
    """Per-volume spin-polarized r2SCAN exchange or correlation energy density (Libxc)."""
    exc = libxc.eval_xc(code, (rho_a, rho_b), spin=1, deriv=0)[0]
    return (rho_a[0] + rho_b[0]) * exc


def delta_xc(mol, mf):
    """E_xc^scalar - E_xc^r2SCAN on the SCF density: exchange by route A (exact spin scaling) and route B (weight and
    Q2D exchange from the total density, mixed with spin-polarized r2SCAN exchange), and correlation."""
    g = mf.grids
    ni = dft.numint.NumInt()
    dm = mf.make_rdm1()
    if dm.ndim == 2:
        dm = np.array([dm / 2, dm / 2])
    # AOs evaluated in grid blocks (full-grid AOs of C4H8/def2-QZVPP with gradients would need ~8 GB)
    rs = [[], []]
    for ao, mask, wb, coords in ni.block_loop(mol, g, mol.nao, deriv=1, max_memory=2000):
        for s_ in (0, 1):
            rs[s_].append(ni.eval_rho(mol, ao, dm[s_], mask, xctype="MGGA", hermi=1, with_lapl=False))
    rs = [np.hstack(r) for r in rs]   # n, grad, tau per spin
    w = g.weights
    delta_xc.last = (np.array(rs), w)
    # exchange: spin scaling of the full scalar mixture
    dEx = 0.0
    for r in rs:
        n2, g2, t2 = 2 * np.maximum(r[0], 0), 4 * np.einsum("ix,ix->x", r[1:4], r[1:4]), 2 * r[4]
        base, scal = pyscf_baseline(n2, g2, t2), evaluate(n2, g2, t2)
        dEx += 0.5 * float(w @ (scal.exchange - base.exchange))
    # correlation: weight and Q2D branch from the total density, r2SCAN correlation spin-polarized
    n = np.maximum(rs[0][0] + rs[1][0], 0)
    gr = rs[0][1:4] + rs[1][1:4]
    g2 = np.einsum("ix,ix->x", gr, gr)
    tau = rs[0][4] + rs[1][4]
    safe = np.where(n > 0, n, 1.0)
    pauli = tau - g2 / (8 * safe)
    act = (n > 1e-60) & (g2 > 0) & (pauli > 0) & (pauli < CTF * n ** (5 / 3))
    dEc = dExB = 0.0
    if act.any():
        weight, exch, corr = _q2d(n[act], g2[act], tau[act])
        ec_r2 = r2scan_polarized("MGGA_C_R2SCAN", rs[0][:, act], rs[1][:, act])
        ex_r2 = r2scan_polarized("MGGA_X_R2SCAN", rs[0][:, act], rs[1][:, act])
        dEc = float(w[act] @ (weight.value * (corr.value - ec_r2)))
        dExB = float(w[act] @ (weight.value * (exch.value - ex_r2)))
    return dEx, dExB, dEc


def species(name, atom_string, spin):
    mol = gto.M(atom=atom_string, basis=BASIS, spin=spin, charge=0, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol)
    mf.xc = "r2scan"
    mf.grids.level = GRID
    mf.conv_tol, mf.max_cycle = 1e-9, 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    if not mf.converged:
        raise SystemExit(f"r2SCAN SCF not converged for {name}")
    dEx, dExB, dEc = delta_xc(mol, mf)
    os.makedirs(DENS, exist_ok=True)
    np.savez_compressed(os.path.join(DENS, name + ".npz"), rho=delta_xc.last[0], w=delta_xc.last[1], e_tot=mf.e_tot,
                        converged=mf.converged)
    print(f"{name:10s} 2S={spin} E(r2SCAN)={mf.e_tot:.8f}  dEx(A)={dEx * 1000:+.4f} dEx(B)={dExB * 1000:+.4f} "
          f"dEc={dEc * 1000:+.4f} mHa", flush=True)
    return dict(species=name, spin2S=spin, nao=mol.nao, ngrid=int(mf.grids.weights.size), e_r2scan=float(mf.e_tot),
                dEx_A=dEx, dEx_B=dExB, dEc=dEc, basis=BASIS, grid=GRID, e_scalarA=float(mf.e_tot) + dEx + dEc,
                e_scalarB=float(mf.e_tot) + dExB + dEc, pyscf=__import__("pyscf").__version__)


def run_species(name):
    if name in ATOM_SPIN:
        d = species(name, f"{name} 0 0 0", ATOM_SPIN[name])
    else:
        xyz, s = MOLECULES[name]
        d = species(name, read_xyz(xyz), s)
    os.makedirs(SPC, exist_ok=True)
    with open(os.path.join(SPC, name + ".json"), "w") as fh:
        json.dump(d, fh, indent=1)


def assemble():
    import pandas as pd
    names = list(ATOM_SPIN) + list(MOLECULES)
    missing = [n for n in names if not os.path.exists(os.path.join(SPC, n + ".json"))]
    if missing:
        print("missing species:", " ".join(missing))
    sp = pd.DataFrame([json.load(open(os.path.join(SPC, n + ".json"))) for n in names if n not in missing])
    sp = sp.set_index("species")
    sp.to_csv(os.path.join(HERE, "ae6_bh6_species.csv"), float_format="%.10f")
    E = {k: sp[k] for k in ("e_r2scan", "e_scalarA", "e_scalarB")}
    have = set(sp.index)

    def atoms_of(xyz):
        return [l.split()[0] for l in read_xyz(xyz).split("\n")]

    ae = []
    for m, ref in AE6_REF.items():
        at = atoms_of(MOLECULES[m][0])
        if not set(at + [m]) <= have:
            continue
        d = dict(molecule=m, label=AE6_LABEL[m], ref=ref)
        for k, lab in (("e_r2scan", "r2scan"), ("e_scalarA", "scalarA"), ("e_scalarB", "scalarB")):
            d[lab] = (sum(E[k][a] for a in at) - E[k][m]) * HA2KCAL
        ae.append(d)
    ae = pd.DataFrame(ae)
    ae.to_csv(os.path.join(HERE, "ae6_results.csv"), index=False, float_format="%.4f")
    bh = []
    for lab, R, ts, P, ff, rr in BH6:
        for direction, side, ref in (("forward", R, ff), ("reverse", P, rr)):
            if not set(side + (ts,)) <= have:
                continue
            d = dict(reaction=lab, direction=direction, ref=ref)
            for k, l in (("e_r2scan", "r2scan"), ("e_scalarA", "scalarA"), ("e_scalarB", "scalarB")):
                d[l] = (E[k][ts] - sum(E[k][x] for x in side)) * HA2KCAL
            bh.append(d)
    bh = pd.DataFrame(bh, columns=["reaction", "direction", "ref", "r2scan", "scalarA", "scalarB"])
    bh.to_csv(os.path.join(HERE, "bh6_results.csv"), index=False, float_format="%.4f")
    for df, name in ((ae, "AE6"), (bh, "BH6")):
        if len(df) == 0:
            continue
        for f in ("r2scan", "scalarA", "scalarB"):
            e = df[f] - df["ref"]
            print(f"{name} ({len(df)}) {f:7s} ME {e.mean():+.2f}  MAE {e.abs().mean():.2f} kcal/mol")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "species":
        for nm in sys.argv[2:]:
            run_species(nm)
    elif sys.argv[1:] == ["assemble"]:
        assemble()
    else:
        raise SystemExit("usage: reproduce_ae6_bh6.py species NAME [NAME ...] | assemble")
