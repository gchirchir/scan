#!/usr/bin/env python3
"""Integrate the PDF's atomic r2SCAN and scalar-Q2D energies on PySCF HF densities.

The PDF does not identify the AtomicOrbitals basis/grid. This script records its
own explicit basis and grid so the calculation is repeatable and the limitation
is visible in the output metadata.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyscf import dft, gto, scf

from q2d_scalar import one_electron, pyscf_baseline, evaluate

ATOMS = ("H", "He", "Ne", "Ar", "Kr", "Xe", "Mg")
# Non-relativistic numerical Hartree-Fock limits (Ha), used to check that the Gaussian HF density is converged
# (C. Froese Fischer, The Hartree-Fock Method for Atoms, Wiley 1977; S. Lehtola, Int. J. Quantum Chem. 119, e25945
# (2019), HelFEM).  The PDF's densities are numerical HF densities from AtomicOrbitals.
HF_LIMIT = {"H": -0.5, "He": -2.861679996, "Ne": -128.547098109, "Ar": -526.817512803, "Kr": -2752.054977347,
            "Xe": -7232.138363872, "Mg": -199.614636425}
# scalar_functional_form.pdf, Sec. 6 table: Ex(r2SCAN), Ex(scalar), Ec(r2SCAN), Ec(scalar) in Ha
PDF_TABLE = {"H": (-0.312499, -0.312499, 0.0, 0.0), "He": (-1.030576, -1.030576, -0.037928, -0.037928),
             "Ne": (-12.144091, -12.140430, -0.347036, -0.347232), "Ar": (-30.182159, -30.181454, -0.697125, -0.697228),
             "Kr": (-93.820243, -93.819752, -1.770054, -1.770151), "Xe": (-178.832464, -178.831801, -2.918254, -2.918396),
             "Mg": (-16.006525, -15.940927, -0.412308, -0.409780)}


def calculate(atom, basis, grid_level):
    mol = gto.M(atom=f"{atom} 0 0 0", basis=basis, unit="Bohr", spin=1 if atom == "H" else 0,
                charge=0, verbose=0)
    mf = scf.UHF(mol) if atom == "H" else scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"HF did not converge for {atom}")

    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()
    ni = dft.numint.NumInt()
    ao = ni.eval_ao(mol, grids.coords, deriv=1)
    dm = mf.make_rdm1()
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    rho = ni.eval_rho(mol, ao, dm, xctype="MGGA", hermi=1, with_lapl=False)
    n = np.maximum(rho[0], 0.0)
    g2 = np.einsum("ix,ix->x", rho[1:4], rho[1:4])
    tau = rho[4]
    if atom == "H":
        base = one_electron(n, g2, tau)
        scalar = base
    else:
        base = pyscf_baseline(n, g2, tau)
        scalar = evaluate(n, g2, tau)
    w = grids.weights
    return {
        "atom": atom, "basis": basis, "grid_level": grid_level,
        "hf_energy_ha": float(mf.e_tot), "hf_limit_ha": HF_LIMIT[atom],
        "hf_minus_limit_mha": 1000 * (float(mf.e_tot) - HF_LIMIT[atom]),
        "electrons_on_grid": float(np.dot(grids.weights, n)),
        "r2scan_exchange_ha": float(np.dot(w, base.exchange)),
        "scalar_exchange_ha": float(np.dot(w, scalar.exchange)),
        "r2scan_correlation_ha": float(np.dot(w, base.correlation)),
        "scalar_correlation_ha": float(np.dot(w, scalar.correlation)),
        "grid_points": int(len(w)), "hf_converged": bool(mf.converged),
        "pdf_r2scan_exchange_ha": PDF_TABLE[atom][0], "pdf_scalar_exchange_ha": PDF_TABLE[atom][1],
        "pdf_r2scan_correlation_ha": PDF_TABLE[atom][2], "pdf_scalar_correlation_ha": PDF_TABLE[atom][3],
    }


def main():
    p = argparse.ArgumentParser()
    # Dyall valence quadruple-zeta (primitive, all-electron): HF within 0.02-0.04 mHa of the numerical HF limit for
    # Kr and Xe (tested 2026-11-23).  Uncontracted ANO-RCC misses the limit by 1.8 (Kr) and 6.3 mHa (Xe), which
    # shifts Ex by +0.7 and +2.7 mHa; contracted ANO-RCC (scalar-relativistic contraction) misses it by 25 and 264 Ha.
    p.add_argument("--basis", default="dyall-v4z")
    p.add_argument("--grid-level", type=int, default=9)
    p.add_argument("--output", default="atomic_results.csv")
    args = p.parse_args()
    rows = [calculate(atom, args.basis, args.grid_level) for atom in ATOMS]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    Path("atomic_run_metadata.json").write_text(json.dumps({
        "basis": args.basis, "grid_level": args.grid_level,
        "pyscf_version": __import__("pyscf").__version__,
        "note": "PDF densities: numerical HF from AtomicOrbitals. Here: PySCF HF in the given Gaussian basis, checked "
                "against the numerical HF limit (hf_minus_limit_mha in atomic_results.csv).",
    }, indent=2) + "\n")
    for r in rows:
        print(f"{r['atom']:>2}  Ex(r2SCAN) {r['r2scan_exchange_ha']: .6f}  Ex(scalar) {r['scalar_exchange_ha']: .6f}  "
              f"Ec(r2SCAN) {r['r2scan_correlation_ha']: .6f}  Ec(scalar) {r['scalar_correlation_ha']: .6f}")


if __name__ == "__main__":
    main()
