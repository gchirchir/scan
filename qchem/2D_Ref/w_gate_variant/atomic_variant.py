#!/usr/bin/env python3
"""Atomic benchmark of jana_ref (Table 2/3 of main.tex) for the EXPERIMENTAL gate variants: HF densities, dyall-v4z,
grid level 9 (as reproduce_atomic.py), closed-shell atoms He..Xe, Mg (SS = TD for closed shells; H: W = 0).
Prints the scalar-minus-r2SCAN shifts dE_x, dE_c (mHa) of the original functional, of the reference PDF, and of
each variant."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from pyscf import dft, gto, scf                                  # noqa: E402
from reproduce_atomic import PDF_TABLE                           # noqa: E402
from verify_variant import ingredients, weight                   # noqa: E402
from q2d_scalar import pyscf_baseline                            # noqa: E402

VARIANTS = {"original": (1e-5, 0, 0), "G2 t=1e-3 p0=4 k=6": (1e-3, 4.0, 6), "G3 t=1e-3 p0=5 k=12": (1e-3, 5.0, 12),
            "TD-best t=1e-4 p0=6 k=12": (1e-4, 6.0, 12), "SS t=1e-2 p0=8 k=12": (1e-2, 8.0, 12),
            "Jana t + p0=5 k=12": (1e-5, 5.0, 12), "Jana t + p0=6 k=12": (1e-5, 6.0, 12)}
rows = []
for atom in ("He", "Ne", "Ar", "Kr", "Xe", "Mg"):
    mol = gto.M(atom=f"{atom} 0 0 0", basis="dyall-v4z", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    grids = dft.gen_grid.Grids(mol)
    grids.level = 9
    grids.build()
    ni = dft.numint.NumInt()
    ao = ni.eval_ao(mol, grids.coords, deriv=1)
    r = ni.eval_rho(mol, ao, mf.make_rdm1(), xctype="MGGA", with_lapl=False)
    n, g2, tau, w = r[0], np.einsum("ix,ix->x", r[1:4], r[1:4]), r[4], grids.weights
    act, W0, qx, qc, a, p = ingredients(n, g2, tau)
    base = pyscf_baseline(n[act], g2[act], tau[act])
    pdf = PDF_TABLE[atom]
    row = dict(atom=atom, pdf_dEx=1e3 * (pdf[1] - pdf[0]), pdf_dEc=1e3 * (pdf[3] - pdf[2]))
    for name, (t, p0, k) in VARIANTS.items():
        W = weight(a, p, t, p0, k)
        row[f"{name}|dEx"] = 1e3 * float(w[act] @ (W * (qx - base.exchange)))
        row[f"{name}|dEc"] = 1e3 * float(w[act] @ (W * (qc - base.correlation)))
    rows.append(row)
    print(atom, {k: round(v, 3) for k, v in row.items() if k != "atom"}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "atomic_variant_v2.csv"), index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print("\ndE_x (mHa):")
print(df[["atom", "pdf_dEx"] + [c for c in df.columns if c.endswith("|dEx")]].round(3).to_string(index=False))
print("\ndE_c (mHa):")
print(df[["atom", "pdf_dEc"] + [c for c in df.columns if c.endswith("|dEc")]].round(3).to_string(index=False))
