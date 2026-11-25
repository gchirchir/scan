#!/usr/bin/env python3
"""Absolute exchange and correlation energies (Ha) of every AE6/BH6 species on its r2SCAN density (dyall-v4z, grid 9):
r2SCAN (spin-polarized Libxc) and scalar Q2D (spin routes A and B for exchange, as in reproduce_ae6_bh6.py).
Input: densities_dyall/<NAME>.npz and species_dyall/<NAME>.json.  Output: ae6_bh6_xc.csv."""
import csv, json, os
import numpy as np
from pyscf.dft import libxc
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for name in list(ATOM_SPIN) + list(MOLECULES):
    d = np.load(os.path.join(HERE, "densities_dyall", name + ".npz"))
    sp = json.load(open(os.path.join(HERE, "species_dyall", name + ".json")))
    ra, rb, w = d["rho"][0], d["rho"][1], d["w"]
    n = ra[0] + rb[0]
    ex = float(w @ (n * libxc.eval_xc("MGGA_X_R2SCAN", (ra, rb), spin=1, deriv=0)[0]))
    ec = float(w @ (n * libxc.eval_xc("MGGA_C_R2SCAN", (ra, rb), spin=1, deriv=0)[0]))
    rows.append(dict(species=name, spin2S=sp["spin2S"], electrons=float(w @ n), ex_r2scan=ex, ex_scalarA=ex + sp["dEx_A"],
                     ex_scalarB=ex + sp["dEx_B"], ec_r2scan=ec, ec_scalar=ec + sp["dEc"],
                     dEx_A_mHa=1000 * sp["dEx_A"], dEx_B_mHa=1000 * sp["dEx_B"], dEc_mHa=1000 * sp["dEc"]))
    print(f"{name:10s} N={rows[-1]['electrons']:.6f} Ex={ex:.6f} Ec={ec:.6f}", flush=True)
with open(os.path.join(HERE, "ae6_bh6_xc.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
