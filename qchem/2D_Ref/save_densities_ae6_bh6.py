#!/usr/bin/env python3
"""Re-run the r2SCAN SCF of every AE6/BH6 species (same settings as reproduce_ae6_bh6.py) and store the density on
the grid (n, grad n, tau per spin) and the weights in densities/<NAME>.npz, for fast post-SCF scans of functional
variants (scan_weight_variants.py)."""
import os, sys
import numpy as np
from pyscf import dft, gto
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES, BASIS, GRID, read_xyz

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "densities")


def run(name):
    if name in ATOM_SPIN:
        atom, spin = f"{name} 0 0 0", ATOM_SPIN[name]
    else:
        atom, spin = read_xyz(MOLECULES[name][0]), MOLECULES[name][1]
    mol = gto.M(atom=atom, basis=BASIS, spin=spin, charge=0, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol)
    mf.xc, mf.grids.level, mf.conv_tol, mf.max_cycle = "r2scan", GRID, 1e-9, 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton(); mf.kernel()
    dm = mf.make_rdm1()
    if dm.ndim == 2:
        dm = np.array([dm / 2, dm / 2])
    ni = dft.numint.NumInt()
    rs = [[], []]
    for ao, mask, wb, coords in ni.block_loop(mol, mf.grids, mol.nao, deriv=1, max_memory=2000):
        for s in (0, 1):
            rs[s].append(ni.eval_rho(mol, ao, dm[s], mask, xctype="MGGA", hermi=1, with_lapl=False))
    rs = np.array([np.hstack(r) for r in rs])
    np.savez_compressed(os.path.join(OUT, name + ".npz"), rho=rs, w=mf.grids.weights, e_tot=mf.e_tot,
                        converged=mf.converged)
    print(f"{name:10s} E={mf.e_tot:.8f} conv={mf.converged} npts={mf.grids.weights.size}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in sys.argv[1:] or list(ATOM_SPIN) + list(MOLECULES):
        run(n)
