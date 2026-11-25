#!/usr/bin/env python3
"""Limit checks W = 0 and W = 1 of the spin-polarized implementation (standard spin treatment) on the r2SCAN densities
of all 21 AE6/BH6 species (dyall-v4z, grid 9; ../densities_dyall).
  W = 0: E_x, E_c of the implementation vs Libxc spin-polarized r2SCAN (MGGA_X_R2SCAN, MGGA_C_R2SCAN)
  W = 1: E_x of the implementation vs the pure Q2D exchange (spin-scaled), computed separately
Writes check_limits_dyall.csv."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from pyscf.dft import libxc                                           # noqa: E402
from validate_limits import polarized_xc, q2d_only                    # noqa: E402
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES                    # noqa: E402
rows = []
for s in list(ATOM_SPIN) + list(MOLECULES):
    d = np.load(os.path.join(os.path.dirname(HERE), "densities_dyall", s + ".npz"))
    rs, w = d["rho"], d["w"]
    n = rs[0][0] + rs[1][0]
    ex_lx = float(w @ (n * libxc.eval_xc("MGGA_X_R2SCAN", (rs[0], rs[1]), spin=1)[0]))
    ec_lx = float(w @ (n * libxc.eval_xc("MGGA_C_R2SCAN", (rs[0], rs[1]), spin=1)[0]))
    ex0, ec0 = polarized_xc(rs, w, 0)
    ex1, _ = polarized_xc(rs, w, 1)
    exq = q2d_only(rs, w)
    rows.append(dict(species=s, ex_libxc=ex_lx, ex_W0=ex0, dEx_W0=ex0 - ex_lx, ec_libxc=ec_lx, ec_W0=ec0,
                     dEc_W0=ec0 - ec_lx, ex_Q2D=exq, ex_W1=ex1, dEx_W1=ex1 - exq))
    print(f"{s:10s} W=0: dEx {ex0-ex_lx:+.1e} dEc {ec0-ec_lx:+.1e} Ha   W=1: dEx {ex1-exq:+.1e} Ha", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "check_limits_dyall.csv"), index=False, float_format="%.15e")
print("max |W=0 dEx| %.1e  max |W=0 dEc| %.1e  max |W=1 dEx| %.1e Ha" % (df.dEx_W0.abs().max(), df.dEc_W0.abs().max(),
                                                                      df.dEx_W1.abs().max()))
