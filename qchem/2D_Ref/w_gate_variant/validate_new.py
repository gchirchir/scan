#!/usr/bin/env python3
"""Validation of the selected new weight (G3: t = 1e-3, p0 = 5, k = 12; SS spin route) by independent recomputation.
  V1  closed-formula weight on the dyall-v4z / grid 9 r2SCAN densities (no cache)       -> AE6, BH6
  V2  independent r2SCAN densities: separate SCF rerun, dyall-v4z, grid 9 (densities_rerun_dyall) -> AE6, BH6
  V5  IBM benchmark recomputed directly with the closed-formula weight (no cache)       -> MAE E_x, E_xc
  V7  sensitivity: neighbours of (t, p0, k) from tune_c3h4_glyoxal.csv
Writes validate_new.csv and prints a summary.  (V3 PBE orbitals: ../pbe_nonscf_ae6/run_pbe_new.py; V4 limits:
validate_limits.log, verify_variant.log; V6 atoms: atomic_exact_exchange.log.)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)

import numpy as np                                                                       # noqa: E402
import pandas as pd                                                                      # noqa: E402
from verify_variant import dexc, ingredients, weight                                     # noqa: E402
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES, AE6_REF, BH6, HA2KCAL, read_xyz       # noqa: E402
from q2d_scalar import pyscf_baseline                                                    # noqa: E402
import reproduce_ibm_benchmark as ibmb                                                   # noqa: E402

from wnew_params import NEW, ORIG                                                        # noqa: E402
G3 = NEW   # proposed W_new
names = list(ATOM_SPIN) + list(MOLECULES)


def ae_bh(dir_, var, route="SS"):
    E = {}
    for s in names:
        d = np.load(os.path.join(PARENT, dir_, s + ".npz"))
        e = float(d["e_tot"])
        if var is not None:
            dx, dc = dexc(d["rho"], d["w"], var, route, [])
            e += dx + dc
        E[s] = e
    ae = [(sum(E[a] for a in [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]) - E[m]) * HA2KCAL - ref
          for m, ref in AE6_REF.items()]
    bh = []
    for _, R, ts, P, ff, rr in BH6:
        bh += [(E[ts] - sum(E[x] for x in R)) * HA2KCAL - ff, (E[ts] - sum(E[x] for x in P)) * HA2KCAL - rr]
    return np.mean(np.abs(ae)), np.mean(np.abs(bh)), ae


def ibm(var):
    istls, _ = ibmb.reference_curves()
    ns = 1 / (np.pi * ibmb.RS2D ** 2)
    ex = {"ge1": [], "lt1": []}
    exc = {"ge1": [], "lt1": []}
    for L in ibmb.WIDTHS:
        x, wq = np.polynomial.legendre.leggauss(ibmb.quad_points(L))
        y, wy = (x + 1) / 2, wq / 2
        f = 2 * np.sin(np.pi * y) ** 2
        fp = 2 * np.pi * np.sin(2 * np.pi * y)
        n = ns / L * f
        g2 = (ns / L ** 2 * fp) ** 2
        tau = g2 / (8 * n) + (np.pi * ns / 2) * n
        we = L * wy / ns
        base = pyscf_baseline(n, g2, tau)
        act, W0, qx, qc, a, p = ingredients(n, g2, tau)
        W = weight(a, p, *var)
        Ex = float(we @ base.exchange) + float(we[act] @ (W * (qx - base.exchange[act])))
        Ec = float(we @ base.correlation) + float(we[act] @ (W * (qc - base.correlation[act])))
        k = "ge1" if L >= 1 else "lt1"
        fock = ibmb.exact_fock_exchange(L)
        ex[k].append(abs(Ex - fock))
        exc[k].append(abs(Ex + Ec - float(istls(L))))
    return np.mean(ex["ge1"]), np.mean(ex["lt1"]), np.mean(exc["ge1"]), np.mean(exc["lt1"])


rows = []
DSETS = [("V1 dyall-v4z/grid 9", "densities_dyall"), ("V2 dyall-v4z/grid 9 (independent SCF rerun)", "densities_rerun_dyall")]
if os.environ.get("SKIP_V2"):
    DSETS = DSETS[:1]
for lab, dir_ in DSETS:
    for vname, var, route in (("r2SCAN", None, "SS"), ("original", ORIG, "SS"), ("original TD", ORIG, "TD"),
                              ("new G3", G3, "SS"), ("new TD", G3, "TD")):
        a, b, errs = ae_bh(dir_, var, route)
        rows.append(dict(test=lab, functional=vname, ae6_mae=a, bh6_mae=b, **{f"err_{m}": e for m, e in zip(AE6_REF, errs)}))
        print(f"{lab:34s} {vname:9s} AE6 MAE {a:9.2f}  BH6 MAE {b:6.2f}", flush=True)
for vname, var in (("original", ORIG), ("new G3", G3)):
    r = ibm(var)
    rows.append(dict(test="V5 IBM direct", functional=vname, ibm_x_ge1=r[0], ibm_x_lt1=r[1], ibm_xc_ge1=r[2], ibm_xc_lt1=r[3]))
    print(f"V5 IBM direct  {vname:9s} MAE_x {r[0]:.4f}/{r[1]:.4f}  MAE_xc {r[2]:.4f}/{r[3]:.4f}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "validate_new.csv"), index=False, float_format="%.4f")

tu = pd.read_csv(os.path.join(HERE, "tune_c3h4_glyoxal.csv"))
# sensitivity of the proposed W_new (Jana constant kept, stored as t = 0.0 in the 4-decimal csv): p0 and k
nb = tu[(tu.t == 0.0) & (tu.p0.isin([4, 5, 6, 7, 8])) & (tu.k.isin([6, 12]))]
nb = nb[["p0", "k", "SS_AE6", "TD_AE6", "BH6_SS", "ibm_x_ge1", "ibm_x_lt1"]].sort_values(["k", "p0"])
nb.to_csv(os.path.join(HERE, "validate_new_sensitivity.csv"), index=False, float_format="%.4f")
print("\nV7 sensitivity:")
print(nb.round(4).to_string(index=False))
