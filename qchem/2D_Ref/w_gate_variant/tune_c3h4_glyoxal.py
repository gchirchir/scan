#!/usr/bin/env python3
"""Tune the EXPERIMENTAL gate  W = W0/G0 * g2(alpha p/t) * g_k(p/p0)  for propyne (C3H4) and glyoxal (C2H2O2),
for both spin routes SS (exact spin scaling of E_x) and TD (W and Q2D E_x from the total density, mixed with
spin-polarized r2SCAN E_x).  Correlation: no spin scaling in both.  r2SCAN densities (dyall-v4z, grid 9).
The other AE6 molecules, BH6 (SS) and the IBM benchmark are reported for every candidate so that nothing is
tuned at their expense.  Uses cache/ from explore.py; adds cache/td_<atom>.npz for the TD exchange of C and O.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, HERE)

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from explore import load, evaluate_gate, correction, g, active, CACHE          # noqa: E402
from reproduce_ae6_bh6 import AE6_REF, MOLECULES, HA2KCAL, read_xyz             # noqa: E402

OPEN = ["C", "O", "Si", "S", "S2"]   # all open-shell AE6 species (TD differs from SS); H has W = 0
# (fixed 2026-11-24: an earlier version listed only C and O, so Si, S and S2 used SS in the TD AE6 MAE)


def td_cache():
    from q2d_scalar import _q2d, CTF, K
    from pyscf.dft import libxc
    for s in OPEN:
        f = os.path.join(CACHE, f"td_{s}.npz")
        if os.path.exists(f):
            continue
        d = np.load(os.path.join(PARENT, "densities_dyall", s + ".npz"))
        rs, w = d["rho"], d["w"]
        n = np.maximum(rs[0][0] + rs[1][0], 0)
        gr = rs[0][1:4] + rs[1][1:4]
        g2 = np.einsum("ix,ix->x", gr, gr)
        tau = rs[0][4] + rs[1][4]
        a = active(n, g2, tau)
        W0, qx, _ = _q2d(n[a], g2[a], tau[a])
        ex = n[a] * libxc.eval_xc("MGGA_X_R2SCAN", (rs[0][:, a], rs[1][:, a]), spin=1, deriv=0)[0]
        alpha = (tau[a] - g2[a] / (8 * n[a])) / (CTF * n[a] ** (5 / 3))
        p = g2[a] / (4 * K ** 2 * n[a] ** (8 / 3))
        np.savez(f, w=w[a], a=alpha, p=p, W0=W0.value, G0=g(alpha * p / 1e-5, 2), d=qx.value - ex)


def errors(gate, data, td):
    """AE6 errors (kcal/mol) for SS and TD."""
    out = {}
    for route in ("SS", "TD"):
        E = {}
        for s, c in data["species"].items():
            if route == "TD" and s in OPEN:
                dx = correction(td[s], "", gate)
            else:
                dx = correction(c, "x0_", gate) + correction(c, "x1_", gate)
            E[s] = c["e_tot"] + dx + correction(c, "c_", gate)
        out[route] = {m: (sum(E[a] for a in [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")])
                          - E[m]) * HA2KCAL - ref for m, ref in AE6_REF.items()}
    return out


if __name__ == "__main__":
    td_cache()
    data = load()
    td = {s: dict(np.load(os.path.join(CACHE, f"td_{s}.npz"))) for s in OPEN}
    rows = []
    for t in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2):
        for p0 in (0, 2, 3, 4, 5, 6, 7, 8):
            for k in ((0,) if p0 == 0 else (4, 6, 12)):
                gate = lambda a, p, t=t, p0=p0, k=k: g(a * p / t, 2) * (g(p / p0, k) if p0 else 1.0)  # noqa: E731
                e = errors(gate, data, td)
                r = evaluate_gate(gate, data)          # SS: AE6 MAE, BH6 MAE, IBM
                rows.append(dict(t=t, p0=p0, k=k,
                                 SS_C3H4=e["SS"]["C3H4"], SS_C2H2O2=e["SS"]["C2H2O2"],
                                 TD_C3H4=e["TD"]["C3H4"], TD_C2H2O2=e["TD"]["C2H2O2"],
                                 SS_AE6=np.mean(np.abs(list(e["SS"].values()))),
                                 TD_AE6=np.mean(np.abs(list(e["TD"].values()))),
                                 BH6_SS=r[1], ibm_x_ge1=r[2], ibm_x_lt1=r[3]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "tune_c3h4_glyoxal.csv"), index=False, float_format="%.4f")
    df["SS_two"] = df[["SS_C3H4", "SS_C2H2O2"]].abs().max(axis=1)
    df["TD_two"] = df[["TD_C3H4", "TD_C2H2O2"]].abs().max(axis=1)
    pd.set_option("display.width", 220)
    for route in ("SS", "TD"):
        print(f"\n=== {route}: best candidates for propyne + glyoxal (max |error|), by IBM thin-well limit")
        for lim in (0.05, 0.06, 0.07, 0.08, 1.0):
            sub = df[df.ibm_x_lt1 <= lim].sort_values(f"{route}_two").head(1)
            print(f"IBM L<1 <= {lim}:")
            print(sub[["t", "p0", "k", f"{route}_C3H4", f"{route}_C2H2O2", f"{route}_AE6", "BH6_SS", "ibm_x_ge1",
                       "ibm_x_lt1"]].round(4).to_string(index=False))
