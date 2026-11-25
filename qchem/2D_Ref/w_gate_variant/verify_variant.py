#!/usr/bin/env python3
"""Independent check of the gate variants (no cache): the weight is rebuilt from its closed formula
    W = cutoff(alpha) * g2(alpha p / t) * g6(p / (6 sqrt(alpha + 1e-6))) * g_k(p / p0)
and compared with the cached shortcut W0 / g2(alpha p / 1e-5) * g2(alpha p / t) g_k(p / p0); then AE6 and BH6 are
recomputed for SS and TD from the stored r2SCAN densities, and the W = 0 / W = 1 limits are re-checked.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import numpy as np                                                            # noqa: E402
from pyscf.dft import libxc                                                   # noqa: E402
from q2d_scalar import _q2d, pyscf_baseline, CTF, K                           # noqa: E402
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES, AE6_REF, BH6, HA2KCAL, read_xyz   # noqa: E402

VARIANTS = {"original": (1e-5, 0, 0), "SS-G3 t=1e-3 p0=5 k=12": (1e-3, 5.0, 12),
            "TD t=1e-4 p0=6 k=12": (1e-4, 6.0, 12)}
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "jana_t":
    VARIANTS = {"Jana constant + p-factor p0=5 k=12": (1e-5, 5.0, 12), "Jana constant + p-factor p0=6 k=12": (1e-5, 6.0, 12)}


def gk(x, k):
    xk = x ** k
    return xk / (1 + xk)


def ingredients(n, g2, tau):
    safe = np.where(n > 0, n, 1.0)
    pauli = tau - g2 / (8 * safe)
    act = (n > 1e-60) & (g2 > 0) & (pauli > 0) & (pauli < CTF * n ** (5 / 3))
    W0, qx, qc = _q2d(n[act], g2[act], tau[act])
    a = pauli[act] / (CTF * n[act] ** (5 / 3))
    p = g2[act] / (4 * K ** 2 * n[act] ** (8 / 3))
    return act, W0.value, qx.value, qc.value, a, p


def weight(a, p, t, p0, k):
    """Closed formula of the (variant) weight, written out independently of _q2d."""
    x = np.clip((a - 0.6) / 0.4, 0, 1)
    cut = np.where(a <= 0.6, 1.0, (1 - x) ** 3 * (1 + 3 * x + 6 * x * x))
    W = cut * gk(a * p / t, 2) * gk(p / (6 * (a + 1e-6) ** 0.5), 6)
    return W * (gk(p / p0, k) if p0 else 1.0)


def dexc(rs, w, var, route, check):
    t, p0, k = var
    dEx = 0.0
    if route == "SS":
        for r in rs:
            n2, g2, t2 = 2 * np.maximum(r[0], 0), 4 * np.einsum("ix,ix->x", r[1:4], r[1:4]), 2 * r[4]
            act, W0, qx, _, a, p = ingredients(n2, g2, t2)
            if not act.any():
                continue
            W = weight(a, p, t, p0, k)
            check.append(np.max(np.abs(weight(a, p, 1e-5, 0, 0) - W0)))       # closed formula == _q2d weight
            dEx += 0.5 * float(w[act] @ (W * (qx - pyscf_baseline(n2[act], g2[act], t2[act]).exchange)))
    n = np.maximum(rs[0][0] + rs[1][0], 0)
    gr = rs[0][1:4] + rs[1][1:4]
    g2 = np.einsum("ix,ix->x", gr, gr)
    tau = rs[0][4] + rs[1][4]
    act, W0, qx, qc, a, p = ingredients(n, g2, tau)
    if not act.any():
        return dEx, 0.0
    W = weight(a, p, t, p0, k)
    check.append(np.max(np.abs(weight(a, p, 1e-5, 0, 0) - W0)))
    ra, rb = rs[0][:, act], rs[1][:, act]
    if route == "TD":
        ex = n[act] * libxc.eval_xc("MGGA_X_R2SCAN", (ra, rb), spin=1, deriv=0)[0]
        dEx = float(w[act] @ (W * (qx - ex)))
    ec = n[act] * libxc.eval_xc("MGGA_C_R2SCAN", (ra, rb), spin=1, deriv=0)[0]
    return dEx, float(w[act] @ (W * (qc - ec)))


def main():
    names = list(ATOM_SPIN) + list(MOLECULES)
    dens = {s: np.load(os.path.join(PARENT, "densities_dyall", s + ".npz")) for s in names}
    check = []
    for vname, var in VARIANTS.items():
        for route in ("SS", "TD"):
            E = {}
            for s in names:
                dx, dc = dexc(dens[s]["rho"], dens[s]["w"], var, route, check)
                E[s] = float(dens[s]["e_tot"]) + dx + dc
            ae = {m: (sum(E[a] for a in [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]) - E[m])
                  * HA2KCAL - ref for m, ref in AE6_REF.items()}
            bh = []
            for _, R, ts, P, ff, rr in BH6:
                bh += [(E[ts] - sum(E[x] for x in R)) * HA2KCAL - ff, (E[ts] - sum(E[x] for x in P)) * HA2KCAL - rr]
            print(f"{vname:24s} {route}: AE6 MAE {np.mean(np.abs(list(ae.values()))):8.2f}  "
                  f"C3H4 {ae['C3H4']:+8.2f}  C2H2O2 {ae['C2H2O2']:+8.2f}  BH6 MAE {np.mean(np.abs(bh)):6.2f}  "
                  f"BH6 ME {np.mean(bh):+6.2f}", flush=True)
    print(f"closed-formula weight vs _q2d weight (original t): max |diff| = {max(check):.2e}")


if __name__ == "__main__":
    main()
