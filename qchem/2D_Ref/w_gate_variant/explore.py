#!/usr/bin/env python3
"""Fast exploration of EXPERIMENTAL gate variants of the scalar Q2D weight (original functional untouched).

Step 1 (python3 explore.py cache): for every active grid point store alpha, p, the original weight W0, the original
        alpha*p gate value G0 = g2(alpha p / 1e-5) and the branch differences
          exchange   dx = e_x^Q - e_x^r2SCAN   on each doubled spin density (spin scaling, weight 1/2)
          correlation dc = e_c^Q(n) - e_c^r2SCAN(n_up, n_dn)  on the total density (no spin scaling)
        for the 21 AE6/BH6 species (r2SCAN densities, dyall-v4z, grid 9) and the 19 IBM widths.
Step 2 (python3 explore.py): evaluate gate variants  W = W0 / G0 * G_new(alpha, p)  and print IBM, AE6, BH6 MAEs.
Gate families:  G = g2(alpha p / t) * g_k(p / p0)     (g_k(x) = x^k / (1 + x^k); p0 = 0 means no p-gate)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import numpy as np                                                    # noqa: E402

CACHE = os.path.join(HERE, "cache")


def g(x, k):
    xk = x ** k
    return xk / (1 + xk)


def active(n, gr2, tau):
    from q2d_scalar import CTF
    safe = np.where(n > 0, n, 1.0)
    pauli = tau - gr2 / (8 * safe)
    return (n > 1e-60) & (gr2 > 0) & (pauli > 0) & (pauli < CTF * n ** (5 / 3))


def points(n, gr2, tau, w, ec_r2=None):
    """Cached arrays on active points: w, alpha, p, W0, G0, dx (exchange) or dc (correlation)."""
    from q2d_scalar import _q2d, pyscf_baseline, CTF, K
    act = active(n, gr2, tau)
    na, ga, ta = n[act], gr2[act], tau[act]
    W0, qx, qc = _q2d(na, ga, ta)
    alpha = (ta - ga / (8 * na)) / (CTF * na ** (5 / 3))
    p = ga / (4 * K ** 2 * na ** (8 / 3))
    base = pyscf_baseline(na, ga, ta)
    diff = qx.value - base.exchange if ec_r2 is None else qc.value - ec_r2[act]
    return dict(w=w[act], a=alpha, p=p, W0=W0.value, G0=g(alpha * p / 1e-5, 2), d=diff)


def build_cache():
    from pyscf.dft import libxc
    import reproduce_ibm_benchmark as ibmb
    from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES
    os.makedirs(CACHE, exist_ok=True)
    for s in list(ATOM_SPIN) + list(MOLECULES):
        d = np.load(os.path.join(PARENT, "densities_dyall", s + ".npz"))
        rs, w = d["rho"], d["w"]
        out = {"e_tot": float(d["e_tot"])}
        for k, r in enumerate(rs):
            pt = points(2 * np.maximum(r[0], 0), 4 * np.einsum("ix,ix->x", r[1:4], r[1:4]), 2 * r[4], 0.5 * w)
            out.update({f"x{k}_{key}": v for key, v in pt.items()})
        n = np.maximum(rs[0][0] + rs[1][0], 0)
        gr = rs[0][1:4] + rs[1][1:4]
        ec = n * libxc.eval_xc("MGGA_C_R2SCAN", (rs[0], rs[1]), spin=1, deriv=0)[0]
        pt = points(n, np.einsum("ix,ix->x", gr, gr), rs[0][4] + rs[1][4], w, ec_r2=ec)
        out.update({f"c_{key}": v for key, v in pt.items()})
        np.savez(os.path.join(CACHE, s + ".npz"), **out)
        print("cached", s, flush=True)
    istls, _ = ibmb.reference_curves()
    ns = 1 / (np.pi * ibmb.RS2D ** 2)
    from q2d_scalar import pyscf_baseline
    for L in ibmb.WIDTHS:
        x, wq = np.polynomial.legendre.leggauss(ibmb.quad_points(L))
        y, wy = (x + 1) / 2, wq / 2
        f = 2 * np.sin(np.pi * y) ** 2
        fp = 2 * np.pi * np.sin(2 * np.pi * y)
        n = ns / L * f
        gr2 = (ns / L ** 2 * fp) ** 2
        tau = gr2 / (8 * n) + (np.pi * ns / 2) * n
        we = L * wy / ns                                   # per-electron weights
        base = pyscf_baseline(n, gr2, tau)
        px = points(n, gr2, tau, we)
        pc = points(n, gr2, tau, we, ec_r2=base.correlation)
        fock = ibmb.exact_fock_exchange(L)
        xc = float(istls(L))
        np.savez(os.path.join(CACHE, f"ibm_{L}.npz"), L=L, ex_r2=float(we @ base.exchange),
                 ec_r2=float(we @ base.correlation), fock=fock, istls=xc,
                 **{f"x_{k}": v for k, v in px.items()}, **{f"c_{k}": v for k, v in pc.items()})
        print("cached IBM", L, flush=True)


def correction(c, prefix, gate):
    W = c[prefix + "W0"] * np.divide(gate(c[prefix + "a"], c[prefix + "p"]), c[prefix + "G0"],
                                     out=np.zeros_like(c[prefix + "G0"]), where=c[prefix + "G0"] > 0)
    return float(c[prefix + "w"] @ (W * c[prefix + "d"]))


def evaluate_gate(gate, data):
    from reproduce_ae6_bh6 import AE6_REF, BH6, MOLECULES, HA2KCAL, read_xyz
    E = {s: c["e_tot"] + correction(c, "x0_", gate) + correction(c, "x1_", gate) + correction(c, "c_", gate)
         for s, c in data["species"].items()}
    ae = []
    for m, ref in AE6_REF.items():
        at = [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
        ae.append((sum(E[a] for a in at) - E[m]) * HA2KCAL - ref)
    bh = []
    for _, R, ts, P, ff, rr in BH6:
        bh += [(E[ts] - sum(E[x] for x in R)) * HA2KCAL - ff, (E[ts] - sum(E[x] for x in P)) * HA2KCAL - rr]
    ex = {"ge1": [], "lt1": []}
    exc = {"ge1": [], "lt1": []}
    for c in data["ibm"]:
        k = "ge1" if c["L"] >= 1 else "lt1"
        x = c["ex_r2"] + correction(c, "x_", gate)
        cc = c["ec_r2"] + correction(c, "c_", gate)
        ex[k].append(abs(x - c["fock"]))
        exc[k].append(abs(x + cc - c["istls"]))
    return (np.mean(np.abs(ae)), np.mean(np.abs(bh)), np.mean(ex["ge1"]), np.mean(ex["lt1"]),
            np.mean(exc["ge1"]), np.mean(exc["lt1"]), ae)


def load():
    import glob
    sp = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in glob.glob(os.path.join(CACHE, "*.npz"))
          if not os.path.basename(f).startswith(("ibm_", "td_"))}
    ibm = [dict(np.load(f)) for f in glob.glob(os.path.join(CACHE, "ibm_*.npz"))]
    return {"species": sp, "ibm": ibm}


if __name__ == "__main__":
    if sys.argv[1:] == ["cache"]:
        build_cache()
        sys.exit()
    data = load()
    rows = []
    print(f"{'t':>8} {'p0':>5} {'k':>2} | AE6 MAE  BH6 MAE | IBM MAE_x L>=1  L<1 | MAE_xc L>=1  L<1")
    for t in (1e-5, 1e-3, 1e-2, 1e-1):
        for p0, k in ((0, 0), (0.5, 6), (1.0, 6), (1.0, 12), (1.5, 6), (2.0, 6), (1.0, 4)):
            gate = (lambda a, p, t=t, p0=p0, k=k:
                    g(a * p / t, 2) * (g(p / p0, k) if p0 else 1.0))
            r = evaluate_gate(gate, data)
            rows.append((t, p0, k) + r[:6])
            print(f"{t:8.0e} {p0:5.2f} {k:2d} | {r[0]:8.2f} {r[1]:7.2f} | {r[2]:13.4f} {r[3]:.4f} |"
                  f" {r[4]:10.4f} {r[5]:.4f}", flush=True)
    import pandas as pd
    pd.DataFrame(rows, columns=["t", "p0", "k", "ae6_mae", "bh6_mae", "ibm_x_ge1", "ibm_x_lt1", "ibm_xc_ge1",
                                "ibm_xc_lt1"]).to_csv(os.path.join(HERE, "explore_results.csv"), index=False)
