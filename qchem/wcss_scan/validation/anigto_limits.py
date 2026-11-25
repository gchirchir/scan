#!/usr/bin/env python3
"""W = 0 / W = 1 limit checks of the scalar Q2D functional (2D_Ref/main.pdf) on aniGTO sheet and wire densities.

Densities: closed-shell, fixed orbitals, anisotropic Gaussian-Hermite (aniGTO trap basis, jcp_main Sec. multiorbital)
    phi_k(r) = prod_d  w_d^(1/4) h_{k_d}(sqrt(w_d) x_d),     doubly occupied, N = 2 * (number of orbitals)
    wire : w = (lam^2, lam^2, 1), orbitals (0,0,k), k < M            (s, p_z, d_zz along the wire)
    sheet: w = (1, 1, lam^2),     orbitals (0,0,0), (1,0,0), (0,1,0) (s, p_x, p_y in the plane)
N = 2 (one orbital) has tau = tau_W, alpha = 0, so W = 0 identically; N = 4, 6 have alpha > 0 and switch W on.

Per system (Ha):
  exact E_x       separable Coulomb kernel 1/r = 2/sqrt(pi) int exp(-t^2 r^2) dt, Gauss-Hermite (exact for the
                  polynomial factors); checked against jcp_main Eqs.(10),(11) (N=2) and codes/08 (N=4,6 wires)
  Libxc           MGGA_X_R2SCAN, MGGA_C_R2SCAN, MGGA_X_SCAN evaluated directly
  W = 0, W = 1    pbe_nonscf_ae6/validate_limits.unpolarized_mix (the code of Table 9), W forced to 0 or 1
  pure Q2D        _q2d branch directly on the active points, r2SCAN elsewhere (independent of the mixing code)
  W[n]            unpolarized_mix 'model' and q2d_scalar.evaluate (reference implementation): must agree
  W_new           W[n] * p^k/(p^k + p0^k), k = 12, p0 = 6
Usage: python3 anigto_limits.py [--m 200] [--lams 1,10,100,1000,10000] [--out anigto_limits.csv]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "pbe_nonscf_ae6")):
    if p not in sys.path:
        sys.path.append(p)
import q2d_scalar as q                                       # noqa: E402
from q2d_scalar import _q2d, evaluate, CTF, K               # noqa: E402
from validate_limits import unpolarized_mix, active_mask    # noqa: E402
from pyscf.dft import libxc                                 # noqa: E402

np.seterr(all="ignore")
PI4 = np.pi ** -0.25
K_NEW, P0_NEW = 12, 6.0

SYSTEMS = {  # name: (geometry, occupied orbitals (k_x, k_y, k_z))
    "wire N=2": ("wire", [(0, 0, 0)]),
    "wire N=4": ("wire", [(0, 0, 0), (0, 0, 1)]),
    "wire N=6": ("wire", [(0, 0, 0), (0, 0, 1), (0, 0, 2)]),
    "sheet N=2": ("sheet", [(0, 0, 0)]),
    "sheet N=4": ("sheet", [(0, 0, 0), (1, 0, 0)]),
    "sheet N=6": ("sheet", [(0, 0, 0), (1, 0, 0), (0, 1, 0)]),
}


def omegas(geom, lam):
    return (lam ** 2, lam ** 2, 1.0) if geom == "wire" else (1.0, 1.0, lam ** 2)


def hermite(y, M):
    """Hermite functions h_k(y) and h_k'(y), k < M."""
    h = [PI4 * np.exp(-y * y / 2)]
    if M > 1:
        h.append(np.sqrt(2.0) * y * h[0])
    for k in range(2, M):
        h.append(np.sqrt(2.0 / k) * y * h[k - 1] - np.sqrt((k - 1) / k) * h[k - 2])
    h = np.array(h)
    dh = np.array([-y * h[k] + (np.sqrt(2.0 * k) * h[k - 1] if k else 0.0) for k in range(M)])
    return h, dh


# ------------------------------------------------------------------ exact exchange
def _log_gl(lo, hi, n, panels=1):
    x, w = np.polynomial.legendre.leggauss(n)
    edges = np.linspace(np.log(lo), np.log(hi), panels + 1)
    X, Wt = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        q_ = 0.5 * (b - a) * x + 0.5 * (b + a)
        X.append(np.exp(q_)); Wt.append(0.5 * (b - a) * w * np.exp(q_))
    return np.concatenate(X), np.concatenate(Wt)


def exact_ex(geom, lam, occ):
    """E_x = -sum_ij (ij|ij) = -(2/sqrt pi) sum_ij int_0^inf prod_d I_d^{ij}(t) dt."""
    om = omegas(geom, lam)
    M = max(max(o) for o in occ) + 1
    t, wt = _log_gl(1e-9, 1e13, 64, panels=60)
    xg, wg = np.polynomial.hermite.hermgauss(24)
    cache = {}

    def I(d, a, b):
        key = (om[d], a, b)
        if key in cache:
            return cache[key]
        c = (t * t / om[d])[:, None, None]
        s = np.sqrt(1 + 2 * c)
        u, v = xg[None, :, None], xg[None, None, :] / s
        y1, y2 = (u + v) / np.sqrt(2), (u - v) / np.sqrt(2)
        h1, _ = hermite(y1, M)
        h2, _ = hermite(y2, M)
        P1 = h1[a] * h1[b] * np.exp(y1 * y1)
        P2 = h2[a] * h2[b] * np.exp(y2 * y2)
        val = np.einsum("k,l,tkl->t", wg, wg, P1 * P2) / s[:, 0, 0]
        cache[key] = val
        return val

    tot = 0.0
    for i in occ:
        for j in occ:
            f = I(0, i[0], j[0]) * I(1, i[1], j[1]) * I(2, i[2], j[2])
            tot += wt @ f + 1e-9 * float(i == j)   # [0, 1e-9]: prod_d I_d(0) = delta_ij
    return -2 / np.sqrt(np.pi) * tot


def exact_closed(geom, lam):
    """jcp_main Eqs. (10), (11): one-orbital sheet / wire."""
    if lam == 1:
        return -np.sqrt(2 / np.pi)
    if lam > 1:
        qq = np.sqrt(lam ** 2 - 1)
        r = np.arctan(qq) / qq if geom == "sheet" else np.arccosh(lam) / qq
    else:                                       # lam < 1: analytic continuation of Eqs. (10), (11)
        qq = np.sqrt(1 - lam ** 2)
        r = np.arctanh(qq) / qq if geom == "sheet" else np.arccos(lam) / qq
    return -np.sqrt(2 / np.pi) * lam * r


# ------------------------------------------------------------------ semilocal fields
def fields(geom, lam, occ, m):
    """n, |grad n|^2, tau on the positive octant (scaled coordinates y_d = sqrt(w_d) x_d), weights include 8x."""
    om = omegas(geom, lam)
    M = max(max(o) for o in occ) + 1
    y, wy = _log_gl(1e-9, 9.5, m)
    h, dh = hermite(y, M)
    sh = (m, m, m)
    n = np.zeros(sh); gx = np.zeros(sh); gy = np.zeros(sh); gz = np.zeros(sh); tau = np.zeros(sh)
    Fs, Ds = [], []
    for o in occ:
        f = [om[d] ** 0.25 * h[o[d]] for d in range(3)]
        df = [om[d] ** 0.75 * dh[o[d]] for d in range(3)]
        F = np.einsum("i,j,k->ijk", f[0], f[1], f[2])
        D = [np.einsum("i,j,k->ijk", *(df[e] if e == d else f[e] for e in range(3))) for d in range(3)]
        n += 2 * F * F
        gx += 4 * F * D[0]; gy += 4 * F * D[1]; gz += 4 * F * D[2]
        tau += D[0] * D[0] + D[1] * D[1] + D[2] * D[2]          # conventional tau, 1/2 x 2 (occupation)
        Fs.append(F); Ds.append(D)
    # Pauli term exactly (Lagrange identity): tau - tau_W = sum_{i<j} |phi_i grad phi_j - phi_j grad phi_i|^2 / sum phi^2
    pauli = np.zeros(sh)
    for i in range(len(occ)):
        for j in range(i + 1, len(occ)):
            for d in range(3):
                c = Fs[i] * Ds[j][d] - Fs[j] * Ds[i][d]
                pauli += c * c
    pauli = 2 * pauli / n
    del Fs, Ds
    w = 8 * np.einsum("i,j,k->ijk", wy / np.sqrt(om[0]), wy / np.sqrt(om[1]), wy / np.sqrt(om[2]))
    n, g2, pauli, w = n.ravel(), (gx * gx + gy * gy + gz * gz).ravel(), pauli.ravel(), w.ravel()
    keep = n > 1e-60
    n, g2, pauli, w = n[keep], g2[keep], pauli[keep], w[keep]
    tau = g2 / (8 * n) + pauli                                   # tau = tau_W + Pauli, alpha = 0 exactly for N = 2
    return n, g2, tau, w, int((pauli == 0).sum())


def lx(code, n, g2, tau):
    z = np.zeros_like(n)
    return n * libxc.eval_xc(code, np.stack((n, np.sqrt(g2), z, z, tau)), spin=0, deriv=0)[0]


def run_one(name, lam, m):
    geom, occ = SYSTEMS[name]
    N = 2 * len(occ)
    t0 = time.time()
    ex_exact = exact_ex(geom, lam, occ)
    n, g2, tau, w, nsnap = fields(geom, lam, occ, m)
    act = active_mask(n, g2, tau)
    r = dict(system=name, lam=lam, N_grid=w @ n, N=N, Ex_exact=ex_exact,
             Ex_closed=exact_closed(geom, lam) if N == 2 else np.nan, n_pauli0=nsnap,
             frac_active=(w[act] @ n[act]) / N)
    # Libxc, direct
    r["Ex_SCAN"] = w @ lx("MGGA_X_SCAN", n, g2, tau)
    exL, ecL = lx("MGGA_X_R2SCAN", n, g2, tau), lx("MGGA_C_R2SCAN", n, g2, tau)
    r["Ex_r2SCAN_libxc"], r["Ec_r2SCAN_libxc"] = w @ exL, w @ ecL
    # W = 0 and W = 1 through the Table-9 mixing code
    ex0, ec0, _ = unpolarized_mix(n, g2, tau, 0)
    ex1, ec1, _ = unpolarized_mix(n, g2, tau, 1)
    exm, ecm, _ = unpolarized_mix(n, g2, tau, "model")
    r["Ex_W0"], r["Ec_W0"], r["Ex_W1"], r["Ec_W1"] = w @ ex0, w @ ec0, w @ ex1, w @ ec1
    # pure Q2D, directly from the branch
    exq, ecq = exL.copy(), ecL.copy()
    Wv = np.zeros_like(n); Wn = np.zeros_like(n); amin = amax = np.nan
    if act.any():
        wq, qx, qc = _q2d(n[act], g2[act], tau[act])
        exq[act], ecq[act] = qx.value, qc.value
        Wv[act] = wq.value
        p = g2[act] / (4 * K ** 2 * n[act] ** (8 / 3))
        Wn[act] = wq.value * p ** K_NEW / (p ** K_NEW + P0_NEW ** K_NEW)
        a = (tau[act] - g2[act] / (8 * n[act])) / (CTF * n[act] ** (5 / 3))
        amin, amax = a.min(), a.max()
    r["Ex_Q2D_pure"], r["Ec_Q2D_pure"] = w @ exq, w @ ecq
    # W[n]: mixing code vs reference implementation
    ref = evaluate(n, g2, tau)
    r["Ex_Worig"], r["Ec_Worig"] = w @ exm, w @ ecm
    r["Ex_Worig_evaluate"], r["Ec_Worig_evaluate"] = w @ ref.exchange, w @ ref.correlation
    r["Ex_Wnew"] = w @ (exL + Wn * (exq - exL))
    r["Ec_Wnew"] = w @ (ecL + Wn * (ecq - ecL))
    r["W_mean"], r["Wnew_mean"], r["W_max"] = (w @ (n * Wv)) / N, (w @ (n * Wn)) / N, Wv.max()
    r["alpha_min_act"], r["alpha_max_act"] = amin, amax
    r["nonfinite"] = int(sum((~np.isfinite(x)).sum() for x in (ex0, ec0, ex1, ec1, exm, ecm, exq, ecq)))
    # limit differences
    r["dEx_W0"] = r["Ex_W0"] - r["Ex_r2SCAN_libxc"]
    r["dEc_W0"] = r["Ec_W0"] - r["Ec_r2SCAN_libxc"]
    r["dEx_W1"] = r["Ex_W1"] - r["Ex_Q2D_pure"]
    r["dEc_W1"] = r["Ec_W1"] - r["Ec_Q2D_pure"]
    r["dEx_model"] = r["Ex_Worig"] - r["Ex_Worig_evaluate"]
    r["dEc_model"] = r["Ec_Worig"] - r["Ec_Worig_evaluate"]
    r["sec"] = time.time() - t0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--lams", default="1,10,100,1000,10000")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--out", default=os.path.join(HERE, "anigto_limits.csv"))
    a = ap.parse_args()
    import pandas as pd
    import pyscf
    print(f"pyscf {pyscf.__version__}  grid {a.m}^3 per system", flush=True)
    rows = []
    for name in a.systems.split(","):
        for lam in map(float, a.lams.split(",")):
            r = run_one(name, lam, a.m)
            rows.append(r)
            print(f"{name:10s} lam {lam:7.0e} N {r['N_grid']:.8f}  exact {r['Ex_exact']:.8f} (closed {r['Ex_closed']:.8f})"
                  f"  act {r['frac_active']:.3f}  W=0 dEx {r['dEx_W0']:+.1e} dEc {r['dEc_W0']:+.1e}"
                  f"  W=1 dEx {r['dEx_W1']:+.1e} dEc {r['dEc_W1']:+.1e}  W[n] dEx {r['dEx_model']:+.1e}"
                  f"  nonfinite {r['nonfinite']}  {r['sec']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format="%.12e")

    print("\nTABLE 1  limit checks (Ha): W=0 vs Libxc r2SCAN, W=1 vs pure Q2D, W[n] mixing code vs evaluate()")
    print(f"{'system':10s}{'lam':>7s}{'Ex r2SCAN':>14s}{'dEx W=0':>10s}{'Ec r2SCAN':>12s}{'dEc W=0':>10s}"
          f"{'Ex Q2D':>12s}{'dEx W=1':>10s}{'Ec Q2D':>11s}{'dEc W=1':>10s}{'dEx W[n]':>10s}{'dEc W[n]':>10s}")
    for r in rows:
        print(f"{r['system']:10s}{r['lam']:7.0e}{r['Ex_r2SCAN_libxc']:14.8f}{r['dEx_W0']:10.1e}{r['Ec_r2SCAN_libxc']:12.8f}"
              f"{r['dEc_W0']:10.1e}{r['Ex_Q2D_pure']:12.6f}{r['dEx_W1']:10.1e}{r['Ec_Q2D_pure']:11.6f}{r['dEc_W1']:10.1e}"
              f"{r['dEx_model']:10.1e}{r['dEc_model']:10.1e}")
    print("max |d|:", " ".join(f"{k} {df[k].abs().max():.1e}" for k in
                               ("dEx_W0", "dEc_W0", "dEx_W1", "dEc_W1", "dEx_model", "dEc_model")))

    print("\nTABLE 2  exchange (Ha) vs exact; <W> electron-averaged weight, act = fraction of electrons with 0<alpha<1")
    print(f"{'system':10s}{'lam':>7s}{'exact':>12s}{'SCAN':>12s}{'r2SCAN':>12s}{'W=1':>12s}{'W orig':>12s}"
          f"{'W new':>12s}{'act':>7s}{'<W>':>7s}{'<Wn>':>7s}{'a_min':>9s}{'a_max':>9s}")
    for r in rows:
        print(f"{r['system']:10s}{r['lam']:7.0e}{r['Ex_exact']:12.6f}{r['Ex_SCAN']:12.4f}{r['Ex_r2SCAN_libxc']:12.4f}"
              f"{r['Ex_W1']:12.4f}{r['Ex_Worig']:12.4f}{r['Ex_Wnew']:12.4f}{r['frac_active']:7.3f}{r['W_mean']:7.3f}"
              f"{r['Wnew_mean']:7.3f}{r['alpha_min_act']:9.2e}{r['alpha_max_act']:9.2e}")
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
