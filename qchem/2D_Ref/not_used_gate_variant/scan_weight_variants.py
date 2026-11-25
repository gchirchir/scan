#!/usr/bin/env python3
"""Scan modifications of the scalar-Q2D switching weight W on stored r2SCAN densities (densities/*.npz) and on the
infinite-barrier well, to look for a weight that stays on in quasi-2D wells but off in molecular cores and bonds.

Variant W'(alpha, s) = W(alpha, s) * G1 * G2 with
  G1 = gate(p / (c sqrt(alpha)), 6) / gate(p / (6 sqrt(alpha)), 6)   (c = 6: original jana_ref gate)
  G2 = s^k / (s0^k + s^k)                                              (s0 = 0: no extra gate)
Exchange and correlation branches are those of jana_ref (q2d_scalar._q2d); only W changes.  Open shells: spin route B
(W and Q2D branch from the total density, mixed with spin-polarized r2SCAN).  Post-SCF on r2SCAN densities.
"""
import glob, itertools, os, sys
import numpy as np
from pyscf.dft import libxc
from q2d_scalar import _q2d, pyscf_baseline, CTF, K
from reproduce_ae6_bh6 import AE6_REF, BH6, MOLECULES, read_xyz, HA2KCAL

HERE = os.path.dirname(os.path.abspath(__file__))
IBM = np.genfromtxt(os.path.join(HERE, "ibm_benchmark.csv"), delimiter=",", names=True)
NS = 1 / (np.pi * 16.0)


def gate(r, power):
    r = np.asarray(r, float)
    return np.where(r <= 1, r**power / (1 + r**power), 1 / (1 + np.minimum(1 / np.maximum(r, 1e-300), 1e300)**power))


def weight_factor(n, g2, tau, c, s0, k):
    n13 = np.cbrt(n)
    alpha = (tau - g2 / (8 * n)) / (CTF * n * n13 * n13)
    p = g2 / (4 * K**2 * n * n * n13 * n13)
    f = np.ones_like(n)
    if c != 6:
        sa = np.sqrt(alpha + 1e-6)
        f *= gate(p / (c * sa), 6) / np.maximum(gate(p / (6 * sa), 6), 1e-300)
    if s0 > 0:
        s = np.sqrt(p)
        f *= s**k / (s0**k + s**k)
    return f


def active(n, g2, tau):
    pauli = tau - g2 / (8 * np.where(n > 0, n, 1))
    return (n > 1e-60) & (g2 > 0) & (pauli > 0) & (pauli < CTF * n ** (5 / 3))


def prepare(d):
    """Variant-independent pieces on the active grid points: base weight, branch-minus-r2SCAN, alpha-p inputs."""
    rho, w = d["rho"], d["w"]
    n = np.maximum(rho[0][0] + rho[1][0], 0)
    gr = rho[0][1:4] + rho[1][1:4]
    g2 = np.einsum("ix,ix->x", gr, gr)
    tau = rho[0][4] + rho[1][4]
    a = active(n, g2, tau)
    if not a.any():
        return None
    W, ex, ec = _q2d(n[a], g2[a], tau[a])
    ra, rb = rho[0][:, a], rho[1][:, a]
    exr = (ra[0] + rb[0]) * libxc.eval_xc("MGGA_X_R2SCAN", (ra, rb), spin=1, deriv=0)[0]
    ecr = (ra[0] + rb[0]) * libxc.eval_xc("MGGA_C_R2SCAN", (ra, rb), spin=1, deriv=0)[0]
    return dict(n=n[a], g2=g2[a], tau=tau[a], wW=w[a] * W.value, diff=ex.value - exr + ec.value - ecr)


def delta_species(P, c, s0, k):
    if P is None:
        return 0.0
    return float(P["wW"] @ (weight_factor(P["n"], P["g2"], P["tau"], c, s0, k) * P["diff"]))


_IBM_CACHE = []


def _ibm_prepare():
    for L, fock, ref_xc in zip(IBM["L_bohr"], IBM["exact_fock_x"], IBM["istls_xc"]):
        npts = 12000 if L < 0.05 else 3000
        x, wq = np.polynomial.legendre.leggauss(npts)
        y, wy = (x + 1) / 2, wq / 2
        n = NS / L * 2 * np.sin(np.pi * y)**2
        g2 = (NS / L**2 * 2 * np.pi * np.sin(2 * np.pi * y))**2
        tau = g2 / (8 * n) + np.pi * NS / 2 * n
        base = pyscf_baseline(n, g2, tau)
        a = active(n, g2, tau)
        W, ex, ec = _q2d(n[a], g2[a], tau[a])
        _IBM_CACHE.append(dict(L=L, fock=fock, ref=ref_xc, wy=wy, a=a, n=n[a], g2=g2[a], tau=tau[a], W=W.value,
                               bx=base.exchange, bc=base.correlation, ex=ex.value, ec=ec.value))


def ibm(c, s0, k):
    if not _IBM_CACHE:
        _ibm_prepare()
    out = []
    for C in _IBM_CACHE:
        L, a = C["L"], C["a"]
        Wv = C["W"] * weight_factor(C["n"], C["g2"], C["tau"], c, s0, k)
        exv, ecv = C["bx"].copy(), C["bc"].copy()
        exv[a] += Wv * (C["ex"] - exv[a]); ecv[a] += Wv * (C["ec"] - ecv[a])
        pe = lambda e: L * float(C["wy"] @ e) / NS
        out.append((L, pe(exv) - C["fock"], pe(exv + ecv) - C["ref"]))
    out = np.array(out)
    th, tk = out[:, 0] < 1, out[:, 0] >= 1
    return [np.abs(out[m, i]).mean() for m in (th, tk) for i in (1, 2)]   # Ex thin, XC thin, Ex thick, XC thick


def main():
    D = {os.path.basename(f)[:-4]: (float(np.load(f)["e_tot"]), prepare(dict(np.load(f)))) for f in glob.glob(os.path.join(HERE, os.environ.get("AE6_DENSITY_DIR", "densities"), "*.npz"))}
    atoms_of = lambda m: [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
    grid = [(6, 0, 4)] + [(c, 0, 4) for c in (3, 4, 8, 10, 15, 20, 30, 50)] + \
           [(6, s0, k) for s0 in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0) for k in (4, 8, 16)] + \
           [(c, s0, 8) for c in (4, 10, 20) for s0 in (1.0, 1.5, 2.0)]
    print(f"{'c':>4} {'s0':>4} {'k':>2} | AE6 ME   MAE | BH6 ME   MAE | IBM MAE (Ha/e): Ex L<1  XC L<1  Ex L>=1 XC L>=1")
    for c, s0, k in grid:
        E = {nm: e0 + delta_species(P, c, s0, k) for nm, (e0, P) in D.items()}
        ae = [(sum(E[a] for a in atoms_of(m)) - E[m]) * HA2KCAL - ref for m, ref in AE6_REF.items()]
        bh = []
        for lab, R, ts, P, ff, rr in BH6:
            bh += [(E[ts] - sum(E[x] for x in R)) * HA2KCAL - ff, (E[ts] - sum(E[x] for x in P)) * HA2KCAL - rr]
        im = ibm(c, s0, k)
        print(f"{c:4g} {s0:4g} {k:2d} | {np.mean(ae):+7.2f} {np.mean(np.abs(ae)):6.2f} | {np.mean(bh):+6.2f} "
              f"{np.mean(np.abs(bh)):5.2f} | {im[0]:.4f}  {im[1]:.4f}  {im[2]:.4f}  {im[3]:.4f}", flush=True)
    # r2SCAN reference line
    E = {nm: e0 for nm, (e0, P) in D.items()}
    ae = [(sum(E[a] for a in atoms_of(m)) - E[m]) * HA2KCAL - ref for m, ref in AE6_REF.items()]
    print(f"r2SCAN       | {np.mean(ae):+7.2f} {np.mean(np.abs(ae)):6.2f} | (IBM r2SCAN: 0.3849 0.3475 0.0425 0.0286)")


if __name__ == "__main__":
    main()
