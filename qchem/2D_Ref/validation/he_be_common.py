#!/usr/bin/env python3
"""He and Be: common rows of Table 9 (tab:limits) and Table 10 (tab:anigtolimits).

One density per atom: self-consistent r2SCAN, RKS, dyall-v4z, PySCF grid level 9 (as for Table 9).
Its occupied orbitals contain only s functions, phi_i(r) = sum_p d_ip exp(-a_p r^2), i.e. isotropic aniGTOs.
The limit checks are evaluated by two independent routes:
  route A (Table 9)  PySCF NumInt on the Becke grid, spin-polarized implementation (validate_limits.polarized_xc,
                     q2d_only), exactly as check_limits_dyall.py
  route B (Table 10) analytic n, grad n, tau of the primitive Gaussians on the aniGTO Cartesian product grid
                     (spherical: radial log Gauss-Legendre, m points per panel, panels split at alpha = 1), spin-unpolarized (unpolarized_mix, _q2d),
                     tau - tau_W from the Lagrange identity (exact zero for He)
Writes he_be_common.csv.  Usage: python3 he_be_common.py [--m 160]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "pbe_nonscf_ae6"), HERE):
    if p not in sys.path:
        sys.path.append(p)
from pyscf import dft, gto                                          # noqa: E402
from pyscf.dft import libxc                                         # noqa: E402
from q2d_scalar import _q2d, CTF                                         # noqa: E402
from validate_limits import polarized_xc, q2d_only, unpolarized_mix, active_mask   # noqa: E402

np.seterr(all="ignore")
RMAX = 40.0                                                      # box (bohr): dyall-v4z has diffuse s functions


def scf(sym):
    mol = gto.M(atom=f"{sym} 0 0 0", basis="dyall-v4z", spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc, mf.grids.level, mf.conv_tol, mf.max_cycle = "r2scan", 9, 1e-10, 300
    mf.kernel()
    return mol, mf


def route_a(mol, mf):
    dm = mf.make_rdm1()
    dm = np.array([dm / 2, dm / 2])
    ni = dft.numint.NumInt()
    rs = [[], []]
    for ao, mask, wb, coords in ni.block_loop(mol, mf.grids, mol.nao, deriv=1, max_memory=2000):
        for s in (0, 1):
            rs[s].append(ni.eval_rho(mol, ao, dm[s], mask, xctype="MGGA", hermi=1, with_lapl=False))
    rs = np.array([np.hstack(r) for r in rs])
    w = mf.grids.weights
    n = rs[0][0] + rs[1][0]
    ex_lx = float(w @ (n * libxc.eval_xc("MGGA_X_R2SCAN", (rs[0], rs[1]), spin=1)[0]))
    ec_lx = float(w @ (n * libxc.eval_xc("MGGA_C_R2SCAN", (rs[0], rs[1]), spin=1)[0]))
    ex0, ec0 = polarized_xc(rs, w, 0)
    ex1, _ = polarized_xc(rs, w, 1)
    exq = q2d_only(rs, w)
    exm, ecm = polarized_xc(rs, w, "model")
    return dict(N=float(w @ n), ex_Wn=exm, ec_Wn=ecm, ex_libxc=ex_lx, ex_W0=ex0, dEx_W0=ex0 - ex_lx, ec_libxc=ec_lx, ec_W0=ec0,
                dEc_W0=ec0 - ec_lx, ex_Q2D=exq, ex_W1=ex1, dEx_W1=ex1 - exq)


def s_orbitals(mol, mf):
    """Occupied MOs as primitive s Gaussians: exponents a_p and coefficients d_ip (checked against eval_gto)."""
    C = mf.mo_coeff[:, mf.mo_occ > 0]
    ao_l = np.concatenate([[mol.bas_angular(b)] * mol.bas_nctr(b) * (2 * mol.bas_angular(b) + 1)
                           for b in range(mol.nbas)])
    non_s = np.abs(C[ao_l > 0]).max() if (ao_l > 0).any() else 0.0
    exps, coef = [], []
    i = 0
    for b in range(mol.nbas):
        l, nc = mol.bas_angular(b), mol.bas_nctr(b)
        if l == 0:
            e = mol.bas_exp(b)
            cc = mol._libcint_ctr_coeff(b) / np.sqrt(4 * np.pi)       # libcint s functions omit Y_00
            for k in range(nc):
                exps.append(e); coef.append(np.outer(C[i + k], cc[:, k]))
        i += nc * (2 * l + 1)
    a = np.concatenate(exps)
    d = np.concatenate(coef, axis=1)                                 # (n_occ, n_prim)
    # check the primitive representation against PySCF's AO values
    rr = np.linspace(0.0, 6.0, 400)[:, None] * np.array([[0.3, -0.5, 0.81]]) / np.linalg.norm([0.3, -0.5, 0.81])
    ref = mol.eval_gto("GTOval", rr) @ C
    mine = np.exp(-np.outer((rr ** 2).sum(1), a)) @ d.T
    return a, d, non_s, np.abs(ref - mine).max()


def route_b(a, d, m):
    # spherical density: radial log Gauss-Legendre in r on [1e-9, RMAX] bohr, 4 pi r^2 dr (exact angular integral),
    # panels split at the alpha = 1 crossings (edge of the active region, where forced W = 1 jumps)
    def alpha(r):
        E = np.exp(-np.outer(r * r, a)); p_ = E @ d.T; dp = -2 * (E * a) @ d.T
        n_ = 2 * (p_ ** 2).sum(1)
        pa = sum((p_[:, i] * dp[:, j] - p_[:, j] * dp[:, i]) ** 2 for i in range(p_.shape[1])
                 for j in range(i + 1, p_.shape[1])) * r * r if p_.shape[1] > 1 else 0 * r
        return 2 * pa / n_ / (CTF * n_ ** (5 / 3))
    rs = np.logspace(-9, np.log10(RMAX), 20001)
    f = alpha(rs) - 1
    edges = [1e-9]
    for i in np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
        lo, hi = rs[i], rs[i + 1]
        for _ in range(100):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if np.sign(alpha(np.array([mid]))[0] - 1) == np.sign(f[i]) else (lo, mid)
        edges.append(np.sqrt(lo * hi))
    edges.append(RMAX)
    y, wy = np.polynomial.legendre.leggauss(m)
    R, WR = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = 0.5 * (np.log(hi) - np.log(lo)) * y + 0.5 * (np.log(hi) + np.log(lo))
        R.append(np.exp(q)); WR.append(0.5 * (np.log(hi) - np.log(lo)) * wy * np.exp(q))
    r, wr = np.concatenate(R), np.concatenate(WR)
    r2, w = r * r, 4 * np.pi * r * r * wr
    route_b.edges = edges
    E = np.exp(-np.outer(r2, a))                                     # (npts, nprim)
    phi = E @ d.T                                                    # (npts, nocc)
    dphi_r = -2 * (E * a) @ d.T                                      # d phi / d r = dphi_r * r_vec
    n = 2 * (phi ** 2).sum(1)
    g2 = (4 * (phi * dphi_r).sum(1)) ** 2 * r2
    tau = (dphi_r ** 2).sum(1) * r2
    pauli = np.zeros_like(n)                                         # Lagrange identity
    for i in range(phi.shape[1]):
        for j in range(i + 1, phi.shape[1]):
            pauli += (phi[:, i] * dphi_r[:, j] - phi[:, j] * dphi_r[:, i]) ** 2 * r2
    keep = n > 1e-60
    n, g2, pauli, w = n[keep], g2[keep], pauli[keep], w[keep]
    tau = g2 / (8 * n) + 2 * pauli / n
    z = np.zeros_like(n)
    rho = np.stack((n, np.sqrt(g2), z, z, tau))
    exL = n * libxc.eval_xc("MGGA_X_R2SCAN", rho, spin=0)[0]
    ecL = n * libxc.eval_xc("MGGA_C_R2SCAN", rho, spin=0)[0]
    ex0, ec0, _ = unpolarized_mix(n, g2, tau, 0)
    ex1, ec1, _ = unpolarized_mix(n, g2, tau, 1)
    exm, ecm, _ = unpolarized_mix(n, g2, tau, "model")
    act = active_mask(n, g2, tau)
    exq, ecq = exL.copy(), ecL.copy()
    if act.any():
        _, qx, qc = _q2d(n[act], g2[act], tau[act])
        exq[act], ecq[act] = qx.value, qc.value
    return dict(N=w @ n, ex_Wn=w @ exm, ec_Wn=w @ ecm, frac_active=(w[act] @ n[act]) / (w @ n), ex_libxc=w @ exL, ex_W0=w @ ex0,
                dEx_W0=w @ ex0 - w @ exL, ec_libxc=w @ ecL, ec_W0=w @ ec0, dEc_W0=w @ ec0 - w @ ecL,
                ex_Q2D=w @ exq, ex_W1=w @ ex1, dEx_W1=w @ ex1 - w @ exq, ec_Q2D=w @ ecq, ec_W1=w @ ec1,
                dEc_W1=w @ ec1 - w @ ecq)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "he_be_common.csv"))
    arg = ap.parse_args()
    rows = []
    for sym in ("He", "Be"):
        mol, mf = scf(sym)
        A = route_a(mol, mf)
        a, d, non_s, fit = s_orbitals(mol, mf)
        B = route_b(a, d, arg.m)
        print(f"{sym}: E(r2SCAN) {mf.e_tot:.10f} conv {mf.converged}; non-s MO coeff {non_s:.1e}; "
              f"primitive vs eval_gto {fit:.1e}; alpha=1 at r = {np.round(route_b.edges[1:-1], 4)}; N  A {A['N']:.10f}  B {B['N']:.10f}  f_act {B['frac_active']:.3f}")
        for k in ("ex_libxc", "ex_W0", "ec_libxc", "ec_W0", "ex_Q2D", "ex_W1", "ex_Wn", "ec_Wn"):
            print(f"   {k:9s} A {A[k]:+.10f}  B {B[k]:+.10f}  B-A {B[k] - A[k]:+.1e}")
        print(f"   A: dEx_W0 {A['dEx_W0']:+.1e} dEc_W0 {A['dEc_W0']:+.1e} dEx_W1 {A['dEx_W1']:+.1e}")
        print(f"   B: dEx_W0 {B['dEx_W0']:+.1e} dEc_W0 {B['dEc_W0']:+.1e} dEx_W1 {B['dEx_W1']:+.1e} dEc_W1 {B['dEc_W1']:+.1e}")
        rows.append(dict(species=sym, e_r2scan=mf.e_tot, non_s=non_s, fit=fit, m=arg.m,
                         **{"A_" + k: v for k, v in A.items()}, **{"B_" + k: v for k, v in B.items()}))
    pd.DataFrame(rows).to_csv(arg.out, index=False, float_format="%.15e")
    print("written", arg.out)


if __name__ == "__main__":
    main()
