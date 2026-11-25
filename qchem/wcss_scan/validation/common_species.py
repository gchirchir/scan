#!/usr/bin/env python3
"""Common species He, Be, H, H2 of Table 9 (tab:limits) and Table 11 (tab:anigtolimits): E_x, E_c with W = 0, W = 1,
W_orig (jana_ref) and W_new on ONE density per species (self-consistent r2SCAN, dyall-v4z, PySCF grid 9; RKS/UKS as
in reproduce_ae6_bh6.py; H2 geometry from geometries/H2.xyz), by three routes:

  pyscf     route of Table 9: PySCF NumInt fields (n, grad n, tau per spin from the AO basis), spin-polarized
            implementation validate_limits.polarized_xc / q2d_only
  ani_pg    aniGTO_Q2D(lambda=1): analytic Cartesian Gaussians x^i y^j z^k exp(-a |r-A|^2) (primitives, contraction
            and MO coefficients of the SCF, cart2sph), analytic gradients, tau_sigma - tau_W,sigma exact by the
            Lagrange identity; on the SAME PySCF grid points and weights
  ani_rad   atoms only (He, Be, H): the same analytic s-Gaussian fields on a radial grid split at alpha = 1
            (converged to 1e-9 Ha)
Spin treatment (Table 9): E_x = 1/2 E_x[2 n_up] + 1/2 E_x[2 n_dn] (W, Q2D, r2SCAN on the doubled spin densities);
E_c = int (1-W[n]) e_c^r2SCAN(n_up, n_dn) + W[n] e_c^Q[n]  (W, Q2D from the total density).
W_new = W_orig p^12/(p^12 + 6^12).  Writes common_species.csv.
"""
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
from q2d_scalar import _q2d, pyscf_baseline, CTF, K                 # noqa: E402
from validate_limits import polarized_xc, q2d_only, active_mask    # noqa: E402
from reproduce_ae6_bh6 import read_xyz                              # noqa: E402

np.seterr(all="ignore")
SPECIES = {"He": ("He 0 0 0", 0, 1e-10), "Be": ("Be 0 0 0", 0, 1e-10),   # He, Be: as in he_be_common.py
           "H": ("H 0 0 0", 1, 1e-9), "H2": (None, 0, 1e-9)}           # H, H2: as in Table 9
MODES = ("W0", "W1", "Worig", "Wnew")
RMAX = 40.0


def scf(name):
    atom, spin, tol = SPECIES[name]
    mol = gto.M(atom=atom or read_xyz("H2"), basis="dyall-v4z", spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol)
    mf.xc, mf.grids.level, mf.conv_tol, mf.max_cycle = "r2scan", 9, tol, 300
    mf.kernel()
    return mol, mf


def occ_coeffs(mf):
    """Occupied spatial MO coefficients per spin (sph AO basis)."""
    if mf.mo_coeff.ndim == 3:
        return [mf.mo_coeff[s][:, mf.mo_occ[s] > 0] for s in (0, 1)]
    C = mf.mo_coeff[:, mf.mo_occ > 0]
    return [C, C]


# ------------------------------------------------------------------ the functional, generic weight
def weight(mode, n, g2, tau):
    act = active_mask(n, g2, tau)
    wq, qx, qc = _q2d(n[act], g2[act], tau[act])
    if mode == "W0":
        W = np.zeros(act.sum())
    elif mode == "W1":
        W = np.ones(act.sum())
    else:
        W = wq.value
        if mode == "Wnew":
            p = g2[act] / (4 * K ** 2 * n[act] ** (8 / 3))
            W = W * p ** 12 / (p ** 12 + 6.0 ** 12)
    return act, W, qx.value, qc.value


def pol_energies(rs, w):
    """E_x, E_c for all four weights + Libxc references + pure Q2D, spin-polarized (Table 9 treatment)."""
    out = {}
    n = rs[0][0] + rs[1][0]
    gr = rs[0][1:4] + rs[1][1:4]
    g2, tau = (gr * gr).sum(0), rs[0][4] + rs[1][4]
    out["Ex_libxc"] = float(w @ (n * libxc.eval_xc("MGGA_X_R2SCAN", (rs[0], rs[1]), spin=1)[0]))
    ec_r2 = n * libxc.eval_xc("MGGA_C_R2SCAN", (rs[0], rs[1]), spin=1)[0]
    out["Ec_libxc"] = float(w @ ec_r2)
    for mode in MODES:
        Ex = 0.0
        for r in rs:
            n2, g22, t2 = 2 * r[0], 4 * (r[1:4] ** 2).sum(0), 2 * r[4]
            ex = pyscf_baseline(n2, g22, t2).exchange.copy()
            act, W, qx, _ = weight(mode, n2, g22, t2)
            ex[act] = (1 - W) * ex[act] + W * qx
            Ex += 0.5 * float(w @ ex)
        ec = ec_r2.copy()
        act, W, _, qc = weight(mode, n, g2, tau)
        ec[act] = (1 - W) * ec[act] + W * qc
        out["Ex_" + mode], out["Ec_" + mode] = Ex, float(w @ ec)
        if mode == "W1":
            ecq = ec_r2.copy(); ecq[act] = qc
            out["Ec_Q2D"] = float(w @ ecq)
    out["Ex_Q2D"] = q2d_only(rs, w)                                   # independent pure-Q2D route of Table 9
    act = active_mask(n, g2, tau)
    out["f_act"] = float(w[act] @ n[act] / (w @ n))
    for mode, key in ((0, "W0"), (1, "W1"), ("model", "Worig")):      # tie to the Table-9 code
        ex, ec = polarized_xc(rs, w, mode)
        out["chk_Ex_" + key], out["chk_Ec_" + key] = ex - out["Ex_" + key], ec - out["Ec_" + key]
    out["N"] = float(w @ n)
    return out


# ------------------------------------------------------------------ aniGTO fields
def cart_shells(mol):
    """Per Cartesian AO: center, exponents, coefficients, (i,j,k); calibrated against libcint GTOval_cart."""
    out = []
    for b in range(mol.nbas):
        l, A, e, c = mol.bas_angular(b), mol.bas_coord(b), mol.bas_exp(b), mol._libcint_ctr_coeff(b)
        comps = [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]
        for k in range(mol.bas_nctr(b)):
            for ijk in comps:
                out.append((A, e, c[:, k], ijk))
    return out


def cart_values(shells, pts, scale=None):
    """Values and gradients of the Cartesian AOs at pts (npts, 3)."""
    nao = len(shells)
    V = np.zeros((nao, len(pts))); G = np.zeros((3, nao, len(pts)))
    for m, (A, e, c, ijk) in enumerate(shells):
        d = pts - A
        r2 = (d * d).sum(1)
        E = np.exp(-np.outer(r2, e))                       # (npts, nprim)
        R = E @ c
        Rp = -2 * (E * e) @ c                              # radial-part derivative / (2 x) factor
        poly = d[:, 0] ** ijk[0] * d[:, 1] ** ijk[1] * d[:, 2] ** ijk[2]
        V[m] = poly * R
        for ax in range(3):
            i = ijk[ax]
            dpoly = (i * d[:, ax] ** (i - 1) if i > 0 else 0.0) * np.prod(
                [d[:, bx] ** ijk[bx] for bx in range(3) if bx != ax], axis=0)
            G[ax, m] = dpoly * R + poly * Rp * d[:, ax]
    if scale is not None:
        V *= scale[:, None]; G *= scale[None, :, None]
    return V, G


def calibrate(mol, shells):
    """Constant prefactor per Cartesian AO (libcint convention), fitted on points at the AO's own length scale."""
    rng = np.random.default_rng(1)
    scale, resid = np.zeros(len(shells)), 0.0
    for m, (A, e, c, ijk) in enumerate(shells):
        pts = A + rng.normal(size=(40, 3)) / np.sqrt(2 * e.min())
        ref = mol.eval_gto("GTOval_cart", pts)[:, m]
        mine, _ = cart_values([shells[m]], pts)
        mine = mine[0]
        scale[m] = (ref @ mine) / (mine @ mine)
        resid = max(resid, np.abs(scale[m] * mine - ref).max() / np.abs(ref).max())
    return scale, resid


def spin_fields(phi, dphi):
    """rs_sigma = (n, grad n, tau) for one spin from occupied orbitals phi (nocc, npts), dphi (3, nocc, npts)."""
    n = (phi ** 2).sum(0)
    gn = 2 * (phi[None] * dphi).sum(1)
    pa = np.zeros_like(n)
    for i in range(phi.shape[0]):
        for j in range(i + 1, phi.shape[0]):
            pa += ((phi[i] * dphi[:, j] - phi[j] * dphi[:, i]) ** 2).sum(0)
    safe = np.where(n > 0, n, 1.0)
    tau = np.where(n > 0, (gn ** 2).sum(0) / (8 * safe) + 0.5 * pa / safe, 0.0)
    return np.vstack((n, gn, tau))


def ani_rs(mol, mf, pts, chunk=20000):
    shells = cart_shells(mol)
    scale, resid = calibrate(mol, shells)
    c2s = mol.cart2sph_coeff()
    Cs = [c2s @ C for C in occ_coeffs(mf)]
    rs = [[], []]
    for s0 in range(0, len(pts), chunk):
        V, G = cart_values(shells, pts[s0:s0 + chunk], scale)
        for s in (0, 1):
            phi = Cs[s].T @ V
            dphi = np.einsum("ai,xan->xin", Cs[s], G)
            rs[s].append(spin_fields(phi, dphi))
    return np.array([np.hstack(r) for r in rs]), resid


def radial_points(mol, mf, m=2000):
    """Radial grid (atoms) split at the alpha = 1 crossings of the doubled spin densities and of the total density."""
    probe = np.logspace(-9, np.log10(RMAX), 20001)
    pts = np.c_[probe, 0 * probe, 0 * probe]
    rs, _ = ani_rs(mol, mf, pts)
    edges = [1e-9, RMAX]
    dens = [(2 * r[0], 4 * (r[1:4] ** 2).sum(0), 2 * r[4]) for r in rs]
    gt = rs[0][1:4] + rs[1][1:4]
    dens.append((rs[0][0] + rs[1][0], (gt * gt).sum(0), rs[0][4] + rs[1][4]))
    def alpha_at(r, which):
        rr, _ = ani_rs(mol, mf, np.array([[r, 0.0, 0.0]]))
        if which < 2:
            n, g2, t = 2 * rr[which][0], 4 * (rr[which][1:4] ** 2).sum(0), 2 * rr[which][4]
        else:
            g = rr[0][1:4] + rr[1][1:4]
            n, g2, t = rr[0][0] + rr[1][0], (g * g).sum(0), rr[0][4] + rr[1][4]
        return ((t - g2 / (8 * n)) / (CTF * n ** (5 / 3)))[0]
    for which, (n, g2, t) in enumerate(dens):
        a = (t - g2 / (8 * n)) / (CTF * n ** (5 / 3))
        f = a - 1
        for i in np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
            lo, hi = probe[i], probe[i + 1]
            for _ in range(50):                                        # bisection on alpha(r) = 1
                mid = np.sqrt(lo * hi)
                if np.sign(alpha_at(mid, which) - 1) == np.sign(f[i]):
                    lo = mid
                else:
                    hi = mid
            edges.append(np.sqrt(lo * hi))
    edges = np.unique(edges)
    y, wy = np.polynomial.legendre.leggauss(m)
    R, WR = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = 0.5 * (np.log(hi) - np.log(lo)) * y + 0.5 * (np.log(hi) + np.log(lo))
        R.append(np.exp(q)); WR.append(0.5 * (np.log(hi) - np.log(lo)) * wy * np.exp(q))
    r, wr = np.concatenate(R), np.concatenate(WR)
    return np.c_[r, 0 * r, 0 * r], 4 * np.pi * r * r * wr, edges


def main():
    import pandas as pd
    rows = []
    for name in SPECIES:
        mol, mf = scf(name)
        # PySCF route
        dm = mf.make_rdm1()
        if dm.ndim == 2:
            dm = np.array([dm / 2, dm / 2])
        ni = dft.numint.NumInt()
        rs = [[], []]
        for ao, mask, wb, c in ni.block_loop(mol, mf.grids, mol.nao, deriv=1, max_memory=2000):
            for s in (0, 1):
                rs[s].append(ni.eval_rho(mol, ao, dm[s], mask, xctype="MGGA", hermi=1, with_lapl=False))
        rs = np.array([np.hstack(r) for r in rs])
        w = mf.grids.weights
        P = pol_energies(rs, w)
        # aniGTO on the same PySCF points
        rsa, resid = ani_rs(mol, mf, mf.grids.coords)
        A = pol_energies(rsa, w)
        row = dict(species=name, e_tot=mf.e_tot, conv=mf.converged, cart_fit=resid,
                   **{"pyscf_" + k: v for k, v in P.items()}, **{"ani_pg_" + k: v for k, v in A.items()})
        if mol.natm == 1:
            pts, wr, edges = radial_points(mol, mf)
            rsr, _ = ani_rs(mol, mf, pts)
            B = pol_energies(rsr, wr)
            row.update({"ani_rad_" + k: v for k, v in B.items()})
        rows.append(row)
        print(f"\n{name}: E {mf.e_tot:.10f} conv {mf.converged}; Cartesian aniGTO vs libcint {resid:.1e}; "
              f"N {P['N']:.10f} / {A['N']:.10f}; f_act PySCF {P['f_act']:.3f} aniGTO {A['f_act']:.3f}")
        for c in ("Ex", "Ec"):
            for md in MODES + ("libxc", "Q2D"):
                k = f"{c}_{md}"
                s = f"   {k:9s} PySCF {P[k]:+.10f}  aniGTO(PySCF grid) {A[k]:+.10f}  diff {A[k] - P[k]:+.1e}"
                if "ani_rad_" + k in row:
                    s += f"  aniGTO(radial) {row['ani_rad_' + k]:+.10f}  diff {row['ani_rad_' + k] - P[k]:+.1e}"
                print(s)
        print("   check vs polarized_xc (PySCF, aniGTO):", " ".join(
            f"{k[4:]} {P[k]:+.0e}/{A[k]:+.0e}" for k in P if k.startswith("chk_")), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "common_species.csv"), index=False, float_format="%.15e")
    print("written common_species.csv")


if __name__ == "__main__":
    main()
