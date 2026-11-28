#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal (PySCF/libxc environment; numba optional):
#
#     cd codes
#     ~/miniconda3/envs/anigto/bin/python 13_trap_ip_gap_gw.py            # all systems
#     ~/miniconda3/envs/anigto/bin/python 13_trap_ip_gap_gw.py --quick    # lam = 1, 2 only, no ions
#
# Needs: numpy, scipy, pyscf (libxc, FCI), [numba].  Results -> codes/results/
# ----------------------------------------------------------------------
r"""Ionization potentials, electron affinities, frontier orbitals, Kohn-Sham inversion and G0W0 for correlated
two-electron wires and sheets (aniGTO / aditi "Why G0W0 works" analogue).

STANDALONE SCRIPT (SCAN_1D_2D/codes): every function it uses is defined in this file; only third-party imports.
The basis, integral, grid and N = 2 inversion functions are identical copies of those in 12_trap_ac_benchmark.py.

Systems: wire (lam^2, lam^2, 1) and sheet (1, 1, lam^2) traps, lam = 1, 2, 5, 10; He-like ions (Z = 1, 2, 3) in the
lam = 2 wire and sheet.  Same aniGTO harmonic-oscillator basis construction as script 12, size target 56
(N = 3 FCI cost), [spdfg] included.

Exact (same-basis FCI) quantities
  E(N=1)  lowest eigenvalue of h;  E(N=2) singlet FCI;  E(N=3) doublet FCI (PySCF direct_spin1)
  I = E(1) - E(2),  A = E(2) - E(3),  fundamental gap I - A
Exact Kohn-Sham (the PyOEP[FCI] analogue)
  v_s from the FCI density by the exact N = 2 inversion phi = sqrt(rho/2) (script 12), constant fixed by the
  HOMO condition eps_H = -I (as PyOEP KSINV with ip given); LUMO and KS gap from the spectrum of T + v_s;
  derivative discontinuity Delta_xc = (I - A) - (eps_L - eps_H).  (For N = 2 the inversion is exact and analytic,
  so the ill-posed basis-set Wu-Yang/KSINV optimisation is not needed.)
  In a trap without nucleus I < 0: the tables use the chemical potentials mu2 = E2 - E1 = -I, mu3 = E3 - E2 = -A
  and the addition energy mu3 - mu2 = I - A (quantum-dot convention).
Approximate frontier orbitals: self-consistent HF, LDA (PW92), PBE, r2SCAN, SCAN, PBE0 in the same basis
  (KS-SCF written here: libxc potential matrix on the anisotropic Gauss-Hermite grid).
G0W0: exact-frequency (Casida RPA poles, full nonlinear QP equation), validated against PySCF gw_exact
  (He/cc-pVTZ, HF and PBE starts: 1e-10 Ha), started from HF, PBE and the exact KS orbitals (G0W0@PyOEP[FCI]).

Outputs (codes/results/): trap_ip_gap.csv, tab_trap_t7_ip_rows.tex, tab_trap_t7_gap_rows.tex,
  tab_trap_t7_gw_rows.tex, trap_ip_gap_log.txt
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)
T0 = time.time()
LOG = []


def log(msg):
    line = f'[{time.time() - T0:8.1f}s] {msg}'
    print(line, flush=True)
    LOG.append(line)


# ==============================================================================================================
# A. Harmonic-oscillator (aniGTO 'dot') basis and exact integrals            [shared block, identical in 12-14]
# ==============================================================================================================
def herm_poly_norm(nmax, y):
    """P_n(y) = H_n(y) / sqrt(2^n n! sqrt(pi)), n = 0..nmax, by the stable three-term recurrence."""
    y = np.asarray(y, float)
    P = np.zeros((nmax + 1,) + y.shape)
    P[0] = np.pi ** -0.25
    if nmax >= 1:
        P[1] = np.sqrt(2.0) * y * P[0]
    for n in range(2, nmax + 1):
        P[n] = np.sqrt(2.0 / n) * y * P[n - 1] - np.sqrt((n - 1.0) / n) * P[n - 2]
    return P


def ho_1d(x, nmax, w):
    """chi_n(x) = w^(1/4) P_n(sqrt(w) x) exp(-w x^2/2) and its first and second derivatives, n = 0..nmax."""
    x = np.asarray(x, float)
    P = herm_poly_norm(nmax + 1, np.sqrt(w) * x) * w ** 0.25 * np.exp(-0.5 * w * x * x)
    n = np.arange(nmax + 1)[:, None]
    val = P[:nmax + 1]
    lower = np.vstack([np.zeros((1,) + x.shape), P[:nmax]])
    d1 = np.sqrt(w / 2.0) * (np.sqrt(n) * lower - np.sqrt(n + 1.0) * P[1:nmax + 2])
    d2 = (w * w * x * x - w * (2 * n + 1)) * val
    return val, d1, d2


def basis_indices(wb, ecut_mult=6.0, degree=4, parity=False):
    """Union of the aniGTO total-degree set (n_x+n_y+n_z <= degree) and the energy cut sum_k wb_k n_k <= ecut_mult*min(wb)."""
    wb = np.asarray(wb, float)
    ecut = ecut_mult * wb.min()
    s = set()
    for n in range(degree + 1):
        for nx in range(n + 1):
            for ny in range(n - nx + 1):
                s.add((nx, ny, n - nx - ny))
    mx = [int(ecut / w + 1e-9) for w in wb]
    for nx in range(mx[0] + 1):
        for ny in range(mx[1] + 1):
            for nz in range(mx[2] + 1):
                if wb[0] * nx + wb[1] * ny + wb[2] * nz <= ecut + 1e-9:
                    s.add((nx, ny, nz))
    idx = sorted(s, key=lambda t: ((wb * np.array(t)).sum(), t))
    idx = np.array(idx, int)
    if parity:
        idx = idx[(idx % 2 == 0).all(1)]
    return idx


def basis_for_target(wb, nb_target, degree=4, nmax_dir=20):
    """Smallest energy cut (steps of 0.25 min(wb)) whose union with the [spdfg] set has >= nb_target functions,
    with at most nmax_dir quanta per direction (keeps the 1D ERI tables small for very anisotropic traps);
    keeps the basis size, hence the FCI basis error, comparable across traps of different anisotropy."""
    ec, last, same = 0.0, -1, 0
    while True:
        idx = basis_indices(wb, ec, degree)
        idx = idx[(idx <= nmax_dir).all(1)]
        if len(idx) >= nb_target or same > 400:
            return idx, ec
        same = same + 1 if len(idx) == last else 0
        last = len(idx)
        ec += 0.25


def u_nodes(wb, n=300, lo_pad=30.0, hi_pad=22.0):
    """Gauss-Legendre nodes in s = ln u for the Boys-type u integral; returns u and the weights of du."""
    wb = np.asarray(wb, float)
    lo = np.log(np.sqrt(wb.min())) - lo_pad
    hi = np.log(np.sqrt(wb.max())) + hi_pad
    x, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    return np.exp(s), 0.5 * (hi - lo) * w * np.exp(s)


def one_dim_eri(w, nmax, u):
    """I(u)[(p,q),(r,s)] = int int chi_p chi_q(x1) chi_r chi_s(x2) exp(-u^2 (x1-x2)^2), exact Gauss-Hermite."""
    ng = 2 * nmax + 2
    t, wt = hermgauss(ng)
    n = nmax + 1
    X, wX = t / np.sqrt(w), wt / np.sqrt(w)
    out = np.empty((len(u), n * n, n * n))
    for iu, uu in enumerate(u):
        b = w + 2.0 * uu * uu
        xi, wxi = t / np.sqrt(b), wt / np.sqrt(b)
        x1 = ((X[:, None] + xi[None, :]) / np.sqrt(2.0)).ravel()
        x2 = ((X[:, None] - xi[None, :]) / np.sqrt(2.0)).ravel()
        W2 = (wX[:, None] * wxi[None, :]).ravel()
        P1 = herm_poly_norm(nmax, np.sqrt(w) * x1) * w ** 0.25
        P2 = herm_poly_norm(nmax, np.sqrt(w) * x2) * w ** 0.25
        A = (P1[:, None, :] * P1[None, :, :]).reshape(n * n, -1)
        B = (P2[:, None, :] * P2[None, :, :]).reshape(n * n, -1)
        out[iu] = (A * W2) @ B.T
    return out


def one_dim_nuc(w, nmax, u):
    """int chi_p chi_q exp(-u^2 x^2) dx for all u, exact Gauss-Hermite."""
    t, wt = hermgauss(nmax + 2)
    out = np.empty((len(u), nmax + 1, nmax + 1))
    for iu, uu in enumerate(u):
        b = w + uu * uu
        x = t / np.sqrt(b)
        P = herm_poly_norm(nmax, np.sqrt(w) * x) * w ** 0.25
        out[iu] = (P * (wt / np.sqrt(b))) @ P.T
    return out


def one_dim_triple(w, nmax):
    """int chi_p chi_q chi_t dx (exact Gauss-Hermite, weight exp(-3 w x^2/2))."""
    t, wt = hermgauss(3 * nmax // 2 + 2)
    a = 1.5 * w
    x = t / np.sqrt(a)
    P = herm_poly_norm(nmax, np.sqrt(w) * x) * w ** 0.25
    return np.einsum('pa,qa,ta,a->pqt', P, P, P, wt / np.sqrt(a))


def h1d(wb, w, nmax):
    """Kinetic and trap matrices <m| p^2/2 |n>, <m| w^2 x^2/2 |n> in the HO(wb) basis (ladder operators)."""
    n = np.arange(nmax + 1)
    X2 = np.diag((2 * n + 1) / (2 * wb))
    if nmax >= 2:
        off = np.sqrt((n[:-2] + 1) * (n[:-2] + 2)) / (2 * wb)
        X2[np.arange(nmax - 1), np.arange(2, nmax + 1)] = off
        X2[np.arange(2, nmax + 1), np.arange(nmax - 1)] = off
    T = np.diag(wb * (n + 0.5)) - 0.5 * wb * wb * X2
    return T, 0.5 * w * w * X2


try:
    from numba import njit, prange

    @njit(parallel=True)
    def _pair_sum(Ix, Iy, Iz, wu, px, py, pz):
        n = px.shape[0]
        nu = wu.shape[0]
        P = np.zeros((n, n))
        for a in prange(n):
            for b in range(a + 1):
                s = 0.0
                for k in range(nu):
                    s += wu[k] * Ix[px[a], px[b], k] * Iy[py[a], py[b], k] * Iz[pz[a], pz[b], k]
                P[a, b] = s
                P[b, a] = s
        return P
except ImportError:
    def _pair_sum(Ix, Iy, Iz, wu, px, py, pz):
        P = np.zeros((len(px), len(px)))
        for k in range(len(wu)):
            P += wu[k] * Ix[:, :, k][np.ix_(px, px)] * Iy[:, :, k][np.ix_(py, py)] * Iz[:, :, k][np.ix_(pz, pz)]
        return P


def build(trap, wb, idx, Z=0.0, nu=300, want_eri=True):
    """One- and two-electron integrals in the orthonormal HO(wb) basis idx: T, V_trap, V_nuc, h, eri (chemists')."""
    trap = np.asarray(trap, float)
    wb = np.asarray(wb, float)
    nb = len(idx)
    T = np.zeros((nb, nb))
    Vt = np.zeros((nb, nb))
    for k in range(3):
        Tk, Vk = h1d(wb[k], trap[k], idx[:, k].max())
        others = [m for m in range(3) if m != k]
        same = np.all(idx[:, None, others] == idx[None, :, others], axis=2)
        T += same * Tk[np.ix_(idx[:, k], idx[:, k])]
        Vt += same * Vk[np.ix_(idx[:, k], idx[:, k])]
    u, wu = u_nodes(wb, nu)
    Vn = np.zeros((nb, nb))
    if Z:
        Nn = [one_dim_nuc(wb[k], idx[:, k].max(), u) for k in range(3)]
        for a in range(len(u)):
            Vn += wu[a] * (Nn[0][a][np.ix_(idx[:, 0], idx[:, 0])] * Nn[1][a][np.ix_(idx[:, 1], idx[:, 1])]
                           * Nn[2][a][np.ix_(idx[:, 2], idx[:, 2])])
        Vn *= -Z * 2.0 / np.sqrt(np.pi)
    out = dict(T=T, Vtrap=Vt, Vnuc=Vn, h=T + Vt + Vn, idx=idx, trap=trap, wb=wb, Z=Z)
    if want_eri:
        n1 = [idx[:, k].max() + 1 for k in range(3)]
        iu, ju = np.triu_indices(nb)
        pair = [idx[iu, k] * n1[k] + idx[ju, k] for k in range(3)]
        Ik = [np.ascontiguousarray(one_dim_eri(wb[k], idx[:, k].max(), u).transpose(1, 2, 0)) for k in range(3)]
        P = _pair_sum(Ik[0], Ik[1], Ik[2], wu, pair[0], pair[1], pair[2]) * (2.0 / np.sqrt(np.pi))
        full = np.zeros((nb, nb, len(iu)))
        full[iu, ju] = P
        full[ju, iu] = P
        eri = np.zeros((nb, nb, nb, nb))
        eri[:, :, iu, ju] = full
        eri[:, :, ju, iu] = full
        out['eri'] = eri
    return out


def jk(eri, D):
    return np.einsum('pqrs,rs->pq', eri, D), np.einsum('prqs,rs->pq', eri, D)


def rhf(h, eri, maxit=500, tol=1e-12):
    """Closed-shell HF for two electrons (one doubly occupied orbital)."""
    e, C = np.linalg.eigh(h)
    D = 2 * np.outer(C[:, 0], C[:, 0])
    Eold = 0.0
    for it in range(maxit):
        J, K = jk(eri, D)
        F = h + J - 0.5 * K
        E = 0.5 * np.sum(D * (h + F))
        e, C = np.linalg.eigh(F)
        Dn = 2 * np.outer(C[:, 0], C[:, 0])
        if abs(E - Eold) < tol and np.abs(Dn - D).max() < 1e-10:
            D = Dn
            break
        D = Dn if it < 2 else 0.5 * (D + Dn)
        Eold = E
    J, K = jk(eri, D)
    F = h + J - 0.5 * K
    e, C = np.linalg.eigh(F)
    return float(0.5 * np.sum(D * (h + F))), e, C, D


def fci_singlet2(h, eri, v0=None, tol=1e-12):
    """Exact two-electron singlet (FCI): Psi = sum_pq C_pq chi_p(1) chi_q(2), C symmetric, sum C^2 = 1,
    (H C)_pq = (h C + C h)_pq + sum_rs (pr|qs) C_rs.  Returns E, C, spin-summed 1-RDM D = 2 C C^T, V_ee."""
    nb = h.shape[0]
    iu, ju = np.triu_indices(nb)
    sc = np.where(iu == ju, 1.0, np.sqrt(2.0))
    eri_f = np.ascontiguousarray(eri.transpose(0, 2, 1, 3).reshape(nb * nb, nb * nb))

    def unpack(v):
        C = np.zeros((nb, nb))
        C[iu, ju] = v / sc
        C[ju, iu] = v / sc
        return C

    def mv(v):
        C = unpack(v)
        HC = h @ C + C @ h + (eri_f @ C.ravel()).reshape(nb, nb)
        return HC[iu, ju] * sc
    n = len(iu)
    if n <= 700:
        H = np.array([mv(e) for e in np.eye(n)])
        w, V = np.linalg.eigh(0.5 * (H + H.T))
        E, v = w[0], V[:, 0]
    else:
        if v0 is None:
            v0 = np.zeros(n)
            v0[0] = 1.0
        else:
            v0 = v0[iu, ju] * sc
        w, V = eigsh(LinearOperator((n, n), matvec=mv), k=1, which='SA', v0=v0, tol=tol)
        E, v = w[0], V[:, 0]
    C = unpack(v)
    if C[0, 0] < 0:
        C = -C
    return float(E), C, 2.0 * C @ C.T, float(C.ravel() @ (eri_f @ C.ravel()))


# ==============================================================================================================
# B. Exact two-electron trap energy (centre-of-mass separation)
# ==============================================================================================================
def exact_rel_energy(trap, nmax, nu=300):
    """sum(w)/2 + lowest eigenvalue of HO(w) + (1/sqrt2)/|s| in the even tensor HO basis n_k <= nmax."""
    trap = np.asarray(trap, float)
    u, wu = u_nodes(trap, nu)
    ev = [np.arange(0, nmax + 1, 2) for _ in range(3)]
    N = [one_dim_nuc(trap[k], nmax, u)[:, ev[k]][:, :, ev[k]] for k in range(3)]
    diag = (trap[0] * (ev[0] + 0.5))[:, None, None] + (trap[1] * (ev[1] + 0.5))[None, :, None] \
        + (trap[2] * (ev[2] + 0.5))[None, None, :]
    shp = diag.shape
    c = (2.0 / np.sqrt(np.pi)) / np.sqrt(2.0) * wu

    def mv(v):
        v = v.reshape(shp)
        out = diag * v
        for a in range(len(u)):
            t = np.einsum('ia,abc->ibc', N[0][a], v)
            t = np.einsum('jb,ibc->ijc', N[1][a], t)
            out = out + c[a] * np.einsum('kc,ijc->ijk', N[2][a], t)
        return out.ravel()
    n = diag.size
    v0 = np.zeros(n)
    v0[0] = 1.0
    e0 = eigsh(LinearOperator((n, n), matvec=mv), k=1, which='SA', tol=1e-12, v0=v0)[0][0]
    return trap.sum() / 2.0 + e0


def exact_trap_energy(trap, nlist=(20, 30, 40)):
    """Extrapolate E(N) = E_inf + a N^-p from three tensor-basis sizes."""
    E = [exact_rel_energy(trap, n) for n in nlist]
    n1, n2, n3 = nlist
    d1, d2 = E[0] - E[1], E[1] - E[2]
    if d1 > 0 and d2 > 0:
        f = lambda p: (n1 ** -p - n2 ** -p) / (n2 ** -p - n3 ** -p) - d1 / d2
        try:
            from scipy.optimize import brentq
            p = brentq(f, 0.3, 6.0)
        except ValueError:
            p = 1.5
    else:
        p = 1.5
    a = d2 / (n2 ** -p - n3 ** -p)
    return E[2] - a * n3 ** -p, E, p


# ==============================================================================================================
# C. Grid, densities and derivatives
# ==============================================================================================================
def tensor_grid(wb, idx, nmin=28, nmax_pts=56):
    """Tensor Gauss-Hermite grid adapted to exp(-wb_k x_k^2) in every direction."""
    axes, wts = [], []
    for k in range(3):
        npt = int(np.clip(2 * idx[:, k].max() + 24, nmin, nmax_pts))
        t, w = hermgauss(npt)
        a = wb[k]
        axes.append(t / np.sqrt(a))
        wts.append(w * np.exp(t * t) / np.sqrt(a))
    return axes, wts


def _log_gl(lo, hi, n):
    """Gauss-Legendre nodes in log coordinates on [lo, hi]: x and weights for dx (as script 03)."""
    p, w = np.polynomial.legendre.leggauss(n)
    a, b = np.log(lo), np.log(hi)
    q = 0.5 * (b - a) * p + 0.5 * (b + a)
    x = np.exp(q)
    return x, 0.5 * (b - a) * w * x


def cylinder_grid(wb, n=300):
    """Axially symmetric quadrature (omega_x = omega_y): points (rho_perp, 0, z) with u = wb_x rho_perp^2 on
    [e^-32, 40] and t = sqrt(wb_z) z on [e^-32, 9] (log Gauss-Legendre, as script 03), weights pi du/wb_x * 2 dz.
    Exact for axially symmetric, z-even integrands such as density functionals of these traps."""
    u, wu = _log_gl(np.exp(-32.0), 40.0, n)
    t, wt = _log_gl(np.exp(-32.0), 9.0, n)
    rp = np.sqrt(u / wb[0])
    z = t / np.sqrt(wb[2])
    W = np.outer(np.pi * wu / wb[0], 2.0 * wt / np.sqrt(wb[2])).ravel()
    return [rp, np.zeros(1), z], W


class Grid:
    """AO values/derivatives of the HO basis on a tensor grid (separable evaluation).  kind='tensor' (Gauss-Hermite,
    for matrix elements) or 'cylinder' (axially symmetric energy integrals; points in the y = 0 half plane)."""

    def __init__(self, wb, idx, kind='tensor', n=300):
        self.idx = idx
        if kind == 'tensor':
            self.axes, self.wts = tensor_grid(wb, idx)
            self.W = np.einsum('i,j,k->ijk', *self.wts).ravel()
        else:
            self.axes, self.W = cylinder_grid(wb, n)
        self.f = [ho_1d(self.axes[k], idx[:, k].max(), wb[k]) for k in range(3)]   # (val, d1, d2) per direction
        self.shape = tuple(len(a) for a in self.axes)

    def ao(self, order=(0, 0, 0)):
        """chi_p(g) with derivative orders (ox, oy, oz) in {0,1,2}; returns (nb, G)."""
        ix, iy, iz = self.idx[:, 0], self.idx[:, 1], self.idx[:, 2]
        X = self.f[0][order[0]][ix]
        Y = self.f[1][order[1]][iy]
        Zz = self.f[2][order[2]][iz]
        return (X[:, :, None, None] * Y[:, None, :, None] * Zz[:, None, None, :]).reshape(len(ix), -1)

    def fields(self, D):
        """rho, grad (3,G), Hessian (3,3,G), laplacian, tau (from D) for a spin-summed density matrix D."""
        c0 = self.ao()
        Dc = D @ c0
        rho = np.einsum('pg,pg->g', c0, Dc)
        d1 = [self.ao(tuple(int(i == k) for i in range(3))) for k in range(3)]
        Dd1 = [D @ d for d in d1]
        grad = np.array([2.0 * np.einsum('pg,pg->g', d1[k], Dc) for k in range(3)])
        hess = np.zeros((3, 3, len(rho)))
        for i in range(3):
            for j in range(i, 3):
                o = [0, 0, 0]
                o[i] += 1
                o[j] += 1
                d2 = self.ao(tuple(o))
                hij = 2.0 * (np.einsum('pg,pg->g', d2, Dc) + np.einsum('pg,pg->g', d1[i], Dd1[j]))
                hess[i, j] = hij
                hess[j, i] = hij
        tau = 0.5 * sum(np.einsum('pg,pg->g', d1[k], Dd1[k]) for k in range(3))
        return rho, grad, hess, np.trace(hess), tau


def ao_points(wb, idx, xyz):
    """chi_p at arbitrary points (nb, npts)."""
    out = np.ones((len(idx), len(xyz)))
    for k in range(3):
        v = ho_1d(xyz[:, k], idx[:, k].max(), wb[k])[0]
        out *= v[idx[:, k]]
    return out


# ==============================================================================================================
# D. Exact N = 2 Kohn-Sham inversion, GL2
# ==============================================================================================================
def ks_exact_2e(grid, T, D_tar, rho_floor=1e-12):
    """phi = sqrt(rho/2): v_s - eps_H = lap(phi)/(2 phi) = lap rho/(4 rho) - |grad rho|^2/(8 rho^2).
    Diagonalising T + v_s in the basis gives eps_H ~ 0 (check) and the virtual KS orbitals for GL2."""
    rho, grad, hess, lap, _ = grid.fields(D_tar)
    ok = rho > rho_floor * rho.max()
    vs = np.zeros_like(rho)
    g2 = np.einsum('cg,cg->g', grad, grad)
    vs[ok] = lap[ok] / (4.0 * rho[ok]) - g2[ok] / (8.0 * rho[ok] ** 2)
    c0 = grid.ao()
    V = np.einsum('pg,qg,g->pq', c0[:, ok], c0[:, ok], vs[ok] * grid.W[ok])
    e, C = np.linalg.eigh(T + V)
    Ts_vW = float(np.sum(grid.W[ok] * g2[ok] / rho[ok])) / 8.0
    D_ks = 2.0 * np.outer(C[:, 0], C[:, 0])
    return dict(eps=e, C=C, Ts=Ts_vW, Ts_orb=float(np.sum(T * D_ks)), D=D_ks, V=V)


def gl2_energy(eri, C, e):
    """E_c^GL2: doubles with KS orbitals (singles vanish for N = 2 because v_x = -v_H/2)."""
    o, v = C[:, :1], C[:, 1:]
    ovov = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri, o, v, o, v, optimize=True).reshape(v.shape[1], v.shape[1])
    den = 2 * e[0] - e[1:, None] - e[None, 1:]
    return float(np.sum(ovov ** 2 / den))


def rho_l2_diff(grid, Da, Db):
    c0 = grid.ao()
    ra = np.einsum('pg,pq,qg->g', c0, Da, c0)
    rb = np.einsum('pg,pq,qg->g', c0, Db, c0)
    return float(np.sqrt(np.sum(grid.W * (ra - rb) ** 2)))


# ==============================================================================================================
# E. Lieb maximisation at lambda = 1 (density-constrained FCI)
# ==============================================================================================================
def triple_tensor(wb, idx):
    """S3[p,q,t] = int chi_p chi_q chi_t (potential basis g_t = chi_t), analytic per direction."""
    S = np.ones((len(idx),) * 3)
    for k in range(3):
        Tk = one_dim_triple(wb[k], idx[:, k].max())
        S *= Tk[np.ix_(idx[:, k], idx[:, k], idx[:, k])]
    return S


def lieb_lambda1(h, eri, S3, D_tar, maxiter=400):
    """max_b { E_FCI[h + sum_t b_t g_t] - sum_t b_t <g_t>_tar }; returns V_ee of the maximiser and its density."""
    tgt = np.einsum('pqt,pq->t', S3, D_tar)
    st = {'C': None}

    def ev(b):
        h1 = h + np.einsum('pqt,t->pq', S3, b)
        E, C, D, Vee = fci_singlet2(h1, eri, v0=st['C'])
        st.update(C=C, E=E, h1=h1, D=D, Vee=Vee)
        return E, D

    def negG(b):
        E, Dl = ev(b)
        return -(E - b @ tgt), -(np.einsum('pqt,pq->t', S3, Dl) - tgt)
    r = minimize(negG, np.zeros(S3.shape[2]), jac=True, method='L-BFGS-B',
                 options={'maxiter': maxiter, 'gtol': 1e-9, 'ftol': 0.0, 'maxcor': 50})
    ev(r.x)
    return dict(Vee=st['Vee'], D=st['D'], nit=r.nit, b=r.x)


# --------------------------------------------------------------------------------------------------------------
# (copied from 12_trap_ac_benchmark.py, section I)
def optimise_zeta(trap, Z, degree=2):
    """Isotropic basis exponent zeta (wb_k = max(omega_k, zeta)) minimising E_HF in a small degree-2 basis."""
    trap = np.asarray(trap, float)

    def f(lz):
        wb = np.maximum(trap, np.exp(lz))
        idx = basis_indices(wb, 0.0, degree)
        d = build(trap, wb, idx, Z=Z)
        return rhf(d['h'], d['eri'])[0]
    res = minimize_scalar(f, bounds=(np.log(trap.min()) - 1, np.log(max(trap.max(), 4.0 * Z * Z)) + 1),
                          method='bounded', options={'xatol': 1e-3})
    return float(np.exp(res.x))


# ==============================================================================================================
# E13. N = 3 FCI, KS-SCF with libxc potentials, Wu-Yang inversion, G0W0
# ==============================================================================================================
def fci_n3(h, eri):
    """Doublet ground state of three electrons (PySCF direct_spin1, nelec = (2, 1))."""
    from pyscf.fci import direct_spin1
    s = direct_spin1.FCI()
    s.conv_tol = 1e-9
    s.verbose = 0
    s.max_cycle = 300
    E, _ = s.kernel(h, eri, h.shape[0], (2, 1))
    return float(E)


XC_SCF = {  # name: (libxc string for PySCF, full-range exact exchange a, long-range extra b, omega)
    'LDA': ('LDA_X,LDA_C_PW', 0.0, 0.0, 0.0),
    'PBE': ('GGA_X_PBE,GGA_C_PBE', 0.0, 0.0, 0.0),
    'r2SCAN': ('MGGA_X_R2SCAN,MGGA_C_R2SCAN', 0.0, 0.0, 0.0),
    'SCAN': ('MGGA_X_SCAN,MGGA_C_SCAN', 0.0, 0.0, 0.0),
    'PBE0': ('0.75*GGA_X_PBE,GGA_C_PBE', 0.25, 0.0, 0.0),
    # CAM-QTP-02 (libxc HYB_GGA_XC_CAM_QTP_02; PySCF rsh coefficients omega = 0.335, alpha = 1, hyb = 0.28):
    # K_eff = 0.28 K + 0.72 K_LR(omega), the semilocal part from libxc (aditi Table 2 "QTP02")
    'QTP02': ('HYB_GGA_XC_CAM_QTP_02', 0.28, 0.72, 0.335),
}


def eri_long_range(wb, idx, omega, nu=120):
    """(pq|rs) with erf(omega r12)/r12 = (2/sqrt(pi)) int_0^omega du exp(-u^2 r12^2): the same exact 1D tables as
    build(), with Gauss-Legendre nodes on u in [0, omega]."""
    nb = len(idx)
    x, w = np.polynomial.legendre.leggauss(nu)
    u = 0.5 * omega * (x + 1.0)
    wu = 0.5 * omega * w
    n1 = [idx[:, k].max() + 1 for k in range(3)]
    iu, ju = np.triu_indices(nb)
    pair = [idx[iu, k] * n1[k] + idx[ju, k] for k in range(3)]
    Ik = [np.ascontiguousarray(one_dim_eri(wb[k], idx[:, k].max(), u).transpose(1, 2, 0)) for k in range(3)]
    P = _pair_sum(Ik[0], Ik[1], Ik[2], wu, pair[0], pair[1], pair[2]) * (2.0 / np.sqrt(np.pi))
    full = np.zeros((nb, nb, len(iu)))
    full[iu, ju] = P
    full[ju, iu] = P
    eri = np.zeros((nb, nb, nb, nb))
    eri[:, :, iu, ju] = full
    eri[:, :, ju, iu] = full
    return eri


def mp2_one_orbital(h, eri, C, e):
    """Closed-shell MP2 for one doubly occupied orbital: t_ab = (0a|0b)/(2e_0 - e_a - e_b),
    E_c = sum (0a|0b) t_ab, unrelaxed spin-summed 1-RDM D_00 = 2 - 2 sum t^2, D_ab = 2 (t t^T)_ab
    (PySCF mp2 _gamma1_intermediates for nocc = 1); returned in the AO (orthonormal HO) basis."""
    o, v = C[:, :1], C[:, 1:]
    ovov = np.einsum('pqrs,pi,qa,rj,sb->ab', eri, o, v, o, v, optimize=True)
    t = ovov / (2 * e[0] - e[1:, None] - e[None, 1:])
    Ec = float(np.sum(ovov * t))
    nv = v.shape[1]
    Dmo = np.zeros((nv + 1, nv + 1))
    Dmo[0, 0] = 2.0 - 2.0 * np.sum(t * t)
    Dmo[1:, 1:] = 2.0 * t @ t.T
    return Ec, C @ Dmo @ C.T


def xc_matrix(grid, D, code, ao_cache):
    """E_xc and V_xc[pq] = int vrho chi_p chi_q + 2 vsigma grad rho . grad(chi_p chi_q) + 1/2 vtau grad chi_p . grad chi_q
    (PySCF/libxc conventions, tau = 1/2 sum_occ |grad phi|^2 spin-summed)."""
    from pyscf.dft import numint
    ni = numint.NumInt()
    c0, d1 = ao_cache
    Dc = D @ c0
    rho = np.einsum('pg,pg->g', c0, Dc)
    grad = np.array([2.0 * np.einsum('pg,pg->g', d1[k], Dc) for k in range(3)])
    tau = 0.5 * sum(np.einsum('pg,pg->g', d1[k], D @ d1[k]) for k in range(3))
    xctype = ni._xc_type(code)
    rho = np.maximum(rho, 1e-30)
    if xctype == 'LDA':
        inp = rho
    elif xctype == 'GGA':
        inp = np.vstack([rho, grad])
    else:
        inp = np.vstack([rho, grad, np.zeros_like(rho), tau])
    exc, vxc = ni.eval_xc(code, inp, spin=0, deriv=1)[:2]
    W = grid.W
    E = float(np.sum(W * rho * exc))
    vrho = vxc[0]
    V = (c0 * (W * vrho)) @ c0.T
    if xctype in ('GGA', 'MGGA'):
        vs = vxc[1]
        Vg = sum((d1[k] * (2.0 * W * vs * grad[k])) @ c0.T for k in range(3))
        V += Vg + Vg.T
    if xctype == 'MGGA':
        vt = vxc[3]
        V += 0.5 * sum((d1[k] * (W * vt)) @ d1[k].T for k in range(3))
    return E, V


def ks_scf(h, eri, grid, name, ao_cache, eri_lr=None, maxit=300, tol=1e-10):
    """Closed-shell KS-SCF (two electrons) with DIIS; returns E, eps, C, D, V_xc (incl. the exact-exchange part
    -(a K + b K_LR)/2 of hybrids / range-separated hybrids)."""
    code, a, b, om = XC_SCF[name]

    def kx(D):
        J, K = jk(eri, D)
        Kx = a * K
        if b:
            Kx = Kx + b * jk(eri_lr, D)[1]
        return J, Kx
    e, C = np.linalg.eigh(h)
    D = 2 * np.outer(C[:, 0], C[:, 0])
    Eold, focks, errs = 0.0, [], []
    for it in range(maxit):
        J, Kx = kx(D)
        Exc, Vxc = xc_matrix(grid, D, code, ao_cache)
        Vx = Vxc - 0.5 * Kx
        F = h + J + Vx
        E = float(np.sum(D * h) + 0.5 * np.sum(D * J) - 0.25 * np.sum(D * Kx) + Exc)
        err = F @ D - D @ F
        focks.append(F)
        errs.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errs.pop(0)
        if len(focks) > 1:
            n = len(focks)
            B = -np.ones((n + 1, n + 1))
            B[n, n] = 0.0
            for i in range(n):
                for j in range(n):
                    B[i, j] = np.sum(errs[i] * errs[j])
            rhs = np.zeros(n + 1)
            rhs[n] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:n]
                F = sum(ci * fi for ci, fi in zip(c, focks))
            except np.linalg.LinAlgError:
                pass
        e, C = np.linalg.eigh(F)
        D = 2 * np.outer(C[:, 0], C[:, 0])
        if abs(E - Eold) < tol and np.abs(err).max() < 1e-7:
            break
        Eold = E
    J, Kx = kx(D)
    Exc, Vxc = xc_matrix(grid, D, code, ao_cache)
    Vx = Vxc - 0.5 * Kx
    e, C = np.linalg.eigh(h + J + Vx)
    return dict(E=E, eps=e, C=C, D=D, Vxc=Vx, it=it)


def g0w0_exact(eps, C, eri_ao, nocc, vxc_mo_diag, orbs, iters=60):
    """Closed-shell G0W0, exact frequency integration through the RPA (direct ring, singlet) Casida poles:
    Sigma_c,pp(w) = sum_s [sum_i w_pi^s^2 / (w - e_i + O_s) + sum_a w_pa^s^2 / (w - e_a - O_s)],
    Sigma_x,pp = -sum_i (pi|ip); QP equation e = eps + Sigma_x + Re Sigma_c(e) - v_xc solved by Newton.
    Returns {p: (e_QP, Z_lin, Sigma_x, Sigma_c(e_QP))}."""
    nmo = len(eps)
    o, v = slice(0, nocc), slice(nocc, nmo)
    eri = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao, C, C, C, C, optimize=True)
    nv = nmo - nocc
    d = (eps[v][None, :] - eps[o][:, None]).ravel()
    K = 2.0 * eri[o, v, o, v].reshape(nocc * nv, nocc * nv)
    sd = np.sqrt(d)
    W2, Zv = np.linalg.eigh(np.diag(d * d) + 2.0 * sd[:, None] * K * sd[None, :])
    Om = np.sqrt(W2)
    XpY = sd[:, None] * Zv / np.sqrt(Om)[None, :]
    wpq = np.sqrt(2.0) * np.einsum('pqia,ias->pqs', eri[:, :, o, v], XpY.reshape(nocc, nv, -1))
    sx = -np.einsum('piip->p', eri[:, o, o, :])
    out = {}
    for p in orbs:
        def sig(w):
            return (np.sum(wpq[p, o, :] ** 2 / (w - eps[o][:, None] + Om[None, :]))
                    + np.sum(wpq[p, v, :] ** 2 / (w - eps[v][:, None] - Om[None, :])))

        def dsig(w):
            return (-np.sum(wpq[p, o, :] ** 2 / (w - eps[o][:, None] + Om[None, :]) ** 2)
                    - np.sum(wpq[p, v, :] ** 2 / (w - eps[v][:, None] - Om[None, :]) ** 2))
        e = eps[p]
        for _ in range(iters):
            f = e - eps[p] - sig(e) - sx[p] + vxc_mo_diag[p]
            e_new = e - f / (1.0 - dsig(e))
            if abs(e_new - e) < 1e-11:
                e = e_new
                break
            e = e_new
        out[p] = (float(e), float(1.0 / (1.0 - dsig(eps[p]))), float(sx[p]), float(sig(e)))
    return out


# ==============================================================================================================
# F13. One system
# ==============================================================================================================
EQ8_STARTS = ('HF', 'PyOEP[HF]', 'PyOEP[MP2]', 'PyOEP[FCI]', 'LDA', 'PBE', 'SCAN', 'PBE0', 'QTP02')


def run_ip(label, family, lam, trap, Z, wb, nb_target, degree):
    idx, ecut = basis_for_target(wb, nb_target, degree)
    t = time.time()
    ints = build(trap, wb, idx, Z=Z)
    h, eri, T = ints['h'], ints['eri'], ints['T']
    Vext = ints['Vtrap'] + ints['Vnuc']
    E1 = float(np.linalg.eigvalsh(h)[0])
    E2, C2, D2, Vee = fci_singlet2(h, eri)
    E3 = fci_n3(h, eri)
    I, A = E1 - E2, E2 - E3
    log(f'{label}: nb={len(idx)}  E1={E1:.9f} E2={E2:.9f} E3={E3:.9f}  I={I:.9f}  A={A:.9f}  ({time.time() - t:.1f}s)')
    grid = Grid(wb, idx)
    ao_cache = (grid.ao(), [grid.ao(tuple(int(i == k) for i in range(3))) for k in range(3)])
    r = dict(system=label, family=family, lam=lam, Z=Z, omega='x'.join(f'{w:g}' for w in trap), nb=len(idx),
             E1=E1, E2_FCI=E2, E3_FCI=E3, I=I, A=A, gap_fund=I - A)
    # exact KS from the FCI density, HOMO condition eps_H = -I
    ks = ks_exact_2e(grid, T, D2)
    shift = -I - ks['eps'][0]
    epsKS = ks['eps'] + shift
    r.update(KS_HOMO=epsKS[0], KS_LUMO=epsKS[1], KS_gap=epsKS[1] - epsKS[0],
             Dxc=(I - A) - (epsKS[1] - epsKS[0]), KS_drho=rho_l2_diff(grid, ks['D'], D2))
    log(f'   exact KS: HOMO={epsKS[0]:.6f} (= mu2 = E2 - E1) LUMO={epsKS[1]:.6f} gap={epsKS[1] - epsKS[0]:.6f} '
        f'Dxc={r["Dxc"]:.6f}  |drho_KS|={r["KS_drho"]:.1e}')
    # HF and DFA frontier orbitals
    E_hf, e_hf, C_hf, D_hf = rhf(h, eri)
    r.update(E_HF=E_hf, HF_HOMO=e_hf[0], HF_LUMO=e_hf[1])
    starts = {'HF': (e_hf, C_hf, np.einsum('pi,pq,qi->i', C_hf, -0.5 * jk(eri, D_hf)[1], C_hf))}
    # PyOEP[HF]: for N = 2 the HF orbital is exactly the lowest eigenfunction of the local h + J/2
    # (v_x = -v_H/2), so the KS inversion of the HF density is analytic: v_xc = -J[rho_HF]/2.
    J_hf = jk(eri, D_hf)[0]
    e_oh, C_oh = np.linalg.eigh(h + 0.5 * J_hf)
    starts['PyOEP[HF]'] = (e_oh, C_oh, np.einsum('pi,pq,qi->i', C_oh, -0.5 * J_hf, C_oh))
    # PyOEP[MP2]: inversion of the unrelaxed MP2 density, constant fixed by the HOMO condition with the
    # Delta-MP2 ionization energy E1 - E_MP2(2) (PyOEP KSINV "ip" option)
    Ec_mp2, D_mp2 = mp2_one_orbital(h, eri, C_hf, e_hf)
    ks_m = ks_exact_2e(grid, T, D_mp2)
    I_mp2 = E1 - (E_hf + Ec_mp2)
    sh_m = -I_mp2 - ks_m['eps'][0]
    starts['PyOEP[MP2]'] = (ks_m['eps'] + sh_m, ks_m['C'], np.einsum(
        'pi,pq,qi->i', ks_m['C'], ks_m['V'] + sh_m * np.eye(len(idx)) - Vext - jk(eri, ks_m['D'])[0], ks_m['C']))
    r.update(E_MP2=E_hf + Ec_mp2, I_MP2=I_mp2, MP2_drho=rho_l2_diff(grid, ks_m['D'], D_mp2))
    eri_lr = eri_long_range(wb, idx, XC_SCF['QTP02'][3])
    for name in XC_SCF:
        try:
            s = ks_scf(h, eri, grid, name, ao_cache, eri_lr=eri_lr)
            r.update({f'{name}_E': s['E'], f'{name}_HOMO': s['eps'][0], f'{name}_LUMO': s['eps'][1]})
            starts[name] = (s['eps'], s['C'], np.einsum('pi,pq,qi->i', s['C'], s['Vxc'], s['C']))
            log(f'   {name:7s} E={s["E"]:.8f} HOMO={s["eps"][0]:.6f} LUMO={s["eps"][1]:.6f} (it {s["it"]})')
        except Exception as exc:                          # record, do not abort the table
            log(f'   {name}: SCF failed ({exc})')
            r.update({f'{name}_E': np.nan, f'{name}_HOMO': np.nan, f'{name}_LUMO': np.nan})
    vxcKS = np.einsum('pi,pq,qi->i', ks['C'], ks['V'] + shift * np.eye(len(idx)) - Vext - jk(eri, D2)[0], ks['C'])
    starts['KS-exact'] = (epsKS, ks['C'], vxcKS)
    starts['PyOEP[FCI]'] = starts['KS-exact']                     # exact KS (= PyOEP[CCSD(T)] for N = 2)
    for st in ('HF', 'PBE', 'KS-exact') + EQ8_STARTS:
        if st not in starts or (st == 'PyOEP[FCI]'):
            continue
        eps, C, vxc = starts[st]
        gw = g0w0_exact(eps, C, eri, 1, vxc, [0, 1])
        for key in ((st, 'PyOEP[FCI]') if st == 'KS-exact' else (st,)):
            r.update({f'GW@{key}_HOMO': gw[0][0], f'GW@{key}_LUMO': gw[1][0], f'GW@{key}_Z': gw[0][1]})
            # eq. (8): eps_QP = eps + <Sigma(eps_QP)> - <v_xc>, <Sigma> = Sigma_x + Sigma_c(eps_QP)
            for p, o in ((0, 'H'), (1, 'L')):
                r.update({f'eq8_{key}_{o}_qp': gw[p][0], f'eq8_{key}_{o}_eps': float(eps[p]),
                          f'eq8_{key}_{o}_sig': gw[p][2] + gw[p][3], f'eq8_{key}_{o}_vxc': float(vxc[p])})
        log(f'   G0W0@{st}: IP={-gw[0][0]:.6f} (exact {I:.6f})  EA={-gw[1][0]:.6f} (exact {A:.6f})')
    return r


# ==============================================================================================================
# G13. Tables and main
# ==============================================================================================================
def mha(x):
    return '--' if not np.isfinite(x) else (f'$+${1e3 * x:.2f}' if x >= 0 else f'$-${abs(1e3 * x):.2f}')


def name_of(r):
    s = f"{r['family']} $\\lambda$={r['lam']:g}"
    return s + (f", $Z$={r['Z']:g}" if r['Z'] else '')


def write_tables(rows):
    """Chemical potentials mu2 = E2 - E1 (= -I), mu3 = E3 - E2 (= -A); addition energy (fundamental gap) mu3 - mu2.
    Orbital errors: eps_HOMO - mu2, eps_LUMO - mu3 (mHa); exact KS: eps_HOMO = mu2 by the HOMO condition."""
    hdr = '% generated by codes/13_trap_ip_gap_gw.py -- do not edit by hand\n'
    meths = ['HF', 'LDA', 'PBE', 'SCAN', 'PBE0']
    L = []
    for r in rows:
        mu2, mu3 = -r['I'], -r['A']
        cells = [name_of(r), f"{mu2:.6f}"] + [mha(r[f'{m}_HOMO'] - mu2) for m in meths] + \
                [f"{mu3:.6f}", mha(r['KS_LUMO'] - mu3)] + [mha(r[f'{m}_LUMO'] - mu3) for m in meths]
        L.append(' & '.join(cells) + r' \\')
    L.append(r'\midrule')
    L.append(r'\textit{MAE (mHa)} & & ' + ' & '.join(
        f"{1e3 * np.nanmean([abs(r[f'{m}_HOMO'] + r['I']) for r in rows]):.1f}" for m in meths) + ' & & ' +
        f"{1e3 * np.nanmean([abs(r['KS_LUMO'] + r['A']) for r in rows]):.1f} & " + ' & '.join(
        f"{1e3 * np.nanmean([abs(r[f'{m}_LUMO'] + r['A']) for r in rows]):.1f}" for m in meths) + r' \\')
    open(os.path.join(RES, 'tab_trap_t7_ip_rows.tex'), 'w').write(hdr + '\n'.join(L) + '\n')
    L = []
    for r in rows:
        cells = [name_of(r), f"{-r['I']:.6f}", f"{-r['A']:.6f}", f"{r['gap_fund']:.6f}", f"{r['KS_gap']:.6f}",
                 f"{r['Dxc']:.6f}", f"{100 * r['Dxc'] / r['gap_fund']:.1f}"] + \
                [f"{r[f'{m}_LUMO'] - r[f'{m}_HOMO']:.6f}" if np.isfinite(r[f'{m}_LUMO']) else '--' for m in meths]
        L.append(' & '.join(cells) + r' \\')
    open(os.path.join(RES, 'tab_trap_t7_gap_rows.tex'), 'w').write(hdr + '\n'.join(L) + '\n')
    L = []
    starts = ['HF', 'PBE', 'KS-exact']
    for r in rows:
        cells = [name_of(r), f"{-r['I']:.6f}"] + [mha(r.get(f'GW@{s}_HOMO', np.nan) + r['I']) for s in starts] + \
                [f"{-r['A']:.6f}"] + [mha(r.get(f'GW@{s}_LUMO', np.nan) + r['A']) for s in starts]
        L.append(' & '.join(cells) + r' \\')
    L.append(r'\midrule')
    L.append(r'\textit{MAE (mHa)} & & ' + ' & '.join(
        f"{1e3 * np.nanmean([abs(r.get(f'GW@{s}_HOMO', np.nan) + r['I']) for r in rows]):.1f}" for s in starts) +
        ' & & ' + ' & '.join(
        f"{1e3 * np.nanmean([abs(r.get(f'GW@{s}_LUMO', np.nan) + r['A']) for r in rows]):.1f}" for s in starts) +
        r' \\')
    open(os.path.join(RES, 'tab_trap_t7_gw_rows.tex'), 'w').write(hdr + '\n'.join(L) + '\n')


def write_eq8_tables(rows):
    """aditi Table 2 analogue: eq. (8) decomposition of the G0W0 HOMO and LUMO for every starting point (Ha).
    Reference: mu2 = E2 - E1 (HOMO) and mu3 = E3 - E2 (LUMO), same-basis FCI."""
    hdr = '% generated by codes/13_trap_ip_gap_gw.py -- do not edit by hand\n'
    for o, ref, fname in (('H', lambda r: -r['I'], 'tab_trap_t8_homo_rows.tex'),
                          ('L', lambda r: -r['A'], 'tab_trap_t8_lumo_rows.tex')):
        L = []
        for r in rows:
            L.append(r'\hline')
            L.append(r'\multicolumn{5}{c}{' + name_of(r) + '} & ' + f'{ref(r):.4f}' + r' \\')
            L.append(r'\hline')
            for st in EQ8_STARTS:
                if st == 'LDA':
                    L.append(r'\noalign{\vskip 3pt}')
                vals = [r.get(f'eq8_{st}_{o}_{q}', np.nan) for q in ('qp', 'eps', 'sig', 'vxc')]
                cells = [st] + [f'{v:.4f}' if np.isfinite(v) else '--' for v in vals] + \
                        [mha(vals[0] - ref(r)) if np.isfinite(vals[0]) else '--']
                L.append(' & '.join(cells) + r' \\')
        open(os.path.join(RES, fname), 'w').write(hdr + '\n'.join(L) + '\n')


def save_csv(rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(RES, 'trap_ip_gap.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.12g}' if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--lams', type=float, nargs='+', default=[1.0, 2.0, 5.0, 10.0])
    ap.add_argument('--nb', type=int, default=56, help='basis-size target (N = 1, 2, 3 in the same basis)')
    ap.add_argument('--degree', type=int, default=4)
    ap.add_argument('--zs', type=float, nargs='+', default=[1, 2, 3])
    ap.add_argument('--tables-only', action='store_true', help='rebuild the table bodies from trap_ip_gap.csv')
    args = ap.parse_args()
    if args.tables_only:
        rows = []
        with open(os.path.join(RES, 'trap_ip_gap.csv')) as fh:
            for r in csv.DictReader(fh):
                d = {}
                for k, v in r.items():
                    try:
                        d[k] = float(v)
                    except (TypeError, ValueError):
                        d[k] = v
                rows.append(d)
        write_tables(rows)
        write_eq8_tables(rows)
        log(f'tables rebuilt from csv ({len(rows)} systems)')
        return
    lams = [1.0, 2.0] if args.quick else args.lams
    rows = []
    for lam in lams:
        for fam in ('wire', 'sheet'):
            if lam == 1.0 and fam == 'sheet':
                continue
            trap = (lam ** 2, lam ** 2, 1.0) if fam == 'wire' else (1.0, 1.0, lam ** 2)
            f = fam if lam != 1.0 else 'iso'
            rows.append(run_ip(f'{f}_{lam:g}', f, lam, trap, 0.0, np.asarray(trap, float), args.nb, args.degree))
            save_csv(rows)
    if not args.quick:
        for fam in ('wire', 'sheet'):
            lam = 2.0
            trap = (lam ** 2, lam ** 2, 1.0) if fam == 'wire' else (1.0, 1.0, lam ** 2)
            for Z in args.zs:
                zeta = optimise_zeta(trap, Z)
                wb = np.maximum(np.asarray(trap, float), zeta)
                rows.append(run_ip(f'{fam}_{lam:g}_Z{Z:g}', fam, lam, trap, float(Z), wb, args.nb, args.degree))
                save_csv(rows)
    save_csv(rows)
    write_tables(rows)
    write_eq8_tables(rows)
    open(os.path.join(RES, 'trap_ip_gap_log.txt'), 'w').write('\n'.join(LOG) + '\n')
    log('done')


if __name__ == '__main__':
    main()
