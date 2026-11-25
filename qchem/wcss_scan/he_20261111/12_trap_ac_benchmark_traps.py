#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal (PySCF/libxc environment; numba optional but ~50x faster ERIs):
#
#     cd codes
#     ~/miniconda3/envs/anigto/bin/python 12_trap_ac_benchmark.py            # wire/sheet + He series
#     ~/miniconda3/envs/anigto/bin/python 12_trap_ac_benchmark.py --quick    # lam = 1, 2 only, no He series
#
# Needs: numpy, scipy, pyscf (libxc), [numba].  Results -> codes/results/
# ----------------------------------------------------------------------
r"""Adiabatic-connection benchmark of correlated two-electron wires and sheets (aniGTO tables I-VI analogue).

STANDALONE SCRIPT (SCAN_1D_2D/codes): every function it uses is defined in this file; only third-party imports.

Systems
  wire   trap omega = (lam^2, lam^2, 1),  sheet  trap omega = (1, 1, lam^2),  lam = 1, 2, 5, 10
         (the same trap families whose noninteracting Gaussian density is the fixed density of scripts 01-11)
  He-like ions  -Z/r + the wire/sheet trap at lam = 2 and the isotropic trap (lam = 1), Z = 1 ... 10

Basis
  aniGTO 'dot' functions: Cartesian Gaussians x^a y^b z^c exp(-sum_k wb_k x_k^2/2) on one centre.  For a common
  exponent per direction their span is exactly that of the harmonic-oscillator (Hermite) functions
  chi_n = prod_k N H_{n_k}(sqrt(wb_k) x_k) exp(-wb_k x_k^2/2), which are used here in orthonormal form.
  The aniGTO [spdfg] basis is the total-degree set n_x+n_y+n_z <= 4 (35 functions); this script uses the union of
  that set with the anisotropic energy cut  sum_k n_k wb_k <= ECUT * min(wb)  (more quanta along soft axes).
  wb = trap frequencies for the traps; for the ions wb_k = max(omega_k, zeta) with zeta optimised variationally (HF).

Integrals (exact up to the u quadrature, < 1e-11 Ha; SI Section S1 of the DC-SCAN paper, Boys-type form)
  1/r12 = (2/sqrt(pi)) int_0^inf du exp(-u^2 r12^2);  each Cartesian factor
     I_k(u) = int int chi_p chi_q(x1) chi_r chi_s(x2) exp(-u^2 (x1-x2)^2) dx1 dx2
  is evaluated EXACTLY by Gauss-Hermite quadrature after rotating to X = (x1+x2)/sqrt2, xi = (x1-x2)/sqrt2;
  the u integral uses Gauss-Legendre in ln u (300 nodes).  -Z/r analogously.  Kinetic and trap matrices analytic.
  Validated: aniGTO [spdfg] HF and FCI energies of the traps (0.1,0.1,0.5), (0.5,0.5,0.5), (1,1,1) reproduced to
  1e-10 Ha; (phi phi|phi phi) equals the closed-form wire/sheet exchange references of scripts 01/02.

Exact reference (pure traps)
  E = sum(omega)/2 (centre of mass) + E_rel, H_rel = HO(omega) + (1/sqrt2)/|s| in s = (r1-r2)/sqrt2, solved in the
  even tensor HO basis n_k <= N (N = 20, 30, 40, matrix-free Lanczos) and extrapolated in N^-p (p fitted).
  Check: Hooke omega = 1/2 gives 2.0000 (exact 2) to < 2e-6 Ha.

Adiabatic-connection ingredients (same definitions as aniGTO anigto_acm.py / star_fixed.py), for the HF and FCI
densities of each system
  W0 = E_x[rho] = -J/2 (N = 2 singlet), W0' = 2 E_c^GL2 (KS orbitals from the exact N = 2 inversion
  phi = sqrt(rho/2)), W1 = V_ee[Psi_1^rho] - J (FCI 2-RDM; Lieb maximisation at lambda = 1 for the HF density),
  W_inf, W_inf' = radialized N = 2 SCE (aniGTO convention: spherical average of the density),
  AC1, AC6, KHAN, SPL, genISI2 (formulas copied verbatim from aniGTO star_fixed.py), T_s = T_vW (exact, N = 2).
  DFA W1 = E_x^DFA + E_c^DFA on the same density (KS tau = tau_W, i.e. alpha = 0): BLYP (aniGTO mix), PBE,
  r2SCAN, SCAN, DC-SCAN (exchange of the DC-SCAN note with SCAN correlation), bounded fits
  W1 ~ beta W0 + (1-beta) E_x^DFA + gamma E_c^DFA (beta in [0,1], gamma in [0,3]) to W1 (Table II) and to W_inf
  (Table IV).

Outputs (codes/results/)
  trap_ac_results.csv           one row per system x density, all ingredients
  trap_exact_reference.csv      exact energies, extrapolation data, same-basis FCI basis error
  trap_basis_convergence.csv    FCI/HF vs ECUT at selected lam
  tab_trap_t1_rows.tex ... tab_trap_t6_rows.tex   table bodies (T1 ingredients+ACM, T2 DFA W1, T3 AC per W1
                                model, T4 SCE-anchored fits, T5 SCE-anchored AC, T6 He series)
  trap_ac_log.txt
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import integrate
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('SCAN_RESULTS', os.path.join(HERE, 'results'))     # cluster jobs: one directory per job
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


# ==============================================================================================================
# F. Radialized N = 2 SCE (copied from aniGTO anigto_acm.py, w1_SCE construction)
# ==============================================================================================================
_trapz = getattr(np, 'trapezoid', None) or np.trapz      # numpy >= 2 / older numpy


def _fibonacci_directions(n):
    n = max(8, int(n))
    i = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack((xy * np.cos(theta), xy * np.sin(theta), z))


def _compute_T_ZP_and_Winf_N2(rho_1d, r_grid):
    """Return (Wprime_inf, Vee_inf, number of valid radial points)."""
    dr = float(r_grid[1] - r_grid[0])
    rho_1d = np.maximum(np.asarray(rho_1d, dtype=float), 0.0)
    Ne = np.clip(np.cumsum(4.0 * np.pi * r_grid ** 2 * rho_1d) * dr, 0.0, 2.0)
    Ne_u, idx = np.unique(Ne, return_index=True)
    r_u = r_grid[idx]
    if Ne_u.size == 0 or Ne_u[-1] < 1.0:
        return np.nan, np.nan, 0
    if Ne_u[0] > 0.0:
        Ne_u = np.r_[0.0, Ne_u]
        r_u = np.r_[r_grid[0], r_u]
    i_half = int(np.searchsorted(Ne, 1.0))
    if i_half < 2:
        return np.nan, np.nan, 0
    Wp_sum = 0.0
    Vee_sum = 0.0
    n_valid = 0
    for i in range(i_half):
        r = float(r_grid[i])
        rho_r = float(rho_1d[i])
        if rho_r < 1.0e-20 or r < 1.0e-12:
            continue
        f_r = float(np.interp(2.0 - float(Ne[i]), Ne_u, r_u, left=r_grid[0], right=r_grid[-1]))
        d = r + f_r
        if d < 1.0e-12 or f_r < 1.0e-12:
            continue
        w_pt = 4.0 * np.pi * r * r * rho_r * dr
        Vee_sum += w_pt / d
        lb = (r * r + f_r * f_r) / (r * f_r * d ** 3)
        if lb <= 0.0:
            continue
        wb_ = np.sqrt(lb)
        rho_f = float(np.interp(f_r, r_grid, rho_1d, left=0.0, right=0.0))
        if rho_f < 1.0e-30:
            continue
        p = (r / f_r) ** 2 * rho_r / rho_f
        if p < 1.0e-12:
            continue
        ls = 2.0 * (p * p + 1.0) / (p * d ** 3)
        Wp_sum += w_pt * (2.0 * wb_ + np.sqrt(max(ls, 0.0)))
        n_valid += 1
    return 0.25 * Wp_sum, Vee_sum, n_valid


def spherical_average(wb, idx, D, r, ntheta=96, nphi=2):
    """rho_sph(r) = (1/4pi) int rho(r, Omega) dOmega; Gauss-Legendre in theta (clusters at the poles, where a wire's
    density sits) times uniform phi; D2h symmetry -> theta in [0, pi/2], phi in [0, pi/2]."""
    t, wt = np.polynomial.legendre.leggauss(ntheta)
    th = 0.25 * np.pi * (t + 1.0)
    wth = 0.25 * np.pi * wt * np.sin(th)
    ph = (np.arange(nphi) + 0.5) * (0.5 * np.pi / nphi)
    rho = np.zeros(len(r))
    for a, wa in zip(th, wth):
        for p in ph:
            d = np.array([np.sin(a) * np.cos(p), np.sin(a) * np.sin(p), np.cos(a)])
            c = ao_points(wb, idx, np.outer(r, d))
            rho += wa * np.einsum('pg,pq,qg->g', c, D, c)
    return np.maximum(rho / nphi, 0.0)          # sum wa over [0,pi/2] = 1 -> average over the sphere


def _cumsimpson(y, x):
    """Cumulative Simpson integral (scipy >= 1.12 cumulative_simpson; older scipy: piecewise-quadratic fallback)."""
    try:
        from scipy.integrate import cumulative_simpson
        return cumulative_simpson(y, x=x)
    except ImportError:
        out = np.empty(len(x) - 1)
        for i in range(len(x) - 1):                     # quadratic through (i-1, i, i+1) or (i, i+1, i+2)
            j = i - 1 if i > 0 else 0
            j = min(j, len(x) - 3)
            xs, ys = x[j:j + 3], y[j:j + 3]
            c = np.polyfit(xs - xs[0], ys, 2)
            P = np.polyint(c)
            out[i] = np.polyval(P, x[i + 1] - xs[0]) - np.polyval(P, x[i] - xs[0])
        return np.cumsum(out)


def sce_n2_radial(r, rho, nq=400):
    """N = 2 radial SCE on a spherical density, as a quadrature in the electron number q = N_e(r) in (0, 1):
    co-motion f(q) = N_out^-1(q) with N_out(r) = int_r^inf 4 pi r'^2 rho (no cancellation in the tail),
    V_ee^SCE = int_0^1 dq / (r + f),  W'_inf = (1/4) int_0^1 dq [2 sqrt((r^2+f^2)/(r f d^3)) + sqrt(2 (p^2+1)/(p d^3))],
    d = r + f, p = (r/f)^2 rho(r)/rho(f)  (same integrand as aniGTO's w1_SCE sum).  q = s^6 removes the q^(-1/6) end."""
    from scipy.interpolate import PchipInterpolator
    dens = 4.0 * np.pi * r * r * rho
    Nin = np.concatenate([[0.0], _cumsimpson(dens, r)])
    Nout = np.concatenate([[0.0], _cumsimpson(dens[::-1], -r[::-1])])[::-1]
    norm = Nin[-1]
    Nin, Nout = Nin * 2.0 / norm, Nout * 2.0 / norm
    keep_i = np.r_[True, np.diff(Nin) > 0]
    keep_o = np.r_[np.diff(Nout) < 0, True]
    rin = PchipInterpolator(Nin[keep_i], r[keep_i])
    rout = PchipInterpolator(Nout[keep_o][::-1], r[keep_o][::-1])
    lrho = PchipInterpolator(r, np.log(np.maximum(rho, 1e-300)))
    s, ws = np.polynomial.legendre.leggauss(nq)
    s = 0.5 * (s + 1.0)
    ws = 0.5 * ws
    q = s ** 6
    wq = ws * 6.0 * s ** 5
    qmin = max(Nin[keep_i][1], Nout[keep_o][-2])
    q = np.maximum(q, qmin)
    rr, ff = rin(q), rout(q)
    d = rr + ff
    V = float(np.sum(wq / d))
    p = (rr / ff) ** 2 * np.exp(lrho(rr) - lrho(ff))
    wp = 2.0 * np.sqrt((rr * rr + ff * ff) / (rr * ff * d ** 3)) + np.sqrt(2.0 * (p * p + 1.0) / (p * d ** 3))
    return V, 0.25 * float(np.sum(wq * wp)), norm


def radialized_sce(wb, idx, D, J, nrad=3001):
    """Radialized N = 2 SCE (aniGTO convention: spherical average of the anisotropic density), converged quadrature.
    Also returns aniGTO's original radial sum (linspace(1e-6, 30, 2001), Fibonacci 256) for comparison."""
    rmax = np.sqrt(80.0 / min(wb))
    x = np.linspace(0.0, 1.0, nrad)
    r = 1e-8 + rmax * x * x                       # dense near the origin
    rho = spherical_average(wb, idx, D, r)
    V, Wp, norm = sce_n2_radial(r, rho)
    # aniGTO legacy (uniform radial grid on [1e-6, 30] in the trap's natural unit, Fibonacci directions)
    rl = np.linspace(1.0e-6, 30.0 / np.sqrt(min(wb)), 2001)
    rho_l = np.zeros(len(rl))
    for dvec in _fibonacci_directions(256):
        c = ao_points(wb, idx, np.outer(rl, dvec))
        rho_l += np.einsum('pg,pq,qg->g', c, D, c)
    rho_l = np.maximum(rho_l / 256, 0.0)
    rho_l *= 2.0 / float(_trapz(4.0 * np.pi * rl * rl * rho_l, rl))
    Wp_l, V_l, _ = _compute_T_ZP_and_Winf_N2(rho_l, rl)
    return dict(W_inf=V - J, Wp_inf=Wp, radial_norm=norm, W_inf_legacy=V_l - J, Wp_inf_legacy=Wp_l)


# ==============================================================================================================
# G. Density functionals (libxc via PySCF) and DC-SCAN exchange (Appendix A of the DC-SCAN manuscript)
# ==============================================================================================================
CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
KF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
A2D = 0.6532258969913366
A1D = 1.0 / 3.0


def f2d(s):
    return A2D / (1.0 + s * s) ** 0.25


def f1d(s):
    return A1D / (1.0 + s * s)


def geometry_weights(D):
    l2 = 0.5 * (D - 2.0) * (D - 3.0)
    l1 = (D - 1.0) * (3.0 - D)
    l3 = 0.5 * (D - 1.0) * (D - 2.0)
    den = l2 * l2 + l1 * l1 + l3 * l3
    return l2 * l2 / den, l1 * l1 / den, l3 * l3 / den


def omega_switch(s, Z, variant='full'):
    s8 = s ** 8
    g = s8 / (1.0 + s8)
    g = np.where(np.isfinite(s8), g, 1.0)
    if variant == 'min':
        return g
    z3 = Z ** 3
    return z3 / (1.0 + z3) * g


def fx_dcscan(s, D, Z, fx_base, variant='full'):
    w2, w1, _ = geometry_weights(D)
    om = omega_switch(s, Z, variant)
    return (1.0 - om * (w2 + w1)) * fx_base + om * (w2 * f2d(s) + w1 * f1d(s))


XC_CODES = {  # name: (exchange code, correlation code) for PySCF/libxc
    'LDA': ('LDA_X', 'LDA_C_PW'),
    'PBE': ('GGA_X_PBE', 'GGA_C_PBE'),
    'BLYP': ('GGA_X_B88', 'GGA_C_LYP'),
    'r2SCAN': ('MGGA_X_R2SCAN', 'MGGA_C_R2SCAN'),
    'SCAN': ('MGGA_X_SCAN', 'MGGA_C_SCAN'),
}


def dfa_energies(grid, D):
    """E_x and E_c of every DFA on the density of D, with the exact N = 2 KS tau = tau_W (alpha = 0);
    DC-SCAN exchange from the density Hessian (descriptors D, Z of the DC-SCAN note)."""
    from pyscf.dft import numint
    ni = numint.NumInt()
    rho, grad, hess, lap, _ = grid.fields(D)
    keep = rho > 1e-30
    rho, grad, hess, lap, w = rho[keep], grad[:, keep], hess[:, :, keep], lap[keep], grid.W[keep]
    g2 = np.sum(grad * grad, axis=0)
    tau_w = g2 / (8.0 * rho)
    inp = np.vstack((rho, grad, lap, tau_w))
    out, eps = {}, {}
    for name, (xc_x, xc_c) in XC_CODES.items():
        for part, code in (('x', xc_x), ('c', xc_c)):
            if code.startswith('LDA'):
                x = inp[:1]
            elif code.startswith('GGA'):
                x = inp[:4]
            else:
                x = inp
            key = code + ',' if part == 'x' else ',' + code
            e = np.asarray(ni.eval_xc(key, x, spin=0, deriv=0)[0])
            eps[(name, part)] = e
            out[f'E{part}_{name}'] = float(np.sum(w * rho * e))
    eps_lda = eps[('LDA', 'x')]
    ok = eps_lda != 0.0
    fx_scan = np.where(ok, eps[('SCAN', 'x')] / np.where(ok, eps_lda, 1.0), 1.0)
    C = -hess / (2.0 * rho) + grad[:, None, :] * grad[None, :, :] / (2.0 * rho * rho)
    C2 = np.einsum('ijg,jkg->ikg', C, C)
    trc2 = np.einsum('iig->g', C2)
    gc2g = np.einsum('ig,ijg,jg->g', grad, C2, grad)
    with np.errstate(invalid='ignore', divide='ignore'):
        Dd = np.where(gc2g > 1e-300 * np.maximum(g2 * trc2, 1e-300), g2 * trc2 / gc2g, 3.0)
    Dd = np.clip(np.nan_to_num(Dd, nan=3.0), 1.0, 3.0)
    kf = KF * np.cbrt(rho)
    s = np.sqrt(g2) / (2.0 * kf * rho)
    Zd = np.sqrt(np.maximum(trc2, 0.0)) / kf ** 2
    out['Ex_DC-SCAN'] = float(np.sum(w * rho * eps_lda * fx_dcscan(s, Dd, Zd, fx_scan, 'full')))
    out['Ec_DC-SCAN'] = out['Ec_SCAN']
    out['N_grid'] = float(np.sum(w * rho))
    return out


DFA_W1 = ('BLYP', 'PBE', 'SCAN', 'DC-SCAN')    # r2SCAN = SCAN exactly for N = 2 (tau_s = tau_W, alpha = 0); kept in the csv
BLYP_BETA, BLYP_GAMMA = 0.5, 1.72375          # aniGTO raw BLYP W1 = 0.5 W0 + 0.5 Ex^B88 + 1.72375 Ec^LYP


def raw_w1(r, model):
    if model == 'BLYP':
        return BLYP_BETA * r['W0'] + (1 - BLYP_BETA) * r['Ex_BLYP'] + BLYP_GAMMA * r['Ec_BLYP']
    return r[f'Ex_{model}'] + r[f'Ec_{model}']


def opt_bg(target, ex_hf, ex_dfa, ec_dfa, tol=1e-8):
    """Bounded fit beta*Ex(HF) + (1-beta)*Ex(DFA) + gamma*Ec(DFA) to a target (aniGTO star_fixed.py::_opt_bg)."""
    vals = (target, ex_hf, ex_dfa, ec_dfa)
    if any(not np.isfinite(v) for v in vals):
        return np.nan, np.nan, np.nan, np.nan

    def obj(p):
        return abs(target - (p[0] * ex_hf + (1.0 - p[0]) * ex_dfa + p[1] * ec_dfa))
    best = None
    for b0, g0 in [(0.5, 1.72375), (0.0, 0.0), (1.0, 3.0), (0.25, 1.5)]:
        res = minimize(obj, [b0, g0], bounds=[(0.0, 1.0), (0.0, 3.0)], method='L-BFGS-B',
                       options={'ftol': tol, 'gtol': tol, 'maxiter': 2000})
        if best is None or res.fun < best.fun:
            best = res
    beta, gamma = (float(x) for x in best.x)
    w1 = float(beta * ex_hf + (1.0 - beta) * ex_dfa + gamma * ec_dfa)
    return beta, gamma, w1, float(abs(target - w1))


# ==============================================================================================================
# H. Adiabatic-connection models (copied verbatim from aniGTO codes/table1/source/star_fixed.py)
# ==============================================================================================================
def _ac1_ac6(W0, W1, gl2, base):
    """
    AC1 / AC6 models on λ ∈ [0,1], with b = W'(0) = 2·GL2:

        AC1:  W(λ) = W0 + bλ/(1+cλ)        c = b/(W1-W0) - 1
        AC6:  W(λ) = W0 + bλ·exp(-cλ)      c = ln(b/(W1-W0))

    Each model fixes its own c from W(1) = W1, and
    E_xc = ∫0^1 W(λ) dλ  (NOT W(λ=1), which would just return W1).

        AC1:  E_xc = W0 + (b/c)·[1 - ln(1+c)/c]
        AC6:  E_xc = W0 + b·[1 - (1+c)·e^(-c)]/c²

    Both need (W1-W0)/b > 0, i.e. W1 below W0 and GL2 < 0.
    Returns E_ac1, E_ac6, c_ac1, Exc_ac1, Exc_ac6.
    """
    nan5 = (np.nan,) * 5
    if np.isnan(W0) or np.isnan(W1) or np.isnan(gl2):
        return nan5

    b = 2.0 * gl2
    if abs(b) < 1.0e-12:
        return base + W0, base + W0, 0.0, W0, W0

    r = (W1 - W0) / b
    if r <= 0.0:
        return nan5

    # AC1 (Padé [1,1])
    c1 = 1.0 / r - 1.0
    if abs(c1) < 1.0e-6:
        f1 = 0.5 - c1 / 3.0 + c1**2 / 4.0
    else:
        f1 = (1.0 - np.log1p(c1) / c1) / c1
    exc1 = W0 + b * f1

    # AC6 (exponential damping)
    c6 = -np.log(r)
    if abs(c6) < 1.0e-6:
        f6 = 0.5 - c6 / 3.0 + c6**2 / 8.0
    else:
        f6 = (1.0 - (1.0 + c6) * np.exp(-c6)) / c6**2
    exc6 = W0 + b * f6

    return float(base + exc1), float(base + exc6), float(c1), float(exc1), float(exc6)


def _khan(W0, W1, gl2, base):
    """
    Khan model on λ ∈ [0,1]:  W(λ) = a + b·sqrt(1+λ)/(1+cλ)
        W(0) = a + b = W0,  W'(0) = b(1/2 - c) = 2·GL2,  W(1) = a + √2·b/(1+c) = W1
    E_xc = ∫0^1 W(λ) dλ  (the original returned W(1), i.e. just W1).
    """
    if np.isnan(W0) or np.isnan(W1) or np.isnan(gl2): return np.nan
    if abs(gl2) < 1e-10: return base+W0
    alpha = (W1-W0)/(2.0*gl2)
    if abs(alpha) < 1e-10: return base+W0
    # c solves α c² + (α/2 - 1) c + √2 - 1 - α/2 = 0 (Khan, arXiv:2503.22661
    # Eq. 16); take his root first, the other root only if it violates a ≤ W1.
    # (The former fallback c → -c did not satisfy W(1) = W1.)
    sq = np.sqrt(max(9*alpha**2 - 16*np.sqrt(2)*alpha + 12*alpha + 4, 0.0))
    for c in ((2.0-alpha+sq)/(4.0*alpha), (2.0-alpha-sq)/(4.0*alpha)):
        den = 0.5-c
        if abs(den) < 1e-10 or c <= -1.0:  # degenerate, or pole inside [0,1]
            continue
        b_k = 2.0*gl2/den; a_k = W0-b_k
        if a_k <= W1:
            break
    else:
        return np.nan
    if abs(b_k) < 1e-10: return base+W0
    integral, _ = integrate.quad(lambda lam: np.sqrt(1.0+lam)/(1.0+c*lam), 0.0, 1.0,
                                 epsabs=1e-12, epsrel=1e-12)
    return float(base + a_k + b_k*integral)


def _spl_corr(W0, gl2, W_inf):
    if abs(gl2) < 1e-12 or abs(W_inf-W0) < 1e-12: return 0.0
    chi = 2.0*gl2/(W_inf-W0)
    return (W0-W_inf)*((np.sqrt(1.0+2.0*chi)-1.0-chi)/chi)


def _gisi2_corr(W0, gl2, W_inf, Wp_inf):
    if abs(gl2) < 1e-12 or abs(Wp_inf) < 1e-30: return 0.0
    def _integrand(lam):
        d=3.5; b_c=(W0-W_inf)*(1.0+d); c_c=b_c**2/(4.0*Wp_inf**2)
        l1,l2=10.65,3.6; w0p=2.0*gl2
        sq = np.sqrt(max(1.0+c_c*lam, 0.0))
        uISI=(W_inf+b_c*(2.0+c_c*lam+2.0*d*sq)/(2.0*sq*(d+sq)**2))
        p=w0p/W0 if abs(W0)>1e-30 else 0.0; pl=p*lam
        a1=W0; a2=(-(uISI-W0)/pl) if abs(pl)>1e-30 else 0.0
        r1=l1*(W0/W_inf)**3 if abs(W_inf)>1e-30 else 0.0
        r2=l2*(W0/W_inf)**3 if abs(W_inf)>1e-30 else 0.0
        Wa=(a1*pl)/(1.0+r1*pl)**3; Wb=(a2*pl)/(1.0+r2*pl)**3
        return uISI+Wa+Wb-W0
    try:
        val,_ = integrate.quad(_integrand, 0.0, 1.0, limit=200, epsabs=1e-10, epsrel=1e-10)
        return float(val)
    except Exception: return 0.0


def ac_models(r, w1, w_inf):
    """AC1, AC6, KHAN with a given W1; SPL, genISI2 with a given W_inf (W_inf' fixed); total energies."""
    base = r['T_s'] + r['E_ext'] + r['J']
    e1, e6, *_ = _ac1_ac6(r['W0'], w1, r['gl2'], base)
    ek = _khan(r['W0'], w1, r['gl2'], base)
    spl = base + r['W0'] + _spl_corr(r['W0'], r['gl2'], w_inf)
    gisi = base + r['W0'] + _gisi2_corr(r['W0'], r['gl2'], w_inf, r['Wp_inf'])
    return [float(e1), float(e6), float(ek), float(spl), float(gisi)]


AC_NAMES = ('AC1', 'AC6', 'KHAN', 'SPL', 'genISI2')


# ==============================================================================================================
# I. One system: HF, FCI, ingredients for both densities
# ==============================================================================================================
def run_system(label, family, lam, trap, Z, wb, nb_target, degree, exact=None):
    idx, ecut = basis_for_target(wb, nb_target, degree)
    t = time.time()
    ints = build(trap, wb, idx, Z=Z)
    h, eri, T = ints['h'], ints['eri'], ints['T']
    E_hf, e_hf, C_hf, D_hf = rhf(h, eri)
    E_fci, C_fci, D_fci, Vee_fci = fci_singlet2(h, eri)
    log(f'{label}: nb={len(idx)}  E_HF={E_hf:.10f}  E_FCI={E_fci:.10f}  Ec={1e3 * (E_fci - E_hf):.4f} mHa  '
        f'({time.time() - t:.1f}s)')
    grid = Grid(wb, idx)
    gcyl = Grid(wb, idx, 'cylinder', 300)
    S3 = triple_tensor(wb, idx)
    Vext = ints['Vtrap'] + ints['Vnuc']
    rows = []
    for dens, D_tar in (('HF', D_hf), ('FCI', D_fci)):
        J = 0.5 * float(np.sum(D_tar * jk(eri, D_tar)[0]))
        W0 = -0.5 * J
        ks = ks_exact_2e(grid, T, D_tar)
        drho_ks = rho_l2_diff(grid, ks['D'], D_tar)
        gl2 = gl2_energy(eri, ks['C'], ks['eps'])
        if dens == 'FCI':
            Vee, src, drho_l, nit = Vee_fci, 'FCI 2-RDM', 0.0, 0
        else:
            lb = lieb_lambda1(h, eri, S3, D_tar)
            Vee, src, drho_l, nit = lb['Vee'], 'Lieb lambda=1', rho_l2_diff(grid, lb['D'], D_tar), lb['nit']
        W1 = Vee - J
        sce = radialized_sce(wb, idx, D_tar, J)
        xc = dfa_energies(gcyl, D_tar)
        E_ext = float(np.sum(Vext * D_tar))
        r = dict(system=label, family=family, lam=lam, Z=Z, omega='x'.join(f'{w:g}' for w in trap),
                 wb='x'.join(f'{w:.4g}' for w in wb), nb=len(idx), ecut=ecut, density=dens, E_HF=E_hf, E_FCI=E_fci,
                 E_exact=(exact if exact is not None else np.nan), E_ref=(E_hf if dens == 'HF' else E_fci),
                 J=J, W0=W0, gl2=gl2, W0p=2 * gl2, W1=W1, W1_source=src, lieb_drho=drho_l, lieb_nit=nit,
                 T_s=ks['Ts'], Ts_orb=ks['Ts_orb'], eps_H_check=float(ks['eps'][0]), ks_drho=drho_ks,
                 ks_gap=float(ks['eps'][1] - ks['eps'][0]), E_ext=E_ext, E_kin=float(np.sum(T * D_tar)),
                 W_inf=sce['W_inf'], Wp_inf=sce['Wp_inf'], sce_norm=sce['radial_norm'], **xc)
        r['E_target'] = E_fci
        acm = ac_models(r, W1, r['W_inf'])
        for k, v in zip(AC_NAMES, acm):
            r[f'E_{k}'] = v
        for m in DFA_W1:
            r[f'W1raw_{m}'] = raw_w1(r, m)
            b, g, w, err = opt_bg(W1, W0, r[f'Ex_{m}'], r[f'Ec_{m}'])
            r.update({f'beta_{m}': b, f'gamma_{m}': g, f'W1fit_{m}': w, f'W1fit_err_{m}': err})
            b, g, w, err = opt_bg(r['W_inf'], W0, r[f'Ex_{m}'], r[f'Ec_{m}'])
            r.update({f'betaSCE_{m}': b, f'gammaSCE_{m}': g, f'W1sce_{m}': w, f'W1sce_err_{m}': err})
        log(f'   {dens}: J={J:.8f} W0={W0:.8f} W0p={2 * gl2:.8f} W1={W1:.8f} ({src}, |drho|={drho_l:.1e}) '
            f'Winf={sce["W_inf"]:.8f} Wpinf={sce["Wp_inf"]:.8f} eps_H(check)={ks["eps"][0]:.1e} '
            f'|drho_KS|={drho_ks:.1e} N={xc["N_grid"]:.8f}')
        log('      ' + '  '.join(f'{k}={1e3 * (v - E_fci):+.3f}' for k, v in zip(AC_NAMES, acm)) + '  (mHa vs FCI)')
        rows.append(r)
    return rows


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
# J. Tables
# ==============================================================================================================
def fmt(v, d=6):
    if v is None or not np.isfinite(v):
        return '--'
    s = f'{v:.{d}f}'
    return s.replace('-', '$-$') if s.startswith('-') else s


def fmte(v, d=2):
    if v is None or not np.isfinite(v):
        return '--'
    return f'$+${v:.{d}f}' if v >= 0 else f'$-${abs(v):.{d}f}'


def sysname(r):
    if r['Z']:
        return f"{r['family']} $\\lambda$={r['lam']:g}, $Z$={r['Z']:g}"
    return f"{r['family']} $\\lambda$={r['lam']:g}"


def write_tables(rows, prefix='tab_trap'):
    hdr = '% generated by codes/12_trap_ac_benchmark.py -- do not edit by hand\n'
    trap_rows = [r for r in rows if not r['Z']]
    he_rows = [r for r in rows if r['Z']]

    def t1(rset, fname):
        lines, errs = [], [[] for _ in AC_NAMES]
        rel = [[] for _ in AC_NAMES]
        for r in rset:
            e = [1e3 * (r[f'E_{k}'] - r['E_FCI']) for k in AC_NAMES]
            for j in range(5):
                errs[j].append(e[j])
                rel[j].append(abs(r[f'E_{AC_NAMES[j]}'] - r['E_FCI']) / abs(r['E_FCI']) * 100)
            cells = [sysname(r), r['density'], fmt(r['E_ref']), fmt(r['W0']), fmt(r['W0p']), fmt(r['W1']),
                     fmt(r['W_inf']), fmt(r['Wp_inf'])] + [fmt(r[f'E_{k}']) for k in AC_NAMES] + [fmte(x) for x in e]
            lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\midrule')
        n = len(rset)
        lines.append(f'\\textit{{MAE (mHa; N={n})}}' + ' &' * 12 + ' & '.join(
            f'{np.nanmean(np.abs(e)):.2f}' for e in errs) + r' \\')
        lines.append(r'\textit{MARE (\%)}' + ' &' * 12 + ' & '.join(f'{np.nanmean(x):.3f}' for x in rel) + r' \\')
        with open(os.path.join(RES, fname), 'w') as fh:
            fh.write(hdr + '\n'.join(lines) + '\n')

    t1(trap_rows, f'{prefix}_t1_rows.tex')
    if he_rows:
        t1(he_rows, f'{prefix}_t6_rows.tex')

    # T2: DFA W1 (raw and fitted) vs exact W1
    lines, raw_e, fit_e = [], [], []
    for r in trap_rows:
        for m in DFA_W1:
            raw, fit = r[f'W1raw_{m}'], r[f'W1fit_{m}']
            raw_e.append(1e3 * (raw - r['W1']))
            fit_e.append(1e3 * (fit - r['W1']))
            lines.append(' & '.join([sysname(r), r['density'], m, fmt(raw), fmte(1e3 * (raw - r['W1'])),
                                     f"{r[f'beta_{m}']:.3f}", f"{r[f'gamma_{m}']:.3f}", fmt(fit),
                                     fmte(1e3 * (fit - r['W1']), 3), fmt(r['W1'])]) + r' \\')
    lines.append(r'\midrule')
    lines.append(f'\\textit{{MAE (mHa; N={len(raw_e)})}} & & & & {np.mean(np.abs(raw_e)):.2f} & & & & '
                 f'{np.mean(np.abs(fit_e)):.3f} & \\\\')
    with open(os.path.join(RES, f'{prefix}_t2_rows.tex'), 'w') as fh:
        fh.write(hdr + '\n'.join(lines) + '\n')

    # T3: AC energies per W1 model (deviation from the exact-W1 row; SPL/genISI2 use W1 as W_inf proxy)
    lines, errs = [], {m: [[] for _ in AC_NAMES] for m in DFA_W1}
    for r in trap_rows:
        ref = ac_models(r, r['W1'], r['W_inf'])
        for m in DFA_W1:
            for kind in ('raw', 'fit'):
                w1 = r[f'W1{kind}_{m}']
                e = ac_models(r, w1, w1)
                d = [1e3 * (x - y) for x, y in zip(e, ref)]
                for j in range(5):
                    errs[m][j].append(d[j])
                lines.append(' & '.join([sysname(r), r['density'], f'{m} {kind}', fmt(w1)] +
                                        [fmt(x) for x in e] + [fmte(x) for x in d]) + r' \\')
    lines.append(r'\midrule')
    for m in DFA_W1:
        lines.append(f'\\textit{{MAE {m} (mHa)}}' + ' &' * 9 + ' & '.join(
            f'{np.nanmean(np.abs(e)):.2f}' for e in errs[m]) + r' \\')
    with open(os.path.join(RES, f'{prefix}_t3_rows.tex'), 'w') as fh:
        fh.write(hdr + '\n'.join(lines) + '\n')

    # T4: SCE-anchored fits (W1-like ansatz fitted to W_inf) ; T5: AC energies with them
    l4, l5 = [], []
    for r in trap_rows:
        ref = ac_models(r, r['W1'], r['W_inf'])
        for m in DFA_W1:
            w = r[f'W1sce_{m}']
            l4.append(' & '.join([sysname(r), r['density'], m, fmt(r['W1']), fmt(r['W_inf']), fmt(r['Wp_inf']),
                                  f"{r[f'betaSCE_{m}']:.3f}", f"{r[f'gammaSCE_{m}']:.3f}", fmt(w),
                                  fmte(1e3 * r[f'W1sce_err_{m}'], 3)]) + r' \\')
            e = ac_models(r, w, r['W_inf'])
            d = [1e3 * (x - y) for x, y in zip(e, ref)]
            l5.append(' & '.join([sysname(r), r['density'], f'{m}@SCE', fmt(w)] + [fmt(x) for x in e] +
                                 [fmte(x) for x in d]) + r' \\')
    for name, ls in (('t4', l4), ('t5', l5)):
        with open(os.path.join(RES, f'{prefix}_{name}_rows.tex'), 'w') as fh:
            fh.write(hdr + '\n'.join(ls) + '\n')


def write_exact_table(exact_rows, conv_rows):
    """Exact energies (COM separation, extrapolated) and the same-basis FCI basis error used in Tables T1-T5."""
    L = ['% generated by codes/12_trap_ac_benchmark.py -- do not edit by hand']
    for e in exact_rows:
        cv = [c for c in conv_rows if c['family'] == e['family'] and float(c['lam']) == float(e['lam'])]
        cells = [f"{e['family']} $\\lambda$={float(e['lam']):g}", f"({e['omega'].replace('x', ', ')})",
                 f"{float(e['E_N40']):.6f}", f"{float(e['p']):.2f}", f"{float(e['E_exact']):.6f}",
                 str(int(float(e['nb']))), f"{float(e['E_FCI_basis']):.6f}", f"{float(e['basis_error_mHa']):.2f}"]
        cells += [f"{1e3 * (float(c['E_FCI']) - float(e['E_exact'])):.2f}" for c in sorted(cv, key=lambda c: int(float(c['nb'])))]
        L.append(' & '.join(cells) + r' \\')
    with open(os.path.join(RES, 'tab_trap_exact_rows.tex'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')


def read_csv_rows(path):
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
    return out


def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.12g}' if isinstance(v, float) else v) for k, v in r.items()})


# ==============================================================================================================
# K. Main
# ==============================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='lam = 1, 2 only; no He series; no convergence study')
    ap.add_argument('--lams', type=float, nargs='+', default=[1.0, 2.0, 5.0, 10.0])
    ap.add_argument('--nb', type=int, default=84, help='basis-size target (energy cut raised until reached)')
    ap.add_argument('--degree', type=int, default=4, help='aniGTO total-degree set always included ([spdfg] = 4)')
    ap.add_argument('--zs', type=float, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument('--he-lams', type=float, nargs='+', default=[1.0, 2.0])
    ap.add_argument('--no-he', action='store_true')
    ap.add_argument('--he-only', action='store_true', help='only the He-like ions (--zs, --he-lams, --he-fams)')
    ap.add_argument('--he-fams', nargs='+', default=['iso', 'wire', 'sheet'])
    ap.add_argument('--no-exact', action='store_true')
    ap.add_argument('--reuse-exact', action='store_true', help='read exact references from trap_exact_reference.csv')
    ap.add_argument('--tables-only', action='store_true', help='rebuild the table bodies from the csv files')
    args = ap.parse_args()
    if args.tables_only:
        import glob
        rows = read_csv_rows(os.path.join(RES, 'trap_ac_results.csv'))
        trap_files = sorted(glob.glob(os.path.join(RES, 'trap_runs', '*', 'trap_ac_results.csv')))
        if trap_files:                                   # traps from the cluster jobs (--lams ... --no-he) ...
            rows = []
            for f in trap_files:
                rows += read_csv_rows(f)
        rows = [r for r in rows if not r['Z']]
        for f in sorted(glob.glob(os.path.join(RES, 'he_runs', '*', 'trap_ac_results.csv'))):
            rows += read_csv_rows(f)                     # ... He-like ions from the cluster jobs (--he-only)
        rows.sort(key=lambda r: (r['Z'] != 0, str(r['family']), float(r['lam']), float(r['Z']), r['density']))
        write_csv(rows, os.path.join(RES, 'trap_ac_results_all.csv'))
        write_tables(rows)
        exs = sorted(glob.glob(os.path.join(RES, 'trap_runs', '*', 'trap_exact_reference.csv')))
        cvs = sorted(glob.glob(os.path.join(RES, 'trap_runs', '*', 'trap_basis_convergence.csv')))
        if not exs:
            exs, cvs = [os.path.join(RES, 'trap_exact_reference.csv')], [os.path.join(RES, 'trap_basis_convergence.csv')]
        exs, cvs = [f for f in exs if os.path.exists(f)], [f for f in cvs if os.path.exists(f)]
        if exs and cvs:
            ex_rows = sum((read_csv_rows(f) for f in exs), [])
            ex_rows.sort(key=lambda e: (float(e['lam']), str(e['family'])))
            write_exact_table(ex_rows, sum((read_csv_rows(f) for f in cvs), []))
        log('tables rebuilt from csv')
        return
    lams = [1.0, 2.0] if args.quick else args.lams
    if args.he_only:                                     # one cluster job = selected He-like ions only
        lams, args.no_exact = [], True

    systems = []
    for lam in lams:
        for fam in ('wire', 'sheet'):
            if lam == 1.0 and fam == 'sheet':
                continue                                     # identical to the wire at lam = 1 (isotropic)
            trap = (lam ** 2, lam ** 2, 1.0) if fam == 'wire' else (1.0, 1.0, lam ** 2)
            systems.append((fam if lam != 1.0 else 'iso', lam, trap))

    # exact references
    exact_rows = []
    exact = {}
    exf = os.path.join(RES, 'trap_exact_reference.csv')
    if args.reuse_exact and os.path.exists(exf):
        for e in read_csv_rows(exf):
            exact[(e['family'], float(e['lam']))] = float(e['E_exact'])
            exact_rows.append({k: e[k] for k in ('family', 'lam', 'omega', 'E_N20', 'E_N30', 'E_N40', 'p', 'E_exact')})
        log(f'exact references reused from {exf}')
    if not args.no_exact:
        for fam, lam, trap in systems:
            if (fam, lam) in exact:
                continue
            t = time.time()
            Einf, Es, p = exact_trap_energy(trap)
            exact[(fam, lam)] = Einf
            exact_rows.append(dict(family=fam, lam=lam, omega='x'.join(f'{w:g}' for w in trap), E_N20=Es[0],
                                   E_N30=Es[1], E_N40=Es[2], p=p, E_exact=Einf))
            log(f'exact {fam} lam={lam:g}: N=20,30,40 -> {Es[0]:.9f} {Es[1]:.9f} {Es[2]:.9f}  p={p:.2f}  '
                f'E_inf={Einf:.9f}  ({time.time() - t:.0f}s)')

    rows = []

    def checkpoint():
        write_csv(rows, os.path.join(RES, 'trap_ac_results.csv'))
        write_tables(rows)
    for fam, lam, trap in systems:
        rows += run_system(f'{fam}_{lam:g}', fam, lam, trap, 0.0, np.asarray(trap, float), args.nb, args.degree,
                           exact.get((fam, lam)))
        checkpoint()
    if exact_rows:
        for e in exact_rows:
            f = [r for r in rows if r['family'] == e['family'] and float(r['lam']) == float(e['lam'])
                 and r['density'] == 'FCI'][0]
            e.update(nb=f['nb'], E_FCI_basis=f['E_FCI'], basis_error_mHa=1e3 * (f['E_FCI'] - e['E_exact']))
        write_csv(exact_rows, os.path.join(RES, 'trap_exact_reference.csv'))

    # basis convergence (SI): FCI vs basis size for every trap with an exact reference
    conv = []
    if not (args.quick or args.he_only):
        for fam, lam, trap in systems:
            for nbt in (35, 56, 84, 120):
                idx, ec = basis_for_target(trap, nbt, args.degree)
                d = build(trap, trap, idx)
                E_hf = rhf(d['h'], d['eri'])[0]
                E_f = fci_singlet2(d['h'], d['eri'])[0]
                conv.append(dict(family=fam, lam=lam, ecut=ec, nb=len(idx), E_HF=E_hf, E_FCI=E_f,
                                 E_exact=exact.get((fam, lam), np.nan)))
                log(f'convergence {fam} lam={lam:g} ecut={ec:g} nb={len(idx)} E_FCI={E_f:.9f}')
        write_csv(conv, os.path.join(RES, 'trap_basis_convergence.csv'))
        write_exact_table(exact_rows, conv)

    # He-like ions in the traps
    if not (args.quick or args.no_he):
        for lam in args.he_lams:
            fams = ('iso',) if lam == 1.0 else ('wire', 'sheet')
            for fam in [f for f in fams if f in args.he_fams]:
                trap = (lam ** 2, lam ** 2, 1.0) if fam == 'wire' else (1.0, 1.0, lam ** 2)
                for Z in args.zs:
                    zeta = optimise_zeta(trap, Z)
                    wb = np.maximum(np.asarray(trap, float), zeta)
                    rows += run_system(f'{fam}_{lam:g}_Z{Z:g}', fam, lam, trap, float(Z), wb, args.nb,
                                       args.degree)
                    checkpoint()

    write_csv(rows, os.path.join(RES, 'trap_ac_results.csv'))
    write_tables(rows)
    if exact_rows and not args.quick:
        write_exact_table(exact_rows, conv)
    with open(os.path.join(RES, 'trap_ac_log.txt'), 'w') as fh:
        fh.write('\n'.join(LOG) + '\n')
    log('done')


if __name__ == '__main__':
    main()
