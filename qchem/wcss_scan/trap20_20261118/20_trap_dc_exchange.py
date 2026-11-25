#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal (PySCF/libxc environment; numba optional but ~50x faster ERIs):
#
#     cd codes
#     ~/miniconda3/envs/anigto/bin/python 20_trap_dc_exchange.py            # sheet + wire, lam = 1 ... 100
#     ~/miniconda3/envs/anigto/bin/python 20_trap_dc_exchange.py --quick    # lam = 1, 2
#
# Needs: numpy, scipy, pyscf (libxc), [numba].  Results -> codes/results/
# ----------------------------------------------------------------------
r"""Exact sheet and wire tests of SCAN and of the dimensional-crossover exchange functionals on CORRELATED
two-electron traps (the interacting counterpart of the fixed Gaussian densities of scripts 01/02).
STANDALONE SCRIPT (SCAN_1D_2D/codes): every function it uses is defined in this file; only third-party imports.

Systems (identical to script 12 and to the fixed-density families of scripts 01/02)
  sheet  trap omega = (1, 1, lam^2)        lam -> inf: quasi-2D sheet
  wire   trap omega = (lam^2, lam^2, 1)    lam -> inf: quasi-1D wire
  The noninteracting density of each trap is exactly the Gaussian sheet/wire of parameter lam of scripts 01/02.
Method (sections A and C copied verbatim from script 12)
  HO ('dot') basis = aniGTO [spdfg] united with the anisotropic energy cut, nb >= 84 (script-12 default); exact
  one- and two-electron integrals; RHF and two-electron FCI.  Exact references in this basis:
      E_x^ex = -1/4 tr(D K[D])  (RHF density),     E_c^ex = E_FCI - E_RHF.
  Every functional is evaluated once on the RHF density (no self-consistency), so differences are functional errors.
  Quadrature: script-12 axially symmetric log-Gauss-Legendre grid (300 x 300), exact for these densities.
Functionals
  exchange + correlation (Libxc): LDA (PW92), PBE, TPSS, SCAN, r2SCAN
  exchange only (spin-unpolarized, closed shell):
    DC-SCAN, DC-SCAN-min, DC-r2SCAN  DC-SCAN note Eqs. (23)-(25), (37): curvature tensor C = -1/2 dd ln n,
                                     D = |grad n|^2 tr C^2 / (grad n.C^2.grad n), Z = sqrt(tr C^2)/k_F^2 (script 01)
    GGA+2D (c = 2, 8), MGGA+2D (c=8) Constantin, PRB 78, 155106 (2008), Eqs. (18), (19), (21)   (script 14)
    Q2D (Libxc GGA_X_Q2D)            Chiodo, Constantin, Fabiano, Della Sala, PRL 108, 126402 (2012)
    Q1D                              Urso and Constantin, EPJB 94, 147 (2021), Eq. (30) (PBEsol base) (script 18)
    MVS, PBEsol                      Libxc
  For N = 2 the RHF density has tau = tau_W (alpha = 0): SCAN = r2SCAN and DC-SCAN = DC-r2SCAN exactly.
Checks (written to the log and the csv)
  * pure Gaussian density 2 chi_000^2 of every trap: grid LDA/SCAN/DC-SCAN vs the analytic Eq. (33) quadrature of
    script 01 (ex_family) and (000|000) vs the exact Gaussian exchange Eq. (44);
  * lam = 2: E_HF, E_FCI, W0 = E_x, E_x^LDA/PBE/SCAN/DC-SCAN reproduce codes/results/trap_ac_results.csv (script 12).
Outputs (codes/results/)
  trap_dc_exchange.csv       one row per system: energies, exact references, every functional, checks
  tab_trap_xc.tex            E_x/E_x^ex and E_c/E_c^ex for LDA, PBE, TPSS, SCAN, r2SCAN (full tabular)
  tab_trap_dc_x.tex          E_x/E_x^ex for SCAN, r2SCAN and the dimensional-crossover exchange functionals
  trap_dc_exchange_log.txt
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import gamma

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('SCAN_RESULTS', os.path.join(HERE, 'results'))
os.makedirs(RES, exist_ok=True)
T0 = time.time()
LOG = []


def log(msg):
    line = f'[{time.time() - T0:8.1f}s] {msg}'
    print(line, flush=True)
    LOG.append(line)


# ==================================================== copied verbatim from codes/12_trap_ac_benchmark.py (A)
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


# ==================================================== end of copy (12 A)


# ==================================================== copied verbatim from codes/12_trap_ac_benchmark.py (C)
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


# ==================================================== end of copy (12 C)


# ==================================================== copied verbatim from codes/01_wire_benchmark.py (DC-SCAN)
CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)        # LDA exchange constant, E_x^LDA = -CX int n^(4/3)

KF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)            # k_F = KF n^(1/3)

C_NORM = 2.0 / np.pi ** 1.5                       # n = lam^k C_NORM exp(-(u + t^2)) in reduced coordinates

SCAN_H0X = 1.174

SCAN_A1 = 4.9479

A1D = 1.0 / 3.0

def m2d():
    """M2D of Eq. (28): the sheet integral of n^(4/3) s^(-1/2) in the lam -> infinity limit."""
    # E_x -> -A2D M2D with  M2D = 2 pi CX c^(4/3) int du dt e^{-4/3(u+t^2)} (K n^{1/3})^{1/2} / |t|^{1/2}
    # (s -> lam^{2/3} |t| / (K n^{1/3}) with lam^{1/3} prefactor cancelling).  Closed form:
    # u and t integrals in closed form:
    a = 4.0 / 3.0 + 1.0 / 6.0                    # s^{-1/2} carries n^{1/6}: exp(-(u+t^2)/6)
    iu = 1.0 / a
    it = 0.5 * gamma(0.25) * a ** (-0.25)        # int_0^inf t^{-1/2} e^{-a t^2} dt
    return 2.0 * np.pi * CX * C_NORM ** (4.0 / 3.0) * np.sqrt(KF) * C_NORM ** (1.0 / 6.0) * iu * it

A2D = np.sqrt(np.pi / 2.0) / m2d()

def ex_exact_sheet(lam):
    """Eq. (10).  lam = 1 -> -sqrt(2/pi)."""
    lam = np.asarray(lam, dtype=float)
    q = np.sqrt(np.maximum(lam ** 2 - 1.0, 0.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(q > 1e-8, np.arctan(q) / np.where(q > 0, q, 1.0), 1.0 - q ** 2 / 3.0)
    return -np.sqrt(2.0 / np.pi) * lam * r

def ex_exact_wire(lam):
    """Eq. (11).  lam = 1 -> -sqrt(2/pi)."""
    lam = np.asarray(lam, dtype=float)
    q = np.sqrt(np.maximum(lam ** 2 - 1.0, 0.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(q > 1e-8, np.arccosh(np.maximum(lam, 1.0)) / np.where(q > 0, q, 1.0), 1.0 - q ** 2 / 3.0)
    return -np.sqrt(2.0 / np.pi) * lam * r

def ex_exact_eq44(diagA, n=4000):
    """Eq. (44): E_x = -(2/sqrt(pi)) int_0^inf sqrt(det A)/sqrt(det(A + 2 t^2 I)) dt, any diagonal A > 0.

    Independent check of Eqs. (10)-(11): substitution t = tan(theta)*sqrt(min a)/sqrt(2) and Gauss-Legendre."""
    a = np.asarray(diagA, dtype=float)
    x, w = np.polynomial.legendre.leggauss(n)
    # t in [0, inf): t = s/(1-s), s in [0,1); integrand ~ t^-3 at large t, so this mapping is benign.
    # Split [0,1) into log-graded panels to resolve the scale 1/sqrt(max a).
    edges = np.concatenate([[0.0], np.logspace(-9, 0, 40)])
    edges[-1] = 1.0 - 1e-15
    tot = 0.0
    sc = np.sqrt(a.min() / 2.0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        ws = 0.5 * (hi - lo) * w
        t = sc * s / (1.0 - s)
        dt = sc / (1.0 - s) ** 2
        f = np.sqrt(np.prod(a)) / np.sqrt(np.prod(a[None, :] + 2.0 * t[:, None] ** 2, axis=1))
        tot += np.sum(ws * dt * f)
    return -2.0 / np.sqrt(np.pi) * tot

def fx_scan_alpha0(s):
    """SCAN F_x(s, alpha = 0) = h0x g_x(s) = 1.174 (1 - exp(-a1/sqrt(s))), g_x(0) = 1 (Eq. 6)."""
    s = np.asarray(s, dtype=float)
    with np.errstate(divide='ignore'):
        return SCAN_H0X * -np.expm1(-SCAN_A1 / np.sqrt(np.where(s > 0, s, 1e-300)))

def f2d(s):
    return A2D / (1.0 + s * s) ** 0.25

def f1d(s):
    return A1D / (1.0 + s * s)

def geometry_weights(D):
    """Eqs. (23)-(24): Lagrange-type selectors of the 2D (D=1), 1D (D=2) and SCAN (D=3) branches."""
    l2 = 0.5 * (D - 2.0) * (D - 3.0)
    l1 = (D - 1.0) * (3.0 - D)
    l3 = 0.5 * (D - 1.0) * (D - 2.0)
    den = l2 * l2 + l1 * l1 + l3 * l3
    return l2 * l2 / den, l1 * l1 / den, l3 * l3 / den

def omega_switch(s, Z, variant='full'):
    """Eq. (25) ('full') or the minimal-information Eq. (37) ('min')."""
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

def family_fields(family, lam, u, t):
    """s, D, Z on the reduced (u, t) grid of Eq. (33) for family 'sheet' or 'wire'.

    wire : u = lam^2 (x^2+y^2), t = z      -> r^T A^2 r = lam^2 u + t^2,  r^T A^4 r = lam^6 u + t^2
    sheet: u = x^2 + y^2,       t = lam z  -> r^T A^2 r = u + lam^2 t^2,  r^T A^4 r = u + lam^6 t^2
    """
    k = 2 if family == 'wire' else 1
    U, T = np.meshgrid(u, t, indexing='ij')
    T2 = T * T
    n = lam ** k * C_NORM * np.exp(-(U + T2))
    kf = KF * np.cbrt(n)
    if family == 'wire':
        a2 = lam ** 2 * U + T2
        a4 = lam ** 6 * U + T2
        trc2 = 2.0 * lam ** 4 + 1.0
    else:
        a2 = U + lam ** 2 * T2
        a4 = U + lam ** 6 * T2
        trc2 = 2.0 + lam ** 4
    s = np.sqrt(a2) / kf                       # |grad n| / (2 k_F n), |grad n|/n = 2 sqrt(r^T A^2 r)
    with np.errstate(invalid='ignore', divide='ignore'):
        D = np.where(a4 > 0, a2 * trc2 / a4, 3.0)
    D = np.clip(D, 1.0, 3.0)
    Z = np.sqrt(trc2) / kf ** 2
    return U, T2, s, D, Z

def ex_family(family, lam, n=400, u_range=(np.exp(-32.0), 26.0), t_range=(np.exp(-32.0), 8.0)):
    """Eq. (33): (E_x^SCAN, E_x^DC-SCAN, E_x^DC-SCAN-min, E_x^LDA) on the Gaussian sheet/wire of parameter lam."""
    k = 2 if family == 'wire' else 1
    u, wu = _log_gl(*u_range, n)
    t, wt = _log_gl(*t_range, n)
    U, T2, s, D, Z = family_fields(family, lam, u, t)
    W = np.outer(wu, wt) * np.exp(-4.0 / 3.0 * (U + T2))
    pref = -2.0 * np.pi * CX * C_NORM ** (4.0 / 3.0) * lam ** (k / 3.0)
    fs = fx_scan_alpha0(s)
    out = {
        'LDA': pref * W.sum(),
        'SCAN': pref * np.sum(W * fs),
        'DC-SCAN': pref * np.sum(W * fx_dcscan(s, D, Z, fs, 'full')),
        'DC-SCAN-min': pref * np.sum(W * fx_dcscan(s, D, Z, fs, 'min')),
    }
    return out
# ==================================================== end of copy (01)


# ==================================================== copied verbatim from codes/14_constantin2008_benchmark.py
AX_PAPER, BX_PAPER = 0.5217, 1.947                  # paper Eqs. (12), (13) -- used in Eqs. (18), (19)

def f_switch(p, c):
    """Eq. (21): p^4 (1 + p^2) / (10^c + p^6), evaluated stably for large p."""
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    small = p <= 1.0
    ps = p[small]
    out[small] = ps ** 4 * (1.0 + ps ** 2) / (10.0 ** c + ps ** 6)
    pl = p[~small]
    out[~small] = (pl ** -2 + 1.0) / (10.0 ** c * pl ** -6 + 1.0)
    return out
# ==================================================== end of copy (14)


# ==================================================== copied verbatim from codes/18_urso2021_prl2025_ibm.py
Q1D_A = 0.06525                                        # Urso & Constantin Eq. (30)

def fx_q1d(s, f_pbesol):
    """Q1D GGA, Urso & Constantin Eq. (30): F = F_PBEsol + w(s) (-F_PBEsol + a/s^2), w = (s^4+s^6)/(1+s^4+s^6)."""
    s2 = s * s
    w = (s2 * s2 + s2 ** 3) / (1.0 + s2 * s2 + s2 ** 3)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        tail = np.where(s > 0, Q1D_A / np.where(s > 0, s2, 1.0), 0.0)
    return f_pbesol + w * (-f_pbesol + tail)
# ==================================================== end of copy (18)


# ==================================================== functionals on the closed-shell density (this script)
XC_PAIRS = {'LDA': ('LDA_X', 'LDA_C_PW'), 'PBE': ('GGA_X_PBE', 'GGA_C_PBE'), 'TPSS': ('MGGA_X_TPSS', 'MGGA_C_TPSS'),
            'SCAN': ('MGGA_X_SCAN', 'MGGA_C_SCAN'), 'R2SCAN': ('MGGA_X_R2SCAN', 'MGGA_C_R2SCAN')}
EXCHANGE_LIBXC = {'LDA': None, 'PBE': 'GGA_X_PBE', 'PBEsol': 'GGA_X_PBE_SOL', 'Q2D': 'GGA_X_Q2D',
                  'TPSS': 'MGGA_X_TPSS', 'MVS': 'MGGA_X_MVS', 'SCAN': 'MGGA_X_SCAN', 'R2SCAN': 'MGGA_X_R2SCAN'}
EXCHANGE_ORDER = ['LDA', 'PBE', 'PBEsol', 'TPSS', 'SCAN', 'R2SCAN', 'MVS', 'Q2D', 'Q1D', 'GGA+2D(c=2)', 'GGA+2D(c=8)',
                  'MGGA+2D(c=8)', 'DC-SCAN', 'DC-SCAN-min', 'DC-R2SCAN', 'DC-PBE', 'DC-PBEsol', 'HSE06(sl)',
                  'DC-HSE06(sl)']
HSE_OMEGA, HSE_BETA = 0.11, 0.25          # PySCF/Libxc HSE06: rsh_coeff = (omega, alpha, beta) = (0.11, 0, 0.25)
TAU_UNIF_X = 0.3 * KF ** 2                 # tau_unif = TAU_UNIF_X n^{5/3}


def density_fields(grid, D, tau_w=False):
    """Closed-shell density of the spin-summed D on the grid: n, grad n, tau, s, p, alpha and the DC descriptors.
    tau_w=True: Kohn-Sham tau of a two-electron singlet, tau = tau_W = |grad n|^2/(8n) (alpha = 0), used for the
    exact (FCI) density, whose one-orbital KS determinant has phi = sqrt(n/2)."""
    rho, grad, hess, lap, tau = grid.fields(D)
    if tau_w:
        tau = np.einsum('ig,ig->g', grad, grad) / (8.0 * np.where(rho > 1e-30, rho, 1.0))
    ok = rho > 1e-30
    nn = np.where(ok, rho, 1.0)
    C = -0.5 * (hess / nn - np.einsum('ig,jg->ijg', grad, grad) / nn ** 2)
    trc2 = np.einsum('ijg,ijg->g', C, C)
    Cg = np.einsum('ijg,jg->ig', C, grad)
    gc2g = np.einsum('ig,ig->g', Cg, Cg)
    gg = np.einsum('ig,ig->g', grad, grad)
    with np.errstate(divide='ignore', invalid='ignore'):
        Dd = np.clip(np.where(gc2g > 0, gg * trc2 / np.where(gc2g > 0, gc2g, 1.0), 3.0), 1.0, 3.0)
    kf = KF * np.cbrt(nn)
    s = np.sqrt(gg) / (2.0 * kf * nn)
    alpha = np.maximum(tau - gg / (8.0 * nn), 0.0) / (TAU_UNIF_X * nn ** (5.0 / 3.0))
    return dict(ok=ok, n=np.where(ok, rho, 0.0), g=grad, tau=tau, s=s, p=s * s, D=Dd, Z=np.sqrt(trc2) / kf ** 2,
                alpha=alpha, W=grid.W)


def libxc_eps(code, f):
    """Libxc energy per particle (unpolarized) of one exchange or correlation code."""
    from pyscf.dft import libxc
    n = np.where(f['ok'], f['n'], 1.0)
    z = np.zeros_like(n)
    if code.startswith('LDA'):
        inp = n
    elif code.startswith('GGA'):
        inp = np.vstack((n, f['g']))
    else:
        inp = np.vstack((n, f['g'], z, f['tau']))
    key = code + ',' if '_X_' in code or code == 'LDA_X' else ',' + code
    return np.asarray(libxc.eval_xc(key, inp, spin=0, deriv=0)[0])


def libxc_eps_key(key, f):
    """Libxc energy per particle (unpolarized) of a full PySCF xc string (GGA input)."""
    from pyscf.dft import libxc
    n = np.where(f['ok'], f['n'], 1.0)
    return np.asarray(libxc.eval_xc(key, np.vstack((n, f['g'])), spin=0, deriv=0)[0])


def eri_longrange(trap, idx, omega, nu=96):
    """(pq|rs) of erf(omega r12)/r12 = (2/sqrt(pi)) int_0^omega du exp(-u^2 r12^2): the ERI construction of build()
    (script 12) with Gauss-Legendre u nodes on [0, omega] (smooth integrand, no log mapping needed)."""
    wb = np.asarray(trap, float)
    nb = len(idx)
    x, w = np.polynomial.legendre.leggauss(nu)
    u, wu = 0.5 * omega * (x + 1.0), 0.5 * omega * w
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


def gaussian_lr_self(A, omega):
    """(000|000) of erf(omega r)/r for the Gaussian 2 chi_000^2 = exp(-r.A.r) density (Eq. 44 restricted to u < omega)."""
    from scipy.integrate import quad
    A = np.asarray(A, float)
    return 2.0 / np.sqrt(np.pi) * quad(lambda t: np.sqrt(np.prod(A)) / np.sqrt(np.prod(A + 2.0 * t * t)), 0.0, omega,
                                       epsabs=1e-14, epsrel=1e-13)[0]


def exchange_factors(f):
    """All exchange enhancement factors F_x = e_x / e_x^LDA on the density f."""
    elda = -CX * np.cbrt(np.where(f['ok'], f['n'], 1.0))
    F = {k: (np.ones_like(f['n']) if c is None else libxc_eps(c, f) / elda) for k, c in EXCHANGE_LIBXC.items()}
    s, p, D, Z = f['s'], f['p'], f['D'], f['Z']
    F['Q1D'] = fx_q1d(s, F['PBEsol'])                                   # Urso & Constantin (2021) Eq. (30)
    with np.errstate(divide='ignore', invalid='ignore'):
        pq = np.where(p > 0, p, 1.0) ** -0.25
    for c in (2.0, 8.0):                                               # Constantin (2008) Eqs. (18), (21)
        F[f'GGA+2D(c={c:g})'] = 1.0 + f_switch(p, c) * (-1.0 + AX_PAPER * pq)
    F['MGGA+2D(c=8)'] = 1.0 + f_switch(p, 8.0) * (-1.0 + BX_PAPER * np.sqrt(f['alpha']))   # Eq. (19)
    F['DC-SCAN'] = fx_dcscan(s, D, Z, F['SCAN'], 'full')               # DC-SCAN note Eqs. (23)-(25), Libxc base
    F['DC-SCAN-min'] = fx_dcscan(s, D, Z, F['SCAN'], 'min')            # minimal-information switch, Eq. (37)
    F['DC-R2SCAN'] = fx_dcscan(s, D, Z, F['R2SCAN'], 'full')
    F['DC-PBE'] = fx_dcscan(s, D, Z, F['PBE'], 'full')                 # the same DC switch on a GGA base
    F['DC-PBEsol'] = fx_dcscan(s, D, Z, F['PBEsol'], 'full')
    # HSE06: semilocal exchange part = Libxc HSE06 (x + PBE c) minus PBE correlation; the 0.25 SR-HF part is added
    # at the energy level in run().  DC-HSE06 applies the DC switch to this semilocal enhancement factor.
    F['HSE06(sl)'] = (libxc_eps_key('HSE06', f) - libxc_eps('GGA_C_PBE', f)) / elda
    F['DC-HSE06(sl)'] = fx_dcscan(s, D, Z, F['HSE06(sl)'], 'full')
    return F


def all_energies(f):
    """E_x of every exchange functional, E_c of every Libxc pair, DC diagnostics."""
    wl = f['W'] * f['n'] * (-CX * np.cbrt(f['n']))
    F = exchange_factors(f)
    out = {f'Ex_{k}': float(np.sum(wl * np.where(f['ok'], F[k], 1.0))) for k in EXCHANGE_ORDER}
    for k, (_, cc) in XC_PAIRS.items():
        out[f'Ec_{k}'] = float(np.sum(f['W'] * f['n'] * np.where(f['ok'], libxc_eps(cc, f), 0.0)))
    w2, w1, _ = geometry_weights(f['D'])
    wn = f['W'] * f['n']
    out['N_grid'] = float(np.sum(wn))
    out['dc_switched_fraction'] = float(np.sum(wn * omega_switch(f['s'], f['Z']) * (w2 + w1)) / out['N_grid'])
    out['mean_D'] = float(np.sum(wn * f['D']) / out['N_grid'])
    return out


def trap_of(family, lam):
    return (1.0, 1.0, lam * lam) if family == 'sheet' else (lam * lam, lam * lam, 1.0)


def gaussian_check(family, lam, eri, idx):
    """Pure Gaussian 2 chi_000^2 of the trap = the fixed density of scripts 01/02: grid vs Eq. (33), ERI vs Eq. (44)."""
    wb = np.array(trap_of(family, lam))
    i0 = int(np.where((idx == 0).all(axis=1))[0][0])
    g = Grid(wb, np.array([[0, 0, 0]]), 'cylinder', 300)
    E = all_energies(density_fields(g, np.array([[2.0]])))
    ref = ex_family(family, lam)
    exact = float(ex_exact_sheet(lam) if family == 'sheet' else ex_exact_wire(lam))
    return dict(gauss_Ex_exact=exact, gauss_d_eri=float(-eri[i0, i0, i0, i0] - exact),
                gauss_d_LDA=E['Ex_LDA'] - ref['LDA'], gauss_d_SCAN=E['Ex_SCAN'] - ref['SCAN'],
                gauss_d_DCSCAN=E['Ex_DC-SCAN'] - ref['DC-SCAN'], gauss_DCSCAN_ratio=ref['DC-SCAN'] / exact,
                gauss_SCAN_ratio=ref['SCAN'] / exact)


def rhf_symmetric(h, eri, idx):
    """RHF with the doubly occupied orbital restricted to the totally symmetric (all-even HO index) subspace, i.e.
    the HF density keeps the trap symmetry.  The unrestricted-parity RHF of script 12 breaks z-parity in the strongly
    correlated wire (lam = 100); its energy lowering is returned as a diagnostic."""
    ev = np.where((idx % 2 == 0).all(axis=1))[0]
    E, _, _, De = rhf(h[np.ix_(ev, ev)], eri[np.ix_(ev, ev, ev, ev)])
    D = np.zeros_like(h)
    D[np.ix_(ev, ev)] = De
    E_free = rhf(h, eri)[0]
    return E, D, E_free - E


def run(family, lam, nb_target, degree, nmax_dir=20):
    trap = np.array(trap_of(family, lam))
    idx, ecut = basis_for_target(trap, nb_target, degree, nmax_dir)
    t = time.time()
    ints = build(trap, trap, idx)
    h, eri = ints['h'], ints['eri']
    E_hf, D, dE_break = rhf_symmetric(h, eri, idx)
    E_fci, _, D_fci, _ = fci_singlet2(h, eri)
    # exact KS exchange of the FCI density: N = 2 singlet, phi_KS = sqrt(rho/2) -> E_x = -U_H/2 = -1/4 tr D J[D]
    Ex_fci = -0.25 * float(np.sum(D_fci * jk(eri, D_fci)[0]))
    J, K = jk(eri, D)
    Ex = -0.25 * float(np.sum(D * K))
    Ec = E_fci - E_hf
    E = all_energies(density_fields(Grid(trap, idx, 'cylinder', 300), D))
    eri_lr = eri_longrange(trap, idx, HSE_OMEGA)
    K_lr = jk(eri_lr, D)[1]
    Ex_sr_hf = Ex + 0.25 * float(np.sum(D * K_lr))                     # -1/4 tr D (K - K_lr)
    i0 = int(np.where((idx == 0).all(axis=1))[0][0])
    d_lr = float(eri_lr[i0, i0, i0, i0] - gaussian_lr_self(np.asarray(trap, float), HSE_OMEGA))
    E['Ex_SR_HF'] = Ex_sr_hf
    # every functional on the exact (FCI) density with the exact KS tau = tau_W, against E_x^FCI = -U_H[n_FCI]/2
    gF = Grid(trap, idx, 'cylinder', 300)
    fF = density_fields(gF, D_fci, tau_w=True)
    EF = all_energies(fF)
    # exact KS correlation of the FCI density (N = 2): E_c^KS = E_FCI - T_W - V_ext - U_H - E_x,  E_x = -U_H/2
    Ts_fci = float(np.sum(fF['W'] * np.where(fF['ok'], fF['tau'], 0.0)))           # T_s = int tau_W
    Vext_fci = float(np.sum(D_fci * ints['Vtrap']))
    UH_fci = -2.0 * Ex_fci
    Ec_ks = E_fci - Ts_fci - Vext_fci - UH_fci - Ex_fci
    Ex_sr_fci = Ex_fci + 0.25 * float(np.sum(D_fci * jk(eri_lr, D_fci)[0]))     # KS SR exchange = -1/4 tr D J_SR[D]
    EF['Ex_HSE06'] = EF['Ex_HSE06(sl)'] + HSE_BETA * Ex_sr_fci
    EF['Ex_DC-HSE06'] = EF['Ex_DC-HSE06(sl)'] + HSE_BETA * Ex_sr_fci
    E['Ex_HSE06'] = E['Ex_HSE06(sl)'] + HSE_BETA * Ex_sr_hf
    E['Ex_DC-HSE06'] = E['Ex_DC-HSE06(sl)'] + HSE_BETA * Ex_sr_hf
    r = dict(system=f'{family}_{lam:g}', family=family, lam=lam, omega='x'.join(f'{w:g}' for w in trap), nb=len(idx),
             ecut=ecut, nmax_dir=nmax_dir, E_HF=E_hf, E_FCI=E_fci, Ex_FCI=Ex_fci, dE_RHF_symmetry_breaking=dE_break, J=0.5 * float(np.sum(D * J)), Ex_exact=Ex, Ec_exact=Ec, **E)
    for k in EXCHANGE_ORDER + ['HSE06', 'DC-HSE06']:
        r[f'Rx_{k}'] = E[f'Ex_{k}'] / Ex
    r['gauss_d_eri_lr'] = d_lr
    for k in EXCHANGE_ORDER + ['HSE06', 'DC-HSE06']:
        r[f'ExF_{k}'] = EF[f'Ex_{k}']
        r[f'RxF_{k}'] = EF[f'Ex_{k}'] / Ex_fci
    r['N_grid_FCI'] = EF['N_grid']
    r.update(Ts_FCI=Ts_fci, Vext_FCI=Vext_fci, UH_FCI=UH_fci, Ec_KS=Ec_ks)
    for k in XC_PAIRS:
        r[f'EcF_{k}'] = EF[f'Ec_{k}']
        r[f'RcF_{k}'] = EF[f'Ec_{k}'] / Ec_ks
    r['dc_switched_fraction_FCI'] = EF['dc_switched_fraction']
    for k in XC_PAIRS:
        r[f'Rc_{k}'] = E[f'Ec_{k}'] / Ec
    r.update(gaussian_check(family, lam, eri, idx))
    log(f'{r["system"]:10s} nb={len(idx)} E_HF={E_hf:.10f} E_FCI={E_fci:.10f} Ex={Ex:.8f} Ec={Ec:.8f} '
        f'Ex_FCI={Ex_fci:.8f} N={E["N_grid"]:.10f} RHF(free)-RHF(sym)={dE_break:.1e} ({time.time() - t:.1f}s)')
    log('    Ex/Ex^ex ' + '  '.join(f'{k}={r[f"Rx_{k}"]:.4f}' for k in EXCHANGE_ORDER + ['HSE06', 'DC-HSE06']))
    log(f'    E_x^SR-HF(omega=0.11) {Ex_sr_hf:.8f}  Gaussian LR (000|000) check {d_lr:.1e}')
    log('    FCI density, Ex/Ex^FCI ' + '  '.join(f'{k}={r[f"RxF_{k}"]:.4f}' for k in ('LDA', 'PBE', 'HSE06', 'SCAN',
                                                                                    'DC-PBE', 'DC-HSE06', 'DC-SCAN'))
        + f'  N={EF["N_grid"]:.10f}')
    log(f'    E_c^KS(FCI density) {Ec_ks:.8f} (E_FCI-E_HF {Ec:.8f})  Ec/Ec^KS ' + '  '.join(f'{k}={r[f"RcF_{k}"]:.4f}' for k in XC_PAIRS))
    log('    Ec/Ec^ex ' + '  '.join(f'{k}={r[f"Rc_{k}"]:.4f}' for k in XC_PAIRS))
    log(f'    Gaussian check: dERI {r["gauss_d_eri"]:.1e}  dLDA {r["gauss_d_LDA"]:.1e}  dSCAN {r["gauss_d_SCAN"]:.1e}  '
        f'dDC-SCAN {r["gauss_d_DCSCAN"]:.1e};  DC switched fraction {r["dc_switched_fraction"]:.3f}')
    return r


def check_script12(rows):
    """lam = 2 rows vs script 12's codes/results/trap_ac_results.csv (HF density, same basis rule nb >= 84)."""
    path = os.path.join(RES, 'trap_ac_results.csv')
    if not os.path.exists(path):
        log('script-12 check skipped (trap_ac_results.csv missing)')
        return
    old = {(q['system'], q['density']): q for q in csv.DictReader(open(path))}
    for r in rows:
        q = old.get((f'{r["family"]}_{r["lam"]:g}', 'HF'))
        if q is None or int(q['nb']) != r['nb']:
            continue
        d = {'E_HF': r['E_HF'] - float(q['E_HF']), 'E_FCI': r['E_FCI'] - float(q['E_FCI']),
             'W0': r['Ex_exact'] - float(q['W0'])}
        for k, kk in (('LDA', 'LDA'), ('PBE', 'PBE'), ('SCAN', 'SCAN'), ('DC-SCAN', 'DC-SCAN')):
            d[f'Ex_{k}'] = r[f'Ex_{k}'] - float(q[f'Ex_{kk}'])
        q2 = old.get((f'{r["family"]}_{r["lam"]:g}', 'FCI'))
        if q2 is not None:
            d['W0(FCI)'] = r['Ex_FCI'] - float(q2['W0'])
            for k in ('LDA', 'PBE', 'SCAN', 'DC-SCAN'):        # script 12 also uses tau = tau_W on the FCI density
                d[f'ExF_{k}'] = r[f'ExF_{k}'] - float(q2[f'Ex_{k}'])
            for k in ('LDA', 'PBE', 'SCAN'):
                d[f'EcF_{k}'] = r[f'EcF_{k}'] - float(q2[f'Ec_{k}'])
            d['Ec_KS'] = r['Ec_KS'] - (float(q2['E_FCI']) - float(q2['T_s']) - float(q2['E_ext']) - float(q2['J'])
                                       - float(q2['W0']))
        r['script12_maxdiff'] = max(abs(v) for v in d.values())
        log(f'script-12 check {r["system"]}: ' + '  '.join(f'{k} {v:.1e}' for k, v in d.items()))


XLAB = {'R2SCAN': r'r$^2$SCAN', 'DC-R2SCAN': r'DC-r$^2$SCAN', 'GGA+2D(c=2)': r'GGA+2D$^a$', 'GGA+2D(c=8)': r'GGA+2D$^b$',
        'MGGA+2D(c=8)': r'MGGA+2D$^b$'}
XC_TAB = ['LDA', 'PBE', 'TPSS', 'SCAN', 'R2SCAN']
DC_TAB = ['PBE', 'DC-PBE', 'HSE06', 'DC-HSE06', 'SCAN', 'DC-SCAN', 'R2SCAN', 'DC-R2SCAN', 'GGA+2D(c=2)', 'MGGA+2D(c=8)',
          'Q2D', 'Q1D']


def lamstr(lam):
    return f'{lam:g}'


def write_tables(rows):
    lines = [r'\begin{tabular}{lrrrr' + 'rr' * len(XC_TAB) + '}', r'\toprule',
             r'& & & & & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{XLAB.get(x, x)}}}' for x in XC_TAB) + r' \\',
             r'System & $\lambda$ & $E_{\rm x}^{\rm ex}$ & $E_{\rm x}^{\rm FCI}$ & $E_{\rm c}^{\rm ex}$ & '
             + ' & '.join([r'$E_{\rm x}$ & $E_{\rm c}$'] * len(XC_TAB)) + r' \\', r'\midrule']
    for fam in ('sheet', 'wire'):
        for r in [q for q in rows if q['family'] == fam]:
            c = []
            for x in XC_TAB:
                c += [f'{r[f"Rx_{x}"]:.3f}', f'{r[f"Rc_{x}"]:.3f}']
            lines.append(' & '.join([fam, lamstr(r['lam']), f'{r["Ex_exact"]:.4f}', f'{r["Ex_FCI"]:.4f}',
                                     f'{r["Ec_exact"]:.4f}'] + c) + r' \\')
        lines.append(r'\midrule' if fam == 'sheet' else r'\bottomrule')
    lines.append(r'\end{tabular}')
    open(os.path.join(RES, 'tab_trap_xc.tex'), 'w').write('\n'.join(lines) + '\n')
    lines = [r'\begin{tabular}{lrrr' + 'r' * len(DC_TAB) + '}', r'\toprule',
             r'System & $\lambda$ & $E_{\rm x}^{\rm ex}$ & $E_{\rm x}^{\rm FCI}$ & ' + ' & '.join(XLAB.get(x, x) for x in DC_TAB) + r' \\',
             r'\midrule']
    for fam in ('sheet', 'wire'):
        for r in [q for q in rows if q['family'] == fam]:
            lines.append(' & '.join([fam, lamstr(r['lam']), f'{r["Ex_exact"]:.4f}', f'{r["Ex_FCI"]:.4f}']
                                    + [f'{r[f"Rx_{x}"]:.3f}' for x in DC_TAB]) + r' \\')
        lines.append(r'\midrule' if fam == 'sheet' else r'\bottomrule')
    lines.append(r'\end{tabular}')
    open(os.path.join(RES, 'tab_trap_dc_x.tex'), 'w').write('\n'.join(lines) + '\n')
    lines = [r'\begin{tabular}{lrr' + 'r' * len(DC_TAB) + '}', r'\toprule',
             r'System & $\lambda$ & $E_{\rm x}^{\rm FCI}$ & ' + ' & '.join(XLAB.get(x, x) for x in DC_TAB) + r' \\',
             r'\midrule']
    for fam in ('sheet', 'wire'):
        for r in [q for q in rows if q['family'] == fam]:
            lines.append(' & '.join([fam, lamstr(r['lam']), f'{r["Ex_FCI"]:.4f}']
                                    + [f'{r[f"RxF_{x}"]:.3f}' for x in DC_TAB]) + r' \\')
        lines.append(r'\midrule' if fam == 'sheet' else r'\bottomrule')
    lines.append(r'\end{tabular}')
    open(os.path.join(RES, 'tab_trap_dc_x_fci.tex'), 'w').write('\n'.join(lines) + '\n')
    lines = [r'\begin{tabular}{lrrr' + 'r' * len(XC_TAB) + '}', r'\toprule',
             r'System & $\lambda$ & $E_{\rm c}^{\rm FCI-HF}$ & $E_{\rm c}^{\rm KS}$ & '
             + ' & '.join(XLAB.get(x, x) for x in XC_TAB) + r' \\', r'\midrule']
    for fam in ('sheet', 'wire'):
        for r in [q for q in rows if q['family'] == fam]:
            lines.append(' & '.join([fam, lamstr(r['lam']), f'{r["Ec_exact"]:.4f}', f'{r["Ec_KS"]:.4f}']
                                    + [f'{r[f"RcF_{x}"]:.3f}' for x in XC_TAB]) + r' \\')
        lines.append(r'\midrule' if fam == 'sheet' else r'\bottomrule')
    lines.append(r'\end{tabular}')
    open(os.path.join(RES, 'tab_trap_c_fci.tex'), 'w').write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='lam = 1, 2 only')
    ap.add_argument('--lams', type=float, nargs='+', default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    ap.add_argument('--nb', type=int, default=84, help='basis-size target (script 12 default)')
    ap.add_argument('--degree', type=int, default=4, help='aniGTO total-degree set always included ([spdfg] = 4)')
    ap.add_argument('--nmax-dir', type=int, default=20, help='max HO quanta per direction (script 12: 20)')
    a = ap.parse_args()
    lams = [1.0, 2.0] if a.quick else a.lams
    rows = [run(fam, lam, a.nb, a.degree, a.nmax_dir) for fam in ('sheet', 'wire') for lam in lams]
    check_script12(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(RES, 'trap_dc_exchange.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_tables(rows)
    open(os.path.join(RES, 'trap_dc_exchange_log.txt'), 'w').write('\n'.join(LOG) + '\n')
    log(f'wrote trap_dc_exchange.csv, tab_trap_xc.tex, tab_trap_dc_x.tex in {RES}')


if __name__ == '__main__':
    main()
