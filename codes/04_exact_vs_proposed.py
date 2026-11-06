#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd codes
#     python3 04_exact_vs_proposed.py
#
# Needs: numpy, scipy, pandas, matplotlib.  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""Exact references versus the proposed numerical methods: (i) default aniGTO repulsion integral versus the logarithmic-variable quadrature; (ii) exact N = 2 SCE of isotropic Gaussians versus axisymmetric optimal transport.

STANDALONE SCRIPT (SCAN_1D_2D/codes): run it alone, from any directory,

    python3 04_exact_vs_proposed.py   (about 5 min: the SCE ladder dominates)

It needs only third-party packages (numpy, scipy, pandas, matplotlib); every project function it uses is defined in this file.
Outputs: results/iee_robust.csv, tab_iee_robust_rows.tex, validation.csv, tab_exact_vs_proposed.tex; figures/dcscan_exact_vs_proposed.pdf/.png
Manuscript items: main-text exact-vs-proposed table and figure

---------------------------------------------------------------------------------------------------------------
(i)  Robust aniGTO repulsion integral: the default aniGTO.IEE (scipy quad on t in [0, 1]) loses
    accuracy when the exponent ratio exceeds ~1e6 and fails (errors of Ha) beyond ~1e9.  IEE_robust
    uses the same aniGTO 1-D kernel (integral3) in the logarithmic variable s = ln u.
    Tested on the self-repulsion of the wire and sheet Gaussians against the closed forms, and timed.
Outputs: results/wire_offset.csv, results/iee_robust.csv, results/tab_wire_offset_rows.tex,
         results/tab_iee_robust_rows.tex

(ii)  the validation set (isotropic Gaussian: exact radial SCE known).
Outputs: results/sce_axisym.csv, results/validation.csv, results/comotion_<fam>_w<omega>.npz, log on stdout.

Table + figure 'exact versus proposed' for the SCAN_1D_2D manuscript.

Inputs (already generated):
  results/iee_robust.csv                            (iee_rows() below)
  sce_axisym/results/validation.csv (sce_axisym/run_validation.py)
Outputs (copied into manuscript/ by manuscript/build.sh):
  generated/tab_exact_vs_proposed.tex, generated/dcscan_exact_vs_proposed.pdf
"""
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
import warnings
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.special import comb
from scipy.special import gamma

# ==============================================================================================================
# DC-SCAN functional and Gaussian-family quadrature (Appendix A of the manuscript; shared by all scripts)
# ==============================================================================================================
CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)        # LDA exchange constant, E_x^LDA = -CX int n^(4/3)
KF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)            # k_F = KF n^(1/3)
C_NORM = 2.0 / np.pi ** 1.5                       # n = lam^k C_NORM exp(-(u + t^2)) in reduced coordinates

# SCAN exchange (Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)); alpha = 0 branch only is needed here.
SCAN_H0X = 1.174
SCAN_A1 = 4.9479

# Sheet coefficient, Eq. (28)-(29): M2D = CX K c^(3/2) (2 pi/3)^(-1/4)... evaluated numerically below.
M2D_NOTE = 1.918653475142493
A2D_NOTE = 0.6532258969913366
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


# --------------------------------------------------------------------------- exact references
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


def ex_wire_asymptotic(lam):
    """Eq. (13): -sqrt(2/pi) ln(2 lam), the leading large-lam form."""
    return -np.sqrt(2.0 / np.pi) * np.log(2.0 * np.asarray(lam, dtype=float))


# --------------------------------------------------------------------------- enhancement factors
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


# --------------------------------------------------------------------------- Gaussian-family quadrature
def _log_gl(lo, hi, n):
    """Gauss-Legendre nodes in log coordinates: x in [lo, hi], returns x and weights for dx."""
    p, w = np.polynomial.legendre.leggauss(n)
    a, b = np.log(lo), np.log(hi)
    q = 0.5 * (b - a) * p + 0.5 * (b + a)
    x = np.exp(q)
    return x, 0.5 * (b - a) * w * x


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


def ex_exact(family, lam):
    return ex_exact_wire(lam) if family == 'wire' else ex_exact_sheet(lam)


def diag_A(family, lam):
    return (lam ** 2, lam ** 2, 1.0) if family == 'wire' else (1.0, 1.0, lam ** 2)

# ==============================================================================================================
# aniGTO two-electron repulsion integral (default quadrature)
# ==============================================================================================================
# The four functions below are copied verbatim from aniGTO.py (aniGT02025-main): the 1-D Gaussian
# moment integral, its products, the u-dependent 1-D kernel and the default repulsion integral IEE,
# i.e. the Boys-type representation (2/sqrt(pi)) int_0^1 dt (1-t^2)^(-3/2) prod_k I_k(u^2),
# u^2 = t^2/(1-t^2), integrated with scipy quad on t in [0, 1] (SI, Section "aniGTO integrals").
def fact2(n):
    """Calculate the double factorial (n - 1)!! of a non-negative integer 'n'."""
    result = 1
    for i in range(n, 1, -2):
        result *= i
    return result


def integral(a,b):
    r"""this function evaluate an integral like this:
    int_{-\infty}^{+\infty}dx x^a e^{-bx^2}
    """
    if a%2==0.00000000000000000:
        return np.sqrt(np.pi/b)*fact2(a-1)/(2*b)**(a/2)
    else:
        return 0.000000000000000000000


def integral2(alpha,U,a,A,b,B,c,C,d,D):
    """this function evaluates an integral like the one above"""
    N=0.000000000000000000000
    for i in range(a+1):
        for j in range(b+1):
            for k in range(c+1):
                for r in range(d+1):
                    N=N+comb(a,i)*comb(b,j)*comb(c,k)*comb(d,r)*(U-A)**(a-i)*(U-B)**(b-j)*(U-C)**(c-k)*(U-D)**(d-r)*integral(i+j+k+r,alpha)
    return N


def integral3(u2,a,n,A,b,m,B,c,p,C,d,q,D):
    """evaluates the integral over x_1 and x_2 only
    which is the on above"""
    xi=(a+b)*u2/(a+b+u2)
    PAB=(a*A+b*B)/(a+b)
    PCD=(c*C+d*D)/(c+d)
    U=((c+d)*PCD+xi*PAB)/(c+d+xi)
    A1=(A*(a+b+u2)-(a+b)*PAB)/u2
    B1=(B*(a+b+u2)-(a+b)*PAB)/u2
    N=0.000000000000000000000000000000
    for i in range(n+1):
        for j in range(m+1):
            if (i+j)%2==0:
                N=N+comb(n,i)*comb(m,j)*fact2(i+j-1)/(2*a+2*b+2*u2)**((i+j)/2)*((u2)/(a+b+u2))**(n+m-i-j)*np.exp(-(c+d)*xi/(c+d+xi)*(PAB-PCD)**2)*integral2(c+d+xi,U,n-i,A1,m-j,B1,p,C,q,D)
            else:
                N=N+0.0000000000000000000000000000
    return np.sqrt(np.pi/(a+b+u2))*N


def integrand2(t, a, l1, A, b, l2, B, c, l3, C, d, l4, D):
    """Define the integrand for numerical integration."""
    ax, ay, az = a
    nx, ny, nz = l1
    Ax, Ay, Az = A
    bx, by, bz = b
    mx, my, mz = l2
    Bx, By, Bz = B
    cx, cy, cz = c
    px, py, pz = l3
    Cx, Cy, Cz = C
    dx, dy, dz = d
    qx, qy, qz = l4
    Dx, Dy, Dz = D
    u2 = t**2 / (1 - t**2)
    return (
        np.power(1 - t**2, -1.50000000000000000000000000000000000000)
        * integral3(u2, ax, nx, Ax, bx, mx, Bx, cx, px, Cx, dx, qx, Dx)
        * integral3(u2, ay, ny, Ay, by, my, By, cy, py, Cy, dy, qy, Dy)
        * integral3(u2, az, nz, Az, bz, mz, Bz, cz, pz, Cz, dz, qz, Dz)
    )


def IEE(a,l1,A,b,l2,B,c,l3,C,d,l4,D):
    ax,ay,az=a
    Ax,Ay,Az=A
    bx,by,bz=b
    Bx,By,Bz=B
    cx,cy,cz=c
    Cx,Cy,Cz=C
    dx,dy,dz=d
    Dx,Dy,Dz,=D
    ExAB=np.exp(-ax*bx*(Ax-Bx)**2/(ax+bx))
    EyAB=np.exp(-ay*by*(Ay-By)**2/(ay+by))
    EzAB=np.exp(-az*bz*(Az-Bz)**2/(az+bz))
    ExCD=np.exp(-cx*dx*(Cx-Dx)**2/(cx+dx))
    EyCD=np.exp(-cy*dy*(Cy-Dy)**2/(cy+dy))
    EzCD=np.exp(-cz*dz*(Cz-Dz)**2/(cz+dz))

    integral,error=quad(integrand2,0.0000000000000,1.00000000000000000000000,(a,l1,A,b,l2,B,c,l3,C,d,l4,D),epsrel =1e-16, epsabs= 1e-16)


    return 2.00000000000000000000000/np.sqrt(np.pi)*ExAB*EyAB*EzAB*ExCD*EyCD*EzCD*integral

# ==============================================================================================================
# Axisymmetric N = 2 SCE optimal-transport solver (stabilized Sinkhorn with rigorous bounds)
# ==============================================================================================================
def gl(lo, hi, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def make_measure(rho_fun, Rmax, Zmax, nR, nz, zsym=True, prune=1e-13):
    """Quadrature points (R, z) and probability weights a of rho/2 on the half plane."""
    R, wR = gl(0.0, Rmax, nR)
    Z, wZ = gl(-Zmax, Zmax, nz)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    W = np.outer(wR, wZ) * 2.0 * np.pi * RR
    rho = np.maximum(rho_fun(RR, ZZ), 0.0)
    mass = rho * W
    Ntot = float(mass.sum())
    a = (mass / Ntot).ravel()
    keep = a > prune * a.max()
    return RR.ravel()[keep], ZZ.ravel()[keep], a[keep] / a[keep].sum(), Ntot


def cost(R, Z):
    return 1.0 / np.sqrt((R[:, None] + R[None, :]) ** 2 + (Z[:, None] - Z[None, :]) ** 2)


def sinkhorn_sym(C, a, eps_final, eps0=None, factor=0.5, iters_per_stage=200, tol=1e-9, maxit=20000):
    """Symmetric Sinkhorn for OT(a, a; C) with epsilon scaling and absorption stabilisation.

    Plan P_ij = a_i a_j exp((f_i + f_j - C_ij)/eps).  Within a stage f = f0 + eps log s and the kernel
    K_ij = exp((f0_i + f0_j - C_ij)/eps) is fixed, so one iteration is one mat-vec:
        s <- sqrt(s / (K (a s)))      (fixed point: s_i (K a s)_i = 1, i.e. row marginal a_i).
    The potential is absorbed into f0 at the end of every stage.  tol is on the marginal residual."""
    f = np.zeros_like(a)
    eps = eps0 or float(C.max() - C.min())
    it = 0
    while True:
        last = eps <= eps_final
        K = np.exp(np.minimum((f[:, None] + f[None, :] - C) / eps, 700.0))
        s = np.ones_like(a)
        for k in range(maxit if last else iters_per_stage):
            Ks = K @ (a * s)
            s = np.sqrt(s / np.maximum(Ks, 1e-300))
            it += 1
            if k % 10 == 0 and np.max(np.abs(s * (K @ (a * s)) - 1.0)) < (tol if last else 1e-3):
                break
        f = f + eps * np.log(s)
        if last:
            return f, eps, it
        eps = max(eps * factor, eps_final)


def plan(C, a, f, eps):
    la = np.log(a)
    logP = (f[:, None] + f[None, :] - C) / eps + la[:, None] + la[None, :]
    return np.exp(logP)


def round_plan(P, a):
    """Altschuler-Weed-Rigollet rounding onto the exact marginals (a, a)."""
    r = np.minimum(a / np.maximum(P.sum(1), 1e-300), 1.0)
    P = P * r[:, None]
    c = np.minimum(a / np.maximum(P.sum(0), 1e-300), 1.0)
    P = P * c[None, :]
    er, ec = a - P.sum(1), a - P.sum(0)
    s = er.sum()
    if s > 0:
        P = P + np.outer(er, ec) / s
    return P


def solve(R, Z, a, eps_rel=2e-4, **kw):
    """Discrete OT value with rigorous discrete bounds.  eps_rel is relative to the mean cost."""
    C = cost(R, Z)
    scale = float(a @ C @ a)
    f, eps, nit = sinkhorn_sym(C, a, eps_rel * scale, **kw)
    P = round_plan(plan(C, a, f, eps), a)
    ub = float(np.sum(P * C))
    uc = np.min(C - f[:, None], axis=0)          # c-transform: u^c_j = min_i C_ij - f_i
    lb = float(a @ f + a @ uc)
    # barycentric co-motion map (diagnostic): partner position of each point
    Pn = P / a[:, None]
    fR, fZ = Pn @ R, Pn @ Z
    return dict(V=0.5 * (ub + lb), V_ub=ub, V_lb=lb, eps=eps, iters=nit, n=len(a), fR=fR, fZ=fZ)


def density_from_table(npz, key='rho_fci'):
    d = np.load(npz)
    it = RegularGridInterpolator((d['R'], d['Z']), d[key], method='cubic', bounds_error=False, fill_value=0.0)

    def rho(RR, ZZ):
        return it(np.stack([RR.ravel(), ZZ.ravel()], axis=1)).reshape(RR.shape)
    return rho, d


# ── exact spherical reference and simple bounds ──────────────────────────────
def radial_sce_n2(rho_r, rmax, n=200001):
    """Exact N=2 SCE for a spherical density rho(r): f(r) = -Ne^{-1}(2 - Ne(r)) r_hat,
    V = int_{Ne(r) < 1} 4 pi r^2 rho / (r + |f|)."""
    r = np.linspace(0.0, rmax, n)
    m = 4 * np.pi * r ** 2 * rho_r(r)
    Ne = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * np.diff(r))])
    Ne *= 2.0 / Ne[-1]
    inner = Ne < 1.0
    fr = np.interp(2.0 - Ne[inner], Ne + 1e-15 * np.arange(n), r)
    y = m[inner] / (r[inner] + fr)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(r[inner])))


def inversion_bound(rho_fun, Rmax, Zmax, nR=400, nz=800):
    """V^SCE <= (1/4) int rho / |r|  (trial co-motion f(r) = -r, admissible for rho(-r) = rho(r))."""
    R, wR = gl(0.0, Rmax, nR)
    Z, wZ = gl(-Zmax, Zmax, nz)
    RR, ZZ = np.meshgrid(R, Z, indexing='ij')
    W = np.outer(wR, wZ) * 2 * np.pi * RR
    rho = np.maximum(rho_fun(RR, ZZ), 0.0)
    N = float((rho * W).sum())
    return 0.25 * float((rho * W / np.sqrt(RR ** 2 + ZZ ** 2)).sum()) * 2.0 / N

# ==============================================================================================================
# Main calculation
# ==============================================================================================================
H = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(H, 'results')
FIG = os.path.join(H, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

def IEE_robust(a, l1, A, b, l2, B, c, l3, C, dd, l4, D):
    """Drop-in replacement for IEE.  Same aniGTO 1-D kernel (integral3), but the Boys-type
    integral is taken in s = ln u instead of t = u / sqrt(1 + u^2):  dt (1 - t^2)^(-3/2) = du, so
        int_0^1 integrand2(t) dt = int ds e^s prod_k integral3(e^{2s}, ...)_k.
    In t the region u^2 ~ exponent ratio sits at 1 - t ~ 1/(2u^2) and is lost to round-off (and 1 - t^2
    underflows to 0) once the ratio exceeds ~1e10; in s it is an ordinary O(1)-wide feature.
    Range: s in [ln sqrt(min exp) - 20, ln sqrt(max exp) + 20], unit breakpoints, plus u0 f(0) for [0, u0]."""
    E = 1.0
    for (p, P), (q, Q) in (((a, A), (b, B)), ((c, C), (dd, D))):
        for k in range(3):
            E *= np.exp(-p[k] * q[k] * (P[k] - Q[k]) ** 2 / (p[k] + q[k]))

    def f(s):
        u2 = np.exp(2.0 * s)
        return np.exp(s) * np.prod([integral3(u2, a[k], l1[k], A[k], b[k], l2[k], B[k], c[k], l3[k], C[k],
                                                     dd[k], l4[k], D[k]) for k in range(3)])
    ex = np.array([a, b, c, dd], dtype=float)
    lo, hi = 0.5 * np.log(ex.min()) - 20.0, 0.5 * np.log(ex.max()) + 20.0
    val, _ = quad(f, lo, hi, points=np.arange(np.floor(lo) + 1.0, hi, 1.0), limit=4000, epsrel=1e-13, epsabs=0.0)
    return 2.0 / np.sqrt(np.pi) * E * (val + f(lo))


def self_repulsion(family, lam, fn):
    Adiag = np.array(diag_A(family, lam))
    a = tuple(Adiag / 2.0)
    O, L = (0.0, 0.0, 0.0), (0, 0, 0)
    norm4 = np.prod(Adiag) / np.pi ** 3
    return -norm4 * fn(a, L, O, a, L, O, a, L, O, a, L, O)


def iee_rows():
    """Default aniGTO IEE versus the logarithmic-variable quadrature (eq. 1 of the main text)."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for fam in ('wire', 'sheet'):
            for lam in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8):   # 1e1, 1e7 added 2026-11-05 (user)
                ex = float(ex_exact(fam, lam))
                t0 = time.time(); v0 = self_repulsion(fam, lam, IEE); t1 = time.time()
                v1 = self_repulsion(fam, lam, IEE_robust); t2 = time.time()
                rows.append(dict(family=fam, lam=lam, exact=ex, err_default=v0 - ex, err_robust=v1 - ex,
                                 ms_default=1e3 * (t1 - t0), ms_robust=1e3 * (t2 - t1)))
                print('%s lam=%.0e  default err %.1e (%.0f ms)  robust err %.1e (%.0f ms)'
                      % (fam, lam, v0 - ex, rows[-1]['ms_default'], v1 - ex, rows[-1]['ms_robust']), flush=True)
        # cost on an ordinary anisotropic primitive quartet with angular momentum (Hookium-like ratio 5)
        a, b = (0.25, 0.25, 0.05), (0.5, 0.5, 0.1)
        L1, L2, O = (2, 0, 1), (0, 1, 1), (0.0, 0.0, 0.0)
        t0 = time.time(); v0 = IEE(a, L1, O, b, L2, O, a, L1, O, b, L2, O); t1 = time.time()
        v1 = IEE_robust(a, L1, O, b, L2, O, a, L1, O, b, L2, O); t2 = time.time()
        rows.append(dict(family='hookium_dg', lam=5.0, exact=np.nan, err_default=np.nan,
                         err_robust=v1 - v0, ms_default=1e3 * (t1 - t0), ms_robust=1e3 * (t2 - t1)))
        print('Hookium (d,g)-type quartet: |robust - default| = %.1e  (%.0f / %.0f ms)'
              % (abs(v1 - v0), 1e3 * (t1 - t0), 1e3 * (t2 - t1)), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, 'iee_robust.csv'), index=False)

    def sci(v):
        if not np.isfinite(v):
            return '--'
        if v == 0:
            return '$0$'
        e = int(np.floor(np.log10(abs(v))))
        return '$%.1f\\times10^{%d}$' % (v / 10 ** e, e)
    with open(os.path.join(RES, 'tab_iee_robust_rows.tex'), 'w') as f:
        for _, r in t[t.family != 'hookium_dg'].iterrows():
            f.write('%s & $10^{%d}$ & %s & %s & %.0f & %.0f \\\\\n'
                    % (r.family, int(round(np.log10(r.lam))), sci(r.err_default), sci(r.err_robust),
                       r.ms_default, r.ms_robust))
    return t


LADDER = (24, 34, 48)          # nodes across the shorter (scaled) extent
MAXPTS = 7500


def extents(rho, d, thr=1e-10):
    RR, ZZ = np.meshgrid(d['R'], d['Z'], indexing='ij')
    r = np.abs(rho(RR, ZZ))
    m = r.max()
    Rext = float(d['R'][np.where(r.max(axis=1) > thr * m)[0][-1]])
    Zext = float(np.abs(d['Z'][np.where(r.max(axis=0) > thr * m)[0]]).max())
    return Rext, Zext


def ladder_solve(rho, Rext, Zext, eps_rel=3e-4):
    L0 = min(Rext, 2 * Zext)
    rows = []
    for n in LADDER:
        nR, nz = int(round(n * Rext / L0)), int(round(n * 2 * Zext / L0))
        while nR * nz > MAXPTS * 1.6:            # pruning removes ~35-40 % of the corners
            nR, nz = int(nR * 0.95), int(nz * 0.95)
        t = time.time()
        R, Z, a, N = make_measure(rho, Rext, Zext, nR, nz)
        r = solve(R, Z, a, eps_rel=eps_rel)
        r.update(nR=nR, nz=nz, h=1.0 / nR, N=N, sec=time.time() - t, R=R, Z=Z, a=a)
        print('   nR=%3d nz=%3d pts=%5d  V=%.7f [lb %.7f ub %.7f]  %.0fs'
              % (nR, nz, r['n'], r['V'], r['V_lb'], r['V_ub'], r['sec']), flush=True)
        rows.append(r)
    h = np.array([r['h'] for r in rows])
    V = np.array([r['V'] for r in rows])
    M = np.stack([np.ones(3), h ** 2, h ** 3], axis=1)
    V3 = float(np.linalg.solve(M, V)[0])
    V2 = float(V[-1] + (V[-1] - V[-2]) / ((h[-2] / h[-1]) ** 2 - 1))
    # two-point h^2 Richardson on the finest pair is the estimate (the h^3 term amplifies the
    # 1e-4-relative Sinkhorn bracket); uncertainty = max(|V3 - V2|, 2e-4 V), floor from validation
    return V2, max(abs(V3 - V2), 2e-4 * abs(V2)), rows


def sce_validation():
    """N = 2 SCE of isotropic Gaussians: axisymmetric optimal transport versus the exact radial solution."""
    out = []
    for w, R0 in ((1.0, 5.0), (0.1, 5.0 / np.sqrt(0.1))):
        rho = lambda R, Z, w=w: 2 * (w / np.pi) ** 1.5 * np.exp(-w * (R ** 2 + Z ** 2))
        ex = radial_sce_n2(lambda r, w=w: 2 * (w / np.pi) ** 1.5 * np.exp(-w * r * r), 1.6 * R0)
        print('validation: isotropic Gaussian w=%g  exact radial V_SCE=%.8f' % (w, ex), flush=True)
        V, err, rows = ladder_solve(rho, R0, R0)
        out.append(dict(case='iso_gauss_w%g' % w, V_exact=ex, V_axisym=V, extrap_err=err,
                        dev=V - ex, finest=rows[-1]['V'], ladder=';'.join('%d:%.9f' % (r['nR'], r['V']) for r in rows)))
        print('   extrapolated %.8f  dev %.2e  (est %.1e)' % (V, V - ex, err), flush=True)
    pd.DataFrame(out).to_csv(os.path.join(RES, 'validation.csv'), index=False)


def make_table_and_figure():
    OUT = RES
    iee = pd.read_csv(os.path.join(RES, 'iee_robust.csv'))
    val = pd.read_csv(os.path.join(RES, 'validation.csv'))
    iee = iee[iee.family.isin(['wire', 'sheet'])].copy()
    iee['default'] = iee.exact + iee.err_default
    iee['proposed'] = iee.exact + iee.err_robust


    def mha(v):
        """|error| in mHa; values below double-precision resolution of the energy print as < 1e-11."""
        v = abs(v) * 1e3
        if v < 1e-11:
            return r'$<10^{-11}$'
        if v >= 0.01:
            return '%.3f' % v if v < 100 else '%.0f' % v
        e = int(np.floor(np.log10(v)))
        return r'$%.1f\times10^{%d}$' % (v / 10 ** e, e)


    def lam(v):
        return r'$\lambda=10^{%d}$' % int(round(np.log10(v)))


    L = [r'% generated by codes/04_exact_vs_proposed.py',
         r'\begin{tabular}{@{}llrrrrrrr@{}}', r'\toprule',
         r' & & & baseline & proposed & \multicolumn{2}{c}{$|E-E^{\mathrm{exact}}|$ (mHa)} & \multicolumn{2}{c}{time per integral (ms)} \\',
         r'\cmidrule(lr){6-7}\cmidrule(l){8-9}',
         r'quantity & parameter & exact & (see block) & (see block) & of baseline & of proposed & of baseline & of proposed \\', r'\midrule',
         r'\multicolumn{9}{@{}l}{\emph{(a) wire, $E_x=-(\phi\phi|\phi\phi)$.  Exact: eq~\ref{eq:dc-exwire};  baseline: aniGTO default (eq~\ref{S-eq:ani-eri-t}, adaptive in $t$);  proposed: aniGTO kernels in $s=\ln u$ (eq~\ref{eq:logvar-eri})}} \\']
    for _, r in iee[iee.family == 'wire'].iterrows():
        L.append(r'wire & %s & %.9f & %.9f & %.9f & %s & %s & %.0f & %.0f \\' % (
            lam(r.lam), r.exact, r.default, r.proposed, mha(r.err_default), mha(r.err_robust), r.ms_default, r.ms_robust))
    L.append(r'\multicolumn{9}{@{}l}{\emph{(b) sheet, $E_x=-(\phi\phi|\phi\phi)$.  Exact: eq~\ref{eq:dc-exsheet};  baseline and proposed as in (a)}} \\')
    for _, r in iee[iee.family == 'sheet'].iterrows():          # all lambda (2026-11-05; was lam >= 1e4)
        L.append(r'sheet & %s & %.9f & %.9f & %.9f & %s & %s & %.0f & %.0f \\' % (
            lam(r.lam), r.exact, r.default, r.proposed, mha(r.err_default), mha(r.err_robust), r.ms_default, r.ms_robust))
    L.append(r'\multicolumn{9}{@{}l}{\emph{(c) $N=2$ SCE, $V_{ee}^{\mathrm{SCE}}$.  Exact: eq~\ref{eq:sce-radial};  baseline: optimal transport (eq~\ref{eq:sce-axicost}), finest grid;  proposed: $h\to0$, eq~\ref{eq:sce-extrap}}} \\')
    for _, r in val.iterrows():
        w = r.case.split('_w')[1]
        L.append(r'Gaussian & $\omega=%s$ & %.9f & %.9f & %.9f & %s & %s & -- & -- \\' % (
            w, r.V_exact, r.finest, r.V_axisym, mha(r.finest - r.V_exact), mha(r.V_axisym - r.V_exact)))
    L += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT, 'tab_exact_vs_proposed.tex'), 'w').write('\n'.join(L) + '\n')

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({'font.size': 7.5, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0})
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.45))
    w = iee[iee.family == 'wire']
    lg = np.logspace(0, 8.3, 200)
    ax[0].semilogx(lg, [ex_exact('wire', x) for x in lg], '-', color='k', lw=1.0, label='exact (closed form)')
    ax[0].semilogx(w.lam, w.proposed, 'o', ms=5, mfc='none', mec='C2', mew=1.2, label='proposed (log variable)')
    ax[0].semilogx(w.lam, w.default, 'x', ms=5, color='C3', mew=1.2, label='aniGTO default')
    ax[0].set_xlabel(r'$\lambda$'); ax[0].set_ylabel(r'$E_x=-(\phi\phi|\phi\phi)$ (Ha)')
    ax[0].set_title('(a) wire: exact vs proposed', fontsize=8)
    ax[0].legend(frameon=False, fontsize=6.3, loc='lower left')

    floor = 1e-12
    for fam, mk in (('wire', 'o'), ('sheet', 's')):
        s = iee[iee.family == fam]
        ax[1].loglog(s.lam, np.maximum(1e3 * np.abs(s.err_default), floor), mk + '-', color='C3', ms=3.5,
                     label='%s, aniGTO default' % fam)
        ax[1].loglog(s.lam, np.maximum(1e3 * np.abs(s.err_robust), floor), mk + '--', color='C2', ms=3.5, mfc='none',
                     label='%s, proposed' % fam)
    ax[1].axhspan(1e-14, 1e-11, color='0.9', lw=0)
    ax[1].text(3e4, 4e-14, r'$\leq10^{-11}$ mHa (double precision)', fontsize=5.6, color='0.35', ha='center')
    ax[1].set_xlabel(r'$\lambda$'); ax[1].set_ylabel(r'$|E_x-E_x^{\mathrm{exact}}|$ (mHa)')
    ax[1].set_ylim(1e-14, 1e5)
    ax[1].set_title('(b) error of the repulsion integral', fontsize=8)
    ax[1].legend(frameon=False, fontsize=5.8, loc='center left')

    for (_, r), c in zip(val.iterrows(), ('C0', 'C1')):
        pts = [tuple(map(float, p.split(':'))) for p in r.ladder.split(';')]
        h2 = np.array([1.0 / n ** 2 for n, _ in pts])
        dv = np.array([1e3 * (v - r.V_exact) for _, v in pts])
        lab = r'Gaussian $\omega=%s$' % r.case.split('_w')[1]
        ax[2].plot(1e3 * h2, dv, 'o', color=c, ms=4, label=lab + ', grids')
        ax[2].plot([0, 1e3 * h2[-1]], [1e3 * (r.V_axisym - r.V_exact), dv[-1]], '--', color=c, lw=0.8)
        ax[2].plot([0], [1e3 * (r.V_axisym - r.V_exact)], '*', color=c, ms=8, mec='k', mew=0.4,
                   label=lab + ', extrapolated')
    ax[2].axhline(0, color='k', lw=0.8)
    ax[2].text(0.97, 0.07, 'exact radial SCE (= 0)', transform=ax[2].transAxes, fontsize=6.3, ha='right')
    ax[2].set_xlabel(r'$10^3\,h^2$ ($h=1/n_R$)'); ax[2].set_ylabel(r'$V_{ee}^{\mathrm{SCE}}-V^{\mathrm{exact}}$ (mHa)')
    ax[2].set_title(r'(c) $N=2$ SCE: exact vs proposed', fontsize=8)
    ax[2].legend(frameon=False, fontsize=5.8, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'dcscan_exact_vs_proposed.pdf'))
    fig.savefig(os.path.join(FIG, 'dcscan_exact_vs_proposed.png'), dpi=200)
    print(open(os.path.join(OUT, 'tab_exact_vs_proposed.tex')).read())


if __name__ == '__main__':
    print('== running: python3 %s   (outputs in %s/results and %s/figures)' % (os.path.basename(__file__),
          os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(__file__))), flush=True)
    iee_rows()
    sce_validation()
    make_table_and_figure()
