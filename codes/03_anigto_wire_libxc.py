#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd codes
#     python3 03_anigto_wire_libxc.py
#
# Needs: numpy, scipy, pyscf.  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""The wire benchmark evaluated with aniGTO ingredients: exact exchange from the aniGTO repulsion integral, density/gradient/Hessian/tau of the aniGTO trap s function, LDA/PBE/SCAN/r2SCAN from libxc.

STANDALONE SCRIPT (SCAN_1D_2D/codes): run it alone, from any directory,

    python3 03_anigto_wire_libxc.py

It needs only third-party packages (numpy, scipy, pyscf (libxc)); every project function it uses is defined in this file.
Outputs: results/anigto_dft_wire.csv, anigto_dft_wire_convergence.csv, anigto_dft_wire_log.txt, tab_dcscan_wire.tex (full main-text wire table), tab_dcscan_anigto_grid_rows.tex
Manuscript items: main-text wire table, SI Table S4

---------------------------------------------------------------------------------------------------------------
DC-SCAN wire benchmark recomputed entirely inside the aniGTO DFT machinery.

    python3 03_anigto_wire_libxc.py [--ngrid 80 160 240] [--lams 1 2 10 100 1000 10000]

For each confinement lam the two-electron Gaussian wire is the aniGTO harmonic-trap ('dot') basis with
omega = (lam^2, lam^2, 1): its s function has exponents (lam^2/2, lam^2/2, 1/2) and is exactly the orbital
phi = sqrt(n_lam/2).  Everything below uses the aniGTO formulas (SI Section S1), defined in this file:

  exact  E_x = -(phi phi|phi phi)       aniGTO repulsion integral IEE (default quadrature)
  exact  E_x, Kantorovich route         -1/4 int int n(z) n(z') w_b(z-z'), the quasi-1D interaction of
                                        w_b below with b = 1/(sqrt(2) lam) (exact here,
                                        because the transverse factor is the same Gaussian for both electrons)
  rho, grad rho, Hessian, laplacian, tau on the anisotropic Gauss-Hermite grid of the aniGTO DFT driver,
                                        from analytic derivatives of the aniGTO basis functions
  LDA, PBE, SCAN, r2SCAN exchange       libxc through PySCF (exchange only: LDA_X, GGA_X_PBE, MGGA_X_SCAN,
                                        MGGA_X_R2SCAN), F_x = e_x / e_x^LDA pointwise
  DC-SCAN                               eq. (20)-(25) of the note with the libxc SCAN F_x and the aniGTO
                                        density Hessian (D, Z from the grid, not from the closed form)
  DC-PBE, DC-r2SCAN                     same crossover on a PBE or r2SCAN base (no refit: A2D, A1D unchanged)
Outputs: results/anigto_dft_wire.csv, results/anigto_dft_wire_convergence.csv, results/anigto_dft_wire_log.txt
"""
import argparse
import numpy as np
import os
import sys
import time
import warnings
from numpy.polynomial.hermite import hermgauss
from pyscf.dft import numint
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal
from scipy.special import comb
from scipy.special import erfcx
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
# Transverse-averaged (quasi-1D Kantorovich) interaction w_b(z) = sqrt(pi)/(2b) erfcx(|z|/(2b))
# ==============================================================================================================
def w_b(x, b):
    return np.sqrt(np.pi) / (2.0 * b) * erfcx(np.abs(x) / (2.0 * b))

# ==============================================================================================================
# Main calculation
# ==============================================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

NI = numint.NumInt()
KEYS = ('LDA', 'PBE', 'SCAN', 'SCAN_closed_form', 'r2SCAN', 'DC-SCAN', 'DC-PBE', 'DC-r2SCAN')
XC = {'LDA': 'LDA_X', 'PBE': 'GGA_X_PBE', 'SCAN': 'MGGA_X_SCAN', 'r2SCAN': 'MGGA_X_R2SCAN'}


class TrapS:
    """The s function of the aniGTO harmonic-trap ('dot') basis for omega = (wx, wy, wz): a normalized anisotropic
    Gaussian with exponents omega/2, i.e. phi(r) = prod_k (omega_k/pi)^(1/4) exp(-omega_k x_k^2 / 2).
    Attributes follow the aniGTO BasisFunction object (exps, shell, origin, coefs, norm, num_coefs)."""

    def __init__(self, omega):
        a = 0.5 * np.asarray(omega, dtype=float)
        self.exps = a.reshape(1, 3)
        self.shell = (0, 0, 0)
        self.origin = (0.0, 0.0, 0.0)
        self.coefs = np.array([1.0])
        self.norm = np.array([np.prod((2.0 * a / np.pi) ** 0.25)])
        self.num_coefs = 1


class TrapMolecule:
    """Minimal stand-in for aniGTO Molecule(basis='dot', n_orbitals={'S': 1}): one trap s function."""

    def __init__(self, omega):
        self.bfs = [TrapS(omega)]


def build_wire(lam, workdir=None):
    """Wire trap omega = (lam^2, lam^2, 1); returns (molecule, exact E_x = -(phi phi|phi phi) from aniGTO IEE)."""
    mol = TrapMolecule([lam ** 2, lam ** 2, 1.0])
    b = mol.bfs[0]
    a, L, O = tuple(b.exps[0]), (0, 0, 0), (0.0, 0.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        eri = b.norm[0] ** 4 * IEE(a, L, O, a, L, O, a, L, O, a, L, O)
    assert np.allclose(b.exps[0], [lam ** 2 / 2, lam ** 2 / 2, 0.5]), b.exps
    return mol, -eri


def eval_bf_hess(b, xyz):
    """Value, gradient (3), Hessian (3x3) of one contracted aniGTO Cartesian Gaussian."""
    r = (xyz - np.asarray(b.origin).reshape(-1, 3)[0]).T
    sh = [int(v) for v in np.asarray(b.shell).reshape(-1, 3)[0]]
    val = np.zeros(r.shape[1])
    grad = np.zeros((3, r.shape[1]))
    hess = np.zeros((3, 3, r.shape[1]))
    for i in range(b.num_coefs):
        a = b.exps[i]
        c = b.norm[i] * b.coefs[i]
        f0, f1, f2 = [], [], []
        for k in range(3):                               # 1D factors t^l e^{-a t^2} and two derivatives
            t, l, ak = r[k], sh[k], a[k]
            e = np.exp(-ak * t * t)
            p = t ** l
            p1 = (l * t ** (l - 1) if l > 0 else 0.0) - 2 * ak * t ** (l + 1)
            p2 = ((l * (l - 1) * t ** (l - 2) if l > 1 else 0.0) - 2 * ak * (2 * l + 1) * t ** l
                  + 4 * ak * ak * t ** (l + 2))
            f0.append(p * e); f1.append(p1 * e); f2.append(p2 * e)
        val += c * f0[0] * f0[1] * f0[2]
        for k in range(3):
            o = [f0[j] for j in range(3)]
            o[k] = f1[k]
            grad[k] += c * o[0] * o[1] * o[2]
            for m in range(3):
                o = [f0[j] for j in range(3)]
                if k == m:
                    o[k] = f2[k]
                else:
                    o[k], o[m] = f1[k], f1[m]
                hess[k, m] += c * o[0] * o[1] * o[2]
    return val, grad, hess


def gh_grid(mol, ngrid):
    """Anisotropic Gauss-Hermite product grid of the aniGTO DFT driver (scale 2*min exponent per axis).

    Only the octant x, y, z >= 0 is kept (the density is even in each coordinate): an even number of nodes
    gives no node on a symmetry plane, and the weights are multiplied by 8."""
    emin = np.array([min(b.exps[:, k].min() for b in mol.bfs) for k in range(3)])
    axes, wts = [], []
    for k in range(3):
        xi, wi = hermgauss(ngrid)
        a = 2.0 * emin[k]
        m = xi > 0
        axes.append(xi[m] / np.sqrt(a))
        wts.append(wi[m] * np.exp(xi[m] ** 2) / np.sqrt(a))
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    w = 8.0 * np.einsum('i,j,k->ijk', *wts).ravel()
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1), w


def cyl_grid(mol, n=400, u_range=(np.exp(-32.0), 26.0), t_range=(np.exp(-32.0), 8.0)):
    """Cylindrical log grid for omega_x = omega_y traps: points (rho, 0, z) of the aniGTO basis functions.

    u = 2 a_perp rho^2, t = sqrt(2 a_z) z (a = s-exponent of the trap basis; u = lam^2 rho^2, t = z for the
    wire), log Gauss-Legendre in u and t as in eq. (33); d^3r = 2 pi rho drho * 2 dz over z >= 0."""
    a = mol.bfs[0].exps[0]
    assert abs(a[0] - a[1]) < 1e-12 * a[0], 'cylindrical grid needs omega_x = omega_y'
    u, wu = _log_gl(*u_range, n)
    t, wt = _log_gl(*t_range, n)
    U, T = np.meshgrid(u, t, indexing='ij')
    rho_c = np.sqrt(U / (2.0 * a[0]))
    z = T / np.sqrt(2.0 * a[2])
    coords = np.stack([rho_c.ravel(), np.zeros(U.size), z.ravel()], axis=1)
    w = (2.0 * np.pi / (4.0 * a[0])) * (2.0 / np.sqrt(2.0 * a[2])) * np.outer(wu, wt).ravel()
    return coords, w


def density_fields(mol, coords):
    """rho, grad rho, Hessian of rho, laplacian, tau for the doubly occupied aniGTO s orbital."""
    phi, g, h = eval_bf_hess(mol.bfs[0], coords)
    rho = 2.0 * phi * phi
    grad = 4.0 * phi * g
    hess = 4.0 * (g[:, None, :] * g[None, :, :] + phi * h)
    lap = np.trace(hess)
    tau = np.sum(g * g, axis=0)                          # 1/2 sum_occ f |grad phi|^2 with f = 2
    return rho, grad, hess, lap, tau


def exchange_on_grid(rho, grad, hess, lap, tau, w):
    keep = rho > 1e-30
    rho, grad, hess, lap, tau, w = rho[keep], grad[:, keep], hess[:, :, keep], lap[keep], tau[keep], w[keep]
    inp = np.vstack((rho, grad, lap, tau))
    ex = {}
    for k, code in XC.items():
        x = inp[:1] if k == 'LDA' else (inp[:4] if k == 'PBE' else inp)
        ex[k] = np.asarray(NI.eval_xc(code + ',', x, spin=0, deriv=0)[0])
    eps_lda = ex['LDA']
    ok = eps_lda != 0.0                                  # libxc returns 0 below its density threshold
    fx = {k: np.where(ok, ex[k] / np.where(ok, eps_lda, 1.0), 1.0) for k in ex}
    # DC-SCAN descriptors from the aniGTO density Hessian: C = -1/2 d d ln n = -H/(2n) + g g^T/(2 n^2)
    C = -hess / (2.0 * rho) + grad[:, None, :] * grad[None, :, :] / (2.0 * rho * rho)
    C2 = np.einsum('ijg,jkg->ikg', C, C)
    trc2 = np.einsum('iig->g', C2)
    gnorm2 = np.sum(grad * grad, axis=0)
    gc2g = np.einsum('ig,ijg,jg->g', grad, C2, grad)
    with np.errstate(invalid='ignore', divide='ignore'):
        D = np.where(gc2g > 1e-300 * np.maximum(gnorm2 * trc2, 1e-300), gnorm2 * trc2 / gc2g, 3.0)
    D = np.clip(np.nan_to_num(D, nan=3.0), 1.0, 3.0)
    kf = KF * np.cbrt(rho)
    s = np.sqrt(gnorm2) / (2.0 * kf * rho)
    Z = np.sqrt(np.maximum(trc2, 0.0)) / kf ** 2
    tau_w = gnorm2 / (8.0 * rho)
    alpha = (tau - tau_w) / (0.3 * kf ** 2 * rho)
    wl = w * rho * eps_lda
    out = {k: float(np.sum(wl * fx[k])) for k in fx}
    for base in ('SCAN', 'PBE', 'r2SCAN'):
        out['DC-' + base] = float(np.sum(wl * fx_dcscan(s, D, Z, fx[base], 'full')))
    # closed-form SCAN alpha=0 on the same grid: isolates libxc vs note eq. (6)
    out['SCAN_closed_form'] = float(np.sum(wl * fx_scan_alpha0(s)))
    big = rho > 1e-6 * rho.max()                         # alpha is roundoff-dominated in the far tail
    diag = dict(max_abs_alpha=float(np.max(np.abs(alpha[big]))), D_median=float(np.median(D)),
                N=float(np.sum(w * rho)))
    return out, diag


def kantorovich_exact(lam, L=12.0):
    """-1/4 int int n(z) n(z') w_b(z - z') dz dz' with n(z) = 2 pi^-1/2 e^{-z^2}, b = 1/(sqrt2 lam)."""
    b = 1.0 / (np.sqrt(2.0) * lam)
    # z = R + r/2, z' = R - r/2: z^2 + z'^2 = 2 R^2 + r^2/2, dz dz' = dR dr; the R integral is analytic.
    # r integral of e^{-r^2/2} w_b(r) on log-graded Gauss-Legendre panels (w_b peaks on the scale b).
    edges = np.concatenate([[0.0], np.logspace(np.log10(b) - 6, np.log10(L), 60)])
    xg, wg = np.polynomial.legendre.leggauss(64)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        r = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        tot += np.sum(0.5 * (hi - lo) * wg * np.exp(-r * r / 2.0) * w_b(r, b))
    ir = 2.0 * tot                                        # even in r
    iR = np.sqrt(np.pi / 2.0)                             # int e^{-2 R^2} dR
    return -0.25 * (4.0 / np.pi) * iR * ir


def write_table(rows, res, refs):
    """Main-text wire table (user 2026-11-05: all computed values first, differences after), one tabular, 3 blocks:
    (a) values: exact three ways, reference-quadrature LDA/SCAN/DC-SCAN/DC-SCAN_min, aniGTO+libxc functionals;
    (b) errors against the exact Eq. (11) value, same columns;
    (c) implementation checks: aniGTO minus reference for the same quantity."""
    NC = 14
    hdr_a = ('$\\lambda$ & eq~\\ref{eq:dc-exwire} & aniGTO, eq~\\ref{S-eq:ani-eri-t} & Kantorovich, eq~\\ref{S-eq:ani-kantorovich} & LDA & SCAN & DC-SCAN & DC-SCAN$_{\\min}$ & '
             'SCAN & DC-SCAN & PBE & DC-PBE & r$^2$SCAN & DC-r$^2$SCAN \\\\')
    L = ['% generated by codes/03_anigto_wire_libxc.py (reference quadrature and aniGTO columns in one run)',
         '% layout (user 2026-11-05): (a) computed values, (b) errors vs exact Eq. (11), (c) aniGTO - reference checks',
         '\\begin{tabular}{@{}r|rrr|rrrr|rrrrrr@{}}', '\\toprule',
         '\\multicolumn{%d}{@{}l}{\\textit{(a) Computed exchange energies (Ha)}} \\\\' % NC,
         ' & \\multicolumn{3}{c|}{exact} & \\multicolumn{4}{c|}{reference quadrature} & '
         '\\multicolumn{6}{c}{aniGTO basis (eq~\\ref{S-eq:ani-density}), fields (eq~\\ref{S-eq:ani-fields}), grid (eq~\\ref{S-eq:ani-cyl}), libxc} \\\\', hdr_a, '\\midrule']
    for r in rows:
        ex, q = refs[r[0]]
        L.append(' & '.join([f'{int(r[0]):,}'] + ['%.6f' % v for v in (
            ex, r[2], r[3], q['LDA'], q['SCAN'], q['DC-SCAN'], q['DC-SCAN-min'], r[7], r[10], r[11], r[12], r[13], r[14])])
                 + ' \\\\')
    L += ['\\midrule', '\\multicolumn{%d}{@{}l}{\\textit{(b) Error against the exact value of eq~\\ref{eq:dc-exwire}, $E_x-E_x^{\\rm exact}$ (Ha)}} \\\\' % NC,
          hdr_a, '\\midrule']
    for r in rows:
        ex, q = refs[r[0]]
        L.append(' & '.join([f'{int(r[0]):,}', '0'] + ['%+.6f' % (v - ex) for v in (
            r[2], r[3], q['LDA'], q['SCAN'], q['DC-SCAN'], q['DC-SCAN-min'], r[7], r[10], r[11], r[12], r[13], r[14])])
                 + ' \\\\')
    L += ['\\midrule', '\\multicolumn{%d}{@{}l}{\\textit{(c) Implementation checks: aniGTO minus reference for the same quantity (Ha)}} \\\\' % NC,
          '$\\lambda$ & \\multicolumn{3}{c|}{exact: aniGTO$-$eq~\\ref{eq:dc-exwire} \\quad Kant.$-$eq~\\ref{eq:dc-exwire}} & LDA & SCAN & DC-SCAN & & '
          '\\multicolumn{6}{l}{(aniGTO+libxc $-$ reference quadrature)} \\\\', '\\midrule']
    for r in rows:
        ex, q = refs[r[0]]
        L.append(' & '.join([f'{int(r[0]):,}', '\\multicolumn{3}{c|}{%+.1e \\qquad %+.1e}' % (r[2] - ex, r[3] - ex),
                             '%+.1e' % (r[5] - q['LDA']), '%+.1e' % (r[7] - q['SCAN']), '%+.1e' % (r[10] - q['DC-SCAN']),
                             '', '\\multicolumn{6}{l}{}']) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(res, 'tab_dcscan_wire.tex'), 'w').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lams', type=float, nargs='+', default=[1, 2, 10, 100, 1000, 10000])
    ap.add_argument('--ngrid', type=int, nargs='+', default=[80, 160, 240],
                    help='Gauss-Hermite nodes per axis of the aniGTO DFT grid (only the positive half is used)')
    ap.add_argument('--ncyl', type=int, nargs='+', default=[300, 400, 560],
                    help='log Gauss-Legendre nodes per variable of the cylindrical grid')
    a = ap.parse_args()
    res = os.path.join(HERE, 'results')
    os.makedirs(res, exist_ok=True)
    t0 = time.time()
    rows, conv, lines, refs = [], [], [], {}
    for lam in a.lams:
        mol, ex_eri = build_wire(lam)
        ex_k = kantorovich_exact(lam)
        ex_cf = float(ex_exact('wire', lam))
        ref = ex_family('wire', lam, n=400)
        refs[lam] = (ex_cf, ref)
        gh = {}
        for ng in a.ngrid:
            coords, w = gh_grid(mol, ng)
            out, diag = exchange_on_grid(*density_fields(mol, coords), w)
            conv.append([lam, 'GH', ng, diag['N']] + [out[k] for k in KEYS])
            gh = out
        for nc in a.ncyl:
            coords, w = cyl_grid(mol, nc)
            out, diag = exchange_on_grid(*density_fields(mol, coords), w)
            conv.append([lam, 'cyl', nc, diag['N']] + [out[k] for k in KEYS])
        ng = nc
        rows.append([lam, ex_cf, ex_eri, ex_k, ref['LDA'], out['LDA'], ref['SCAN'], out['SCAN'],
                     out['SCAN_closed_form'], ref['DC-SCAN'], out['DC-SCAN'], out['PBE'], out['DC-PBE'],
                     out['r2SCAN'], out['DC-r2SCAN'], diag['N'], diag['max_abs_alpha'], ng,
                     gh['SCAN'], gh['DC-SCAN']])
        lines.append(('lam=%-7g VALUES  exact Eq.11 %.6f | aniGTO ERI %.6f | Kantorovich %.6f | LDA ani %.6f quad %.6f | '
                      'SCAN ani %.6f quad %.6f | DC-SCAN ani %.6f quad %.6f | PBE %.6f DC-PBE %.6f | r2SCAN %.6f DC-r2SCAN %.6f'
                      '\n            DIFFS   ERI-exact %+.1e | Kant-exact %+.1e | LDA ani-quad %+.1e | SCAN ani-quad %+.1e | '
                      'DC-SCAN ani-quad %+.1e | SCAN-exact %+.6f | DC-SCAN-exact %+.6f | DC-PBE-exact %+.6f | '
                      'N=%.8f max|alpha|=%.1e')
                     % (lam, ex_cf, ex_eri, ex_k, out['LDA'], ref['LDA'], out['SCAN'], ref['SCAN'], out['DC-SCAN'],
                        ref['DC-SCAN'], out['PBE'], out['DC-PBE'], out['r2SCAN'], out['DC-r2SCAN'],
                        ex_eri - ex_cf, ex_k - ex_cf, out['LDA'] - ref['LDA'], out['SCAN'] - ref['SCAN'],
                        out['DC-SCAN'] - ref['DC-SCAN'], out['SCAN'] - ex_cf, out['DC-SCAN'] - ex_cf,
                        out['DC-PBE'] - ex_cf, diag['N'], diag['max_abs_alpha']))
        print(lines[-1], flush=True)
    hdr = ['lambda', 'E_exact', 'E_exact_aniGTO_ERI', 'E_exact_Kantorovich', 'LDA_quad', 'LDA_aniGTO',
           'SCAN_quad', 'SCAN_aniGTO_libxc', 'SCAN_closedform_aniGTO_grid', 'DCSCAN_quad', 'DCSCAN_aniGTO',
           'PBE_aniGTO', 'DCPBE_aniGTO', 'r2SCAN_aniGTO', 'DCr2SCAN_aniGTO', 'N_grid', 'max_abs_alpha',
           'ncyl', 'SCAN_aniGTO_GHgrid', 'DCSCAN_aniGTO_GHgrid']
    with open(os.path.join(res, 'anigto_dft_wire.csv'), 'w') as f:
        f.write(','.join(hdr) + '\n')
        for r in rows:
            f.write(','.join('%.15g' % v for v in r) + '\n')
    with open(os.path.join(res, 'anigto_dft_wire_convergence.csv'), 'w') as f:
        f.write('lambda,grid,nodes,N,' + ','.join(KEYS) + '\n')
        for r in conv:
            f.write(','.join(v if isinstance(v, str) else '%.15g' % v for v in r) + '\n')
    write_table(rows, res, refs)
    with open(os.path.join(res, 'tab_dcscan_anigto_grid_rows.tex'), 'w') as f:   # SI table body (values, then diffs)
        for r in conv:
            if r[0] >= 100 or r[0] == 10:
                o = dict(zip(KEYS, r[4:]))
                q = refs[r[0]][1]
                f.write(' & '.join([f'{int(r[0]):,}', 'Gauss--Hermite' if r[1] == 'GH' else 'cylindrical log', '%d' % r[2],
                                    '%.8f' % r[3]] + ['%.6f' % o[k] for k in ('LDA', 'PBE', 'SCAN', 'r2SCAN', 'DC-SCAN',
                                                                         'DC-PBE', 'DC-r2SCAN')]
                                   + ['%+.1e' % (o['LDA'] - q['LDA']), '%+.1e' % (o['SCAN'] - q['SCAN']),
                                      '%+.1e' % (o['DC-SCAN'] - q['DC-SCAN']), '%+.1e' % (o['SCAN_closed_form'] - o['SCAN'])])
                        + ' \\\\\n')
    lines.append('total %.1f s' % (time.time() - t0))
    open(os.path.join(res, 'anigto_dft_wire_log.txt'), 'w').write('\n'.join(lines) + '\n')
    print(lines[-1])


if __name__ == '__main__':
    print('== running: python3 %s   (outputs in %s/results and %s/figures)' % (os.path.basename(__file__),
          os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(__file__))), flush=True)
    main()
