#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd codes
#     python3 02_sheet_benchmark.py
#
# Needs: numpy, scipy, matplotlib.  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""DC-SCAN on the two-electron Gaussian sheet (quasi-2D): main-text sheet table, SI checks, approach to the lam -> infinity limits and the sheet figure.

STANDALONE SCRIPT (SCAN_1D_2D/codes): run it alone, from any directory,

    python3 02_sheet_benchmark.py

It needs only third-party packages (numpy, scipy, matplotlib); every project function it uses is defined in this file.
Outputs: results/sheet_table.csv, sheet_convergence.csv, sheet_exact_crosscheck.csv, sheet_asymptotics.csv, sheet_curve.csv, tab_dcscan_sheet_reference.tex, tab_dcscan_sheet_check_rows.tex, tab_dcscan_sheet_asym_rows.tex, run_log_sheet.txt; figures/sheet_benchmark.pdf/.png
Manuscript items: main-text sheet table, SI Tables S5-S6, SI Figure S2

---------------------------------------------------------------------------------------------------------------
2D (sheet) DC-SCAN benchmark: reproduces Table 1 and Figure 1 of the DC-SCAN note, plus checks.

    python3 02_sheet_benchmark.py            # writes results/ and figures/ next to this script

Density family (Eq. 7): n_lam(x,y,z) = 2 lam pi^(-3/2) exp(-(x^2 + y^2 + lam^2 z^2)), N = 2, alpha = 0.
Exact E_x (Eq. 10) is finite as lam -> inf: -sqrt(pi/2) (the "finite sheet limit" fixing A2D).
On the alpha = 0 branch SCAN also stays finite here: F_x^SCAN -> h0x a1 s^(-1/2) at large s, so with the
same integral M2D that fixes A2D,  E_x^SCAN(lam -> inf) = -h0x a1 M2D  (about -11.15 Ha, 8.9x the exact limit).
Outputs
  results/sheet_table.csv, tab_dcscan_sheet_reference.tex  Table 1: exact, LDA, SCAN, DC-SCAN, DC-exact,
                                     DC-SCAN_min - exact at lam = 1 ... 1e4 (400 GL nodes)
  results/sheet_curve.csv           Figure 1 data: 39 log-spaced lam, 300 GL nodes
  results/sheet_convergence.csv     400 vs 560 nodes, plus wider u/t ranges
  results/sheet_exact_crosscheck.csv, tab_dcscan_sheet_check_rows.tex  exact E_x three ways
  results/sheet_asymptotics.csv, tab_dcscan_sheet_asym_rows.tex  approach to the lam -> inf limits
  results/run_log.txt
  figures/sheet_benchmark.pdf/.png
"""
import numpy as np
import os
import platform
import sys
import time
import warnings
from scipy.integrate import quad
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
# Main calculation
# ==============================================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
FAMILY = 'sheet'
TABLE_LAMS = [1.0, 2.0, 10.0, 100.0, 1000.0, 10000.0]
NOTE_TABLE1 = {  # values printed in the note (DC_SCAN_detailed_theory_results.pdf, Table 1), hartree
    1.0: (-0.797885, -0.795058, -0.795058),
    2.0: (-0.964802, -1.000228, -0.869377),
    10.0: (-1.179303, -1.672690, -1.082995),
    100.0: (-1.245397, -3.179886, -1.184779),
    1000.0: (-1.252517, -5.130958, -1.221408),
    10000.0: (-1.253234, -7.045312, -1.238506),
}
E_SHEET_LIMIT = -np.sqrt(np.pi / 2.0)
E_SCAN_LIMIT = -SCAN_H0X * SCAN_A1 * m2d()


def write_csv(path, header, rows, fmt='%.15g'):
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(fmt % v if isinstance(v, float) else str(v) for v in r) + '\n')


def sci(v):
    """LaTeX scientific notation for small differences."""
    if v == 0:
        return '$0$'
    e = int(np.floor(np.log10(abs(v))))
    return '$%.1f\\times10^{%d}$' % (v / 10 ** e, e)


def anigto_exact(lam, high_limit=False):
    """E_x = -(phi phi|phi phi) from the aniGTO primitive ERI, phi = sqrt(n/2), exponents A/2."""
    import warnings
    from scipy.integrate import quad
    A = np.array(diag_A(FAMILY, lam))
    a = tuple(A / 2.0)
    O, L = (0.0, 0.0, 0.0), (0, 0, 0)
    norm4 = np.prod(A) / np.pi ** 3                       # N^4, N^2 = sqrt(det A) pi^(-3/2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if not high_limit:
            eri = IEE(a, L, O, a, L, O, a, L, O, a, L, O)
        else:                                             # same aniGTO integrand, more adaptive subintervals
            val, _ = quad(integrand2, 0.0, 1.0, (a, L, O, a, L, O, a, L, O, a, L, O),
                          epsrel=1e-14, epsabs=1e-14, limit=2000,
                          points=[1.0 - 10.0 ** (-k) for k in range(1, 10)])
            eri = 2.0 / np.sqrt(np.pi) * val
    return -norm4 * eri


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    log = []
    t0 = time.time()

    # ---- Table 1 ---------------------------------------------------------------------------------
    rows, worst = [], 0.0
    for lam in TABLE_LAMS:
        r = ex_family(FAMILY, lam, n=400)
        ex = float(ex_exact(FAMILY, lam))
        rows.append([lam, ex, r['LDA'], r['SCAN'], r['DC-SCAN'], r['DC-SCAN'] - ex, r['DC-SCAN-min'],
                     r['DC-SCAN-min'] - ex])
        ref = NOTE_TABLE1[lam]
        worst = max(worst, abs(round(ex, 6) - ref[0]), abs(round(r['SCAN'], 6) - ref[1]),
                    abs(round(r['DC-SCAN'], 6) - ref[2]))
    write_csv(os.path.join(RES, 'sheet_table.csv'),
              ['lambda', 'E_exact', 'E_LDA', 'E_SCAN', 'E_DCSCAN', 'DCSCAN_minus_exact', 'E_DCSCAN_min',
               'DCSCANmin_minus_exact'], rows)
    with open(os.path.join(RES, 'tab_dcscan_sheet_reference.tex'), 'w') as f:
        f.write('% generated by codes/02_sheet_benchmark.py; Gaussian sheet, exchange energies in hartree\n')
        f.write('\\begin{tabular}{@{}rrrrrrr@{}}\n\\toprule\n'
                '$\\lambda$ & Exact & LDA & SCAN & DC-SCAN & DC-SCAN$-$exact & DC-SCAN$_{\\min}-$exact \\\\\n\\midrule\n')
        for r in rows:
            f.write('%s & %.6f & %.6f & %.6f & %.6f & %+.6f & %+.6f \\\\\n'
                    % (f'{int(r[0]):,}', r[1], r[2], r[3], r[4], r[5], r[7]))
        f.write('\\bottomrule\n\\end{tabular}\n')
    log.append(f'Table 1 regression vs note (6-decimal values): max |diff| = {worst:.1e} Ha')

    # ---- convergence: nodes and integration ranges ------------------------------------------------
    conv, dcv = [], {}
    for lam in TABLE_LAMS:
        a = ex_family(FAMILY, lam, n=400)
        b = ex_family(FAMILY, lam, n=560)
        c = ex_family(FAMILY, lam, n=560, u_range=(np.exp(-40.0), 34.0), t_range=(np.exp(-40.0), 10.0))
        conv.append([lam, b['SCAN'] - a['SCAN'], b['DC-SCAN'] - a['DC-SCAN'],
                     c['SCAN'] - b['SCAN'], c['DC-SCAN'] - b['DC-SCAN']])
        dcv[lam] = (a['DC-SCAN'], b['DC-SCAN'])          # DC-SCAN values for the SI check table (11-05)
    write_csv(os.path.join(RES, 'sheet_convergence.csv'),
              ['lambda', 'SCAN_560_minus_400', 'DCSCAN_560_minus_400', 'SCAN_wide_minus_560',
               'DCSCAN_wide_minus_560'], conv)
    log.append('quadrature 400 -> 560 nodes: max |dE(DC-SCAN)| = %.1e, |dE(SCAN)| = %.1e Ha; '
               'wider u/t ranges: %.1e / %.1e Ha'
               % (max(abs(c[2]) for c in conv), max(abs(c[1]) for c in conv),
                  max(abs(c[4]) for c in conv), max(abs(c[3]) for c in conv)))

    # ---- exact reference cross-check ---------------------------------------------------------------
    xc = []
    for lam in TABLE_LAMS:
        e10 = float(ex_exact(FAMILY, lam))
        e44 = ex_exact_eq44(diag_A(FAMILY, lam))
        ea = anigto_exact(lam)
        eh = anigto_exact(lam, high_limit=True)
        xc.append([lam, e10, e44, ea, eh, e44 - e10, ea - e10, eh - e10])
    write_csv(os.path.join(RES, 'sheet_exact_crosscheck.csv'),
              ['lambda', 'Eq10', 'Eq44_quadrature', 'aniGTO_IEE', 'aniGTO_integrand_quad_limit2000',
               'Eq44_minus_Eq10', 'aniGTO_minus_Eq10', 'aniGTO_hl_minus_Eq10'], xc)
    convd = {c[0]: (dcv[c[0]][0], dcv[c[0]][1], c[2]) for c in conv}   # DC-SCAN 400, 560, 560-400 (11-05)
    with open(os.path.join(RES, 'tab_dcscan_sheet_check_rows.tex'), 'w') as f:
        for x in xc:   # values first (exact, eq A44, aniGTO, aniGTO fine, DC-SCAN 400/560), then differences (user 11-05)
            d4, d5, dd = convd[x[0]]
            f.write('%s & %.10f & %.10f & %.10f & %.10f & %.10f & %.10f & %s & %s & %s & %s \\\\\n'
                    % (f'{int(x[0]):,}', x[1], x[2], x[3], x[4], d4, d5, sci(x[5]), sci(x[6]), sci(x[7]), sci(dd)))
    log.append('exact E_x: max |Eq44 - Eq10| = %.1e; aniGTO IEE: max |diff| = %.1e (lam <= 1e3), %.1e (lam = 1e4); '
               'finer quadrature of the aniGTO integrand: max %.1e Ha'
               % (max(abs(x[5]) for x in xc), max(abs(x[6]) for x in xc[:-1]), abs(xc[-1][6]),
                  max(abs(x[7]) for x in xc)))

    # ---- asymptotics: approach to the finite lam -> inf limits ------------------------------------
    lams = np.logspace(2, 6, 9)
    asym, avals = [], []
    for lam in lams:
        r = ex_family(FAMILY, lam, n=400)
        ex = float(ex_exact(FAMILY, lam))
        asym.append([lam, ex - E_SHEET_LIMIT, r['DC-SCAN'] - E_SHEET_LIMIT, r['DC-SCAN'] - ex,
                     r['SCAN'] - E_SCAN_LIMIT])
        avals.append((ex, r['DC-SCAN'], r['SCAN']))                # values for the SI table (values first, 11-05)
    asym = np.array(asym)
    ll = np.log(asym[:, 0])
    pw = lambda col: np.diff(np.log(np.abs(asym[:, col]))) / np.diff(ll)   # noqa: E731
    p_exact, p_dc, p_scan = pw(1), pw(3), pw(4)
    write_csv(os.path.join(RES, 'sheet_asymptotics.csv'),
              ['lambda', 'Eexact_minus_limit', 'EDCSCAN_minus_limit', 'DCSCAN_minus_exact', 'ESCAN_minus_SCANlimit'],
              [list(map(float, a)) for a in asym])
    with open(os.path.join(RES, 'tab_dcscan_sheet_asym_rows.tex'), 'w') as f:
        for a, v in zip(asym, avals):
            e = np.log10(a[0])
            lab = '%d' % round(e) if abs(e - round(e)) < 1e-9 else '%.1f' % e
            f.write('$10^{%s}$ & %.6f & %.6f & %.6f & %s & %+.6f & %+.4f \\\\\n'
                    % (lab, v[0], v[1], v[2], sci(a[1]), a[3], a[4]))
    log.append('lam -> inf limits: exact -sqrt(pi/2) = %.6f, SCAN -h0x a1 M2D = %.6f (ratio %.2f)'
               % (E_SHEET_LIMIT, E_SCAN_LIMIT, E_SCAN_LIMIT / E_SHEET_LIMIT))
    log.append('approach exponents (last decade): exact-limit ~ lam^%.3f, DC-exact ~ lam^%.3f, SCAN-limit ~ lam^%.3f'
               % (p_exact[-1], p_dc[-1], p_scan[-1]))
    log.append('DC-SCAN - exact at lam = 1e5, 1e6: %+.6f, %+.6f Ha' % (asym[-3, 3], asym[-1, 3]))

    # ---- Figure 1 curve ----------------------------------------------------------------------------
    curve = []
    for lam in np.logspace(0, 4, 39):
        r = ex_family(FAMILY, lam, n=300)
        curve.append([lam, float(ex_exact(FAMILY, lam)), r['SCAN'], r['DC-SCAN'], r['DC-SCAN-min']])
    write_csv(os.path.join(RES, 'sheet_curve.csv'), ['lambda', 'E_exact', 'E_SCAN', 'E_DCSCAN', 'E_DCSCAN_min'],
              curve)
    plot(np.array(curve))

    log.append(f'total time {time.time() - t0:.1f} s')
    head = [f'DC-SCAN 2D (sheet) benchmark  {time.strftime("%Y-%m-%d %H:%M")}',
            f'python {platform.python_version()}  numpy {np.__version__}  host {platform.node()}',
            f'A2D = {A2D:.16f} (note {A2D_NOTE})  M2D = {m2d():.15f}', '']
    table = ['lambda      exact        SCAN          DC-SCAN      DC-exact    DC-SCAN(min)  min-exact']
    table += ['%-8g %11.6f %13.6f %12.6f %+10.6f %12.6f %+10.6f' % (r[0], r[1], r[3], r[4], r[5], r[6], r[7])
              for r in rows]
    text = '\n'.join(head + table + [''] + log) + '\n'
    open(os.path.join(RES, 'run_log_sheet.txt'), 'w').write(text)
    print(text)
    return 0 if worst <= 1.5e-6 else 1


def plot(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(c[:, 0], c[:, 1], 'k-', lw=2, label='Exact two-electron exchange')
    ax.semilogx(c[:, 0], c[:, 2], 'C3--', lw=1.6, label='SCAN')
    ax.semilogx(c[:, 0], c[:, 3], 'C0-', lw=1.6, label='DC-SCAN candidate')
    ax.axhline(E_SHEET_LIMIT, color='0.5', lw=0.8, ls=':')
    ax.set_xlabel(r'Confinement parameter $\lambda$')
    ax.set_ylabel('Exchange energy (Ha)')
    ax.legend(loc='lower left', frameon=False)
    ins = ax.inset_axes([0.47, 0.50, 0.48, 0.42])
    ins.semilogx(c[:, 0], c[:, 1], 'k-', lw=1.8, label='Exact')
    ins.semilogx(c[:, 0], c[:, 3], 'C0-', lw=1.4, label='DC-SCAN')
    ins.axhline(E_SHEET_LIMIT, color='0.5', lw=0.8, ls=':')
    ins.set_title('Exact and DC-SCAN (magnified)', fontsize=8)
    ins.tick_params(labelsize=7)
    ins.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, 'sheet_benchmark.' + ext), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    print('== running: python3 %s   (outputs in %s/results and %s/figures)' % (os.path.basename(__file__),
          os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(__file__))), flush=True)
    sys.exit(main())
