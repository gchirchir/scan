#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd codes
#     python3 05_refit_switch_scales.py
#
# Needs: numpy, scipy.  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""PBE base and the exploratory refit of the two activation scales (s0, Z0) of DC-SCAN on the wire, with the sheet as test set.

STANDALONE SCRIPT (SCAN_1D_2D/codes): run it alone, from any directory,

    python3 05_refit_switch_scales.py

It needs only third-party packages (numpy, scipy); every project function it uses is defined in this file.
Outputs: results/switch_refit.csv, switch_refit_log.txt, tab_dcscan_refit_rows.tex
Manuscript items: SI Table S7

---------------------------------------------------------------------------------------------------------------
Exploratory refit of the DC-SCAN activation scales on the wire, tested on the sheet.


Omega(s, Z) = (Z/Z0)^3/(1+(Z/Z0)^3) * (s/s0)^8/(1+(s/s0)^8);  the note uses s0 = Z0 = 1.
Only s0 and Z0 are fitted (least squares on the wire errors at lam = 2, 10, 100, 1e3, 1e4).  A2D and A1D are not
touched, so the uniform-gas limit (Omega = 0 at s = 0), the Gaussian sheet limit and the wire log coefficient
remain exact for any s0, Z0 > 0.  The sheet family is NOT used in the fit and serves as the test set.
Bases: SCAN (alpha = 0 closed form, identical to libxc SCAN/r2SCAN at alpha = 0) and PBE (kappa = 0.804,
mu = 0.2195149727645171).  The PBE and SCAN bases alone are listed for reference.
Output: results/switch_refit.csv, results/switch_refit_log.txt.  Fixed-density, two-parameter, one-family fit:
an exploratory diagnostic, not a validated functional.
"""
import numpy as np
import os
import re
import sys
from scipy.optimize import minimize
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
# Main calculation
# ==============================================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

KAPPA, MU = 0.804, 0.2195149727645171
LAMS = [2.0, 10.0, 100.0, 1000.0, 10000.0]


def fx_pbe(s):
    return 1.0 + KAPPA - KAPPA / (1.0 + MU * s * s / KAPPA)


def ex(family, lam, base, s0=1.0, z0=1.0, crossover=True, n=200):
    k = 2 if family == 'wire' else 1
    u, wu = _log_gl(np.exp(-32.0), 26.0, n)
    t, wt = _log_gl(np.exp(-32.0), 8.0, n)
    U, T2, s, D, Z = family_fields(family, lam, u, t)
    W = np.outer(wu, wt) * np.exp(-4.0 / 3.0 * (U + T2))
    pref = -2.0 * np.pi * CX * C_NORM ** (4.0 / 3.0) * lam ** (k / 3.0)
    fb = fx_scan_alpha0(s) if base == 'SCAN' else fx_pbe(s)
    if not crossover:
        return pref * np.sum(W * fb)
    w2, w1, _ = geometry_weights(D)
    with np.errstate(over='ignore', invalid='ignore'):
        zs, ss = (Z / z0) ** 3, (s / s0) ** 8
        om = np.nan_to_num(zs / (1.0 + zs) * ss / (1.0 + ss), nan=1.0)
    return pref * np.sum(W * ((1.0 - om * (w2 + w1)) * fb + om * (w2 * f2d(s) + w1 * f1d(s))))


def errors(family, base, **kw):
    return [ex(family, l, base, **kw) - float(ex_exact(family, l)) for l in LAMS]


def main():
    res = os.path.join(HERE, 'results')
    os.makedirs(res, exist_ok=True)
    rows, log, vals = [], [], []
    for base in ('SCAN', 'PBE'):
        def obj(p):
            s0, z0 = np.exp(p)
            return sum(e * e for e in errors('wire', base, s0=s0, z0=z0, n=120))
        r = minimize(obj, [0.0, 0.0], method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-10, maxiter=400))
        s0, z0 = np.exp(r.x)
        for label, kw in ((base + ' alone', dict(crossover=False)),
                          ('DC-' + base + ' (note: s0=Z0=1)', {}),
                          ('DC-' + base + ' refit (s0=%.3f, Z0=%.3f)' % (s0, z0), dict(s0=s0, z0=z0))):
            for fam in ('wire', 'sheet'):
                e = errors(fam, base, **kw)
                rows.append([label, fam] + e + [float(np.sqrt(np.mean(np.square(e))))])
                vals.append([float(ex_exact(fam, l)) + v for v, l in zip(e, LAMS)])   # E_x values (11-05)
                log.append('%-40s %-5s ' % (label, fam) + ' '.join('%+9.4f' % v for v in e)
                           + '   RMS %.4f' % rows[-1][-1])
    with open(os.path.join(res, 'switch_refit.csv'), 'w') as f:
        f.write('variant,family,' + ','.join('err_lam_%g' % l for l in LAMS) + ',rms\n')
        for r in rows:
            f.write('"%s",%s,' % (r[0], r[1]) + ','.join('%.10g' % v for v in r[2:]) + '\n')
    with open(os.path.join(res, 'tab_dcscan_refit_rows.tex'), 'w') as f:      # SI table body
        # user 2026-11-05: print the data (E_x^exact and E_x) first, the errors E_x - exact after them
        for fam in ('wire', 'sheet'):
            f.write('exact, eq~\\ref{M-eq:dc-ex%s} & %s & ' % (fam, fam) + ' & '.join('%.6f' % float(ex_exact(fam, l)) for l in LAMS)
                    + ' & ' * (len(LAMS) + 1) + ' \\\\\n')
        f.write('\\midrule\n')
        for r, v in zip(rows, vals):
            name = r[0].replace('DC-', 'DC-').replace('(note: s0=Z0=1)', '($s_0=Z_0=1$)')
            name = re.sub(r'refit \(s0=([0-9.]+), Z0=([0-9.]+)\)', r'refit ($s_0=\1$, $Z_0=\2$)', name)
            f.write('%s & %s & ' % (name, r[1]) + ' & '.join('%.6f' % x for x in v) + ' & '
                    + ' & '.join('%+.4f' % x for x in r[2:-1]) + ' & %.4f \\\\\n' % r[-1])
    head = ['errors E_x - exact (Ha) at lam = ' + ', '.join('%g' % l for l in LAMS),
            'fit set: wire only; sheet = test set']
    text = '\n'.join(head + log) + '\n'
    open(os.path.join(res, 'switch_refit_log.txt'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    print('== running: python3 %s   (outputs in %s/results and %s/figures)' % (os.path.basename(__file__),
          os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(__file__))), flush=True)
    main()
