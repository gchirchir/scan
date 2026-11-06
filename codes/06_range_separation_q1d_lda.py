#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd codes
#     python3 06_range_separation_q1d_lda.py
#
# Needs: numpy, scipy.  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""Beyond the semilocal form (exploratory): HSE/LC-type range separation and the Kantorovich quasi-1D LDA on the Gaussian wire.

STANDALONE SCRIPT (SCAN_1D_2D/codes): run it alone, from any directory,

    python3 06_range_separation_q1d_lda.py

It needs only third-party packages (numpy, scipy); every project function it uses is defined in this file.
Outputs: results/explore_rsh_q1d.txt
Manuscript items: SI Section S6 (range separation, quasi-1D LDA)

---------------------------------------------------------------------------------------------------------------
Exploratory: can range separation (HSE-type, LC-type) or a Kantorovich-based quasi-1D LDA reach mHa
accuracy for the exchange energy of the two-electron Gaussian wire?  Fixed density, exchange only.

Exact split  erf(w r)/r = (2/sqrt pi) int_0^w e^{-t^2 r^2} dt  ->  E_x^LR(w) = -(2/sqrt pi) int_0^w
    sqrt(det A)/sqrt(det(A + 2 t^2 I)) dt,  E_x^SR = E_x - E_x^LR  (Eq. 44 with a finite upper limit).
SR-DFA: ITYH (Iikura-Tsuneda-Yanai-Hirao 2001) attenuation of any F_x: e_x^LDA F_x F_att(a),
    a = w / (2 k),  k = k_F / sqrt(F_x),  F_att = LDA erfc attenuation (Savin / Gill).
q1D-LDA: local in z on the 1D density n1(z) with the 1D uniform-gas exchange of the transverse-averaged
    interaction w_b (defined below), b = 1/sqrt(2 C_perp) from the density curvature:
    e_x(n1) = -2 int_0^inf [sin(k r)/(pi r)]^2 w_b(r) dr,  k = pi n1 / 2.
"""
import numpy as np
import os
import sys
from scipy.special import erf, erfcx
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

LAMS = [2.0, 10.0, 100.0, 1000.0, 10000.0]


def ex_lr_exact(A, w, n=200):
    A = np.asarray(A, float)
    x, wt = np.polynomial.legendre.leggauss(n)
    # t in [0, w]; integrand varies on the scale sqrt(min A) and sqrt(max A): log-graded panels
    edges = np.concatenate([[0.0], np.logspace(-6, 0, 25) * w])
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        f = np.sqrt(np.prod(A)) / np.sqrt(np.prod(A[None, :] + 2 * t[:, None] ** 2, axis=1))
        tot += np.sum(0.5 * (hi - lo) * wt * f)
    return -2.0 / np.sqrt(np.pi) * tot


def f_att(a):
    a = np.asarray(a, float)
    out = np.empty_like(a)
    big = a > 8.0
    ab = a[big]
    out[big] = 1 / (36 * ab ** 2) - 1 / (960 * ab ** 4) + 1 / (26880 * ab ** 6)
    s = a[~big]
    with np.errstate(over='ignore', under='ignore'):
        out[~big] = 1 - 8.0 / 3.0 * s * (np.sqrt(np.pi) * erf(1 / (2 * s)) + (2 * s - 4 * s ** 3)
                                         * np.exp(-1 / (4 * s * s)) - 3 * s + 4 * s ** 3)
    return out


def dfa_parts(lam, w, n=400, family='wire'):
    """Full and SR (ITYH) energies for LDA, SCAN, DC-SCAN on the reduced grid of dcscan.ex_family."""
    k = 2 if family == 'wire' else 1
    u, wu = _log_gl(np.exp(-32.0), 26.0, n)
    t, wt = _log_gl(np.exp(-32.0), 8.0, n)
    U, T2, s, D, Z = family_fields(family, lam, u, t)
    W = np.outer(wu, wt) * np.exp(-4.0 / 3.0 * (U + T2))
    pref = -2.0 * np.pi * CX * C_NORM ** (4.0 / 3.0) * lam ** (k / 3.0)
    kf = KF * np.cbrt(lam ** k * C_NORM * np.exp(-(U + T2)))
    fs = fx_scan_alpha0(s)
    out = {}
    for name, F in (('LDA', np.ones_like(s)), ('SCAN', fs), ('DC-SCAN', fx_dcscan(s, D, Z, fs))):
        full = pref * np.sum(W * F)
        sr = pref * np.sum(W * F * f_att(w / (2 * kf / np.sqrt(F)))) if w > 0 else full
        out[name] = (full, sr)
    return out


def ex_q1d_lda(lam, nz=161, zmax=6.0):
    """Local quasi-1D LDA on n1(z) = 2 pi^-1/2 e^{-z^2} with the w_b kernel, b = 1/(sqrt2 lam)."""
    b = 1.0 / (np.sqrt(2.0) * lam)
    wb = lambda r: np.sqrt(np.pi) / (2 * b) * erfcx(r / (2 * b))   # noqa: E731
    zg, wz = np.polynomial.legendre.leggauss(nz)
    z = zmax * zg
    wz = zmax * wz
    n1 = 2 / np.sqrt(np.pi) * np.exp(-z * z)
    kk = np.pi * n1 / 2
    # x = k r: e_x = -(2 k / pi^2) int_0^inf sin^2 x / x^2 w_b(x / k) dx; panels: log near 0, unit width beyond
    xg, wg = np.polynomial.legendre.leggauss(24)
    edges = np.concatenate([[0.0], np.logspace(-12, 0, 49), np.arange(2.0, 4001.0)])
    ex = np.zeros_like(z)
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        ww = 0.5 * (hi - lo) * wg
        X, K = np.meshgrid(x, kk, indexing='ij')
        ex += np.sum(ww[:, None] * np.sin(X) ** 2 / X ** 2 * wb(X / K), axis=0)
    X = edges[-1]
    ex += 1.0 / (4 * X * X) * kk          # tail: w_b ~ k/x, <sin^2> = 1/2
    ex *= -2 * kk / np.pi ** 2
    return float(np.sum(wz * ex))


def main():
    rows = []
    print('wire, errors E - exact in mHa\n')
    hdr = '%-26s' % 'method' + ''.join('%11s' % ('lam=%g' % l) for l in LAMS) + '%10s' % 'RMS'
    print(hdr)
    res = {}
    ex = {l: float(ex_exact_wire(l)) for l in LAMS}
    for l in LAMS:
        A = diag_A('wire', l)
        r = {}
        f0 = dfa_parts(l, 0.0)
        r['SCAN'] = f0['SCAN'][0]
        r['DC-SCAN'] = f0['DC-SCAN'][0]
        for w, tag in ((0.11, 'HSE'), (0.3, 'LC0.3'), (1.0, 'LC1.0')):
            p = dfa_parts(l, w)
            lr = ex_lr_exact(A, w)
            sr = ex[l] - lr
            for base in ('LDA', 'SCAN', 'DC-SCAN'):
                full, srd = p[base]
                if tag == 'HSE':
                    r['HSE06-' + base] = 0.25 * sr + 0.75 * srd + (full - srd)
                else:
                    r[tag + '-' + base] = srd + lr
            r['SRfrac_' + tag] = sr / ex[l]
        r['q1D-LDA'] = ex_q1d_lda(l)
        res[l] = r
    keys = ['SCAN', 'DC-SCAN', 'HSE06-SCAN', 'HSE06-DC-SCAN', 'LC0.3-SCAN', 'LC0.3-DC-SCAN', 'LC1.0-LDA',
            'LC1.0-SCAN', 'LC1.0-DC-SCAN', 'q1D-LDA']
    for kname in keys:
        e = np.array([1e3 * (res[l][kname] - ex[l]) for l in LAMS])
        print('%-26s' % kname + ''.join('%+11.1f' % v for v in e) + '%10.1f' % np.sqrt(np.mean(e * e)))
        rows.append([kname] + list(e))
    # one-parameter fit: E = q1D-LDA + c * (E_exact(lam=1 shape) ...) -> simplest: scale factor on the O(1) error
    q = np.array([res[l]['q1D-LDA'] for l in LAMS])
    e0 = np.array([ex[l] for l in LAMS])
    print('\nfraction of |E_x| in the SR part:', ' '.join('%.3f' % res[l]['SRfrac_HSE'] for l in LAMS),
          '(w=0.11)', ' '.join('%.3f' % res[l]['SRfrac_LC1.0'] for l in LAMS), '(w=1)')
    print('q1D-LDA - exact (mHa):', ' '.join('%+.2f' % v for v in 1e3 * (q - e0)))
    return res


class Tee:
    """Print to the terminal and to a results file at the same time."""

    def __init__(self, path):
        self.f, self.o = open(path, 'w'), sys.stdout

    def write(self, s):
        self.f.write(s); self.o.write(s)

    def flush(self):
        self.f.flush(); self.o.flush()


if __name__ == '__main__':
    print('== running: python3 %s   (outputs in %s/results and %s/figures)' % (os.path.basename(__file__),
          os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(__file__))), flush=True)
    HERE = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    sys.stdout = Tee(os.path.join(HERE, 'results', 'explore_rsh_q1d.txt'))
    main()
