#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd codes
#     python3 08_local_hybrid_wires.py
#
# Needs: numpy, scipy, pyscf.  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""Local hybrids (a = tau_W/tau) and DC-SCAN on N = 2, 4, 6 Gaussian wires built from the s, p_z, d_zz functions of the aniGTO trap basis (fixed orbitals, exchange only).

STANDALONE SCRIPT (SCAN_1D_2D/codes): run it alone, from any directory,

    python3 08_local_hybrid_wires.py   (about 2 min)

It needs only third-party packages (numpy, scipy, pyscf (libxc)); every project function it uses is defined in this file.
Outputs: results/explore_local_hybrid.txt, tab_dcscan_lh_rows.tex
Manuscript items: SI Table S9

---------------------------------------------------------------------------------------------------------------
Exploratory: local hybrids on N = 2, 4, 6 Gaussian-wire densities (fixed orbitals, exchange only).

Orbitals: phi_k(r) = g(rho) h_k(z), k = 0..M-1, doubly occupied (N = 2M); g^2 = lam^2/pi e^{-lam^2 rho^2},
h_k the harmonic-oscillator (Hermite) functions -- the s, p_z, d_zz functions of the aniGTO trap basis with
omega = (lam^2, lam^2, 1).  Exact (HF) exchange of these orbitals, two independent ways:
  (a) Kantorovich: E_x = -sum_ij int int p_ij(z) p_ij(z') w_b(z - z'),  p_ij = h_i h_j,  b = 1/(sqrt2 lam);
  (b) integral of the conventional exchange-energy density
      e_x(r) = -sum_ij phi_i phi_j(r) v_ij(r),  v_ij(r) = int phi_i phi_j(r') / |r - r'| d^3r'
      = (2/sqrt pi) int_0^inf dt  lam^2/(lam^2+t^2) e^{-t^2 u/(lam^2+t^2)}  Q_ij(z,t),   u = lam^2 rho^2,
      Q_ij(z,t) = int p_ij(z') e^{-t^2 (z-z')^2} dz'  (Gaussian moments, exact).
Local hybrid: E = int [a e_x^HF + (1 - a) e_x^DFA],  a = tau_W / tau  (iso-orbital mixing, Jaramillo-Scuseria-
Ernzerhof type);  a = 1 for one orbital, so N = 2 is exact by construction.  DFA from libxc (via PySCF) plus
DC-SCAN (eq. 4) on the full SCAN F_x(s, alpha).
"""
import numpy as np
import os
import sys
from numpy.polynomial.hermite import hermgauss
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
# Main calculation
# ==============================================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))

PI4 = np.pi ** -0.25


def hermite_funcs(z, M):
    """h_k, h_k', h_k'' for k < M (physicists' Hermite functions)."""
    h = [PI4 * np.exp(-z * z / 2)]
    if M > 1:
        h.append(np.sqrt(2.0) * z * h[0])
    for k in range(2, M):
        h.append(np.sqrt(2.0 / k) * z * h[k - 1] - np.sqrt((k - 1) / k) * h[k - 2])
    h = np.array(h)
    dh = np.zeros_like(h)
    for k in range(M):          # h_k' = sqrt(k/2) h_{k-1} - sqrt((k+1)/2) h_{k+1}
        dh[k] = -z * h[k] + (np.sqrt(2.0 * k) * h[k - 1] if k > 0 else 0.0)
    d2h = (z * z - 2 * np.arange(M)[:, None] - 1) * h    # h'' = (z^2 - 2k - 1) h
    return h, dh, d2h


def poly_pair(M):
    """p_ij(z) = P_ij(z) e^{-z^2}: return a callable P_ij for the pair list."""
    pairs = [(i, j) for i in range(M) for j in range(M)]

    def P(z):
        h, _, _ = hermite_funcs(z, M)
        e = np.exp(z * z / 2)
        return np.array([h[i] * e * h[j] * e for i, j in pairs])
    return pairs, P


def Q_table(z, t, M):
    """Q_ij(z,t) for all pairs: shape (npair, nt, nz)."""
    pairs, P = poly_pair(M)
    x, w = hermgauss(8)
    T2 = (t * t)[:, None]
    c = 1 + T2
    mu = T2 * z[None, :] / c
    pre = np.exp(-T2 * z[None, :] ** 2 / c) / np.sqrt(c)          # sqrt(pi/c) * (1/sqrt(pi)) from GH weights
    out = np.zeros((len(pairs), t.size, z.size))
    for xk, wk in zip(x, w):
        out += wk * P((mu + xk / np.sqrt(c)).ravel()).reshape(len(pairs), t.size, z.size)
    return pairs, out * pre[None]


def fields(lam, M, nu=300, nz=300, nt=500):
    u, wu = _log_gl(np.exp(-34.0), 45.0, nu)
    z, wz = _log_gl(np.exp(-34.0), 11.0, nz)
    U, Zg = np.meshgrid(u, z, indexing='ij')
    W = (np.pi / lam ** 2) * 2.0 * np.outer(wu, wz)           # d^3r = pi du / lam^2 dz, z<0 mirrored
    g2 = lam ** 2 / np.pi * np.exp(-U)
    h, dh, d2h = hermite_funcs(z, M)
    nz_ = 2 * np.sum(h * h, axis=0)
    dnz = 4 * np.sum(h * dh, axis=0)
    d2nz = 4 * np.sum(dh * dh + h * d2h, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        L1 = np.where(nz_ > 0, dnz / nz_, 0.0)
        L2 = np.where(nz_ > 0, d2nz / nz_ - L1 ** 2, 0.0)
    L1, L2 = L1[None, :], L2[None, :]
    n = g2 * nz_[None, :]
    rho_c = np.sqrt(U) / lam
    gx = -2 * lam ** 2 * rho_c * n
    gz = L1 * n
    grad2 = gx * gx + gz * gz
    lap = n * ((4 * lam ** 2 * U + L1 ** 2) + (-4 * lam ** 2 + L2))
    tau = g2 * (lam ** 2 * U * np.sum(h * h, axis=0)[None, :] + np.sum(dh * dh, axis=0)[None, :])
    tauW = grad2 / (8 * n)
    # density curvature C = diag(lam^2, lam^2, -L2/2)
    czz = -0.5 * L2 + 0 * U
    trc2 = 2 * lam ** 4 + czz ** 2
    gc2g = n * n * (lam ** 4 * 4 * lam ** 4 * rho_c ** 2 + czz ** 2 * L1 ** 2)   # gx^2 C_xx^2 + gz^2 C_zz^2
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.clip(np.where(gc2g > 0, grad2 * trc2 / gc2g, 3.0), 1.0, 3.0)
    # exact exchange-energy density
    t, wt = _log_gl(1e-7, 1e10, nt)
    T = (2 / np.sqrt(np.pi)) * wt[None, :] * lam ** 2 / (lam ** 2 + t * t)[None, :] \
        * np.exp(-(t * t)[None, :] * u[:, None] / (lam ** 2 + t * t)[None, :])
    pairs, Q = Q_table(z, t, M)
    _, P = poly_pair(M)
    pz = P(z) * np.exp(-z * z)                                 # p_ij(z)
    eHF = np.zeros_like(U)
    for k in range(len(pairs)):
        eHF -= g2 * pz[k][None, :] * (T @ Q[k])
    return dict(n=n, gx=gx, gz=gz, grad2=grad2, lap=lap, tau=tau, tauW=tauW, D=D, trc2=trc2, eHF=eHF, W=W)


def kantorovich(lam, M):
    b = 1 / (np.sqrt(2) * lam)
    pairs, P = poly_pair(M)
    zg, wzg = np.polynomial.legendre.leggauss(600)
    z = 10 * zg
    wz = 10 * wzg
    xg, wg = np.polynomial.legendre.leggauss(48)
    edges = np.concatenate([[0.0], np.logspace(np.log10(b) - 6, np.log10(20.0), 80)])
    pz = P(z) * np.exp(-z * z)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        r = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        ww = 0.5 * (hi - lo) * wg * np.sqrt(np.pi) / (2 * b) * erfcx(r / (2 * b))
        for s in (+1, -1):
            zz = (z[None, :] + s * r[:, None]).ravel()
            pp = (P(zz) * np.exp(-zz * zz)).reshape(len(pairs), r.size, z.size)
            tot += np.einsum('r,kz,krz,z->', ww, pz, pp, wz)
    return -tot


def dfa(F):
    from pyscf.dft import numint
    NI = numint.NumInt()
    n = F['n'].ravel()
    keep = n > 1e-30
    inp = np.vstack((n, F['gx'].ravel(), 0 * n, F['gz'].ravel(), F['lap'].ravel(), F['tau'].ravel()))[:, keep]
    out = {}
    for k, code in (('LDA', 'LDA_X'), ('PBE', 'GGA_X_PBE'), ('SCAN', 'MGGA_X_SCAN')):
        x = inp[:1] if k == 'LDA' else (inp[:4] if k == 'PBE' else inp)
        e = np.zeros_like(n)
        e[keep] = np.asarray(NI.eval_xc(code + ',', x, spin=0, deriv=0)[0]) * n[keep]
        out[k] = e.reshape(F['n'].shape)
    # DC-SCAN on the full SCAN F_x(s, alpha)
    with np.errstate(divide='ignore', invalid='ignore'):
        fscan = np.where(out['LDA'] != 0, out['SCAN'] / out['LDA'], 1.0)
    kf = KF * np.cbrt(F['n'])
    s = np.sqrt(F['grad2']) / (2 * kf * F['n'])
    Zs = np.sqrt(F['trc2']) / kf ** 2
    out['DC-SCAN'] = out['LDA'] * fx_dcscan(s, F['D'], Zs, fscan)
    return out


def _sci(v):
    e = int(np.floor(np.log10(abs(v))))
    return '$%.1f\\times10^{%d}$' % (v / 10 ** e, e)


def _lh(v):
    a = abs(v)
    return '$%+.1f$' % v if a >= 10 else ('$%+.2f$' % v if a >= 1 else '$%+.3f$' % v)


def write_si_rows(allres, lams):
    """SI table body for N = 4 and 6: exact E_x (Ha), <1-a>_n and errors (mHa) of SCAN, DC-SCAN and local hybrids."""
    L = []
    for M in (2, 3):
        res = allres[M]
        L.append('%d & $E_x^{\\mathrm{exact}}$, eq~\\ref{eq:ani-eed} (Ha) & ' % (2 * M) + ' & '.join('$%.6f$' % res[l]['exact'] for l in lams) + ' \\\\')
        # values first (E_x of every method = exact + error, Ha), then <1-a> and the errors (mHa) -- user 2026-11-05
        for k in ('SCAN', 'DC-SCAN', 'LH-PBE', 'LH-SCAN', 'LH-DC-SCAN'):
            L.append('  & %s $E_x$ (Ha) & ' % k + ' & '.join('$%.6f$' % (res[l]['exact'] + res[l][k]) for l in lams) + ' \\\\')
        L.append('  & $\\langle1-a\\rangle_n$ & ' + ' & '.join(_sci(res[l]['mean_1-a']) for l in lams) + ' \\\\')
        for k in ('SCAN', 'DC-SCAN'):
            L.append('  & %s error (mHa) & ' % k + ' & '.join('$%+.1f$' % (1e3 * res[l][k]) for l in lams) + ' \\\\')
        for k in ('LH-PBE', 'LH-SCAN', 'LH-DC-SCAN'):
            L.append('  & %s error (mHa) & ' % k + ' & '.join(_lh(1e3 * res[l][k]) for l in lams) + ' \\\\')
        if M == 2:
            L.append('\\midrule')
    open(os.path.join(HERE, 'results', 'tab_dcscan_lh_rows.tex'), 'w').write('\n'.join(L) + '\n')


def main():
    lams = [2.0, 10.0, 100.0, 1000.0, 10000.0]
    print('fixed-orbital exchange on N-electron Gaussian wires; errors vs exact in mHa\n')
    allres = {}
    for M in (1, 2, 3):
        res = {}
        for lam in lams:
            F = fields(lam, M)
            W = F['W']
            ek = kantorovich(lam, M)
            eh = float(np.sum(W * F['eHF']))
            X = dfa(F)
            with np.errstate(divide='ignore', invalid='ignore'):
                a = np.clip(np.where(F['tau'] > 0, F['tauW'] / F['tau'], 1.0), 0, 1)
            r = {'N_check': float(np.sum(W * F['n'])), 'exact': ek, 'EED-exact': eh - ek,
                 'mean_1-a': float(np.sum(W * F['n'] * (1 - a)) / np.sum(W * F['n']))}
            for k, e in X.items():
                r[k] = float(np.sum(W * e)) - ek
                r['LH-' + k] = float(np.sum(W * (a * F['eHF'] + (1 - a) * e))) - ek
            res[lam] = r
        allres[M] = res
        print(f'N = {2 * M}')
        print('%-14s' % 'lam' + ''.join('%14g' % l for l in lams))
        print('%-14s' % 'N on grid' + ''.join('%14.8f' % res[l]['N_check'] for l in lams))
        print('%-14s' % 'E_x exact' + ''.join('%14.6f' % res[l]['exact'] for l in lams))
        print('%-14s' % 'EED-Kant (Ha)' + ''.join('%14.1e' % res[l]['EED-exact'] for l in lams))
        print('%-14s' % '<1-a>_n' + ''.join('%14.2e' % res[l]['mean_1-a'] for l in lams))
        for k in ('LDA', 'PBE', 'SCAN', 'DC-SCAN', 'LH-LDA', 'LH-PBE', 'LH-SCAN', 'LH-DC-SCAN'):
            print('%-14s' % k + ''.join('%+14.3f' % (1e3 * res[l][k]) for l in lams))
        print()
    write_si_rows(allres, lams)


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
    sys.stdout = Tee(os.path.join(HERE, 'results', 'explore_local_hybrid.txt'))
    main()
