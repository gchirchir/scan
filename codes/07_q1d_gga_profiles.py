#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd codes
#     python3 07_q1d_gga_profiles.py
#
# Needs: numpy, scipy.  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""Quasi-1D LDA and the Gaussian-fitted quasi-1D GGA on two-electron wires with seven axial profiles (exact exchange from the Kantorovich expression).

STANDALONE SCRIPT (SCAN_1D_2D/codes): run it alone, from any directory,

    python3 07_q1d_gga_profiles.py

It needs only third-party packages (numpy, scipy); every project function it uses is defined in this file.
Outputs: results/explore_q1d_gga.txt, tab_dcscan_q1d_rows.tex
Manuscript items: SI Table S8

---------------------------------------------------------------------------------------------------------------
Exploratory: transferability of a quasi-1D (Kantorovich w_b) exchange LDA and a one-parameter 1D GGA.

Two-electron product densities n(r) = n1(z) g_b(rho), g_b the transverse Gaussian of width b = 1/(sqrt2 lam).
Exact exchange (one orbital): E_x = -1/4 int int n1(z) n1(z') w_b(z - z') dz dz'  -- exact for ANY n1(z).
q1D-LDA:  E = int e_x^{1D,b}(n1(z)) dz  (1D uniform gas with interaction w_b).
q1D-GGA:  E = q1D-LDA - c int n1^2 h(s1) dz,  s1 = |n1'| / n1^2 (dimensionless, same scaling as n1^2),
          h(s) = s^2 / (1 + s^2).  c fitted on the Gaussian profile only, then tested on other profiles.
"""
import numpy as np
import os
import sys
from scipy.special import erfcx


# ==============================================================================================================
# Main calculation
# ==============================================================================================================
ZL = 14.0
NZ = 801
zg, wzg = np.polynomial.legendre.leggauss(NZ)
Z = ZL * zg
WZ = ZL * wzg


def wb(r, b):
    return np.sqrt(np.pi) / (2 * b) * erfcx(np.abs(r) / (2 * b))


def normalize(f):
    return 2.0 * f / np.sum(WZ * f)


PROFILES = {
    'Gaussian (fit)': lambda z: np.exp(-z * z),
    'sech^2': lambda z: 1 / np.cosh(z) ** 2,
    'Lorentzian^2': lambda z: (1 + z * z) ** -2,
    'p-like z^2 e^-z^2': lambda z: z * z * np.exp(-z * z),
    'double well d=3': lambda z: np.exp(-(z - 1.5) ** 2) + np.exp(-(z + 1.5) ** 2),
    'wide Gaussian x3': lambda z: np.exp(-(z / 3) ** 2),
    'quartic-like e^-z^4': lambda z: np.exp(-z ** 4),
}


def ex_exact(fn, b):
    """-1/4 int dz n1(z) int dr n1(z + r) w_b(r), r on graded panels (w_b peaks on the scale b)."""
    n1 = normalize(fn(Z))
    c = 2.0 / np.sum(WZ * fn(Z))
    xg, wg = np.polynomial.legendre.leggauss(48)
    edges = np.concatenate([[0.0], np.logspace(np.log10(b) - 6, np.log10(2 * ZL), 80)])
    conv = np.zeros_like(Z)
    for lo, hi in zip(edges[:-1], edges[1:]):
        r = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        ww = 0.5 * (hi - lo) * wg * wb(r, b)
        conv += np.sum(ww[:, None] * c * (fn(Z[None, :] + r[:, None]) + fn(Z[None, :] - r[:, None])), axis=0)
    return -0.25 * float(np.sum(WZ * n1 * conv))


def ex_q1d_lda(n1, b):
    kk = np.pi * n1 / 2
    ok = kk > 1e-14
    xg, wg = np.polynomial.legendre.leggauss(24)
    edges = np.concatenate([[0.0], np.logspace(-12, 0, 49), np.arange(2.0, 3001.0)])
    ex = np.zeros(ok.sum())
    k = kk[ok]
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        ww = 0.5 * (hi - lo) * wg
        X, K = np.meshgrid(x, k, indexing='ij')
        ex += np.sum(ww[:, None] * np.sin(X) ** 2 / X ** 2 * wb(X / K, b), axis=0)
    ex += k / (4 * edges[-1] ** 2)
    out = np.zeros_like(n1)
    out[ok] = -2 * k / np.pi ** 2 * ex
    return float(np.sum(WZ * out))


def gga_term(fn, h):
    n1 = normalize(fn(Z))
    dz = 1e-5
    dn = (normalize(fn(Z + dz)) - normalize(fn(Z - dz))) / (2 * dz)   # normalize uses the same constant
    with np.errstate(divide='ignore', invalid='ignore'):
        s1 = np.where(n1 > 1e-300, np.abs(dn) / n1 ** 2, 0.0)
    return float(np.sum(WZ * n1 ** 2 * np.nan_to_num(h(s1))))


HERE = os.path.dirname(os.path.abspath(__file__))
TEX_NAMES = [('Gaussian (fit)', '$e^{-z^2}$ (fit)'), ('wide Gaussian x3', '$e^{-z^2/9}$'), ('sech^2', '$\\mathrm{sech}^2z$'),
             ('quartic-like e^-z^4', '$e^{-z^4}$'), ('Lorentzian^2', '$(1+z^2)^{-2}$'),
             ('double well d=3', '$e^{-(z-1.5)^2}+e^{-(z+1.5)^2}$'), ('p-like z^2 e^-z^2', '$z^2e^{-z^2}$')]


def write_si_rows(data, c, lam=1000.0, hn='s^2/(1+s^2)'):
    """SI table body: exact E_x (Ha), q1D-LDA and Gaussian-fitted q1D-GGA errors (mHa) at lam = 1e3."""
    with open(os.path.join(HERE, 'results', 'tab_dcscan_q1d_rows.tex'), 'w') as f:
        for key, tex in TEX_NAMES:
            ee, el = data[key]['E'][lam]
            eg = el - c * data[key]['G'][hn]                     # q1D-GGA E_x (Ha); values first, errors after (11-05)
            f.write('%s & $%.5f$ & $%.5f$ & $%.5f$ & $%+.1f$ & $%+.2f$ \\\\\n' % (tex, ee, el, eg, 1e3 * (el - ee), 1e3 * (eg - ee)))


def main():
    lams = [10.0, 100.0, 1000.0, 10000.0]
    hs = {'s^2/(1+s^2)': lambda s: s * s / (1 + s * s), 's/(1+s)': lambda s: s / (1 + s)}
    data = {}
    for name, fn in PROFILES.items():
        n1 = normalize(fn(Z))
        data[name] = {'G': {k: gga_term(fn, h) for k, h in hs.items()}, 'E': {}}
        for lam in lams:
            b = 1 / (np.sqrt(2) * lam)
            data[name]['E'][lam] = (ex_exact(fn, b), ex_q1d_lda(n1, b))
    g = data['Gaussian (fit)']
    # fit c on the Gaussian at lam >= 100 (lam-independent offset)
    for hn in hs:
        c = np.mean([g['E'][l][1] - g['E'][l][0] for l in lams[1:]]) / g['G'][hn]
        if hn == 's^2/(1+s^2)':
            write_si_rows(data, c)
        print(f'\nq1D-GGA with h = {hn}: c = {c:.5f} (fit on Gaussian, lam >= 100)')
        print('%-22s' % 'profile' + ''.join('%22s' % ('lam=%g LDA | GGA' % l) for l in lams) + '   (mHa)')
        for name, dd in data.items():
            row = ''
            for l in lams:
                ee, el = dd['E'][l]
                row += '%11.1f |%+8.2f' % (1e3 * (el - ee), 1e3 * (el - c * dd['G'][hn] - ee))
            print('%-22s' % name + row)
    print('\nexact E_x at lam=1e3:', ', '.join('%s %.5f' % (n, d['E'][1000.0][0]) for n, d in data.items()))


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
    sys.stdout = Tee(os.path.join(HERE, 'results', 'explore_q1d_gga.txt'))
    main()
