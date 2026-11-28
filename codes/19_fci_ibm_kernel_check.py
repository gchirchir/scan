#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal (after code 10 has written results/fci_gaussian_validation.csv):
#
#     cd SCAN_1D_2D        # (the folder that contains codes/)
#     python3 codes/19_fci_ibm_kernel_check.py
#
# Needs: numpy, scipy, matplotlib.  Runtime: a few seconds.
# ----------------------------------------------------------------------
r"""FCI exchange of the two-electron Gaussian sheet/wire recomputed with the IBM exact-exchange kernels.

STANDALONE SCRIPT (SCAN_1D_2D/codes): imports only third-party packages; reads the FCI data written by code 10
(results/fci_gaussian_validation.csv; data, not code).

The exact single-subband exchange of the infinite-barrier (IBM) systems (codes 14, 17, 18) is a Coulomb form-factor
integral: free-direction Fourier transform x confinement kernel.  For a single doubly occupied orbital
phi = phi_free * phi_conf the same route gives E_x = -J exactly:

  sheet  A = diag(1,1,lam^2):  E_x = -int_0^inf exp(-q^2/2) F(q) dq,
         F(q) = <exp(-q|z-z'|)> = exp(q^2/(2 lam^2)) erfc(q/(sqrt2 lam))         (IBM-well form factor)
  wire   A = diag(lam^2,lam^2,1):  E_x = -(1/pi) int_0^inf exp(-k^2/2) V(k) dk,
         V(k) = 2 int_0^inf p exp(-p^2/(2 lam^2))/(p^2+k^2) dp = exp(x) E1(x),  x = k^2/(2 lam^2)
                                                                               (IBM-wire Hankel kernel)
Limits: sheet -> -sqrt(pi/2) (F -> 1); wire -> -sqrt(2/pi) ln(2 lam) + const.

Outputs: results/fci_ibm_kernel_check.csv, results/tab_fci_ibm_kernel_rows.tex,
         figures/fci_ibm_kernel_check.pdf/.eps/.png
"""
import csv
import os

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, exp1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ----------------------------------------------------------------- publication style (identical copy in codes 14-17, 19)
# Modelled on the JPCL He-series figures (style() of their plot.py): serif Times/STIX text, inward ticks on all four
# sides, heavy frame, framed white legend with a black edge, bold in-panel labels instead of titles, no grid, a
# marker + line style + colour per method.  Sizes are PRINT sizes: make the figure exactly as wide as it is printed
# (PUB_W) and the fonts come out at the stated points (the JPCL 40/30/27 pt on a 14-in figure printed one column wide
# are 9.6/7.2/6.5 pt; here 10/9/7.5 pt).  The colour of a method is the same in every figure (validated palette:
# normal-vision dE >= 15 for all pairs; CVD separation is carried by the distinct marker + line style of each method).
PUB_W = {'column': 3.37, 'main_wide': 6.5, 'main_geometry': 6.6,          # revtex jcp reprint (main text)
         'sm_055': 3.6, 'sm_062': 4.05, 'sm_095': 6.2}                     # SM: article, a4, 22 mm margins
PUB = {  # method -> colour, line style, marker
    'exact':  ('#000000', '-', None),
    'LDA':    ('#6e6e6e', (0, (5, 2)), 'o'),
    'PBE':    ('#c23b4f', (0, (1, 1.3)), 's'),
    'SCAN':   ('#c47a00', (0, (4, 1.5)), '^'),
    'r2SCAN': ('#9b59b6', (0, (1, 1.3)), 'v'),
    'DC':     ('#0a8f6a', '-', 'D'),
    'DCr2':   ('#1a3f8a', (0, (5, 1.5, 1, 1.5)), 'P'),
    'A':      ('#3aa6d8', (0, (5, 1.5, 1, 1.5)), 'X'),
    'B':      ('#9b59b6', (0, (4, 1.5)), '*'),
    'C':      ('#c47a00', (0, (5, 1.5, 1, 1.5)), 'h'),
    'sheet':  ('#1f5f9e', '-', 'o'),
    'wire':   ('#0a8f6a', '-', 's'),
    'sheetF': ('#b8650a', 'none', 'o'),
    'wireF':  ('#a03fb0', 'none', 's'),
    'guide':  ('#7a7a7a', (0, (5, 2)), None),
}


def pub_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9.5,
        'axes.labelsize': 10.5,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 7.5,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.major.size': 4.0, 'ytick.major.size': 4.0,
        'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'axes.grid': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 4.5,
        'lines.markeredgewidth': 0.8,
        'legend.frameon': True, 'legend.framealpha': 1.0, 'legend.fancybox': False,
        'legend.edgecolor': 'black', 'legend.facecolor': 'white',
        'legend.handlelength': 2.4, 'legend.borderpad': 0.35, 'legend.labelspacing': 0.25,
        'legend.columnspacing': 0.9, 'patch.linewidth': 0.8,
        'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def pub_line(ax, x, y, key, label=None, marks=10, lw=None, ms=None, filled=False, **kw):
    """Plot one method in its fixed colour / line style / marker; `marks` markers spread evenly along the curve."""
    c, ls, mk = PUB[key]
    st = dict(color=c, ls=ls, lw=lw or (2.0 if key == 'exact' else 1.5), label=label)
    if mk is not None:
        st.update(marker=mk, ms=ms or 4.5, mfc=c if filled else 'white', mec=c, mew=0.9,
                  markevery=(1.0 / marks) if ls != 'none' else None)
    st.update(kw)
    return ax.plot(x, y, **st)[0]


def pub_legend(ax, **kw):
    return ax.legend(**kw)


def pub_panel(ax, text, x=0.03, y=0.96, ha='left', va='top'):
    """Bold in-panel label, e.g. '(a) quasi-2D well', in place of an axes title."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=10, fontweight='bold')


def pub_save(fig, path_no_ext):
    for ext in ('pdf', 'eps', 'png'):
        fig.savefig(f'{path_no_ext}.{ext}', dpi=300 if ext == 'png' else None)
    plt.close(fig)
# ----------------------------------------------------------------- end of publication style

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

TOL = dict(epsabs=1e-14, epsrel=1e-13, limit=400)


def ex_sheet_ibm(lam):
    """Sheet exchange from the IBM-well route: in-plane FT exp(-q^2/4)^2 times the z form factor F(q)."""
    f = lambda q: np.exp(-q * q / 2.0) * erfcx(q / (np.sqrt(2.0) * lam))
    return -quad(f, 0.0, np.inf, **TOL)[0]


def _exp_e1(x):
    """exp(x) E1(x), with the asymptotic series where exp(x) would overflow."""
    return np.exp(x) * exp1(x) if x < 500.0 else (1.0 - 1.0 / x + 2.0 / x ** 2 - 6.0 / x ** 3) / x


def ex_wire_ibm(lam):
    """Wire exchange from the IBM-wire route: longitudinal FT times the transverse Hankel kernel V(k)."""
    f = lambda k: np.exp(-k * k / 2.0) * _exp_e1(k * k / (2.0 * lam * lam)) / np.pi
    return -(quad(f, 0.0, 1.0 / lam, **TOL)[0] + quad(f, 1.0 / lam, np.inf, **TOL)[0])


def read_fci():
    rows = {}
    with open(os.path.join(RES, 'fci_gaussian_validation.csv')) as fh:
        for r in csv.DictReader(fh):
            rows[(r['family'], float(r['lambda']))] = float(r['ex_fci'])
    return rows


def main():
    fci = read_fci()
    out = []
    for fam, fn in (('sheet', ex_sheet_ibm), ('wire', ex_wire_ibm)):
        for lam in sorted(l for f, l in fci if f == fam):
            e = fn(lam)
            out.append((fam, lam, fci[(fam, lam)], e, e - fci[(fam, lam)]))
    with open(os.path.join(RES, 'fci_ibm_kernel_check.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['family', 'lambda', 'ex_fci', 'ex_ibm_kernel', 'ibm_minus_fci'])
        w.writerows(out)
    with open(os.path.join(RES, 'tab_fci_ibm_kernel_rows.tex'), 'w') as fh:
        fh.write('% generated by codes/19_fci_ibm_kernel_check.py\n')
        for fam, lam, ef, ek, d in out:
            ls = f'10^{{{int(round(np.log10(lam)))}}}' if lam >= 1000 else f'{lam:g}'
            fh.write(f'{fam} & ${ls}$ & ${ef:.9f}$ & ${ek:.9f}$ & ${d:+.1e}$ \\\\\n'.replace('e-', 'e{-}'))

    # ------------------------------------------------------------------ figure (publication style)
    pub_style()
    lam = np.logspace(0, 4, 121)
    es = np.array([ex_sheet_ibm(l) for l in lam])
    ew = np.array([ex_wire_ibm(l) for l in lam])
    fig, (a, b) = plt.subplots(1, 2, figsize=(PUB_W['main_wide'], 2.75),
                               gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.32))
    a.semilogx(lam, -np.sqrt(np.pi / 2) * np.ones_like(lam), color=PUB['guide'][0], ls=PUB['guide'][1], lw=1.0,
               label=r'sheet limit $-\sqrt{\pi/2}$')
    lw_ = lam[lam >= 3]
    a.semilogx(lw_, -np.sqrt(2 / np.pi) * np.log(2 * lw_), color='k', ls=(0, (1, 1.2)), lw=1.3, zorder=5,
               label=r'wire law $-\sqrt{2/\pi}\,\ln 2\lambda$')
    pub_line(a, lam, es, 'sheet', label='sheet, IBM-well kernel', lw=2.0, marker='')
    pub_line(a, lam, ew, 'wire', label='wire, IBM-wire kernel', lw=2.0, marker='')
    for fam in ('sheet', 'wire'):
        x = [r[1] for r in out if r[0] == fam]
        y = [r[2] for r in out if r[0] == fam]
        pub_line(a, x, y, fam + 'F', label=f'{fam}, FCI (Table I)', ms=6.5, mew=1.4, zorder=6)
    a.set_xlim(0.8, 1.3e4)
    a.set_ylim(-8.6, -0.3)
    a.set_xlabel(r'Confinement strength $\lambda$')
    a.set_ylabel(r'Exact exchange $E_x$ (Ha)')
    pub_panel(a, '(a)', x=0.95, ha='right')
    pub_legend(a, loc='lower left', fontsize=7)

    floor = 1e-17
    b.axhspan(1e-18, 2.2e-16, color='0.88', zorder=0, lw=0)
    for fam in ('sheet', 'wire'):
        x = [r[1] for r in out if r[0] == fam]
        y = [max(abs(r[4]), floor) for r in out if r[0] == fam]
        c, _, mk = PUB[fam + 'F']
        b.loglog(x, y, ls='-', lw=1.2, color=c, marker=mk, ms=6, mfc='white', mec=c, mew=1.4, label=fam,
                 clip_on=False, zorder=4)
    b.text(12, 3e-17, 'double-precision roundoff', fontsize=7.5, color='0.25')
    b.annotate(r'wire $\lambda=1$: 0', (1.0, floor), (1.35, 8e-17), fontsize=7.5, color='0.25',
               arrowprops=dict(arrowstyle='-', color='0.4', lw=0.6))
    b.set_ylim(1e-17, 1e-7)
    b.set_xlim(0.8, 1.3e4)
    b.set_xlabel(r'Confinement strength $\lambda$')
    b.set_ylabel(r'$|E_x^{\rm kernel}-E_x^{\rm FCI}|$ (Ha)')
    pub_panel(b, '(b)', x=0.95, ha='right')
    pub_legend(b, loc='upper left')
    pub_save(fig, os.path.join(FIG, 'fci_ibm_kernel_check'))

    print('family  lambda   E_x^FCI        E_x^IBM-kernel   diff')
    for fam, l, ef, ek, d in out:
        print(f'{fam:5s} {l:8g}  {ef:.9f}  {ek:.9f}  {d:+.1e}')
    print(f'max |diff| = {max(abs(r[4]) for r in out):.2e} Ha')
    sl = (ex_wire_ibm(1e4) - ex_wire_ibm(1e3)) / np.log(10.0)
    print(f'wire slope 1e3->1e4 = {sl:.6f}  (exact -sqrt(2/pi) = {-np.sqrt(2 / np.pi):.6f})')
    print(f'sheet E_x(1e4) = {ex_sheet_ibm(1e4):.9f}  (limit {-np.sqrt(np.pi / 2):.9f})')


if __name__ == '__main__':
    main()
