#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal (after codes 01 and 02 have written their results):
#
#     cd SCAN_1D_2D        # (the folder that contains codes/)
#     python3 codes/16_subrata_paper_figures.py
#
# Needs: numpy, pandas, matplotlib.  Figures -> codes/figures/subrata_paper/
# ----------------------------------------------------------------------
r"""The four figures of the subrata JCP manuscript (upload/JCIM/subrata/jcp_main.tex), regenerated from data.

STANDALONE SCRIPT (SCAN_1D_2D/codes): every function it uses is defined here (the enhancement factors are copied
verbatim from codes/01_wire_benchmark.py); it reads the CSV results of codes 01 and 02 (data, not code).

  geometry.pdf    Fig. 1  contours of the normalized two-electron density n = 2 det(A)^{1/2} pi^{-3/2} exp(-r^T A r)
                          in the x-z plane: isotropic (lam = 1), sheet A = diag(1,1,lam^2), wire A = diag(lam^2,lam^2,1),
                          drawn at lam = LAM_DRAW (illustration only; widths scale as 1/lam)
  branches.pdf    Fig. 5  enhancement factors vs s: SCAN alpha = 0 (Eq. 6), F_2D and F_1D (Eq. 9/10 of the paper)
  comparison.pdf  Fig. 3  |E_x| / |E_x^exact| for SCAN and DC-SCAN on the sheet and wire, from
                          results/sheet_curve.csv (code 02) and results/wire_curve.csv (code 01)
  wire_slope.pdf  Fig. 4  secant slope dE_x^DC-SCAN / d ln lam vs the upper end of each interval, from
                          results/wire_offset.csv (code 01); dashed: -sqrt(2/pi)
Outputs: figures/subrata_paper/{geometry,branches,comparison,wire_slope}.pdf/.eps/.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.special import gamma


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
OUT = os.path.join(HERE, 'figures', 'subrata_paper')
os.makedirs(OUT, exist_ok=True)
LAM_DRAW = 3.0

# ---------------------------------------------------- copied verbatim from codes/01_wire_benchmark.py
CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)        # LDA exchange constant, E_x^LDA = -CX int n^(4/3)
KF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)            # k_F = KF n^(1/3)
C_NORM = 2.0 / np.pi ** 1.5                       # n = lam^k C_NORM exp(-(u + t^2)) in reduced coordinates

# SCAN exchange (Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)); alpha = 0 branch only is needed here.
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


def fx_scan_alpha0(s):
    """SCAN F_x(s, alpha = 0) = h0x g_x(s) = 1.174 (1 - exp(-a1/sqrt(s))), g_x(0) = 1 (Eq. 6)."""
    s = np.asarray(s, dtype=float)
    with np.errstate(divide='ignore'):
        return SCAN_H0X * -np.expm1(-SCAN_A1 / np.sqrt(np.where(s > 0, s, 1e-300)))


def f2d(s):
    return A2D / (1.0 + s * s) ** 0.25


def f1d(s):
    return A1D / (1.0 + s * s)
# ---------------------------------------------------- end of verbatim copy


def save(fig, name):
    pub_save(fig, os.path.join(OUT, name))


def fig_geometry():
    x = np.linspace(-3.2, 3.2, 500)
    X, Z = np.meshgrid(x, x)
    shapes = (('(a) isotropic', (1.0, 1.0)), ('(b) sheet: thin in $z$', (1.0, LAM_DRAW ** 2)),
              ('(c) wire: thin in $x,y$', (LAM_DRAW ** 2, 1.0)))
    fig, axs = plt.subplots(1, 3, figsize=(PUB_W['main_geometry'], 2.35))
    for ax, (title, (ax_x, ax_z)) in zip(axs, shapes):
        n = 2.0 * np.sqrt(ax_x * ax_x * ax_z) * np.pi ** -1.5 * np.exp(-ax_x * X * X - ax_z * Z * Z)
        levels = np.max(n) * np.array([0.03, 0.15, 0.4, 0.75, 1.0])
        ax.contourf(X, Z, n, levels=levels, colors=['#dce8f3', '#b3cde4', '#82add2', '#4f87bb'])
        ax.contour(X, Z, n, levels=levels[:-1], colors='#1f4e7a', linewidths=0.7)
        ax.axhline(0, color='0.35', lw=0.7)
        ax.axvline(0, color='0.35', lw=0.7)
        ax.text(3.15, 0.15, '$x$', fontsize=10, color='0.15', ha='right')
        ax.text(0.18, 3.05, '$z$', fontsize=10, color='0.15', va='top')
        ax.set_aspect('equal')
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.axis('off')
        ax.text(0.5, -0.04, title, transform=ax.transAxes, ha='center', va='top', fontsize=10, fontweight='bold')
    save(fig, 'geometry')


def fig_branches():
    s = np.logspace(-1, 4, 600)
    fig, ax = plt.subplots(figsize=(PUB_W['column'], 2.75))
    pub_line(ax, s, fx_scan_alpha0(s), 'SCAN', label=r'SCAN, $\alpha=0$', lw=1.8, marks=9)
    pub_line(ax, s, f2d(s), 'sheet', label=r'sheet branch $F_{\rm 2D}$', lw=1.8, marks=9)
    pub_line(ax, s, f1d(s), 'wire', label=r'wire branch $F_{\rm 1D}$', lw=1.8, marks=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.text(300, 4e-3, r'$\propto s^{-1/2}$', color=PUB['sheet'][0], fontsize=10)
    ax.text(300, 3e-5, r'$\propto s^{-2}$', color=PUB['wire'][0], fontsize=10)
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1e-9, 3.0)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r'Reduced density gradient $s$')
    ax.set_ylabel(r'Enhancement factor $F_x$')
    pub_legend(ax, loc='lower left')
    save(fig, 'branches')


def fig_comparison(trap=True, name='comparison'):
    """trap=False: Gaussian references only (no SCAN/DC-SCAN on the correlated FCI trap densities), for manuscripts
    without the correlated-trap section (upload/JCIM/subrata/main.tex) -> comparison_gauss."""
    fig, axs = plt.subplots(1, 2, figsize=(PUB_W['main_wide'], 2.6), gridspec_kw=dict(wspace=0.25))
    for ax, fam, title in ((axs[0], 'sheet', '(a) sheet'), (axs[1], 'wire', '(b) wire')):
        d = pd.read_csv(os.path.join(RES, f'{fam}_curve.csv'))
        d = d[d['lambda'] <= 1e4]
        lam = d['lambda'].values
        pub_line(ax, lam, np.ones_like(lam), 'exact', label='exact reference')
        pub_line(ax, lam, (d['E_SCAN'] / d['E_exact']).values, 'SCAN', label='SCAN')
        pub_line(ax, lam, (d['E_DCSCAN'] / d['E_exact']).values, 'DC', label='DC-SCAN')
        # FCI checks: one-orbital FCI exchange of the same Gaussian (code 10) and the correlated two-electron traps
        # (code 20): SCAN / DC-SCAN on the exact FCI density against its exact KS exchange E_x^FCI
        f = pd.read_csv(os.path.join(RES, 'fci_gaussian_validation.csv'))
        f = f[(f['family'] == fam) & (f['lambda'] <= 1e4)]
        ex_ref = np.interp(np.log(f['lambda']), np.log(d['lambda']), d['E_exact'])
        ax.plot(f['lambda'], f['ex_fci'] / ex_ref, 'D', color='k', ms=4.5, mfc='white', mew=1.0, ls='none',
                label=r'FCI $E_x$ (Gaussian)')
        t = pd.read_csv(os.path.join(RES, 'trap_dc_exchange.csv'))
        t = t[t['family'] == fam]
        for key, col, lab in ((('SCAN', 'RxF_SCAN', 'SCAN, FCI density'), ('DC', 'RxF_DC-SCAN', 'DC-SCAN, FCI density'))
                              if trap else ()):
            c, _, mk = PUB[key]
            ax.plot(t['lam'], t[col], mk, color=c, ms=5.0, mfc=c, mec='k', mew=0.5, ls='none', label=lab)
        ax.set_xscale('log')
        ax.set_xlim(1, 1e4)
        ax.set_ylim(0.8 if fam == 'sheet' else 0.0, 2.0)   # user 2026-11-23: both panels up to 2 (SCAN leaves the panel)
        ax.set_xlabel(r'Confinement strength $\lambda$')
        pub_panel(ax, title)
    axs[0].set_ylabel(r'$E_x/E_x^{\rm exact}$')
    pub_legend(axs[0], loc='upper right', fontsize=6.5)   # 2026-11-23: free corner once the y axis stops at 2
    save(fig, name)


def fig_correlation():
    """Correlation of the correlated two-electron sheet/wire traps (code 20): LDA, PBE, SCAN correlation relative to
    the FCI references -- lines: RHF density vs E_FCI - E_HF; filled markers: exact FCI density vs exact KS E_c."""
    t = pd.read_csv(os.path.join(RES, 'trap_dc_exchange.csv'))
    t = t[t['lam'] <= 100]                      # basis-converged range (20/40/60 quanta)
    fig, axs = plt.subplots(1, 2, figsize=(PUB_W['main_wide'], 2.6), gridspec_kw=dict(wspace=0.25))
    for ax, fam, title in ((axs[0], 'sheet', '(a) sheet'), (axs[1], 'wire', '(b) wire')):
        d = t[t['family'] == fam].sort_values('lam')
        lam = d['lam'].values
        pub_line(ax, lam, np.ones_like(lam), 'exact', label='FCI reference', marks=0)
        for key in ('LDA', 'PBE', 'SCAN'):
            c, ls, mk = PUB[key]
            ax.plot(lam, d[f'Rc_{key}'], ls=ls, color=c, lw=1.5, label=f'{key}, HF density')
            ax.plot(lam, d[f'RcF_{key}'], mk, color=c, ms=5.0, mfc=c, mec='k', mew=0.5, ls='none',
                    label=f'{key}, FCI density')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.9, 110)
        ax.set_ylim(0.03 if fam == 'sheet' else 0.005, 12.0)
        ax.set_xlabel(r'Confinement strength $\lambda$')
        pub_panel(ax, title, x=0.97, ha='right')
    axs[0].set_ylabel(r'$E_c^{\rm DFA}/E_c^{\rm FCI}$')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7.5, frameon=True,
               edgecolor='k', fancybox=False)
    save(fig, 'correlation')


def fig_wire_slope():
    d = pd.read_csv(os.path.join(RES, 'wire_offset.csv')).dropna(subset=['slope_dcscan'])
    fig, ax = plt.subplots(figsize=(PUB_W['column'], 2.6))
    ax.axhline(-np.sqrt(2.0 / np.pi), color='k', ls=(0, (5, 2)), lw=1.5, label=r'exact $-\sqrt{2/\pi}$')
    c, _, mk = PUB['DC']
    ax.plot(d['lam'], d['slope_dcscan'], '-', color=c, marker=mk, ms=5.5, mfc='white', mec=c, mew=1.2, lw=1.6,
            label='DC-SCAN secant slope')
    f = pd.read_csv(os.path.join(RES, 'fci_gaussian_validation.csv'))
    f = f[f['family'] == 'wire'].sort_values('lambda')
    lf, ef = f['lambda'].values, f['ex_fci'].values
    sl = np.diff(ef) / np.diff(np.log(lf))                       # FCI secant slope between consecutive lam
    sel = lf[1:] >= 7e3
    ax.plot(lf[1:][sel], sl[sel], 'D', color='k', ms=5.0, mfc='white', mew=1.0, ls='none',
            label=r'FCI secant slope ($10^3\to10^4$)')
    ax.set_xscale('log')
    ax.set_xlim(7e3, 1.5e12)
    ax.set_ylim(-0.8125, -0.7775)
    ax.set_xlabel(r'Confinement strength $\lambda$')
    ax.set_ylabel(r'$dE_x/d\ln\lambda$ (Ha)')
    pub_legend(ax, loc='lower left')
    save(fig, 'wire_slope')


def main():
    pub_style()
    fig_geometry()
    fig_branches()
    fig_comparison()
    fig_comparison(trap=False, name='comparison_gauss')
    fig_wire_slope()
    fig_correlation()
    d = pd.read_csv(os.path.join(RES, 'wire_offset.csv')).dropna(subset=['slope_dcscan'])
    print('wire secant slopes (upper end lam, slope):')
    for lam, sl in zip(d['lam'], d['slope_dcscan']):
        print(f'  {lam:9.0e}  {sl:.6f}')
    print(f'exact -sqrt(2/pi) = {-np.sqrt(2 / np.pi):.6f};  A2D = {A2D:.10f}')
    print(f'wrote {OUT}/geometry, branches, comparison, wire_slope (.pdf/.png)')


if __name__ == '__main__':
    main()
