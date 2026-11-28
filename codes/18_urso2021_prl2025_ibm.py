#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd SCAN_1D_2D        # (the folder that contains codes/)
#     ~/miniconda3/envs/anigto/bin/python codes/18_urso2021_prl2025_ibm.py
#
# Needs: numpy, scipy, matplotlib, pyscf (libxc), poppler (pdftocairo).  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""IBM quasi-2D / quasi-1D figures of Urso & Constantin, EPJB 94, 147 (2021) and the slab exchange asymptote of
Horowitz, Proetto & Pitarke, PRL 134, 256401 (2025), recomputed with SCAN_1D_2D codes, plus DC-SCAN / DC-r2SCAN.

NOTE: the EPJB article PDF (upload/JCIM/subrata/s10051-021-00159-y.pdf) carries a "RETRACTED ARTICLE" watermark.
Our exact references are computed independently here; the paper's curves are used only for comparison.

STANDALONE SCRIPT (SCAN_1D_2D/codes).  Copied verbatim: the DC-SCAN block of codes/01_wire_benchmark.py (via 14);
eps_x_2d, lmax, _autocorr_table, form_factor, well_exact_exchange from codes/14; J01/J11/J1J, rmax, WireExact from
codes/17; _matrix, stroked_polylines from codes/15.

Urso Fig. 3   quasi-2D IBM eps_x vs L/Lmax (rs2D = 2, 5): exact (single-subband, code 14 route), MVS, SCAN, Q2D GGA
              (Libxc, true tau) + DC-SCAN, DC-r2SCAN.
Urso Fig. 4   quasi-1D IBM density, s, q vs rho (kF1D = 0.5; L = Lmax/2, /5, /10) + our D descriptor and Omega.
Urso Fig. 7   F_x(s): Q1D GGA (Eq. 30, coded here), Q2D, PBEsol, MVS/SCAN (alpha = 1) + DC branches F_2D, F_1D.
Urso Fig. 8   quasi-1D IBM eps_x vs L/Lmax (kF1D = 2, 0.5): exact (V(qR) route, code 17), MVS, SCAN, Q2D, Q1D
              + DC-SCAN, DC-r2SCAN.  Lmax = sqrt(x02^2 - x01^2)/kF (paper Eq. 18); the m = +-1 subband makes the
              strict one-subband range L < 0.600 Lmax (all points shown in the paper, L <= 0.5 Lmax, satisfy it).
PRL 2025      exact conventional-gauge eps_x(z) of single-subband slabs, d = 0.05 lambda_F, rs = 1..6:
              eps_x(z) = -k_F int phi(z')^2 g(k_F|z - z'|) dz',  g(y) = int J1(x)^2 / (x sqrt(x^2 + y^2)) dx,
              -> -1/(2z) for z >> d (PRL Eq. 1 for slabs, universal); checked: int phi^2 eps_x = well exact exchange.
              Inside the slab compared with LDA, SCAN, DC-SCAN, DC-r2SCAN energies per particle (gauge-dependent).
The paper figures are vector graphics: every stroked curve is digitized (pdftocairo -svg), calibrated on the
frame ticks, and matched to the closest curve of this work (rms printed).
Outputs: results/urso2021_fig{3,4,7,8}*.csv, urso2021_figure_comparison.csv, prl2025_*.csv, urso2021_prl2025_log.txt;
         figures/urso2021_fig{3,4,7,8}.pdf/.png, prl2025_slab_epsx.pdf/.png
"""
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, j0, j1, jn_zeros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf.dft import libxc

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

# methods of this script that have no colour in the shared PUB table (kept outside the identical copy above)
PUB.update({
    'MVS': ('#6e6e6e', (0, (1, 1.3)), 's'),
    'Q2D': ('#3aa6d8', (0, (5, 1.5, 1, 1.5)), 'X'),
    'Q1D': ('#a03fb0', (0, (3, 1, 1, 1, 1, 1)), 'h'),
    'HPP19': ('#2ca02c', (0, (6, 1.5)), 'o'),               # Horowitz-Proetto-Pitarke Eq. 19 MGGA, green
    'DCSCAN': ('#d62728', '-', 'D'),                       # DC-SCAN in red (user request 2026-11-19)
    'DCr2SCAN': ('#7f0000', (0, (5, 1.5, 1, 1.5)), 'P'),   # DC-r2SCAN dark red
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)            # LDA exchange: e_x = -CX n^{4/3} (unpolarized)

# ==================================================== copied verbatim from codes/01_wire_benchmark.py (DC-SCAN)
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


def _log_gl(lo, hi, n):
    """Gauss-Legendre nodes in log coordinates: x in [lo, hi], returns x and weights for dx."""
    p, w = np.polynomial.legendre.leggauss(n)
    a, b = np.log(lo), np.log(hi)
    q = 0.5 * (b - a) * p + 0.5 * (b + a)
    x = np.exp(q)
    return x, 0.5 * (b - a) * w * x
# ==================================================== end of verbatim copy


# ==================================================== copied verbatim from codes/14_constantin2008_benchmark.py
def eps_x_2d(rs):
    """Eq. (1)."""
    return -4.0 * math.sqrt(2.0) / (3.0 * math.pi * rs)


def lmax(rs):
    """Single-subband condition L < Lmax = sqrt(3/2) pi rs2D (paper text after Eq. 6)."""
    return math.sqrt(1.5) * math.pi * rs


def _autocorr_table(m=400):
    """C(s) = int_0^{1-s} rho(u) rho(u+s) du for rho = 2 sin^2(pi u), on Gauss-Legendre s in (0,1)."""
    xs, ws = np.polynomial.legendre.leggauss(m)
    s = 0.5 * (xs + 1.0)
    ws = 0.5 * ws
    xu, wu = np.polynomial.legendre.leggauss(m)
    C = np.empty_like(s)
    for i, si in enumerate(s):
        u = 0.5 * (1.0 - si) * (xu + 1.0)
        C[i] = 0.5 * (1.0 - si) * np.sum(wu * 4.0 * np.sin(np.pi * u) ** 2 * np.sin(np.pi * (u + si)) ** 2)
    return s, ws, C


_S, _WS, _C = _autocorr_table()


def form_factor(Q):
    """F(Q) = 2 int_0^1 C(s) e^{-Q s} ds  (F(0) = 1)."""
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    return 2.0 * np.sum(_WS * _C * np.exp(-np.outer(Q, _S)), axis=1)


def well_exact_exchange(rs, L, n=400):
    """Exact exchange per electron of the single-subband IBM gas."""
    kf = math.sqrt(2.0) / rs
    n2d = 1.0 / (math.pi * rs * rs)
    x, w = np.polynomial.legendre.leggauss(n)
    xq = 0.5 * (x + 1.0)                             # q / (2 kF)
    A = 2.0 * kf * kf * (np.arccos(xq) - xq * np.sqrt(1.0 - xq * xq))
    q = 2.0 * kf * xq
    F = form_factor(q * L) if L > 0 else np.ones_like(q)
    return -float(np.sum(0.5 * w * 2.0 * kf * F * A)) / (4.0 * np.pi ** 2 * n2d)
# ==================================================== end of copy (14)


# ==================================================== copied verbatim from codes/17_ibm_quasi_1d_2d.py
J01 = float(jn_zeros(0, 1)[0])
J11 = float(jn_zeros(1, 1)[0])
J1J = float(j1(J01))


def rmax(rs1d):
    kf = math.pi / (4.0 * rs1d)
    return math.sqrt(J11 ** 2 - J01 ** 2) / kf


class WireExact:
    """Exact single-subband exchange of the IBM wire; V(Q) tabulated once (depends on Q = qR only)."""

    def __init__(self, nu=4000, xmax=3000.0):
        u, wu = np.polynomial.legendre.leggauss(nu)
        u, wu = 0.5 * (u + 1.0), 0.5 * wu
        dens = 2.0 * u * j0(J01 * u) ** 2 / J1J ** 2          # transverse radial density, int_0^1 = 1
        self.norm = float(np.sum(wu * dens))
        xs, wx = _log_gl(1e-20, 1.0, 900)
        g, wg = np.polynomial.legendre.leggauss(24)
        edges = np.linspace(1.0, xmax, 1200)
        xl = np.concatenate([0.5 * (b - a) * g + 0.5 * (b + a) for a, b in zip(edges[:-1], edges[1:])])
        wl = np.concatenate([0.5 * (b - a) * wg for a, b in zip(edges[:-1], edges[1:])])
        self.x = np.concatenate((xs, xl))
        self.wx = np.concatenate((wx, wl))
        F = np.empty_like(self.x)
        for k in range(0, len(self.x), 2000):
            F[k:k + 2000] = j0(np.outer(self.x[k:k + 2000], u)) @ (wu * dens)
        self.F = F
        self.lnQ = np.linspace(np.log(1e-18), np.log(60.0), 1600)
        Q = np.exp(self.lnQ)
        xf = self.x * self.F ** 2 * self.wx
        self.V = np.array([2.0 * np.sum(xf / (q * q + self.x ** 2)) for q in Q])
        # small-Q law V = -2 ln Q + v0
        self.v0 = float(self.V[0] + 2.0 * self.lnQ[0])

    def eps_x(self, rs1d, R, nq=500):
        n1d = 1.0 / (2.0 * rs1d)
        kf = math.pi * n1d / 2.0
        q, wq = _log_gl(2.0 * kf * 1e-16, 2.0 * kf, nq)
        lnQ = np.log(q * R)
        V = np.where(lnQ < self.lnQ[0], -2.0 * lnQ + self.v0, np.interp(lnQ, self.lnQ, self.V))
        return float(-np.sum(wq * (2.0 * kf - q) * V) / (2.0 * math.pi ** 2 * n1d))
# ==================================================== end of copy (17)


# ==================================================== copied verbatim from codes/15_constantin2008_figure_compare.py
def _matrix(tr):
    M = np.eye(3)
    for kind, args in re.findall(r'(matrix|translate|scale)\(([^)]*)\)', tr or ''):
        v = [float(x) for x in re.split(r'[ ,]+', args.strip())]
        if kind == 'matrix':
            m = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif kind == 'translate':
            m = np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0], [0, 0, 1]])
        else:
            m = np.array([[v[0], 0, 0], [0, v[1] if len(v) > 1 else v[0], 0], [0, 0, 1]])
        M = M @ m
    return M


def stroked_polylines(svg_file):
    """All stroked sub-paths (end points of M/L/C segments) in page coordinates, with their style string."""
    out = []

    def walk(el, M):
        M2 = M @ _matrix(el.get('transform'))
        if el.tag.split('}')[-1] == 'path':
            st = (el.get('style') or '') + ' ' + ' '.join(
                f'{k}:{v}' for k, v in el.attrib.items() if k in ('stroke', 'stroke-dasharray'))
            if 'stroke:' in st and 'stroke:none' not in st.replace(' ', ''):
                toks = re.findall(r'[MLCZmlcz]|-?\d*\.?\d+(?:e-?\d+)?', el.get('d'))
                subs, sub, nums, cmd = [], [], [], None
                for tk in toks:
                    if tk in 'MLCZmlcz':
                        cmd = tk
                        if tk in 'Mm' and sub:
                            subs.append(sub)
                            sub = []
                        continue
                    nums.append(float(tk))
                    if len(nums) == {'M': 2, 'L': 2, 'C': 6}.get(cmd, 2):
                        sub.append((nums[-2], nums[-1]))
                        nums = []
                if sub:
                    subs.append(sub)
                for s in subs:
                    a = np.c_[np.array(s), np.ones(len(s))] @ M2.T
                    out.append((a[:, :2], st))
        for ch in el:
            walk(ch, M2)

    walk(ET.parse(svg_file).getroot(), np.eye(3))
    return out
# ==================================================== end of copy (15)


# ==================================================== Libxc helpers and functionals on the IBM densities (this script)
TAU_UNIF = 0.3 * KF ** 2                               # tau_unif = TAU_UNIF n^{5/3}
Q1D_A = 0.06525                                        # Urso & Constantin Eq. (30)


def libxc_fx(code, rho, g, lap, tau):
    """Enhancement factor e_x / e_x^LDA of a Libxc exchange functional (unpolarized)."""
    z = np.zeros_like(rho)
    if code.startswith('MGGA'):
        inp = np.vstack((rho, g, z, z, lap, tau))
    else:
        inp = np.vstack((rho, g, z, z))
    e = np.asarray(libxc.eval_xc(code + ',', inp, spin=0, deriv=0)[0])
    return e / (-CX * np.cbrt(rho))


def fx_q1d(s, f_pbesol):
    """Q1D GGA, Urso & Constantin Eq. (30): F = F_PBEsol + w(s) (-F_PBEsol + a/s^2), w = (s^4+s^6)/(1+s^4+s^6)."""
    s2 = s * s
    w = (s2 * s2 + s2 ** 3) / (1.0 + s2 * s2 + s2 ** 3)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        tail = np.where(s > 0, Q1D_A / np.where(s > 0, s2, 1.0), 0.0)
    return f_pbesol + w * (-f_pbesol + tail)


def fx_hpp19(s, alpha):
    """MGGA exchange enhancement factor of Horowitz, Proetto & Pitarke, PRB 108, 115119 (2023), Eq. (19):
    F = 2.7 a^(1/2) / [1 + 2.7 a (0.8788 + s) - A(s) a^B(s) + 0.924 ln(1 + a^3) e^(-10 s)],
    A(s) = 2.5 / (5 + s)^0.4,  B(s) = 0.96 exp(-0.5 s^0.3);  F(s=0, a=1) = 1.000, F ~ 2.7 a^(1/2) for a -> 0."""
    a = np.maximum(alpha, 0.0)
    A = 2.5 / (5.0 + s) ** 0.4
    B = 0.96 * np.exp(-0.5 * s ** 0.3)
    den = 1.0 + 2.7 * a * (0.8788 + s) - A * a ** B + 0.924 * np.log1p(a ** 3) * np.exp(-10.0 * s)
    return 2.7 * np.sqrt(a) / den


def all_fx(rho, g, lap, tau, s, D, Z):
    f = {k: libxc_fx(c, rho, g, lap, tau) for k, c in
         (('MVS', 'MGGA_X_MVS'), ('SCAN', 'MGGA_X_SCAN'), ('r2SCAN', 'MGGA_X_R2SCAN'), ('Q2D', 'GGA_X_Q2D'),
          ('PBEsol', 'GGA_X_PBE_SOL'))}
    f['Q1D'] = fx_q1d(s, f['PBEsol'])
    alpha = (tau - g * g / (8.0 * rho)) / (0.3 * KF ** 2 * rho ** (5.0 / 3.0))     # (tau - tau_W) / tau_unif
    f['HPP19'] = fx_hpp19(s, alpha)
    f['LDA'] = np.ones_like(rho)
    f['DCSCAN'] = fx_dcscan(s, D, Z, f['SCAN'], 'full')
    f['DCr2SCAN'] = fx_dcscan(s, D, Z, f['r2SCAN'], 'full')
    return f


def well_fields(rs, L, n=700):
    """IBM quasi-2D well (as code 14 well_dcscan): grid, density, s, tau, D (= 1), Z."""
    n2d = 1.0 / (math.pi * rs * rs)
    kf2d = math.sqrt(2.0) / rs
    ua, wa = _log_gl(1e-14, 0.25, n)
    vb, wb = _log_gl(1e-14, 0.25, n)
    u = np.concatenate((ua, 0.5 - vb))
    w = 2.0 * L * np.concatenate((wa, wb))
    amp = 2.0 * n2d / L
    sn, cs = np.sin(np.pi * u), np.cos(np.pi * u)
    rho = amp * sn * sn
    g = np.abs(amp * (np.pi / L) * 2.0 * sn * cs)
    lap = amp * 2.0 * (np.pi / L) ** 2 * np.cos(2.0 * np.pi * u)
    tau = g * g / (8.0 * rho) + kf2d ** 4 / (4.0 * np.pi * L) * sn * sn
    kf = KF * np.cbrt(rho)
    s = g / (2.0 * kf * rho)
    czz = (np.pi / L) ** 2 / (sn * sn)
    D = np.ones_like(rho)
    Z = czz / kf ** 2
    return dict(u=u, w=w / n2d, rho=rho, g=g, lap=lap, tau=tau, s=s, D=D, Z=Z)


def wire_fields(kf1d, L, n=700):
    """IBM quasi-1D wire of radius L (Urso Eq. 20): grid, density, s, q, tau, D, Z (as code 17 wire_semilocal)."""
    n1d = 2.0 * kf1d / math.pi
    ua, wa = _log_gl(1e-13, 0.5, n)
    vb, wb = _log_gl(1e-13, 0.5, n)
    u = np.concatenate((ua, 1.0 - vb))
    w = 2.0 * math.pi * L * L * u * np.concatenate((wa, wb))
    x = J01 * u
    J0, J1 = j0(x), j1(x)
    amp = n1d / (math.pi * L * L * J1J ** 2)
    k = J01 / L
    rho = amp * J0 * J0
    g = np.abs(amp * k * 2.0 * J0 * J1)
    lap = amp * k * k * (2.0 * J1 * J1 - 2.0 * J0 * J0)
    tau = g * g / (8.0 * rho) + rho * kf1d ** 2 / 6.0
    kf = KF * np.cbrt(rho)
    s = g / (2.0 * kf * rho)
    rt = J1 / J0
    rx = np.where(x > 1e-6, J1 / (x * J0), 0.5 - x * x / 16.0)
    crr = k * k * (1.0 - rx + rt * rt)
    cpp = k * k * rx
    D = np.clip(np.where(crr != 0.0, (crr * crr + cpp * cpp) / np.where(crr != 0.0, crr * crr, 1.0), 3.0), 1.0, 3.0)
    Z = np.sqrt(crr * crr + cpp * cpp) / kf ** 2
    q = lap / (4.0 * KF ** 2 * rho ** (5.0 / 3.0))
    return dict(u=u, w=w / n1d, rho=rho, g=g, lap=lap, tau=tau, s=s, q=q, D=D, Z=Z)


def energies(fd):
    f = all_fx(fd['rho'], fd['g'], fd['lap'], fd['tau'], fd['s'], fd['D'], fd['Z'])
    e_lda = -CX * np.cbrt(fd['rho'])
    out = {k: float(np.sum(fd['w'] * fd['rho'] * e_lda * v)) for k, v in f.items()}
    out['N_check'] = float(np.sum(fd['w'] * fd['rho']))
    return out


def lmax_wire_urso(kf1d):
    """Urso Eq. (18): Lmax = sqrt(x02^2 - x01^2) / kF1D (second J0 subband).  The m = +-1 subband (j_{1,1}) lies
    lower; the strict one-subband limit is sqrt(j11^2 - j01^2)/kF = 0.600 Lmax."""
    return math.sqrt(J02 ** 2 - J01 ** 2) / kf1d


J02 = float(jn_zeros(0, 2)[1])


# ==================================================== exact conventional-gauge eps_x(z) of a single-subband slab
def g_slab_table():
    """g(y) = int_0^inf J1(x)^2 / (x sqrt(x^2 + y^2)) dx;  g(0) = 4/(3 pi), g(y) -> 1/(2y)."""
    ys = np.concatenate(([0.0], np.logspace(-5, 5, 400)))
    gv = []
    for y in ys:
        f = lambda x: j1(x) ** 2 / (x * math.sqrt(x * x + y * y))
        tot, a = 0.0, 0.0
        for b in np.concatenate((np.linspace(0.5, 50.0, 100), np.logspace(np.log10(60.0), 6, 200))):
            tot += quad(f, a, b, limit=200, epsabs=1e-14, epsrel=1e-11)[0]
            a = b
        tot += 1.0 / (2.0 * np.pi) * (1.0 / 1e6) ** 2 * 0.0       # tail ~ x^-3 cos^2: < 1e-12, neglected
        gv.append(tot)
    return ys, np.array(gv)


def slab_eps_x_profile(rs2d, L, z, gtab, n=600):
    """Exact exchange energy per particle at position z (conventional gauge) for the single-subband IBM slab on
    [0, L]:  eps_x(z) = -k_F int phi(z')^2 g(k_F |z - z'|) dz'  (n(z) cancels: valid at any z, also outside)."""
    kf = math.sqrt(2.0) / rs2d
    x, w = np.polynomial.legendre.leggauss(n)
    zp = 0.5 * L * (x + 1.0)
    wp = 0.5 * L * w * (2.0 / L) * np.sin(np.pi * zp / L) ** 2
    ys, gv = gtab
    Y = kf * np.abs(np.subtract.outer(np.atleast_1d(z), zp))
    G = np.where(Y > 1e5, 0.5 / np.maximum(Y, 1e-300), np.interp(np.log(np.maximum(Y, 1e-5)), np.log(ys[1:]), gv[1:]))
    G = np.where(Y < 1e-5, gv[0], G)
    return -kf * (G @ wp)


# ==================================================== digitizing the vector figures of the paper
URSO = os.path.join(ROOT, 'upload', 'JCIM', 'subrata', 's10051-021-00159-y.pdf')
# panel: page, frame (x0, x1, y0_top, y1_bottom), x tick range (first, last), y tick range (top, bottom),
#        y pixel of the top and bottom tick (None -> frame edges)
PANELS = {
    'Fig. 3 (rs2D=2)': (3, (344.6, 538.9, 68.1, 184.9), (0, 1), (0.0, -1.2), None),
    'Fig. 3 (rs2D=5)': (3, (349.2, 538.9, 232.4, 349.2), (0, 1), (-0.05, -0.45), None),
    'Fig. 4 (n)': (4, (348.0, 537.3, 68.3, 192.4), (0, 5), (0.4, 0.0), None),
    'Fig. 4 (s, q)': (4, (343.5, 537.3, 235.0, 359.1), (0, 4), (3.0, -1.0), None),
    'Fig. 7': (6, (89.5, 283.3, 460.5, 584.6), (0, 5), (1.8, 0.0), None),
    'Fig. 8 (kF=2)': (6, (339.4, 534.5, 67.6, 182.2), (0, 0.5), (0.0, -14.0), (67.6, 174.6)),
    'Fig. 8 (kF=0.5)': (6, (343.9, 534.5, 226.1, 340.7), (0, 0.5), (0.0, -4.0), None),
}


def digitize_panel(polys, frame, xr, yr, ytick_px):
    """Merge every stroked polyline of one style inside the frame into one curve (legend samples and ticks,
    i.e. 2-point straight segments, are dropped).  Returns {style: (X, Y)} in data units."""
    x0, x1, y0, y1 = frame
    ya, yb = ytick_px if ytick_px else (y0, y1)
    groups = {}
    for a, st in polys:
        if len(a) < 3:
            continue
        if not (x0 - 0.5 <= a[:, 0].min() and a[:, 0].max() <= x1 + 0.5 and y0 - 0.5 <= a[:, 1].min()
                and a[:, 1].max() <= y1 + 0.5):
            continue
        if np.ptp(a[:, 0]) > 0.97 * (x1 - x0) and np.ptp(a[:, 1]) > 0.97 * (y1 - y0):
            continue                                       # the frame itself
        key = re.sub(r'stroke-width:[^;]+;?', '', st)
        key = (re.search(r'stroke:\s*([^;]+)', key).group(1).strip() if 'stroke:' in key else '?',
               re.search(r'stroke-dasharray:\s*([^;]+)', key).group(1).split()[0][:4] if 'dasharray' in key else 'solid')
        groups.setdefault(key, []).append(a)
    out = {}
    for key, arrs in groups.items():
        a = np.vstack(arrs)
        X = xr[0] + (a[:, 0] - x0) / (x1 - x0) * (xr[1] - xr[0])
        Y = yr[0] + (a[:, 1] - ya) / (yb - ya) * (yr[1] - yr[0])
        o = np.argsort(X)
        out[key] = (X[o], Y[o])
    return out


def match(digit, curves, yspan):
    """Assign each digitized style to the closest of our curves {name: (x, y)} (rms); unmatched if > 4 % of span."""
    res = []
    for key, (X, Y) in digit.items():
        best = None
        for name, (x, y) in curves.items():
            m = (X >= x.min()) & (X <= x.max())
            if m.sum() < 5:
                continue
            d = Y[m] - np.interp(X[m], x, y)
            r = float(np.sqrt(np.mean(d * d)))
            if best is None or r < best[1]:
                best = (name, r, float(np.max(np.abs(d))), int(m.sum()))
        if best is None or best[1] > 0.04 * yspan:
            res.append((key, None, best, X, Y))
        else:
            res.append((key, best[0], best, X, Y))
    return res


# ==================================================== driver
def write_csv(path, cols):
    keys = list(cols)
    np.savetxt(path, np.column_stack([np.asarray(cols[k], dtype=float) for k in keys]), delimiter=',',
               header=','.join(keys), comments='', fmt='%.12g')


STY = {'exact': dict(color='k', lw=2.0), 'LDA': dict(color='0.6', lw=1.0),
       'MVS': dict(color='#2ca02c', lw=1.2, ls='--'), 'SCAN': dict(color='#1f77b4', lw=1.2, ls=':'),
       'Q2D': dict(color='#c71585', lw=1.2, ls='-.'), 'Q1D': dict(color='#17becf', lw=1.2, ls=(0, (6, 2, 1, 2))),
       'DCSCAN': dict(color='#d62728', lw=1.8), 'DCr2SCAN': dict(color='#ff7f0e', lw=1.8, ls=(0, (1, 1.2)))}
LAB = {'exact': 'exact reference (this work)', 'LDA': 'LDA', 'MVS': 'MVS', 'SCAN': 'SCAN', 'Q2D': 'Q2D GGA',
       'Q1D': 'Q1D GGA', 'DCSCAN': 'DC-SCAN (this work)', 'DCr2SCAN': r'DC-r$^2$SCAN (this work)'}


def overlay(ax, res, colors):
    for key, name, best, X, Y in res:
        if name is None:
            continue
        st = max(1, len(X) // 40)
        ax.plot(X[::st], Y[::st], 'o', ms=3.2, mfc='none', mew=0.7, color=colors.get(name, 'k'))
    ax.plot([], [], 'o', ms=3.2, mfc='none', mew=0.7, color='k', label='Urso & Constantin 2021 (digitized)')


def main():
    warnings.filterwarnings('ignore')
    log = []
    say = lambda t='': (print(t, flush=True), log.append(t))
    say('Urso & Constantin, EPJB 94, 147 (2021) [RETRACTED] and Horowitz-Proetto-Pitarke, PRL 134, 256401 (2025):')
    say('IBM quasi-2D / quasi-1D figures with SCAN_1D_2D codes + DC-SCAN / DC-r2SCAN (script 18)')
    data = {}

    # ---------------- Fig. 3: quasi-2D eps_x vs L/Lmax, rs2D = 2, 5
    xs = np.concatenate((np.logspace(-3, -1, 25), np.linspace(0.11, 1.0, 46)))
    for rs in (2.0, 5.0):
        Lm = lmax(rs)
        rows = {k: [] for k in ('x', 'L', 'exact', 'LDA', 'MVS', 'SCAN', 'r2SCAN', 'Q2D', 'PBEsol', 'Q1D',
                                'HPP19', 'DCSCAN', 'DCr2SCAN', 'N_check')}
        for x in xs:
            e = energies(well_fields(rs, x * Lm))
            rows['x'].append(x)
            rows['L'].append(x * Lm)
            rows['exact'].append(well_exact_exchange(rs, x * Lm))
            for k in rows:
                if k in e:
                    rows[k].append(e[k])
        rows = {k: np.array(v) for k, v in rows.items()}
        data[f'fig3_rs{rs:g}'] = rows
        write_csv(os.path.join(RES, f'urso2021_fig3_rs{rs:g}.csv'), rows)
        say(f'Fig. 3 rs2D={rs:g}: Lmax={Lm:.4f}; eps_x^2D={eps_x_2d(rs):.5f}; max|N-1|={np.max(np.abs(rows["N_check"] - 1)):.1e}')
        for x in (1.0, 0.5, 0.1, 0.01, 0.001):
            i = int(np.argmin(np.abs(np.log(rows['x'] / x))))
            say('   L/Lmax=%6.3f exact %.5f MVS %.5f SCAN %.5f Q2D %.5f DC-SCAN %.5f DC-r2SCAN %.5f'
                % tuple([rows[k][i] for k in ('x', 'exact', 'MVS', 'SCAN', 'Q2D', 'DCSCAN', 'DCr2SCAN')]))

    # ---------------- Fig. 8: quasi-1D eps_x vs L/Lmax, kF1D = 2, 0.5
    wx = WireExact()
    xs1 = np.concatenate((np.logspace(-3, -1, 25), np.linspace(0.11, 0.5, 30)))
    for kf in (2.0, 0.5):
        Lm = lmax_wire_urso(kf)
        rows = {k: [] for k in ('x', 'L', 'exact', 'LDA', 'MVS', 'SCAN', 'r2SCAN', 'Q2D', 'PBEsol', 'Q1D',
                                'HPP19', 'DCSCAN', 'DCr2SCAN', 'N_check')}
        for x in xs1:
            e = energies(wire_fields(kf, x * Lm))
            rows['x'].append(x)
            rows['L'].append(x * Lm)
            rows['exact'].append(wx.eps_x(math.pi / (4.0 * kf), x * Lm))
            for k in rows:
                if k in e:
                    rows[k].append(e[k])
        rows = {k: np.array(v) for k, v in rows.items()}
        data[f'fig8_kf{kf:g}'] = rows
        write_csv(os.path.join(RES, f'urso2021_fig8_kf{kf:g}.csv'), rows)
        say(f'Fig. 8 kF1D={kf:g}: Lmax(Urso, x02)={Lm:.4f}; strict one-subband limit {math.sqrt(J11**2-J01**2)/kf:.4f}'
            f' = {math.sqrt(J11**2-J01**2)/math.sqrt(J02**2-J01**2):.3f} Lmax; max|N-1|={np.max(np.abs(rows["N_check"] - 1)):.1e}')
        for x in (0.5, 0.2, 0.05, 0.01, 0.001):
            i = int(np.argmin(np.abs(np.log(rows['x'] / x))))
            say('   L/Lmax=%6.3f exact %.4f MVS %.4f SCAN %.4f Q2D %.4f Q1D %.4f DC-SCAN %.4f DC-r2SCAN %.4f'
                % tuple([rows[k][i] for k in ('x', 'exact', 'MVS', 'SCAN', 'Q2D', 'Q1D', 'DCSCAN', 'DCr2SCAN')]))

    # ---------------- Fig. 4: quasi-1D density, s, q (+ our D and Omega), kF1D = 0.5
    kf = 0.5
    Lm = lmax_wire_urso(kf)
    f4 = {}
    for div in (2, 5, 10):
        fd = wire_fields(kf, Lm / div, n=300)
        o = np.argsort(fd['u'])
        rho_r = fd['u'][o] * Lm / div
        om = omega_switch(fd['s'][o], fd['Z'][o], 'full')
        f4[div] = dict(r=rho_r, n=fd['rho'][o], s=fd['s'][o], q=fd['q'][o], D=fd['D'][o], Omega=om)
        write_csv(os.path.join(RES, f'urso2021_fig4_L{div}.csv'), f4[div])
    say(f'Fig. 4: kF1D=0.5, Lmax={Lm:.4f}; n(0) at L=Lmax/10: {f4[10]["n"][0]:.4f}; q(0) = {f4[2]["q"][0]:.4f} '
        f'(paper: -0.365/(kF L)^(2/3) = {-0.365 / (kf * Lm / 2) ** (2 / 3):.4f})')

    # ---------------- Fig. 7: enhancement factors
    sgrid = np.linspace(1e-4, 5.0, 500)
    rho = np.ones_like(sgrid)
    g = sgrid * 2.0 * KF
    tw = g * g / 8.0
    f7 = {'s': sgrid}
    for a in (0.0, 1.0):
        tau = tw + a * TAU_UNIF
        for k, c in (('MVS', 'MGGA_X_MVS'), ('SCAN', 'MGGA_X_SCAN'), ('r2SCAN', 'MGGA_X_R2SCAN')):
            f7[f'{k}_a{a:g}'] = libxc_fx(c, rho, g, np.zeros_like(rho), tau)
    f7['PBEsol'] = libxc_fx('GGA_X_PBE_SOL', rho, g, 0 * rho, tw)
    f7['Q2D'] = libxc_fx('GGA_X_Q2D', rho, g, 0 * rho, tw)
    f7['Q1D'] = fx_q1d(sgrid, f7['PBEsol'])
    f7['F2D'] = f2d(sgrid)
    f7['F1D'] = f1d(sgrid)
    write_csv(os.path.join(RES, 'urso2021_fig7.csv'), f7)

    # ---------------- digitize the paper and match
    tmp = tempfile.mkdtemp()
    polys = {}
    for page in (3, 4, 6):
        svg = os.path.join(tmp, f'p{page}.svg')
        subprocess.run(['pdftocairo', '-svg', '-f', str(page), '-l', str(page), URSO, svg], check=True)
        polys[page] = stroked_polylines(svg)
    cand = {
        'Fig. 3 (rs2D=2)': {k: (data['fig3_rs2']['x'], data['fig3_rs2'][k]) for k in ('exact', 'MVS', 'SCAN', 'Q2D')},
        'Fig. 3 (rs2D=5)': {k: (data['fig3_rs5']['x'], data['fig3_rs5'][k]) for k in ('exact', 'MVS', 'SCAN', 'Q2D')},
        'Fig. 8 (kF=2)': {k: (data['fig8_kf2']['x'], data['fig8_kf2'][k]) for k in ('exact', 'MVS', 'SCAN', 'Q2D', 'Q1D')},
        'Fig. 8 (kF=0.5)': {k: (data['fig8_kf0.5']['x'], data['fig8_kf0.5'][k]) for k in ('exact', 'MVS', 'SCAN', 'Q2D', 'Q1D')},
        'Fig. 4 (n)': {f'n L=Lmax/{d}': (f4[d]['r'], f4[d]['n']) for d in (2, 5, 10)},
        'Fig. 4 (s, q)': dict([(f's L=Lmax/{d}', (f4[d]['r'], f4[d]['s'])) for d in (2, 5, 10)] +
                              [(f'q L=Lmax/{d}', (f4[d]['r'], f4[d]['q'])) for d in (2, 5, 10)]),
        'Fig. 7': {'Q1D': (sgrid, f7['Q1D']), 'Q2D': (sgrid, f7['Q2D']), 'PBEsol': (sgrid, f7['PBEsol']),
                   'MVS alpha=1': (sgrid, f7['MVS_a1']), 'SCAN alpha=1': (sgrid, f7['SCAN_a1'])},
    }
    matches, cmp_rows = {}, []
    say('')
    say('digitized published curves vs this work (rms, max in figure units; pt = one printed point):')
    for panel, (page, frame, xr, yr, ytp) in PANELS.items():
        dig = digitize_panel(polys[page], frame, xr, yr, ytp)
        span = abs(yr[1] - yr[0])
        res = match(dig, cand[panel], span)
        matches[panel] = res
        pt = span / ((ytp[1] - ytp[0]) if ytp else (frame[3] - frame[2]))
        for key, name, best, X, Y in res:
            if name is None:
                continue
            say(f'  {panel:16s} {str(key):42s} -> {name:14s} rms {best[1]:.4f} max {best[2]:.4f} ({best[3]} pts; pt {pt:.4f})')
            cmp_rows.append((panel, name, best[1], best[2], best[3], pt))
    with open(os.path.join(RES, 'urso2021_figure_comparison.csv'), 'w') as fh:
        fh.write('panel,curve,rms,max,points,pt\n')
        for r in cmp_rows:
            fh.write('%s,%s,%.6g,%.6g,%d,%.6g\n' % r)

    # ---------------- PRL 2025 insight: exact eps_x(z) of single-subband slabs, d = 0.05 lambda_F
    gtab = g_slab_table()
    say('')
    say(f'g(y): g(0) = {gtab[1][0]:.10f} (4/(3 pi) = {4 / (3 * math.pi):.10f});  2 y g(y) at y = 1e5: {2e5 * gtab[1][-1]:.6f}')
    prl = {}
    for rs3 in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        kf3 = (9.0 * math.pi / 4.0) ** (1.0 / 3.0) / rs3
        lamF = 2.0 * math.pi / kf3
        d = 0.05 * lamF
        n2d = d * 3.0 / (4.0 * math.pi * rs3 ** 3)
        rs2d = 1.0 / math.sqrt(math.pi * n2d)
        zz = np.concatenate((np.linspace(0.0, d, 60), d + lamF * np.logspace(-3, 3, 200)))
        eps = slab_eps_x_profile(rs2d, d, zz, gtab)
        x, w = np.polynomial.legendre.leggauss(400)
        zq = 0.5 * d * (x + 1.0)
        tot = float(np.sum(0.5 * d * w * (2.0 / d) * np.sin(np.pi * zq / d) ** 2 * slab_eps_x_profile(rs2d, d, zq, gtab)))
        ref = well_exact_exchange(rs2d, d)
        zs = zz - d                                     # vacuum side z > 0, slab in [-d, 0]
        prl[rs3] = dict(z=zs, z_over_lamF=zs / lamF, eps=eps, lamF=lamF, d=d, rs2d=rs2d)
        say(f'  rs={rs3:g}: d={d:.4f} bohr, rs2D={rs2d:.3f}, int phi^2 eps_x(z) = {tot:.8f} vs well exact {ref:.8f}; '
            f'-2 z eps_x at z = 1, 10, 1000 lamF: ' + ', '.join(f'{-2 * (zz[i] - d) * eps[i]:.4f}' for i in (np.argmin(np.abs(zz - d - lamF)), np.argmin(np.abs(zz - d - 10 * lamF)), len(zz) - 1)))
        write_csv(os.path.join(RES, f'prl2025_slab_epsx_rs{rs3:g}.csv'), {'z': zs, 'z_over_lamF': zs / lamF, 'eps_x': eps})
    # semilocal energy per particle inside one slab (rs = 3)
    p3 = prl[3.0]
    fd = well_fields(p3['rs2d'], p3['d'], n=200)
    f = all_fx(fd['rho'], fd['g'], fd['lap'], fd['tau'], fd['s'], fd['D'], fd['Z'])
    zin = fd['u'] * p3['d'] - p3['d']
    o = np.argsort(zin)
    in_slab = {'z': np.r_[zin[o], -p3['d'] - zin[o][::-1]]}
    for k in ('LDA', 'SCAN', 'DCSCAN', 'DCr2SCAN'):
        v = -CX * np.cbrt(fd['rho'][o]) * f[k][o]
        in_slab[k] = np.r_[v, v[::-1]]
    in_slab['exact'] = slab_eps_x_profile(p3['rs2d'], p3['d'], in_slab['z'] + p3['d'], gtab)
    write_csv(os.path.join(RES, 'prl2025_slab_inside_rs3.csv'), in_slab)

    plots(data, f4, f7, matches, prl, in_slab)
    pub_figure()
    with open(os.path.join(RES, 'urso2021_prl2025_log.txt'), 'w') as fh:
        fh.write('\n'.join(log) + '\n')
    say('wrote results/urso2021_*.csv, prl2025_*.csv, urso2021_prl2025_log.txt; figures/urso2021_fig{3,4,7,8}.pdf, prl2025_slab_epsx.pdf')


def plots(data, f4, f7, matches, prl, in_slab):
    def save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=160, bbox_inches='tight')
        plt.close(fig)
    col = {k: v['color'] for k, v in STY.items()}

    # Fig. 3 and Fig. 8
    for tag, panels, keys, ylims, xlim, note in (
            ('fig3', (('fig3_rs2', 'Fig. 3 (rs2D=2)', r'$r_s^{2D}=2$'), ('fig3_rs5', 'Fig. 3 (rs2D=5)', r'$r_s^{2D}=5$')),
             ('exact', 'MVS', 'SCAN', 'Q2D', 'DCSCAN', 'DCr2SCAN'), ((-1.2, 0.0), (-0.45, -0.05)), (0, 1), 'quasi-2D IBM'),
            ('fig8', (('fig8_kf2', 'Fig. 8 (kF=2)', r'$k_F^{1D}=2$'), ('fig8_kf0.5', 'Fig. 8 (kF=0.5)', r'$k_F^{1D}=0.5$')),
             ('exact', 'MVS', 'SCAN', 'Q2D', 'Q1D', 'DCSCAN', 'DCr2SCAN'), ((-14, 0.0), (-4.0, 0.0)), (0, 0.5), 'quasi-1D IBM')):
        fig, axs = plt.subplots(2, 1, figsize=(5.4, 7.6), sharex=True)
        for ax, (dk, pk, lab), yl in zip(axs, panels, ylims):
            d = data[dk]
            for k in keys:
                ax.plot(d['x'], d[k], label=LAB[k], **STY[k])
            if tag == 'fig3':
                rs = 2.0 if 'rs2' in dk else 5.0
                ax.plot([0], [eps_x_2d(rs)], '*', color='#17becf', ms=9, label='2D UEG')
            overlay(ax, matches[pk], col)
            ax.set_ylim(*yl)
            ax.set_xlim(*xlim)
            ax.set_ylabel(r'$\epsilon_x=E_x/N$ (hartree)')
            ax.text(0.97, 0.06, f'{note}, {lab}', transform=ax.transAxes, ha='right', fontsize=9)
            ax.grid(alpha=0.2)
        axs[0].legend(fontsize=7, loc='lower right', bbox_to_anchor=(1.0, 0.12))
        axs[1].set_xlabel(r'$L/L_{\max}$')
        plt.tight_layout()
        save(fig, f'urso2021_{tag}')

    # Fig. 4 (+ D and Omega)
    fig, axs = plt.subplots(3, 1, figsize=(5.4, 9.6))
    cl = {2: '#d62728', 5: '#2ca02c', 10: '#1f77b4'}
    for dv in (2, 5, 10):
        r = f4[dv]
        axs[0].plot(r['r'], r['n'], color=cl[dv], lw=1.4, label=rf'$L=L_{{\max}}/{dv}$')
        axs[1].plot(r['r'], r['s'], color=cl[dv], lw=1.4, label=rf'$s$, $L=L_{{\max}}/{dv}$')
        axs[1].plot(r['r'], r['q'], color=cl[dv], lw=1.2, ls='--', label=rf'$q$, $L=L_{{\max}}/{dv}$')
        axs[2].plot(r['r'] / r['r'].max(), r['D'], color='k', lw=1.6, label='$D$ (any $L$)' if dv == 2 else None)
        axs[2].plot(r['r'] / r['r'].max(), r['Omega'], color=cl[dv], lw=1.2, ls='--', label=rf'$\Omega$, $L=L_{{\max}}/{dv}$')
    colmap = {f'n L=Lmax/{d}': cl[d] for d in cl}
    colmap.update({f's L=Lmax/{d}': cl[d] for d in cl})
    colmap.update({f'q L=Lmax/{d}': cl[d] for d in cl})
    overlay(axs[0], matches['Fig. 4 (n)'], colmap)
    overlay(axs[1], matches['Fig. 4 (s, q)'], colmap)
    axs[0].set_xlim(0, 5)
    axs[0].set_ylim(0, 0.4)
    axs[0].set_xlabel(r'$\rho$ (bohr)')
    axs[0].set_ylabel('$n$')
    axs[1].set_xlim(0, 4)
    axs[1].set_ylim(-1, 3)
    axs[1].set_xlabel(r'$\rho$ (bohr)')
    axs[1].set_ylabel('$s$ and $q$')
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 2.2)
    axs[2].set_xlabel(r'$\rho/L$')
    axs[2].set_ylabel(r'DC-SCAN $D$ and $\Omega$')
    axs[2].axhline(2, color='0.6', lw=0.6)
    axs[2].axhline(1, color='0.6', lw=0.6)
    axs[2].text(0.02, 2.05, 'wire branch ($D=2$)', fontsize=7, color='0.4')
    axs[2].text(0.02, 1.05, 'sheet branch ($D=1$)', fontsize=7, color='0.4')
    for a in axs:
        a.legend(fontsize=6.5, loc='upper right')
    for a in axs[:2]:
        a.text(0.03, 0.9, r'$k_F^{1D}=0.5$', transform=a.transAxes, fontsize=8)
    axs[2].set_title('(c) this work: DC-SCAN descriptor across the wire', fontsize=8, loc='left')
    plt.tight_layout()
    save(fig, 'urso2021_fig4')

    # Fig. 7
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    s = f7['s']
    for k, lab, st in (('Q1D', 'Q1D GGA', dict(color='#d62728', lw=1.4)), ('Q2D', 'Q2D GGA', dict(color='#2ca02c', lw=1.2, ls='--')),
                       ('PBEsol', 'PBEsol', dict(color='#1f77b4', lw=1.2, ls=':')),
                       ('MVS_a1', r'MVS, $\alpha=1$', dict(color='#c71585', lw=1.2, ls='-.')),
                       ('SCAN_a1', r'SCAN, $\alpha=1$', dict(color='0.4', lw=1.0, ls='-.')),
                       ('SCAN_a0', r'SCAN, $\alpha=0$ (DC base)', dict(color='0.4', lw=1.0, ls='--')),
                       ('F2D', r'DC sheet branch $F_{2D}$ (this work)', dict(color='#ff7f0e', lw=1.8)),
                       ('F1D', r'DC wire branch $F_{1D}$ (this work)', dict(color='#8c564b', lw=1.8))):
        ax.plot(s, f7[k], label=lab, **st)
    overlay(ax, matches['Fig. 7'], {'Q1D': '#d62728', 'Q2D': '#2ca02c', 'PBEsol': '#1f77b4', 'MVS alpha=1': '#c71585',
                                    'SCAN alpha=1': '0.4'})
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.8)
    ax.set_xlabel('$s$')
    ax.set_ylabel('$F_x$')
    ax.legend(fontsize=6.5, loc='upper right', ncol=2)
    save(fig, 'urso2021_fig7')

    # PRL 2025: exact eps_x(z) of single-subband slabs
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.2))
    cm = plt.get_cmap('viridis')
    for i, (rs3, p) in enumerate(prl.items()):
        c = cm(i / 5.0)
        axs[0].plot(p['z_over_lamF'], p['eps'], color=c, lw=1.4, label=rf'$r_s={rs3:g}$')
        out = p['z'] > 0
        axs[1].semilogx(p['z'][out] / p['lamF'], -2.0 * p['z'][out] * p['eps'][out], color=c, lw=1.4,
                        label=rf'$r_s={rs3:g}$')
    zz = np.linspace(0.05, 5, 200)
    axs[0].axvline(0, color='0.5', lw=0.6)
    axs[0].set_xlim(-0.06, 3)
    axs[0].set_ylim(-0.23, 0.005)
    axs[0].set_xlabel(r'$z/\lambda_F$ (slab in $[-d,0]$, $d=0.05\lambda_F$; vacuum $z>0$)')
    axs[0].set_ylabel(r'exact $\epsilon_x(z)$ (hartree)')
    axs[0].legend(fontsize=7)
    axs[0].set_title(r'(a) exact exchange energy per particle, single-subband slabs', fontsize=9, loc='left')
    axs[1].axhline(1.0, color='k', ls=':', lw=1.0, label=r'$\epsilon_x\to-1/(2z)$ [PRL Eq. (1)]')
    axs[1].set_xlabel(r'$z/\lambda_F$')
    axs[1].set_xlim(1e-3, 1e3)
    axs[1].set_ylabel(r'$-2z\,\epsilon_x(z)$')
    axs[1].set_ylim(0, 1.2)
    axs[1].legend(fontsize=7, loc='lower right')
    axs[1].set_title('(b) universal slab asymptote', fontsize=9, loc='left')
    ins = axs[2]
    for k, st in (('exact', dict(color='k', lw=1.8)), ('LDA', dict(color='0.6', lw=1.0)),
                  ('SCAN', dict(color='#1f77b4', ls=':', lw=1.2)), ('DCSCAN', dict(color='#d62728', lw=1.4)),
                  ('DCr2SCAN', dict(color='#ff7f0e', ls=(0, (1, 1.2)), lw=1.4))):
        ins.plot(in_slab['z'] / prl[3.0]['d'], in_slab[k], label=LAB.get(k, k).replace(' (this work)', ''), **st)
    ins.set_xlim(-1, 0)
    ins.set_xlabel(r'$z/d$ (inside the slab, $r_s=3$)')
    ins.set_ylabel(r'$\epsilon_x(z)$ (hartree), gauge-dependent')
    ins.set_title('(c) exact vs semilocal energy per particle', fontsize=9, loc='left')
    ins.legend(fontsize=7)
    plt.tight_layout()
    save(fig, 'prl2025_slab_epsx')


def dc_note(d, xmax, tol=0.05):
    """Comment on DC-SCAN vs exact for one panel: where it overlaps exact (|DC/exact - 1| <= tol for all larger L),
    where it crosses exact, and how far it is too negative at small L."""
    m = d['x'] <= xmax + 1e-12
    x, r = d['x'][m], d['DCSCAN'][m] / d['exact'][m] - 1.0
    bad = x[np.abs(r) > tol]
    x0 = float(bad.max()) if len(bad) else 0.0
    cross = [0.5 * (x[i] + x[i + 1]) for i in range(len(x) - 1) if r[i] * r[i + 1] < 0]
    at = lambda xv: 100.0 * float(np.interp(xv, x, r))
    line1 = rf'DC-SCAN overlaps exact ($|\Delta|\leq{100 * tol:.0f}\%$) for $L/L_{{\max}}\geq{x0:.2f}$'
    if cross:
        line1 += rf' (crosses it at {cross[0]:.2f})'
    line2 = (rf'$\Delta$ = {at(0.2):+.0f}%, {at(0.05):+.0f}%, {at(0.01):+.0f}% at $L/L_{{\max}}$ = 0.2, 0.05, 0.01'
             rf'  ($\Delta>0$: too negative)').replace('-', '\u2212')
    return line1 + '\n' + line2


def pub_figure():
    """Main-text figure (publication style): eps_x = E_x/N versus L/Lmax for the quasi-2D IBM well (rs2D = 2, 5) and the
    quasi-1D IBM wire (kF1D = 2, 0.5), read back from results/urso2021_fig{3,8}_*.csv written by main().  Each panel
    has an inset zoomed on 0 <= L/Lmax <= 0.2 (y range set by exact, Q2D/Q1D, DC-SCAN, DC-r2SCAN there)."""
    import csv
    def read(name):
        with open(os.path.join(RES, name)) as fh:
            rows = list(csv.DictReader(fh))
        return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    pub_style()
    lab = {'exact': 'exact exchange', 'MVS': 'MVS', 'SCAN': 'SCAN', 'Q2D': 'Q2D GGA', 'Q1D': 'Q1D GGA',
           'HPP19': 'MGGA Eq. (19)', 'DCSCAN': 'DC-SCAN', 'DCr2SCAN': r'DC-r$^2$SCAN'}
    fig, axs = plt.subplots(2, 2, figsize=(PUB_W['main_wide'], 7.6))
    panels = (('urso2021_fig3_rs2.csv', axs[0, 0], '(a) quasi-2D well', r'$r_s^{\rm 2D}=2$', (-0.45, -0.15), 1.0, 2.0),
              ('urso2021_fig8_kf2.csv', axs[0, 1], '(b) quasi-1D wire', r'$k_F^{\rm 1D}=2$', (-15.0, 0.0), 0.5, None),
              ('urso2021_fig3_rs5.csv', axs[1, 0], '(c) quasi-2D well', r'$r_s^{\rm 2D}=5$', (-0.18, -0.05), 1.0, 5.0),
              ('urso2021_fig8_kf0.5.csv', axs[1, 1], '(d) quasi-1D wire', r'$k_F^{\rm 1D}=0.5$', (-4.0, 0.0), 0.5, None))
    zoom = 0.2
    handles = {}
    for name, ax, title, par, yl, xmax, rs in panels:
        d = read(name)
        # user 2026-11-23: SCAN and MVS curves removed from this figure (their values stay in the csv files)
        keys = ('exact', 'Q2D', 'HPP19', 'DCSCAN', 'DCr2SCAN') if rs else \
               ('exact', 'Q2D', 'Q1D', 'HPP19', 'DCSCAN', 'DCr2SCAN')
        # well panels (rs set): inset lowered below the curves, which the tighter y range now spreads out
        ins = ax.inset_axes([0.40, 0.23, 0.57, 0.32] if rs else [0.40, 0.25, 0.57, 0.45])
        for k in keys:
            st = (0.5 / 8, 1.0 / 8) if k == 'DCr2SCAN' else None      # stagger: DC-r2SCAN lies within 1.4 mHa
            handles[k] = pub_line(ax, d['x'], d[k], k, label=lab[k], marks=8, **({'markevery': st} if st else {}))
            st = (0.5 / 6, 1.0 / 6) if k == 'DCr2SCAN' else None      # of DC-SCAN (markers between DC-SCAN's)
            pub_line(ins, d['x'], d[k], k, marks=6, lw=1.2, ms=3.2, **({'markevery': st} if st else {}))
        if rs:
            for a_ in (ax, ins):
                h = a_.plot([0.0], [eps_x_2d(rs)], ls='none', marker='*', ms=10 if a_ is ax else 8, color='#3aa6d8',
                            mec='black', mew=0.6, clip_on=False, zorder=5, label='2D UEG')[0]
            handles['UEG'] = h
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(*yl)
        ax.set_ylabel(r'$\epsilon_x=E_x/N$ (Ha)')
        ax.set_xlabel(r'$L/L_{\max}$')
        # inset: zoom on 0 <= L/Lmax <= 0.2; y range from the curves that stay close to exact
        m = (d['x'] >= (0.004 if rs else 0.01)) & (d['x'] <= zoom)
        close = [d[k][m] for k in keys if k in ('exact', 'Q2D', 'Q1D', 'HPP19', 'DCSCAN', 'DCr2SCAN')]
        lo, hi = min(c.min() for c in close), max(d[k][d['x'] <= zoom].max() for k in keys)
        pad = 0.06 * (hi - lo)
        ins.set_xlim(0.0, zoom)
        ins.set_ylim(lo - pad, hi + pad)
        ins.tick_params(labelsize=7, length=2.5)
        ins.set_xticks([0.0, 0.1, 0.2])
        ins.patch.set_facecolor('white')
        ax.indicate_inset_zoom(ins, edgecolor='0.45', lw=0.6)
        pub_panel(ax, f'{title}, {par}', x=0.97, y=0.03, ha='right', va='bottom')
        ax.text(0.5, -0.27, dc_note(d, xmax), transform=ax.transAxes, ha='center', va='top', fontsize=7,
                color='#d62728', linespacing=1.2)
    order = ['exact', 'Q2D', 'Q1D', 'HPP19', 'DCSCAN', 'DCr2SCAN', 'UEG']
    fig.legend([handles[k] for k in order], [lab.get(k, '2D UEG') for k in order], loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.0), frameon=True)
    # exact-exchange formulas used for the black curves
    fig.text(0.5, 0.918, r'exact exchange:  $E_x=\frac{1}{2}\int\!\!\int d\mathbf{r}\,d\mathbf{r}^\prime\,'
             r'\frac{n(\mathbf{r})\,n_x(\mathbf{r},\mathbf{r}^\prime)}{|\mathbf{r}-\mathbf{r}^\prime|}$,   '
             r'$n_x=-\frac{|n_1(\mathbf{r},\mathbf{r}^\prime)|^2}{2\,n(\mathbf{r})}$,   $\epsilon_x=E_x/N$',
             ha='center', va='center', fontsize=9)
    fig.text(0.27, 0.872, r'well:  $\epsilon_x=-\frac{1}{4\pi^2 n_{2D}}\int_0^{2k_F}F(qL)\,A(q)\,dq$',
             ha='center', va='center', fontsize=9)
    fig.text(0.76, 0.872, r'wire:  $\epsilon_x=-\frac{1}{2\pi^2 n_{1D}}\int_0^{2k_F}(2k_F-q)\,V(qL)\,dq$',
             ha='center', va='center', fontsize=9)
    fig.text(0.5, 0.829, r'MGGA Eq. (19) [PRB 108, 115119]:  $F_x=\frac{2.7\,\alpha^{1/2}}{1+2.7\alpha(0.8788+s)'
             r'-A(s)\,\alpha^{B(s)}+0.924\ln(1+\alpha^3)\,e^{-10s}}$,  $A=\frac{2.5}{(5+s)^{0.4}}$,  '
             r'$B=0.96\,e^{-0.5s^{0.3}}$', ha='center', va='center', fontsize=8.5, color=PUB['HPP19'][0])
    fig.tight_layout(rect=(0, 0, 1, 0.805), h_pad=0.4, w_pad=0.6)
    pub_save(fig, os.path.join(FIG, 'ibm_epsx_urso'))

def add_hpp19_to_csv():
    """Add/refresh the HPP19 column (Horowitz-Proetto-Pitarke Eq. 19 MGGA) in results/urso2021_fig{3,8}_*.csv on the
    same L grid, without recomputing the other columns; also check the 2D limit against the paper (rs2D = 4)."""
    import csv
    for name, fields, par in (('urso2021_fig3_rs2.csv', well_fields, 2.0), ('urso2021_fig3_rs5.csv', well_fields, 5.0),
                              ('urso2021_fig8_kf2.csv', wire_fields, 2.0), ('urso2021_fig8_kf0.5.csv', wire_fields, 0.5)):
        path = os.path.join(RES, name)
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r['HPP19'] = repr(energies(fields(par, float(r['L'])))['HPP19'])
        keys = list(rows[0].keys())
        if 'HPP19' in keys:
            keys.remove('HPP19')
        keys.insert(keys.index('DCSCAN'), 'HPP19')
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'{name}: HPP19 added ({len(rows)} points)')
    # paper check: Fig. 6 of PRB 108, 115119, rs2D = 4: eps_x -> 2D exact -0.150 Ha within 1% for L -> 0
    for x in (1e-3, 1e-2, 0.1, 0.5, 1.0):
        e = energies(well_fields(4.0, x * lmax(4.0)))['HPP19']
        print(f'check rs2D=4 L/Lmax={x:g}: HPP19 {e:.5f}  exact {well_exact_exchange(4.0, x * lmax(4.0)):.5f}'
              f'  2D UEG {eps_x_2d(4.0):.5f}')


if __name__ == '__main__':
    if '--hpp19-only' in sys.argv:   # add the HPP Eq. 19 MGGA column to the csv files, then redraw
        add_hpp19_to_csv()
        pub_figure()
    elif '--fig-only' in sys.argv:   # redraw figures/ibm_epsx_urso.* from the csv files of a previous full run
        pub_figure()
    else:
        main()
