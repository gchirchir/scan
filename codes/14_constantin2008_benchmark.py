#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal (needs the PySCF/Libxc environment used by codes 03, 08, 10):
#
#     cd SCAN_1D_2D        # (the folder that contains codes/)
#     ~/miniconda3/envs/anigto/bin/python codes/14_constantin2008_benchmark.py
#
# Needs: numpy, scipy, matplotlib, pyscf (libxc).  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""Benchmark of our quadrature / functional machinery against L. A. Constantin, "Dimensional crossover of the
exchange-correlation energy at the semilocal level", Phys. Rev. B 78, 155106 (2008)  [file subrata/PhysRevB.78.155106.pdf].

STANDALONE SCRIPT (SCAN_1D_2D/codes): every function it uses is defined here; only third-party imports.

Reproduced (paper numbering):
  Eq. (1)   2D uniform-gas exchange  eps_x^2D = -4 sqrt2 / (3 pi rs2D)            (rs2D = 4: -0.15005 Ha)
  Eq. (12)  a_x = 0.5217  (large-p GGA coefficient)       -- by quadrature of the L -> 0 well
  Eq. (13)  b_x = 1.947   (large-p meta-GGA coefficient)  -- by quadrature of the L -> 0 well
  Fig. 1    exchange energy per particle of the infinite-barrier (IBM) quasi-2D gas, rs2D = 4, vs 1/L:
            exact, LSDA, PBE, GGA+2D (Eq. 18) and MGGA+2D (Eq. 19) with f(p) of Eq. (21), c = 2, 8, 16
  Fig. 3    xc energy per particle of the IBM gas, rs2D = 4, vs L: LSDA, PBE, MGGA+2D (Eqs. 19 + 22),
            Eq. (18) + Eq. (22); the exact 2D limit (ISTLS is a many-body method and is not reproduced)
  Fig. 4    as Fig. 3 for rs2D = 2/sqrt3
  Fig. 5    xc energy of the nonuniformly scaled hydrogen atom vs lambda: exact Eq. (23), LSDA, PBE, MGGA+2D,
            Eq. (18) + Eq. (22)
Not reproduced: Fig. 2 (jellium-slab exchange energy density with self-consistent LSDA orbitals) and ISTLS.
Validation of OUR DC-SCAN code (added 2026-11-17): fx_scan_alpha0, fx_dcscan (+ geometry_weights, omega_switch,
  f2d, f1d), _log_gl, m2d/A2D are copied verbatim from codes/01_wire_benchmark.py and evaluated on the IBM well
  (rs2D = 4) down to L = 1e-9 bohr; L -> 0 limits extrapolated in L^{1/3} ln L, L^{1/3}, L^{2/3} and compared with the analytic
  limits: SCAN -> -CX h0x a1 K I / n2D = -1.670599 Ha (Kaplan et al. 2018: -1.671), DC-SCAN -> (A2D / a_x) eps_x^2D
  = -0.187865 Ha (A2D is fitted on the Gaussian sheet, so DC-SCAN is NOT exact on this slab; exact -0.150053).
  r2SCAN and DC-r2SCAN (Libxc MGGA_X_R2SCAN base, true tau; alpha != 0 in the well) are added to the well
  table and to the Fig. 1 table (columns DCSCAN, DCr2SCAN, SCAN, r2SCAN; DC-* with Libxc base, true tau).
  Outputs: results/constantin2008_well_dcscan.csv, figures/constantin2008_well_dcscan.pdf.

Methods
  IBM well of current thickness L (= Lmax / lambda), one occupied subband:
      n(z) = (2 n2D / L) sin^2(pi z / L),  n2D = 1/(pi rs2D^2),  kF = sqrt(2)/rs2D,
      tau  = tau_W + kF^4 / (4 pi L) sin^2(pi z / L)                                   (paper Eq. 11)
  Exact exchange per electron of the single-subband gas (Fermi-disk overlap A(q), form factor F):
      eps_x = -(1 / (4 pi^2 n2D)) int_0^{2kF} F(qL) A(q) dq,
      A(q) = 2 kF^2 [acos x - x sqrt(1-x^2)],  x = q/(2kF),   F(Q) = int int rho(u) rho(u') e^{-Q|u-u'|} du du'.
      F = 1 gives Eq. (1) exactly (checked).
  Semilocal energies per electron: (1/n2D) int_0^L n eps(n, n', tau) dz, Gauss-Legendre in u = z/L.
  LSDA (Slater + PW92), PBE (x + c) from Libxc; GGA+2D/MGGA+2D/Eq. (22) coded from the paper:
      eps_x^{GGA+2D}  = eps_x^LSDA {1 + f(p) [-1 + 0.5217 p^{-1/4}]}                       (18)
      eps_x^{MGGA+2D} = eps_x^LSDA {1 + f(p) [-1 + 1.947 alpha^{1/2}]}                      (19)
      f(p) = p^4 (1 + p^2) / (10^c + p^6)                                                   (21)
      eps_c^{MGGA+2D} = eps_c^LSDA + f(t) [-eps_c^LSDA + eps_c^2D(rs2D_loc)],  c = 8       (22)
      eps_c^2D: Eq. (4);  rs2D_loc = 0.4173 n^{-1/3} alpha^{-1/2}                           (17)
      p = |grad n|^2 / (4 (3 pi^2)^{2/3} n^{8/3}),  t = |grad n| / (4 (3/pi)^{1/6} n^{7/6})
  Scaled H atom (one electron, fully spin polarized), n = (lam/pi) exp(-2 sqrt(x^2+y^2+(lam z)^2)):
      substitute z' = lam z -> spherical (r', mu): |grad n| = 2 n sqrt(1 - mu^2 + lam^2 mu^2), d^3r = 2 pi r'^2 dr' dmu / lam.
      Libxc spin-polarized LSDA/PBE.  Eq. (18)/(19) exchange by spin scaling E_x[n,0] = E_x^unpol[2n]/2;
      Correlation: the paper's Fig. 5 is reproduced (1.3 mHa rms, code 15) only when the "+2D" curves use plain
      zeta = 1 PW92 correlation, i.e. the Eq. (22) switch term does not act on this one-electron density.  The
      literal Eq. (22) (zeta = 1 PW92, t of the total density, alpha = 0 -> rs2D_loc = inf -> eps_c^2D = 0) is
      kept as extra columns *_literal22; it lies ~20-37 mHa above the paper curves for lambda > 50.
      (The paper does not spell out its spin treatment; p is evaluated at 2n by exchange spin scaling.)
      Exact E_xc = -J; checked against an independent Fourier-space Hartree quadrature.
Outputs: results/constantin2008_fig{1,3,4,5}.csv, results/constantin2008_log.txt,
         results/tab_constantin2008_rows.tex, figures/constantin2008_fig{1,3,4,5}.pdf/.png
"""
import math
import warnings
import os

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_fn
from scipy.special import gamma

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

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)          # LDA exchange: e_x = -CX n^{4/3} (unpolarized)
KF3 = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
AX_PAPER, BX_PAPER = 0.5217, 1.947                  # paper Eqs. (12), (13) -- used in Eqs. (18), (19)
C_CORR = 8.0                                        # paper: c = 8 in Eq. (22)


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


# ------------------------------------------------------------------------------------------- paper formulas
def eps_x_2d(rs):
    """Eq. (1)."""
    return -4.0 * math.sqrt(2.0) / (3.0 * math.pi * rs)


def eps_c_2d(rs):
    """Eq. (4), Seidl-Perdew-Levy interpolation; vectorized, -> 0 for rs -> inf."""
    rs = np.asarray(rs, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        v = 0.5058 * (1.3311 / rs ** 2 * (np.sqrt(1.0 + 1.5026 * rs) - 1.0) - 1.0 / rs)
    return np.where(np.isfinite(rs), v, 0.0)


def f_switch(p, c):
    """Eq. (21): p^4 (1 + p^2) / (10^c + p^6), evaluated stably for large p."""
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    small = p <= 1.0
    ps = p[small]
    out[small] = ps ** 4 * (1.0 + ps ** 2) / (10.0 ** c + ps ** 6)
    pl = p[~small]
    out[~small] = (pl ** -2 + 1.0) / (10.0 ** c * pl ** -6 + 1.0)
    return out


def lmax(rs):
    """Single-subband condition L < Lmax = sqrt(3/2) pi rs2D (paper text after Eq. 6)."""
    return math.sqrt(1.5) * math.pi * rs


# ------------------------------------------------------------------------------------------- Libxc wrappers
def xc_unpol(code, rho, grad):
    """Energy per particle from Libxc, unpolarized; grad = |grad n| (put on x)."""
    if 'GGA' not in code:
        return np.asarray(libxc.eval_xc(code, rho, spin=0, deriv=0)[0])
    z = np.zeros_like(rho)
    return np.asarray(libxc.eval_xc(code, np.vstack((rho, grad, z, z)), spin=0, deriv=0)[0])


def xc_pol(code, rho, grad):
    """Energy per particle (of the total density) from Libxc, fully polarized (n_up = n, n_dn = 0)."""
    z = np.zeros_like(rho)
    if 'GGA' not in code:
        return np.asarray(libxc.eval_xc(code, (rho, z), spin=1, deriv=0)[0])
    up = np.vstack((rho, grad, z, z))
    dn = np.zeros_like(up)
    return np.asarray(libxc.eval_xc(code, (up, dn), spin=1, deriv=0)[0])


# ------------------------------------------------------------------------------------------- IBM quantum well
class Well:
    """IBM quasi-2D gas of 2D parameter rs and current thickness L, Gauss-Legendre in u = z/L."""

    def __init__(self, rs, L, n=600):
        self.rs, self.L = rs, L
        self.n2d = 1.0 / (math.pi * rs * rs)
        self.kf = math.sqrt(2.0) / rs
        x, w = np.polynomial.legendre.leggauss(n)
        u = 0.5 * (x + 1.0)
        self.w = 0.5 * w * L                         # weights for dz
        s, c = np.sin(np.pi * u), np.cos(np.pi * u)
        self.rho = 2.0 * self.n2d / L * s * s
        self.grad = np.abs(2.0 * self.n2d / L * (np.pi / L) * 2.0 * s * c)
        self.tau_w = self.grad ** 2 / (8.0 * self.rho)
        self.tau = self.tau_w + self.kf ** 4 / (4.0 * np.pi * L) * s * s
        self.p = self.grad ** 2 / (4.0 * KF3 ** 2 * self.rho ** (8.0 / 3.0))
        self.t = self.grad / (4.0 * (3.0 / np.pi) ** (1.0 / 6.0) * self.rho ** (7.0 / 6.0))
        self.alpha = (self.tau - self.tau_w) / (0.3 * KF3 ** 2 * self.rho ** (5.0 / 3.0))
        self.ex_lda_closed = -CX * np.cbrt(self.rho)

    def per_particle(self, eps):
        return float(np.sum(self.w * self.rho * eps) / self.n2d)

    def energies(self, cs=(2.0, 8.0, 16.0)):
        r, g = self.rho, self.grad
        ex_lda = xc_unpol('LDA_X,', r, g)
        ec_lda = xc_unpol(',LDA_C_PW', r, g)
        out = {
            'x_LSDA': self.per_particle(ex_lda),
            'x_LSDA_closed': self.per_particle(self.ex_lda_closed),
            'x_PBE': self.per_particle(xc_unpol('GGA_X_PBE,', r, g)),
            'c_LSDA': self.per_particle(ec_lda),
            'c_PBE': self.per_particle(xc_unpol(',GGA_C_PBE', r, g)),
        }
        for c in cs:
            fp = f_switch(self.p, c)
            out[f'x_GGA2D_c{c:g}'] = self.per_particle(ex_lda * (1.0 + fp * (-1.0 + AX_PAPER * self.p ** -0.25)))
            out[f'x_MGGA2D_c{c:g}'] = self.per_particle(ex_lda * (1.0 + fp * (-1.0 + BX_PAPER * np.sqrt(self.alpha))))
        rs_loc = 0.4173 * self.rho ** (-1.0 / 3.0) * self.alpha ** -0.5
        ft = f_switch(self.t, C_CORR)
        out['c_MGGA2D'] = self.per_particle(ec_lda + ft * (-ec_lda + eps_c_2d(rs_loc)))
        return out


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


K = np.sqrt(2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0))   # n^{4/3} s^{-1/2} = K n^2 |dn/dz|^{-1/2}


def slab_integral(rs2d):
    """int n^2 |n'|^(-1/2) dz per unit area for the infinite-barrier slab n = n_2D (2/L) sin^2(pi z/L).
    Closed form (L cancels): with x = pi z/L,  I = (2 n_2D)^(3/2) 2^(-1/2) pi^(-3/2) int_0^pi sin^(7/2)x |cos x|^(-1/2) dx
    and int_0^pi sin^(7/2)x |cos x|^(-1/2) dx = B(9/4, 1/4)  (the integral of Constantin's eq. 12)."""
    n2d = 1.0 / (np.pi * rs2d ** 2)
    return n2d, (2.0 * n2d) ** 1.5 / np.sqrt(2.0) * np.pi ** -1.5 * beta_fn(2.25, 0.25)
    # (copied verbatim from codes/09_literature_validation.py)


def limit_coefficients(rs=4.0):
    """a_x (Eq. 12) and b_x (Eq. 13): the coefficients that make the large-p forms give Eq. (1) as L -> 0.
    a_x: closed form (as code 09) and an independent quadrature on the L -> 0 well with the |cos|^{-1/2}
    singularity at the well centre handled by an algebraic quadrature weight.  b_x: quadrature (smooth)."""
    n2d, I = slab_integral(rs)
    target = eps_x_2d(rs)
    ax_closed = (n2d * target) / (-CX * K * I)
    L = 1e-7 * lmax(rs)
    kf = math.sqrt(2.0) / rs

    def lda_weighted(u, kind):
        s, c = math.sin(math.pi * u), math.cos(math.pi * u)
        n = 2.0 * n2d / L * s * s
        g = abs(2.0 * n2d / L * (math.pi / L) * 2.0 * s * c)
        e = n * (-CX * n ** (1.0 / 3.0)) * L
        if kind == 'p':
            p = g * g / (4.0 * KF3 ** 2 * n ** (8.0 / 3.0))
            return e * p ** -0.25 * math.sqrt(0.5 - u)          # remove the (1/2 - u)^{-1/2} factor
        tw = g * g / (8.0 * n)
        alpha = (kf ** 4 / (4.0 * math.pi * L) * s * s) / (0.3 * KF3 ** 2 * n ** (5.0 / 3.0))
        return e * math.sqrt(alpha)

    ip = 2.0 * quad(lambda u: lda_weighted(u, 'p'), 0.0, 0.5, weight='alg', wvar=(0.0, -0.5),
                    epsabs=0, epsrel=1e-12, limit=400)[0]
    ia = 2.0 * quad(lambda u: lda_weighted(u, 'a'), 0.0, 0.5, epsabs=0, epsrel=1e-12, limit=400)[0]
    return target * n2d / ip, target * n2d / ia, ax_closed


# ------------------------------------------------------------------------------------------- scaled H atom
def h_atom_energies(lam, nr=200, nmu=200, cs=(8.0,)):
    """E_xc (Ha) of n = (lam/pi) exp(-2 sqrt(rho^2 + lam^2 z^2)), fully spin polarized."""
    # Gauss-Legendre on log r' (robust for every lambda)
    a, b = np.log(1e-8), np.log(60.0)
    x, w = np.polynomial.legendre.leggauss(nr)
    r = np.exp(0.5 * (b - a) * x + 0.5 * (b + a))
    wr = 0.5 * (b - a) * w * r
    xm, wm = np.polynomial.legendre.leggauss(nmu)
    mu = 0.5 * (xm + 1.0)                            # symmetric in mu -> integrate (0,1), times 2
    wm = 0.5 * wm
    R, M = np.meshgrid(r, mu, indexing='ij')
    W = (2.0 * np.pi * R * R * np.outer(wr, wm) * 2.0 / lam).ravel()
    rho = (lam / np.pi * np.exp(-2.0 * R)).ravel()
    grad = (2.0 * rho.reshape(R.shape) * np.sqrt(1.0 - M * M + lam * lam * M * M)).ravel()

    def E(eps):
        return float(np.sum(W * rho * eps))

    out = {'N': float(np.sum(W * rho))}
    ex_lda = xc_pol('LDA_X,', rho, grad)
    ec_lda = xc_pol(',LDA_C_PW', rho, grad)
    out['xc_LSDA'] = E(ex_lda + ec_lda)
    out['xc_PBE'] = E(xc_pol('GGA_X_PBE,', rho, grad) + xc_pol(',GGA_C_PBE', rho, grad))
    # spin scaling: per-particle exchange of the polarized density = unpolarized form at density 2n
    rho2, grad2 = 2.0 * rho, 2.0 * grad
    p2 = grad2 ** 2 / (4.0 * KF3 ** 2 * rho2 ** (8.0 / 3.0))
    ex_lda2 = -CX * np.cbrt(rho2)                    # e_x^unpol(2n)/(2n) * (2n)/(2n): per particle of 2n
    t = grad / (4.0 * (3.0 / np.pi) ** (1.0 / 6.0) * rho ** (7.0 / 6.0))
    ft = f_switch(t, C_CORR)
    ec_22 = ec_lda + ft * (-ec_lda + 0.0)            # alpha = 0 -> rs2D_loc = inf -> eps_c^2D = 0
    for c in cs:
        fp = f_switch(p2, c)
        ex18 = ex_lda2 * (1.0 + fp * (-1.0 + AX_PAPER * p2 ** -0.25))
        ex19 = ex_lda2 * (1.0 + fp * (-1.0))         # alpha = 0
        # E_x[n,0] = E_x^unpol[2n] / 2 = (1/2) int (2n) eps(2n) = int n eps(2n)
        out[f'xc_Eq18+Eq22_c{c:g}'] = E(ex18 + ec_lda)          # as plotted in the paper (see docstring)
        out[f'xc_MGGA2D_c{c:g}'] = E(ex19 + ec_lda)
        out[f'xc_Eq18+Eq22_c{c:g}_literal22'] = E(ex18 + ec_22)
        out[f'xc_MGGA2D_c{c:g}_literal22'] = E(ex19 + ec_22)
    out['x_LSDA_closed_check'] = E(ex_lda2) - E(ex_lda)   # closed-form spin-scaled LDA vs Libxc
    return out


def h_atom_exact(lam):
    """Paper Eq. (23)."""
    if abs(lam - 1.0) < 1e-12:
        return -5.0 / 16.0
    if lam > 1.0:
        q = math.sqrt(lam * lam - 1.0)
        return -5.0 / 16.0 * lam / q * math.atan(q)
    q = math.sqrt(1.0 - lam * lam)
    return -5.0 / 16.0 * lam / q * math.log((q + 1.0) / lam)


def h_atom_hartree_fourier(lam):
    """Independent check: J = (1/2) int d^3k/(2pi)^3 4pi/k^2 |n(k)|^2 with n1(q) = (1 + q^2/4)^{-2},
    q^2 = k^2 (1 - mu^2 + mu^2/lam^2); radial k integral numerically, mu integral numerically."""
    def inner(mu):
        g = 1.0 - mu * mu + mu * mu / (lam * lam)
        return quad(lambda k: (1.0 + k * k * g / 4.0) ** -4, 0.0, np.inf, epsabs=0, epsrel=1e-13)[0]
    val = quad(inner, -1.0, 1.0, epsabs=0, epsrel=1e-12, limit=200)[0]
    return -val / (2.0 * math.pi)                     # (1/2)(4pi)(2pi)/(2pi)^3 = 1/(2pi); E_xc = -J


# ------------------------------------------------------------------------------------------- driver
def write_csv(path, header, rows):
    with open(path, 'w') as fh:
        fh.write(','.join(header) + '\n')
        for r in rows:
            fh.write(','.join(f'{v:.12g}' for v in r) + '\n')


# ------------------------------------------------------------------------------------------- our DC-SCAN on the IBM well
def well_dcscan(rs, L, n=700):
    """Exchange per electron of the IBM well with OUR functions (code 01, verbatim above) and Libxc SCAN.

    Grid: by symmetry u in (0, 1/2), split at 1/4; Gauss-Legendre in log(u) near the wall and in log(1/2 - u)
    near the centre (where |n'| = 0 and the s^{-1/2} integrands are singular in the limit), via _log_gl.
    Curvature tensor C = -1/2 dd ln n = diag(0, 0, (pi/L)^2 / sin^2(pi u)) -> D = |g|^2 trC^2 / (g C^2 g) = 1
    (sheet branch) wherever n' != 0, and Z = sqrt(trC^2) / k_F^2.
    """
    n2d = 1.0 / (math.pi * rs * rs)
    kf2d = math.sqrt(2.0) / rs
    ua, wa = _log_gl(1e-14, 0.25, n)                    # from the wall
    vb, wb = _log_gl(1e-14, 0.25, n)                    # from the centre, u = 1/2 - v
    u = np.concatenate((ua, 0.5 - vb))
    w = 2.0 * L * np.concatenate((wa, wb))              # dz = L du, both halves
    amp = 2.0 * n2d / L
    sn, cs = np.sin(np.pi * u), np.cos(np.pi * u)
    rho = amp * sn * sn
    g = np.abs(amp * (np.pi / L) * 2.0 * sn * cs)
    lap = amp * 2.0 * (np.pi / L) ** 2 * np.cos(2.0 * np.pi * u)
    tau_w = g * g / (8.0 * rho)
    tau = tau_w + kf2d ** 4 / (4.0 * np.pi * L) * sn * sn
    kf = KF * np.cbrt(rho)
    s = g / (2.0 * kf * rho)
    czz = (np.pi / L) ** 2 / (sn * sn)
    gc2g = g * g * czz * czz
    D = np.clip(np.where(gc2g > 0, g * g * czz * czz / np.where(gc2g > 0, gc2g, 1.0), 3.0), 1.0, 3.0)
    Z = czz / kf ** 2
    e_lda = -CX * np.cbrt(rho)
    z = np.zeros_like(rho)
    inp = np.vstack((rho, g, z, z, lap, tau))
    e_scan = np.asarray(libxc.eval_xc('MGGA_X_SCAN,', inp, spin=0, deriv=0)[0])
    e_r2 = np.asarray(libxc.eval_xc('MGGA_X_R2SCAN,', inp, spin=0, deriv=0)[0])
    ec_scan = np.asarray(libxc.eval_xc(',MGGA_C_SCAN', inp, spin=0, deriv=0)[0])     # correlation for xc (Fig. 3/4)
    ec_r2 = np.asarray(libxc.eval_xc(',MGGA_C_R2SCAN', inp, spin=0, deriv=0)[0])
    ok = e_lda != 0.0
    fs0 = fx_scan_alpha0(s)
    fs_lx = np.where(ok, e_scan / np.where(ok, e_lda, 1.0), fs0)
    fr2_lx = np.where(ok, e_r2 / np.where(ok, e_lda, 1.0), fs0)
    per = lambda f: float(np.sum(w * rho * e_lda * f) / n2d)
    return {
        'LDA': per(np.ones_like(rho)),
        'SCAN_libxc': per(fs_lx),
        'SCAN_alpha0': per(fs0),
        'DCSCAN': per(fx_dcscan(s, D, Z, fs0, 'full')),
        'DCSCAN_libxc_base': per(fx_dcscan(s, D, Z, fs_lx, 'full')),
        'r2SCAN_libxc': per(fr2_lx),
        'DCr2SCAN_libxc_base': per(fx_dcscan(s, D, Z, fr2_lx, 'full')),
        'c_SCAN': float(np.sum(w * rho * ec_scan) / n2d),
        'c_r2SCAN': float(np.sum(w * rho * ec_r2) / n2d),
        'N_check': float(np.sum(w * rho) / n2d),
        'D_max_dev': float(np.max(np.abs(D[gc2g > 0] - 1.0))),
    }


def extrapolate_cuberoot(L, E):
    """Fit E(L) = E_inf + a L^{1/3} ln L + b L^{1/3} + c L^{2/3} and return E_inf.
    The s^{-1/2} tail gives the finite limit; the next SCAN term (a1^2 s^{-1}) has a |cos|^{-1} singularity at the
    well centre, cut off where s ~ 1 (width ~ L^{2/3}), hence the L^{1/3} ln L term."""
    X = np.vstack((np.ones_like(L), np.cbrt(L) * np.log(L), np.cbrt(L), np.cbrt(L) ** 2)).T
    return float(np.linalg.lstsq(X, E, rcond=None)[0][0])


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    from scipy.integrate import IntegrationWarning
    warnings.filterwarnings('ignore', category=IntegrationWarning)   # alg-weighted a_x quadrature; checked vs closed form
    log = []
    say = lambda s='': (print(s), log.append(s))
    say('Constantin, PRB 78, 155106 (2008): reproduction with SCAN_1D_2D codes (script 14)')

    # --- constants and limits
    ax, bx, ax_closed = limit_coefficients()
    e2d4 = eps_x_2d(4.0)
    ex_L0 = well_exact_exchange(4.0, 0.0)
    ex_small = well_exact_exchange(4.0, 1e-6)
    f9 = float(f_switch(np.array([9.0]), 8.0)[0])
    checks = [
        ('2D gas exchange, rs2D=4 (Ha), Eq. (1)', e2d4, -0.15005),
        ('exact well exchange, L->0 (F=1 route)', ex_L0, -0.15005),
        ('a_x (Eq. 12), L->0 quadrature', ax, AX_PAPER),
        ('a_x (Eq. 12), closed form (as code 09)', ax_closed, AX_PAPER),
        ('b_x (Eq. 13), L->0 quadrature', bx, BX_PAPER),
        ('Lmax(rs2D=4) (bohr)', lmax(4.0), 15.39),
        ('Lmax(rs2D=2/sqrt3) (bohr)', lmax(2.0 / math.sqrt(3.0)), 4.44),
        ('f(p=9), c=8, Eq. (21)', f9, 0.0053),
        ('H atom E_xc(lam=1), Eq. (23)', h_atom_exact(1.0), -5.0 / 16.0),
        ('H atom E_xc(lam->inf) = -5 pi/32', h_atom_exact(1e9), -0.4909),
    ]
    say(f'exact well exchange at L = 1e-6 bohr: {ex_small:.8f}')
    say('')
    say(f'{"quantity":44s} {"ours":>14s} {"paper":>10s}  rel.diff')
    for name, ours, ref in checks:
        say(f'{name:44s} {ours:14.6f} {ref:10.5f}  {(ours - ref) / abs(ref):+.2e}')

    # --- independent Hartree check of Eq. (23)
    say('')
    say('H atom: Eq. (23) vs Fourier-space Hartree quadrature')
    for lam in (0.5, 1.0, 2.0, 10.0, 100.0, 350.0):
        a, b = h_atom_exact(lam), h_atom_hartree_fourier(lam)
        say(f'  lam={lam:7.1f}  Eq.(23) {a:.12f}  Fourier {b:.12f}  diff {a - b:+.1e}')

    # --- Fig. 1: exchange per particle, rs2D = 4, vs 1/L
    rs = 4.0
    inv_L = np.concatenate((np.linspace(1.0 / lmax(rs), 1.0, 40), np.linspace(1.02, 40.0, 120)))
    rows1 = []
    for il in inv_L:
        L = 1.0 / il
        e = Well(rs, L).energies()
        d = well_dcscan(rs, L)                          # DC-SCAN / DC-r2SCAN: Libxc base with the true tau
        rows1.append([il, L, well_exact_exchange(rs, L), e['x_LSDA'], e['x_PBE'],
                      e['x_GGA2D_c2'], e['x_GGA2D_c8'], e['x_GGA2D_c16'],
                      e['x_MGGA2D_c2'], e['x_MGGA2D_c8'], e['x_MGGA2D_c16'], e['x_LSDA_closed'],
                      d['DCSCAN_libxc_base'], d['DCr2SCAN_libxc_base'], d['SCAN_libxc'], d['r2SCAN_libxc']])
    rows1 = np.array(rows1)
    h1 = ['inv_L', 'L', 'exact', 'LSDA', 'PBE', 'GGA2D_c2', 'GGA2D_c8', 'GGA2D_c16',
          'MGGA2D_c2', 'MGGA2D_c8', 'MGGA2D_c16', 'LSDA_closed_form', 'DCSCAN', 'DCr2SCAN', 'SCAN', 'r2SCAN']
    write_csv(os.path.join(RES, 'constantin2008_fig1.csv'), h1, rows1)
    say('')
    say(f'Fig. 1: max |LSDA Libxc - closed form| = {np.max(np.abs(rows1[:, 3] - rows1[:, 11])):.2e} Ha')
    for il in (1.0, 5.0, 10.0, 20.0, 40.0):
        i = int(np.argmin(np.abs(rows1[:, 0] - il)))
        r = rows1[i]
        say(f'  1/L={r[0]:6.2f}: exact {r[2]:.5f} LSDA {r[3]:.5f} PBE {r[4]:.5f} '
            f'GGA+2D(a,b,c) {r[5]:.5f} {r[6]:.5f} {r[7]:.5f} MGGA+2D(a,b,c) {r[8]:.5f} {r[9]:.5f} {r[10]:.5f}')

    # --- Figs. 3, 4: xc per particle vs L
    figs34 = {}
    for tag, rs_, Lhi in (('fig3', 4.0, 8.0), ('fig4', 2.0 / math.sqrt(3.0), 3.0)):
        Ls = np.linspace(Lhi / 400.0, Lhi, 200)
        rows = []
        for L in Ls:
            e = Well(rs_, L).energies(cs=(8.0,))
            d = well_dcscan(rs_, L)                   # DC-SCAN / DC-r2SCAN xc = DC exchange + Libxc SCAN / r2SCAN c
            rows.append([L, e['x_LSDA'] + e['c_LSDA'], e['x_PBE'] + e['c_PBE'],
                         e['x_MGGA2D_c8'] + e['c_MGGA2D'], e['x_GGA2D_c8'] + e['c_MGGA2D'],
                         well_exact_exchange(rs_, L),
                         d['DCSCAN_libxc_base'] + d['c_SCAN'], d['DCr2SCAN_libxc_base'] + d['c_r2SCAN'],
                         d['SCAN_libxc'] + d['c_SCAN'], d['r2SCAN_libxc'] + d['c_r2SCAN'],
                         d['c_SCAN'], d['c_r2SCAN']])
        rows = np.array(rows)
        write_csv(os.path.join(RES, f'constantin2008_{tag}.csv'),
                  ['L', 'LSDA', 'PBE', 'MGGA2D', 'Eq18_Eq22', 'exact_exchange_only', 'DCSCAN_xc', 'DCr2SCAN_xc',
                   'SCAN_xc', 'r2SCAN_xc', 'c_SCAN', 'c_r2SCAN'], rows)
        exact2d = eps_x_2d(rs_) + float(eps_c_2d(rs_))
        figs34[tag] = (rs_, rows, exact2d)
        say(f'{tag}: rs2D={rs_:.4f} exact 2D xc limit {exact2d:.5f}; at smallest L={rows[0,0]:.4f}: '
            f'MGGA+2D {rows[0,3]:.5f}, Eq18+22 {rows[0,4]:.5f}, LSDA {rows[0,1]:.5f}, PBE {rows[0,2]:.5f}, '
            f'DC-SCAN xc {rows[0,6]:.5f}, DC-r2SCAN xc {rows[0,7]:.5f}, SCAN xc {rows[0,8]:.5f}')
        for Lq in (Lhi / 400.0, 0.5, 1.0, 2.0, Lhi):
            i = int(np.argmin(np.abs(rows[:, 0] - Lq)))
            r = rows[i]
            say(f'  L={r[0]:6.3f}: LSDA {r[1]:.5f} PBE {r[2]:.5f} MGGA+2D {r[3]:.5f} DC-SCAN xc {r[6]:.5f} '
                f'DC-r2SCAN xc {r[7]:.5f} SCAN xc {r[8]:.5f} r2SCAN xc {r[9]:.5f} (c_SCAN {r[10]:.5f})')

    # --- Fig. 5: scaled H atom
    lams = np.concatenate((np.linspace(1.0, 10.0, 19), np.linspace(15.0, 350.0, 68)))
    rows5 = []
    for lam in lams:
        e = h_atom_energies(lam)
        rows5.append([lam, h_atom_exact(lam), e['xc_LSDA'], e['xc_PBE'], e['xc_MGGA2D_c8'],
                      e['xc_Eq18+Eq22_c8'], e['N'], e['x_LSDA_closed_check'],
                      e['xc_MGGA2D_c8_literal22'], e['xc_Eq18+Eq22_c8_literal22']])
    rows5 = np.array(rows5)
    write_csv(os.path.join(RES, 'constantin2008_fig5.csv'),
              ['lambda', 'exact', 'LSDA', 'PBE', 'MGGA2D', 'Eq18_Eq22', 'N_check', 'LSDA_x_closed_minus_libxc',
               'MGGA2D_literal22', 'Eq18_Eq22_literal22'], rows5)
    say(f'Fig. 5: max |N - 1| = {np.max(np.abs(rows5[:, 6] - 1.0)):.1e}; '
        f'max |LSDA_x closed - Libxc| = {np.max(np.abs(rows5[:, 7])):.1e} Ha')
    for lam in (1.0, 10.0, 100.0, 350.0):
        i = int(np.argmin(np.abs(rows5[:, 0] - lam)))
        r = rows5[i]
        say(f'  lam={r[0]:6.1f}: exact {r[1]:.5f} LSDA {r[2]:.5f} PBE {r[3]:.5f} MGGA+2D {r[4]:.5f} Eq18+22 {r[5]:.5f}')

    # --- figures (paper styling: same axes ranges and labels)
    def save(fig, name):
        fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(FIG, name + '.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(5.0, 7.6), sharex=True)
    for axp, kind in zip(axs, ('GGA2D', 'MGGA2D')):
        axp.plot(rows1[:, 0], rows1[:, 2], 'k-', lw=1.8, label='exact reference (this work)')
        axp.plot(rows1[:, 0], rows1[:, 3], 'k--', lw=1.0, label='LSDA')
        axp.plot(rows1[:, 0], rows1[:, 4], 'k:', lw=1.2, label='PBE')
        col = {'GGA2D': (5, 6, 7), 'MGGA2D': (8, 9, 10)}[kind]
        nm = 'GGA+2D' if kind == 'GGA2D' else 'MGGA+2D'
        for j, lab, st in zip(col, ('(a) c=2', '(b) c=8', '(c) c=16'), ('-', '-.', '--')):
            axp.plot(rows1[:, 0], rows1[:, j], st, lw=1.3, label=f'({nm}){lab}')
        axp.set_ylim(-0.55, -0.05)
        axp.set_xlim(0, 40)
        axp.set_ylabel(r'$\epsilon_x$ (hartree)')
        axp.text(30, -0.36, r'$r_s^{2D}=4$')
        axp.legend(fontsize=7, loc='lower left')
    axs[1].set_xlabel(r'$1/L=\lambda/L_{max}$ (bohr$^{-1}$)')
    save(fig, 'constantin2008_fig1')

    for tag, ylim in (('fig3', (-0.4, -0.1)), ('fig4', (-1.2, -0.4))):
        rs_, rows, exact2d = figs34[tag]
        fig, axp = plt.subplots(figsize=(5.0, 3.8))
        axp.plot(rows[:, 0], rows[:, 1], 'k--', label='LSDA')
        axp.plot(rows[:, 0], rows[:, 2], 'k:', label='PBE')
        axp.plot(rows[:, 0], rows[:, 3], '-', label='MGGA+2D')
        axp.plot(rows[:, 0], rows[:, 4], '-.', label='Eq.(18)+Eq.(22)')
        axp.plot([0.0], [exact2d], 'ko', label='exact 2D limit')
        axp.set_xlim(0, rows[-1, 0])
        axp.set_ylim(*ylim)
        axp.set_xlabel(r'$L=L_{max}/\lambda$ (bohr)')
        axp.set_ylabel(r'$\epsilon_{xc}$ (hartree)')
        axp.set_title(rf'$r_s^{{2D}}={rs_:.4g}$ (ISTLS not reproduced)', fontsize=9)
        axp.legend(fontsize=8)
        save(fig, f'constantin2008_{tag}')

    fig, axp = plt.subplots(figsize=(5.0, 3.8))
    axp.plot(rows5[:, 0], rows5[:, 1], 'k-', lw=1.8, label='exact reference, Eq. (23) (this work)')
    axp.plot(rows5[:, 0], rows5[:, 2], 'k--', label='LSDA')
    axp.plot(rows5[:, 0], rows5[:, 3], 'k:', label='PBE')
    axp.plot(rows5[:, 0], rows5[:, 4], '-', label='MGGA+2D')
    axp.plot(rows5[:, 0], rows5[:, 5], '-.', label='Eq.(18)+Eq.(22)')
    axp.set_xlim(0, 350)
    axp.set_ylim(-1.6, 0.0)
    axp.set_xlabel(r'$\lambda$')
    axp.set_ylabel(r'$E_{xc}$ (hartree)')
    axp.legend(fontsize=8)
    save(fig, 'constantin2008_fig5')

    # --- our DC-SCAN functions (code 01) on the IBM well, rs2D = 4: limits L -> 0
    rs = 4.0
    n2d_, I_ = slab_integral(rs)
    lim_scan = -CX * SCAN_H0X * SCAN_A1 * K * I_ / n2d_       # Kaplan 2018 App. A (code 09)
    lim_dc = -CX * A2D * K * I_ / n2d_                        # = (A2D / a_x) eps_x^2D: Gaussian-fitted A2D on the slab
    inv_Ls = np.logspace(np.log10(1.0 / lmax(rs)), 9.0, 46)
    rowsD = []
    for il in inv_Ls:
        L = 1.0 / il
        e = well_dcscan(rs, L)
        rowsD.append([il, L, well_exact_exchange(rs, L), e['LDA'], e['SCAN_libxc'], e['SCAN_alpha0'],
                      e['DCSCAN'], e['DCSCAN_libxc_base'], e['N_check'], e['D_max_dev'],
                      e['r2SCAN_libxc'], e['DCr2SCAN_libxc_base']])
    rowsD = np.array(rowsD)
    write_csv(os.path.join(RES, 'constantin2008_well_dcscan.csv'),
              ['inv_L', 'L', 'exact', 'LDA', 'SCAN_libxc', 'SCAN_alpha0_code01', 'DCSCAN_code01',
               'DCSCAN_libxc_base', 'N_check', 'D_max_dev', 'r2SCAN_libxc', 'DCr2SCAN_libxc_base'], rowsD)
    tail = rowsD[:, 0] >= 1e5
    ext = {k: extrapolate_cuberoot(rowsD[tail, 1], rowsD[tail, j]) for k, j in
           (('SCAN_libxc', 4), ('SCAN_alpha0', 5), ('DCSCAN', 6), ('DCSCAN_libxc_base', 7),
            ('r2SCAN_libxc', 10), ('DCr2SCAN_libxc_base', 11))}
    say('')
    say('Our DC-SCAN / SCAN functions (code 01, verbatim) on the IBM well, rs2D = 4')
    say(f'  max |N - 1| = {np.max(np.abs(rowsD[:, 8] - 1.0)):.1e};  max |D - 1| (sheet branch) = {np.max(rowsD[:, 9]):.1e}')
    for il in (1.0 / lmax(rs), 1.0, 10.0, 1e3, 1e6, 1e9):
        i = int(np.argmin(np.abs(np.log(rowsD[:, 0] / il))))
        r = rowsD[i]
        say(f'  1/L={r[0]:9.3g}: exact {r[2]:.5f} SCAN(libxc) {r[4]:.5f} SCAN(a=0, code01) {r[5]:.5f} '
            f'DC-SCAN(code01) {r[6]:.5f} DC-SCAN(libxc base) {r[7]:.5f}')
    say(f'  L -> 0, extrapolated in L^(1/3) ln L, L^(1/3), L^(2/3) from 1/L >= 1e5:')
    say(f'    SCAN (libxc)            {ext["SCAN_libxc"]:.6f}   analytic {lim_scan:.6f}   Kaplan 2018: -1.671')
    say(f'    SCAN alpha=0 (code 01)  {ext["SCAN_alpha0"]:.6f}   analytic {lim_scan:.6f}')
    say(f'    DC-SCAN (code 01)       {ext["DCSCAN"]:.6f}   analytic (A2D/a_x) eps_2D = {lim_dc:.6f}   exact 2D {eps_x_2d(rs):.6f}')
    say(f'    DC-SCAN (libxc base)    {ext["DCSCAN_libxc_base"]:.6f}')
    say(f'    r2SCAN (libxc)          {ext["r2SCAN_libxc"]:.6f}')
    say(f'    DC-r2SCAN (libxc base)  {ext["DCr2SCAN_libxc_base"]:.6f}')
    checks += [('SCAN slab 2D limit (Ha), code 01 fx, extrapolated', ext['SCAN_alpha0'], -1.671),
               ('SCAN slab 2D limit (Ha), Libxc, extrapolated', ext['SCAN_libxc'], -1.671),
               ('DC-SCAN slab 2D limit (Ha), code 01, extrapolated', ext['DCSCAN'], lim_dc)]

    pub_style()
    fig, (pa, pb) = plt.subplots(1, 2, figsize=(PUB_W['sm_095'], 2.9), gridspec_kw=dict(wspace=0.42))
    for axp, zoom in ((pa, False), (pb, True)):
        x = rowsD[:, 0]
        pub_line(axp, x, rowsD[:, 2], 'exact', label='exact reference (this work)')
        pub_line(axp, x, rowsD[:, 4], 'SCAN', label='SCAN (Libxc)', marks=8)
        pub_line(axp, x, rowsD[:, 5], 'SCAN', label=r'SCAN $\alpha=0$ (code 01)', marks=8, color='#7a4a00',
                 ls=(0, (1, 1.3)), marker='<', mec='#7a4a00')
        pub_line(axp, x, rowsD[:, 6], 'DC', label='DC-SCAN (code 01)', marks=8)
        pub_line(axp, x, rowsD[:, 7], 'DC', label='DC-SCAN, Libxc SCAN base', marks=8, ls=(0, (4, 1.5)),
                 marker='d', color='#5fbf8f', mec='#2f8f5f', lw=1.2)
        pub_line(axp, x, rowsD[:, 11], 'DCr2', label=r'DC-r$^2$SCAN, Libxc r$^2$SCAN base', marks=8)
        axp.set_xscale('log')
        lines = [(eps_x_2d(rs), 'exact 2D' if zoom else None), (lim_dc, 'DC-SCAN limit' if zoom else None)]
        if not zoom:
            lines.append((lim_scan, 'SCAN limit'))
        for y, lab in lines:
            axp.axhline(y, color='0.45', lw=0.8, ls=(0, (5, 2)), zorder=0)
            if lab:
                axp.text(1.4e9, y, lab, fontsize=7.5, color='0.2', va='center', ha='left')
        axp.set_xlim(x[0], 1e9)
        axp.set_ylim(*((-0.20, -0.085) if zoom else (-1.8, -0.05)))
        axp.set_xlabel(r'$1/L=\lambda/L_{\max}$ (bohr$^{-1}$)')
        axp.set_ylabel(r'$\epsilon_x$ (Ha)')
    pub_panel(pa, '(a) IBM well,\n' + r'$\mathbf{r_s^{2D}=4}$', x=0.96, y=0.62, ha='right')
    pub_panel(pb, '(b) zoom', x=0.96, y=0.93, ha='right')
    pub_legend(pa, loc='lower left', fontsize=6.5)
    pub_save(fig, os.path.join(FIG, 'constantin2008_well_dcscan'))
    plt.rcdefaults()

    # --- LaTeX rows: paper value vs ours
    tex = ['% generated by codes/14_constantin2008_benchmark.py: quantity & paper & this work & rel. diff']
    for name, ours, ref in checks:
        tex.append(f'{name} & ${ref:.5g}$ & ${ours:.6f}$ & ${(ours - ref) / abs(ref):+.1e}$ \\\\')
    with open(os.path.join(RES, 'tab_constantin2008_rows.tex'), 'w') as fh:
        fh.write('\n'.join(tex) + '\n')
    with open(os.path.join(RES, 'constantin2008_log.txt'), 'w') as fh:
        fh.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
