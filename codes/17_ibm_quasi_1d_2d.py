#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal (after code 14 has written results/constantin2008_well_dcscan.csv):
#
#     cd SCAN_1D_2D        # (the folder that contains codes/)
#     ~/miniconda3/envs/anigto/bin/python codes/17_ibm_quasi_1d_2d.py
#
# Needs: numpy, scipy, matplotlib, pyscf (libxc).  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""DC-SCAN on infinite-barrier-model (IBM) quasi-2D wells and quasi-1D wires: exact exchange as reference.

STANDALONE SCRIPT (SCAN_1D_2D/codes): every function it uses is defined here.  The DC-SCAN pieces (fx_scan_alpha0,
f2d, f1d, geometry_weights, omega_switch, fx_dcscan, _log_gl, m2d/A2D) are copied verbatim from
codes/01_wire_benchmark.py.  The quasi-2D well data are read from results/constantin2008_well_dcscan.csv (code 14).

Quasi-1D IBM wire (cylinder of radius R, infinite barrier, lowest subband only, spin unpolarized, n1D = 1/(2 rs1D)):
    n(rho) = n1D J0(j rho/R)^2 / (pi R^2 J1(j)^2),  j = j_{0,1} = 2.404826,   k_F = pi n1D / 2,
    tau    = tau_W + n k_F^2 / 6            (axial plane waves, <k^2>/2 = k_F^2/6 per electron),
    single-subband condition  k_F^2/2 < (j11^2 - j01^2)/(2 R^2)  ->  R < R_max = sqrt(j11^2 - j01^2)/k_F.
  Exact exchange per electron (Fermi-segment overlap 2k_F - |q|, effective 1D interaction V):
    eps_x = -(1/(2 pi^2 n1D)) int_0^{2k_F} (2k_F - q) V(qR) dq,
    V(Q)  = 2 int_0^inf x F(x)^2 / (Q^2 + x^2) dx,   F(x) = int_0^1 2u J0(ju)^2 J0(xu) du / J1(j)^2.
  Leading thin-wire law (V -> -2 ln Q):  eps_x -> -(n1D/2) ln(1/R) + const   (universal, profile independent).
  Curvature C = -1/2 dd ln n: radial C_rr = (j/R)^2 [1 - J1/(x J0) + (J1/J0)^2], azimuthal C_pp = (j/R)^2 J1/(x J0),
  axial 0;  D = (C_rr^2 + C_pp^2)/C_rr^2 (= 2 on the axis, -> 1 at the wall),  Z = sqrt(C_rr^2 + C_pp^2)/k_F^2.
Quasi-2D IBM well (code 14): n(z) = (2 n2D/L) sin^2(pi z/L), D = 1 everywhere, exact single-subband exchange.

Functionals (exchange only): Libxc LDA, SCAN, r2SCAN (true tau); DC-SCAN / DC-r2SCAN = fx_dcscan with the Libxc
SCAN / r2SCAN enhancement factor as base; DC-SCAN(alpha=0) = code-01 production form (fx_scan_alpha0 base).
Outputs: results/ibm_wire_dcscan.csv, results/ibm_quasi_1d_2d_log.txt, results/tab_ibm_rows.tex,
         figures/ibm_quasi_1d_2d.pdf/.png
"""
import math
import os
import warnings

import numpy as np
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

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)            # LDA exchange constant

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


# ==================================================== quasi-1D IBM wire
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


def wire_semilocal(rs1d, R, n=700):
    """Exchange per electron of the IBM wire: Libxc LDA/SCAN/r2SCAN and DC-SCAN variants (code-01 functions)."""
    n1d = 1.0 / (2.0 * rs1d)
    kf1 = math.pi * n1d / 2.0
    ua, wa = _log_gl(1e-13, 0.5, n)                     # from the axis
    vb, wb = _log_gl(1e-13, 0.5, n)                     # from the wall, u = 1 - v
    u = np.concatenate((ua, 1.0 - vb))
    w = 2.0 * math.pi * R * R * u * np.concatenate((wa, wb))      # 2 pi rho drho
    x = J01 * u
    J0, J1 = j0(x), j1(x)
    amp = n1d / (math.pi * R * R * J1J ** 2)
    k = J01 / R
    rho = amp * J0 * J0
    g = np.abs(amp * k * 2.0 * J0 * J1)
    lap = amp * k * k * (2.0 * J1 * J1 - 2.0 * J0 * J0)
    tau = g * g / (8.0 * rho) + rho * kf1 ** 2 / 6.0
    kf = KF * np.cbrt(rho)
    s = g / (2.0 * kf * rho)
    rt = J1 / J0
    rx = np.where(x > 1e-6, J1 / (x * J0), 0.5 - x * x / 16.0)
    crr = k * k * (1.0 - rx + rt * rt)
    cpp = k * k * rx
    D = np.clip(np.where(crr != 0.0, (crr * crr + cpp * cpp) / np.where(crr != 0.0, crr * crr, 1.0), 3.0), 1.0, 3.0)
    Z = np.sqrt(crr * crr + cpp * cpp) / kf ** 2
    e_lda = -CX * np.cbrt(rho)
    zz = np.zeros_like(rho)
    inp = np.vstack((rho, g, zz, zz, lap, tau))
    ok = e_lda != 0.0
    fs0 = fx_scan_alpha0(s)
    fx = {}
    for key, code in (('SCAN', 'MGGA_X_SCAN,'), ('r2SCAN', 'MGGA_X_R2SCAN,')):
        e = np.asarray(libxc.eval_xc(code, inp, spin=0, deriv=0)[0])
        fx[key] = np.where(ok, e / np.where(ok, e_lda, 1.0), fs0)
    per = lambda f: float(np.sum(w * rho * e_lda * f) / n1d)
    return {
        'LDA': per(np.ones_like(rho)),
        'SCAN': per(fx['SCAN']),
        'r2SCAN': per(fx['r2SCAN']),
        'DCSCAN': per(fx_dcscan(s, D, Z, fx['SCAN'], 'full')),
        'DCr2SCAN': per(fx_dcscan(s, D, Z, fx['r2SCAN'], 'full')),
        'DCSCAN_alpha0': per(fx_dcscan(s, D, Z, fs0, 'full')),
        'DCSCAN_D2': per(fx_dcscan(s, np.full_like(D, 2.0), Z, fx['SCAN'], 'full')),   # diagnostic: wire branch forced
        'N_check': float(np.sum(w * rho) / n1d),
        'D_axis': float(D[0]), 'D_half': float(D[-1]), 'D_wall': float(D[n]),   # u -> 0, u = 1/2, u -> 1
    }


class Table(dict):
    """Columns as numpy arrays, attribute access (keeps this script free of pandas)."""
    __getattr__ = dict.__getitem__

    @classmethod
    def read(cls, path):
        a = np.genfromtxt(path, delimiter=',', names=True)
        return cls({k: np.asarray(a[k], dtype=float) for k in a.dtype.names})

    def write(self, path):
        keys = list(self)
        np.savetxt(path, np.column_stack([self[k] for k in keys]), delimiter=',', header=','.join(keys),
                   comments='', fmt='%.15g')

    def row(self, i):
        return Table({k: v[i] for k, v in self.items()})


def secant(x, y):
    return np.r_[np.nan, np.diff(y) / np.diff(np.log(x))]


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    log = []
    say = lambda t='': (print(t, flush=True), log.append(t))
    say('IBM quasi-2D wells and quasi-1D wires: DC-SCAN vs exact exchange (script 17)')

    ex = WireExact()
    ex2 = WireExact(nu=6000, xmax=5000.0)
    say(f'transverse density norm = {ex.norm:.12f};  V(Q) small-Q law: V = -2 ln Q + {ex.v0:.6f}')
    rs1d = 1.0
    n1d = 1.0 / (2.0 * rs1d)
    Rm = rmax(rs1d)
    say(f'rs1D = {rs1d}, n1D = {n1d}, k_F = {math.pi * n1d / 2:.6f}, R_max = {Rm:.6f} bohr (one subband)')
    say(f'quadrature check: eps_x(R_max) {ex.eps_x(rs1d, Rm):.10f} vs finer table {ex2.eps_x(rs1d, Rm):.10f}; '
        f'eps_x(1e-6) {ex.eps_x(rs1d, 1e-6):.10f} vs {ex2.eps_x(rs1d, 1e-6):.10f}')

    inv_R = np.logspace(np.log10(1.0 / Rm), 9.0, 46)
    rows = []
    for ir in inv_R:
        R = 1.0 / ir
        e = wire_semilocal(rs1d, R)
        rows.append(dict(inv_R=ir, R=R, exact=ex.eps_x(rs1d, R), **e))
    d = Table({k: np.array([r[k] for r in rows]) for k in rows[0]})
    chk = wire_semilocal(rs1d, d.R[-1], n=1100)
    say(f'grid check at R = {d.R[-1]:.1e}: DC-SCAN {d.DCSCAN[-1]:.8f} (n=700) vs {chk["DCSCAN"]:.8f} (n=1100); '
        f'max |N-1| = {np.max(np.abs(d.N_check - 1.0)):.1e};  D(axis) = {d.D_axis[0]:.4f}, D(u=1/2) = {d.D_half[0]:.4f}, D(wall) = {d.D_wall[0]:.4f}')
    for c in ('exact', 'LDA', 'SCAN', 'r2SCAN', 'DCSCAN', 'DCr2SCAN', 'DCSCAN_alpha0', 'DCSCAN_D2'):
        d['slope_' + c] = secant(d.inv_R, d[c])
    d.write(os.path.join(RES, 'ibm_wire_dcscan.csv'))

    say('')
    say('quasi-1D IBM wire, rs1D = 1: exchange per electron (Ha)')
    say('     1/R        exact       LDA      SCAN    r2SCAN   DC-SCAN DC-r2SCAN DC-SCAN(a=0)  DC/exact')
    for ir in (1.0 / Rm, 1.0, 10.0, 1e2, 1e3, 1e4, 1e6, 1e9):
        r = d.row(int(np.argmin(np.abs(np.log(d.inv_R / ir)))))
        say(f'  {r.inv_R:9.3g} {r.exact:9.5f} {r.LDA:9.5f} {r.SCAN:9.5f} {r.r2SCAN:9.5f} {r.DCSCAN:9.5f} '
            f'{r.DCr2SCAN:9.5f} {r.DCSCAN_alpha0:9.5f}   {r.DCSCAN / r.exact:.4f}')
    say(f'slope d eps / d ln(1/R) at the thinnest wires (analytic exact: -n1D/2 = {-n1d / 2:.4f}):')
    for c in ('exact', 'DCSCAN', 'DCr2SCAN', 'SCAN', 'DCSCAN_D2'):
        say(f'  {c:9s} ' + '  '.join(f'{v:.5f}' for v in d['slope_' + c][-4:]))
    i3, i9 = int(np.argmin(np.abs(np.log(d.R / 1e-3)))), len(d.R) - 1
    sl = (d.DCSCAN_D2[i9] - d.DCSCAN_D2[i3]) / np.log(d.R[i3] / d.R[i9])
    sx = (d.exact[i9] - d.exact[i3]) / np.log(d.R[i3] / d.R[i9])
    say(f'diagnostic, D forced to 2 (wire branch everywhere): mean slope R = {d.R[i3]:.1e} -> {d.R[i9]:.0e}: '
        f'{-sl:.4f} vs exact {-sx:.4f} (ratio {sl / sx:.3f})')

    w2 = Table.read(os.path.join(RES, 'constantin2008_well_dcscan.csv'))
    say('')
    say('quasi-2D IBM well, rs2D = 4 (from code 14): exchange per electron (Ha)')
    for il in (w2.inv_L[0], 1.0, 10.0, 1e3, 1e6, 1e9):
        r = w2.row(int(np.argmin(np.abs(np.log(w2.inv_L / il)))))
        say(f'  1/L={r.inv_L:9.3g}  exact {r.exact:.5f}  SCAN {r.SCAN_libxc:.5f}  r2SCAN {r.r2SCAN_libxc:.5f}  '
            f'DC-SCAN {r.DCSCAN_libxc_base:.5f}  DC-r2SCAN {r.DCr2SCAN_libxc_base:.5f}  DC/exact {r.DCSCAN_libxc_base / r.exact:.4f}')

    # --- LaTeX rows for the main-text table
    tex = ['% generated by codes/17_ibm_quasi_1d_2d.py: system & thickness (bohr) & exact & SCAN & r2SCAN & DC-SCAN & '
           'DC-r2SCAN & DC-SCAN/exact  (exchange per electron, Ha)']
    pick = lambda df, col, v: df.row(int(np.argmin(np.abs(np.log(df[col] / v)))))
    fmt = lambda v: f'{v:.4f}' if abs(v) < 100 else f'{v:.0f}'
    for v in (w2.L[0], 1.0, 1e-2, 1e-4, 1e-9):
        r = pick(w2, 'L', v)
        tex.append(f'Quasi-2D well & $L={_tex_num(r.L)}$ & ${fmt(r.exact)}$ & ${fmt(r.SCAN_libxc)}$ & '
                   f'${fmt(r.r2SCAN_libxc)}$ & ${fmt(r.DCSCAN_libxc_base)}$ & ${fmt(r.DCr2SCAN_libxc_base)}$ & '
                   f'{r.DCSCAN_libxc_base / r.exact:.3f} \\\\')
    for v in (Rm, 1.0, 1e-2, 1e-4, 1e-9):
        r = pick(d, 'R', v)
        tex.append(f'Quasi-1D wire & $R={_tex_num(r.R)}$ & ${fmt(r.exact)}$ & ${fmt(r.SCAN)}$ & ${fmt(r.r2SCAN)}$ & '
                   f'${fmt(r.DCSCAN)}$ & ${fmt(r.DCr2SCAN)}$ & {r.DCSCAN / r.exact:.3g} \\\\')
    with open(os.path.join(RES, 'tab_ibm_rows.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')

    plot(w2, d, n1d)
    with open(os.path.join(RES, 'ibm_quasi_1d_2d_log.txt'), 'w') as f:
        f.write('\n'.join(log) + '\n')
    say('wrote results/ibm_wire_dcscan.csv, tab_ibm_rows.tex, ibm_quasi_1d_2d_log.txt, figures/ibm_quasi_1d_2d.pdf')


def _tex_num(v):
    if 0.1 <= v < 100:
        return f'{v:.2f}'
    e = int(math.floor(math.log10(v)))
    m = v / 10 ** e
    return f'10^{{{e}}}' if abs(m - 1) < 0.05 else f'{m:.1f}\\times10^{{{e}}}'


def plot(w2, d, n1d):
    pub_style()
    fig, axs = plt.subplots(2, 2, figsize=(PUB_W['main_wide'], 4.9), sharex='col',
                            gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.06, wspace=0.27))
    sets = ((0, w2, 'inv_L', dict(exact='exact', LDA='LDA', SCAN='SCAN_libxc', r2SCAN='r2SCAN_libxc',
                                    DC='DCSCAN_libxc_base', DCr2='DCr2SCAN_libxc_base'),
             '(a) quasi-2D well\n' + r'$\mathbf{r_s^{2D}=4}$', r'$1/L$ (bohr$^{-1}$)'),
            (1, d, 'inv_R', dict(exact='exact', LDA='LDA', SCAN='SCAN', r2SCAN='r2SCAN', DC='DCSCAN', DCr2='DCr2SCAN'),
             '(b) quasi-1D wire\n' + r'$\mathbf{r_s^{1D}=1}$', r'$1/R$ (bohr$^{-1}$)'))
    lab = dict(exact='exact reference', LDA='LDA', SCAN='SCAN', r2SCAN=r'r$^2$SCAN', DC='DC-SCAN',
               DCr2=r'DC-r$^2$SCAN')
    for col, df, xc, cols, title, xl in sets:
        ax, axr = axs[0, col], axs[1, col]
        x = np.asarray(df[xc])
        for key in ('exact', 'LDA', 'SCAN', 'r2SCAN', 'DC', 'DCr2'):
            y = np.asarray(df[cols[key]])
            pub_line(ax, x, y, key, label=lab[key], marks=8)
            pub_line(axr, x, y / np.asarray(df['exact']), key, marks=8)
        if col == 1:
            ref = df['exact'][-1] - n1d / 2.0 * (np.log(x) - np.log(x[-1]))
            ax.plot(x, ref, color=PUB['guide'][0], lw=1.0, ls=(0, (1, 1.5)),
                    label=r'exact law $-\frac{n_{1D}}{2}\ln\frac{1}{R}$')
        ax.set_xscale('log')
        axr.set_xscale('log')
        if col == 0:
            pub_panel(ax, title, x=0.96, y=0.60, ha='right')
        else:
            pub_panel(ax, title, x=0.05, y=0.78)
        ax.set_ylabel(r'$\epsilon_x$ (Ha)')
        axr.axhline(1.0, color='k', lw=0.8)
        axr.set_ylabel(r'$\epsilon_x/\epsilon_x^{\rm exact}$')
        axr.set_xlabel(xl)
        axr.set_ylim(0.0, 3.0)
        ax.set_xlim(x[0], x[-1])
    axs[0, 0].set_ylim(-1.8, -0.05)
    axs[0, 1].set_ylim(min(d.DCSCAN.min(), d.exact.min()) * 1.6, -0.0)
    pub_legend(axs[0, 0], loc='lower left', fontsize=7)
    h, l = axs[0, 1].get_legend_handles_labels()
    pub_legend(axs[0, 1], handles=h[-1:], labels=l[-1:], loc='lower left', fontsize=7)
    pub_save(fig, os.path.join(FIG, 'ibm_quasi_1d_2d'))


if __name__ == '__main__':
    main()
