#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal:
#
#     cd codes
#     python3 09_literature_validation.py
#
# Needs: numpy, scipy.  Results -> codes/results/
# ----------------------------------------------------------------------
r"""Validation of the sheet-limit construction of DC-SCAN against independent literature values.

STANDALONE SCRIPT (SCAN_1D_2D/codes): run it alone; only numpy and scipy are needed.
Outputs: results/literature_validation.csv, results/literature_validation_log.txt, results/tab_literature_validation_rows.tex

In the lambda -> infinity (2D) limit of a one-orbital (alpha = 0) density the reduced gradient s diverges and any
enhancement factor that tends to  F_x -> A s^(-1/2)  gives a finite exchange energy
    E_x -> -C_X A K  int n^2 |dn/dz|^(-1/2) d^3r ,   K = (2 (3 pi^2)^(1/3))^(1/2),
because n^(4/3) s^(-1/2) = K n^2 |dn/dz|^(-1/2).  We use this single formula in two ways, exactly as in the main text:
  (i)  A fixed so that E_x equals the exact 2D limit (the DC-SCAN sheet coefficient A_2D, eqs A26-A28);
  (ii) A = h_x^0 a_1 (the alpha = 0 SCAN enhancement factor at large s) -> the SCAN limit (eq A25).
Applied to the infinite-barrier slab n(z) = n_2D (2/L) sin^2(pi z/L) with n_2D = 1/(pi r_s^2) this must reproduce
  Constantin, Phys. Rev. B 78, 155106 (2008) / arXiv:0906.5037:  a_x = 0.5217  (F_x -> a_x s^(-1/2), his eq. 12),
  Kaplan, Wagle, Perdew, Phys. Rev. B 98, 085147 (2018):        E_x^SCAN/N -> -1.671 Ha (their Appendix A estimate),
                                                                 -1.655 Ha (their numerical integration),
                                                                 exact 2D value -0.1501 Ha at r_s^2D = 4 (QMC/HF 2D gas),
and applied to the Gaussian sheet of the main text it must reproduce our A_2D = 0.6532 and SCAN/exact = 8.9.
"""
import os

import numpy as np
from scipy.integrate import quad
from scipy.special import beta

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)           # LDA exchange constant (unpolarized), E_x^LDA = -CX int n^(4/3)
K = np.sqrt(2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0))
SCAN_H0X, SCAN_A1 = 1.174, 4.9479                   # SCAN alpha = 0 exchange: F_x -> h0x a1 s^(-1/2) at large s
A2D_NOTE = 0.6532258969913366                       # main text, eq A28 (Gaussian sheet)


def slab_integral(rs2d):
    """int n^2 |n'|^(-1/2) dz per unit area for the infinite-barrier slab n = n_2D (2/L) sin^2(pi z/L).
    Closed form (L cancels): with x = pi z/L,  I = (2 n_2D)^(3/2) 2^(-1/2) pi^(-3/2) int_0^pi sin^(7/2)x |cos x|^(-1/2) dx
    and int_0^pi sin^(7/2)x |cos x|^(-1/2) dx = B(9/4, 1/4)  (the integral of Constantin's eq. 12)."""
    n2d = 1.0 / (np.pi * rs2d ** 2)
    return n2d, (2.0 * n2d) ** 1.5 / np.sqrt(2.0) * np.pi ** -1.5 * beta(2.25, 0.25)


def slab_integral_quadrature(rs2d):
    """Numerical cross-check of slab_integral: half interval with the algebraic weight (L/2 - z)^(-1/2)."""
    n2d = 1.0 / (np.pi * rs2d ** 2)
    g = lambda x: np.sin(x) ** 3.5 / np.sqrt(np.cos(x) / (np.pi / 2 - x)) if x < np.pi / 2 - 1e-12 else 1.0
    v = quad(g, 0.0, np.pi / 2, weight='alg', wvar=(0.0, -0.5), epsabs=0, epsrel=1e-13, limit=400)[0]
    return (2.0 * n2d) ** 1.5 / np.sqrt(2.0) * np.pi ** -1.5 * 2.0 * v


def gaussian_sheet_integral():
    """int n^2 |dn/dz|^(-1/2) d^3r for the two-electron Gaussian sheet n = 2 pi^(-3/2) lam e^{-(x^2+y^2)-lam^2 z^2}
    in the lam -> infinity limit (lam cancels): radial and z integrals done numerically."""
    c = 2.0 / np.pi ** 1.5
    # n^2 |n_z|^(-1/2) = c^2 lam^2 e^{-2r^2-2t^2} / sqrt(c lam e^{-r^2-t^2} 2 lam^2 |z|) with t = lam z, dz = dt/lam
    # -> c^(3/2) 2^(-1/2) e^{-3/2 (r^2+t^2)} |t|^(-1/2) dt   (all lam powers cancel)
    ir = quad(lambda r: 2.0 * np.pi * r * np.exp(-1.5 * r * r), 0, np.inf, epsrel=1e-13)[0]
    it = 2.0 * quad(lambda t: np.exp(-1.5 * t * t) / np.sqrt(t), 0, np.inf, epsrel=1e-13, limit=400)[0]
    return c ** 1.5 / np.sqrt(2.0) * ir * it


def anigto_self_repulsion(a):
    """aniGTO self-repulsion (phi phi|phi phi) of one s Gaussian with exponents a = (a_x, a_y, a_z):
    eq S16 (eq:ani-selfrep) with the closed-form aniGTO kernels I_k(u^2) = pi / (2 sqrt(a_k (a_k + u^2)))
    (eq S15, eq:ani-Ik-s), normalization N_s^4 = prod_k 2 a_k / pi, integrated in the logarithmic variable
    s = ln u of eq 1 (exact at any exponent ratio)."""
    a = np.asarray(a, float)
    ns4 = np.prod(2.0 * a / np.pi)
    f = lambda s_: np.exp(s_) * np.prod(np.pi / (2.0 * np.sqrt(a * (a + np.exp(2.0 * s_)))))
    lo, hi = 0.5 * np.log(a.min()) - 20.0, 0.5 * np.log(a.max()) + 20.0
    v = quad(f, lo, hi, points=np.arange(np.floor(lo) + 1.0, hi, 1.0), limit=4000, epsrel=1e-13, epsabs=0.0)[0]
    return ns4 * 2.0 / np.sqrt(np.pi) * (v + f(lo))      # + [0, u0] contribution (integrand ~ constant there)


TEXQ = {'Gaussian sheet: exact 2D limit E_x (Ha)': 'Gaussian sheet: exact 2D limit $E_x$ (Ha)',
        'slab: exact 2D exchange per electron (Ha), r_s^2D=4': 'slab: exact 2D $E_x/N$ (Ha), $r_s^{2D}=4$',
        'slab: coefficient a_x for the exact 2D limit': 'slab: $a_x$ giving the exact 2D limit',
        'slab: SCAN 2D-limit E_x/N (Ha), r_s^2D=4': 'slab: SCAN 2D limit $E_x/N$ (Ha), analytic',
        'slab: SCAN 2D-limit E_x/N (Ha), numerical': 'slab: SCAN 2D limit $E_x/N$ (Ha), numerical',
        'slab: SCAN/exact ratio': 'slab: SCAN/exact',
        'Gaussian sheet: coefficient A_2D': 'Gaussian sheet: $A_{2D}$',
        'Gaussian sheet: SCAN limit (Ha)': 'Gaussian sheet: SCAN limit (Ha)',
        'Gaussian sheet: SCAN/exact ratio': 'Gaussian sheet: SCAN/exact'}
TEXSRC = {'eq A12 limit': 'eq~\\ref{M-eq:dc-sheetlim}',
          'Kaplan et al. 2018 (2D gas)': 'ref~\\citenum{kaplan2018collapse}',
          'Constantin 2008, eq. 12': 'ref~\\citenum{constantin2008dimensional}, eq~12',
          'Kaplan et al. 2018, App. A estimate': 'ref~\\citenum{kaplan2018collapse}, App.~A',
          'Kaplan et al. 2018, numerical': 'ref~\\citenum{kaplan2018collapse}, numerical',
          'Kaplan et al. 2018 ("order of magnitude")': 'ref~\\citenum{kaplan2018collapse}',
          'this work, eq A28': 'eq~\\ref{M-eq:dc-A2D}',
          'this work, Table 2 / eq A25': 'eq~\\ref{M-eq:dc-M2D}, Table~\\ref{M-tab:dcscan-sheet}',
          'this work, abstract (8.9)': 'main text'}


DOI = {'Kaplan et al. 2018 (2D gas)': '10.1103/PhysRevB.98.085147',
       'Constantin 2008, eq. 12': '10.1103/PhysRevB.78.155106',
       'Kaplan et al. 2018, App. A estimate': '10.1103/PhysRevB.98.085147',
       'Kaplan et al. 2018, numerical': '10.1103/PhysRevB.98.085147',
       'Kaplan et al. 2018 ("order of magnitude")': '10.1103/PhysRevB.98.085147'}


def refcell(ref, src):
    """reference value; literature values are red, clickable hyperlinks to the paper (doi.org)"""
    v = '%.4f' % ref
    return '\\href{https://doi.org/%s}{\\textcolor{red}{%s}}' % (DOI[src], v) if src in DOI else v   # red, clickable


def main():
    os.makedirs(RES, exist_ok=True)
    rows, log = [], []
    # ---- slab of Constantin (2009) and Kaplan-Wagle-Perdew (2018)
    rs = 4.0
    n2d, I = slab_integral(rs)
    ex2d = -4.0 * np.sqrt(2.0) / (3.0 * np.pi * rs)            # 2D uniform-gas exchange per electron
    a_exact = (n2d * ex2d) / (-CX * K * I)                      # coefficient giving the exact 2D limit
    scan_lim = -CX * SCAN_H0X * SCAN_A1 * K * I / n2d            # SCAN 2D-limit exchange per electron
    rows += [('slab: exact 2D exchange per electron (Ha), r_s^2D=4', ex2d, -0.1501, 'Kaplan et al. 2018 (2D gas)'),
             ('slab: coefficient a_x for the exact 2D limit', a_exact, 0.5217, 'Constantin 2008, eq. 12'),
             ('slab: SCAN 2D-limit E_x/N (Ha), r_s^2D=4', scan_lim, -1.671, 'Kaplan et al. 2018, App. A estimate'),
             ('slab: SCAN 2D-limit E_x/N (Ha), numerical', scan_lim, -1.655, 'Kaplan et al. 2018, numerical'),
             ('slab: SCAN/exact ratio', scan_lim / ex2d, -1.655 / -0.1501, 'Kaplan et al. 2018 ("order of magnitude")')]
    log.append('slab integral: closed form %.15f, quadrature %.15f (r_s^2D=4)' % (I, slab_integral_quadrature(rs)))
    # r_s independence of the coefficient (the slab limit is scale-free)
    for rs_ in (1.0, 10.0):
        n2_, I_ = slab_integral(rs_)
        log.append('r_s^2D=%g: a_x = %.10f' % (rs_, (n2_ * (-4 * np.sqrt(2) / (3 * np.pi * rs_))) / (-CX * K * I_)))
    # ---- our Gaussian sheet: same formula must give A_2D (eq A28) and SCAN/exact = 8.9
    G = gaussian_sheet_integral()
    ex_sheet = -np.sqrt(np.pi / 2.0)
    a_sheet = ex_sheet / (-CX * K * G)
    scan_sheet = -CX * SCAN_H0X * SCAN_A1 * K * G
    LAM_ANI = 1.0e8                                            # aniGTO at lambda = 1e8 (exponent ratio 1e16)
    ex_ani = -anigto_self_repulsion((0.5, 0.5, 0.5 * LAM_ANI ** 2))
    log.append('aniGTO exact sheet exchange at lambda=%g: %.12f; closed form eq A10: %.12f; limit -sqrt(pi/2): %.12f'
               % (LAM_ANI, ex_ani, ex_sheet + np.sqrt(2.0 / np.pi) / LAM_ANI, ex_sheet))
    rows += [('Gaussian sheet: exact 2D limit E_x (Ha)', ex_sheet, ex_sheet, 'eq A12 limit', ex_ani),
             ('Gaussian sheet: coefficient A_2D', a_sheet, A2D_NOTE, 'this work, eq A28', ex_ani / (-CX * K * G)),
             ('Gaussian sheet: SCAN limit (Ha)', scan_sheet, -11.145141, 'this work, Table 2 / eq A25', None),
             ('Gaussian sheet: SCAN/exact ratio', scan_sheet / ex_sheet, 8.89, 'this work, abstract (8.9)', scan_sheet / ex_ani)]
    rows = [r if len(r) == 5 else r + (None,) for r in rows]     # slab rows: aniGTO not applicable (non-Gaussian)
    with open(os.path.join(RES, 'literature_validation.csv'), 'w') as f:
        f.write('quantity,this_formula,aniGTO,literature_or_reference,relative_difference,source\n')
        for q, v, ref, src, va in rows:
            f.write('"%s",%.10g,%s,%.10g,%.3e,"%s"\n' % (q, v, '' if va is None else '%.12g' % va, ref,
                                                        (v - ref) / abs(ref), src))
    with open(os.path.join(RES, 'tab_literature_validation_rows.tex'), 'w') as f:   # values first, difference at the end
        for q, v, ref, src, va in rows:
            f.write('%s & %.6f & %s & %s & %s & $%+.2f$\\%% & %s \\\\\n' % (TEXQ.get(q, q), v, ('--' if va is None else '%.6f' % va), refcell(ref, src),
                    TEXSRC.get(src, src), 100 * (v - ref) / abs(ref), ('--' if va is None else '$%+.2f$\\%%' % (100 * (va - ref) / abs(ref)))))
    main_src = {'eq~\\ref{M-eq:dc-A2D}': 'eq~\\ref{eq:dc-A2D}', 'eq~\\ref{M-eq:dc-sheetlim}': 'eq~\\ref{eq:dc-sheetlim}',
                'eq~\\ref{M-eq:dc-M2D}, Table~\\ref{M-tab:dcscan-sheet}': 'eq~\\ref{eq:dc-M2D}, Table~\\ref{tab:dcscan-sheet}'}
    with open(os.path.join(RES, 'tab_literature_validation_main_rows.tex'), 'w') as f:   # main-text copy
        for q, v, ref, src, va in rows:
            sr = TEXSRC.get(src, src)
            f.write('%s & %.6f & %s & %s & %s & $%+.2f$\\%% & %s \\\\\n' % (TEXQ.get(q, q), v, ('--' if va is None else '%.6f' % va), refcell(ref, src),
                    main_src.get(sr, sr), 100 * (v - ref) / abs(ref), ('--' if va is None else '$%+.2f$\\%%' % (100 * (va - ref) / abs(ref)))))
    lines = ['%-46s this formula %12.6f | aniGTO %12s | reference %10.4f | rel.diff %+.2e | %s'
             % (q, v, '--' if va is None else '%.6f' % va, ref, (v - ref) / abs(ref), src)
             for q, v, ref, src, va in rows] + log
    open(os.path.join(RES, 'literature_validation_log.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
