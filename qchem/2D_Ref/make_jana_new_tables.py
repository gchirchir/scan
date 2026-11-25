#!/usr/bin/env python3
"""Mirrors of Tables 2-4 (reproduction of jana_ref) with three columns: jana_ref | PySCF original W | PySCF W_new
(G3: t = 1e-3, p0 = 5, k = 12).  Same HF densities (dyall-v4z, grid 9) and the same analytic IBM well.
  tab_jn_atomic.tex   Table 2 mirror: scalar E_x and E_c (Ha)   (r2SCAN columns do not depend on W)
  tab_jn_shift.tex    Table 3 mirror: Delta E_x, Delta E_c (mHa)
  tab_jn_ibm.tex      Table 4 mirror: IBM MAEs (Ha/electron)
Sources: atomic_results.csv (original), w_gate_variant/atomic_variant.csv (W_new shifts on the same densities),
ibm_mae.csv (jana_ref values), ibm_benchmark_new.csv (original and W_new per width)."""
import os

import numpy as np
import pandas as pd

from reproduce_atomic import PDF_TABLE

HERE = os.path.dirname(os.path.abspath(__file__))
RED = "\\textcolor{pyscfred}{%s}"
at = pd.read_csv(os.path.join(HERE, "atomic_results.csv")).set_index("atom")
av = pd.read_csv(os.path.join(HERE, "w_gate_variant", "atomic_variant_v2.csv")).set_index("atom")
import sys  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "w_gate_variant"))
from wnew_params import ATOMIC_KEY as NEW  # noqa: E402
ATOMS = ["H", "He", "Ne", "Ar", "Kr", "Xe", "Mg"]


def new_shift(a, kind):
    return 0.0 if a == "H" else float(av.loc[a, f"{NEW}|d{kind}"])          # mHa; H: one-electron, W = 0


A = ["\\begin{tabular}{@{}l rrr rrr@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{$E_x^{\\rm scalar}$} & \\multicolumn{3}{c}{$E_c^{\\rm scalar}$} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
     "Atom & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ \\\\", "\\midrule"]
S = ["\\begin{tabular}{@{}l rrr rrr@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{$\\Delta E_x$} & \\multicolumn{3}{c}{$\\Delta E_c$} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
     "Atom & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ \\\\", "\\midrule"]
for a in ATOMS:
    p = PDF_TABLE[a]
    r = at.loc[a]
    xo, co = r.scalar_exchange_ha, r.scalar_correlation_ha
    xn = r.r2scan_exchange_ha + new_shift(a, "Ex") / 1e3
    cn = r.r2scan_correlation_ha + new_shift(a, "Ec") / 1e3
    A.append(" & ".join([a, f"{p[1]:.6f}", RED % f"{xo:.6f}", RED % f"{xn:.6f}",
                         f"{p[3]:.6f}", RED % f"{co:.6f}", RED % f"{cn:.6f}"]) + " \\\\")
    S.append(" & ".join([a, f"{1e3 * (p[1] - p[0]):+.3f}", RED % f"{1e3 * (xo - r.r2scan_exchange_ha):+.3f}",
                         RED % f"{new_shift(a, 'Ex'):+.3f}", f"{1e3 * (p[3] - p[2]):+.3f}",
                         RED % f"{1e3 * (co - r.r2scan_correlation_ha):+.3f}", RED % f"{new_shift(a, 'Ec'):+.3f}"])
             + " \\\\")
for T, f in ((A, "tab_jn_atomic.tex"), (S, "tab_jn_shift.tex")):
    T += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(HERE, f), "w").write("\n".join(T) + "\n")

m = pd.read_csv(os.path.join(HERE, "ibm_mae.csv"))
b = pd.read_csv(os.path.join(HERE, "ibm_benchmark_new.csv"))
I = ["\\begin{tabular}{@{}l rrr rrr rrr@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{exchange} & \\multicolumn{3}{c}{correlation} & \\multicolumn{3}{c}{XC} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "Widths & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$"
     " & $W^{\\rm new}$ \\\\", "\\midrule"]
for lab, w, sel in (("$L\\geq1$ bohr (13)", "L >= 1", b.L >= 1), ("$L<1$ bohr (6)", "L < 1", b.L < 1)):
    j = m[(m.widths == w) & (m.functional == "scalar")].iloc[0]
    d = b[sel]
    new = (np.mean(np.abs(d.new_x - d.fock)), np.mean(np.abs(d.new_c - (d.istls - d.fock))),
           np.mean(np.abs(d.new_x + d.new_c - d.istls)))
    I.append(" & ".join([lab, f"{j.pdf_x:.6f}", RED % f"{j.mae_x:.6f}", RED % f"{new[0]:.6f}",
                         f"{j.pdf_c:.6f}", RED % f"{j.mae_c:.6f}", RED % f"{new[1]:.6f}",
                         f"{j.pdf_xc:.6f}", RED % f"{j.mae_xc:.6f}", RED % f"{new[2]:.6f}"]) + " \\\\")
I += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_jn_ibm.tex"), "w").write("\n".join(I) + "\n")
for f in ("tab_jn_atomic.tex", "tab_jn_shift.tex", "tab_jn_ibm.tex"):
    print("=====", f)
    print(open(os.path.join(HERE, f)).read().replace("\\textcolor{pyscfred}", ""))
