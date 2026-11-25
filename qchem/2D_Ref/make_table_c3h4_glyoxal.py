#!/usr/bin/env python3
"""tab_c3h4_glyoxal.tex: propyne and glyoxal atomization energies (kcal/mol) of the original functional (Table 7) and of
the tuned gate variants (w_gate_variant/tune_c3h4_glyoxal.csv), r2SCAN densities, dyall-v4z, grid 9."""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
RED = "\\textcolor{pyscfred}{%s}"
ae = pd.read_csv(os.path.join(HERE, "ae6_results.csv")).set_index("molecule")
tu = pd.read_csv(os.path.join(HERE, "w_gate_variant", "tune_c3h4_glyoxal.csv"))
ibm = pd.read_csv(os.path.join(HERE, "ibm_mae.csv"))
ix = lambda fn, w: float(ibm[(ibm.functional == fn) & (ibm.widths == w)].mae_x.iloc[0])
ref = ae.ref
rows = [("r$^2$SCAN", "--", ae.loc["C3H4", "r2scan"] - ref["C3H4"], ae.loc["C2H2O2", "r2scan"] - ref["C2H2O2"],
         (ae.r2scan - ref).abs().mean(), ix("r2scan", "L >= 1"), ix("r2scan", "L < 1")),
        ("SS", "original", ae.loc["C3H4", "scalarA"] - ref["C3H4"], ae.loc["C2H2O2", "scalarA"] - ref["C2H2O2"],
         (ae.scalarA - ref).abs().mean(), ix("scalar", "L >= 1"), ix("scalar", "L < 1")),
        ("TD", "original", ae.loc["C3H4", "scalarB"] - ref["C3H4"], ae.loc["C2H2O2", "scalarB"] - ref["C2H2O2"],
         (ae.scalarB - ref).abs().mean(), ix("scalar", "L >= 1"), ix("scalar", "L < 1"))]
for route, t, p0, k in (("SS", 1e-3, 5, 12), ("SS", 1e-2, 8, 12), ("TD", 1e-2, 4, 4), ("TD", 1e-4, 6, 12)):
    r = tu[(abs(tu.t - t) < 1e-12) & (tu.p0 == p0) & (tu.k == k)].iloc[0]
    rows.append((route, f"$t{{=}}10^{{{int(round(__import__('math').log10(t)))}}}$, $p_0{{=}}{p0}$, $k{{=}}{k}$",
                 r[f"{route}_C3H4"], r[f"{route}_C2H2O2"], r[f"{route}_AE6"], r.ibm_x_ge1, r.ibm_x_lt1))
L = ["\\begin{tabular}{@{}ll rr rr r rr@{}}", "\\toprule",
     " & & \\multicolumn{2}{c}{C$_3$H$_4$ (propyne)} & \\multicolumn{2}{c}{C$_2$H$_2$O$_2$ (glyoxal)} & AE6 &"
     " \\multicolumn{2}{c}{IBM MAE $E_x$ (Ha)} \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){8-9}",
     f"Route & Gate & $E$ & $E-E^{{\\rm ref}}$ & $E$ & $E-E^{{\\rm ref}}$ & MAE & $L\\geq1$ & $L<1$ \\\\",
     "\\midrule",
     f"Reference & & {ref['C3H4']:.2f} & & {ref['C2H2O2']:.2f} & & & & \\\\", "\\midrule"]
for route, gate, e1, e2, mae, i1, i2 in rows:
    L.append(" & ".join([route, gate, RED % f"{ref['C3H4'] + e1:.2f}", RED % f"{e1:+.2f}", RED % f"{ref['C2H2O2'] + e2:.2f}",
                         RED % f"{e2:+.2f}", RED % f"{mae:.2f}", RED % f"{i1:.4f}", RED % f"{i2:.4f}"]) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_c3h4_glyoxal.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
