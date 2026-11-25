#!/usr/bin/env python3
"""tab_new_xc.tex: Table 5 with W_new.  E_x and E_c (Ha) of the 21 AE6/BH6 species on their r2SCAN densities
(dyall-v4z, grid 9): r2SCAN, scalar with the original weight (SS), scalar with W_new (SS).
Sources: ae6_bh6_xc.csv (r2SCAN and original, as Table 5), new_scheme_species_shifts.csv (W_new shifts)."""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
RED = "\\textcolor{pyscfred}{%s}"
x = pd.read_csv(os.path.join(HERE, "ae6_bh6_xc.csv")).set_index("species")
n = pd.read_csv(os.path.join(HERE, "new_scheme_species_shifts.csv")).set_index("species")
chk = max(abs(1e3 * (x.ex_scalarA - x.ex_r2scan) - n.dEx_orig).max(), abs(1e3 * (x.ec_scalar - x.ec_r2scan) - n.dEc_orig).max())
print(f"consistency original shifts (mHa): {chk:.1e}")
LABEL = {"SiH4": "SiH$_4$", "S2": "S$_2$", "C3H4": "C$_3$H$_4$", "C2H2O2": "C$_2$H$_2$O$_2$", "C4H8": "C$_4$H$_8$",
         "CH4": "CH$_4$", "CH3": "CH$_3$", "H2O": "H$_2$O", "H2": "H$_2$", "H2S": "H$_2$S", "HS": "SH",
         "TS_OH_CH4": "TS, OH+CH$_4$", "TS_H_OH": "TS, H+OH", "TS_H_H2S": "TS, H+H$_2$S"}
GROUP = {"H": "atoms", "SiH4": "AE6 molecules", "OH": "BH6 species"}
L = ["\\begin{tabular}{@{}lc rrr rrr@{}}", "\\toprule",
     " & & \\multicolumn{3}{c}{$E_x$ (Ha)} & \\multicolumn{3}{c}{$E_c$ (Ha)} \\\\",
     "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
     "Species & $2S$ & r$^2$SCAN & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & r$^2$SCAN & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ \\\\"]
for s, r in x.iterrows():
    if s in GROUP:
        L += ["\\midrule", f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{GROUP[s]}}}}} \\\\"]
    L.append(" & ".join([LABEL.get(s, s), str(int(r.spin2S))] + [RED % f"{round(v, 6) + 0.0:.6f}" for v in (
        r.ex_r2scan, r.ex_scalarA, r.ex_r2scan + n.loc[s, "dEx_new"] / 1e3,
        r.ec_r2scan, r.ec_scalar, r.ec_r2scan + n.loc[s, "dEc_new"] / 1e3)]) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_new_xc.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L[5:12]).replace("\\textcolor{pyscfred}", ""))
