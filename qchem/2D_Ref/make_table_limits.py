#!/usr/bin/env python3
"""tab_limits.tex: full data of the W = 0 and W = 1 checks (pbe_nonscf_ae6/check_limits_dyall.csv, full precision):
energies (Ha) of both sides and the exact differences, all 21 AE6/BH6 species, r2SCAN densities, dyall-v4z, grid 9."""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "pbe_nonscf_ae6", "check_limits_dyall.csv"))
LAB = {"H": "H$^{*}$", "SiH4": "SiH$_4$", "S2": "S$_2$", "C3H4": "C$_3$H$_4$", "C2H2O2": "C$_2$H$_2$O$_2$", "C4H8": "C$_4$H$_8$",
       "CH4": "CH$_4$", "CH3": "CH$_3$", "H2O": "H$_2$O", "H2": "H$_2^{*}$", "H2S": "H$_2$S", "HS": "SH",
       "TS_OH_CH4": "TS, OH+CH$_4$", "TS_H_OH": "TS, H+OH", "TS_H_H2S": "TS, H+H$_2$S"}
G = "\\textcolor{green!50!black}{%s}"


def d(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


L = ["\\begin{tabular}{@{}l rrr rrr rrr@{}}", "\\toprule",
     " & \\multicolumn{6}{c}{$W=0$ against Libxc spin-polarized r$^2$SCAN} & \\multicolumn{3}{c}{$W=1$ against pure Q2D} \\\\",
     "\\cmidrule(lr){2-7}\\cmidrule(lr){8-10}",
     "Species & $E_x^{\\rm Libxc}$ & $E_x^{W=0}$ & $\\Delta E_x$ & $E_c^{\\rm Libxc}$ & $E_c^{W=0}$ & $\\Delta E_c$"
     " & $E_x^{\\rm Q2D}$ & $E_x^{W=1}$ & $\\Delta E_x$ \\\\", "\\midrule"]
for r in df.itertuples():
    L.append(" & ".join([LAB.get(r.species, r.species),
                         f"{r.ex_libxc:.8f}", f"{r.ex_W0:.8f}", G % d(r.dEx_W0),
                         f"{round(r.ec_libxc, 8) + 0.0:.8f}", f"{round(r.ec_W0, 8) + 0.0:.8f}", G % d(r.dEc_W0),
                         f"{r.ex_Q2D:.8f}", f"{r.ex_W1:.8f}", G % d(r.dEx_W1)]) + " \\\\")
# common species He, Be (H, H2 are rows above): PySCF route of validation/common_species.py, same density as the
# aniGTO_Q2D(lambda=1) rows of Table tab:anigtolimits
cs_path = os.path.join(HERE, "validation", "common_species.csv")
if os.path.exists(cs_path):
    cs = pd.read_csv(cs_path)
    L.append("\\midrule")
    for r in cs[cs.species.isin(["He", "Be"])].itertuples():
        L.append(" & ".join([r.species + "$^{*}$", f"{r.pyscf_Ex_libxc:.8f}", f"{r.pyscf_Ex_W0:.8f}",
                             G % d(r.pyscf_Ex_W0 - r.pyscf_Ex_libxc),
                             f"{round(r.pyscf_Ec_libxc, 8) + 0.0:.8f}", f"{round(r.pyscf_Ec_W0, 8) + 0.0:.8f}",
                             G % d(r.pyscf_Ec_W0 - r.pyscf_Ec_libxc),
                             f"{r.pyscf_Ex_Q2D:.8f}", f"{r.pyscf_Ex_W1:.8f}", G % d(r.pyscf_Ex_W1 - r.pyscf_Ex_Q2D)]) + " \\\\")
L += ["\\midrule", "max $|\\Delta|$ & & & " + G % d(df.dEx_W0.abs().max()) + " & & & " + G % d(df.dEc_W0.abs().max())
      + " & & & " + G % d(df.dEx_W1.abs().max()) + " \\\\", "\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_limits.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L[5:9]))
