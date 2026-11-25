#!/usr/bin/env python3
"""LaTeX table ../tab_ae6_basis.tex: AE6 ME/MAE (kcal/mol) of non-SCF PBE-orbital energies for each basis
(ae6_pbe_nonscf_<basis>_g6.csv from basis_scan.sh, plus ae6_pbe_nonscf.csv = dyall-v4z grid 9)."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RED = "\\textcolor{pyscfred}{%s}"
RUNS = [("def2-TZVP", "def2-tzvp", 6, ""), ("aug-cc-pVTZ", "aug-cc-pvtz", 6, "diffuse"),
        ("def2-QZVPP", "def2-qzvpp", 6, ""), ("def2-QZVPPD", "def2-qzvppd", 6, "diffuse"),
        ("aug-cc-pVQZ", "aug-cc-pvqz", 6, "diffuse"), ("dyall-v4z", "dyall-v4z", 6, ""),
        ("dyall-v4z", None, 9, "")]
K = ("pbe", "r2scan", "SS", "TD")
L = ["\\begin{tabular}{@{}llc rrrr rrrr@{}}", "\\toprule",
     " & & & \\multicolumn{4}{c}{ME} & \\multicolumn{4}{c}{MAE} \\\\",
     "\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}",
     "Basis & & grid & $E^{\\rm PBE}$ & $E^{{\\rm r}^2{\\rm SCAN}}$ & $E^{\\rm SS}$ & $E^{\\rm TD}$"
     " & $E^{\\rm PBE}$ & $E^{{\\rm r}^2{\\rm SCAN}}$ & $E^{\\rm SS}$ & $E^{\\rm TD}$ \\\\", "\\midrule"]
for lab, tag, grid, note in RUNS:
    f = "ae6_pbe_nonscf.csv" if tag is None else f"ae6_pbe_nonscf_{tag}_g{grid}.csv"
    d = pd.read_csv(os.path.join(HERE, f))
    assert len(d) == 6, f
    e = {k: d[k] - d["ref"] for k in K}
    L.append(" & ".join([lab, note, str(grid)] + [RED % f"{e[k].mean():+.2f}" for k in K]
                        + [RED % f"{e[k].abs().mean():.2f}" for k in K]) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(os.path.dirname(HERE), "tab_ae6_basis.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
