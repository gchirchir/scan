#!/usr/bin/env python3
"""LaTeX table ../tab_ae6_pbe.tex: AE6 errors of non-SCF r2SCAN and scalar functional on PBE orbitals
(species/*.json from run_pbe_nonscf.py), next to the r2SCAN-orbital results of ../ae6_results.csv.
Molecules whose species are not finished yet are shown as '--' and left out of ME/MAE."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import pandas as pd                                                        # noqa: E402
from reproduce_ae6_bh6 import AE6_REF, AE6_LABEL, MOLECULES, read_xyz, HA2KCAL   # noqa: E402

RED = "\\textcolor{pyscfred}{%s}"
sp = {f[:-5]: json.load(open(os.path.join(HERE, "species", f)))
      for f in os.listdir(os.path.join(HERE, "species")) if f.endswith(".json")}
old = pd.read_csv(os.path.join(PARENT, "ae6_results.csv")).set_index("molecule")
COLS = [("pbe", "PBE"), ("r2scan", "r2SCAN"), ("SS", "SS"), ("TD", "TD")]
err = {k: [] for k in [c for c, _ in COLS] + ["r2_old", "SS_old", "TD_old"]}
L = ["\\begin{tabular}{@{}l r rrrr rrr@{}}", "\\toprule",
     " & & \\multicolumn{4}{c}{PBE orbitals (non-SCF)} & \\multicolumn{3}{c}{r$^2$SCAN orbitals (Table~\\ref{tab:ae6})} \\\\",
     "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}",
     "Molecule & $E^{\\rm ref}$ & $E^{\\rm PBE}$ & $E^{{\\rm r}^2{\\rm SCAN}}$ & $E^{\\rm SS}$ & $E^{\\rm TD}$"
     " & $E^{{\\rm r}^2{\\rm SCAN}}$ & $E^{\\rm SS}$ & $E^{\\rm TD}$ \\\\", "\\midrule"]
for m, ref in AE6_REF.items():
    at = [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
    row = [AE6_LABEL[m], f"{ref:.2f}"]
    if set(at + [m]) <= set(sp):
        for k, _ in COLS:
            e = (sum(sp[a]["e_" + k] for a in at) - sp[m]["e_" + k]) * HA2KCAL - ref
            err[k].append(e)
            row.append(RED % f"{e:+.2f}")
        done = True
    else:
        row += ["--"] * 4
        done = False
    for k, c in (("r2_old", "r2scan"), ("SS_old", "scalarA"), ("TD_old", "scalarB")):
        e = old.loc[m, c] - ref
        if done:
            err[k].append(e)
        row.append(RED % f"{e:+.2f}")
    L.append(" & ".join(row) + " \\\\")
L.append("\\midrule")
order = [c for c, _ in COLS] + ["r2_old", "SS_old", "TD_old"]
for lab, fn, fmt in (("ME", lambda e: sum(e) / len(e), "+.2f"), ("MAE", lambda e: sum(map(abs, e)) / len(e), ".2f")):
    L.append(lab + " & & " + " & ".join(RED % format(fn(err[k]), fmt) for k in order) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(PARENT, "tab_ae6_pbe.tex"), "w").write("\n".join(L) + "\n")
n = len(err["pbe"])
print(f"wrote tab_ae6_pbe.tex ({n} of {len(AE6_REF)} molecules)")
