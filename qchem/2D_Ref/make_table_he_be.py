#!/usr/bin/env python3
"""tab_he_be.tex: common species He, Be, H, H2 side by side (validation/common_species.py): E_x, E_c with W = 0, W = 1,
W_orig (jana_ref), W_new across the header, for three routes on one r2SCAN density (dyall-v4z, grid 9):
PySCF (route of Table tab:limits) | aniGTO_Q2D(lambda=1), PySCF grid | aniGTO_Q2D(lambda=1), radial grid (atoms).
Rows: E_x, E_c and Delta = E(aniGTO) - E(PySCF)."""
import os
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "validation", "common_species.csv"))
MODES = ("W0", "W1", "Worig", "Wnew")
WL = ["$W=0$", "$W=1$", "$W^{\\rm orig}_{\\rm jana\\_ref}$", "$W^{\\rm new}$"]
ROUTES = [("pyscf", "PySCF (Table~\\ref{tab:limits})"), ("ani_pg", "aniGTO\\_Q2D($\\lambda=1$), PySCF grid"),
          ("ani_rad", "aniGTO\\_Q2D($\\lambda=1$), radial grid")]
LAB = {"He": "He", "Be": "Be", "H": "H", "H2": "H$_2$"}


def d(x):
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def v(x):
    return "--" if not np.isfinite(x) else f"{round(x, 8) + 0.0:.8f}"


L = ["\\begin{tabular}{@{}ll " + " ".join(["rrrr"] * 3) + "@{}}", "\\toprule",
     " & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{t}}}" for _, t in ROUTES) + " \\\\",
     "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}",
     "Atom & & " + " & ".join(WL * 3) + " \\\\", "\\midrule"]
for i, r in df.iterrows():
    if i:
        L.append("\\midrule")
    for j, c in enumerate(("Ex", "Ec")):
        cells = [LAB[r.species] if j == 0 else "", f"$E_{c[1]}$"]
        for rt, _ in ROUTES:
            cells += [v(r.get(f"{rt}_{c}_{m}", np.nan)) for m in MODES]
        L.append(" & ".join(cells) + " \\\\")
    for c in ("Ex", "Ec"):
        cells = ["", f"$\\Delta E_{c[1]}$"] + [""] * 4
        for rt, _ in ROUTES[1:]:
            cells += [d(r.get(f"{rt}_{c}_{m}", np.nan) - r[f"pyscf_{c}_{m}"]) for m in MODES]
        L.append(" & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_he_be.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
