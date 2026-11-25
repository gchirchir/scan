#!/usr/bin/env python3
"""tab_new_pbe_ae6.tex / tab_new_pbe_bh6.tex: AE6 and BH6 on PBE orbitals (non-SCF, energy only, no potential;
dyall-v4z, grid level 9): PBE, r2SCAN, original weight (SS, TD), proposed W_new (SS, TD); errors vs reference.
Sources: pbe_nonscf_ae6/ae6_pbe_new_dyall-v4z_g9.csv and bh6_pbe_new_dyall-v4z_g9.csv (run_pbe_new.py)."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RED = "\\textcolor{pyscfred}{%s}"
COLS = [("pbe", "PBE"), ("r2scan", "r$^2$SCAN"), ("orig_SS", "SS"), ("orig_TD", "TD"), ("newJ_SS", "SS"),
        ("newJ_TD", "TD")]


def table(df, labcol, head, out):
    L = ["\\begin{tabular}{@{}l r rr rr rr@{}}", "\\toprule",
         " & & & & \\multicolumn{2}{c}{$W^{\\rm orig}_{\\rm jana\\_ref}$} & \\multicolumn{2}{c}{$W^{\\rm new}$} \\\\",
         "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         f"{head} & $E^{{\\rm ref}}$ & " + " & ".join(c[1] for c in COLS) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        L.append(" & ".join([r[labcol], f"{r.ref:.2f}"] + [RED % f"{r[c] - r.ref:+.2f}" for c, _ in COLS]) + " \\\\")
    L.append("\\midrule")
    L.append("ME & & " + " & ".join(RED % f"{(df[c] - df.ref).mean():+.2f}" for c, _ in COLS) + " \\\\")
    L.append("MAE & & " + " & ".join(RED % f"{(df[c] - df.ref).abs().mean():.2f}" for c, _ in COLS) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(HERE, out), "w").write("\n".join(L) + "\n")
    print("\n".join(L).replace("\\textcolor{pyscfred}", ""))


ae = pd.read_csv(os.path.join(HERE, "pbe_nonscf_ae6", "ae6_pbe_new_dyall-v4z_g9.csv"))
table(ae, "label", "Molecule", "tab_new_pbe_ae6.tex")
bh = pd.read_csv(os.path.join(HERE, "pbe_nonscf_ae6", "bh6_pbe_new_dyall-v4z_g9.csv"))
lab = []
for _, r in bh.iterrows():
    s = r.reaction.replace("+ HS", "+ SH")
    if r.direction == "reverse":
        lhs, rhs = s.split(" $\\to$ ")
        s = rhs + " $\\to$ " + lhs
    lab.append(s)
bh["rx"] = lab
table(bh, "rx", "Reaction", "tab_new_pbe_bh6.tex")
