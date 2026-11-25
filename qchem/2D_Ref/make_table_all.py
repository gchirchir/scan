#!/usr/bin/env python3
"""Comparison tables of all approaches tried for AE6 (and BH6, IBM where defined), against the reference
(Haunschild & Klopper E_nrel).  Writes tab_all_ae6.tex (per-molecule errors) and tab_all_summary.tex.

Sources (all kept; nothing recomputed here except the gate variants, which are evaluated from the cache):
  ae6_results.csv / bh6_results.csv                 r2SCAN densities (Tables 7, 8)
  pbe_nonscf_ae6/ae6_pbe_nonscf.csv                 PBE orbitals, dyall-v4z, grid 9 (Table 9)
  w_gate_variant/cache + explore.evaluate_gate      gate variants on r2SCAN densities
  ibm_mae.csv                                       IBM MAEs of the original functional and r2SCAN
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "w_gate_variant"))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
from explore import load, evaluate_gate, g             # noqa: E402

RED = "\\textcolor{pyscfred}{%s}"
MOL = ["SiH4", "SiO", "S2", "C3H4", "C2H2O2", "C4H8"]
LAB = {"SiH4": "SiH$_4$", "SiO": "SiO", "S2": "S$_2$", "C3H4": "C$_3$H$_4$", "C2H2O2": "C$_2$H$_2$O$_2$",
       "C4H8": "C$_4$H$_8$"}

ae = pd.read_csv(os.path.join(HERE, "ae6_results.csv")).set_index("molecule")
bh = pd.read_csv(os.path.join(HERE, "bh6_results.csv"))
pbe = pd.read_csv(os.path.join(HERE, "pbe_nonscf_ae6", "ae6_pbe_nonscf.csv")).set_index("molecule")
ibm = pd.read_csv(os.path.join(HERE, "ibm_mae.csv"))
ref = ae.loc[MOL, "ref"]


def ibm_of(fn):
    s = ibm[ibm.functional == fn].set_index("widths")
    return s.loc["L >= 1", "mae_x"], s.loc["L < 1", "mae_x"]


def bh_mae(col):
    return float(np.mean(np.abs(bh[col] - bh.ref)))


data = load()
GATES = [("W1: $t{=}10^{-1}$", 1e-1, 0, 0), ("W2: $t{=}10^{-3}$, $p_0{=}4$", 1e-3, 4.0, 6),
         ("W3: $t{=}10^{-3}$, $p_0{=}5$", 1e-3, 5.0, 12), ("$W^{\\rm new}$, SS: $p_0{=}6$", 1e-5, 6.0, 12)]
gate_res = {}
for lab, t, p0, k in GATES:
    r = evaluate_gate(lambda a, p, t=t, p0=p0, k=k: g(a * p / t, 2) * (g(p / p0, k) if p0 else 1.0), data)
    gate_res[lab] = r

# columns: (header, density label, errors per molecule, BH6 MAE or None, IBM (ge1, lt1), functional changed?)
r2i, sci = ibm_of("r2scan"), ibm_of("scalar")
cols = [
    ("r$^2$SCAN", "r$^2$SCAN", ae.loc[MOL, "r2scan"] - ref, bh_mae("r2scan"), r2i, "no (baseline)"),
    ("SS", "r$^2$SCAN", ae.loc[MOL, "scalarA"] - ref, bh_mae("scalarA"), sci, "no"),
    ("TD", "r$^2$SCAN", ae.loc[MOL, "scalarB"] - ref, bh_mae("scalarB"), sci, "no (other spin route)"),
    ("PBE", "PBE", pbe.loc[MOL, "pbe"] - ref, None, None, "--"),
    ("r$^2$SCAN", "PBE", pbe.loc[MOL, "r2scan"] - ref, None, r2i, "no (baseline)"),
    ("SS", "PBE", pbe.loc[MOL, "SS"] - ref, None, sci, "no"),
]
for lab, *_ in GATES:
    r = gate_res[lab]
    cols.append((lab, "r$^2$SCAN", pd.Series(r[6], index=MOL), r[1], (r[2], r[3]), "yes (weight $W$)"))

# proposed W_new with the TD spin route (from make_table_orig_vs_new.py output)
_on = pd.read_csv(os.path.join(HERE, "orig_vs_new_ae6.csv")).set_index("molecule")
_ob = pd.read_csv(os.path.join(HERE, "orig_vs_new_bh6.csv"))
cols.append(("$W^{\\rm new}$, TD: $p_0{=}6$", "r$^2$SCAN", _on.loc[MOL, "new_TD"] - ref,
             float(np.mean(np.abs(_ob.new_TD - _ob.ref))), (cols[-1][4][0], cols[-1][4][1]), "yes (weight $W$)"))

# ---- per-molecule table, columns grouped by density
GROUPS = [("r$^2$SCAN density", [0, 1, 2, 6, 7, 8], ["r$^2$SCAN", "SS", "TD", "W1", "W2", "W3"]),
          ("PBE orbitals", [3, 4, 5], ["PBE", "r$^2$SCAN", "SS"])]
order = [i for _, idx, _ in GROUPS for i in idx]
heads = [h for _, _, hs in GROUPS for h in hs]
L = ["\\begin{tabular}{@{}l r " + " ".join("r" * len(idx) for _, idx, _ in GROUPS) + "@{}}", "\\toprule"]
row, rule, c0 = " & ", [], 3
for name, idx, _ in GROUPS:
    row += f" & \\multicolumn{{{len(idx)}}}{{c}}{{{name}}}"
    rule.append(f"\\cmidrule(lr){{{c0}-{c0 + len(idx) - 1}}}")
    c0 += len(idx)
L += [row + " \\\\", "".join(rule), "Molecule & $E^{\\rm ref}$ & " + " & ".join(heads) + " \\\\", "\\midrule"]
for m in MOL:
    L.append(" & ".join([LAB[m], f"{ref[m]:.2f}"] + [RED % f"{cols[i][2][m]:+.1f}" for i in order]) + " \\\\")
L.append("\\midrule")
L.append("ME & & " + " & ".join(RED % f"{cols[i][2].mean():+.1f}" for i in order) + " \\\\")
L.append("MAE & & " + " & ".join(RED % f"{cols[i][2].abs().mean():.1f}" for i in order) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_all_ae6.tex"), "w").write("\n".join(L) + "\n")

# ---- summary table
S = ["\\begin{tabular}{@{}llrrrrl@{}}", "\\toprule",
     "Approach & Density & AE6 MAE & BH6 MAE & \\multicolumn{2}{c}{IBM MAE $E_x$ (Ha)} & Functional changed \\\\",
     "\\cmidrule(lr){5-6}", " & & (kcal/mol) & (kcal/mol) & $L\\geq1$ & $L<1$ & \\\\", "\\midrule"]
for name, dens, err, bmae, ib, changed in cols:
    S.append(" & ".join([name, dens, RED % f"{err.abs().mean():.2f}",
                         "--" if bmae is None else RED % f"{bmae:.2f}",
                         "--" if ib is None else RED % f"{ib[0]:.4f}", "--" if ib is None else RED % f"{ib[1]:.4f}",
                         changed]) + " \\\\")
S += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_all_summary.tex"), "w").write("\n".join(S) + "\n")
for c in cols:
    print(f"{c[0]:32s} {c[1]:10s} AE6 MAE {c[2].abs().mean():9.2f}  BH6 {c[3] if c[3] is None else round(c[3],2)}"
          f"  IBM {c[4]}")
