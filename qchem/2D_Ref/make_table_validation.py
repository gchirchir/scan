#!/usr/bin/env python3
"""tab_validation.tex and tab_sensitivity.tex for Sec. 4.4 (proposed W_new = W_orig * p^k/(p^k+p0^k), p0 = 6, k = 12,
Jana constant kept; both spin routes).  All data dyall-v4z, grid level 9.
Sources: w_gate_variant/validate_new.csv (V1, V2, V5), pbe_nonscf_ae6/ae6_pbe_new_dyall-v4z_g9.csv (V3, newJ columns),
w_gate_variant/atomic_variant_v2.csv (V6), w_gate_variant/validate_new_sensitivity.csv (V7)."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "w_gate_variant"))
from wnew_params import ATOMIC_KEY  # noqa: E402

RED = "\\textcolor{pyscfred}{%s}"
v = pd.read_csv(os.path.join(HERE, "w_gate_variant", "validate_new.csv"))
pbe = pd.read_csv(os.path.join(HERE, "pbe_nonscf_ae6", "ae6_pbe_new_dyall-v4z_g9.csv"))
ibm = pd.read_csv(os.path.join(HERE, "ibm_mae.csv"))
av = pd.read_csv(os.path.join(HERE, "w_gate_variant", "atomic_variant_v2.csv")).set_index("atom")
FN = ["r2SCAN", "original", "original TD", "new G3", "new TD"]


def get(test, fn, col):
    sel = v[(v.test == test) & (v.functional == fn)]
    return float("nan") if sel.empty else float(sel[col].iloc[0])


def mae(col):
    return float((pbe[col] - pbe.ref).abs().mean())


r2_ibm = [float(ibm[(ibm.functional == "r2scan") & (ibm.widths == w)].mae_x.iloc[0]) for w in ("L >= 1", "L < 1")]
rows = []
# V1 (closed-formula weight, same densities) muted 2026-11-24 at user request; V2 kept
for test, lab, dens in (("V2 dyall-v4z/grid 9 (independent SCF rerun)", "V2", "r$^2$SCAN SCF, separate run"),):
    for q, col in (("AE6 MAE", "ae6_mae"), ("BH6 MAE", "bh6_mae")):
        rows.append((lab, dens, q, *[("--" if get(test, f, col) != get(test, f, col) else f"{get(test, f, col):.2f}") for f in FN]))
rows.append(("V3", "PBE orbitals, non-SCF", "AE6 MAE", f"{mae('r2scan'):.2f}", f"{mae('orig_SS'):.2f}",
             f"{mae('orig_TD'):.2f}", f"{mae('newJ_SS'):.2f}", f"{mae('newJ_TD'):.2f}"))
_pb = pd.read_csv(os.path.join(HERE, "pbe_nonscf_ae6", "bh6_pbe_new_dyall-v4z_g9.csv"))
rows.append(("V3", "PBE orbitals, non-SCF", "BH6 MAE", *[f"{(_pb[c] - _pb.ref).abs().mean():.2f}" for c in
                                                         ("r2scan", "orig_SS", "orig_TD", "newJ_SS", "newJ_TD")]))
for i, w in enumerate(("L\\geq1", "L<1")):
    k = "ibm_x_ge1" if i == 0 else "ibm_x_lt1"
    o, n = get("V5 IBM direct", "original", k), get("V5 IBM direct", "new G3", k)
    rows.append(("V5", "IBM well, $r_s^{2D}=4$", f"MAE $E_x$, ${w}$ (Ha)", f"{r2_ibm[i]:.4f}", f"{o:.4f}", f"{o:.4f}",
                 f"{n:.4f}", f"{n:.4f}"))
mg = -12.23 + float(av.loc["Mg", f"{ATOMIC_KEY}|dEx"])
rows.append(("V6", "Mg, HF density", "$E_x-E_x^{\\rm HF}$ (mHa)", "-12.23", "+53.91", "+53.91", f"{mg:+.2f}",
             f"{mg:+.2f}"))
L = ["\\begin{tabular}{@{}lll rrrrr@{}}", "\\toprule",
     " & & & & \\multicolumn{2}{c}{$W^{\\rm orig}_{\\rm jana\\_ref}$} & \\multicolumn{2}{c}{$W^{\\rm new}$} \\\\",
     "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
     "Test & Density & Quantity & r$^2$SCAN & SS & TD & SS & TD \\\\", "\\midrule"]
prev = None
for lab, dens, q, *vals in rows:
    if prev is not None and lab != prev:
        L.append("\\addlinespace")
    L.append(" & ".join([lab if lab != prev else "", dens if lab != prev else "", q] + [RED % x for x in vals]) + " \\\\")
    prev = lab
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_validation.tex"), "w").write("\n".join(L) + "\n")

s = pd.read_csv(os.path.join(HERE, "w_gate_variant", "validate_new_sensitivity.csv"))
T = ["\\begin{tabular}{@{}cc rrr rr@{}}", "\\toprule",
     " & & \\multicolumn{2}{c}{AE6 MAE (kcal/mol)} & BH6 MAE & \\multicolumn{2}{c}{IBM MAE $E_x$ (Ha)} \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}",
     "$k$ & $p_0$ & SS & TD & SS (kcal/mol) & $L\\geq1$ & $L<1$ \\\\", "\\midrule"]
for k in (6, 12):
    for _, r in s[s.k == k].sort_values("p0").iterrows():
        T.append(" & ".join([str(k), f"{int(r.p0)}"] + [RED % f"{r.SS_AE6:.2f}", RED % f"{r.TD_AE6:.2f}",
                                                         RED % f"{r.BH6_SS:.2f}", RED % f"{r.ibm_x_ge1:.4f}",
                                                         RED % f"{r.ibm_x_lt1:.4f}"]) + " \\\\")
    if k == 6:
        T.append("\\midrule")
T += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_sensitivity.tex"), "w").write("\n".join(T) + "\n")
print("\n".join(L).replace("\\textcolor{pyscfred}", ""))
print("\n".join(T).replace("\\textcolor{pyscfred}", ""))
