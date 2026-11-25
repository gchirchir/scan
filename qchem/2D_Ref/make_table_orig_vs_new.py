#!/usr/bin/env python3
"""Clean comparison: original functional of jana_ref vs the new (gated) weight, against the reference.
r2SCAN densities, dyall-v4z, grid 9 (densities_dyall).  Weight in jana_ref notation (Eq. 4):
    original  W = S(alpha) u^2/(u^2 + 1e-10)  chi^6/(chi^6 + 6^6)
    new       W = S(alpha) u^2/(u^2 + t^2)    chi^6/(chi^6 + 6^6)  p^k/(p^k + p0^k)
Proposed (2026-11-24): W_new = W_orig * p^k/(p^k+p0^k), p0 = 6, k = 12, Jana constant kept (wnew_params.NEW);
new-SS / new-TD: the two spin routes.
Writes orig_vs_new_ae6.csv, orig_vs_new_bh6.csv, tab_orig_new_ae6.tex, tab_orig_new_bh6.tex, tab_orig_new_summary.tex.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "w_gate_variant"))

import numpy as np                                                                   # noqa: E402
import pandas as pd                                                                  # noqa: E402
from verify_variant import dexc                                                      # noqa: E402
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES, AE6_REF, AE6_LABEL, BH6, HA2KCAL, read_xyz   # noqa: E402

RED = "\\textcolor{pyscfred}{%s}"
from wnew_params import NEW, ORIG                                                   # noqa: E402
METHODS = [("r2scan", None, None), ("orig_SS", ORIG, "SS"), ("orig_TD", ORIG, "TD"),
           ("new_SS", NEW, "SS"), ("new_TD", NEW, "TD")]
HEAD = ["r$^2$SCAN", "SS", "TD", "SS", "TD"]

names = list(ATOM_SPIN) + list(MOLECULES)
dens = {s: np.load(os.path.join(HERE, "densities_dyall", s + ".npz")) for s in names}
E = {}
for key, var, route in METHODS:
    E[key] = {}
    for s in names:
        e = float(dens[s]["e_tot"])
        if var is not None:
            dx, dc = dexc(dens[s]["rho"], dens[s]["w"], var, route, [])
            e += dx + dc
        E[key][s] = e

ae = []
for m, ref in AE6_REF.items():
    at = [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
    ae.append(dict(molecule=m, label=AE6_LABEL[m], ref=ref,
                   **{k: (sum(E[k][a] for a in at) - E[k][m]) * HA2KCAL for k, _, _ in METHODS}))
ae = pd.DataFrame(ae)
bh = []
for lab, R, ts, P, ff, rr in BH6:
    for side, ref, rx in ((R, ff, lab), (P, rr, None)):
        if rx is None:
            lhs, rhs = lab.replace("+ HS", "+ SH").split(" $\\to$ ")
            rx = rhs + " $\\to$ " + lhs
        bh.append(dict(reaction=rx.replace("+ HS", "+ SH"), ref=ref,
                       **{k: (E[k][ts] - sum(E[k][x] for x in side)) * HA2KCAL for k, _, _ in METHODS}))
bh = pd.DataFrame(bh)
ae.to_csv(os.path.join(HERE, "orig_vs_new_ae6.csv"), index=False, float_format="%.4f")
bh.to_csv(os.path.join(HERE, "orig_vs_new_bh6.csv"), index=False, float_format="%.4f")
K = [k for k, _, _ in METHODS]


def table(df, key, head, path):
    L = ["\\begin{tabular}{@{}l r rrrrr rrrrr@{}}", "\\toprule",
         " & & \\multicolumn{5}{c}{value} & \\multicolumn{5}{c}{error, $E-E^{\\rm ref}$} \\\\",
         "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}",
         " & & & \\multicolumn{2}{c}{$W^{\\rm orig}_{\\rm jana\\_ref}$} & \\multicolumn{2}{c}{$W^{\\rm new}$} & & \\multicolumn{2}{c}{$W^{\\rm orig}_{\\rm jana\\_ref}$}"
         " & \\multicolumn{2}{c}{$W^{\\rm new}$} \\\\",
         "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}",
         f"{head} & $E^{{\\rm ref}}$ & " + " & ".join(HEAD * 2) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        L.append(" & ".join([r[key], f"{r.ref:.2f}"] + [RED % f"{r[k]:.2f}" for k in K]
                            + [RED % f"{r[k] - r.ref:+.2f}" for k in K]) + " \\\\")
    err = {k: df[k] - df.ref for k in K}
    L.append("\\midrule")
    L.append("ME & " + " & " * 6 + " & ".join(RED % f"{err[k].mean():+.2f}" for k in K) + " \\\\")
    L.append("MAE & " + " & " * 6 + " & ".join(RED % f"{err[k].abs().mean():.2f}" for k in K) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(HERE, path), "w").write("\n".join(L) + "\n")
    return {k: err[k].abs().mean() for k in K}


mae_ae = table(ae, "label", "Molecule", "tab_orig_new_ae6.tex")
mae_bh = table(bh, "reaction", "Reaction", "tab_orig_new_bh6.tex")

# summary: AE6, BH6, IBM exchange MAE, Mg atomic exchange error vs exact HF exchange
ibm = pd.read_csv(os.path.join(HERE, "ibm_mae.csv"))
ix = {w: float(ibm[(ibm.functional == "r2scan") & (ibm.widths == w)].mae_x.iloc[0]) for w in ("L >= 1", "L < 1")}
ox = {w: float(ibm[(ibm.functional == "scalar") & (ibm.widths == w)].mae_x.iloc[0]) for w in ("L >= 1", "L < 1")}
tu = pd.read_csv(os.path.join(HERE, "w_gate_variant", "tune_c3h4_glyoxal.csv"))


def tun(t, p0, k):
    r = tu[(abs(tu.t - round(t, 4)) < 1e-12) & (tu.p0 == p0) & (tu.k == k)].iloc[0]   # csv stores t to 4 decimals
    return r.ibm_x_ge1, r.ibm_x_lt1


nx = {"new_SS": tun(*NEW), "new_TD": tun(*NEW)}
IB = {"r2scan": (ix["L >= 1"], ix["L < 1"]), "orig_SS": (ox["L >= 1"], ox["L < 1"]),
      "orig_TD": (ox["L >= 1"], ox["L < 1"]), **nx}
# Mg: E_x - E_x(exact HF), mHa (w_gate_variant/atomic_exact_exchange.log; closed shell, SS = TD)
from wnew_params import ATOMIC_KEY                                                  # noqa: E402
_av = pd.read_csv(os.path.join(HERE, "w_gate_variant", "atomic_variant_v2.csv")).set_index("atom")
_mgn = -12.23 + float(_av.loc["Mg", f"{ATOMIC_KEY}|dEx"])          # r2SCAN error + W_new shift
MG = {"r2scan": -12.23, "orig_SS": 53.91, "orig_TD": 53.91, "new_SS": _mgn, "new_TD": _mgn}
S = ["\\begin{tabular}{@{}l rrrrr@{}}", "\\toprule",
     " & & \\multicolumn{2}{c}{$W^{\\rm orig}_{\\rm jana\\_ref}$} & \\multicolumn{2}{c}{$W^{\\rm new}$} \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
     " & " + " & ".join(HEAD) + " \\\\", "\\midrule",
     "AE6 MAE (kcal/mol) & " + " & ".join(RED % f"{mae_ae[k]:.2f}" for k in K) + " \\\\",
     "BH6 MAE (kcal/mol) & " + " & ".join(RED % f"{mae_bh[k]:.2f}" for k in K) + " \\\\",
     "IBM MAE $E_x$, $L\\geq1$ (Ha) & " + " & ".join(RED % f"{IB[k][0]:.4f}" for k in K) + " \\\\",
     "IBM MAE $E_x$, $L<1$ (Ha) & " + " & ".join(RED % f"{IB[k][1]:.4f}" for k in K) + " \\\\",
     "Mg: $E_x-E_x^{\\rm HF}$ (mHa) & " + " & ".join(RED % f"{MG[k]:+.2f}" for k in K) + " \\\\",
     "\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_orig_new_summary.tex"), "w").write("\n".join(S) + "\n")
print(ae.round(2).to_string(index=False))
print(bh.round(2).to_string(index=False))
print("AE6 MAE", {k: round(v, 2) for k, v in mae_ae.items()})
print("BH6 MAE", {k: round(v, 2) for k, v in mae_bh.items()})
