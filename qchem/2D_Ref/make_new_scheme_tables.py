#!/usr/bin/env python3
"""Every result table of main.tex recomputed with the selected new weight (G3: t = 1e-3, p0 = 5, k = 12; standard spin
treatment: E_x spin-scaled, E_c not), side by side with the original functional of jana_ref.
  tab_new_atomic.tex   Tables 2-3 (atomic shifts on HF densities) + exchange error vs exact HF exchange
  tab_new_ibm.tex      Table 4 (IBM MAEs) + figure ibm_benchmark_new.pdf (Figure 1 with the new weight added)
  tab_new_dxc.tex      Table 6 (scalar-minus-r2SCAN shifts of the 21 AE6/BH6 species, SS route)
  tab_new_pbe.tex      Table 12 (AE6 on PBE orbitals, non-SCF)
(Tables 7-8 with the new weight: tab_orig_new_ae6.tex / tab_orig_new_bh6.tex, already in main.tex.)
Table 1 (HF densities) and Table 5's r2SCAN columns do not depend on the weight."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "w_gate_variant"))

import numpy as np                                                              # noqa: E402
import pandas as pd                                                             # noqa: E402
from verify_variant import ingredients, weight, dexc                            # noqa: E402
from q2d_scalar import pyscf_baseline                                           # noqa: E402
from reproduce_atomic import PDF_TABLE                                          # noqa: E402
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES                              # noqa: E402
import reproduce_ibm_benchmark as ibmb                                          # noqa: E402

RED = "\\textcolor{pyscfred}{%s}"
from wnew_params import NEW as G3, ORIG, ATOMIC_KEY        # noqa: E402  (G3 = proposed W_new here)

# ---------------- atomic (Tables 2-3)
av = pd.read_csv(os.path.join(HERE, "w_gate_variant", "atomic_variant_v2.csv")).set_index("atom")
EXX_ERR = {  # E_x - E_x^HF (mHa) from w_gate_variant/atomic_exact_exchange.log: r2SCAN, original, new
    "He": (-4.81, -4.81, -4.81), "Ne": (-35.75, -32.09, -35.75), "Ar": (2.77, 3.48, 2.77),
    "Kr": (35.76, 36.26, 35.76), "Xe": (264.66, 265.33, 264.67), "Mg": (-12.23, 53.91, -4.78)}
L = ["\\begin{tabular}{@{}l rrr rrr rrr@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{$\\Delta E_x$ (mHa)} & \\multicolumn{3}{c}{$\\Delta E_c$ (mHa)}"
     " & \\multicolumn{3}{c}{$E_x-E_x^{\\rm HF}$ (mHa)} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "Atom & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & \\janaref{} & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & r$^2$SCAN & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ \\\\",
     "\\midrule", "H & +0.000 & " + RED % "+0.000" + " & " + RED % "+0.000" + " & +0.000 & " + RED % "+0.000"
     + " & " + RED % "+0.000" + " & -- & -- & -- \\\\"]
for at in ("He", "Ne", "Ar", "Kr", "Xe", "Mg"):
    p = PDF_TABLE[at]
    r = av.loc[at]
    EXX_ERR[at] = (EXX_ERR[at][0], EXX_ERR[at][1], EXX_ERR[at][0] + float(r[f"{ATOMIC_KEY}|dEx"]))
    L.append(" & ".join([at, f"{1e3 * (p[1] - p[0]):+.3f}", RED % f"{r['original|dEx']:+.3f}",
                         RED % f"{r[f'{ATOMIC_KEY}|dEx']:+.3f}", f"{1e3 * (p[3] - p[2]):+.3f}",
                         RED % f"{r['original|dEc']:+.3f}", RED % f"{r[f'{ATOMIC_KEY}|dEc']:+.3f}"]
                        + [RED % f"{x:+.2f}" for x in EXX_ERR[at]]) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_new_atomic.tex"), "w").write("\n".join(L) + "\n")

# ---------------- IBM (Table 4, Figure 1)
istls, _ = ibmb.reference_curves()
ns = 1 / (np.pi * ibmb.RS2D ** 2)
rows = []
for Lw in ibmb.WIDTHS:
    x, wq = np.polynomial.legendre.leggauss(ibmb.quad_points(Lw))
    y, wy = (x + 1) / 2, wq / 2
    f = 2 * np.sin(np.pi * y) ** 2
    fp = 2 * np.pi * np.sin(2 * np.pi * y)
    n = ns / Lw * f
    g2 = (ns / Lw ** 2 * fp) ** 2
    tau = g2 / (8 * n) + (np.pi * ns / 2) * n
    we = Lw * wy / ns
    base = pyscf_baseline(n, g2, tau)
    act, W0, qx, qc, a, p = ingredients(n, g2, tau)
    r = dict(L=Lw, fock=ibmb.exact_fock_exchange(Lw), istls=float(istls(Lw)),
             r2_x=float(we @ base.exchange), r2_c=float(we @ base.correlation))
    for lab, var in (("orig", ORIG), ("new", G3)):
        W = weight(a, p, *var)
        r[f"{lab}_x"] = r["r2_x"] + float(we[act] @ (W * (qx - base.exchange[act])))
        r[f"{lab}_c"] = r["r2_c"] + float(we[act] @ (W * (qc - base.correlation[act])))
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "ibm_benchmark_new.csv"), index=False, float_format="%.8f")
T = ["\\begin{tabular}{@{}ll rrr@{}}", "\\toprule",
     "Widths & functional & exchange & correlation & XC \\\\", "\\midrule"]
for lab, sel in (("$L\\geq1$ bohr (13)", df.L >= 1), ("$L<1$ bohr (6)", df.L < 1)):
    d = df[sel]
    for fn, key in (("r$^2$SCAN", "r2"), ("scalar, original", "orig"), ("scalar, new", "new")):
        mx = np.mean(np.abs(d[f"{key}_x"] - d.fock))
        mc = np.mean(np.abs(d[f"{key}_c"] - (d.istls - d.fock)))
        mxc = np.mean(np.abs(d[f"{key}_x"] + d[f"{key}_c"] - d.istls))
        T.append(f"{lab} & {fn} & " + " & ".join(RED % f"{v:.6f}" for v in (mx, mc, mxc)) + " \\\\")
    T.append("\\midrule" if lab.startswith("$L\\geq") else "")
T = [t for t in T if t] + ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_new_ibm.tex"), "w").write("\n".join(T) + "\n")

import matplotlib                                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                 # noqa: E402
Lf = np.linspace(0.0, 10.0, 201)
fock_f = np.array([ibmb.exact_fock_exchange(max(Lv, 1e-8)) for Lv in Lf])
ist_f = istls(Lf)
fig, axs = plt.subplots(3, 1, figsize=(3.6, 6.4), sharex=True, gridspec_kw=dict(hspace=0.07))
for ax, (k, ref, ylab, tag) in zip(axs, (("xc", ist_f, r"$\epsilon_{xc}$ (Ha)", "(a) XC"),
                                         ("x", fock_f, r"$\epsilon_{x}$ (Ha)", "(b) exchange"),
                                         ("c", ist_f - fock_f, r"$\epsilon_{c}$ (Ha)", "(c) correlation"))):
    ax.plot(Lf, ref, color="k", lw=2.0, label="reference")
    for key, col, mk, lab in (("r2", "#c47a00", "^", r"r$^2$SCAN"), ("orig", "#0a8f6a", "D", "scalar, original"),
                              ("new", "#c0392b", "o", r"scalar, $W^{\rm new}$")):
        yv = df[f"{key}_x"] + df[f"{key}_c"] if k == "xc" else df[f"{key}_{k}"]
        ax.plot(df.L, yv, color=col, marker=mk, ms=3.6, mfc="white", lw=1.2, label=lab)
    ax.axvline(1.0, color="#7a7a7a", ls=":", lw=0.9)
    ax.set_ylabel(ylab)
    ax.text(0.97, 0.06 if k != "c" else 0.90, tag, transform=ax.transAxes, ha="right")
axs[0].set_ylim(-0.36, -0.035)
axs[1].set_ylim(-0.31, -0.025)
axs[2].set_ylim(-0.075, 0.0)
axs[1].legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.15))
axs[2].set_xlabel("Well width $L$ (bohr)")
fig.savefig(os.path.join(HERE, "ibm_benchmark_new.pdf"), bbox_inches="tight")

# ---------------- AE6/BH6 species shifts (Table 6), SS route
LABEL = {"SiH4": "SiH$_4$", "SiO": "SiO", "S2": "S$_2$", "C3H4": "C$_3$H$_4$", "C2H2O2": "C$_2$H$_2$O$_2$",
         "C4H8": "C$_4$H$_8$", "OH": "OH", "CH4": "CH$_4$", "CH3": "CH$_3$", "H2O": "H$_2$O", "H2": "H$_2$",
         "H2S": "H$_2$S", "HS": "SH", "TS_OH_CH4": "TS, OH+CH$_4$", "TS_H_OH": "TS, H+OH",
         "TS_H_H2S": "TS, H+H$_2$S"}
GROUP = {"H": "atoms", "SiH4": "AE6 molecules", "OH": "BH6 species"}
D = ["\\begin{tabular}{@{}lc rr rr@{}}", "\\toprule",
     " & & \\multicolumn{2}{c}{$\\Delta E_x^{\\rm SS}$ (mHa)} & \\multicolumn{2}{c}{$\\Delta E_c$ (mHa)} \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", "Species & $2S$ & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ \\\\"]
spc = []
for s in list(ATOM_SPIN) + list(MOLECULES):
    d = np.load(os.path.join(HERE, "densities_dyall", s + ".npz"))
    o = dexc(d["rho"], d["w"], ORIG, "SS", [])
    nw = dexc(d["rho"], d["w"], G3, "SS", [])
    spin = ATOM_SPIN.get(s, MOLECULES.get(s, (None, 0))[1])
    spc.append(dict(species=s, dEx_orig=1e3 * o[0], dEx_new=1e3 * nw[0], dEc_orig=1e3 * o[1], dEc_new=1e3 * nw[1]))
    if s in GROUP:
        D += ["\\midrule", f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{GROUP[s]}}}}} \\\\"]
    D.append(" & ".join([LABEL.get(s, s), str(spin)] + [RED % f"{v:+.3f}" for v in
                                                         (1e3 * o[0], 1e3 * nw[0], 1e3 * o[1], 1e3 * nw[1])]) + " \\\\")
D += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_new_dxc.tex"), "w").write("\n".join(D) + "\n")
pd.DataFrame(spc).to_csv(os.path.join(HERE, "new_scheme_species_shifts.csv"), index=False, float_format="%.4f")

# PBE orbitals: see make_table_new_pbe.py (needs the grid-9 PBE run with the proposed W_new)
for f in ("tab_new_atomic.tex", "tab_new_ibm.tex", "tab_new_dxc.tex"):
    print("=====", f)
    print(open(os.path.join(HERE, f)).read().replace("\\textcolor{pyscfred}", ""))
