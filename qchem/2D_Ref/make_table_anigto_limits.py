#!/usr/bin/env python3
"""tab_anigto_limits.tex: independent W = 0 / W = 1 limit checks on aniGTO sheets and wires
(validation/anigto_limits.py, WCSS; largest grid available, the smaller grid gives the grid-convergence check)."""
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "validation", "anigto_limits_m*.csv")),
               key=lambda f: int(re.search(r"_m(\d+)", f).group(1)))
df = pd.read_csv(files[-1])
M = int(re.search(r"_m(\d+)", files[-1]).group(1))
# lambda < 1 (approach to lambda = 0) from the same grid, and the analytic lambda = 0 limit (n -> 0: every energy -> 0)
small = os.path.join(HERE, "validation", f"anigto_limits_small_m{M}.csv")
if os.path.exists(small):
    df = pd.concat([df, pd.read_csv(small)], ignore_index=True)
    zero = df.drop_duplicates("system").copy()
    for c in zero.columns:
        if c not in ("system", "N"):
            zero[c] = 0.0
    zero["lam"] = 0.0
    zero["frac_active"] = np.nan; zero["W_mean"] = np.nan; zero["Wnew_mean"] = np.nan
    df = pd.concat([df, zero], ignore_index=True)
order = {s: i for i, s in enumerate(dict.fromkeys(df.system))}
df = df.assign(_o=df.system.map(order)).sort_values(["_o", "lam"]).drop(columns="_o").reset_index(drop=True)
dfx = df[df.lam > 0]                                    # computed rows (for the max |Delta| and the printed checks)
G = "\\textcolor{green!50!black}{%s}"


def d(x):
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def lam(x):
    if x == 0:
        return "0$^{\\dagger}$"
    return "1" if x == 1 else f"$10^{{{int(round(__import__('math').log10(x)))}}}$"


def fa(x):
    return "--" if not np.isfinite(x) else f"{x:.3f}"


L = ["\\begin{tabular}{@{}llr rrr rrr rrr rrr@{}}", "\\toprule",
     " & & & \\multicolumn{6}{c}{$W=0$ against Libxc r$^2$SCAN} & \\multicolumn{6}{c}{$W=1$ against pure aniGTO\\_Q2D($\\lambda$)} \\\\",
     "\\cmidrule(lr){4-9}\\cmidrule(lr){10-15}",
     "System & $\\lambda$ & $f_{\\rm act}$ & $E_x^{\\rm Libxc}$ & $E_x^{W=0}$ & $\\Delta E_x$ & $E_c^{\\rm Libxc}$ & $E_c^{W=0}$"
     " & $\\Delta E_c$ & $E_x^{\\rm aniGTO\\_Q2D}$ & $E_x^{W=1}$ & $\\Delta E_x$ & $E_c^{\\rm aniGTO\\_Q2D}$ & $E_c^{W=1}$ & $\\Delta E_c$ \\\\",
     "\\midrule"]
prev = None
for r in df.itertuples():
    geom, N = r.system.split()
    if prev is not None and r.system != prev:
        L.append("\\addlinespace")
    lab = f"aniGTO {geom}, ${N}$" if r.system != prev else ""
    prev = r.system
    L.append(" & ".join([lab, lam(r.lam), fa(r.frac_active),
                         f"{r.Ex_r2SCAN_libxc:.8f}", f"{r.Ex_W0:.8f}", G % d(r.dEx_W0),
                         f"{r.Ec_r2SCAN_libxc:.8f}", f"{r.Ec_W0:.8f}", G % d(r.dEc_W0),
                         f"{r.Ex_Q2D_pure:.8f}", f"{r.Ex_W1:.8f}", G % d(r.dEx_W1),
                         f"{r.Ec_Q2D_pure:.8f}", f"{r.Ec_W1:.8f}", G % d(r.dEc_W1)]) + " \\\\")
# common species He, Be, H, H2 (same density as in Table tab:limits; validation/common_species.py):
# aniGTO_Q2D(lambda=1) fields, radial grid for the atoms, PySCF grid for H2 (spin-polarized treatment for H)
cs_path = os.path.join(HERE, "validation", "common_species.csv")
if os.path.exists(cs_path):
    cs = pd.read_csv(cs_path)
    L.append("\\midrule")
    LBL = {"He": "He", "Be": "Be", "H": "H", "H2": "H$_2$"}
    for _, r in cs.iterrows():
        rt = "ani_rad_" if pd.notna(r.get("ani_rad_Ex_W0")) else "ani_pg_"
        g = lambda k: r[rt + k]
        L.append(" & ".join(["aniGTO " + LBL[r.species] + "$^{*}$", "1", f"{g('f_act'):.3f}",
                             f"{g('Ex_libxc'):.8f}", f"{g('Ex_W0'):.8f}", G % d(g('Ex_W0') - g('Ex_libxc')),
                             f"{round(g('Ec_libxc'), 8) + 0.0:.8f}", f"{round(g('Ec_W0'), 8) + 0.0:.8f}", G % d(g('Ec_W0') - g('Ec_libxc')),
                             f"{g('Ex_Q2D'):.8f}", f"{g('Ex_W1'):.8f}", G % d(g('Ex_W1') - g('Ex_Q2D')),
                             f"{round(g('Ec_Q2D'), 8) + 0.0:.8f}", f"{round(g('Ec_W1'), 8) + 0.0:.8f}", G % d(g('Ec_W1') - g('Ec_Q2D'))]) + " \\\\")
mx = {k: dfx[k].abs().max() for k in ("dEx_W0", "dEc_W0", "dEx_W1", "dEc_W1", "dEx_model", "dEc_model")}
L += ["\\midrule", "max $|\\Delta|$ & & & & & " + G % d(mx["dEx_W0"]) + " & & & " + G % d(mx["dEc_W0"])
      + " & & & " + G % d(mx["dEx_W1"]) + " & & & " + G % d(mx["dEc_W1"]) + " \\\\", "\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_anigto_limits.tex"), "w").write("\n".join(L) + "\n")

# numbers quoted in the text
print(f"grid {M}^3; max |d| " + " ".join(f"{k} {v:.1e}" for k, v in mx.items()))
print("max |N_grid - N| %.1e" % (dfx.N_grid - dfx.N).abs().max())
n2 = dfx[dfx.N == 2]
print("N=2: max |Ex_exact - closed| %.1e, max f_act %.3f" % ((n2.Ex_exact - n2.Ex_closed).abs().max(), n2.frac_active.max()))
print("N=4,6: min f_act (lam>=10) %.3f" % dfx[(dfx.N > 2) & (dfx.lam >= 10)].frac_active.min())
if len(files) > 1:
    d0 = pd.read_csv(files[-2])
    for k in ("Ex_r2SCAN_libxc", "Ec_r2SCAN_libxc", "Ex_Q2D_pure", "Ex_Worig"):
        big = pd.read_csv(files[-1])
        print(f"grid change {k}: max |d| {(big[k] - d0[k]).abs().max():.1e} Ha")

# ---------------------------------------------------------------- tab_anigto_ex.tex: exchange vs lambda (same run)
X = ["\\begin{tabular}{@{}llr rrrrrr rr@{}}", "\\toprule",
     "System & $\\lambda$ & $f_{\\rm act}$ & $E_x^{\\rm exact}$ & r$^2$SCAN & aniGTO\\_Q2D($\\lambda$) & aniGTO\\_Q2D($\\lambda$) & aniGTO\\_Q2D($\\lambda$)"
     " & $\\Delta^{\\rm orig}$ & $\\langle W^{\\rm orig}\\rangle$ & $\\langle W^{\\rm new}\\rangle$ \\\\",
     " & & & & ($W=0$) & $W=1$ & $W^{\\rm orig}_{\\rm jana\\_ref}$ & $W^{\\rm new}$ & & & \\\\", "\\midrule"]
prev = None
for r in df.itertuples():
    geom, N = r.system.split()
    if prev is not None and r.system != prev:
        X.append("\\addlinespace")
    lab = f"aniGTO {geom}, ${N}$" if r.system != prev else ""
    prev = r.system
    X.append(" & ".join([lab, lam(r.lam), fa(r.frac_active), f"{r.Ex_exact:.6f}", f"{r.Ex_r2SCAN_libxc:.4f}",
                         f"{r.Ex_W1:.4f}", f"{r.Ex_Worig:.4f}", f"{r.Ex_Wnew:.4f}", f"{r.Ex_Worig - r.Ex_exact:+.4f}",
                         fa(r.W_mean), fa(r.Wnew_mean)]) + " \\\\")
X += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tab_anigto_ex.tex"), "w").write("\n".join(X) + "\n")
