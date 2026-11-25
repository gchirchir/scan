#!/usr/bin/env python3
"""Write the LaTeX tables of main.tex from the PySCF results (atomic_results.csv, ibm_mae.csv).
PySCF numbers are printed in red, the reference values of jana_ref (scalar_functional_form.pdf) in black.
Run:  python3 make_tables.py   (after reproduce_atomic.py and reproduce_ibm_benchmark.py)"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RED = "\\textcolor{pyscfred}{%s}"


def atomic():
    rows = list(csv.DictReader(open(os.path.join(HERE, "atomic_results.csv"))))
    L = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{$E_x^{\\mathrm{r^2SCAN}}$} & \\multicolumn{2}{c}{$E_x^{\\mathrm{scalar}}$}"
         " & \\multicolumn{2}{c}{$E_c^{\\mathrm{r^2SCAN}}$} & \\multicolumn{2}{c}{$E_c^{\\mathrm{scalar}}$} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
         "Atom & jana\\_ref & PySCF & jana\\_ref & PySCF & jana\\_ref & PySCF & jana\\_ref & PySCF \\\\", "\\midrule"]
    for r in rows:
        c = [r["atom"]]
        for k in ("r2scan_exchange", "scalar_exchange", "r2scan_correlation", "scalar_correlation"):
            c += [f"{float(r['pdf_' + k + '_ha']):.6f}", RED % f"{float(r[k + '_ha']):.6f}"]
        L.append(" & ".join(c) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(HERE, "tab_atomic.tex"), "w").write("\n".join(L) + "\n")

    # effect of the correction (scalar - r2SCAN) and density check
    L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{$\\Delta E_x=E_x^{\\mathrm{scalar}}-E_x^{\\mathrm{r^2SCAN}}$}"
         " & \\multicolumn{2}{c}{$\\Delta E_c=E_c^{\\mathrm{scalar}}-E_c^{\\mathrm{r^2SCAN}}$} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Atom & jana\\_ref & PySCF & jana\\_ref & PySCF \\\\", "\\midrule"]
    for r in rows:
        f = lambda k: float(r[k])
        c = [r["atom"],
             f"{1000 * (f('pdf_scalar_exchange_ha') - f('pdf_r2scan_exchange_ha')):+.3f}",
             RED % f"{1000 * (f('scalar_exchange_ha') - f('r2scan_exchange_ha')):+.3f}",
             f"{1000 * (f('pdf_scalar_correlation_ha') - f('pdf_r2scan_correlation_ha')):+.3f}",
             RED % f"{1000 * (f('scalar_correlation_ha') - f('r2scan_correlation_ha')):+.3f}"]
        L.append(" & ".join(c) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(HERE, "tab_atomic_shift.tex"), "w").write("\n".join(L) + "\n")

    # HF densities: numerical HF limit, PySCF HF energy, then their difference
    L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         "Atom & $E_{\\rm HF}^{\\rm limit}$ (Ha) & $E_{\\rm HF}^{\\rm PySCF}$ (Ha) & "
         "$E_{\\rm HF}^{\\rm PySCF}-E_{\\rm HF}^{\\rm limit}$ (mHa) & $N_{\\rm grid}$ \\\\", "\\midrule"]
    for r in rows:
        f = lambda k: float(r[k])
        L.append(" & ".join([r["atom"], f"{f('hf_limit_ha'):.6f}", RED % f"{f('hf_energy_ha'):.6f}",
                             RED % f"{f('hf_minus_limit_mha'):+.3f}", RED % f"{f('electrons_on_grid'):.6f}"]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(HERE, "tab_hf.tex"), "w").write("\n".join(L) + "\n")


def ibm():
    rows = list(csv.DictReader(open(os.path.join(HERE, "ibm_mae.csv"))))
    L = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
         " & & \\multicolumn{2}{c}{exchange} & \\multicolumn{2}{c}{correlation} & \\multicolumn{2}{c}{XC} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
         "Widths & functional & jana\\_ref & PySCF & jana\\_ref & PySCF & jana\\_ref & PySCF \\\\", "\\midrule"]
    for r in rows:
        lab = r["widths"].replace(">=", "\\geq").replace("<", "<")
        c = [f"${lab}$ bohr ({r['n']})", "scalar Q2D" if r["functional"] == "scalar" else "r$^2$SCAN"]
        for k in ("x", "c", "xc"):
            p = r["pdf_" + k]
            c += ["--" if p in ("", "nan") else f"{float(p):.6f}", RED % f"{float(r['mae_' + k]):.6f}"]
        L.append(" & ".join(c) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(HERE, "tab_ibm_mae.tex"), "w").write("\n".join(L) + "\n")


def ae6_bh6():
    """AE6 and BH6 tables (kcal/mol): reference, r2SCAN, scalar Q2D (spin routes A and B), errors; ME and MAE rows."""
    p_ae, p_bh = os.path.join(HERE, "ae6_results.csv"), os.path.join(HERE, "bh6_results.csv")
    if not (os.path.exists(p_ae) and os.path.exists(p_bh)):
        return
    F = ("r2scan", "scalarA", "scalarB")
    for path, out, key, head in ((p_ae, "tab_ae6.tex", "label", "Molecule"), (p_bh, "tab_bh6.tex", "reaction", "Reaction")):
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            if os.path.exists(os.path.join(HERE, out)):
                os.remove(os.path.join(HERE, out))
            continue
        bh = key == "reaction"
        nlead = 1  # BH6: each direction written out as its own reaction (as in Table 4 of Haunschild & Klopper)
        L = ["\\begin{tabular}{@{}l rrrr rrr@{}}", "\\toprule",
             " & " * nlead + " & \\multicolumn{3}{c}{%s} & \\multicolumn{3}{c}{error, $E-E^{\\rm ref}$} \\\\" % ("barrier height" if bh else "atomization energy"),
             "\\cmidrule(lr){%d-%d}\\cmidrule(lr){%d-%d}" % (nlead + 2, nlead + 4, nlead + 5, nlead + 7),
             head + " & $E^{\\rm ref}$" + " & $E^{{\\rm r}^2{\\rm SCAN}}$ & $E^{\\rm SS}$ & $E^{\\rm TD}$" * 2 + " \\\\",
             "\\midrule"]
        err = {k: [] for k in F}
        for r in rows:
            v = {k: float(r[k]) for k in F + ("ref",)}
            c = [rxn_label(r) if bh else r[key]] + [f"{v['ref']:.2f}"]
            c += [RED % f"{v[k]:.2f}" for k in F]
            for k in F:
                err[k].append(v[k] - v["ref"])
                c.append(RED % f"{v[k] - v['ref']:+.2f}")
            L.append(" & ".join(c) + " \\\\")
        L.append("\\midrule")
        for lab, fn, fmt in (("ME", lambda e: sum(e) / len(e), "+.2f"),
                             ("MAE", lambda e: sum(abs(x) for x in e) / len(e), ".2f")):
            L.append(lab + " & " * (nlead + 4) + " & ".join(RED % format(fn(err[k]), fmt) for k in F) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(HERE, out), "w") as fh:
            fh.write("\n".join(L) + "\n")



def rxn_label(r):
    """Reaction as written for the given direction; reverse = products -> reactants (Haunschild & Klopper, Table 4)."""
    lab = r["reaction"].replace("+ HS", "+ SH")
    if r["direction"] == "reverse":
        lhs, rhs = lab.split(" $\\to$ ")
        lab = rhs + " $\\to$ " + lhs
    return lab

LABEL = {"SiH4": "SiH$_4$", "SiO": "SiO", "S2": "S$_2$", "C3H4": "C$_3$H$_4$", "C2H2O2": "C$_2$H$_2$O$_2$",
         "C4H8": "C$_4$H$_8$", "OH": "OH", "CH4": "CH$_4$", "CH3": "CH$_3$", "H2O": "H$_2$O", "H2": "H$_2$",
         "H2S": "H$_2$S", "HS": "SH", "TS_OH_CH4": "\\textcolor{red}{transition state (TS)}, OH+CH$_4$",
         "TS_H_OH": "\\textcolor{red}{transition state (TS)}, H+OH",
         "TS_H_H2S": "\\textcolor{red}{transition state (TS)}, H+H$_2$S"}
GROUP_START = {"SiH4": "AE6 molecules", "OH": "BH6 species", "H": "atoms"}


def ae6_bh6_xc():
    """Tables of Ex and Ec (Ha) and of the scalar-minus-r2SCAN differences (mHa) for all AE6/BH6 species."""
    path = os.path.join(HERE, "ae6_bh6_xc.csv")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    A = ["\\begin{tabular}{@{}lc rrr rr@{}}", "\\toprule",
         "Species & $2S$ & $E_x^{{\\rm r}^2{\\rm SCAN}}$ & $E_x^{\\rm SS}$ & $E_x^{\\rm TD}$ & $E_c^{{\\rm r}^2{\\rm SCAN}}$ & $E_c^{\\rm SS}=E_c^{\\rm TD}$ \\\\"]
    D = ["\\begin{tabular}{@{}lc rrr@{}}", "\\toprule",
         "Species & $2S$ & $\\Delta E_x^{\\rm SS}$ & $\\Delta E_x^{\\rm TD}$ & $\\Delta E_c$ \\\\"]
    for r in rows:
        nm = r["species"]
        if nm in GROUP_START:
            for T, ncol in ((A, 7), (D, 5)):
                T.append("\\midrule")
                T.append("\\multicolumn{%d}{@{}l}{\\textit{%s}} \\\\" % (ncol, GROUP_START[nm]))
        lab = LABEL.get(nm, nm)
        A.append(" & ".join([lab, r["spin2S"]] + [RED % f"{round(float(r[k]), 6) + 0.0:.6f}" for k in
                 ("ex_r2scan", "ex_scalarA", "ex_scalarB", "ec_r2scan", "ec_scalar")]) + " \\\\")
        D.append(" & ".join([lab, r["spin2S"]] + [RED % f"{float(r[k]):+.3f}" for k in
                 ("dEx_A_mHa", "dEx_B_mHa", "dEc_mHa")]) + " \\\\")
    for T, out in ((A, "tab_ae6bh6_xc.tex"), (D, "tab_ae6bh6_dxc.tex")):
        T += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(HERE, out), "w").write("\n".join(T) + "\n")


if __name__ == "__main__":
    atomic()
    ibm()
    ae6_bh6()
    ae6_bh6_xc()
    print("wrote tab_hf.tex, tab_atomic.tex, tab_atomic_shift.tex, tab_ibm_mae.tex (+ tab_ae6.tex, tab_bh6.tex)")
