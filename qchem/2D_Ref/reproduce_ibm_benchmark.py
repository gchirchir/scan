#!/usr/bin/env python3
"""Infinite-barrier-model (IBM) benchmark of scalar_functional_form.pdf, Sec. 7: figure and MAE table.

Fixed-density single-subband well, r_s^2D = 4, width L.  Semilocal energies (scalar Q2D of q2d_scalar.py and its
r2SCAN baseline, PySCF Libxc) come from calculate() of reproduce_ibm.py on the analytic density
    n(z) = (n_s/L) 2 sin^2(pi z/L),   tau = tau_W + (pi n_s/2) n.
References, as in the PDF:
  exchange     exact Fock exchange of the same single-subband well, computed here (exact_fock_exchange);
  XC           digitized ISTLS of Chiodo et al., PRL 108, 126402 (2012), Fig. 1 (chiodo2012_fig1_digitized.csv);
  correlation  ISTLS XC minus exact Fock exchange.
MAEs over the PDF's 19 widths, split at L = 1 bohr (13 widths >= 1, 6 widths < 1).

Exact exchange of one occupied subband phi(z) of an unpolarized 2D gas (k_F = sqrt(2 pi n_s) = sqrt(2)/r_s):
    eps_x(L) = -k_F int_0^1 [1 - S_0(x)] F(2 k_F L x) dx,
    1 - S_0(x) = 1 - (2/pi)[arcsin x + x sqrt(1 - x^2)]        (free 2D gas, x = q/2k_F),
    F(q) = int int |phi(z)|^2 |phi(z')|^2 exp(-q|z - z'|) dz dz'  (subband form factor),
which gives the strict-2D value -4 sqrt(2)/(3 pi r_s) = -0.150053 Ha at L -> 0 (F = 1).

Run:  python3 reproduce_ibm_benchmark.py      (needs PySCF for the r2SCAN baseline; writes ibm_benchmark.csv,
      ibm_mae.csv, ibm_benchmark.{pdf,eps,png})
"""
import os

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from reproduce_ibm import calculate

HERE = os.path.dirname(os.path.abspath(__file__))
RS2D = 4.0
# the 19 widths of the PDF's Sec. 7 figure (markers read from the plot; the PDF does not list them): 6 below 1 bohr,
# 13 at or above; L = 0.002 and 0.01 are the two markers at Ex = -0.158 and -0.165 next to L = 0
WIDTHS = [0.002, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]


def quad_points(L):
    """Gauss-Legendre points across the well: the switching layer near the walls narrows as L -> 0 (tested
    2026-11-23: scalar Ex at L = 0.002 is -0.15278 / -0.15837 / -0.15818 Ha with 800 / 3000 / 12000 points;
    converged to < 1e-6 Ha for L >= 0.1 with 800)."""
    return 12000 if L < 0.05 else 3000
PDF_MAE = {"L >= 1": (0.026433, 0.008969, 0.017464), "L < 1": (0.035057, 0.005234, 0.037071)}


def form_factor(Q, npts=400):
    """F(qL) of the infinite well, |phi|^2 = f(y)/L, f(y) = 2 sin^2(pi y): F = 2 int_0^1 A(d) exp(-Q d) dd with the
    autocorrelation A(d) = int_0^{1-d} f(y) f(y+d) dy (Gauss-Legendre in d and y)."""
    xd, wd = np.polynomial.legendre.leggauss(npts)
    d, wd = (xd + 1) / 2, wd / 2
    xy, wy = np.polynomial.legendre.leggauss(npts)
    y = (xy[None, :] + 1) / 2 * (1 - d[:, None])
    A = (1 - d) * np.sum(wy / 2 * 4 * np.sin(np.pi * y) ** 2 * np.sin(np.pi * (y + d[:, None])) ** 2, axis=1)
    Q = np.atleast_1d(np.asarray(Q, float))
    return 2 * np.exp(-np.outer(Q, d)) @ (wd * A)


def exact_fock_exchange(L, rs=RS2D, npts=2000):
    """Exact (Fock) exchange per electron (Ha) of the single-subband IBM of width L (bohr)."""
    kf = np.sqrt(2) / rs
    x, w = np.polynomial.legendre.leggauss(npts)
    x, w = (x + 1) / 2, w / 2
    one_minus_s0 = 1 - (2 / np.pi) * (np.arcsin(x) + x * np.sqrt(1 - x * x))
    return float(-kf * np.dot(w * one_minus_s0, form_factor(2 * kf * L * x)))


def reference_curves():
    ref = pd.read_csv(os.path.join(HERE, "chiodo2012_fig1_digitized.csv"), comment="#")
    ist = ref[ref.curve == "ISTLS"].sort_values("L_bohr").drop_duplicates("L_bohr")
    exx = ref[ref.curve == "EXX"].sort_values("L_bohr").drop_duplicates("L_bohr")
    return (PchipInterpolator(ist.L_bohr.values, ist.eps_ha_per_e.values), exx)


def main():
    istls, exx = reference_curves()
    # self-checks of the exact exchange: strict-2D limit and the digitized EXX curve of Chiodo Fig. 1
    lim = -4 * np.sqrt(2) / (3 * np.pi * RS2D)
    assert abs(exact_fock_exchange(1e-8) - lim) < 1e-7, (exact_fock_exchange(1e-8), lim)
    dev = [exact_fock_exchange(L) - y for L, y in zip(exx.L_bohr, exx.eps_ha_per_e)]
    print(f"exact Fock vs digitized EXX of Chiodo Fig. 1 ({len(dev)} points): max |dev| = {np.max(np.abs(dev)):.2e} Ha,"
          f" mean |dev| = {np.mean(np.abs(dev)):.2e} Ha")

    ns = 1 / (np.pi * RS2D ** 2)
    rows = []
    for L in WIDTHS:
        r = calculate(L, ns, quad_points(L))
        fock = exact_fock_exchange(L)
        xc_ref = float(istls(L))
        rows.append(dict(L_bohr=L, exact_fock_x=fock, istls_xc=xc_ref, ref_c=xc_ref - fock,
                         scalar_x=r["scalar_exchange_ha_per_e"], scalar_c=r["scalar_correlation_ha_per_e"],
                         scalar_xc=r["scalar_xc_ha_per_e"], r2scan_x=r["r2scan_exchange_ha_per_e"],
                         r2scan_c=r["r2scan_correlation_ha_per_e"], r2scan_xc=r["r2scan_xc_ha_per_e"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "ibm_benchmark.csv"), index=False, float_format="%.8f")

    mae = []
    for lab, sel in (("L >= 1", df.L_bohr >= 1), ("L < 1", df.L_bohr < 1)):
        d = df[sel]
        for fn in ("scalar", "r2scan"):
            mae.append(dict(widths=lab, n=int(sel.sum()), functional=fn,
                            mae_x=float(np.mean(np.abs(d[f"{fn}_x"] - d.exact_fock_x))),
                            mae_c=float(np.mean(np.abs(d[f"{fn}_c"] - d.ref_c))),
                            mae_xc=float(np.mean(np.abs(d[f"{fn}_xc"] - d.istls_xc))),
                            pdf_x=PDF_MAE[lab][0] if fn == "scalar" else np.nan,
                            pdf_c=PDF_MAE[lab][1] if fn == "scalar" else np.nan,
                            pdf_xc=PDF_MAE[lab][2] if fn == "scalar" else np.nan))
    m = pd.DataFrame(mae)
    m.to_csv(os.path.join(HERE, "ibm_mae.csv"), index=False, float_format="%.6f")
    print(m.to_string(index=False))
    figure(df, istls)


# ------------------------------------------------------------- publication style (as codes/16_subrata_paper_figures.py)
def figure(df, istls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 10.5, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 7.5, "axes.linewidth": 1.0, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "lines.linewidth": 1.5,
        "legend.frameon": True, "legend.framealpha": 1.0, "legend.fancybox": False, "legend.edgecolor": "black",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42})
    Lf = np.linspace(0.0, 10.0, 201)
    fock_f = np.array([exact_fock_exchange(max(L, 1e-8)) for L in Lf])
    ist_f = istls(Lf)
    fig, axs = plt.subplots(3, 1, figsize=(3.37, 6.2), sharex=True, gridspec_kw=dict(hspace=0.07))
    pan = (("xc", ist_f, "ISTLS (digitized)", r"$\epsilon_{xc}$ (Ha)", (-0.36, -0.035), "(a) XC"),
           ("x", fock_f, "exact Fock exchange", r"$\epsilon_{x}$ (Ha)", (-0.31, -0.025), "(b) exchange"),
           ("c", ist_f - fock_f, "ISTLS $-$ Fock", r"$\epsilon_{c}$ (Ha)", (-0.075, 0.0), "(c) correlation"))
    for ax, (k, ref, rlab, ylab, ylim, tag) in zip(axs, pan):
        ax.plot(Lf, ref, color="k", lw=2.0, label=rlab)
        ax.plot(df.L_bohr, df[f"r2scan_{k}"], color="#c47a00", ls=(0, (4, 1.5)), marker="^", ms=4, mfc="white",
                mec="#c47a00", mew=0.9, label=r"r$^2$SCAN")
        ax.plot(df.L_bohr, df[f"scalar_{k}"], color="#0a8f6a", ls="-", marker="D", ms=3.8, mfc="white",
                mec="#0a8f6a", mew=0.9, label="scalar Q2D")
        ax.axvline(1.0, color="#7a7a7a", ls=":", lw=0.9)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.text(0.97, 0.06 if k != "c" else 0.94, tag, transform=ax.transAxes, ha="right",
                va="bottom" if k != "c" else "top", fontsize=10, fontweight="bold")
        ax.legend(loc="lower right" if k == "c" else "upper left", fontsize=7.0, ncol=1 if k == "c" else 3,
                  handlelength=1.8, columnspacing=0.7,
                  bbox_to_anchor=(0.80, 0.02) if k == "c" else None)
    axs[-1].set_xlim(0, 10)
    axs[-1].set_xlabel(r"Well width $L$ (bohr)")
    for ext in ("pdf", "eps", "png"):
        fig.savefig(os.path.join(HERE, f"ibm_benchmark.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
