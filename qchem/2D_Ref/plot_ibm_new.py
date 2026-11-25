#!/usr/bin/env python3
"""Figure 1 with W_new added (from ibm_benchmark_new.csv); writes ibm_benchmark_new.pdf."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import reproduce_ibm_benchmark as ibmb
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "ibm_benchmark_new.csv"))
istls, _ = ibmb.reference_curves()
Lf = np.linspace(0.0, 10.0, 101)
fock_f = np.array([ibmb.exact_fock_exchange(max(L, 1e-8)) for L in Lf])
ist_f = istls(Lf)
fig, axs = plt.subplots(3, 1, figsize=(3.6, 6.4), sharex=True, gridspec_kw=dict(hspace=0.07))
for ax, (k, ref, ylab, tag) in zip(axs, (("xc", ist_f, r"$\epsilon_{xc}$ (Ha)", "(a) XC"),
                                         ("x", fock_f, r"$\epsilon_{x}$ (Ha)", "(b) exchange"),
                                         ("c", ist_f - fock_f, r"$\epsilon_{c}$ (Ha)", "(c) correlation"))):
    ax.plot(Lf, ref, color="k", lw=2.0, label="reference")
    for key, col, mk, lab in (("r2", "#c47a00", "^", r"r$^2$SCAN"), ("orig", "#0a8f6a", "D", "scalar, original $W$"),
                              ("new", "#c0392b", "o", r"scalar, $W^{\rm new}$")):
        yv = df[f"{key}_x"] + df[f"{key}_c"] if k == "xc" else df[f"{key}_{k}"]
        ax.plot(df.L, yv, color=col, marker=mk, ms=3.6, mfc="white", lw=1.2, label=lab)
    ax.axvline(1.0, color="#7a7a7a", ls=":", lw=0.9)
    ax.set_ylabel(ylab)
    ax.text(0.97, 0.06 if k != "c" else 0.90, tag, transform=ax.transAxes, ha="right")
axs[0].set_ylim(-0.36, -0.035); axs[1].set_ylim(-0.31, -0.025); axs[2].set_ylim(-0.075, 0.0)
axs[1].legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.15))
axs[2].set_xlabel("Well width $L$ (bohr)")
fig.savefig(os.path.join(HERE, "ibm_benchmark_new.pdf"), bbox_inches="tight")
