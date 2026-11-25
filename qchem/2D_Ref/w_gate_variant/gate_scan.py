#!/usr/bin/env python3
"""EXPERIMENTAL variant of the scalar Q2D weight (not the functional of jana_ref; original results are kept).

Diagnosis: the AE6 failure comes from single-orbital regions of atoms (1s cores, the Be-like 1s2 2s2 doubled beta
density of C), where alpha ~ 1e-3 and alpha*p ~ 1e-4..1e-2, and the gate g(alpha p / t) with t = 1e-5 is fully on.
In the quasi-2D wells (IBM, r_s^2D = 4) the correction acts at alpha*p ~ 0.07..15.

Variant: the only change is the threshold t of that gate,
    W_t = cutoff(alpha) * g2(alpha p / t) * g6(p / (6 sqrt(alpha + 1e-6))),   g_k(x) = x^k / (1 + x^k),
everything else (Q2D branches, r2SCAN baseline, mixing, spin treatment) unchanged.  t = 1e-5 is the original.
Spin treatment (standard): exchange by exact spin scaling, correlation without spin scaling (W, Q2D from total n).

Tests for each t:
  IBM      MAE of E_x, E_c, E_xc vs exact Fock / ISTLS over the 19 widths (as reproduce_ibm_benchmark.py)
  atoms    dE_x, dE_c (mHa) on the HF-limit-free r2SCAN densities of the AE6 atoms
  AE6/BH6  on the stored self-consistent r2SCAN densities (../densities_dyall, dyall-v4z, grid 9), as Table 7/8
Writes gate_scan_ibm.csv, gate_scan_species.csv, gate_scan_ae6.csv, gate_scan_bh6.csv.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402
from pyscf.dft import libxc                                            # noqa: E402

from q2d_scalar import _q2d, pyscf_baseline, CTF, K                    # noqa: E402
from reproduce_ae6_bh6 import (ATOM_SPIN, MOLECULES, AE6_REF, BH6, HA2KCAL, read_xyz)   # noqa: E402
import reproduce_ibm_benchmark as ibmb                                 # noqa: E402

T0 = 1e-5
THRESHOLDS = [float(x) for x in sys.argv[1:]] or [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SUFFIX = "" if not sys.argv[1:] else "_" + "_".join(sys.argv[1:])


def g2(x):
    return x * x / (1 + x * x)


def branches(n, gr2, tau, t):
    """Per-volume (e_x, e_c) of the unpolarized mixture with gate threshold t, and active mask."""
    base = pyscf_baseline(n, gr2, tau)
    ex, ec = base.exchange.copy(), base.correlation.copy()
    safe = np.where(n > 0, n, 1.0)
    pauli = tau - gr2 / (8 * safe)
    act = (n > 1e-60) & (gr2 > 0) & (pauli > 0) & (pauli < CTF * n ** (5 / 3))
    if act.any():
        w0, qx, qc = _q2d(n[act], gr2[act], tau[act])
        na = n[act]
        alpha = pauli[act] / (CTF * na ** (5 / 3))
        p = gr2[act] / (4 * K ** 2 * na ** (8 / 3))
        old = g2(alpha * p / T0)
        W = np.where(old > 0, w0.value * g2(alpha * p / t) / np.where(old > 0, old, 1), 0.0)
        ex[act] = (1 - W) * ex[act] + W * qx.value
        ec[act] = (1 - W) * ec[act] + W * qc.value
    return ex, ec, act


def weight_only(n, gr2, tau, t):
    """W_t on active points (zeros elsewhere), for correlation from the total density."""
    safe = np.where(n > 0, n, 1.0)
    pauli = tau - gr2 / (8 * safe)
    act = (n > 1e-60) & (gr2 > 0) & (pauli > 0) & (pauli < CTF * n ** (5 / 3))
    W = np.zeros_like(n)
    qc = np.zeros_like(n)
    if act.any():
        w0, _, q = _q2d(n[act], gr2[act], tau[act])
        na = n[act]
        alpha = pauli[act] / (CTF * na ** (5 / 3))
        p = gr2[act] / (4 * K ** 2 * na ** (8 / 3))
        old = g2(alpha * p / T0)
        W[act] = np.where(old > 0, w0.value * g2(alpha * p / t) / np.where(old > 0, old, 1), 0.0)
        qc[act] = q.value
    return W, qc


def polarized(rs, w, t):
    """dE_x (spin scaled) and dE_c (no spin scaling) relative to spin-polarized r2SCAN (Ha)."""
    dEx = 0.0
    for r in rs:
        n2, gg, t2 = 2 * np.maximum(r[0], 0), 4 * np.einsum("ix,ix->x", r[1:4], r[1:4]), 2 * r[4]
        ex, _, _ = branches(n2, gg, t2, t)
        dEx += 0.5 * float(w @ (ex - pyscf_baseline(n2, gg, t2).exchange))
    n = np.maximum(rs[0][0] + rs[1][0], 0)
    gr = rs[0][1:4] + rs[1][1:4]
    W, qc = weight_only(n, np.einsum("ix,ix->x", gr, gr), rs[0][4] + rs[1][4], t)
    ec_r2 = n * libxc.eval_xc("MGGA_C_R2SCAN", (rs[0], rs[1]), spin=1, deriv=0)[0]
    dEc = float(w @ (W * (qc - ec_r2)))
    return dEx, dEc


def ibm(t):
    istls, _ = ibmb.reference_curves()
    ns = 1 / (np.pi * ibmb.RS2D ** 2)
    rows = []
    for L in ibmb.WIDTHS:
        x, wq = np.polynomial.legendre.leggauss(ibmb.quad_points(L))
        y, wy = (x + 1) / 2, wq / 2
        f = 2 * np.sin(np.pi * y) ** 2
        fp = 2 * np.pi * np.sin(2 * np.pi * y)
        n = ns / L * f
        gr2 = (ns / L ** 2 * fp) ** 2
        tau = gr2 / (8 * n) + (np.pi * ns / 2) * n
        ex, ec, _ = branches(n, gr2, tau, t)
        pe = lambda e: float(L * np.dot(wy, e) / ns)            # noqa: E731
        fock = ibmb.exact_fock_exchange(L)
        xc = float(istls(L))
        rows.append(dict(L=L, x=pe(ex) - fock, c=pe(ec) - (xc - fock), xc=pe(ex + ec) - xc))
    d = pd.DataFrame(rows)
    out = {}
    for lab, sel in (("ge1", d.L >= 1), ("lt1", d.L < 1)):
        for k in ("x", "c", "xc"):
            out[f"mae_{k}_{lab}"] = float(d[sel][k].abs().mean())
    return out


def main():
    species = list(ATOM_SPIN) + list(MOLECULES)
    dens = {s: np.load(os.path.join(PARENT, "densities_dyall", s + ".npz")) for s in species}
    e_r2 = {s: float(dens[s]["e_tot"]) for s in species}
    ibm_rows, sp_rows, ae_rows, bh_rows = [], [], [], []
    for t in THRESHOLDS:
        ibm_rows.append(dict(t=t, **ibm(t)))
        E = {}
        for s in species:
            dEx, dEc = polarized(dens[s]["rho"], dens[s]["w"], t)
            E[s] = e_r2[s] + dEx + dEc
            sp_rows.append(dict(t=t, species=s, dEx_mHa=1e3 * dEx, dEc_mHa=1e3 * dEc))
        for m, ref in AE6_REF.items():
            at = [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
            ae_rows.append(dict(t=t, molecule=m, ref=ref, r2scan=(sum(e_r2[a] for a in at) - e_r2[m]) * HA2KCAL,
                                scalar=(sum(E[a] for a in at) - E[m]) * HA2KCAL))
        for lab, R, ts, P, ff, rr in BH6:
            for direc, side, ref in (("forward", R, ff), ("reverse", P, rr)):
                bh_rows.append(dict(t=t, reaction=lab, direction=direc, ref=ref,
                                    r2scan=(e_r2[ts] - sum(e_r2[x] for x in side)) * HA2KCAL,
                                    scalar=(E[ts] - sum(E[x] for x in side)) * HA2KCAL))
        a = pd.DataFrame([r for r in ae_rows if r["t"] == t])
        b = pd.DataFrame([r for r in bh_rows if r["t"] == t])
        i = ibm_rows[-1]
        print(f"t={t:7.0e}  IBM MAE_x {i['mae_x_ge1']:.4f}/{i['mae_x_lt1']:.4f}  MAE_xc {i['mae_xc_ge1']:.4f}/"
              f"{i['mae_xc_lt1']:.4f} Ha (L>=1/L<1) | AE6 MAE {np.mean(np.abs(a.scalar - a.ref)):8.2f} "
              f"(r2SCAN {np.mean(np.abs(a.r2scan - a.ref)):.2f}) | BH6 MAE {np.mean(np.abs(b.scalar - b.ref)):6.2f} "
              f"(r2SCAN {np.mean(np.abs(b.r2scan - b.ref)):.2f}) kcal/mol", flush=True)
    for rows, name in ((ibm_rows, "ibm"), (sp_rows, "species"), (ae_rows, "ae6"), (bh_rows, "bh6")):
        pd.DataFrame(rows).to_csv(os.path.join(HERE, f"gate_scan_{name}{SUFFIX}.csv"), index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
