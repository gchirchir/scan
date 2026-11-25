#!/usr/bin/env python3
"""He, Be side by side: E_x and E_c with W = 0, W = 1, W_orig (jana_ref) and W_new, by three routes on ONE density
(self-consistent r2SCAN, RKS, dyall-v4z, PySCF grid 9).

  PySCF          route of Table 9: PySCF NumInt fields (n, grad n, tau from the AO basis) on the PySCF grid;
                 W = 0, 1, W_orig through validate_limits.polarized_xc (spin-polarized implementation)
  aniGTO(PySCF)  aniGTO_Q2D(lambda = 1): analytic aniGTO fields (s Gaussians, tau - tau_W exact by the Lagrange
                 identity) on the SAME PySCF grid points and weights
  aniGTO(radial) aniGTO_Q2D(lambda = 1): same analytic fields, radial grid split at alpha = 1 (converged, 1e-9 Ha)
PySCF vs aniGTO(PySCF): only tau - tau_W differs (round-off vs exact).  aniGTO(PySCF) vs aniGTO(radial): only the grid.
W_new = W_orig * p^12/(p^12 + 6^12).  Writes he_be_side_by_side.csv.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import he_be_common as hb                                           # noqa: E402
from he_be_common import CTF, _q2d, active_mask, polarized_xc     # noqa: E402
from q2d_scalar import K, pyscf_baseline                            # noqa: E402
from pyscf import dft                                               # noqa: E402

np.seterr(all="ignore")
MODES = ("W0", "W1", "Worig", "Wnew")


def mix(n, g2, tau, w):
    """Unpolarized E_x, E_c for W = 0, 1, W_orig, W_new (closed shells: equal to the spin-polarized forms)."""
    base = pyscf_baseline(n, g2, tau)
    act = active_mask(n, g2, tau)
    wq, qx, qc = _q2d(n[act], g2[act], tau[act])
    p = g2[act] / (4 * K ** 2 * n[act] ** (8 / 3))
    Ws = {"W0": 0.0, "W1": 1.0, "Worig": wq.value, "Wnew": wq.value * p ** 12 / (p ** 12 + 6.0 ** 12)}
    out = {}
    for k, W in Ws.items():
        ex, ec = base.exchange.copy(), base.correlation.copy()
        ex[act] = (1 - W) * ex[act] + W * qx.value
        ec[act] = (1 - W) * ec[act] + W * qc.value
        out["Ex_" + k], out["Ec_" + k] = float(w @ ex), float(w @ ec)
    out["f_act"] = float(w[act] @ n[act] / (w @ n))
    return out


def anigto_fields(a, d, r2):
    E = np.exp(-np.outer(r2, a))
    phi, dp = E @ d.T, -2 * (E * a) @ d.T
    n = 2 * (phi ** 2).sum(1)
    g2 = (4 * (phi * dp).sum(1)) ** 2 * r2
    pa = np.zeros_like(n)
    for i in range(phi.shape[1]):
        for j in range(i + 1, phi.shape[1]):
            pa += (phi[:, i] * dp[:, j] - phi[:, j] * dp[:, i]) ** 2 * r2
    k = n > 1e-60
    return n[k], g2[k], g2[k] / (8 * n[k]) + 2 * pa[k] / n[k], k


def main(m=2000):
    import pandas as pd
    rows = []
    for sym in ("He", "Be"):
        mol, mf = hb.scf(sym)
        a, d, _, _ = hb.s_orbitals(mol, mf)
        # PySCF route (Table 9)
        dm = mf.make_rdm1()
        ni = dft.numint.NumInt()
        R = [ni.eval_rho(mol, ao, dm, mask, xctype="MGGA", hermi=1, with_lapl=False)
             for ao, mask, wb, c in ni.block_loop(mol, mf.grids, mol.nao, deriv=1, max_memory=2000)]
        r = np.hstack(R)
        w = mf.grids.weights
        n, g2, tau = r[0], (r[1:4] ** 2).sum(0), r[4]
        k = n > 1e-60
        A = mix(n[k], g2[k], tau[k], w[k])
        rs = np.array([r / 2, r / 2])
        for mode, key in ((0, "W0"), (1, "W1"), ("model", "Worig")):        # tie to the Table-9 code
            ex, ec = polarized_xc(rs, w, mode)
            A[f"dpol_Ex_{key}"], A[f"dpol_Ec_{key}"] = ex - A["Ex_" + key], ec - A["Ec_" + key]
        # aniGTO fields on the same PySCF points
        nb, gb, tb, kb = anigto_fields(a, d, (mf.grids.coords ** 2).sum(1))
        B1 = mix(nb, gb, tb, w[kb])
        # aniGTO fields, converged radial grid (panels split at alpha = 1)
        hb.route_b(a, d, m)
        y, wy = np.polynomial.legendre.leggauss(m)
        rr, ww = [], []
        for lo, hi in zip(hb.route_b.edges[:-1], hb.route_b.edges[1:]):
            q = 0.5 * (np.log(hi) - np.log(lo)) * y + 0.5 * (np.log(hi) + np.log(lo))
            rr.append(np.exp(q)); ww.append(0.5 * (np.log(hi) - np.log(lo)) * wy * np.exp(q))
        rr, ww = np.concatenate(rr), np.concatenate(ww)
        nr, gr, tr, kr = anigto_fields(a, d, rr * rr)
        B2 = mix(nr, gr, tr, (4 * np.pi * rr * rr * ww)[kr])
        row = dict(species=sym)
        for lab, D in (("pyscf", A), ("ani_pyscfgrid", B1), ("ani_radial", B2)):
            row.update({f"{lab}_{k_}": v for k_, v in D.items()})
        rows.append(row)
        print(f"\n{sym}  f_act: PySCF {A['f_act']:.3f}  aniGTO(PySCF grid) {B1['f_act']:.3f}  aniGTO(radial) {B2['f_act']:.3f}")
        print(f"{'':10s}{'PySCF':>16s}{'aniGTO(PySCF)':>16s}{'aniGTO(radial)':>16s}{'aniPG-PySCF':>13s}{'radial-PySCF':>13s}")
        for c in ("Ex", "Ec"):
            for md in MODES:
                key = f"{c}_{md}"
                print(f"{key:10s}{A[key]:16.10f}{B1[key]:16.10f}{B2[key]:16.10f}{B1[key] - A[key]:13.1e}{B2[key] - A[key]:13.1e}")
        print("   polarized (Table 9 code) - unpolarized:", " ".join(f"{k_} {v:+.0e}" for k_, v in A.items() if k_.startswith("dpol")))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "he_be_side_by_side.csv"), index=False, float_format="%.15e")


if __name__ == "__main__":
    main()
