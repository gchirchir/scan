#!/usr/bin/env python3
"""Limit checks of the spin-polarized scalar functional (standard spin treatment) before the AE6 runs.

Standard spin treatment:
  exchange    exact spin scaling   E_x[n_up, n_dn] = 1/2 E_x[2 n_up] + 1/2 E_x[2 n_dn]
              with the unpolarized mixture  e_x[n] = (1 - W[n]) e_x^r2SCAN[n] + W[n] e_x^Q[n]
  correlation no spin-scaling relation; W and the Q2D branch from the total density,
              e_c = (1 - W[n]) e_c^r2SCAN(n_up, n_dn) + W[n] e_c^Q[n]

Checks on PBE densities (grid integrals, Ha):
  W = 0      : must equal Libxc spin-polarized r2SCAN (MGGA_X_R2SCAN, MGGA_C_R2SCAN), and PySCF's
               own E_xc('r2scan') from nr_uks/nr_rks
  W = 1      : must equal the pure Q2D branch (exchange spin-scaled, correlation from n)
  W = W[n]   : closed shells must equal q2d_scalar.evaluate(n) (the reference implementation of jana_ref)
               and open shells the delta_xc() route SS + common E_c used for the AE6 tables

Usage: python3 validate_limits.py [species ...]   (default: H C O SiH4 C3H4; basis env AE6_BASIS, grid AE6_GRID)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                     # noqa: E402
from pyscf import dft, gto                             # noqa: E402
from pyscf.dft import libxc                            # noqa: E402

from q2d_scalar import _q2d, pyscf_baseline, evaluate, CTF   # noqa: E402
from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES, read_xyz, delta_xc   # noqa: E402

BASIS = os.environ.get("AE6_BASIS", "def2-tzvp")
GRID = int(os.environ.get("AE6_GRID", "6"))


def active_mask(n, g2, tau):
    """Points where the Q2D branch is defined (0 < tau - tau_W < tau_unif); W = 0 elsewhere."""
    safe = np.where(n > 0, n, 1.0)
    pauli = tau - g2 / (8 * safe)
    return (n > 1e-60) & (g2 > 0) & (pauli > 0) & (pauli < CTF * n ** (5 / 3))


def unpolarized_mix(n, g2, tau, weight_mode):
    """Per-volume (e_x, e_c) of the unpolarized mixture with W = 0, 1 or the model W[n]."""
    base = pyscf_baseline(n, g2, tau)
    ex, ec = base.exchange.copy(), base.correlation.copy()
    act = active_mask(n, g2, tau)
    if weight_mode == 0 or not act.any():
        return ex, ec, act
    w, qx, qc = _q2d(n[act], g2[act], tau[act])
    W = np.ones(act.sum()) if weight_mode == 1 else w.value
    ex[act] = (1 - W) * ex[act] + W * qx.value
    ec[act] = (1 - W) * ec[act] + W * qc.value
    return ex, ec, act


def polarized_xc(rs, w, weight_mode):
    """E_x (spin scaled) and E_c (no spin scaling) of the spin-polarized functional on spin densities rs."""
    Ex = 0.0
    for r in rs:                                              # exact spin scaling of exchange
        n2 = 2 * np.maximum(r[0], 0)
        g2 = 4 * np.einsum("ix,ix->x", r[1:4], r[1:4])
        ex, _, _ = unpolarized_mix(n2, g2, 2 * r[4], weight_mode)
        Ex += 0.5 * float(w @ ex)
    n = np.maximum(rs[0][0] + rs[1][0], 0)                    # correlation: W, Q2D from total density
    gr = rs[0][1:4] + rs[1][1:4]
    g2 = np.einsum("ix,ix->x", gr, gr)
    tau = rs[0][4] + rs[1][4]
    ec_r2 = n * libxc.eval_xc("MGGA_C_R2SCAN", (rs[0], rs[1]), spin=1, deriv=0)[0]
    ec = ec_r2.copy()
    act = active_mask(n, g2, tau)
    if weight_mode != 0 and act.any():
        wv, _, qc = _q2d(n[act], g2[act], tau[act])
        W = np.ones(act.sum()) if weight_mode == 1 else wv.value
        ec[act] = (1 - W) * ec_r2[act] + W * qc.value
    return Ex, float(w @ ec)


def q2d_only(rs, w):
    """Pure Q2D energies computed directly from _q2d (independent of the mixing code), on active points only."""
    Ex = 0.0
    for r in rs:
        n2 = 2 * np.maximum(r[0], 0)
        g2 = 4 * np.einsum("ix,ix->x", r[1:4], r[1:4])
        t2 = 2 * r[4]
        act = active_mask(n2, g2, t2)
        base = pyscf_baseline(n2, g2, t2).exchange
        ex = base.copy()
        if act.any():
            ex[act] = _q2d(n2[act], g2[act], t2[act])[1].value
        Ex += 0.5 * float(w @ ex)
    return Ex


def run(name):
    if name in ATOM_SPIN:
        atom, spin = f"{name} 0 0 0", ATOM_SPIN[name]
    else:
        xyz, spin = MOLECULES[name]
        atom = read_xyz(xyz)
    mol = gto.M(atom=atom, basis=BASIS, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol)
    mf.xc, mf.grids.level, mf.conv_tol = "pbe", GRID, 1e-9
    mf.kernel()
    dm = mf.make_rdm1()
    dEx_SS, _, dEc = delta_xc(mol, mf)              # reference route of the AE6 tables
    rs, w = delta_xc.last
    ni = dft.numint.NumInt()
    ex_lx = float(w @ ((rs[0][0] + rs[1][0]) * libxc.eval_xc("MGGA_X_R2SCAN", (rs[0], rs[1]), spin=1)[0]))
    ec_lx = float(w @ ((rs[0][0] + rs[1][0]) * libxc.eval_xc("MGGA_C_R2SCAN", (rs[0], rs[1]), spin=1)[0]))
    exc_pyscf = float((ni.nr_uks if spin else ni.nr_rks)(mol, mf.grids, "r2scan", dm)[1])

    ex0, ec0 = polarized_xc(rs, w, 0)
    ex1, ec1 = polarized_xc(rs, w, 1)
    exW, ecW = polarized_xc(rs, w, "model")
    exQ = q2d_only(rs, w)
    print(f"\n{name} (2S={spin}, {BASIS}, grid {GRID}, PBE density, N={float(w @ (rs[0][0] + rs[1][0])):.6f})")
    print(f"  W=0 : Ex {ex0:.10f}  Libxc r2SCAN Ex {ex_lx:.10f}  diff {ex0 - ex_lx:+.2e}")
    print(f"        Ec {ec0:.10f}  Libxc r2SCAN Ec {ec_lx:.10f}  diff {ec0 - ec_lx:+.2e}")
    print(f"        Exc {ex0 + ec0:.10f}  PySCF nr_*ks('r2scan') {exc_pyscf:.10f}  diff {ex0 + ec0 - exc_pyscf:+.2e}")
    print(f"  W=1 : Ex {ex1:.10f}  pure Q2D Ex {exQ:.10f}  diff {ex1 - exQ:+.2e}   (Ec(W=1) {ec1:.8f})")
    print(f"  W[n]: dEx {1e3 * (exW - ex0):+.4f} mHa vs delta_xc SS {1e3 * dEx_SS:+.4f}  "
          f"dEc {1e3 * (ecW - ec0):+.4f} vs {1e3 * dEc:+.4f}")
    if not spin:   # closed shell: polarized functional must reduce to the unpolarized reference evaluate(n)
        n = rs[0][0] + rs[1][0]
        gr = rs[0][1:4] + rs[1][1:4]
        ref = evaluate(n, np.einsum("ix,ix->x", gr, gr), rs[0][4] + rs[1][4])
        print(f"        closed shell: Ex {exW:.10f} vs evaluate(n) {float(w @ ref.exchange):.10f}  "
              f"Ec {ecW:.10f} vs {float(w @ ref.correlation):.10f}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["H", "C", "O", "SiH4", "C3H4"]):
        run(s)
