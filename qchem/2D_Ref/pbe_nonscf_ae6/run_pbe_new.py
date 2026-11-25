#!/usr/bin/env python3
"""Steps 1-2 of the supervisor's list on PBE orbitals (non-SCF, energy only, no potential):
    E^F = E^PBE - E_xc^PBE[n_up, n_dn] + E_xc^F[n_up, n_dn]
for F = r2SCAN, original scalar (W_orig, jana_ref Eq. 4) and new scalar (W_new), each with the SS and TD spin routes
(exchange spin-scaled in SS; correlation never spin-scaled).  dyall-v4z, grid level 6 (env).  For each species also
the W = 0 check (scalar machinery with W = 0 vs Libxc spin-polarized r2SCAN).
Run: python3 run_pbe_new.py species NAME ; python3 run_pbe_new.py assemble
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, os.path.join(PARENT, "w_gate_variant"))

import numpy as np                                                                 # noqa: E402
from pyscf import dft, gto                                                         # noqa: E402
from pyscf.dft import libxc                                                        # noqa: E402

from reproduce_ae6_bh6 import ATOM_SPIN, MOLECULES, AE6_REF, AE6_LABEL, BH6, HA2KCAL, read_xyz, delta_xc   # noqa: E402
from verify_variant import dexc                                                    # noqa: E402
from q2d_scalar import pyscf_baseline                                              # noqa: E402

BASIS = os.environ.get("AE6_BASIS", "dyall-v4z")
GRID = int(os.environ.get("AE6_GRID", "6"))
OUT = os.path.join(HERE, f"species_new_{BASIS}_g{GRID}")
VARIANTS = {"orig": (1e-5, 0, 0), "new_SS": (1e-3, 5.0, 12), "new_TD": (1e-4, 6.0, 12),
            "newJ": (1e-5, 6.0, 12)}   # newJ: W_new = W_orig * p^k/(p^k+p0^k), Jana constant kept (proposed)


def species(name):
    if name in ATOM_SPIN:
        atom, spin = f"{name} 0 0 0", ATOM_SPIN[name]
    else:
        xyz, spin = MOLECULES[name]
        atom = read_xyz(xyz)
    mol = gto.M(atom=atom, basis=BASIS, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol)
    mf.xc, mf.grids.level, mf.conv_tol, mf.max_cycle = "pbe", GRID, 1e-9, 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    assert mf.converged, name
    ni = dft.numint.NumInt()
    dm = mf.make_rdm1()
    nr = ni.nr_uks if spin else ni.nr_rks
    e_r2 = mf.e_tot - float(nr(mol, mf.grids, "pbe", dm)[1]) + float(nr(mol, mf.grids, "r2scan", dm)[1])
    delta_xc(mol, mf)
    rs, w = delta_xc.last
    # W = 0 check: spin-scaled unpolarized r2SCAN exchange + polarized correlation == Libxc polarized r2SCAN
    ex0 = sum(0.5 * float(w @ pyscf_baseline(2 * np.maximum(r[0], 0), 4 * np.einsum("ix,ix->x", r[1:4], r[1:4]),
                                              2 * r[4]).exchange) for r in rs)
    ex_lx = float(w @ ((rs[0][0] + rs[1][0]) * libxc.eval_xc("MGGA_X_R2SCAN", (rs[0], rs[1]), spin=1)[0]))
    d = dict(species=name, spin2S=spin, basis=BASIS, grid=GRID, e_pbe=float(mf.e_tot), e_r2scan=e_r2,
             w0_check_Ha=ex0 - ex_lx)
    for v, var in VARIANTS.items():
        for route in ("SS", "TD"):
            dx, dc = dexc(rs, w, var, route, [])
            d[f"e_{v}_{route}"] = e_r2 + dx + dc
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, name + "_dens.npz"), rho=rs, w=w, e_r2scan=e_r2)
    json.dump(d, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(f"{name:7s} E(r2SCAN@PBE)={e_r2:.8f}  W=0 check {d['w0_check_Ha']:+.1e} Ha", flush=True)


def assemble():
    import pandas as pd
    sp = {n: json.load(open(os.path.join(OUT, n + ".json"))) for n in list(ATOM_SPIN) + list(AE6_REF)}
    keys = ["e_pbe", "e_r2scan"] + [f"e_{v}_{r}" for v in VARIANTS for r in ("SS", "TD")]
    rows = []
    for m, ref in AE6_REF.items():
        at = [l.split()[0] for l in read_xyz(MOLECULES[m][0]).split("\n")]
        rows.append(dict(molecule=m, label=AE6_LABEL[m], ref=ref,
                         **{k[2:]: (sum(sp[a][k] for a in at) - sp[m][k]) * HA2KCAL for k in keys}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, f"ae6_pbe_new_{BASIS}_g{GRID}.csv"), index=False, float_format="%.4f")
    print(df.round(2).to_string(index=False))
    for k in keys:
        e = df[k[2:]] - df.ref
        print(f"{k[2:]:10s} ME={e.mean():+9.2f} MAE={e.abs().mean():9.2f}")
    print("max |W=0 check| =", max(abs(v["w0_check_Ha"]) for v in sp.values()))
    # BH6 (if all BH6 species are present)
    need = {x for _, R, ts, P, _, _ in BH6 for x in R + P + (ts,)}
    if all(os.path.exists(os.path.join(OUT, n + ".json")) for n in need):
        sb = {n: json.load(open(os.path.join(OUT, n + ".json"))) for n in need}
        rows = []
        for lab, R, ts, P, ff, rr in BH6:
            for side, ref, direc in ((R, ff, "forward"), (P, rr, "reverse")):
                rows.append(dict(reaction=lab, direction=direc, ref=ref,
                                 **{k[2:]: (sb[ts][k] - sum(sb[x][k] for x in side)) * HA2KCAL for k in keys}))
        bh = pd.DataFrame(rows)
        bh.to_csv(os.path.join(HERE, f"bh6_pbe_new_{BASIS}_g{GRID}.csv"), index=False, float_format="%.4f")
        print(bh.round(2).to_string(index=False))
        for k in keys:
            e = bh[k[2:]] - bh.ref
            print(f"BH6 {k[2:]:10s} ME={e.mean():+9.2f} MAE={e.abs().mean():9.2f}")
        print("max |W=0 check| BH6 species =", max(abs(v["w0_check_Ha"]) for v in sb.values()))


if __name__ == "__main__":
    species(sys.argv[2]) if sys.argv[1] == "species" else assemble()
