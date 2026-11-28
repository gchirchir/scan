#!/usr/bin/env python3
"""Generate VASP inputs for PRL 108, 126402 (2012) and JCP 152, 151101 (2020).

Standalone: imports only the Python standard library and NumPy. POTCAR files are
never generated or copied; POTCAR.spec records the required element/potential.
Seed slabs are geometry checks. Production slabs must be regenerated with
--lattice-json after the functional-specific bulk optimizations are complete.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import numpy as np

FUNCTIONALS = ("LDA", "PBE", "PBEsol", "Q2D-GGA")
FCC = {"Ag": 4.09, "Au": 4.08, "Cu": 3.61, "Ir": 3.84, "Pd": 3.89,
       "Pt": 3.92, "Rh": 3.80}
BCC = {"Cr": 2.88, "Fe": 2.87, "Mo": 3.15, "Nb": 3.30, "Ta": 3.30,
       "V": 3.03, "W": 3.17}
HCP = {"Co": (2.51, 4.07), "Cd": (2.98, 5.62), "Hf": (3.19, 5.05),
       "Ru": (2.71, 4.28), "Ti": (2.95, 4.68), "Zn": (2.66, 4.95),
       "Zr": (3.23, 5.15)}
PRL_FCC = {"Al": 4.02, "Cu": 3.60, "Rh": 3.79, "Au": 4.07,
           "Pt": 3.92, "Pd": 3.88, "Ti": 4.10}
POTCAR = {"Ag": "Ag", "Al": "Al", "Au": "Au", "Cd": "Cd", "Co": "Co",
          "Cr": "Cr_pv", "Cu": "Cu", "Fe": "Fe", "Hf": "Hf_pv",
          "Ir": "Ir", "Mo": "Mo_pv", "Nb": "Nb_pv", "Pd": "Pd",
          "Pt": "Pt", "Rh": "Rh", "Ru": "Ru_pv", "Ta": "Ta_pv",
          "Ti": "Ti_pv", "V": "V_pv", "W": "W_pv", "Zn": "Zn",
          "Zr": "Zr_sv"}


def lattice(kind, a, c=None):
    if kind == "fcc":
        return np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0.0]]), [[0, 0, 0]]
    if kind == "bcc":
        return np.array([[-a/2, a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, -a/2]]), [[0, 0, 0]]
    return np.array([[a, 0, 0], [a/2, math.sqrt(3)*a/2, 0], [0, 0, c]]), [[0, 0, 0], [2/3, 1/3, 1/2]]


def normal(kind, facet, direct):
    hkl = {"100": (1, 0, 0), "110": (1, 1, 0), "111": (1, 1, 1),
           "0001": (0, 0, 1), "10-10": (1, 0, 0), "11-20": (1, 1, 0)}[facet]
    if kind in ("fcc", "bcc"):
        return np.array(hkl, float) / np.linalg.norm(hkl)
    rec = np.linalg.inv(direct).T
    n = np.array(hkl, float) @ rec
    return n / np.linalg.norm(n)


def plane_vectors(direct, n):
    cand = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i, j, k) == (0, 0, 0):
                    continue
                v = np.array([i, j, k], float) @ direct
                if abs(np.dot(v, n)) < 1e-7:
                    cand.append(v)
    cand.sort(key=lambda x: np.dot(x, x))
    best = None
    for u in cand[:80]:
        for v in cand[:80]:
            area = np.linalg.norm(np.cross(u, v))
            if area < 1e-8:
                continue
            alignment = abs(np.dot(np.cross(u, v) / area, n))
            if alignment > 1 - 1e-8 and (best is None or area < best[0] - 1e-8):
                best = (area, u.copy(), v.copy())
    if best is None:
        raise RuntimeError("could not find surface translations")
    _, u, v = best
    if np.dot(np.cross(u, v), n) < 0:
        v *= -1
    return u, v


def slab(kind, facet, a, c, layers, vacuum):
    direct, basis = lattice(kind, a, c)
    n = normal(kind, facet, direct)
    u, v = plane_vectors(direct, n)
    gram_inv = np.linalg.inv(np.array([[np.dot(u, u), np.dot(u, v)],
                                       [np.dot(u, v), np.dot(v, v)]]))
    raw = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            for k in range(-10, 11):
                trans = np.array([i, j, k], float)
                for b in basis:
                    p = (trans + np.array(b)) @ direct
                    xy = gram_inv @ np.array([np.dot(p, u), np.dot(p, v)])
                    if -1e-7 <= xy[0] < 1-1e-7 and -1e-7 <= xy[1] < 1-1e-7:
                        raw.append((float(np.dot(p, n)), float(xy[0] % 1), float(xy[1] % 1)))
    zs = []
    for z, _, _ in sorted(raw):
        if not zs or abs(z-zs[-1]) > 1e-6:
            zs.append(z)
    mid = len(zs)//2
    chosen = zs[mid-layers//2:mid+layers//2+1]
    if len(chosen) != layers:
        raise RuntimeError(f"only {len(chosen)} layers generated")
    points = []
    seen = set()
    for z, x, y in raw:
        if any(abs(z-q) < 1e-6 for q in chosen):
            key = (round(z, 7), round(x, 7), round(y, 7))
            if key not in seen:
                seen.add(key)
                points.append((z, x, y))
    z0, z1 = min(chosen), max(chosen)
    length = z1-z0+vacuum
    frac = sorted([(x, y, (z-z0+vacuum/2)/length) for z, x, y in points], key=lambda q: q[2])
    cell = np.array([u, v, n*length])
    return cell, frac, len(chosen), vacuum


def write_poscar(path, symbol, cell, frac, title, normal_only=False):
    lines = [title, "1.0"] + ["  " + "  ".join(f"{x:16.10f}" for x in row) for row in cell]
    lines += [symbol, str(len(frac))]
    if normal_only:
        lines += ["Selective dynamics", "Direct"]
        lines += ["  " + "  ".join(f"{x:16.10f}" for x in p) + "  F F T" for p in frac]
    else:
        lines += ["Direct"]
        lines += ["  " + "  ".join(f"{x:16.10f}" for x in p) for p in frac]
    path.write_text("\n".join(lines) + "\n")


def xc_lines(xc):
    if xc == "LDA": return ["GGA = CA"]
    if xc == "PBE": return ["GGA = PE"]
    if xc == "PBEsol": return ["GGA = PS"]
    return ["GGA = LIBXC", "LIBXC1 = GGA_X_Q2D", "LIBXC2 = GGA_C_Q2D"]


def incar(xc, slab_run, magnetic, natoms, encut, sigma, ediff):
    lines = ["SYSTEM = surface-energy reproduction", "PREC = Accurate", f"ENCUT = {encut}",
             f"EDIFF = {ediff:.1e}", "ALGO = Normal", "LREAL = .FALSE.", "LASPH = .TRUE.",
             "ADDGRID = .TRUE.", "ISMEAR = 0", f"SIGMA = {sigma}", "LWAVE = .FALSE.",
             "LCHARG = .TRUE.", "NELM = 200"] + xc_lines(xc)
    if slab_run:
        lines += ["IBRION = 2", "NSW = 200", "ISIF = 2", "EDIFFG = -0.01", "ISYM = 2"]
    else:
        lines += ["IBRION = 2", "NSW = 120", "ISIF = 3", "EDIFFG = -0.005", "ISYM = 2"]
    if magnetic:
        lines += ["ISPIN = 2", f"MAGMOM = {natoms}*3.0"]
    return "\n".join(lines) + "\n"


def kpoints(mesh):
    return "Automatic mesh\n0\nGamma\n%d %d %d\n0 0 0\n" % mesh


def cases(protocol):
    if protocol == "prl2012":
        return [(e, "fcc", "111", a, None) for e, a in PRL_FCC.items()]
    out = []
    for e, a in FCC.items():
        out += [(e, "fcc", f, a, None) for f in ("100", "110", "111")]
    for e, a in BCC.items():
        out += [(e, "bcc", f, a, None) for f in ("100", "110", "111")]
    for e, (a, c) in HCP.items():
        out += [(e, "hcp", f, a, c) for f in ("0001", "10-10", "11-20")]
    return out


def load_lattice(path):
    return json.loads(Path(path).read_text()) if path else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=("prl2012", "jcp2020"), default="jcp2020")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lattice-json", help="functional-specific optimized a/c values")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        if not args.force: raise SystemExit(f"refusing nonempty output: {root}; use --force")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    opt = load_lattice(args.lattice_json)
    is_jcp = args.protocol == "jcp2020"
    layers, vacuum, encut, mesh, sigma, ediff = ((13, 21.0, 800, (16,16,1), 0.05, 1e-6)
                                                  if is_jcp else (11, 10.5, 476, (10,10,1), 0.05, 1e-6))
    manifest = []
    unique_bulk = {}
    geometry_cache = {}
    for el, kind, facet, seed_a, seed_c in cases(args.protocol):
        unique_bulk[el] = (kind, seed_a, seed_c)
        for xc in FUNCTIONALS:
            vals = opt.get(xc, {}).get(el, {})
            a, c = float(vals.get("a", seed_a)), vals.get("c", seed_c)
            c = None if c is None else float(c)
            gkey = (kind, facet, a, c, layers, vacuum)
            if gkey not in geometry_cache:
                geometry_cache[gkey] = slab(kind, facet, a, c, layers, vacuum)
            cell, xyz, nlay, vac = geometry_cache[gkey]
            d = root/"slabs"/xc/el/facet
            d.mkdir(parents=True, exist_ok=True)
            write_poscar(d/"POSCAR", el, cell, xyz, f"{el} {kind}({facet}) {layers}L {xc}",
                         normal_only=not is_jcp)
            (d/"INCAR").write_text(incar(xc, True, el in ("Fe", "Co"), len(xyz), encut, sigma, ediff))
            (d/"KPOINTS").write_text(kpoints(mesh))
            (d/"POTCAR.spec").write_text(POTCAR[el] + "\n")
            (d/"STATUS").write_text(("production-from-optimized-lattice\n" if vals else
                                      "SEED ONLY: regenerate using --lattice-json before production\n"))
            manifest.append({"protocol": args.protocol, "stage": "slab", "xc": xc, "element": el,
                             "structure": kind, "facet": facet, "layers": nlay, "atoms": len(xyz),
                             "a": a, "c": c, "vacuum_A": vac, "kmesh": mesh, "encut_eV": encut,
                             "production_geometry": bool(vals), "path": str(d.relative_to(root))})
    for el, (kind, seed_a, seed_c) in sorted(unique_bulk.items()):
        for xc in FUNCTIONALS:
            d = root/"bulk"/xc/el
            d.mkdir(parents=True, exist_ok=True)
            cell, basis = lattice(kind, seed_a, seed_c)
            write_poscar(d/"POSCAR", el, cell, basis, f"{el} {kind} bulk {xc}")
            (d/"INCAR").write_text(incar(xc, False, el in ("Fe", "Co"), len(basis), encut, sigma, ediff))
            (d/"KPOINTS").write_text(kpoints((16,16,16)))
            (d/"POTCAR.spec").write_text(POTCAR[el] + "\n")
            manifest.append({"protocol": args.protocol, "stage": "bulk", "xc": xc, "element": el,
                             "structure": kind, "a_seed": seed_a, "c_seed": seed_c,
                             "kmesh": (16,16,16), "encut_eV": encut, "path": str(d.relative_to(root))})
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root/"jobs.tsv").write_text("stage\txc\telement\tfacet\tpath\n" + "".join(
        f"{m['stage']}\t{m['xc']}\t{m['element']}\t{m.get('facet','-')}\t{m['path']}\n" for m in manifest))
    example = {xc: {el: ({"a": a, "c": c} if c is not None else {"a": a})
                    for el, (_, a, c) in sorted(unique_bulk.items())} for xc in FUNCTIONALS}
    (root/"lattice.example.json").write_text(json.dumps(example, indent=2) + "\n")
    (root/"README.txt").write_text(
        f"VASP inputs for {args.protocol}.\n\n"
        "STAGE 1: run every bulk/<XC>/<element> directory. Extract the optimized lattice parameters from CONTCAR "
        "into a copy of lattice.example.json named lattice.json. For cubic cells store a; for hcp store a and c.\n"
        "STAGE 2: rerun codes/23_vasp_surface_inputs.py with --lattice-json lattice.json and a new output path. "
        "Only slabs whose STATUS says production-from-optimized-lattice may enter a surface-energy table.\n"
        "STAGE 3: run every slab, verify ionic/electronic convergence, and compute "
        "sigma=(E_slab-N*E_bulk_atom)/(2*A), using matching XC, POTCAR, ENCUT and converged settings.\n\n"
        "Q2D-GGA requires VASP >=6.3 compiled with -DUSELIBXC and a Libxc build containing "
        "GGA_X_Q2D/GGA_C_Q2D. Test those identifiers before production. POTCAR.spec is a manifest only; licensed "
        "POTCAR files are not included. Confirm the potential labels and independently converge ENCUT, bulk and "
        "slab k meshes, layer count and vacuum. The papers do not fully specify the 2020 PAW datasets or bulk "
        "k mesh, so those choices are explicit reproduction settings, not claimed verbatim paper parameters.\n")
    print(f"wrote {len(manifest)} calculations under {root}")


if __name__ == "__main__":
    main()
