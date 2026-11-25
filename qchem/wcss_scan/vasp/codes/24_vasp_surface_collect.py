#!/usr/bin/env python3
"""Collect optimized lattices and surface energies from script-23 VASP trees."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import numpy as np


def poscar(path):
    s = Path(path).read_text().splitlines()
    scale = float(s[1].split()[0])
    cell = np.array([[float(x) for x in s[i].split()[:3]] for i in range(2, 5)]) * scale
    counts = [int(x) for x in s[6].split()]
    return cell, sum(counts)


def final_toten(path):
    text = Path(path).read_text(errors="replace")
    vals = re.findall(r"free\s+energy\s+TOTEN\s+=\s+([-+0-9.Ee]+)\s+eV", text)
    if not vals:
        raise ValueError(f"no TOTEN in {path}")
    if "reached required accuracy" not in text and "aborting loop because EDIFF is reached" not in text:
        raise ValueError(f"no convergence marker in {path}")
    return float(vals[-1])


def collect_lattice(root, out):
    root = Path(root)
    data = json.loads((root/"manifest.json").read_text())
    result = {}
    for m in data:
        if m["stage"] != "bulk": continue
        cell, _ = poscar(root/m["path"]/"CONTCAR")
        kind = m["structure"]
        if kind == "fcc": a, c = math.sqrt(2)*np.linalg.norm(cell[0]), None
        elif kind == "bcc": a, c = 2*np.linalg.norm(cell[0])/math.sqrt(3), None
        else: a, c = np.linalg.norm(cell[0]), np.linalg.norm(cell[2])
        result.setdefault(m["xc"], {})[m["element"]] = ({"a": a, "c": c} if c else {"a": a})
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {sum(map(len,result.values()))} optimized cells to {out}")


def collect_surface(bulk_root, slab_root, out):
    bulk_root, slab_root = Path(bulk_root), Path(slab_root)
    bm = json.loads((bulk_root/"manifest.json").read_text())
    sm = json.loads((slab_root/"manifest.json").read_text())
    bulk = {}
    for m in bm:
        if m["stage"] != "bulk": continue
        d = bulk_root/m["path"]
        _, n = poscar(d/"POSCAR")
        bulk[(m["xc"],m["element"])] = final_toten(d/"OUTCAR")/n
    rows = []
    for m in sm:
        if m["stage"] != "slab": continue
        if not m.get("production_geometry"):
            raise ValueError(f"seed slab rejected: {m['path']}")
        d = slab_root/m["path"]
        cell, n = poscar(d/"POSCAR")
        area = np.linalg.norm(np.cross(cell[0],cell[1]))
        eslab = final_toten(d/"OUTCAR")
        ebulk = bulk[(m["xc"],m["element"])]
        sigma_ev_a2 = (eslab-n*ebulk)/(2*area)
        rows.append({**m, "E_slab_eV": eslab, "E_bulk_atom_eV": ebulk,
                     "area_A2": area, "sigma_eV_A2": sigma_ev_a2,
                     "sigma_J_m2": sigma_ev_a2*16.02176634})
    fields = list(rows[0])
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} surface energies to {out}")


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("lattice"); a.add_argument("root"); a.add_argument("out")
    a=sub.add_parser("surface"); a.add_argument("bulk_root"); a.add_argument("slab_root"); a.add_argument("out")
    x=ap.parse_args()
    if x.cmd=="lattice": collect_lattice(x.root,x.out)
    else: collect_surface(x.bulk_root,x.slab_root,x.out)


if __name__ == "__main__": main()
