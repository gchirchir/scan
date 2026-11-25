#!/usr/bin/env python3
"""Evaluate scalar-Q2D and r2SCAN on the PDF's analytic infinite-barrier well."""
import argparse
import csv

import numpy as np

from q2d_scalar import evaluate, pyscf_baseline


def calculate(width, ns, points):
    # Gauss-Legendre nodes avoid the zero-density hard-wall endpoints.
    x, w = np.polynomial.legendre.leggauss(points)
    y, wy = (x + 1) / 2, w / 2
    f = 2 * np.sin(np.pi * y) ** 2
    fp = 2 * np.pi * np.sin(2 * np.pi * y)
    n = ns / width * f
    grad2 = (ns / width**2 * fp) ** 2
    tau_w = grad2 / (8 * n)
    tau = tau_w + (np.pi * ns / 2) * n
    base = pyscf_baseline(n, grad2, tau)
    q2d = evaluate(n, grad2, tau, baseline=pyscf_baseline)
    def per_electron(e):
        return float(width * np.dot(wy, e) / ns)
    return {
        "width_bohr": float(width), "rs2d": float(1 / np.sqrt(np.pi * ns)),
        "r2scan_exchange_ha_per_e": per_electron(base.exchange),
        "scalar_exchange_ha_per_e": per_electron(q2d.exchange),
        "r2scan_correlation_ha_per_e": per_electron(base.correlation),
        "scalar_correlation_ha_per_e": per_electron(q2d.correlation),
        "r2scan_xc_ha_per_e": per_electron(base.exchange + base.correlation),
        "scalar_xc_ha_per_e": per_electron(q2d.exchange + q2d.correlation),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--widths", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5,
                            3, 3.5, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--rs2d", type=float, default=4.0)
    p.add_argument("--points", type=int, default=800)
    p.add_argument("--output", default="ibm_results.csv")
    args = p.parse_args()
    ns = 1 / (np.pi * args.rs2d**2)
    rows = [calculate(L, ns, args.points) for L in args.widths]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"L={row['width_bohr']:5.2f}  Ex={row['scalar_exchange_ha_per_e']: .7f}  "
              f"Ec={row['scalar_correlation_ha_per_e']: .7f}  "
              f"Exc={row['scalar_xc_ha_per_e']: .7f} Ha/e")


if __name__ == "__main__":
    main()
