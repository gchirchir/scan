# Scalar Q2D functional: PySCF reproduction of scalar_functional_form.pdf

Report: `main.tex` / `main.pdf` (PySCF numbers in red, PDF values in black).  Previous version in `bak_20261123/`.

| step | command | output |
|---|---|---|
| atomic table (PDF Sec. 6) | `python3 reproduce_atomic.py` | `atomic_results.csv`, `atomic_run_metadata.json` |
| IBM figure + MAEs (PDF Sec. 7) | `python3 reproduce_ibm_benchmark.py` | `ibm_benchmark.{csv,pdf,eps,png}`, `ibm_mae.csv` |
| LaTeX tables | `python3 make_tables.py` | `tab_atomic.tex`, `tab_atomic_shift.tex`, `tab_ibm_mae.tex` |
| report | `pdflatex main.tex` (twice) | `main.pdf` |

Python: `python3` with PySCF 2.14, numpy, scipy, pandas, matplotlib.  Functional: `q2d_scalar.py` (supplied;
checked term by term against the PDF equations).  C++ files: Q-Chem route, not used here.

## Results
- Atoms: HF densities in **uncontracted** ANO-RCC (contracted ANO-RCC is a relativistic contraction; its
  non-relativistic HF misses the HF limit by 25 Ha for Kr and 264 Ha for Xe).  All atoms within 0.02-6 mHa of the
  numerical HF limit; Ex, Ec within 3 mHa of the PDF; scalar minus r2SCAN shifts within 0.3 mHa (Mg +65.3 vs +65.6 mHa).
- IBM: exact Fock exchange computed from the subband form factor (strict-2D limit -0.150053 Ha checked); ISTLS from
  `chiodo2012_fig1_digitized.csv` (Chiodo et al. PRL 108, 126402, Fig. 1).  Widths read from the PDF figure:
  0.002, 0.01, 0.1, 0.25, 0.5, 0.75 | 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, ..., 10.  Exchange MAEs 0.026433 / 0.035057 equal
  the PDF to all digits; correlation and XC within 0.4 mHa (ISTLS digitization).
- Quadrature: 3000 Gauss-Legendre points, 12000 for L < 0.05 bohr (800 points err by 5 mHa at L = 0.002).

## AE6 / BH6 (2026-11-23)
- References: Haunschild & Klopper, Theor. Chem. Acc. 131, 1112 (2012) (`s00214-012-1112-3.pdf`), Table 4 E_nrel, kJ/mol -> kcal/mol.
- Functional: original `q2d_scalar.py` (md5 ab13dc61e77911c0073217fe71e184f8), no modification.
- Basis/grid: dyall-v4z, grid level 9 (as `reproduce_atomic.py`).  Per species:
  `AE6_SPECIES_DIR=species_dyall AE6_DENSITY_DIR=densities_dyall python3 reproduce_ae6_bh6.py species NAME`
  (all 21 run locally, log `ae6_bh6_dyall_local.log`), then
  `AE6_SPECIES_DIR=species_dyall python3 reproduce_ae6_bh6.py assemble && python3 make_tables.py && pdflatex main.tex` (x2).
- Ex/Ec per species (Tables 5, 6 of main.pdf): `python3 ae6_bh6_xc_table.py` -> `ae6_bh6_xc.csv`, then `make_tables.py`.
- def2-QZVPP / grid 6 results kept in `species_def2qzvpp/`, `densities_def2qzvpp/`.
- Cluster (WCSS `~/pyscf/gpu/scan/AE6_BH6`): `sbatch ae6_bh6_cpu.sbatch NAME ...` (lem-cpu-short); not needed in the end.
- `not_used_gate_variant/`: exploratory modified-gate scan, NOT used (user: no modification of the functional).
