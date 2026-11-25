# Qchem branch: scalar Q2D (jana_ref) validation codes

Codes and results for the scalar, Laplacian-free Q2D correction to r2SCAN (W_orig of jana_ref and the proposed
W_new), and its independent verification with anisotropic Gaussian (aniGTO) densities.

## Layout
- `2D_Ref/` — local codes (PySCF 2.7 / 2.14).
  - `q2d_scalar.py` — the functional (energies and derivatives, PySCF adapter).
  - `reproduce_*.py`, `save_densities_ae6_bh6.py`, `pbe_nonscf_ae6/` — atoms, IBM well, AE6/BH6, PBE non-SCF,
    limit checks W = 0 / W = 1 (`pbe_nonscf_ae6/check_limits_dyall.py`, `validate_limits.py`).
  - `make_table_*.py` and `tab_*.tex` — the table generators and their outputs.
  - `validation/`
    - `merge_check.py` — Gaussian sheets/wires and the IBM well, compared with the DC-SCAN manuscript.
    - `anigto_limits.py`, `run_anigto.sbatch` — W = 0 / W = 1 / W[n] limit checks and exchange versus lambda on
      aniGTO sheets and wires (N = 2, 4, 6; lambda = 1 ... 1e4); exact exchange by the separable Coulomb kernel.
    - `common_species.py` — He, Be, H, H2 on one r2SCAN/dyall-v4z density by three routes: PySCF; analytic
      Cartesian aniGTO fields (exact tau - tau_W, Lagrange identity) on the PySCF grid; and on a radial grid split
      at alpha = 1.
    - `he_be_common.py`, `he_be_side_by_side.py` — earlier He/Be versions of the same check.
  - Density grids (`*.npz`, several GB) are not stored; regenerate them with `save_densities_ae6_bh6.py`.
- `wcss_scan/` — copy of the WCSS cluster folder `~/pyscf/gpu/scan` (AE6_BH6, he_20261111, trap20_20261118,
  validation, vasp; VASP inputs only, POTCAR files not included).

## Run
    python3 2D_Ref/validation/anigto_limits.py --m 220     # aniGTO sheets and wires (cluster: run_anigto.sbatch)
    python3 2D_Ref/validation/common_species.py            # He, Be, H, H2: PySCF vs aniGTO
    python3 2D_Ref/make_table_anigto_limits.py; python3 2D_Ref/make_table_he_be.py
