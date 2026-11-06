# codes/ — standalone scripts (one file = one complete calculation)

Every number and figure of `../manuscript/main.pdf` and `si.pdf` comes from one of these nine files.  Each file
defines every function it needs (DC-SCAN functional, aniGTO integrals copied verbatim from aniGTO, Kantorovich
kernel, SCE solver) and imports only third-party packages, so a single command reproduces its data:

    cd codes && python3 01_wire_benchmark.py

Outputs go to `results/` (csv, txt, LaTeX table bodies) and `figures/` (pdf, png) next to the scripts.

| Script | Produces (manuscript item) | Needs | Time |
|---|---|---|---|
| `01_wire_benchmark.py` | wire reference quadrature, SI Tables S1–S3 (checks, local slopes, asymptotic offset), SI Fig. S1 | numpy, scipy, pandas, matplotlib | 1 min |
| `02_sheet_benchmark.py` | main-text sheet table, SI Tables S5–S6, SI Fig. S2 | numpy, scipy, matplotlib | 2 min |
| `03_anigto_wire_libxc.py` | full main-text wire table (reference + aniGTO/libxc columns), SI Table S4 (grids) | numpy, scipy, **pyscf** | <1 min |
| `04_exact_vs_proposed.py` | main-text exact-vs-proposed table and figure (log-variable ERI, axisymmetric SCE) | numpy, scipy, pandas, matplotlib | 2–5 min |
| `05_refit_switch_scales.py` | SI Table S7 (PBE base, refit of s0, Z0) | numpy, scipy | <1 min |
| `06_range_separation_q1d_lda.py` | SI §S6 range separation and quasi-1D LDA (text values) | numpy, scipy | <1 min |
| `07_q1d_gga_profiles.py` | SI Table S8 (quasi-1D LDA/GGA on seven axial profiles) | numpy, scipy | <1 min |
| `08_local_hybrid_wires.py` | SI Table S9 (local hybrids, N = 4, 6) | numpy, scipy, **pyscf** | <1 min |
| `09_literature_validation.py` | validation of the sheet-limit construction against Constantin (2009, a_x = 0.5217) and Kaplan, Wagle, Perdew (2018, SCAN slab limit -1.671 Ha, exact -0.1501 Ha) | numpy, scipy | <1 s |

Python: `python3` for 01, 02, 04–07; an environment with PySCF (here `~/miniconda3/envs/anigto/bin/python`) for 03
and 08.  `../refresh_all.sh` runs all eight and builds the manuscript.

Verified 2026-11-05 against the earlier modular pipeline (now in `../legacy/`): all printed table values identical;
raw CSVs identical except timing columns and 1e-15–1e-16 Ha roundoff (aniGTO ERI now from aniGTO's Python `IEE`
instead of its Cython ERI; SCE extrapolation).  One SI cell changed by rounding: N = 6 LH-SCAN at lam = 1e3 is
-5.585 mHa, now printed -5.58 (the hand-typed table had -5.59).
