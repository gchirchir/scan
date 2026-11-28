# codes/ — standalone scripts (one file = one complete calculation)

Every number and figure of `../manuscript/main.pdf` and `si.pdf` comes from one of these production files.  Each file
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
| `11_table_11_validation.py` | standalone FCI audit of all exact-reference rows: two-electron sheet/wire, wire asymptotic, and multiorbital N=4,6 tables | numpy, scipy, **pyscf** | <1 min |
| `10_fci_gaussian_validation.py` | FCI check of the Gaussian exchange reference for sheet and wire traps, plus Libxc SCAN/r2SCAN and DC-SCAN/DC-r2SCAN exchange (2026-11-17); writes `fci_gaussian_validation.csv`, `tab_fci_*_exchange_rows.tex`, `tab_r2scan_*_rows.tex` | numpy, scipy, **pyscf** | ~2 s |
| `14_constantin2008_benchmark.py` | reproduces Constantin, PRB 78, 155106 (2008): Eqs. (1), (12), (13), (23) and Figs. 1, 3, 4, 5 (IBM quantum well, scaled H atom; LSDA/PBE via Libxc, GGA+2D/MGGA+2D/Eq. 22 coded); also runs the code-01 SCAN/DC-SCAN functions (verbatim copy) on the well: SCAN -> -1.670597 (Kaplan -1.671), DC-SCAN -> (A2D/a_x) eps_2D = -0.187865 | numpy, scipy, matplotlib, **pyscf** | ~2-4 min |
| `15_constantin2008_figure_compare.py` | digitizes the published curves from the vector figures of `subrata/PhysRevB.78.155106.pdf` (pdftocairo) and compares them with code 14 (rms/max per curve, overlay figure) | numpy, pandas, matplotlib, poppler | <10 s |
| `16_subrata_paper_figures.py` | the four figures of the subrata JCP paper (geometry, branches, comparison, wire_slope) from the results of codes 01/02 (functions copied from 01) -> `figures/subrata_paper/` | numpy, pandas, matplotlib | <5 s |
| `17_ibm_quasi_1d_2d.py` | DC-SCAN / DC-r2SCAN vs exact single-subband exchange on infinite-barrier systems: quasi-1D cylindrical wire (computed here, exact V(qR) route) + quasi-2D well (read from code 14 CSV); D-descriptor diagnostics (D=2 axis -> 1 wall; forced-D=2 slope) -> `results/ibm_wire_dcscan.csv`, `tab_ibm_rows.tex`, `figures/ibm_quasi_1d_2d.pdf`; run after 14 | numpy, scipy, matplotlib, pyscf (anigto env) | ~50 s |
| `18_urso2021_prl2025_ibm.py` | Urso & Constantin EPJB 94, 147 (2021) [PDF watermarked RETRACTED] Figs. 3, 4, 7, 8 recomputed (exact single-subband exchange, MVS, SCAN, Q2D, Q1D) + DC-SCAN/DC-r2SCAN, overlaid on the digitized vector curves; PRL 134, 256401 (2025) slab asymptote: exact eps_x(z) of single-subband slabs -> -1/(2z) -> `results/urso2021_*.csv`, `prl2025_*.csv`, `figures/urso2021_fig{3,4,7,8}.pdf`, `prl2025_slab_epsx.pdf` | numpy, scipy, matplotlib, pyscf (anigto env), pdftocairo | ~4 min |
| `19_fci_ibm_kernel_check.py` | FCI exchange of the Gaussian sheet/wire (code 10 CSV) recomputed with the IBM exact-exchange kernels (slab form factor F(q), transverse Hankel kernel V(k)); max diff 1.5e-9 Ha -> `results/fci_ibm_kernel_check.csv`, `tab_fci_ibm_kernel_rows.tex`, `figures/fci_ibm_kernel_check.pdf` (main Fig. 2) | numpy, scipy, matplotlib | seconds |
| `12_trap_ac_benchmark.py` | adiabatic-connection benchmark of correlated two-electron wire (lam^2,lam^2,1) and sheet (1,1,lam^2) traps, lam = 1, 2, 5, 10 (aniGTO tables I-VI analogue) | numpy, scipy, **pyscf** | minutes-hours |
| `13_trap_ip_gap_gw.py` | IPs, EAs, frontier orbitals, N = 2 Kohn-Sham inversion and G0W0 of the same traps (functions copied from 12) | numpy, scipy, **pyscf** | minutes-hours |
| `20_trap_dc_exchange.py` | SCAN and dimensional-crossover exchange on the CORRELATED sheet/wire traps (needs `results/trap_ac_results.csv` of 12) | numpy, scipy, **pyscf** | minutes |
| `21_trap_gap_gcf.py` | Kohn-Sham gaps of the traps, PBE->HSE06 gap-correction (GCF) fits, DC-PBE/DC-SCAN/DC-r2SCAN/DC-HSE06 gaps | numpy, scipy, **pyscf** | minutes-hours |
| `22_surface_density_dcscan.py` | metal (111) slab surface densities and fixed-density surface XC energies incl. DC-SCAN (Chiodo et al. PRL 108, 126402; Patra et al. JCP 152, 151101); reads `reference_*.csv` and VASP outputs | numpy, scipy, matplotlib | minutes |
| `23_vasp_surface_inputs.py` | VASP bulk/slab inputs for PRL 108, 126402 (2012) and JCP 152, 151101 (2020): LDA/PBE/PBEsol/Q2D-GGA, 7 PRL (111) systems and 21 metals x 3 JCP facets; outputs under `../material/vasp/surface_q2d/` | numpy; VASP 6.3+ with Libxc for Q2D production | seconds |
| `24_vasp_surface_collect.py` | extracts functional-specific optimized bulk cells and computes converged surface energies from script-23 VASP output trees | numpy | seconds |

Python: `python3` for 01, 02, 04–07; an environment with PySCF (here `~/miniconda3/envs/anigto/bin/python`) for 03,
08, and 10.  `../refresh_all.sh` runs the nine production generators and builds the manuscript.

Script 10 is an independent validation and is intentionally not part of
`refresh_all.sh`: it runs a one-spatial-orbital, two-electron singlet FCI
calculation and checks `E_FCI = 2 h00 + J` and `E_x = -J` against Eq. (7)
and the closed-form sheet/wire references.  Run it from the project root with

    $HOME/miniconda3/envs/anigto/bin/python codes/10_fci_gaussian_validation.py

Script 11 is also independent of the manuscript and generated table files. It
extends the FCI audit to every exact-reference row in the main/SI tables. Its
multiorbital check uses exact anisotropic Coulomb integrals in a PySCF FCI
repulsion Hamiltonian for the fixed occupied determinant; SCAN, DC-SCAN, and
local-hybrid columns are functional values and are not FCI observables.

Figure style: codes 14-17 and 19 share an identical publication-style block (pub_style etc.; print-size fonts, fixed colour per method, .pdf/.eps/.png).

Verified 2026-11-05 against the earlier modular pipeline (now in `../legacy/`): all printed table values identical;
raw CSVs identical except timing columns and 1e-15–1e-16 Ha roundoff (aniGTO ERI now from aniGTO's Python `IEE`
instead of its Cython ERI; SCE extrapolation).  One SI cell changed by rounding: N = 6 LH-SCAN at lam = 1e3 is
-5.585 mHa, now printed -5.58 (the hand-typed table had -5.59).

Not included in the public repository: the publisher PDFs digitized by 15 (`subrata/PhysRevB.78.155106.pdf`) and
18 (`s10051-021-00159-y.pdf`) -- obtain them from the journals and put them at the paths named in the scripts.
