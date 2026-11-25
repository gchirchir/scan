# JCP2020_Patra — reproduction of J. Chem. Phys. 152, 151101 (2020), with SCAN and DC-SCAN

Reference: A. Patra, S. Jana, L. A. Constantin, L. Chiodo, P. Samal, "Improved transition metal surface energies
from a generalized gradient approximation developed for quasi two-dimensional systems", J. Chem. Phys. 152, 151101
(2020) (`material/151101_1_online.pdf`).  One standalone script, `jcp2020_scf.py` (numpy, scipy, pyscf; matplotlib for
figures).  Every functional of the paper (LDA, PBE, PBEsol, Q2D-GGA) is solved self-consistently; in addition every
calculation is done with SCAN (self-consistent) and DC-SCAN (DC-SCAN exchange evaluated on the SCAN density).

| Paper item | Reproduced by | Output |
|---|---|---|
| Table I: σ of 21 metals (7 fcc, 7 bcc, 7 hcp) × 3 facets (fcc 100/110/111, bcc 100/110/111, hcp 0001/10-10/11-20) | `surface` | `results/jcp2020_table1.{csv,tex}` (+ SCAN, DC-SCAN columns) |
| Table II: ME/MAE/MARE of the most stable facet and of the Wulff average | `table` (own Wulff construction) | `results/jcp2020_table2.{csv,tex}` |
| Fig. 1: σ vs number of layers, Pt and Ta, Q2D-GGA | `surface --layers N` | `figures/jcp2020_fig1.pdf` (+ DC-SCAN curves) |
| Fig. 2: Δn(z) of fcc(111) Pt/Pd/Rh/Al, bcc(110) W/Na, hcp(0001) Ru/Hf/Mg | profiles saved by `surface` | `figures/jcp2020_fig2.pdf` |
| own lattice constants of each functional | `lattice` (BM3 fit, 9 points ±5 %) | `results/lattice_<M>_<F>.json` (SCAN run also gives DC-SCAN a0) |

Reference data: `reference_jcp2020_table1.csv` (paper Table I) and `reference_jcp2020_table2.csv` (paper Table II).
Validation of the analysis code: the paper's own Table I values put through `wulff_fractions` + `stats` reproduce
the paper's Table II to ±0.01 J/m² (one MARE, hcp PBEsol most-stable, prints 19.93 % in the paper vs 16.9 % from
its own Table I values, while the MAE agrees).

## Protocol differences from the paper
GTH-PBE pseudopotentials + gth-dzvp-molopt-sr Gaussian basis with an FFT cutoff (3d 300 Ha, Al 80 Ha, others 150 Ha) instead
of VASP PAW 800 eV; slabs not relaxed; vacuum 12 Å (paper > 20 Å); 12×12×1 k (paper 16×16×1); hcp c/a fixed at
experiment; Fe and Co spin-polarised with KUKS/DIIS (no DC-SCAN for spin-polarised runs).

Slab/bulk consistency: the bulk reference of every facet is the oriented bulk cell with the slab's in-plane cell and a
lattice vector normal to the surface; the slab atoms are bulk atoms shifted by whole FFT-grid steps and the slab cell
height is a whole number of grid steps, so slab and bulk have identical grid positions and the FFT-grid error
cancels in σ (checked for all nine facet types).

## Run
```bash
PY=$HOME/miniconda3/envs/anigto/bin/python
$PY jcp2020_scf.py geometry --system Ti --facet 10-10        # inspect a cell, no SCF
$PY jcp2020_scf.py tasks                                     # tasks_lattice.txt (120), tasks_surface.txt (330), tasks_fig1.txt (60)
# WCSS (heavy runs only there): copy this folder, then
TASKS=tasks_lattice.txt sbatch --array=0-119%40 run_jcp2020.sbatch
TASKS=tasks_surface.txt sbatch --dependency=afterok:<lattice job> --array=0-329%40 run_jcp2020.sbatch
TASKS=tasks_fig1.txt    sbatch --dependency=afterok:<lattice job> --array=0-<n-1>%40 run_jcp2020.sbatch
# locally after fetching results/*.json, *.npz:
$PY jcp2020_scf.py figures --functional PBEsol && $PY jcp2020_scf.py report   # -> results/jcp2020_report.pdf
```
Extending: add a metal to `METALS` (structure, experimental a, c/a), or a facet to `FACETS`/`MILLER3`; the slab
builder works for any Miller plane of fcc, bcc and hcp.
