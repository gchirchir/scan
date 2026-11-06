# scan — Dimensional crossover in SCAN exchange (DC-SCAN)

Source, compiled PDFs and code of

**Probing Dimensional Crossover in SCAN Exchange with Exact Quasi-One-Dimensional Wires and Quasi-Two-Dimensional Sheets**
G. Chirchir, S. Jana, S. Śmiga (Nicolaus Copernicus University in Toruń)

| File | Content |
|---|---|
| `main.tex`, `main.pdf` | manuscript (Appendix A: all DC-SCAN equations A1–A44) |
| `si.tex`, `si.pdf` | Supporting Information |
| `eri_note.tex`, `eri_note.pdf` | reviewer support document: exact aniGTO electron-repulsion integrals at extreme anisotropy with the logarithmic variable s = ln u in the Boys-type integral |
| `figures/` | figures used by the three documents |
| `codes/` | nine standalone Python scripts `01`–`09` that produce every number and figure, with `results/` and `figures/` |

All results are fixed-density and exchange-only. The orbitals are exact (analytic) eigenfunctions of the
anisotropic harmonic-trap Hamiltonian; no Kohn–Sham/SCF equation is solved.

## Reproduce
Each script is a single file that defines every function it uses:

    cd codes
    python3 04_exact_vs_proposed.py     # writes results/ and figures/

`python3` with numpy, scipy, pandas, matplotlib for scripts 01, 02, 04–07 and 09; an environment with PySCF (libxc)
for 03 and 08. See `codes/README.md` for what each script produces and its run time.

## Compile
Each `.tex` compiles on its own with pdfLaTeX (bibliography embedded, no BibTeX), e.g. on Overleaf
(see `README.txt`).
