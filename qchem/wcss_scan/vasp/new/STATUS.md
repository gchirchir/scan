# Surface-metal production status

Last updated: 2026-11-21, Europe/Warsaw.

WCSS root: `/home/gabchi5762/pyscf/gpu/scan/vasp/new`

Production protocol: 8-layer (111) slab, 8x8x1 slab k mesh, 8x8x4
three-layer bulk k mesh, 30 Ha density cutoff, PBEsol, 0.05 eV Gaussian
smearing, GTH-PBE pseudopotentials, and `gth-dzvp-molopt-sr` basis.

Submitted jobs:

- Pt: `6379650_0` (KRKS production pilot)
- Pd and Ti: `6379651_4`, `6379651_5` (KRKS)
- Cu: first attempt `6379652_1` failed before SCF because the KUKS total-spin
  parity was set for one cell rather than all 256 k points. Corrected targeted
  retry `6379659_1` is the active KUKS common-chemical-potential pilot.
- Rh and Au: not submitted; release with `./submit.sh odd-remaining` only
  after Cu has a terminal successful log, complete JSON/bulk NPZ/slab NPZ,
  finite fields, and electron-count closure.

Driver SHA-256:
`c8a6dad2ba70fb9be02e91988875c62bbc35944a05f351fb3c27c33fd2623885`.

The low-cost Cu/KUKS control converged at 8 Ha and exercised the unrestricted
spin-summed code, but the intentionally coarse grid did not close the electron
count. A 30 Ha one-k-point control did not converge in 50 cycles and is not a
scientific result. The full 8x8x4 Cu job is therefore the production validation
gate; do not report or release Rh/Au until it passes.

Simple launch folders were installed on 2026-11-21 for Al, Pt, Cu, Rh, Au,
Pd, and Ti. Inside any one of these folders, use `python3 input.py` for an
interactive-compute-node run or `sbatch slurm.sh` for a batch run. The active
Pt/Cu/Pd/Ti jobs predate these convenience wrappers and must not be duplicated.
