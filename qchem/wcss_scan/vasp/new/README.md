# WCSS metal (111) extension

Production root: `/home/gabchi5762/pyscf/gpu/scan/vasp/new`.

All metals use the Al production protocol: 8 layers, 8x8x1 slab mesh,
8x8x4 bulk mesh, 30 Ha cutoff, PBEsol, and 0.05 eV Gaussian smearing.
KRKS is used for even-valence cells. Cu, Rh, and Au one-atom bulk cells use
KUKS with `fix_spin=False`, which applies one common chemical potential to
both spin channels; all reported density fields are spin summed.

Enter WCSS only with `./gab_wcss.sh`, obtain a compute node with
`sub-interactive-lem-cpu`, and verify PySCF 2.7.0 before submission.

    ./submit.sh pilot          # Pt production validation
    ./submit.sh even           # Pd and Ti (validated KRKS path)
    ./submit.sh odd-pilot      # Cu KUKS production validation
    ./submit.sh odd-remaining  # Rh and Au, only after Cu passes

The submitter skips any task with a complete JSON and any already queued
array index. A result is complete only when JSON, bulk NPZ, and slab NPZ all
exist and its log terminates normally.

## Simple per-metal folders

Each metal also has its own directory at the WCSS root: `Al/`, `Pt/`, `Cu/`,
`Rh/`, `Au/`, `Pd/`, and `Ti/`.  Each contains an executable `input.py` and
`slurm.sh`.  After loading PySCF on an allocated compute node, run directly:

    cd /home/gabchi5762/pyscf/gpu/scan/vasp/new/Rh
    python3 input.py

Or submit the same calculation to Slurm:

    cd /home/gabchi5762/pyscf/gpu/scan/vasp/new/Rh
    sbatch slurm.sh

Replace `Rh` with the desired metal. Do not run or submit a metal already
listed by `squeue -u gabchi5762 -n surf22`. The input wrapper skips an already
complete JSON, but it cannot detect a second copy that is currently running.
