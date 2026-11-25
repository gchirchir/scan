#!/bin/bash -l
#SBATCH --job-name=surf22
#SBATCH --output=slurm-%j.out
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
set -euo pipefail

module purge
export PAGER=cat
module load PySCF/2.7.0-foss-2023a

cd "$SLURM_SUBMIT_DIR"
NT=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$NT
export MKL_NUM_THREADS=$NT
export OPENBLAS_NUM_THREADS=$NT
export PYSCF_MAX_MEMORY=76000

python3 input.py
