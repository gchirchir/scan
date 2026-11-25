#!/usr/bin/env bash
#SBATCH --job-name=ae6cpu
#SBATCH --partition=lem-cpu-short,lem-cpu-normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# AE6/BH6 species given as arguments (dyall-v4z, grid 9; original q2d_scalar.py), one after another, 16 threads.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc >/dev/null 2>&1 || true
export MODULES_PAGER=cat
module purge >/dev/null 2>&1; module load PySCF/2.7.0-foss-2023a >/dev/null 2>&1
python3 -c "import pyscf; print('PySCF', pyscf.__version__)"
md5sum q2d_scalar.py
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYSCF_MAX_MEMORY=100000
mkdir -p species logs
for s in "$@"; do python3 reproduce_ae6_bh6.py species "$s"; done
