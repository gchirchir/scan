#!/bin/bash -l
# Merge check: scalar-Q2D (2D_Ref/main.pdf) on the jcp_main Gaussian sheets/wires and the common IBM well.
#SBATCH -J q2d_merge
#SBATCH -p lem-cpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o merge_%j.out
cd "$SLURM_SUBMIT_DIR" || exit 1
export MODULES_PAGER= LMOD_PAGER=none
module purge >/dev/null 2>&1; module load PySCF/2.7.0-foss-2023a >/dev/null 2>&1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -c "import pyscf; print('pyscf', pyscf.__version__)"
python3 -u merge_check.py
