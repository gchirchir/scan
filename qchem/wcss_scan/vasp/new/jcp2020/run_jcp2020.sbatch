#!/bin/bash -l
#SBATCH --job-name=jcp2020
#SBATCH --output=logs/jcp2020_%A_%a.out
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
# One array task = one line of $TASKS (tasks_lattice.txt, tasks_surface.txt or tasks_fig1.txt), e.g.
#   TASKS=tasks_lattice.txt sbatch --array=0-$(( $(wc -l < tasks_lattice.txt) - 1 ))%40 --export=ALL run_jcp2020.sbatch
# Surface tasks read results/lattice_<M>_<F>.json: submit them after the lattice array (--dependency=afterok:<id>).
set -euo pipefail
module purge; export PAGER=cat; module load PySCF/2.7.0-foss-2023a
cd "$SLURM_SUBMIT_DIR"
NT=${SLURM_CPUS_PER_TASK:-16}; export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
export PYSCF_MAX_MEMORY=$(( ${SLURM_MEM_PER_NODE:-180000} * 80 / 100 ))
export TMPDIR=/lustre/pd03/hpc-gabchi5762-1768308766/jcp2020_tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; export PYSCF_TMPDIR=$TMPDIR
mkdir -p "$TMPDIR" logs results; trap 'rm -rf "$TMPDIR"' EXIT
LINE="$(grep -v '^#' "${TASKS:?set TASKS}" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
echo "task=$SLURM_ARRAY_TASK_ID [$LINE] host=$(hostname) date=$(date -Is)"
python -u jcp2020_scf.py $LINE
echo "done date=$(date -Is)"
