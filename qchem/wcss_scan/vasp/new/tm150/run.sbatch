#!/bin/bash -l
#SBATCH --job-name=tm150
#SBATCH --output=logs/tm150_%A_%a.out
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
set -euo pipefail
module purge; export PAGER=cat; module load PySCF/2.7.0-foss-2023a
cd "$SLURM_SUBMIT_DIR"
NT=${SLURM_CPUS_PER_TASK:-16}; export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
export PYSCF_MAX_MEMORY=$(( ${SLURM_MEM_PER_NODE:-150000} * 80 / 100 ))
export TMPDIR=/lustre/pd03/hpc-gabchi5762-1768308766/tm150_tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; export PYSCF_TMPDIR=$TMPDIR
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
read -r SYS A CUT BL KZ <<< "$(grep -v '^#' ${TASKS:-tasks.txt} | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
echo "task=$SLURM_ARRAY_TASK_ID $SYS $A host=$(hostname) date=$(date -Is)"
python -u 22_surface_density_dcscan.py run --system "$SYS" --a "$A" --layers 8 --bulk-layers "${BL:-3}" --vacuum 14 \
  --kxy 8 --kz "${KZ:-4}" --cutoff "${CUT:-150}" --sigma 0.05 --kerker-alpha 0.3 --kerker-q0 0.8 --kerker-hist 12 --max-cycle 200
echo "done date=$(date -Is)"
