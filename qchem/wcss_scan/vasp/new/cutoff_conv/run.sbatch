#!/bin/bash -l
#SBATCH --job-name=surfconv
#SBATCH --output=logs/conv_%A_%a.out
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
set -euo pipefail
module purge
export PAGER=cat
module load PySCF/2.7.0-foss-2023a
cd "$SLURM_SUBMIT_DIR"
NT=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
export PYSCF_MAX_MEMORY=$(( ${SLURM_MEM_PER_NODE:-120000} * 80 / 100 ))
export TMPDIR=/lustre/pd03/hpc-gabchi5762-1768308766/surfconv_tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYSCF_TMPDIR=$TMPDIR
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
LINE=$(grep -v '^#' ${TASKS:-tasks.txt} | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r SYS A KIND L KXY KZ CUT DAMP SHIFT <<< "$LINE"
echo "task=$SLURM_ARRAY_TASK_ID $LINE host=$(hostname) date=$(date -Is)"
python -u cutoff_conv.py --system "$SYS" --a "$A" --kind "$KIND" --layers "$L" --kxy "$KXY" --kz "$KZ" \
  --cutoff "$CUT" --damp "$DAMP" --shift "$SHIFT"
echo "done date=$(date -Is)"
