#!/bin/bash -l
#SBATCH --job-name=surfscf
#SBATCH --output=logs/scf_%A_%a.out
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
set -euo pipefail
module purge; export PAGER=cat; module load PySCF/2.7.0-foss-2023a
cd "$SLURM_SUBMIT_DIR"
NT=${SLURM_CPUS_PER_TASK:-16}; export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
export PYSCF_MAX_MEMORY=$(( ${SLURM_MEM_PER_NODE:-120000} * 80 / 100 ))
export TMPDIR=/lustre/pd03/hpc-gabchi5762-1768308766/surfconv_tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; export PYSCF_TMPDIR=$TMPDIR
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
LINE=$(grep -v "^#" ${TASKS:-tasks_scf.txt} | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r LAB SYS A L KXY CUT SIG DAMP SHIFT DS GUESS DSP MAXC PA PH KR KQ <<< "$LINE"
echo "task=$SLURM_ARRAY_TASK_ID $LINE host=$(hostname) date=$(date -Is)"
python -u cutoff_conv.py --system "$SYS" --a "$A" --kind slab --layers "$L" --kxy "$KXY" --cutoff "$CUT" --sigma "$SIG" \
  --damp "$DAMP" --shift "$SHIFT" --diis-start "$DS" --guess "$GUESS" --diis-space "$DSP" --label "$LAB" --max-cycle "${MAXC:-200}" --pulay-alpha "${PA:-0}" --pulay-hist "${PH:-8}" --kerker-rho "${KR:-0}" --kerker-q0 "${KQ:-0.8}"
echo "done date=$(date -Is)"
