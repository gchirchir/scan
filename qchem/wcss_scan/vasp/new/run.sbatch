#!/bin/bash -l
#SBATCH --job-name=surf22
#SBATCH --output=logs/surf22_%A_%a.out
set -euo pipefail
module purge
export PAGER=cat
module load PySCF/2.7.0-foss-2023a
ROOT=$HOME/pyscf/gpu/scan/vasp/new
NT=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
export PYSCF_MAX_MEMORY=$(( ${SLURM_MEM_PER_NODE:-96000} * 80 / 100 ))
LINE=$(grep -v '^#' "$ROOT/tasks.txt" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read -r NAME SYSTEM A LAYERS KXY KZ CUTOFF <<< "$LINE"
mkdir -p "$ROOT/logs" "$ROOT/code/results" "$ROOT/code/figures"
test ! -s "$ROOT/code/results/surface22_${NAME}.json" || { echo "already complete: $NAME"; exit 0; }
echo "task=$SLURM_ARRAY_TASK_ID name=$NAME host=$(hostname) date=$(date -Is)"
python -c 'import pyscf, scipy; print("PySCF",pyscf.__version__,"SciPy",scipy.__version__)'
python -u "$ROOT/code/22_surface_density_dcscan.py" run \
  --system "$SYSTEM" --a "$A" --layers "$LAYERS" --kxy "$KXY" --kz "$KZ" --cutoff "$CUTOFF"
test -s "$ROOT/code/results/surface22_${NAME}.json"
echo "done name=$NAME date=$(date -Is)"
