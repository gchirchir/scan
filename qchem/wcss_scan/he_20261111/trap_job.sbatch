#!/usr/bin/env bash
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
# Wire/sheet traps for the given lam values (exact references, AC ingredients, basis convergence); no He series.
set -eo pipefail
TAG=$1; shift
cd "$SLURM_SUBMIT_DIR"
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
source ~/.bashrc >/dev/null 2>&1 || true
export MODULES_PAGER=cat
module load PySCF/2.7.0-foss-2023a >/tmp/pyscf_module_load_${SLURM_JOB_ID}.log 2>&1
python -c "import pyscf, numpy, scipy, numba; print('PySCF', pyscf.__version__, 'numba', numba.__version__)"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export SCAN_RESULTS="$SLURM_SUBMIT_DIR/trap_runs/$TAG"
mkdir -p "$SCAN_RESULTS"
python3 12_trap_ac_benchmark_traps.py --no-he --lams "$@" 2>&1 | tee "$SCAN_RESULTS/run.log"
sha256sum 12_trap_ac_benchmark_traps.py > "$SCAN_RESULTS/script.sha256"
