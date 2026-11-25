#!/usr/bin/env bash
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
# One He-like ion in a trap: args FAM LAM Z.  Results -> runs/<FAM>_<LAM>_Z<Z>/
set -eo pipefail
FAM=$1; LAM=$2; Z=$3
cd "$SLURM_SUBMIT_DIR"
# batch shells do not define module: initialise it as in acm/anigto/G0W0100/aditi/slurm.sh
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
source ~/.bashrc >/dev/null 2>&1 || true
export MODULES_PAGER=cat
module load PySCF/2.7.0-foss-2023a >/tmp/pyscf_module_load_${SLURM_JOB_ID}.log 2>&1
python -c "import pyscf, numpy, scipy; print('PySCF', pyscf.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
python -c "import numba; print('numba', numba.__version__)" || echo "numba missing: numpy fallback for ERIs"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export SCAN_RESULTS="$SLURM_SUBMIT_DIR/runs/${FAM}_${LAM}_Z${Z}"
mkdir -p "$SCAN_RESULTS"
python3 12_trap_ac_benchmark.py --he-only --he-fams "$FAM" --he-lams "$LAM" --zs "$Z" 2>&1 | tee "$SCAN_RESULTS/run.log"
sha256sum 12_trap_ac_benchmark.py > "$SCAN_RESULTS/script.sha256"
