#!/bin/bash -l
#SBATCH --job-name=trap20
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0
#SBATCH --output=logs/trap20_%A_%a.out
set -o pipefail
export MODULES_PAGER=cat
module load PySCF/2.7.0-foss-2023a >/tmp/pyscf_module_load_${SLURM_JOB_ID}.log 2>&1
python -c "import pyscf; print('PySCF', pyscf.__version__)"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMBA_NUM_THREADS=16
ND=40; OUT=nd40_ec
cd $HOME/pyscf/gpu/scan/trap20_20261118
mkdir -p $OUT && cp trap_ac_results.csv $OUT/
SCAN_RESULTS=$PWD/$OUT python3 -u 20_trap_dc_exchange.py --nmax-dir $ND --lams 1 2 5 10 20 50 100 1000 10000 2>&1 | grep -v -i 'warn'
