#!/bin/bash -l
#SBATCH --job-name=trap20
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --array=2
#SBATCH --output=logs/trap20_%A_%a.out
set -o pipefail
export MODULES_PAGER=cat
module load PySCF/2.7.0-foss-2023a >/tmp/pyscf_module_load_${SLURM_JOB_ID}.log 2>&1
python -c "import pyscf; print('PySCF', pyscf.__version__)"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMBA_NUM_THREADS=16
NDS=(20 40 60); ND=${NDS[$SLURM_ARRAY_TASK_ID]}
cd $HOME/pyscf/gpu/scan/trap20_20261118
mkdir -p nd$ND && cp trap_ac_results.csv nd$ND/
SCAN_RESULTS=$PWD/nd$ND python3 -u 20_trap_dc_exchange.py --nmax-dir $ND 2>&1 | grep -v -i 'warn'
