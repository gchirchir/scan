#!/usr/bin/env bash
#SBATCH --job-name=ae6bh6g
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# AE6/BH6 species given as arguments, on a lem-gpu node (CPU PySCF; GPU unused; max 16 cpus per GPU), 2 at a time x 8 threads.
# Submit: sbatch ae6_bh6_gpunode.sbatch C4H8 SiH4 ...
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
source ~/.bashrc >/dev/null 2>&1 || true
export MODULES_PAGER=cat
module load PySCF/2.7.0-foss-2023a >/dev/null 2>&1
python3 -c "import pyscf, numpy; print('PySCF', pyscf.__version__, 'numpy', numpy.__version__)"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYSCF_MAX_MEMORY=50000
mkdir -p species logs
run() {
  [ -s "species/$1.json" ] && { echo "$1 done already"; return 0; }
  python3 reproduce_ae6_bh6.py species "$1" > "logs/sp_$1.log" 2>&1 && tail -1 "logs/sp_$1.log" || echo "$1 FAILED (logs/sp_$1.log)"
}
export -f run
printf '%s\n' "$@" | xargs -P 2 -I{} bash -c 'run {}'
echo "all done"; ls species
