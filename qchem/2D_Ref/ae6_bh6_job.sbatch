#!/usr/bin/env bash
#SBATCH --job-name=ae6bh6
#SBATCH --partition=lem-cpu-short
#SBATCH --array=0-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
# One AE6/BH6 species per array task: r2SCAN SCF (def2-QZVPP, grid 6) + post-SCF scalar Q2D -> species/<NAME>.json
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
source ~/.bashrc >/dev/null 2>&1 || true
export MODULES_PAGER=cat
module load PySCF/2.7.0-foss-2023a >/dev/null 2>&1
SPECIES=(C4H8 C2H2O2 C3H4 TS_OH_CH4 SiH4 S2 SiO TS_H_H2S H2S HS TS_H_OH CH4 CH3 H2O OH H2 S Si O C H)
NAME=${SPECIES[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYSCF_MAX_MEMORY=16000
python3 -c "import pyscf, numpy; print('PySCF', pyscf.__version__, 'numpy', numpy.__version__)"
if [ -s "species/$NAME.json" ]; then echo "$NAME done already"; exit 0; fi
python3 reproduce_ae6_bh6.py species "$NAME"
