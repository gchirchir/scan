#!/bin/bash -l
# W=0 / W=1 limit checks of the scalar Q2D functional on aniGTO sheets and wires (N = 2, 4, 6).
# Usage: sbatch --export=ALL,M=200 run_anigto.sbatch     (M = grid nodes per direction)
#SBATCH -J anigto_lim
#SBATCH -p lem-cpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o anigto_%j.out
cd "$SLURM_SUBMIT_DIR" || exit 1
M=${M:-200}
export MODULES_PAGER= LMOD_PAGER=none
module purge >/dev/null 2>&1; module load PySCF/2.7.0-foss-2023a >/dev/null 2>&1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -u anigto_limits.py --m $M --lams 1,10,100,1000,10000 --out anigto_limits_m${M}.csv
