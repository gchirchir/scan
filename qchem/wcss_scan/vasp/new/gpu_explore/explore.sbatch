#!/bin/bash -l
#SBATCH --job-name=gpuexplore
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --output=explore_%j.out
echo "host $(hostname)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== modules"; module -t avail 2>&1 | grep -iE "^cuda/|^cupy|^pyscf|gpu4pyscf|^cutensor|^jax|^nccl|^cudnn|^magma|^python/" | sort | head -40
echo "=== PySCF module contents"; module purge; module load PySCF/2.7.0-foss-2023a 2>&1 | tail -2; python -c "import pyscf,sys;print(pyscf.__version__,sys.version)"; python -c "import cupy;print(\"cupy\",cupy.__version__)" 2>&1 | tail -1; python -c "import gpu4pyscf;print(\"gpu4pyscf\",gpu4pyscf.__version__)" 2>&1 | tail -1
for e in ~/envs/pyscf_gpu ~/envs/pyscf214 ~/envs/jaxsce_env; do echo "=== $e"; $e/bin/python -c "import gpu4pyscf;print(gpu4pyscf.__version__)" 2>&1 | tail -1; $e/bin/python -c "import cupy;print(\"cupy\",cupy.__version__)" 2>&1 | tail -1; done
echo "=== pip reachability"; timeout 20 python -m pip download --no-deps -d /tmp/x gpu4pyscf-cuda12x==1.4.0 2>&1 | tail -2
