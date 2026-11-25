#!/bin/bash
set -euo pipefail
ROOT=$HOME/pyscf/gpu/scan/vasp/new
cd "$ROOT"
mkdir -p logs code/results code/figures
stage=${1:-pilot}
case "$stage" in
  pilot) candidates="0" ;;
  even) candidates="4 5" ;;
  odd-pilot) candidates="1" ;;
  odd-remaining) candidates="2 3" ;;
  remaining) candidates="1 2 3 4 5" ;;
  all) candidates="0 1 2 3 4 5" ;;
  *) echo "usage: ./submit.sh [pilot|even|odd-pilot|odd-remaining|remaining|all]" >&2; exit 2 ;;
esac
queued=$(squeue -u gabchi5762 -h -r -n surf22 -o '%K' | sort -u)
ids=""
for i in $candidates; do
  name=$(grep -v '^#' tasks.txt | sed -n "$((i+1))p" | awk '{print $1}')
  if [ ! -s "code/results/surface22_${name}.json" ] && ! grep -qx "$i" <<< "$queued"; then
    ids+="$i,"
  fi
done
ids=${ids%,}
test -n "$ids" || { echo "No missing unqueued $stage tasks"; exit 0; }
echo "Submitting $stage indices $ids"
sbatch -p lem-cpu-short -c 16 --mem=96G -t 3-00:00:00 --array="$ids" run.sbatch
