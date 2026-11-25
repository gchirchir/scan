#!/usr/bin/env bash
# Submit the 30 He-series jobs (iso lam=1; wire, sheet lam=2; Z = 1..10) unless an equivalent job is queued.
set -euo pipefail
cd "$(dirname "$0")"
queued=$(squeue -u "$USER" -h -o '%j')
for spec in "iso 1" "wire 2" "sheet 2"; do
  set -- $spec
  for Z in 1 2 3 4 5 6 7 8 9 10; do
    name="scanhe_${1}_${2}_Z${Z}"
    if grep -qx "$name" <<<"$queued"; then echo "skip $name (queued)"; continue; fi
    if [ -s "runs/${1}_${2}_Z${Z}/trap_ac_results.csv" ] && grep -q "\] done" "runs/${1}_${2}_Z${Z}/run.log" 2>/dev/null; then echo "skip $name (done)"; continue; fi
    sbatch -J "$name" he_job.sbatch "$1" "$2" "$Z"
  done
done
