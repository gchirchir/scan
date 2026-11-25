#!/bin/bash
# cancel running tasks whose SCF has stalled (find_stalled.py) and resubmit the same task line with the guarded code
cd ~/pyscf/gpu/scan/vasp/new/jcp2020
python3 find_stalled.py | while read tag jid name rest; do
  [ "$tag" = "STALLED" ] || continue
  idx=${jid##*_}
  case $name in jcp20surf) T=tasks_surface.txt;; jcp20lat) T=tasks_lattice.txt;; jcp20fig1) T=tasks_fig1.txt;; jcp20relat) T=tasks_relat.txt;; *) continue;; esac
  scancel $jid && N=$(TASKS=$T sbatch --parsable --job-name=$name --array=$idx --export=ALL run_jcp2020.sbatch) && echo "restarted stalled $jid ($rest) -> $N"
done
