#!/bin/bash
# Release held surface / fig1 array tasks whose lattice constant (results/lattice_<M>_<F>.json) exists.
cd ~/pyscf/gpu/scan/vasp/new/jcp2020
n=0
for pair in jcp20surf:tasks_surface.txt jcp20fig1:tasks_fig1.txt; do
  name=${pair%%:*}; file=${pair#*:}
  squeue -h -r -u $USER -n $name -t PENDING -o "%i %r" | while read id reason; do
    [ "$reason" = "JobHeldUser" ] || continue
    idx=${id##*_}
    line=$(grep -v "^#" $file | sed -n "$((idx + 1))p")
    sys=$(awk "{for(i=1;i<=NF;i++) if(\$i==\"--system\") print \$(i+1)}" <<< "$line")
    fun=$(awk "{for(i=1;i<=NF;i++) if(\$i==\"--functional\") print \$(i+1)}" <<< "$line")
    [ "$fun" = "DC-SCAN" ] && fun=SCAN
    if [ -f results/lattice_${sys}_${fun}.json ] && ! grep -q "\"minimum_near_edge\": true" results/lattice_${sys}_${fun}.json; then scontrol release $id && echo "released $id ($sys $fun)"; fi
  done
done
