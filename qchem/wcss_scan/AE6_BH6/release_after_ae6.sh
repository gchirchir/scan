#!/usr/bin/env bash
# Release the jobs held so that AE6/BH6 (ae6g1-3) get the next free slots; runs until all ae6g jobs left PENDING.
cd ~/pyscf/gpu/scan/AE6_BH6
for i in $(seq 1 720); do
  pend=$(squeue -u $USER -h -t PD -o "%j" | grep -c "^ae6g")
  [ "$pend" -eq 0 ] && break
  sleep 60
done
for j in $(cat held_for_ae6.txt); do scontrol release $j; done
echo "$(date) released: $(tr "\n" " " < held_for_ae6.txt) (ae6 pending left: $pend)" >> release_after_ae6.log
