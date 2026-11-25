#!/usr/bin/env bash
# Recompute (dyall-v4z, grid 9) every species whose JSON lacks the proposed W_new (key e_newJ_SS); then assemble.
cd "$(dirname "$0")"
while ! grep -q BH6_DONE run_bh6_pbe_g9.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=8 PYSCF_MAX_MEMORY=6000 AE6_BASIS=dyall-v4z AE6_GRID=9
for s in H C O Si S SiH4 SiO S2 C3H4 C2H2O2 C4H8 OH CH4 CH3 H2O H2 H2S HS TS_OH_CH4 TS_H_OH TS_H_H2S; do
  f=species_new_dyall-v4z_g9/$s.json
  if [ -s "$f" ] && grep -q e_newJ_SS "$f"; then continue; fi
  python3 -u run_pbe_new.py species $s 2>&1 | grep -v Warn | tail -1
done
python3 run_pbe_new.py assemble 2>&1
echo NEWJ_DONE
