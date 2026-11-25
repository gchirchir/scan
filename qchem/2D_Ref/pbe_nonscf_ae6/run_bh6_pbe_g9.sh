#!/usr/bin/env bash
# BH6 species on PBE orbitals (dyall-v4z, grid 9) after the AE6 grid-9 run has finished; then assemble AE6 + BH6.
cd "$(dirname "$0")"
while ! grep -q ALL_DONE run_pbe_new_g9.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=8 PYSCF_MAX_MEMORY=6000 AE6_BASIS=dyall-v4z AE6_GRID=9
for s in OH CH4 CH3 H2O H2 H2S HS TS_OH_CH4 TS_H_OH TS_H_H2S; do
  [ -s species_new_dyall-v4z_g9/$s.json ] && continue
  python3 -u run_pbe_new.py species $s 2>&1 | grep -v Warn | tail -1
done
python3 run_pbe_new.py assemble 2>&1
echo BH6_DONE
