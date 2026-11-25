#!/usr/bin/env bash
# Basis-set benchmark of non-SCF r2SCAN / scalar AE6 on PBE orbitals, grid level 6 (cheap first; diffuse: D, aug-).
# Waits for the default dyall-v4z/grid-9 run to finish first (memory is limited on this machine).
cd "$(dirname "$0")"
while pgrep -f "run_pbe_nonscf.py species" >/dev/null || ! grep -q ALL_DONE run.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=12 PYSCF_MAX_MEMORY=3500 AE6_GRID=6
for b in def2-tzvp aug-cc-pvtz def2-qzvpp def2-qzvppd dyall-v4z aug-cc-pvqz; do
  export AE6_BASIS=$b
  echo "=== $b grid $AE6_GRID $(date +%T)"
  for s in H C O Si S SiH4 SiO S2 C3H4 C2H2O2 C4H8; do
    [ -s "species_${b}_g6/$s.json" ] && continue
    python3 run_pbe_nonscf.py species $s 2>&1 | tail -2
  done
  python3 run_pbe_nonscf.py assemble 2>&1 | tail -4
done
echo SCAN_DONE
