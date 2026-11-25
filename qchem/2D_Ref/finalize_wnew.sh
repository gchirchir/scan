#!/usr/bin/env bash
# Wait for all W_new jobs, then run the validation and rebuild every table (dyall-v4z, grid 9 throughout).
cd "$(dirname "$0")"
until grep -q SCHEME_DONE make_new_scheme_tables.log 2>/dev/null && grep -q NEWJ_DONE pbe_nonscf_ae6/fill_newJ_g9.log 2>/dev/null \
      && grep -q ALL_DONE rerun_dyall_scf.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=12
echo "== validation V1, V2, V5, V7"; (cd w_gate_variant && python3 -u validate_new.py 2>&1 | grep -v Warn | tee validate_new.log)
echo "== tables"
python3 make_table_orig_vs_new.py 2>&1 | grep -v Warn | tail -2
python3 make_table_all.py 2>&1 | grep -v Warn | tail -3
python3 make_table_validation.py 2>&1 | tail -30
python3 make_table_new_pbe.py 2>&1 | tail -40
python3 make_jana_new_tables.py >/dev/null 2>&1; python3 make_table_new_xc.py >/dev/null 2>&1
echo FINAL_DONE
