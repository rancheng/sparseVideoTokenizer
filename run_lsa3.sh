#!/bin/bash
# rerun of the LSA tables after the bucket-eligibility fix (completed blocks searchable, window keys skipped)
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY exp_lsa.py --part stale2 > runs_lsa_stale2.log 2>&1
$PY exp_lsa.py --part stale > runs_lsa_stale.log 2>&1
for n in hnet_v5 hnet_v5_s1 hnet_v5s hnet_v5s_s1 hnet_v5sr; do $PY exp_lsa.py --name $n --part native > runs_lsa_native_$n.log 2>&1; done
$PY exp_lsa.py > runs_lsa_budget.log 2>&1
echo R_DONE
