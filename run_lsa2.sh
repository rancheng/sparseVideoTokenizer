#!/bin/bash
# seed check: is native sparse training (v5s) reproducibly better than dense (v5)?
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LONG="--freeze_fenc --alibi --mem_read --mem_src tok --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 6000 --B 1 --T 16384 --min_gap 1500 --max_gap 12000 --lr 3e-4 --warmup 100"
run() { n=$1; shift; mkdir -p runs/$n; [ -f runs/$n/ckpt.pt ] || $PY train.py --name $n --chunkers learned,learned --Ns 4,4 --router tied --init runs/hnet_v4/ckpt.pt $LONG "$@" > runs/$n/stdout.log 2>&1; }
run hnet_v5_s1 --seed 1
run hnet_v5s_s1 --sparse_read 32 --seed 1
for n in hnet_v5_s1 hnet_v5s_s1; do $PY exp_lsa.py --name $n --part native > runs_lsa_native_$n.log 2>&1; done
$PY eval.py hnet_v5s hnet_v5s_s1 hnet_v5_s1 --suite hour_longgap --n_env 24 > runs/eval_longgap_v5s.log 2>&1
echo S2_DONE
