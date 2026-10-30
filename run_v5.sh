#!/bin/bash
# v5 = long-gap stage: 27-minute clips (16384 frames) with a frozen frame encoder, cue->query gaps 2.5-20 min,
# initialised from the v4 models. Then a larger long-gap eval (24 hour-long streams) so the numbers mean something.
while pgrep -f run_v4b.sh >/dev/null; do sleep 30; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
LONG="--freeze_fenc --alibi --mem_read --mem_src tok --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 3000 --B 2 --T 16384 --min_gap 1500 --max_gap 12000 --lr 3e-4 --warmup 100"
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
best=hnet_v4; [ -f runs/hnet_v4d/ckpt.pt ] && best=$(cat runs/v5_hnet_init 2>/dev/null || echo hnet_v4)
run hnet_v5  --chunkers learned,learned --Ns 4,4 --router tied --init runs/$best/ckpt.pt $LONG
run fixed_v5 --chunkers fixed,fixed --Ns 4,4 --init runs/fixed_v4/ckpt.pt $LONG
$PY eval.py hnet_v5 fixed_v5 hnet_v4 fixed_v4 --suite hour_longgap --n_env 24 > runs/eval_longgap_v5.log 2>&1
$PY eval.py hnet_v5 fixed_v5 --suite hour > runs/eval_hour_v5.log 2>&1
echo V5_DONE
