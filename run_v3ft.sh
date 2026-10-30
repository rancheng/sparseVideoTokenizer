#!/bin/bash
# continue the hierarchy and the dense control on long gaps (100-1500 frames), same settings for both
while kill -0 627836 2>/dev/null; do sleep 30; done
PY=.venv/bin/python
COMMON="--alibi --mem_read --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 8000 --min_gap 100 --max_gap 1500 --lr 3e-4 --warmup 100"
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
run hnet_v3_ft --chunkers learned,learned --Ns 4,4 --router tied --init runs/hnet_v3/ckpt.pt $COMMON
run iso_v3_ft  --chunkers all --Ns 1 --dims 256,512 --init runs/iso_v3/ckpt.pt $COMMON
$PY eval.py hnet_v3_ft iso_v3_ft --suite hour > runs/eval_hour_v3ft.log 2>&1
$PY eval.py hnet_v3_ft iso_v3_ft --suite hour_longgap > runs/eval_longgap_v3ft.log 2>&1
echo V3FT_DONE
