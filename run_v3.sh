#!/bin/bash
# v3 = v2 fixes + balanced loss (w_pix 10; pixel loss was drowning the sparse decision signal) + gap curriculum.
PY=.venv/bin/python
COMMON="--alibi --mem_read --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 8000 --curriculum 0:5:60,1500:30:400,3000:100:1500"
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
run hnet_v3  --chunkers learned,learned --Ns 4,4 --router tied $COMMON
run iso_v3   --chunkers all --Ns 1 --dims 256,512 $COMMON
run heur_v3  --chunkers external,external --Ns 4,4 $COMMON
run fixed_v3 --chunkers fixed,fixed --Ns 4,4 $COMMON
$PY eval.py hnet_v3 iso_v3 heur_v3 fixed_v3 hnet_v2 --suite hour > runs/eval_hour_v3.log 2>&1
$PY eval.py hnet_v3 iso_v3 heur_v3 fixed_v3 --suite hour_longgap > runs/eval_longgap_v3.log 2>&1
echo V3_DONE
