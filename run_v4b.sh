#!/bin/bash
# after hnet_v4: held-out evals first, then matched controls with the same event-token memory, then the router fix
while pgrep -f run_v4.sh >/dev/null; do sleep 30; done
PY=.venv/bin/python
COMMON="--alibi --mem_read --mem_src tok --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 8000 --curriculum 0:5:60,1500:30:400,3000:100:1500"
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
$PY eval.py hnet_v4 --suite hour > runs/eval_hour_v4.log 2>&1
$PY eval.py hnet_v4 --suite hour_longgap > runs/eval_longgap_v4.log 2>&1
echo V4_EVAL_DONE
run hnet_v4d --chunkers learned,learned --Ns 4,4 --router tied --router_detach $COMMON
run fixed_v4 --chunkers fixed,fixed --Ns 4,4 $COMMON
run heur_v4  --chunkers external,external --Ns 4,4 $COMMON
run iso_v4   --chunkers all --Ns 1 --dims 256,512 $COMMON
$PY eval.py hnet_v4d fixed_v4 heur_v4 iso_v4 --suite hour > runs/eval_hour_v4b.log 2>&1
$PY eval.py hnet_v4d fixed_v4 heur_v4 iso_v4 --suite hour_longgap > runs/eval_longgap_v4b.log 2>&1
echo V4B_DONE
