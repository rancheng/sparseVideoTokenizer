#!/bin/bash
# memory fix: anti-copycat action dropout + strong decision-frame weight, hierarchy vs dense
while kill -0 252265 2>/dev/null; do sleep 30; done
PY=.venv/bin/python
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
run hnet_16x_mem --chunkers learned,learned --Ns 4,4 --act_drop 0.3 --w_query 100
run iso_1x_mem --chunkers all --Ns 1 --dims 256,512 --act_drop 0.3 --w_query 100
$PY eval.py hnet_16x_mem iso_1x_mem --suite hour > runs/eval_hour_mem.log 2>&1
echo FOLLOWUP2_DONE
