#!/bin/bash
# Experiment queue (one GPU, sequential). Logs in runs/<name>/stdout.log
set -x
PY=.venv/bin/python
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
# E1: matched 16x compression, 2-stage hierarchy, different chunkers (+ isotropic no-compression reference)
run hnet_16x   --chunkers learned,learned   --Ns 4,4
run fixed_16x  --chunkers fixed,fixed       --Ns 4,4
run heur_16x   --chunkers external,external --Ns 4,4
run iso_1x     --chunkers all --Ns 1 --dims 256,512
# E2: compression sweep with learned chunking
run hnet_4x    --chunkers learned --Ns 4 --dims 256,512
run hnet_64x   --chunkers learned,learned --Ns 8,8 --heartbeats 128,512
# E3: action causality ablation: pure video tokenizer (no action input, policy head is a detached probe)
run hnet_16x_videoonly --chunkers learned,learned --Ns 4,4 --no_actions --detach_policy
# E4: long-context finetune (13.6 min clips, memory gaps up to 10 min)
run hnet_16x_long --chunkers learned,learned --Ns 4,4 --init runs/hnet_16x/ckpt.pt --T 8192 --B 1 --steps 1500 --max_gap 6000 --lr 3e-4 --warmup 100
run iso_1x_long   --chunkers all --Ns 1 --dims 256,512 --init runs/iso_1x/ckpt.pt --T 8192 --B 1 --steps 1500 --max_gap 6000 --lr 3e-4 --warmup 100
# evaluation
$PY eval.py iso_1x fixed_16x heur_16x hnet_16x hnet_4x hnet_64x hnet_16x_videoonly hnet_16x_long iso_1x_long --suite hour > runs/eval_hour.log 2>&1
$PY eval.py hnet_16x iso_1x hnet_16x_long iso_1x_long hnet_64x --suite hour_longgap > runs/eval_longgap.log 2>&1
$PY eval.py hnet_16x_long hnet_64x iso_1x_long --suite day > runs/eval_day.log 2>&1
