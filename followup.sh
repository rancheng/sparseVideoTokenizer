#!/bin/bash
# runs after run_all.sh: budget-cap training, then evals + streaming benchmarks
while kill -0 219311 2>/dev/null; do sleep 30; done
PY=.venv/bin/python
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
run hnet_cap_r02 --chunkers learned,learned --Ns 4,4 --ratio_mode cap --rate_w 0.02
run hnet_cap_r10 --chunkers learned,learned --Ns 4,4 --ratio_mode cap --rate_w 0.1
$PY eval.py hnet_cap_r02 hnet_cap_r10 --suite hour > runs/eval_hour_cap.log 2>&1
$PY bench_stream.py iso_1x fixed_16x heur_16x hnet_16x hnet_64x hnet_cap_r02 hnet_cap_r10 --frames 36000 > runs/bench_stream.log 2>&1
$PY bench_scale.py > runs/bench_scale.log 2>&1
echo FOLLOWUP_DONE
