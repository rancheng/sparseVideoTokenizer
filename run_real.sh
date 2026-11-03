#!/bin/bash
# real robot video (LeRobot, real/prep_lerobot.py): sparse vs dense vs pixel-change front, model size, data scale
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { n=$1; shift; mkdir -p runs_real/$n; [ -f runs_real/$n/ckpt.pt ] || $PY -m real.train_real --name $n "$@" > runs_real/$n/stdout.log 2>&1; }
S="--steps 8000 --B 8 --T 256"
run rs_base $S
run rs_dense $S --router dense --fovea all
run rs_heur $S --router heur --fovea all
run rs_small $S --size small
run rs_large $S --size large
run rs_d1 $S --sets pusht
run rs_d4 $S --sets pusht,columbia_cairlab_pusht_real,aloha_static_coffee,droid_100
for n in rs_base rs_dense rs_heur rs_small rs_large rs_d1 rs_d4; do $PY -m real.eval_real $n > runs_real/$n/eval.log 2>&1; done
echo REAL_DONE
