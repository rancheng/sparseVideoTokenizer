#!/bin/bash
# st_pp_L2 rerun: batch 8 needs smaller front pieces (8 x 512 frames rendered at once)
PY=.venv/bin/python
while kill -0 1437011 2>/dev/null; do sleep 60; done
PP="--router tied --fovea learned --fov_credit task --fov_pos --rule_mode pair --pop_exit --n_exit 256"
mkdir -p runs2/st_pp_L2
$PY train2.py --name st_pp_L2 $PP --init runs2/st_pp/ckpt.pt --freeze_front --steps 4000 --warmup 100 --lr 3e-4 --T 2048 --B 8 --chunk 512 > runs2/st_pp_L2/stdout.log 2>&1
$PY eval2.py st_pp_L2 --rule_mode pair --pop_exit --tag hour_pp > runs2/eval_hour_L2.log 2>&1
$PY eval2.py st_pp_L2 --rule_mode pair --pop_exit --N 2048 --n_env 48 --tag clip2k > runs2/eval_clip2k_L2.log 2>&1
echo L2_DONE
