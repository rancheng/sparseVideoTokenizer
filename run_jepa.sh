#!/bin/bash
# JEPA objective on the toy stream (vt/jepa.py), after the real-data queue: pure latent prediction, + inverse dynamics,
# + future actions. Same front, curriculum and env as st_pp (pixel objective), for the per-entity budget comparison.
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (on the 4090 this waited for the real-data queue; run it on its own: one GPU job at a time on 24 GB)
CUR="--steps 7000 --fov_warm 1000 --stages 0:256:8:1:fixed,2000:1024:4:1:short,4000:2048:2:0.5:normal"
PP="--router tied --fovea learned --fov_credit task --fov_pos --rule_mode pair --pop_exit --n_exit 256"
run() { name=$1; shift; mkdir -p runs2/$name; [ -f runs2/$name/ckpt.pt ] || $PY train2.py --name $name "$@" > runs2/$name/stdout.log 2>&1; }
run st_pj0 $PP $CUR --obj jepa --w_pix 0 --w_inv 0 --w_fut 0
run st_pj1 $PP $CUR --obj jepa --w_pix 0 --w_fut 0
run st_pj2 $PP $CUR --obj jepa --w_pix 0
echo J_DONE
