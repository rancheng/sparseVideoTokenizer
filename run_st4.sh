#!/bin/bash
# after run_st3.sh: main-network loss variants (w_main 1 hurt the policy: the loss reshaped the token encoders)
PY=.venv/bin/python
CUR="--steps 7000 --fov_warm 1000 --stages 0:256:8:1:fixed,2000:1024:4:1:short,4000:2048:2:0.5:normal"
run() { name=$1; shift; mkdir -p runs2/$name; [ -f runs2/$name/ckpt.pt ] || $PY train2.py --name $name "$@" > runs2/$name/stdout.log 2>&1; }
while kill -0 1365732 2>/dev/null; do sleep 30; done
PP="--router tied --fovea learned --fov_credit task --fov_pos --rule_mode pair --pop_exit --n_exit 256"
run st_pp_md   $PP --w_main 0.3 --main_detach $CUR
run st_pp_m01  $PP --w_main 0.1 $CUR
$PY eval2.py st_pp_md st_pp_m01 --rule_mode pair --pop_exit --N 2048 --n_env 48 --tag clip2k > runs2/eval_clip2k_pp_m.log 2>&1
$PY eval2.py st_pp_md st_pp_m01 --rule_mode pair --pop_exit --tag hour_pp > runs2/eval_hour_pp_m.log 2>&1
echo I_DONE
