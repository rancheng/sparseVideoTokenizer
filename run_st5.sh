#!/bin/bash
# long-clip training of the temporal model on a frozen sparse front (the v5 lesson: memory spans what the clips contain)
# st_pp_L16: 16k-frame clips; st_pp_L2: same frames per step in 2k clips (control); st_heur_L16: heuristic front
PY=.venv/bin/python
run() { name=$1; shift; mkdir -p runs2/$name; [ -f runs2/$name/ckpt.pt ] || $PY train2.py --name $name "$@" > runs2/$name/stdout.log 2>&1; }
PP="--router tied --fovea learned --fov_credit task --fov_pos --rule_mode pair --pop_exit --n_exit 256"
LG="--freeze_front --steps 4000 --warmup 100 --lr 3e-4"
run st_pp_L16 $PP --init runs2/st_pp/ckpt.pt $LG --T 16384 --B 1
run st_pp_L2 $PP --init runs2/st_pp/ckpt.pt $LG --T 2048 --B 8
run st_heur_L16 --router heur --fovea all --rule_mode pair --init runs2/st_pair_heur/ckpt.pt $LG --T 16384 --B 1
$PY eval2.py st_pp_L16 st_pp_L2 st_pp --rule_mode pair --pop_exit --tag hour_pp > runs2/eval_hour_L.log 2>&1
$PY eval2.py st_pp_L16 st_pp_L2 st_pp --rule_mode pair --pop_exit --N 2048 --n_env 48 --tag clip2k > runs2/eval_clip2k_L.log 2>&1
$PY eval2.py st_heur_L16 st_pair_heur --rule_mode pair --tag hour_pair > runs2/eval_hour_heurL.log 2>&1
echo L_DONE
