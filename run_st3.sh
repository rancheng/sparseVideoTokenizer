#!/bin/bash
# st_pp_main rerun (stale main-loss graph fixed), then hnet_v5 at batch 1 (B=2, T=16384 runs out of memory with learned chunking)
PY=.venv/bin/python
CUR="--steps 7000 --fov_warm 1000 --stages 0:256:8:1:fixed,2000:1024:4:1:short,4000:2048:2:0.5:normal"
run() { name=$1; shift; mkdir -p runs2/$name; [ -f runs2/$name/ckpt.pt ] || $PY train2.py --name $name "$@" > runs2/$name/stdout.log 2>&1; }
PP="--router tied --fovea learned --fov_credit task --fov_pos --rule_mode pair --pop_exit --n_exit 256"
run st_pp_main $PP --w_main 1 $CUR
$PY eval2.py st_pp_main --rule_mode pair --pop_exit --tag hour_pp > runs2/eval_hour_pp_main.log 2>&1
$PY eval2.py st_pp_main --rule_mode pair --pop_exit --N 2048 --n_env 48 --tag clip2k > runs2/eval_clip2k_pp_main.log 2>&1
echo H_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LONG="--freeze_fenc --alibi --mem_read --mem_src tok --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 6000 --B 1 --T 16384 --min_gap 1500 --max_gap 12000 --lr 3e-4 --warmup 100"
[ -f runs/hnet_v5/ckpt.pt ] || $PY train.py --name hnet_v5 --chunkers learned,learned --Ns 4,4 --router tied --init runs/hnet_v4/ckpt.pt $LONG > runs/hnet_v5/stdout.log 2>&1
$PY eval.py hnet_v5 fixed_v5 hnet_v4 fixed_v4 --suite hour_longgap --n_env 24 > runs/eval_longgap_v5.log 2>&1
echo V5_DONE
