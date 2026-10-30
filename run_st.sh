#!/bin/bash
# v2 testbed experiment matrix (docs/v2-spatiotemporal.md). Fixes so far: per-frame event cap, router/fovea read the
# content embedding without writing it, beta-NLL residual cell forward model, decision loss with its own mean,
# fixed-rule curriculum stage, half of the clips centred on a decision.
PY=.venv/bin/python
mkdir -p runs2
CUR="--steps 7000 --fov_warm 1000 --stages 0:256:8:1:fixed,2000:1024:4:1:short,4000:2048:2:0.5:normal"
run() { name=$1; shift; mkdir -p runs2/$name; [ -f runs2/$name/ckpt.pt ] || $PY train2.py --name $name "$@" > runs2/$name/stdout.log 2>&1; }
run st_learned --router tied --fovea learned $CUR
run st_short   --router tied --fovea learned --delay 20,50 $CUR
$PY eval2.py st_learned > runs2/eval_hour_A.log 2>&1
$PY eval2.py st_short st_learned --delay 20,50 --tag hour_short > runs2/eval_hour_short.log 2>&1
echo A_DONE
run st_rel     --router rel  --fovea learned $CUR
run st_lowres  --router tied --fovea none $CUR
$PY eval2.py st_rel st_lowres > runs2/eval_hour_B.log 2>&1
echo B_DONE
run st_heur    --router heur --fovea all $CUR
run st_dense   --router dense --fovea all $CUR
$PY eval2.py st_heur st_dense > runs2/eval_hour_C.log 2>&1
echo C_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LONG="--freeze_fenc --alibi --mem_read --mem_src tok --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 3000 --B 2 --T 16384 --min_gap 1500 --max_gap 12000 --lr 3e-4 --warmup 100"
[ -f runs/hnet_v5/ckpt.pt ] || $PY train.py --name hnet_v5 --chunkers learned,learned --Ns 4,4 --router tied --init runs/hnet_v4/ckpt.pt $LONG > runs/hnet_v5/stdout.log 2>&1
$PY eval.py hnet_v5 fixed_v5 hnet_v4 fixed_v4 --suite hour_longgap --n_env 24 > runs/eval_longgap_v5.log 2>&1
echo V5_DONE
