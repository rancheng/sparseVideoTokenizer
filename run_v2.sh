#!/bin/bash
# v2: fixed router (tied init + calibrated temperature), time-ALiBi main network, per-frame memory read,
# answerable-only memory labels, denser cues, anti-copycat action dropout.
PY=.venv/bin/python
COMMON="--alibi --mem_read --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --steps 8000"
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
run hnet_v2 --chunkers learned,learned --Ns 4,4 --router tied $COMMON
run iso_v2 --chunkers all --Ns 1 --dims 256,512 $COMMON
echo V2_DONE
