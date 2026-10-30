#!/bin/bash
# v4 = v3 + memory entries are the event tokens themselves (main-network input) next to the main output.
# Probes on v3 showed main outputs overwrite what an event contained, so the read could only relay, not retrieve.
PY=.venv/bin/python
COMMON="--alibi --mem_read --mem_src tok --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 8000 --curriculum 0:5:60,1500:30:400,3000:100:1500"
run() { name=$1; shift; mkdir -p runs/$name; [ -f runs/$name/ckpt.pt ] || $PY train.py --name $name "$@" > runs/$name/stdout.log 2>&1; }
run hnet_v4  --chunkers learned,learned --Ns 4,4 --router tied $COMMON
echo V4_HNET_DONE
