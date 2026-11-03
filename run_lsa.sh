#!/bin/bash
# native LSA memory read: hnet_v5 recipe (long-gap fine-tuning from v4), each frame reads 32 content-selected entries
# + 32 recent ones. v5s: content-only bucket score (as in the paper); v5sr: with a recency term
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LONG="--freeze_fenc --alibi --mem_read --mem_src tok --mask_unanswerable --cue_rate 0.01 --act_drop 0.3 --w_query 30 --w_pix 10 --steps 6000 --B 1 --T 16384 --min_gap 1500 --max_gap 12000 --lr 3e-4 --warmup 100"
run() { n=$1; shift; mkdir -p runs/$n; [ -f runs/$n/ckpt.pt ] || $PY train.py --name $n --chunkers learned,learned --Ns 4,4 --router tied --init runs/hnet_v4/ckpt.pt $LONG "$@" > runs/$n/stdout.log 2>&1; }
run hnet_v5s --sparse_read 32
run hnet_v5sr --sparse_read 32 --sparse_rec 0.05
echo S_DONE
