# sparseVideoTokenizer

A causal, action-conditioned **hierarchical video tokenizer for robot policies**. It adapts the H-Net
idea from Cartesia's "From Tokens to Bytes" posts (learned dynamic chunking inside the model, with no
fixed tokenizer) from bytes to video frames. The goal is to keep 1 hour, and later 1 day, of video
context on a single RTX 4090, using sparse, event-driven compute.

```
frame + a_{t-1} → frame encoder → L1 stage (local, router: "new moment?") → L2 stage ("new event?")
                → main network (full causal attention over events only, NoPE + Δt embedding)
                → dechunk back to frame rate → policy a_t  +  forward model o_{t+1} | a_t
```

The design and rationale are in [DESIGN.md](DESIGN.md). Results tables are in `runs/report/tables.md`.

## Layout

| path | what |
|---|---|
| `vt/env.py` | synthetic long-horizon tabletop robot stream with ground-truth events, distractors, memory task; GPU renderer |
| `vt/model.py` | `VideoHNet`: router, ratio / budget-cap loss, heartbeat boundaries, chunk/dechunk, sliding-window + full attention |
| `vt/stream.py` | event-driven streaming engine: the main network runs only on top-level boundaries; KV caches; FLOP accounting |
| `train.py` | training (learned / fixed / heuristic / isotropic chunkers) |
| `eval.py` | hour / long-gap / day evaluation suites in one causal pass |
| `bench_stream.py`, `bench_scale.py` | frame-by-frame streaming benchmark; real-scale (680M) main-network latency vs KV length |
| `tests/` | causality test, streaming == full-forward equivalence test |
| `run_all.sh`, `followup*.sh` | the experiment queues that produced `runs/` |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py --name hnet_16x --chunkers learned,learned --Ns 4,4
.venv/bin/python eval.py hnet_16x --suite hour
.venv/bin/python bench_stream.py hnet_16x --frames 36000
.venv/bin/python tests/test_causal.py && .venv/bin/python tests/test_stream.py hnet_16x
```

## Status (2026-10-29)

**Working**
- **Streaming is exact.** The frame-by-frame engine matches the full-sequence forward to 1.6e-5 (fp32) with identical boundaries.
- **Sparse compute works.** Main-network FLOPs per frame drop from 85% of total (dense) to 0.3–4%. Main KV after 1 h drops from 590 MB to 2–26 MB.
- **A full day fits.** A 24 h stream (864 K frames) runs on the 4090 in one pass.
- **Action conditioning matters.** It makes boundaries more action-aligned: level-1 action-event recall is 0.84 vs 0.66 without it.

**Open**
- **Long-range memory doesn't work yet.** The cue → query memory task is at chance for all models, the dense baseline included. The cause is copycat shortcut learning from the previous-action input; anti-copycat runs are in progress.
- **Budget-cap rate loss over-compresses.** Idle cost falls to the heartbeat floor as intended, but level 2 also collapses to heartbeat-only, because nothing in the objective yet rewards long-range tokens.
- **The frame encoder is now the per-frame cost floor.** A sparse (changed-patch-only) frame encoder is the next step.
