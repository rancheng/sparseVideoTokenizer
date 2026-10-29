# sparseVideoTokenizer

A causal, action-conditioned **hierarchical video tokenizer for robot policies**. It adapts the H-Net
idea from Cartesia's "From Tokens to Bytes" posts ([part 1](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1),
[part 2](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2)) from bytes to video frames: the model learns
where to cut the stream, instead of using a fixed tokenizer. The goal is to keep **1 hour, and later 1 day, of
video context on one RTX 4090**, with **event-driven sparse compute**: the expensive network runs only when
something new happens.

![streaming demo](docs/figures/stream.gif)

*A held-out robot stream at 5× speed. Counters show tokens emitted so far by a dense tokenizer (one per frame)
vs the learned hierarchy (level 1 and level 2) and the pixel-change heuristic.*

## How it works

![pipeline](docs/figures/arch_pipeline.png)

- **Frames are the "bytes".** Each 64×64 frame becomes one 256-d vector, plus an embedding of the previous action.
- **Two learned chunking stages.** Cheap sliding-window layers look at the recent past, and an H-Net router
  (`p_t = ½(1 − cos(W_q h_t, W_k h_{t−1}))`) decides which moments are new. Level 1 keeps about 1 in 4
  frames, and level 2 about 1 in 16–23. A forced *heartbeat* boundary every 64 / 256 frames bounds staleness.
- **The main network** (full causal attention, NoPE + Δt embedding) sees only level-2 tokens. Its KV cache
  is the long-term memory.
- **Back down to frame rate.** The main network's output is dechunked (EMA smoothing, causal repeat, STE)
  and feeds two heads: the **policy** `a_t` and an action-conditioned **forward model** `o_{t+1} | a_t`.
- **Everything is strictly causal**, so the same weights run as a live streaming policy.

![streaming engine](docs/figures/arch_streaming.png)

The streaming engine (`vt/stream.py`) runs the frame encoder and level 1 every frame, level 2 only on
level-1 boundaries, and the main network only on level-2 boundaries. It is numerically identical to the
full-sequence model (max error 1.6e-5, identical boundaries).

Text-to-video mapping, and what is new for robots:

![mapping](docs/figures/hnet_mapping.png)

Full explanation: **[docs/how-it-works.md](docs/how-it-works.md)**. Real-scale design and memory budget: **[DESIGN.md](DESIGN.md)**.

## Results so far

All runs use a synthetic long-horizon tabletop stream with ground-truth events, long idle periods,
exogenous light toggles, a blinking LED, sensor noise and a cue→query memory task. The full log, with every
number, is in **[docs/experiment-log.md](docs/experiment-log.md)**.

**Sparse streaming removes the main-network cost.** The frame encoder becomes the new floor.

![stream compute](docs/figures/stream_compute.png)

**Memory grows with events, not frames.** Main-network KV after 1 h: 590 MB dense vs 26 MB (16×) vs 2.3 MB (budget cap).

![kv growth](docs/figures/kv_growth.png)

**Real-scale projection for a 680M main network on one 4090.** The per-step timings are measured; the
bars are projections from them. Dense needs 127 GB of KV for a day; event-driven needs 1–6 GB.

![scale projection](docs/figures/scale_projection.png)

**What doesn't work yet:**

- **The learned router barely discriminates.** Its `p_t` is pinned at about 0.5 (see the timeline below),
  so boundaries are close to coin flips. The pixel-change heuristic is idle-adaptive, but it chases lighting changes.
- **Prediction quality is flat from 1× to 255× compression,** and memory recall is at chance for every
  model, dense included. Nothing in the objective rewards remembering yet.

![timeline](docs/figures/timeline.png)

![boundary quality](docs/figures/boundary_quality.png)

![compression vs quality](docs/figures/compression_quality.png)

<details>
<summary>More figures: what each tokenizer chose to keep, token rate over an hour, training curves</summary>

![filmstrip](docs/figures/filmstrip.png)

![token rate](docs/figures/token_rate.png)

![training curves](docs/figures/training_curves.png)

</details>

## What's next

Fix the router so it is sensitive at init: tied or temperature-scaled cosine, or difference routing. Then
make long-range memory pay through the objective: action chunking, and a self-supervised retrieval loss.
After that comes a sparse frame encoder that recomputes only the patches that changed, and an L3 stage for
day-long context. Details: **[docs/future-plans.md](docs/future-plans.md)**.

## Layout

| path | what |
|---|---|
| `vt/env.py` | synthetic long-horizon robot stream with ground-truth events, distractors, memory task; GPU renderer |
| `vt/model.py` | `VideoHNet`: router, ratio / budget-cap loss, heartbeat, chunk/dechunk, sliding-window + full attention |
| `vt/stream.py` | event-driven streaming engine with KV caches and FLOP accounting |
| `train.py`, `eval.py` | training; hour / long-gap / day evaluation in one causal pass |
| `bench_stream.py`, `bench_scale.py` | frame-by-frame streaming benchmark; real-scale main-network latency vs KV length |
| `tests/` | causality test; streaming == full-forward equivalence test |
| `viz/` | figure generators (`arch_diagrams.py`, `behavior.py`, `stream_gif.py`, `results.py`) |
| `docs/` | how it works, experiment log, future plans, figures |
| `run_all.sh`, `followup*.sh` | the experiment queues that produced `runs/` |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py --name hnet_16x --chunkers learned,learned --Ns 4,4      # ~11 min on a 4090
.venv/bin/python eval.py hnet_16x --suite hour
.venv/bin/python bench_stream.py hnet_16x --frames 36000
.venv/bin/python tests/test_causal.py && .venv/bin/python tests/test_stream.py hnet_16x
.venv/bin/python viz/results.py        # regenerate result charts from runs/*.json
```
