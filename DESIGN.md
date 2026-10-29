# Event-Chunked Video Tokenizer for Long-Context Robot Policies

Goal: a causal video tokenizer for action-generating robot policies. It should keep **1 hour** of
multi-camera history resident on one **RTX 4090 (24 GB)**, and grow to **1 day** without changing
the architecture. It is built on the Cartesia H-Net ideas ("From Tokens to Bytes" I & II):
learned dynamic chunking inside the model, a recursive hierarchy, compression that adapts to
content, and no hand-written tokenizer.

## 1. Why a fixed video tokenizer cannot reach an hour

| tokenizer | tokens / frame | 10 Hz × 1 h | 10 Hz × 24 h |
|---|---|---|---|
| ViT patches 224² (256) | 256 | 9.2 M | 221 M |
| VLM-style pooled (64) | 64 | 2.3 M | 55 M |
| fixed 3D tokenizer (4×16×16 on 256²) | 64 / 4 frames | 0.58 M | 14 M |
| **this design, top level** | adaptive | **~3–10 K** | **~20–60 K** (with L3) |

A 4090 holds about 100–200 K tokens of KV cache for a ~1 B main network (see §5). Fixed tokenizers
spend the same budget on a robot standing still as on a grasp. Robot streams are mostly idle
or repetitive, and the decision-relevant content is sparse: contacts, grasps, failures, human
interventions, scene changes. Those events are what an H-Net-style learned chunker should
spend tokens on.

The lessons from the Cartesia posts that carry over directly:
1. **Learned boundaries beat heuristic ones.** BLT-style entropy thresholds trail end-to-end
   routing by 1.3–1.8× compute, and frozen proxy boundaries are worse too. The video analogue of
   the entropy heuristic is codec-style keyframes chosen by pixel change. We test this directly
   (`heur_16x`).
2. **Compression ≠ efficiency.** Chunks need to be *semantic*: action-relevant events, not merely
   "pixels changed".
3. **Hierarchy is recursive.** Outer stages are cheap, local networks (Mamba or SWA), and the
   expensive main network only sees the coarsest tokens. In H-Net-XL, 43% of FLOPs sit in the outer
   stages and 57% in the main network, which runs at about 1/8 of the input positions.

## 2. Architecture

```
camera(s) ──► L0 spatial: frozen/finetuned ViT patches ─► spatial dynamic chunking ─► k≈8–16 tok/frame
            (proprio, a_{t-1}, language) are added as tokens/embeddings at L0
L1 temporal stage (frames → "motion primitives", ~4×):   SWA/Mamba encoder · router · chunk
  L2 temporal stage (→ "events/contacts", ~4–8×):       SWA/Mamba encoder · router · chunk
    L3 temporal stage (→ "sub-tasks/episodes", ~8–16×):  added when going from 1 h to 1 day
      MAIN network: full causal attention, NoPE + Δt embedding, KV cache = the long-term memory
    dechunk (EMA smoothing · causal repeat · STE confidence) + residual + decoder (each stage)
heads at frame rate: policy a_t (BC / flow / diffusion head) · forward model o_{t+1} | h_t, a_t
```

Key design decisions. All are implemented in `vt/model.py` and exercised in the experiments.

* **Strictly causal everywhere.** The router compares `q_t` with `k_{t-1}` only. Dechunking at frame
  t uses the chunk that started at or before t, and windowed attention only looks back. The same weights
  therefore run as a streaming policy with O(1) work per frame in the outer stages
  (`tests/test_causal.py` checks this).
* **Action-conditioned tokenizer.** `a_{t-1}` (and proprioception) enter at L0, so the router can tell
  self-caused visual change (the arm moving) from exogenous change. The training objective is
  action-causal: (i) a policy head predicts `a_t` from `h_≤t`, and (ii) a forward model predicts
  `o_{t+1}` from `(h_≤t, a_t)`. This is the "next byte prediction" of video for robots: chunk boundaries
  are learned where they help predict *what happens when I act*.
* **Router.** This is H-Net routing: `p_t = ½(1 − cos(W_q x_t, W_k x_{t−1}))`, `b_t = [p_t ≥ ½]`, with the
  ratio loss `L = N/(N−1)·((N−1)FG + (1−F)(1−G))`, α = 0.03. **A video-specific change: `W_q, W_k` are
  randomly initialised, not set to the identity.** Adjacent frames are near-identical, so the identity init
  gives p≈0 and the router never wakes up (observed). Independent random init then leaves p≈0.5 (q ⊥ k), so the router barely discriminates (see docs/experiment-log.md). A tied or temperature-scaled init is the planned fix.
* **Heartbeat boundaries.** A boundary is forced whenever absolute time crosses a multiple of M_l frames
  (64 / 256 here). This bounds the staleness of the main network's view, which matters for a control loop.
  It also prevents collapse to zero tokens and costs only 3600·10/M tokens per hour.
* **Time-aware main network.** Chunks are irregular in time, so the main network uses NoPE and adds a
  log-bucketed Δt embedding of the gap since the previous chunk. Positions never exceed the training
  range, so it extrapolates to hours and days, and the model knows how long ago something happened.
* **EMA smoothing via a log-depth parallel scan.** This is a Hillis–Steele scan, so dechunking costs
  O(L log L) with no Python loop. It runs on 216 K chunks per day.

## 3. Training recipe (single 4090)

1. **L0 is pretrained or frozen.** Cache L0 latents to disk: 1 h × 10 Hz × 16 tok × 256 d × bf16 ≈ 295 MB,
   so cached latents for 100 h of robot logs fit on an NVMe SSD.
2. **Train L1/L2 + main network on 3–15 min clips.** The main network sees 16–64× fewer positions,
   so 13-minute clips cost what a 50-second clip costs an isotropic model (exp. E4).
3. **Long-context finetune** on 30–60 min clips. Use gradient checkpointing on the outer stages; the
   main network sees only ~2–5 K tokens per hour.
4. **Grow to a day.** Add an L3 stage *around* the trained main network, the same way H-Net nests a
   stage. Train it on cached L2 outputs from day-long logs; only L3 and the main network need
   long-sequence training.
5. **Policy finetune** with the target action head (diffusion / flow), keeping the tokenizer objective
   as an auxiliary loss.

## 4. Experiments run here (see REPORT.md)

The test bed is a synthetic tabletop stream with ground-truth events, long idle periods, exogenous
distractors (light toggles), a predictable nuisance (a blinking LED), sensor noise, and a memory task
whose correct action depends on a cue seen minutes to hours earlier.

* E1: learned vs fixed-stride vs pixel-change-heuristic chunking at matched 16× compression, plus an
  uncompressed isotropic frame transformer.
* E2: compression sweep at 4×, 16× and 64×.
* E3: action causality. The same tokenizer trained with no action input and no policy gradient
  (a pure video tokenizer with a detached policy probe).
* E4: long context. A 1-hour single causal pass, cue→query gaps up to 40 min, and a 24-hour stream
  (864 K frames) in one pass on the 4090.

## 5. Memory budget on a 4090 (real scale)

Main network: 24 layers, d = 1536, GQA with 4 KV heads × 128, bf16. KV per token = 2·24·512·2 B = 48 KB.

| horizon | frames (10 Hz) | L0 tokens (12/frame) | L1 (÷4) | L2 (÷8) | L3 (÷8) | main KV |
|---|---|---|---|---|---|---|
| 1 h | 36 K | 432 K (SWA only) | 108 K | **13.5 K** | – | **0.65 GB** |
| 8 h | 288 K | 3.5 M | 864 K | 108 K | **13.5 K** | 0.65 GB |
| 24 h | 864 K | 10 M | 2.6 M | 324 K | **40 K** | **1.9 GB** |

The outer stages keep only sliding-window state of a few MB, independent of horizon. Weights for a ~1.3 B
main network plus outer stages take about 3 GB in bf16. Everything fits in 24 GB with room for 4–8×
more cameras or tokens. These are *average* rates. Learned chunking spends fewer tokens in idle
stretches and more during manipulation; the heartbeat sets the floor.
