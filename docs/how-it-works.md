# How it works

This page explains the model in plain language first, then gives the exact mechanics. It covers
what the code in `vt/` does *today*. The longer-term real-scale design is in [../DESIGN.md](../DESIGN.md).

![pipeline](figures/arch_pipeline.png)

## 1. The idea in one paragraph

A robot camera produces 36,000 frames per hour at 10 Hz. Most of them are near-duplicates: the
robot is waiting, or doing the same motion as a moment ago. A fixed video tokenizer spends the same
number of tokens on every frame, so an hour does not fit in memory. Cartesia's H-Net argues that, for
text, the model should *learn where to cut* the input instead of using a fixed tokenizer. We apply that
to video. The model watches every frame with cheap local layers, and a small learned **router**
decides which moments are new. Only those moments are passed up to the expensive **main network**,
which keeps the long-term memory. One hour becomes about 300–2,800 top-level tokens instead of 36,000
(or the 2–9 million a patch tokenizer would produce).

## 2. Mapping from H-Net (text) to this model (video)

![mapping](figures/hnet_mapping.png)

| Text H-Net / blog | Video model here |
|---|---|
| byte | one frame (plus the action just taken) |
| BPE tokenizer | fixed-stride chunking (`fixed_16x`) |
| BLT entropy-threshold chunking | pixel-change "keyframe" heuristic (`heur_16x`) |
| byte-level Transformer | frame-level Transformer, no compression (`iso_1x`) |
| H-Net dynamic chunking | `hnet_4x`, `hnet_16x`, `hnet_64x` |
| next-byte prediction | next-frame prediction given the action, plus predicting the action |

What is taken from H-Net (Hwang, Wang & Gu, 2025):
- **Shape.** Encoder, then main network, then decoder, nested recursively.
- **Router.** `p_t = ½(1 − cos(W_q h_t, W_k h_{t−1}))`, and the position is a boundary if `p_t ≥ 0.5`.
- **Ratio loss.** `N/(N−1)·((N−1)·F·G + (1−F)·(1−G))`, with weight α = 0.03.
- **Smoothing.** An EMA over chunks: `z̄_k = P_k z_k + (1 − P_k) z̄_{k−1}`.
- **Straight-through confidence and residual skip.**

What is new for video and robots:

| change | why |
|---|---|
| random router init (not identity) | adjacent frames are near-identical, so the identity init gives p≈0 and the router never wakes up. **Known issue:** independent random W_q, W_k leave p≈0.5 everywhere, so the router barely discriminates; see the experiment log and future-plans §1 |
| heartbeat boundaries every 64 / 256 frames | bounded staleness for a control loop; prevents collapse to zero tokens |
| NoPE + log-bucketed Δt embedding in the main network | chunks are irregular in time; positions never exceed the training range, so it extrapolates to hours and days |
| actions in, actions out | the router can tell self-caused change from exogenous change; boundaries become action-relevant |
| sliding-window attention instead of Mamba-2 in outer stages | same role (cheap, local, constant memory), no custom kernels |
| strict causality, tested | the same weights run as a live streaming policy |

## 3. Data format at every level

The tokens are **continuous vectors with timestamps**, not discrete codebook IDs. Each level is a ragged
sequence: a busy minute produces more tokens than an idle one.

```
input frames     (B, T, 3, 64, 64)    10 Hz
input actions    (B, T, 3)            a_{t-1} = (vx, vy, gripper)
frame tokens     (B, T, 256)          1 per frame, + Linear(a_{t-1})
  router         p (B, T), b (B, T)
L1 tokens        (B, ~T/4, 384)       + start frame of each chunk
  router         p, b over L1 tokens
L2 tokens        (B, ~T/16, 512)      + start frame         ← what the main network sees & remembers
main output      (B, ~T/16, 512)
dechunked        (B, T, 256)          every frame gets the latest event's context + its own detail
outputs          policy (B, T, 3); next-frame prediction (B, T, 3, 64, 64)
```

For a single live stream, the tokenizer output is a list `[(vector_512, start_frame), …]`. In memory,
each top-level token is one key/value entry per main-network layer (16 KB per token at the current size).

## 4. The pieces, in order

1. **Frame encoder.** A 4-layer conv net maps 64×64×3 to 256 numbers. The previous action is
   embedded and added.
2. **Stage 1 encoder.** Two attention blocks, each looking at the last 64 frames (6.4 s), with local RoPE.
3. **Router.** It compares the current frame's state with the previous one, and a large change makes a
   boundary. A heartbeat also forces a boundary whenever the clock crosses a multiple of 64 frames.
4. **Chunk.** The boundary positions are gathered. Their vectors are projected to 384 dims, and an
   embedding of "frames since the previous chunk" is added.
5. **Stage 2.** The same machinery runs over the L1 tokens, with a heartbeat every 256 frames (25.6 s).
   Its tokens are projected to 512 dims.
6. **Main network.** 8 full causal attention blocks with no position encoding, run over L2 tokens
   only. Its KV cache is the long-term memory.
7. **Dechunk, on the way down at each stage.** The main network's output for each chunk is smoothed
   with the EMA and repeated to every position until the next boundary (always using a chunk that
   started *at or before* that position). It is multiplied by the straight-through confidence and added
   to a skip connection from that stage's encoder. Then two decoder blocks run.
8. **Heads at frame rate.** The policy predicts `a_t`. The forward model takes `(h_t, a_t)`, and a conv
   decoder renders the predicted `o_{t+1}`.

**Training loss:** `100·MSE(next frame) + MSE(action) + 0.1·BCE(gripper) + 0.03·ratio_loss`.

The ratio loss has two modes:
- **`target`** (H-Net default): pushes the *average* rate to 1/N.
- **`cap`**: the ratio loss applies only when over budget, plus a small per-token cost. Idle time then
  falls to the heartbeat floor.

## 5. Streaming: the sparse part

![streaming](figures/arch_streaming.png)

`vt/stream.py` runs the same weights one frame at a time:

| runs | when | state it keeps |
|---|---|---|
| frame encoder, stage-1 enc/dec, heads | every frame | 64-frame sliding KV window |
| stage-2 enc/dec | only when stage 1 emits a token (~1 in 4 frames) | 64-token sliding window |
| main network, 1 step + 1 KV append | only when stage 2 emits a token (~1 in 16–120 frames) | grows by one entry per event |

Between boundaries, each level reuses its cached smoothed chunk state. That is exactly what dechunking
computes in training, so the streaming engine is **numerically identical** to the full-sequence model
(max policy error 1.6e-5 in fp32, identical boundaries; `tests/test_stream.py`). The only state that grows
with time is the main network's KV cache, one entry per top-level event.

Measured over 1 hour of streaming (see [experiment-log.md](experiment-log.md)):
- **The main network becomes a small cost.** It takes 85% of per-frame compute in the dense model, but
  4% (hnet_16x), 0.7% (hnet_64x) or 0.3% (budget-cap) in the sparse ones.
- **Main KV cache after 1 h:** 590 MB dense vs 26 / 4.9 / 2.3 MB.
- **The frame encoder (56 MFLOP/frame) is now the cost floor.** It is the next thing to make sparse.

## 6. Guarantees and tests

- `tests/test_causal.py`: perturbing frames or actions after time t leaves every output before t unchanged.
- `tests/test_stream.py`: the frame-by-frame streaming engine reproduces the full-sequence forward for
  learned, fixed, heuristic and isotropic chunkers.
