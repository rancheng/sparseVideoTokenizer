# Future plans

The plans are ordered by what currently blocks the goal: a policy that *uses* an hour (later a day)
of video context on a single RTX 4090, with compute that scales with events rather than with frames.

## 0. What we learned so far (drives the order below)

1. **Sparse streaming works.** The main network runs at the event rate, its KV cache grows with events,
   and the engine matches training exactly. The per-frame **frame encoder is now the cost floor**.
2. **Prediction metrics do not discriminate.** Next-frame PSNR and action error are flat from 1× to
   255× compression, because the local outer layers handle short-range dynamics. Nothing in the
   objective *pays* for long-range tokens. That is why the budget-cap loss collapses level 2 to
   heartbeats only.
3. **The learned router is barely learning.** `p_t` is pinned at about 0.5, so the boundaries are
   almost random. The pixel-change heuristic is more idle-adaptive (but chases lighting changes).
4. **The models don't remember, and the objective doesn't ask them to.**
   - The cue colour is 96% decodable from the frame latent.
   - It is at chance in `h_t` just 50 frames later, for every model including dense full attention.
   - Copycat shortcut learning from `a_{t−1}` leaves only one decision frame per query, so memory is
     almost never rewarded. Action-input dropout with a 100× decision weight was not enough.

## 1. Fix the router so it actually discriminates (blocking everything else)

The learned router's `p_t` sits within ±0.006 of the 0.5 threshold on idle frames, so its boundaries are
close to coin flips (experiment log, "Router diagnosis"). Independent random `W_q`, `W_k` make q ⊥ k, so
cos ≈ 0 and p ≈ 0.5 regardless of content.

Candidate fixes, to compare on the same 4,000-step setup:
- **Tied init** (`W_k = W_q`, random orthogonal) plus a learned scalar temperature τ and offset β:
  `p = σ((1 − cos − β) / τ)`, with β initialised so that p ≈ 1/N on typical adjacent frames.
- **Difference routing:** `p = σ(w · norm(h_t − h_{t−1}) + b)`. Change magnitude is then explicit and the
  router learns *which* changes matter.
- **Diagnostics for every run:** the `p_t` histogram (it must be bimodal), router-vs-random event recall,
  and idle vs active rate. `viz/behavior.py` already computes all of these.

**Success criterion:** idle tokens approach the heartbeat floor (as the heuristic does), *and* light
toggles are recalled less often than action events (unlike the heuristic).

## 2. Make long-range memory pay

Without this, no chunker can be judged on what matters.

- **Action chunking.** Predict `a_{t:t+16}` (ACT / diffusion-policy style) so every frame near a
  decision carries memory-dependent targets, and drop `a_{t−1}` from the policy input entirely, using
  proprioception instead.
- **Retrieval auxiliary loss.** At random times, predict a property of a frame k steps back
  (k ∈ [64, 36,000]) from `h_t`. This is self-supervised (no labels), and it rewards the router for
  keeping tokens that stay useful.
- **Harder, denser memory tasks.** Several things to remember at once, plus "where did I leave object X"
  queries, so memory is needed for many frames rather than one.
- **Retest the chunkers** (fixed / heuristic / learned / budget-cap) once memory is learned. The
  hypothesis from the blog, learned > heuristic > fixed at equal tokens, can only show up here.

## 3. Sparse frame encoder (biggest remaining compute)

- **Patch-level change router.** Recompute only the 16×16 patches whose router says "changed", and
  reuse cached features elsewhere, in the spirit of Eventful Transformers and token reuse.
- **2-D H-Net spatial chunking** from Cartesia's Part 4 (pixel-space H-Net) as stage 0, giving
  k ≈ 8–16 tokens per frame instead of 256 patches.
- **Target:** per-frame FLOPs proportional to *changed pixels*, so an idle camera costs almost nothing.

## 4. Budget-cap loss that keeps level 2 alive

- Keep the cap (spend at most 1/N), but replace the constant per-token cost with a **cost tied to
  prediction gain**, for example a rate–distortion Lagrangian on the retrieval loss from item 1.
- **Per-level floors and caps tied to the controller's needs.** For example, at least 1 token every
  2.5 s at L1 during motion, nothing extra during idle.

## 5. Engineering for real scale

- **Packed variable-length batches.** Use flex_attention document masks or flash-attn varlen instead of
  padding to the longest chunk sequence. Chunk counts vary 5–30× between clips.
- **Outer stages on Mamba-2** (as in H-Net) for O(1) streaming state and faster long-clip training.
- **CUDA Graphs or a compiled step function** for the streaming engine. At toy size, latency is kernel-launch
  bound (2–3 ms/frame) and doesn't show the FLOP gap.
- **GQA in the main network.** It cuts KV per token about 4×, so 24 h × 40 K events fits in about 2 GB
  (see DESIGN.md §5).
- **Ragged multi-robot batching** in the streaming engine: one GPU serving N robots, each with its
  own event-rate-sized KV cache.

## 6. From 1 hour to 1 day

- **Add an L3 stage** ("sub-task / episode") *around* the trained main network, as H-Net nests a
  stage. Train it on cached L2 outputs from day-long logs, so only L3 and the main network see
  day-length sequences.
- **Heartbeat at L3**, for example every 5 min, so the day-level memory is never blind for long.
- **Evaluate on the 24 h suite** (`eval.py --suite day`) with memory gaps of hours, which needs items 1–2 first.

## 7. Real data

- Move from the synthetic tabletop to LeRobot / DROID-style robot logs. Use a pretrained ViT for L0,
  cache its latents to disk, and train L1, L2 and the main network on 15–60 min clips.
- **Boundary analysis on real data:** do learned boundaries line up with contacts, grasps, failures and
  human interventions?
