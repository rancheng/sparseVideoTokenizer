# v2: sparse spatio-temporal tokenizer with a fovea, and a testbed for each assumption

The v1 to v4 experiments got the memory working, but they also showed why H-Net's level-2 router learned nothing on
the v1 stream. Removing the main network cost **0.03 dB** of next-frame PSNR, so the dense loss gave the router no
reason to care where it cut. The v1 frame encoder was also dense: 56 MFLOP per frame, every frame, whatever happened.

v2 changes both the testbed and the model, so that every assumption becomes a falsifiable prediction.

## Testbed: `vt/env2.py` (128×128, an 8×8 grid of 16×16 cells, 10 Hz)

| element | what it forces | hypothesis it tests |
|---|---|---|
| conveyor and opaque **tunnel**, transit time D | to predict the item at the exit, the model must remember it from D frames ago; with D = 150–600 that is past the local windows | **H1**: the router learns where to cut only if the main network matters for the dense loss |
| **glyph-coded sorting** (solid, horizontal stripes, vertical stripes, checker, all with the same mean colour) | the glyph is invisible at 4× downsampling, and the target bin depends on it | **H4**: a learned fovea finds the few cells that need high resolution |
| **rule symbol** flashed for 5 frames, minutes apart (identity or mirrored glyph→bin mapping) | every sorting decision depends on a small symbol seen minutes ago | **H3**: credit for long memory is dense (every episode), not one query every few minutes |
| **noisy TV** (one cell, random position per env, unpredictable) | a pure change detector fires on it every frame | **H6**: a relevance term lets the router ignore change that cannot be predicted |
| static textured background, light toggles, sensor noise, ~89% idle | spatial and temporal sparsity | compute should scale with events, not pixels or frames |
| ground truth: phase, primitive boundaries, task starts, exits, rule, glyph, bin | — | **H2**: L1 matches primitives, L2 matches tasks |

## Model: `vt/model2.py` (`SparseVideoModel`)

1. **Cheap embedding** of every cell from its 4×-downsampled pixels: 48 numbers. This is the only work done
   for every cell on every frame, about 0.4 MFLOP per frame, against 56 MFLOP for the v1 frame encoder.
2. **Per-cell temporal router.** H-Net dynamic chunking over time, applied at each location:
   `p = σ((1 − cos(W_q c_t, W_k c_{t−1}) − β)/τ)`, using the tied initialisation and calibration from v3,
   plus a staggered per-cell heartbeat. `router=rel` adds a zero-initialised learned relevance logit (H6).
3. **Fovea gate on event cells.** The high-resolution conv encoder of the full 16×16 cell runs only where the
   gate is on. The gate is trained with a straight-through estimator and the H-Net ratio loss (target: 1 in 4
   events).
4. **Sparse spatial transformer.** Event cells are recomputed and attend to all cells; non-event cells keep their
   cached state. Training emulates this exactly by gathering stale states from each cell's last event time. The
   high-resolution encoder runs only on event cells during training too.
5. **Per-cell dechunk.** H-Net's EMA over each cell's events with weight p, held between events, with
   straight-through confidence. This is what gives the router its gradient.
6. **Pooling.** K = 4 latent queries pool the 64 cell states into one frame vector. The temporal H-Net from v4
   follows: L1 and L2 learned chunking, main network, event-token memory.
7. **Heads.** The policy; and a **per-cell forward model** that predicts o_{t+1} for sampled cells from the temporal
   latent, the cell state and the action. Half of the sampled cells are cells that change, so tunnel exits and
   arm motion dominate the loss rather than a static background.

Streaming (`vt/stream2.py`) does per frame: cheap embedding and router for all cells; then, for event cells only,
the gate, the high-resolution encoder, the transformer rows (against cached keys and values) and the EMA; then
re-pooling. `tests/test_stream2.py` checks that it matches the training forward pass. Chunked long-stream
processing carries the same state and is exact (`SparseFront.forward(state=...)`).

## Experiments and the result each hypothesis predicts

| run | front end | tunnel delay | prediction |
|---|---|---|---|
| `st_learned` | learned router + learned fovea | 150–600 | removing the main network should cost many dB at exits; L2 task recall well above random; fovea finds the glyph and rule cells |
| `st_short` | same | 20–50 | control for H1: exits are solvable locally, so removing the main network should cost little, and L2 should be near random again |
| `st_rel` | relevance router + fovea | 150–600 | H6: TV event rate falls below the pure change router's, at equal budget |
| `st_heur` | pixel-change cells, high resolution on every event | 150–600 | spends its budget on the TV and light toggles; cell budget matched |
| `st_lowres` | learned router, no fovea | 150–600 | H4 control: sorting near chance (glyph and rule invisible); exit colour still fine |
| `st_dense` | every cell, every frame, full resolution | 150–600 | upper bound on quality; compute about 15–30× the sparse runs |

Metrics come from `eval2.py`, on held-out hour-long streams in one causal pass:

- events per frame, idle and active;
- recall of relevant cell changes;
- TV event rate;
- fovea hits on the glyph and rule cells;
- L1 recall of primitive boundaries and L2 recall of task starts, each against random at the same rate;
- decision accuracy by age of the rule flash;
- exit-cell PSNR against copying the last frame;
- every task metric again with the main network removed;
- front-end FLOPs per frame against the dense patch model.
