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

## Results, round 1 (held-out 1-hour streams, 4 envs, one causal pass; `runs2/eval_hour_*.log`)

Training: 7,000 steps, three stages (rule fixed / short rule period / normal), `--fov_warm 1000`.

### Sparsity and the fovea (H4, H6)

| run | events / frame (of 64) | high-res cells / frame | TV event rate | high-res share on TV | rule cell high-res per flash | glyph read before decision | front MFLOP / frame |
|---|---|---|---|---|---|---|---|
| st_learned (tied router, STE fovea) | 4.1 | 2.0 | 1.00 | 49% | 0% | 100% | 11.8 |
| st_rel (+ relevance logit) | 4.3 | 2.1 | 1.00 | 47% | 1% | 99% | 14.1 |
| **st_task** (counterfactual task-credit fovea) | 4.3 | 2.0 | 0.90 | **20%** | **35%** | 100% | 11.9 |
| st_lowres (no fovea) | 3.7 | 0 | 1.00 | — | — | **53% (chance)** | 8.4 |
| st_heur (pixel-change cells, all high-res) | 4.2 | 4.2 | 0.51 | 12% | 100% | 100% | 15.1 |
| st_dense | 64 | 64 | 1.00 | 2% | 100% | 100% | 154 |

- **H4 holds:** the glyph is only readable with the fovea (53% → 100%), and 2 high-res cells per frame suffice:
  about 8% of the dense front's compute.
- **The STE fovea falls for the noisy TV.** It learned only on cells it already picked, and from every loss, pixels
  included. The TV does help next-frame pixels: it holds its value half the time. The rule cell, an event on every
  flash, got high resolution 0% of the time.
- **The fix, `--fov_credit task`.** In training only, every event cell is encoded at high resolution; a 0/1 gate (a
  leaf) keeps only the picked ones in the forward pass, so training equals inference. The score regresses
  −∂(action losses)/∂gate on *every* event cell (counterfactual, first order), with Gumbel top-k exploration and the
  cell position as input. The TV's share fell from 49% to 20%, and the rule cell went from 0% to 35%.
- **H6 not shown:** the learned relevance logit (`st_rel`) did not suppress the TV.

### Rule memory (H3): a flaw in the testbed, not (only) the model

Decision accuracy stayed at ≈ 0.5 for every model, the dense front and a 30× decision weight included. The scores
were identical to three decimals across models: all of them apply one fixed glyph→bin mapping. Linear probes
(66 flashes, grouped split) show where the rule is lost:

| probe of the current rule | st_dense | st_heur | st_learned |
|---|---|---|---|
| rule cell state at the flash | 1.00 | 0.94 | 0.97 |
| pooled frame vector at the flash | 0.81 | 0.92 | 0.51 |
| temporal output h, 5 frames later | 0.53 | 0.47 | 0.44 |
| h at the decision | 0.34 | 0.36 | 0.50 |

Flashes do become tokens: L2 tokens land on 68% (heur) and 52% (learned) of flashes, against 27% for random cuts.
The cause is the mirror rule: bin = glyph XOR rule. The heading head is linear in h, and the 8 (rule, glyph) → bin
cases are not linearly separable from additive rule and glyph features. Remembering the rule pays nothing at first
order until the network builds the interaction, so the toy tested XOR learning, not memory. The v1/v4 cue mapped
straight to the answer, and memory learned there.

**Fix:** `rule_mode=pair`. The rule picks the left or right pair of bins and the glyph picks within the pair, so each
helps on its own. Runs: `st_pair_heur`, `st_pair_task`.

### Object permanence (H1)

- **The exit test was broken.** Items crawled out of the tunnel at 0.13–0.5 px/frame, so next-frame prediction at
  exits was solved by copying: the model scored 28.6 dB, noisy copy 28.7 dB.
- **`--pop_exit`.** The item now appears in one frame, announced by a lamp one frame earlier, so the timing is
  local. `exit0_psnr` scores only that frame: copy 20.7 dB.

`st_pop` (pop exits, 256 exit cells sampled per step) results:
- **Exit frame:** 21.9 dB against copy 20.7, and **21.9 with the main network removed**. The gain is a generic
  "an item appears" prediction, not memory.
- **Colour probes (4-way, chance 0.25):** frame vector at the entry 0.63; h at the entry 0.30; h at the lamp 0.25.
- **Why the dense loss still doesn't pay for memory:** the cell decoder gets each cell's own state directly, so h
  only has to carry what that state lacks. That is needed on a handful of exit frames. It is the v1 situation again,
  one level down.
- **The temporal hierarchy did become event-driven:**
  - L1: 43 tokens/min idle vs 227/min active (`st_task`: 137 vs 200).
  - L2: catches 98% of task starts, against 28% for random cuts at the same rate.

### Rule memory with the additive rule (`rule_mode=pair`)

The rule is now learnable. Training decision accuracy (flash inside the clip) climbs past the fixed-rule stage:
`st_pair_heur` from 0.42 to about 0.8, `st_pair_task` from 0.23 to about 0.7–0.9. Accuracy on the glyph half of the
decision is 1.00 and 0.94.

It is only remembered while the flash is recent. Rule half (`side_ok`), 48 held-out 2,048-frame streams:

| flash age (frames) | 0–600 | 600–1500 | 1500–3000 |
|---|---|---|---|
| st_pair_heur | 0.68 | 0.50 | 0.22 |
| st_pair_heur, main removed | 0.66 | 0.33 | 0.11 |
| st_pair_task | 0.62 | 0.42 | 0.00 |

The hour streams give 0.55 overall. Removing the main network barely changes the short-age accuracy, so the rule is
carried by the local stages' windowed attention (the L2 window spans about 1,000 frames). Once the flash leaves that
window, the decision falls back to a prior. The main network and the memory are not used for it.

Every v2 test now shows the same thing: the local stages solve what they can see, and no loss trains the main
network to reach beyond them. Next test: `--w_main`, which gives the main network a loss at its own rate (predict the
next event token, as next-byte loss at an H-Net boundary does on text). Runs `st_pp` vs `st_pp_main` (pair rule, pop
exits, task-credit fovea).

### Main network: does its own loss make memory pay? (`st_pp` vs `st_pp_main`)

Both runs use the pair rule, pop exits (256 exit cells per step) and the task-credit fovea. `st_pp_main` adds
`--w_main 1`: the main network predicts the next event token (cosine, stop-gradient target).

| | st_pp | st_pp_main |
|---|---|---|
| rule cell high-res per flash (hour) | **60%** | 5% |
| TV high-res share | 21% | 14% |
| rule half, 2k streams, age 0–600 (main removed) | **0.71 (0.48)** | 0.26 (0.24) |
| rule half, 2k streams, all ages | 0.64 | 0.46 |
| glyph half, 2k streams | 0.99 | 0.47 |
| rule half, hour streams | 0.54 | 0.45 |
| exit frame PSNR / copy / main removed | 21.8 / 20.7 / 21.8 | 21.8 / 20.7 / 21.8 |
| item colour probe in h at the lamp (chance 0.25) | 0.22 | 0.24 |
| L1 tokens/min idle vs active | 34 vs 323 | 10 vs 137 |
| L2 task-start recall (random at same rate) | 0.59 (0.28) | 0.16 (0.33) |

- **`st_pp` is the first run where the main network matters for a decision.** Removing it drops rule decisions on
  recent flashes from 0.71 to 0.48. When the rule matters additively, the fovea also finds the rule cell on 60% of
  flashes.
- **Memory still does not reach the exit or old flashes.** The exit frame is predicted no better with main than
  without, and the item's colour is not in h. Hour-long streams stay at chance: training clips hold at most 2,048
  frames and one or two flashes, while an hour holds about 30 flashes, all in the memory.
- **`--w_main 1` hurt.** The next-token loss flowed back into the token encoders and overrode task features: the
  glyph half fell from 0.99 to 0.47, and the policy did not recover in training. Queued:
  - `--main_detach`: the loss trains only the main network, on a second pass over detached tokens;
  - `--w_main 0.1`.

### What holds so far

| assumption | status | evidence |
|---|---|---|
| per-cell event routing is sparse and recalls relevant change | **holds** | 4 of 64 cells per frame, 0.92–0.99 recall of relevant cell changes, 12 vs 150 MFLOP per frame |
| a learned fovea finds the few cells that need high resolution (H4) | **holds, with task credit** | glyph unreadable without it (0.53); counterfactual task credit moves high-res off the TV (49% → 20%) and onto the rule (0% → 60%) |
| a relevance logit lets the cell router ignore the noisy TV (H6) | not shown | `st_rel` TV event rate 1.00 |
| temporal chunking is event-driven; L2 matches tasks (H2) | **partly** | L1 34 tokens/min idle vs 323 active; L2 task recall 0.59–1.00 against random 0.28–0.51, but not in every run |
| the main network and memory carry long-range content (H1, H3) | **not yet** | short-range rule only (≤ 600 frames); exits and colour not remembered; hour streams at chance |
