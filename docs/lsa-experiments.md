# Latent Sparse Attention on the video memory

Paper: Kubicek et al., *Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention*,
[arXiv:2610.42147](https://arxiv.org/abs/2610.42147). Keys are clustered by content (Ward's method) once per block of
consecutive keys; a query scores bucket centroids, expands the best buckets into k keys, and also reads a recent window.

Here it replaces the frame-rate memory read of the temporal model (`MemoryRead`, `vt/model.py`): blocks of 32 entries,
8 buckets per block, window of 32 entries. Code: `vt/lsa.py` (index), `exp_lsa.py` (experiments), `bench_lsa.py`
(latency), `tests/test_lsa.py`. Native sparse training: `train.py --sparse_read 32 [--sparse_rec 0.05]`.

Test set: 12 held-out hour streams of the cue→query task with gaps of 1.5k–6k frames (where `hnet_v5` answers with the
full read), 102 queries, chance 0.25; projections are fitted on 4 other streams. SE is about ±4%.

## 1. One hour of memory (`hnet_v5`, `runs_lsa/budget_hnet_v5.json`)

Accuracy by long-range entries read per frame (k), plus the 32 most recent; full read 0.78 (868 entries on average):

| selection | k=8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| recent entries only (same count) | 0.27 | 0.27 | 0.27 | 0.35 | 0.50 |
| oracle (top-k of the full attention) | 0.78 | 0.78 | 0.78 | 0.78 | 0.78 |
| **LSA on the model's own keys, no training** | 0.62 | 0.69 | **0.76** | 0.78 | 0.78 |
| LSA, distilled on uniform frames (the paper's D-LSA recipe) | 0.41 | 0.49 | 0.62 | 0.74 | 0.78 |
| LSA, distilled with half the queries at query frames | 0.63 | 0.72 | 0.78 | 0.78 | 0.78 |
| LSA, PCA projection (the paper's training-free variant) | 0.27 | 0.27 | 0.27 | 0.27 | 0.29 |
| block sparse (4 consecutive entries, distilled) | 0.52 | 0.62 | 0.75 | 0.78 | 0.78 |

- Untrained LSA on the raw keys reaches the full read at k = 32, scoring about a quarter as many centroids as entries.
- **Uniform distillation hurts**: the head-mean attention it matches is dominated by idle frames; the rare query frames
  that need retrieval get almost no weight. Weighting query frames fixes it.
- **PCA fails completely**: the leading directions of the keys are the idle background; the cue direction is dropped.
- The oracle matches the full read with 48% of the attention mass: mass is a poor proxy for what the task needs.

## 2. Long memory: stale cues (`stale_hnet_v5.json`, `stale2_hnet_v5.json`)

0–15 hours of other streams' memory (shifted back in time, about 130 old cues) are prepended to each test hour:

| | 0 h | 1 h | 3 h | 7 h | 15 h |
|---|---|---|---|---|---|
| full read | 0.78 | 0.74 | 0.74 | 0.75 | 0.74 |
| LSA, content only, k=32 | 0.76 | 0.58 | 0.45 | 0.46 | **0.36** |
| block sparse, k=32 | 0.75 | 0.52 | 0.37 | 0.37 | 0.34 |
| **LSA + recency, k=32** | 0.78 | 0.74 | 0.74 | 0.75 | **0.74** |
| LSA + recency, k=8 | 0.78 | | | 0.75 | 0.74 |
| LSA + recency, tree search, k=32 | 0.78 | | | 0.75 | 0.74 (1,198 centroids scored vs 9,115) |

Recency: bucket score − λ·log₂(1 + frames since the bucket's newest entry), λ = 0.02–0.1 all give the same result.

- Content-only selection (the paper's N-LSA scores without position) fills up with old cues: at 15 h, 31% of the
  long-range picks are older cues and 0.2% the current one. All cue tokens look alike; content cannot tell the latest.
- With recency, 40 entries out of ~36,000 give the full-read accuracy at 15 h.
- Expanding buckets newest-first does not help; the ranking of buckets is the problem.

## 3. Latency (`bench.json`, one read layer, one query, idle 4090)

| memory | entries | KV cache | full read | LSA flat | LSA tree | 64-d index |
|---|---|---|---|---|---|---|
| 1 hour | 2,342 | 0.005 GB | 0.12 ms | 0.08 ms | 0.12 ms | 0.1 MB |
| 1 day | 56,208 | 0.12 GB | 2.6 ms | 0.08 ms | 0.12 ms | 2 MB |
| 1 week | 393,456 | 0.81 GB | 17 ms | 0.10 ms | 0.12 ms | 13 MB |
| 1 month | 1,686,240 | 3.45 GB | 74 ms | 0.13 ms | 0.12 ms | 56 MB |

At one hour the H-Net chunking already compresses 36,000 frames to ~2,300 entries and LSA buys nothing. From about a
day on it matters; at a month one full read layer uses 74 of the 100 ms a 10 Hz frame allows. LSA does not shrink the
KV cache; keeping centroids on the GPU and fetching the few selected entries from host memory is untested.

## 4. Native sparse training and a seed check (`native_*.json`, `runs/eval_longgap_v5s.log`)

`hnet_v5s`: the v5 recipe with the read restricted to 32 selected + 32 recent entries during training (selection as a
mask over dense logits: exact, 0.24 s/step vs 0.18 dense). `hnet_v5sr` adds recency in training. Seed 1 repeats both.

| model | full read, 0 h | full, 15 h | content k=32, 15 h | + recency k=32, 15 h | + recency k=8, 15 h |
|---|---|---|---|---|---|
| hnet_v5 (dense, seed 0) | 0.78 | 0.74 | 0.34 | 0.74 | 0.74 |
| hnet_v5_s1 (dense, seed 1) | 0.99 | 0.82 | 0.28 | 0.70 | 0.57 |
| hnet_v5s (sparse, seed 0) | 0.97 | 0.75 | 0.30 | 0.87 | 0.86 |
| hnet_v5s_s1 (sparse, seed 1) | 0.77 | 0.64 | 0.44 | 0.62 | 0.60 |
| hnet_v5sr (sparse + recency) | 0.67 | 0.62 | 0.38 | 0.60 | 0.60 |

- **No evidence that native sparse training helps.** Seeds of the same recipe differ more (0.78 vs 0.99) than the two
  recipes do. The single-seed claim (0.97 vs 0.78) is retracted, as is "sparse reads make the chunker write more entries
  at cues" (seed 0: 3.0× vs 1.3×; seed 1: dense 4.5×, sparse 3.3×).
- Holds on every model: content-only selection collapses with long history; recency recovers most of it, by how much
  depends on the model (k = 32 within −0.12 to +0.12 of the full read). The full read itself is diluted by stale cues.
- The cue colour is linearly decodable from main-network outputs 1.5k–6k frames after the cue (0.60–0.76, chance 0.25)
  in every model, so a selection does not have to hit the cue token itself.
- The three sparse-trained models were trained before the bucket-eligibility fix below.

## Bug found by the unit test

Buckets were searchable only when their whole block was older than the recent window, so entries of the block that
straddles the window edge were unreachable. Fixed to the paper's rule (completed blocks are searchable; keys already in
the window are skipped). `tests/test_lsa.py` checks that the sparse read equals the dense read when it selects every
entry. All tables above are from the rerun after the fix (they moved by at most 0.01).
