# Continuing the experiments on the RTX 3090 machine (`ubuntu-3090`)

The experiments were moved from the RTX 4090 machine on 2026-11-03. Paths are the same on both machines:

| what | path |
|---|---|
| repository (git, remote `origin` = `rancheng/sparseVideoTokenizer`, private) | `/home/ubuntu/Codes/videoTokenizer` |
| robot video data (not in git) | `/home/ubuntu/data/robot_video/{raw,prep}` |
| checkpoints (not in git) | `runs/*/ckpt.pt`, `runs2/*/ckpt.pt`, `runs_real/*/ckpt.pt` |

Read first: `README.md`, `docs/experiment-log.md`, `docs/v2-spatiotemporal.md`, `docs/lsa-experiments.md`.

## Getting the code and checkpoints (Hugging Face, no SSH)

Everything is in the private Hugging Face model repo **`rancheng222/sparseVideoTokenizer`**: the working tree at
commit time (code, docs, logs, eval results, all 75 checkpoints, about 7.5 GB) and `videoTokenizer.bundle`, a git
bundle with the full history. Not included: `.venv`, `runs_lsa/*.pt` (LSA caches, rebuilt by `exp_lsa.py`), the
robot video data (section 2).

```bash
hf auth login                                   # account rancheng222 (the repo is private)
cd /home/ubuntu/Codes/videoTokenizer            # this folder may already hold a partial copy from an aborted rsync:
                                                # the download overwrites it with the same files
hf download rancheng222/sparseVideoTokenizer --local-dir /home/ubuntu/Codes/videoTokenizer
# git history: restore it from the bundle, keep GitHub as origin
git init -q && git fetch -q videoTokenizer.bundle "refs/heads/*:refs/heads/*" && git reset -q --hard main
git remote add origin https://github.com/rancheng/sparseVideoTokenizer.git   # push needs GitHub credentials
```
`hf` comes with `pip install huggingface_hub` (section 1); install that first if `hf` is missing.

## 0. Machine state when handed over

- **GPU driver is not loaded.** `nvidia-smi` fails. `nvidia-driver-595-open` (595.91.07) is installed, but no NVIDIA
  kernel module exists for the running kernel `7.0.0-38-generic` (`lsmod | grep nvidia` is empty, nothing under
  `/usr/lib/modules/$(uname -r)/updates/dkms`). Secure Boot is off. Fix (needs sudo), then reboot:
  ```bash
  sudo apt install --reinstall nvidia-dkms-595-open     # rebuilds the module for the running kernel
  # or, if Ubuntu ships a prebuilt one for this kernel:
  # sudo apt install linux-modules-nvidia-595-open-$(uname -r)
  sudo reboot
  nvidia-smi                                            # must list the RTX 3090
  ```
- 24 GB GPU (same as the 4090), but **31 GB RAM** (the 4090 box had 125 GB): keep preprocessing to 2 workers.
- 3090 is Ampere: bf16 autocast works; expect about 0.6x the 4090's speed.

## 1. Python environment

```bash
cd /home/ubuntu/Codes/videoTokenizer
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install numpy pandas pyarrow pillow huggingface_hub
# versions used on the 4090: torch 2.11.0+cu128, numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.0, pillow 13.0.0
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```
`ffmpeg` with the dav1d AV1 decoder is already installed (LeRobot videos are AV1).

## 2. Robot video datasets (LeRobot v3 on the Hugging Face Hub, public)

`real/datasets.txt` lists 17 streams: 15 fixed third-person cameras and 2 moving cameras (ALOHA mobile head camera,
DROID wrist camera), one camera each. Only `meta/`, `data/` and the first video file of that camera are downloaded
(about 4 GB in total).

```bash
cd /home/ubuntu/Codes/videoTokenizer
PATH=$PWD/.venv/bin:$PATH ./real/download.sh          # -> /home/ubuntu/data/robot_video/raw, prints DL_DONE
PREP_WORKERS=2 .venv/bin/python real/prep_lerobot.py /home/ubuntu/data/robot_video/raw /home/ubuntu/data/robot_video/prep real/datasets.txt
```
Preprocessing (`real/prep_lerobot.py`): decode with ffmpeg, short side to 128, centre crop 128×128, subsample to
about 10 fps, up to 30,000 frames per stream, actions z-scored per dataset and padded to 14 dims, last 10% of episodes
held out. Output per stream: `frames.npy` (uint8 N×128×128×3), `actions.npy`, `meta.json`. Expected result (from
the 4090): 17 streams, about 380,000 frames, 18 GB. Check with
`ls /home/ubuntu/data/robot_video/prep/*/meta.json | wc -l` → 17.

## 3. Where the experiments stopped

### Done on the 4090

- Everything in `docs/experiment-log.md`, `docs/v2-spatiotemporal.md` and `docs/lsa-experiments.md`, including the
  long-clip runs (`st_pp_L16`, `st_pp_L2`, `st_heur_L16`) and the LSA study (`hnet_v5*`).
- Real robot video, 3 of 7 baseline runs trained (8,000 steps, B 8, T 256, all 17 streams, `run_real.sh`),
  **not evaluated yet**. Last training step:

  | run | front | train cell MSE (copy) | train action MSE | events / cell | high-res / cell | time |
  |---|---|---|---|---|---|---|
  | `rs_base` | learned router + task-credit fovea | 0.0075 (0.0087) | 0.126 | 0.091 | 0.026 | 18 min |
  | `rs_dense` | every cell, full resolution | 0.0059 (0.0085) | 0.127 | 1.0 | 1.0 | 20 min |
  | `rs_heur` | pixel-change cells, all high-res | 0.0073 (0.0085) | 0.141 | 0.081 | 0.081 | 11 min |

  Training action MSE includes the previous action as input half of the time (`--act_drop 0.5`), so it is easy; the
  held-out metric that matters is `act_nmse_vision` (previous actions zeroed).

### Not done (in this order)

1. **Evaluate the three real baselines** (a few minutes each):
   ```bash
   .venv/bin/python -m real.eval_real rs_base rs_dense rs_heur     # -> runs_real/<run>/eval.json, one line per stream
   ```
   Per stream: next-frame PSNR vs copying, action nMSE from vision only, events and high-res cells per frame, event
   recall/precision against changed cells, per-cell event / high-res / change heatmaps, temporal tokens per second in
   moving vs still frames, front MFLOP vs dense.
   **Check first:** on a 300-step PushT model the level-1 temporal router fired only on its heartbeat (1 in 64
   frames). If `tok_per_s_L1` is about `fps/64` on most streams, the temporal router is not firing on real video.
2. **Remaining real baselines**: `./run_real.sh` (skips runs whose `ckpt.pt` exists): `rs_small`, `rs_large`
   (model size), `rs_d1` (PushT only), `rs_d4` (4 streams) for data scaling, then the evals of all seven.
3. **JEPA objective on the toy stream**: `./run_jepa.sh` (about 50 min per run on the 4090), then
   ```bash
   .venv/bin/python eval_jepa.py st_pp st_pj0 st_pj1 st_pj2 --pop_exit
   .venv/bin/python eval2.py st_pj0 st_pj1 st_pj2 --rule_mode pair --pop_exit --tag hour_pp
   ```
   Runs: `st_pj0` latent prediction only, `st_pj1` + multi-step inverse dynamics, `st_pj2` + future actions. All
   with `--w_pix 0`, against `st_pp` (pixel objective, same front and curriculum).

## 4. The JEPA objective (`--obj jepa` in `train2.py`, `vt/jepa.py`)

Why: the pixel objective never needed the main network (removing it cost 0.03 dB) and the cell decoder could read
each cell's own state, so nothing pushed the model to compress or to remember. See the report's discussion.

- Target: an EMA copy of the sparse front encodes the predicted frame densely (`SparseFront.encode_dense`), layer
  norm, no gradient. EMA 0.996, updated after every step.
- Predictor: from h_t, the actions a_t..a_{t+k-1}, cell position and horizon k ∈ {1, 4, 16, 64}. It sees the target
  cell's current state only for k = 1 (removes the shortcut). Loss: smooth L1, averaged over horizons.
- Multi-step inverse dynamics (`--w_inv`): a_t from the front's action-free frame vectors f_t, f_{t+k}, k ≤ 16.
- Future actions (`--w_fut`): a_{t+k} from h_t, k ∈ {4, 16, 64}.
- Surprise router (`--w_surp`): BCE pushing the cell router towards the top 1/N_cell cells by one-step latent error.
- The task-credit fovea is credited for behaviour losses (actions + inverse + future), not for latent prediction.
- Logged: `j1 j4 j16 j64` (per-horizon loss), `y_std` (target spread across cells and frames: watch for collapse),
  `l_inv l_fut l_surp`.

**What would show it works** (`eval_jepa.py`):

| test | pixel objective `st_pp` (measured) | wanted |
|---|---|---|
| noisy TV share of events / high-res | 25% / 21% | falls |
| rule symbol: events while shown | 20% of shown frames | rises, with `st_pj2` (future actions) |
| main network removed: j16, j64 | (pixel: 0.03 dB) | clearly worse without main |
| collapse | — | `y_std` and `erank_f` stay up |

Predicted budget per variant: pure JEPA keeps arm, item, lights and belt but not the TV or the rule; inverse dynamics
keeps the arm and touched items; future actions should add the incoming item and the rule symbol. Only the last one
appearing on its own would show that action relevance was learned rather than built in.

Next steps after that: JEPA on the real robot streams (`real/train_real.py` does not have `--obj jepa` yet);
optionally a frozen pretrained V-JEPA 2 encoder as the target (check its licence first), reported separately from
the self-supervised result; a rate term (bits per token and per event) in place of the H-Net ratio loss.

## 5. Practical notes

- One GPU job at a time on 24 GB. `hnet_v5` at B 2, T 16384 runs out of memory; use B 1.
- `pkill -f <pattern>` (and `pgrep -f` in a wait loop) matches the shell running it. Stop jobs by PID, or use a
  bracket pattern: `pgrep -f "[t]rain2.py"`.
- Queue scripts skip finished runs (they check for `ckpt.pt`), so a queue can be restarted after a crash.
- Logs: `runs*/<run>/stdout.log` (one JSON line per 50 steps), `train_log.json`, `eval*.json`. `*.log` files outside
  `runs*/<run>/stdout.log` are gitignored.
- The interactive report (claude.ai artifact, private) was built by a script that is not in the repository.
