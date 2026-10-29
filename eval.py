"""Evaluate tokenizers on held-out hour-long streams (single causal pass over the whole hour).

Metrics
  compression   frames per token at each level, tokens/hour, idle vs active token rate
  dynamics      next-frame PSNR given (history, a_t) on all / dynamic frames (vs copy-last-frame)
  policy        action MSE on moving frames, gripper accuracy
  memory        recall accuracy at query frames, bucketed by cue->query gap (chance 25%)
  boundaries    recall of action events / exogenous light events within +-3 frames
  systems       wall-clock, frames/s, peak VRAM, streaming-state bytes (KV caches)
"""
import argparse, json, math, os, time
import numpy as np
import torch
from vt.env import simulate, Renderer, flat, BINS, ACTION_EVENTS, EV_LIGHT
from vt.model import VideoHNet

dev = "cuda"


def load(name):
    ck = torch.load(f"runs/{name}/ckpt.pt", map_location=dev)
    m = VideoHNet(**ck["cfg"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


@torch.no_grad()
def encode_stream(model, st, env, chunk=8192):
    N = st["action"].shape[1]
    r = Renderer(dev)
    gen = torch.Generator(device=dev).manual_seed(1234 + env)
    lat, scores, prev = [], [], None
    for t0 in range(0, N, chunk):
        f, _ = r(flat(st, env, t0, min(N, t0 + chunk)), gen=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lat.append(torch.cat([model.fenc(f[i:i + 4096]) for i in range(0, len(f), 4096)]))
        ff = f if prev is None else torch.cat([prev[None], f])
        d = (ff[1:] - ff[:-1]).abs().mean((1, 2, 3))
        scores.append(torch.cat([torch.full((1,), 1e9, device=dev), d]) if prev is None else d)
        prev = f[-1]
    return torch.cat(lat)[None], torch.cat(scores)[None]


def frame_masks(infos, T):
    """Boundary masks in frame time for every level."""
    masks, frame_of = [], torch.arange(T, device=dev)[None]
    for inf in infos:
        b = inf["b"][0]
        pos = frame_of[0][b]  # frame index of each boundary at this level
        m = torch.zeros(T, dtype=torch.bool, device=dev); m[pos] = True
        masks.append(m)
        frame_of = frame_of[:, inf["order"][0][inf["cvalid"][0]]]
    return masks


def near(mask_np, idx, tol=3):
    c = np.concatenate([[0], np.cumsum(mask_np)])
    lo = np.clip(idx - tol, 0, len(mask_np)); hi = np.clip(idx + tol + 1, 0, len(mask_np))
    return (c[hi] - c[lo]) > 0


@torch.no_grad()
def eval_stream(model, ck, st, env, decode=True):
    T = st["action"].shape[1]
    torch.cuda.synchronize(); t0 = time.time()
    x, score = encode_stream(model, st, env)
    torch.cuda.synchronize(); t_enc = time.time() - t0
    act = torch.as_tensor(st["action"][env], device=dev)[None]
    prev_act = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)
    ext = {i: score > th for i, th in enumerate(ck["thresholds"])} if ck["thresholds"] else None
    torch.cuda.reset_peak_memory_stats(); t1 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, infos = model(None, prev_act, ext=ext, x_lat=x)
        pol, z = model.heads(h, act)
    torch.cuda.synchronize(); t_body = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1e9
    res = {"T": T, "t_encode_s": t_enc, "t_body_s": t_body, "fps_body": T / t_body, "peak_gb_body": peak}

    # ---- compression ----
    masks = frame_masks(infos, T)
    a_np = st["action"][env]
    moving = np.abs(a_np[:, :2]).sum(1) > 0.2
    ev = st["events"][env]
    act_ev = np.nonzero(ev & ACTION_EVENTS)[0]; light_ev = np.nonzero(ev & EV_LIGHT)[0]
    busy = near(moving | (ev > 0), np.arange(T), tol=5)  # active segments
    hours = T / 36000
    res["levels"] = []
    for l, m in enumerate(masks):
        mn = m.cpu().numpy()
        res["levels"].append({
            "frames_per_token": T / mn.sum(), "tokens_per_hour": mn.sum() / hours,
            "tok_per_min_idle": mn[~busy].sum() / max(1, (~busy).sum()) * 600,
            "tok_per_min_active": mn[busy].sum() / max(1, busy.sum()) * 600,
            "recall_action_events": near(mn, act_ev).mean() if len(act_ev) else float("nan"),
            "recall_light_events": near(mn, light_ev).mean() if len(light_ev) else float("nan"),
            "precision_events": near((ev > 0), np.nonzero(mn)[0]).mean(),
        })
    # ---- policy ----
    pol = pol[0].float().cpu().numpy()
    res["act_mse_moving"] = float(((pol[moving, :2] - a_np[moving, :2]) ** 2).mean())
    res["act_mse_idle"] = float(((pol[~moving, :2] - a_np[~moving, :2]) ** 2).mean())
    res["grip_acc"] = float(((pol[:, 2] > 0) == (a_np[:, 2] > 0.5)).mean())
    # ---- memory recall ----
    qa = st["query_answer"][env]; qt = np.nonzero(qa >= 0)[0]
    recs = []
    for t in qt:
        pos = st["agent"][env, t]
        dirs = BINS - pos; dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6
        v = pol[t, :2]; ve = a_np[t, :2]
        recs.append({"gap": int(t - st["query_cue_t"][env, t]), "ok": int(np.argmax(dirs @ v) == qa[t]),
                     "expert_ok": int(np.argmax(dirs @ ve) == qa[t])})
    res["queries"] = recs
    # ---- forward model (next frame given history + a_t) ----
    if decode:
        r = Renderer(dev)
        se_all, se_dyn, se_copy_dyn, n_all, n_dyn = 0.0, 0.0, 0.0, 0, 0
        for s in range(0, T - 1, 4096):
            e = min(T - 1, s + 4096)
            _, clean = r(flat(st, env, s, e + 1), noise=0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model.fdec(z[0, s:e]).float()
            err = ((pred - clean[1:]) ** 2).mean((1, 2, 3))
            cp = ((clean[:-1] - clean[1:]) ** 2).mean((1, 2, 3))
            dyn = cp > 0
            se_all += err.sum().item(); n_all += len(err)
            se_dyn += err[dyn].sum().item(); se_copy_dyn += cp[dyn].sum().item(); n_dyn += int(dyn.sum())
        res["psnr_all"] = 10 * math.log10(n_all / se_all)
        res["psnr_dyn"] = 10 * math.log10(n_dyn / se_dyn)
        res["psnr_dyn_copy"] = 10 * math.log10(n_dyn / se_copy_dyn)
    # ---- streaming state (what must stay resident to continue the stream) ----
    cfg = ck["cfg"]
    bytes_ = 0
    for l in range(len(cfg["chunkers"])):  # sliding-window KV of enc+dec stacks (2 layers each)
        bytes_ += 2 * 2 * 2 * 64 * cfg["dims"][l] * 2
    n_inner = int(masks[-1].sum()) if cfg["chunkers"][-1] != "all" else T
    bytes_ += cfg["n_main"] * 2 * cfg["dims"][len(cfg["chunkers"])] * n_inner * 2
    res["stream_state_mb"] = bytes_ / 1e6
    res["inner_tokens"] = n_inner
    return res, masks, score[0].cpu().numpy()


def summarize(rs):
    out = {}
    for k in ["psnr_all", "psnr_dyn", "psnr_dyn_copy", "act_mse_moving", "act_mse_idle", "grip_acc", "fps_body",
              "peak_gb_body", "stream_state_mb", "inner_tokens", "t_body_s", "t_encode_s"]:
        if k in rs[0]:
            out[k] = float(np.mean([r[k] for r in rs]))
    out["levels"] = [{k: float(np.nanmean([r["levels"][l][k] for r in rs])) for k in rs[0]["levels"][l]}
                     for l in range(len(rs[0]["levels"]))]
    q = [x for r in rs for x in r["queries"]]
    buckets = [(0, 1500), (1500, 6000), (6000, 18000), (18000, 10 ** 9)]
    out["memory"] = {f"{a}-{b}": {"acc": float(np.mean([x["ok"] for x in q if a <= x["gap"] < b])) if any(a <= x["gap"] < b for x in q) else None,
                                  "n": sum(a <= x["gap"] < b for x in q)} for a, b in buckets}
    out["memory"]["expert_acc"] = float(np.mean([x["expert_ok"] for x in q])) if q else None
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--n_env", type=int, default=6)
    ap.add_argument("--suite", default="hour", choices=["hour", "hour_longgap", "day"])
    a = ap.parse_args()
    if a.suite == "hour":
        st = simulate(a.n_env, 36000, seed=9999, gap=(100, 1500))
    elif a.suite == "hour_longgap":
        st = simulate(a.n_env, 36000, seed=8888, gap=(1500, 24000), cue_rate=1 / 300)
    else:
        st = simulate(1, 864000, seed=7777, gap=(100, 36000), cue_rate=1 / 1200)
    for name in a.names:
        model, ck = load(name)
        rs = []
        for e in range(st["action"].shape[0]):
            r, masks, score = eval_stream(model, ck, st, e, decode=(a.suite != "day"))
            rs.append(r)
            if e == 0:
                np.savez_compressed(f"runs/{name}/{a.suite}_env0.npz", masks=np.stack([m.cpu().numpy() for m in masks]),
                                    score=score, events=st["events"][0], action=st["action"][0])
        s = summarize(rs)
        json.dump({"summary": s, "per_stream": rs}, open(f"runs/{name}/eval_{a.suite}.json", "w"), default=float)
        print(name, json.dumps(s, default=float))
        del model; torch.cuda.empty_cache()
