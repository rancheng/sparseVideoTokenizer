"""Held-out evaluation on real robot video: per dataset, the test episodes laid end to end (up to --max_frames), one
causal pass (front in chunks with carried state, then the temporal model).

  psnr / psnr_copy          next-frame PSNR of the forward model over all cells, against copying the current frame
  psnr_move / ..._copy      the same on cells that change by more than 0.02 (mean abs) at t+1
  act_nmse_vision           action MSE (normalised units, real dims only) with every previous-action input zeroed
  act_nmse_prev             ... with the previous actions given; act_nmse_copy: predicting a_t = a_{t-1}
  events / hires per frame, front MFLOP per frame against every cell at full resolution
  event_recall / precision  events vs cells that changed (> 0.02 since t-1); base rate = share of changed cells
  tok_per_s_L1/L2           temporal tokens per second, in the 25% most and least moving frames
usage: python -m real.eval_real <run> [<run> ...]
"""
import argparse, json, math, os
import numpy as np
import torch
from vt.model2 import SparseVideoModel, to_cells
from vt.env2 import G
from vt.stream2 import StreamingSparseModel

dev = "cuda"
DATA = "/home/ubuntu/data/robot_video/prep"


def load(name):
    ck = torch.load(f"runs_real/{name}/ckpt.pt", map_location=dev)
    c = ck["cfg"]
    m = SparseVideoModel(c["front"], c["temporal"], n_dom=c["n_dom"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def front_cost(m):
    try:
        c = StreamingSparseModel(m).cost
    except Exception:
        return None
    return c


@torch.no_grad()
def eval_set(m, ck, tag, max_frames, chunk=512):
    meta = json.load(open(f"{DATA}/{tag}/meta.json"))
    a0 = meta["starts"][meta["test_from"]]; a1 = min(meta["n_frames"], a0 + max_frames)
    fr = np.load(f"{DATA}/{tag}/frames.npy", mmap_mode="r")[a0:a1]
    act = torch.as_tensor(np.load(f"{DATA}/{tag}/actions.npy")[a0:a1], device=dev)[None]
    ad = meta["act_dim"]
    frames = torch.as_tensor(np.ascontiguousarray(fr), device=dev).permute(0, 3, 1, 2).float().div_(255)[None]
    T = frames.shape[1]
    dom = torch.tensor([ck["cfg"]["doms"].index(tag)], device=dev) if tag in ck["cfg"]["doms"] else None
    fs, bs, us, cells, state = [], [], [], [], None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t0 in range(0, T, chunk):
            f, fi, state = m.front(frames[:, t0:t0 + chunk], t0=t0, state=state)
            fs.append(f); bs.append(fi["b"]); us.append(fi["use"]); cells.append(fi["cell"])
        x = torch.cat(fs, 1)
        if m.dom is not None and dom is not None:
            x = x + m.dom(dom)[:, None].to(x.dtype)
        b, use, cell = torch.cat(bs, 1)[0], torch.cat(us, 1)[0], torch.cat(cells, 1)[0]
        prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)
        h0, infos = m.temporal(None, torch.zeros_like(prev), x_lat=x)      # vision only
        pol0, z = m.temporal.heads(h0, act)
        h1, _ = m.temporal(None, prev, x_lat=x)
        pol1, _ = m.temporal.heads(h1, act)
    r = {"tag": tag, "frames": T}
    a = act[0, :, :ad].float()
    r["act_nmse_vision"] = ((pol0[0, :, :ad].float() - a) ** 2).mean().item()
    r["act_nmse_prev"] = ((pol1[0, :, :ad].float() - a) ** 2).mean().item()
    r["act_nmse_copy"] = ((a[1:] - a[:-1]) ** 2).mean().item()
    r["act_nmse_zero"] = (a ** 2).mean().item()
    # forward model on every cell of every frame
    cc, cn = to_cells(frames[:, :T - 1])[0], to_cells(frames[:, 1:])[0]           # (T-1, J, 3,16,16)
    se, se_c, sm, sm_c, nm = 0.0, 0.0, 0.0, 0.0, 0
    J = G * G
    for t0 in range(0, T - 1, 256):
        t1 = min(T - 1, t0 + 256); n = t1 - t0
        tt = torch.arange(t0, t1, device=dev).repeat_interleave(J); jj = torch.arange(J, device=dev).repeat(n)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = m.cdec(z[0, tt], cell[tt, jj], jj, cc[tt, jj]).float()
        tg = cn[tt, jj]; e = ((pred - tg) ** 2).mean((1, 2, 3)); ec = ((cc[tt, jj] - tg) ** 2).mean((1, 2, 3))
        mv = (cc[tt, jj] - tg).abs().mean((1, 2, 3)) > 0.02
        se += e.sum().item(); se_c += ec.sum().item(); sm += e[mv].sum().item(); sm_c += ec[mv].sum().item(); nm += int(mv.sum())
    N = (T - 1) * J
    r["psnr"] = 10 * math.log10(N / se); r["psnr_copy"] = 10 * math.log10(N / max(se_c, 1e-12))
    r["psnr_move"] = 10 * math.log10(nm / sm) if nm else None; r["psnr_move_copy"] = 10 * math.log10(nm / max(sm_c, 1e-12)) if nm else None
    r["moving_cell_share"] = nm / N
    # sparsity and where events go
    r["events_per_frame"] = b.float().sum(-1).mean().item(); r["hires_per_frame"] = use.float().sum(-1).mean().item()
    chg = torch.zeros_like(b); chg[1:] = (to_cells(frames[:, 1:])[0] - to_cells(frames[:, :-1])[0]).abs().mean((-1, -2, -3)) > 0.02
    r["changed_share"] = chg.float().mean().item()
    r["event_recall"] = (b & chg).sum().item() / max(1, chg.sum().item())
    r["event_precision"] = (b & chg).sum().item() / max(1, b.sum().item())
    r["hires_on_changed"] = (use & chg).sum().item() / max(1, use.sum().item())
    r["event_map"] = b.float().mean(0).view(G, G).cpu().numpy().round(3).tolist()
    r["hires_map"] = use.float().mean(0).view(G, G).cpu().numpy().round(3).tolist()
    r["change_map"] = chg.float().mean(0).view(G, G).cpu().numpy().round(3).tolist()
    c = front_cost(m)
    if c:
        per = J * (c["cheap"] + c["router"]) + r["events_per_frame"] * (c["fovea"] + sum(c["block"]) + c["pool_row"]) + r["hires_per_frame"] * c["hires"] + c["pool"]
        dense = J * (c["cheap"] + c["fovea"] + c["hires"] + sum(c["block"]) + c["pool_row"]) + c["pool"]
        r["front_mflops"] = per / 1e6; r["dense_front_mflops"] = dense / 1e6
    # temporal tokens vs motion
    fps = meta["fps_out"]
    motion = (frames[0, 1:] - frames[0, :-1]).abs().mean((1, 2, 3)); motion = torch.cat([motion[:1], motion])
    lo, hi = torch.quantile(motion.float(), torch.tensor([0.25, 0.75], device=dev))
    frame_of = torch.arange(T, device=dev)
    for li, inf in enumerate(infos):  # boundary masks in frame time (as eval.frame_masks)
        bb = torch.zeros(T, dtype=torch.bool, device=dev); bb[frame_of[inf["b"][0]]] = True
        frame_of = frame_of[inf["order"][0][inf["cvalid"][0]]]
        r[f"tok_per_s_L{li + 1}"] = bb.float().mean().item() * fps
        r[f"tok_per_s_L{li + 1}_moving"] = bb[motion >= hi].float().mean().item() * fps
        r[f"tok_per_s_L{li + 1}_still"] = bb[motion <= lo].float().mean().item() * fps
    mid = T // 2
    r["sample_frame_index"] = int(a0 + mid)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--sets", default="all")
    ap.add_argument("--max_frames", type=int, default=3000)
    a = ap.parse_args()
    tags = sorted(d for d in os.listdir(DATA) if os.path.exists(f"{DATA}/{d}/meta.json"))
    if a.sets != "all":
        tags = [t for t in tags if t in a.sets.split(",")]
    for name in a.names:
        m, ck = load(name)
        res = []
        for t in tags:
            r = eval_set(m, ck, t, a.max_frames)
            res.append(r)
            print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("_map")}), flush=True)
        json.dump(res, open(f"runs_real/{name}/eval.json", "w"))
