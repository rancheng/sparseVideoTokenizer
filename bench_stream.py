"""Stream a held-out hour frame-by-frame through the event-driven engine; measure sparse vs dense cost.

Outputs runs/<name>/stream_<frames>.json with per-component FLOPs, FLOPs/frame idle vs active, latency
percentiles, state growth, and memory-recall accuracy computed from the *streamed* policy.
"""
import argparse, json, time
import numpy as np
import torch
from vt.env import simulate, Renderer, flat, BINS
from vt.stream import StreamingVideoHNet
from eval import load

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--frames", type=int, default=36000)
ap.add_argument("--seed", type=int, default=9999)
a = ap.parse_args()

st = simulate(1, a.frames, seed=a.seed)
r = Renderer()
gen = torch.Generator(device="cuda").manual_seed(1234)
act = torch.as_tensor(st["action"][0], device="cuda")
prev = torch.cat([torch.zeros_like(act[:1]), act[:-1]])
moving = np.abs(st["action"][0, :, :2]).sum(1) > 0.2
ev = st["events"][0]
busy = np.convolve((moving | (ev > 0)).astype(float), np.ones(11), "same") > 0

for name in a.names:
    model, ck = load(name)
    eng = StreamingVideoHNet(model, ck["thresholds"])
    T = a.frames
    fl = np.zeros(T); lat = np.zeros(T); pols = np.zeros((T, 3), np.float32)
    growth = []
    CH = 2048
    for c0 in range(0, T, CH):
        frames, _ = r(flat(st, 0, c0, min(T, c0 + CH)), gen=gen)
        for i in range(len(frames)):
            t = c0 + i
            f0 = sum(eng.flops.values())
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pol, _, _ = eng.step(frames[i], prev[t], act[t])
            torch.cuda.synchronize(); lat[t] = time.perf_counter() - t0
            fl[t] = sum(eng.flops.values()) - f0
            pols[t] = pol.float().cpu().numpy()
        growth.append({"t": min(T, c0 + CH), **eng.state_bytes()})
    # memory recall from the streamed policy
    qa = st["query_answer"][0]; qt = np.nonzero(qa >= 0)[0]
    ok = []
    for t in qt:
        dirs = BINS - st["agent"][0, t]; dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        ok.append(int(np.argmax(dirs @ pols[t, :2]) == qa[t]))
    lv = [sum(1 for (l, _) in eng.events if l == k) for k in range(len(eng.levels))]
    res = {
        "name": name, "frames": T, "hours": T / 36000,
        "flops_total": {k: float(v) for k, v in eng.flops.items()},
        "gflops_per_frame_mean": fl.mean() / 1e9,
        "gflops_per_frame_idle": fl[~busy].mean() / 1e9, "gflops_per_frame_active": fl[busy].mean() / 1e9,
        "gflops_per_frame_last1000": fl[-1000:].mean() / 1e9,
        "latency_ms": {"mean": lat.mean() * 1e3, "p50": np.percentile(lat, 50) * 1e3, "p99": np.percentile(lat, 99) * 1e3,
                       "max": lat.max() * 1e3},
        "events_per_level": lv, "events_per_hour": [x / (T / 36000) for x in lv],
        "state": growth[-1], "state_growth": growth,
        "memory_acc": float(np.mean(ok)) if ok else None, "n_queries": len(ok),
        "act_mse_moving": float(((pols[moving, :2] - st["action"][0, moving, :2]) ** 2).mean()),
    }
    json.dump(res, open(f"runs/{name}/stream_{T}.json", "w"), default=float, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "state_growth"}, default=float))
    del model, eng; torch.cuda.empty_cache()
