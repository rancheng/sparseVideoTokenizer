"""Export compact JSON of all measured results for the blog's interactive charts."""
import json, os, sys
OUT = sys.argv[1]
RUNS = {"iso_1x": "dense (no compression)", "fixed_16x": "fixed stride 16x", "heur_16x": "pixel-change heuristic 16x",
        "hnet_4x": "learned 4x (1 stage)", "hnet_16x": "learned 16x", "hnet_64x": "learned 64x",
        "hnet_16x_videoonly": "learned 16x, video-only", "hnet_cap_r02": "learned + budget cap (0.02)",
        "hnet_cap_r10": "learned + budget cap (0.1)", "hnet_16x_long": "learned 16x, long finetune",
        "iso_1x_long": "dense, long finetune", "hnet_16x_mem": "learned 16x, anti-copycat", "iso_1x_mem": "dense, anti-copycat"}

def rnd_recall(tok_per_hour, tol=3):  # P(>=1 random boundary within +-tol frames) at this rate
    r = tok_per_hour / 36000.0
    return 1 - (1 - r) ** (2 * tol + 1)

res = {}
for n, label in RUNS.items():
    p = f"runs/{n}/eval_hour.json"
    if not os.path.exists(p):
        continue
    s = json.load(open(p))["summary"]
    lv = [{k: round(v, 4) for k, v in L.items()} | {"random_recall": round(rnd_recall(L["tokens_per_hour"]), 4)} for L in s["levels"]]
    res[n] = {"label": label, "levels": lv, "psnr_dyn": round(s["psnr_dyn"], 3), "psnr_copy": round(s["psnr_dyn_copy"], 3),
              "act_mse_moving": round(s["act_mse_moving"], 5), "grip_acc": round(s["grip_acc"], 4),
              "memory": s["memory"], "state_mb_per_hour": round(s["stream_state_mb"], 2)}
    for suite in ("hour_longgap", "day"):
        q = f"runs/{n}/eval_{suite}.json"
        if os.path.exists(q):
            ss = json.load(open(q))["summary"]
            res[n][suite] = {"memory": ss["memory"], "top_tokens_per_hour": round(ss["levels"][-1]["tokens_per_hour"], 1),
                             "t_body_s": round(ss.get("t_body_s", 0), 3), "t_encode_s": round(ss.get("t_encode_s", 0), 3),
                             "peak_gb": round(ss.get("peak_gb_body", 0), 2), "state_mb": round(ss["stream_state_mb"], 1),
                             "inner_tokens": ss.get("inner_tokens")}
stream = {}
for n in RUNS:
    p = f"runs/{n}/stream_36000.json"
    if os.path.exists(p):
        b = json.load(open(p))
        stream[n] = {"label": RUNS[n], "mflop_total": {k: round(v / 36000 / 1e6, 3) for k, v in b["flops_total"].items()},
                     "mflop_idle": round(b["gflops_per_frame_idle"] * 1e3, 2), "mflop_active": round(b["gflops_per_frame_active"] * 1e3, 2),
                     "latency_ms": {k: round(v, 3) for k, v in b["latency_ms"].items()}, "events_per_hour": b["events_per_hour"],
                     "state_growth": [{"t": g["t"], "main_kv_mb": round(g["main_kv"] / 1e6, 3), "windows_mb": round(g["windows"] / 1e6, 3),
                                       "main_tokens": g["main_tokens"]} for g in b["state_growth"]],
                     "memory_acc_streamed": b["memory_acc"], "n_queries": b["n_queries"]}
scale = json.load(open("runs/bench_scale.json"))
memory = {
    "query_examples": [  # iso_1x at query onset frame t vs t+1 (memdiag2): expert vs predicted (vx, vy)
        {"answer": "red", "expert_t": [-0.97, -1.0], "pred_t": [-0.06, 0.14], "expert_t1": [-1.0, -1.0], "pred_t1": [-0.97, -0.98]},
        {"answer": "yellow", "expert_t": [0.98, 1.0], "pred_t": [-0.11, 0.33], "expert_t1": [1.0, 0.9], "pred_t1": [0.95, 0.98]},
        {"answer": "blue", "expert_t": [-1.0, 0.94], "pred_t": [-0.11, 0.22], "expert_t1": [-1.0, 0.97], "pred_t1": [-1.0, 1.03]},
        {"answer": "yellow", "expert_t": [1.0, 0.96], "pred_t": [-0.11, -0.01], "expert_t1": [1.0, 1.0], "pred_t1": [0.96, 0.96]},
        {"answer": "blue", "expert_t": [-1.0, 1.0], "pred_t": [-0.58, 0.85], "expert_t1": [-0.93, 1.0], "pred_t1": [-0.97, 0.99]},
        {"answer": "red", "expert_t": [-0.98, -1.0], "pred_t": [-0.04, 0.05], "expert_t1": [-1.0, -1.0], "pred_t1": [-0.97, -1.01]}],
    "probe_leaky": {"hnet_16x": {"0": 1.0, "50": 0.727, "300": 0.713}, "iso_1x": {"0": 1.0, "50": 0.70, "300": 0.602}},
    "probe_grouped": {"hnet_16x": {"0": [0.96, 0.02], "50": [0.24, 0.08], "300": [0.25, 0.04], "1000": [0.30, 0.08]},
                      "iso_1x": {"0": [0.96, 0.02], "50": [0.20, 0.05], "300": [0.24, 0.01], "1000": [0.26, 0.04]}},
    "probe_notes": "190 cue episodes; grouped = one sample per cue episode, split by episode, 5 seeds (mean, std). Leaky = 4 sibling frames per cue split randomly.",
}
json.dump({"results": res, "stream": stream, "scale": scale, "memory": memory}, open(OUT, "w"), separators=(",", ":"), default=float)
print(len(res), "runs;", len(stream), "stream runs;", os.path.getsize(OUT), "bytes")
