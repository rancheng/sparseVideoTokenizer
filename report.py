"""Collect eval JSONs into markdown tables + figures (runs/report/)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vt.env import ACTION_EVENTS, EV_LIGHT

os.makedirs("runs/report", exist_ok=True)
RUNS = ["iso_1x", "fixed_16x", "heur_16x", "hnet_16x", "hnet_4x", "hnet_64x", "hnet_16x_videoonly", "hnet_16x_long", "iso_1x_long",
        "hnet_cap_r02", "hnet_cap_r10", "hnet_16x_mem", "iso_1x_mem"]


def get(name, suite):
    p = f"runs/{name}/eval_{suite}.json"
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def fmt(x, f="{:.2f}"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


lines = []
for suite in ["hour", "hour_longgap"]:
    lines += [f"\n### suite: {suite}\n",
              "| run | frames/token (top) | top tokens/hour | idle tok/min | active tok/min | PSNR dyn (copy) | act MSE moving | grip acc | mem <2.5min | mem 2.5-10min | mem 10-30min | mem >30min | action-event recall (top) | light recall (top) | state MB/hour |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in RUNS:
        s = get(n, suite)
        if s is None:
            continue
        top = s["levels"][-1]
        m = s["memory"]
        mem = [fmt(m[k]["acc"], "{:.0%}") + f" (n={m[k]['n']})" if m[k]["acc"] is not None else "—" for k in ["0-1500", "1500-6000", "6000-18000", "18000-1000000000"]]
        lines.append(f"| {n} | {top['frames_per_token']:.1f} | {top['tokens_per_hour']:.0f} | {top['tok_per_min_idle']:.1f} | {top['tok_per_min_active']:.1f} | "
                     f"{fmt(s.get('psnr_dyn'))} ({fmt(s.get('psnr_dyn_copy'))}) | {s['act_mse_moving']:.4f} | {s['grip_acc']:.3f} | "
                     + " | ".join(mem) + f" | {top['recall_action_events']:.2f} | {top['recall_light_events']:.2f} | {s['stream_state_mb']:.1f} |")
    lines += ["", "| run | level | frames/token | tokens/hour | idle tok/min | active tok/min | action-event recall | light recall | boundary precision |", "|---|---|---|---|---|---|---|---|---|"]
    for n in RUNS:
        s = get(n, suite)
        if s is None:
            continue
        for l, L in enumerate(s["levels"]):
            lines.append(f"| {n} | {l+1} | {L['frames_per_token']:.1f} | {L['tokens_per_hour']:.0f} | {L['tok_per_min_idle']:.1f} | {L['tok_per_min_active']:.1f} | "
                         f"{L['recall_action_events']:.2f} | {L['recall_light_events']:.2f} | {L['precision_events']:.2f} |")
# ---- streaming engine benchmark (1 hour, frame by frame) ----
lines += ["\n### streaming engine, 1 hour frame-by-frame\n",
          "| run | MFLOP/frame (mean) | idle | active | main-net share | L1 events/h | top events/h | main KV after 1 h (MB) | latency p50 / p99 (ms) | mem acc (streamed) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for n in RUNS:
    p = f"runs/{n}/stream_36000.json"
    if not os.path.exists(p):
        continue
    b = json.load(open(p)); ft = b["flops_total"]
    lines.append(f"| {n} | {b['gflops_per_frame_mean']*1e3:.1f} | {b['gflops_per_frame_idle']*1e3:.1f} | {b['gflops_per_frame_active']*1e3:.1f} | "
                 f"{ft['main']/sum(ft.values()):.1%} | {b['events_per_hour'][0]:.0f} | {b['events_per_hour'][-1]:.0f} | {b['state']['main_kv']/1e6:.1f} | "
                 f"{b['latency_ms']['p50']:.1f} / {b['latency_ms']['p99']:.1f} | {fmt(b['memory_acc'], '{:.0%}')} (n={b['n_queries']}) |")
if os.path.exists("runs/bench_scale.json"):
    bs = json.load(open("runs/bench_scale.json"))
    lines += ["\n### real-scale main network (680M, d=1536, 24 layers) one step vs KV length\n", "| KV tokens | KV GB | ms / step |", "|---|---|---|"]
    for pt in bs["points"]:
        ms = "OOM" if pt.get("oom") else f"{pt['ms_per_step']:.2f}"
        lines.append(f"| {pt['kv_tokens']} | {pt['kv_gb']:.1f} | {ms} |")
open("runs/report/tables.md", "w").write("\n".join(lines))
print("\n".join(lines))

# ---- boundary raster on a 2-minute window ----
names = [n for n in ["fixed_16x", "heur_16x", "hnet_16x", "hnet_16x_videoonly"] if os.path.exists(f"runs/{n}/hour_env0.npz")]
if names:
    d0 = np.load(f"runs/{names[0]}/hour_env0.npz")
    ev, act = d0["events"], d0["action"]
    moving = np.abs(act[:, :2]).sum(1) > 0.2
    # pick a window that contains a cue or query
    t0 = max(0, int(np.nonzero(ev & (1 << 4))[0][0]) - 600) if (ev & (1 << 4)).any() else 3000
    sl = slice(t0, t0 + 1200)
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(14, 1.3 * (len(names) + 1)), sharex=True)
    t = np.arange(sl.start, sl.stop) / 10
    ax = axes[0]
    ax.fill_between(t, 0, moving[sl], color="#bbb", step="mid", label="robot moving")
    for bit, c, lab in [(ACTION_EVENTS, "#1f6feb", "action events"), (EV_LIGHT, "#d4a017", "light toggle (exogenous)")]:
        idx = np.nonzero(ev[sl] & bit)[0]
        ax.vlines(t[idx], 0, 1, color=c, lw=1.2, label=lab)
    for bit, lab in [(1 << 3, "cue"), (1 << 4, "query")]:
        for i in np.nonzero(ev[sl] & bit)[0]:
            ax.text(t[i], 1.05, lab, fontsize=8)
    ax.set_yticks([]); ax.legend(loc="upper right", fontsize=7, ncol=3); ax.set_title("ground truth")
    for ax, n in zip(axes[1:], names):
        m = np.load(f"runs/{n}/hour_env0.npz")["masks"]
        for l, (mk, c) in enumerate(zip(m, ["#8aa", "#c33"])):
            idx = np.nonzero(mk[sl])[0]
            ax.vlines(t[idx], l, l + 0.9, color=c, lw=0.8)
        ax.set_yticks([0.45, 1.45]); ax.set_yticklabels(["L1", "L2"], fontsize=8); ax.set_title(n, fontsize=9, loc="left")
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig("runs/report/boundaries.png", dpi=130)

# ---- compression/quality frontier ----
pts = [(n, get(n, "hour")) for n in ["iso_1x", "hnet_4x", "hnet_16x", "hnet_64x", "fixed_16x", "heur_16x"]]
pts = [(n, s) for n, s in pts if s]
if pts:
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for n, s in pts:
        x = s["levels"][-1]["tokens_per_hour"]
        mk = "o" if n.startswith(("hnet", "iso")) else "s"
        axs[0].scatter(x, s["psnr_dyn"], marker=mk); axs[0].annotate(n, (x, s["psnr_dyn"]), fontsize=7)
        axs[1].scatter(x, s["act_mse_moving"], marker=mk); axs[1].annotate(n, (x, s["act_mse_moving"]), fontsize=7)
        acc = s["memory"]["0-1500"]["acc"]
        axs[2].scatter(x, acc, marker=mk); axs[2].annotate(n, (x, acc), fontsize=7)
    for a, yl in zip(axs, ["next-frame PSNR, dynamic frames (dB)", "action MSE (moving)", "memory recall acc (<2.5 min)"]):
        a.set_xscale("log"); a.set_xlabel("top-level tokens / hour"); a.set_ylabel(yl); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("runs/report/frontier.png", dpi=130)
