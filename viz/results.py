"""Results figures for the README / docs  ->  docs/figures/*.png

Reads only the JSON artefacts written by eval.py, bench_stream.py, bench_scale.py and train.py. Every figure skips
runs whose files do not exist yet, so this can be rerun as more runs land:

    .venv/bin/python viz/results.py
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "docs", "figures")

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "text.color": INK, "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})

# One colour per model family, fixed across every figure. Variants of a model share its hue and differ by
# marker / line style (long = hollow / dashed, mem = diamond / dotted). The dense baseline is the neutral gray.
# Validated with the dataviz palette checker: the 5 hues of the main streaming set clear the normal-vision floor
# all-pairs; CVD separation is in the 6-8 band, so markers + direct labels are always used as secondary encoding.
C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300", "red": "#e34948",
     "orange": "#eb6834", "magenta": "#e87ba4", "violet": "#6250d6", "gray": "#52514e"}
MODEL = {  # name: (label, colour, marker, filled, linestyle)
    "iso_1x":             ("dense (iso_1x)", C["gray"], "s", True, "-"),
    "fixed_16x":          ("fixed_16x", C["yellow"], "^", True, "-"),
    "heur_16x":           ("heur_16x (pixel-change)", C["red"], "v", True, "-"),
    "hnet_4x":            ("hnet_4x", C["orange"], "o", True, "-"),
    "hnet_16x":           ("hnet_16x", C["blue"], "o", True, "-"),
    "hnet_64x":           ("hnet_64x", C["green"], "o", True, "-"),
    "hnet_cap_r02":       ("hnet_cap_r02", C["aqua"], "P", True, "-"),
    "hnet_cap_r10":       ("hnet_cap_r10", C["violet"], "X", True, "-"),
    "hnet_16x_videoonly": ("hnet_16x_videoonly", C["magenta"], "o", True, "-"),
    "hnet_16x_long":      ("hnet_16x_long", C["blue"], "o", False, "--"),
    "iso_1x_long":        ("iso_1x_long", C["gray"], "s", False, "--"),
    "hnet_16x_mem":       ("hnet_16x_mem", C["blue"], "D", True, ":"),
    "iso_1x_mem":         ("iso_1x_mem", C["gray"], "D", True, ":"),
}
HOUR_RUNS = list(MODEL)
STREAM_RUNS = ["iso_1x", "fixed_16x", "heur_16x", "hnet_16x", "hnet_64x", "hnet_cap_r02"]
BOUNDARY_RUNS = ["fixed_16x", "heur_16x", "hnet_16x", "hnet_16x_videoonly", "hnet_cap_r02"]
TRAIN_RUNS = ["iso_1x", "fixed_16x", "heur_16x", "hnet_4x", "hnet_16x", "hnet_64x", "hnet_16x_videoonly",
              "hnet_cap_r02", "hnet_16x_mem", "iso_1x_mem"]
FPS, FRAMES_PER_HOUR = 10, 36000


def lab(n): return MODEL[n][0]
def col(n): return MODEL[n][1]


def mstyle(n, size=7):
    _, c, m, filled, _ = MODEL[n]
    return dict(marker=m, markersize=size, markerfacecolor=c if filled else "white", markeredgecolor=c,
                markeredgewidth=1.6, color=c)


def load(*parts):
    p = os.path.join(RUNS, *parts)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def header(fig, title, sub, y=1.0):
    fig.text(0.01, y, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.01, y - 0.012, sub, fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.4)


def fmt_bytes_gb(gb):
    if gb >= 1000: return f"{gb / 1000:.1f} TB"
    if gb >= 1: return f"{gb:.1f} GB"
    return f"{gb * 1000:.0f} MB"


def fmt_s(s):
    if s >= 3600: return f"{s / 3600:.1f} h"
    if s >= 60: return f"{s / 60:.1f} min"
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"


# ---------------------------------------------------------------- 1. stream compute
def fig_stream_compute():
    runs = [(n, load(n, "stream_36000.json")) for n in STREAM_RUNS]
    runs = [(n, d) for n, d in runs if d]
    if not runs:
        return print("skip stream_compute (no stream_36000.json)")
    comps = [("frame_enc+heads", "frame encoder + heads", "#52514e"), ("level1", "level-1 chunk stack", "#898781"),
             ("level2", "level-2 chunk stack", "#c3c2b7"), ("main", "main network", C["orange"])]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ys = np.arange(len(runs))[::-1]
    for y, (n, d) in zip(ys, runs):
        left = 0.0
        tot = sum(d["flops_total"].values()) / d["frames"] / 1e6
        for key, _, c in comps:
            v = d["flops_total"].get(key, 0.0) / d["frames"] / 1e6
            if v > 0:
                ax.barh(y, v, left=left, height=0.62, color=c, edgecolor="white", linewidth=1.5)
            left += v
        share = d["flops_total"].get("main", 0.0) / sum(d["flops_total"].values())
        ax.text(tot + 6, y, f"{tot:.0f} MFLOP/frame  ·  main {share:.1%}", va="center", fontsize=9.5, color=INK,
                fontweight="bold" if n == "iso_1x" else "normal")
    fe = runs[0][1]["flops_total"]["frame_enc+heads"] / runs[0][1]["frames"] / 1e6
    ax.axvline(fe, color=INK, lw=1, ls=(0, (3, 3)), zorder=3)
    ax.text(fe + 3, ys[0] + 0.55, f"frame-encoder floor ≈ {fe:.0f} MFLOP", fontsize=9, color=INK, va="bottom")
    ax.set_yticks(ys, [lab(n) for n in dict(runs)])
    for t, (n, _) in zip(ax.get_yticklabels(), runs):
        t.set_color(INK)
    ax.set_xlabel("measured compute per frame while streaming 1 hour frame-by-frame (MFLOP, mean over 36,000 frames)")
    ax.set_xlim(0, max(sum(d["flops_total"].values()) / d["frames"] / 1e6 for _, d in runs) * 1.42)
    ax.set_ylim(-0.6, len(runs) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in comps], loc="lower right", ncol=1,
              bbox_to_anchor=(1.0, 0.02), handlelength=1.2)
    header(fig, "Event-driven streaming removes the main-network cost; the frame encoder becomes the floor",
           "Per-frame compute by component, event-driven engine (bench_stream.py), toy-scale models. "
           "The main network only runs when a top-level token is emitted.", y=0.985)
    fig.subplots_adjust(top=0.86)
    save(fig, "stream_compute.png")


# ---------------------------------------------------------------- 2. KV growth
def fig_kv_growth():
    runs = [(n, load(n, "stream_36000.json")) for n in STREAM_RUNS]
    runs = [(n, d) for n, d in runs if d]
    if not runs:
        return print("skip kv_growth")
    fig, ax = plt.subplots(figsize=(9, 5))
    ends = []
    for n, d in runs:
        g = d["state_growth"]
        t = np.array([x["t"] for x in g]) / FPS / 60
        kv = np.array([x["main_kv"] for x in g]) / 1e6
        ax.plot(t, kv, ls=MODEL[n][4], **mstyle(n, 5), zorder=3)
        ends.append((kv[-1], n, d["events_per_hour"][-1]))
    # direct end labels, nudged apart in log space
    ends.sort()
    ypos, last = [], -1e9
    for v, n, e in ends:
        ly = max(math.log10(v), last + 0.13); ypos.append(ly); last = ly
    for (v, n, e), ly in zip(ends, ypos):
        ax.annotate(f"{lab(n)}  {v:.1f} MB  ({e:,.0f} tok/h)", xy=(60, v), xytext=(63.5, 10 ** ly), fontsize=9,
                    color=INK, va="center", arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=0, shrinkB=3))
    w = [x["windows"] for _, d in runs for x in d["state_growth"]]
    wmax = max(w) / 1e6
    ax.axhline(wmax, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax.text(60, wmax * 0.86, f"tokenizer sliding-window caches: constant {wmax:.2f} MB", fontsize=8.5, color=INK2,
            ha="right", va="top")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} MB"))
    ax.set_xlim(0, 61)
    ax.set_xticks(range(0, 61, 10))
    ax.set_xlabel("stream time (minutes at 10 Hz)")
    ax.set_ylabel("main-network KV cache (MB, log scale)")
    header(fig, "Main-network KV cache grows with events, not frames",
           "Measured state size every 2,048 frames while streaming one held-out hour (bench_stream.py). "
           "Linear growth; slope = top-level token rate.", y=0.985)
    fig.subplots_adjust(top=0.88)
    save(fig, "kv_growth.png")


# ---------------------------------------------------------------- 3. real-scale projection
def fig_scale_projection():
    b = load("bench_scale.json")
    if not b:
        return print("skip scale_projection (no bench_scale.json)")
    pts = [p for p in b["points"] if not p.get("oom")]
    n = np.array([p["kv_tokens"] for p in pts], float)
    ms = np.array([p["ms_per_step"] for p in pts], float)
    nmax, slope = n[-1], (ms[-1] - ms[-2]) / (n[-1] - n[-2])
    kvb = b["kv_bytes_per_token"]

    def ms_at(k):  # piecewise-linear interpolation, flat below the first point, linear extrapolation above the last
        k = np.asarray(k, float)
        return np.where(k <= nmax, np.interp(k, n, ms), ms[-1] + slope * (k - nmax))

    def gpu_s(n_steps, tok_per_step):  # sum over steps of ms(KV length before that step)
        k = np.linspace(0, n_steps, 4001)
        return np.trapezoid(ms_at(k * tok_per_step), k) / 1e3

    cfgs = [("dense, 1 token/frame", 1.0 * FRAMES_PER_HOUR, 1.0, "#52514e", None),
            ("dense, 16 tokens/frame", 1.0 * FRAMES_PER_HOUR, 16.0, "#898781", None)]
    for name in ["hnet_16x", "hnet_64x"]:
        s = load(name, "stream_36000.json")
        if s:
            cfgs.append((f"event-driven {name}\n({s['events_per_hour'][-1]:,.0f} events/h measured)",
                         s["events_per_hour"][-1], 1.0, col(name), name))
    horizons = [1, 24]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    W = 0.8 / len(cfgs)
    for pi, ax in enumerate(axes):
        for ci, (lbl, steps_h, tps, c, _) in enumerate(cfgs):
            for hi, H in enumerate(horizons):
                tokens = steps_h * H * tps
                extrap = tokens > nmax
                if pi == 0:
                    val = tokens * kvb / 1e9; txt = fmt_bytes_gb(val)
                else:
                    val = gpu_s(steps_h * H, tps) / H; txt = fmt_s(val)
                x = hi + (ci - (len(cfgs) - 1) / 2) * W
                ax.bar(x, val, width=W * 0.92, color=c if not extrap else "white", edgecolor=c,
                       hatch="////" if extrap else None, linewidth=1.4)
                ax.text(x, val * 1.15, txt, ha="center", va="bottom", fontsize=8.5, color=INK, zorder=5,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
        ax.set_yscale("log")
        ax.set_xticks(range(len(horizons)), [f"{H} hour" + ("s" if H > 1 else "") + " of 10 Hz video" for H in horizons])
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0, labelsize=10, labelcolor=INK)
    a0, a1 = axes
    a0.axhline(24, color=C["red"], lw=1.3, ls=(0, (4, 3)))
    a0.text(1.49, 24 * 1.25, "RTX 4090: 24 GB", color=INK, fontsize=9, ha="right", va="bottom", fontweight="bold")
    a0.set_ylabel("main-network KV cache at end of horizon (log)")
    a0.set_title("(a) memory", loc="left")
    a0.set_ylim(0.01, 2e4)
    a1.axhline(3600, color=C["red"], lw=1.3, ls=(0, (4, 3)))
    a1.text(-0.47, 3600 * 1.2, "real-time limit", color=INK, fontsize=9, ha="left", va="bottom", fontweight="bold")
    a1.set_ylabel("main-network GPU time per hour of video (log)")
    a1.set_title("(b) compute", loc="left")
    a1.set_ylim(0.3, 3e5)
    tk = [1, 10, 60, 600, 3600, 36000, 360000]
    a1.set_yticks(tk, ["1 s", "10 s", "1 min", "10 min", "1 h", "10 h", "100 h"])
    a1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_bytes_gb(v)))
    handles = [Patch(facecolor=c, edgecolor=c, label=l) for l, _, _, c, _ in cfgs]
    handles.append(Patch(facecolor="white", edgecolor=INK2, hatch="////",
                     label=f"needs KV > {nmax / 1e3:.0f}K tokens:\nms/step extrapolated"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04), fontsize=9,
               handlelength=1.6, columnspacing=1.6)
    header(fig, f"Real-scale projection: {b['params'] / 1e6:.0f}M-parameter main network on one RTX 4090",
           f"MEASURED: one main-network step (bf16, d=1536, 24 layers, KV {kvb / 1e3:.0f} KB/token) at KV length "
           f"{int(n[0]):,}–{int(nmax):,} tokens (bench_scale.py);\nand the top-level event rates of the trained toy models "
           f"over a 1-hour stream (bench_stream.py).\nPROJECTED: every bar — ms/step interpolated piecewise-linearly in "
           f"KV length, extrapolated linearly beyond {int(nmax / 1e3)}K tokens (hatched). Dense 16 tokens/frame assumes "
           f"one batched step per frame\n(optimistic for dense). Main network only; tokenizer cost and weights excluded.", y=1.02)
    fig.subplots_adjust(top=0.8, wspace=0.28)
    save(fig, "scale_projection.png")
    return cfgs, ms_at, gpu_s


# ---------------------------------------------------------------- 4. compression vs quality
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def fig_compression_quality():
    rows = []
    for n in HOUR_RUNS:
        e = load(n, "eval_hour.json")
        if not e:
            continue
        s = e["summary"]
        q = [x for r in e["per_stream"] for x in r.get("queries", [])]
        k = sum(x["ok"] for x in q)
        rows.append(dict(name=n, tph=s["levels"][-1]["tokens_per_hour"], psnr=s.get("psnr_dyn"),
                         copy=s.get("psnr_dyn_copy"), amse=s["act_mse_moving"], mem=wilson(k, len(q)), nq=len(q),
                         expert=s["memory"].get("expert_acc")))
    if not rows:
        return print("skip compression_quality")
    # offset exactly-coincident x values slightly so every marker stays visible
    seen = {}
    for r in rows:
        key = round(math.log10(r["tph"]), 2)
        r["x"] = r["tph"] * (1.09 ** seen.get(key, 0)); seen[key] = seen.get(key, 0) + 1
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    panels = [("psnr", "next-frame PSNR on dynamic frames (dB)", "(a) world-model prediction"),
              ("amse", "action MSE on moving frames (lower is better)", "(b) policy"),
              ("mem", "memory-recall accuracy (95% CI)", "(c) memory recall")]
    for ax, (key, yl, title) in zip(axes, panels):
        for r in rows:
            if key == "mem":
                p, lo, hi = r["mem"]
                ax.errorbar(r["x"], p, yerr=[[p - lo], [hi - p]], ls="none", elinewidth=1.2, capsize=0,
                            ecolor=col(r["name"]), alpha=0.6, zorder=2)
                ax.plot(r["x"], p, ls="none", **mstyle(r["name"]), zorder=3)
            elif r[key] is not None:
                ax.plot(r["x"], r[key], ls="none", **mstyle(r["name"]), zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(70, 1.2e5)
        ax.set_xlabel("top-level tokens per hour (log)")
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left")
        sec = ax.secondary_xaxis("top", functions=(lambda t: FRAMES_PER_HOUR / np.maximum(t, 1e-9),
                                                   lambda c: FRAMES_PER_HOUR / np.maximum(c, 1e-9)))
        sec.set_xticks([1, 4, 16, 64, 256], ["1×", "4×", "16×", "64×", "256×"])
        sec.tick_params(labelsize=8.5, colors=MUTED)
        sec.spines["top"].set_color(AXIS)
    a0, a1, a2 = axes
    copy = np.nanmean([r["copy"] for r in rows if r["copy"] is not None])
    a0.axhline(copy, color=INK2, lw=1, ls=(0, (3, 3)))
    a0.text(85, copy + 0.15, f"copy-last-frame baseline {copy:.2f} dB", fontsize=8.5, color=INK2, va="bottom")
    ps = [r["psnr"] for r in rows if r["psnr"] is not None]
    a0.text(85, max(ps) + 0.35, f"all runs: {min(ps):.2f}–{max(ps):.2f} dB", fontsize=9, color=INK, va="bottom",
            fontweight="bold")
    a0.set_ylim(copy - 1, max(ps) + 1.4)
    a1.set_ylim(0, max(r["amse"] for r in rows) * 1.25)
    for r in rows:
        if r["name"] == "hnet_16x_videoonly":
            a1.annotate("no action input\n(videoonly)", xy=(r["x"], r["amse"]), xytext=(r["x"] * 2.3, r["amse"] * 0.92),
                        fontsize=8.5, color=INK, va="top", arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    others = [r["amse"] for r in rows if r["name"] != "hnet_16x_videoonly"]
    a1.text(85, max(others) * 1.6, f"all others: {min(others):.4f}–{max(others):.4f}", fontsize=9, color=INK,
            fontweight="bold")
    a2.axhline(0.25, color=C["red"], lw=1.3, ls=(0, (4, 3)))
    a2.text(1.01, 0.25, "chance\n25%", color=INK, fontsize=9, fontweight="bold", va="center", ha="left",
            transform=matplotlib.transforms.blended_transform_factory(a2.transAxes, a2.transData))
    ex = [r["expert"] for r in rows if r["expert"] is not None]
    a2.set_ylim(0, 0.62)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a2.text(85, 0.58, f"n = {rows[0]['nq']} queries per run; expert policy {np.mean(ex):.0%}", fontsize=8.5,
            color=INK2, va="top")
    handles = [Line2D([], [], ls="none", label=lab(r["name"]), **mstyle(r["name"])) for r in rows]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), bbox_to_anchor=(0.5, -0.13),
               handletextpad=0.3, columnspacing=1.3)
    header(fig, "Prediction quality is flat across 1×–255× compression; memory recall is at chance for every run",
           "Held-out hour suite (6 streams × 36,000 frames, eval.py). Top axis: frames per top-level token. "
           "Coincident points offset slightly in x.", y=1.04)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "compression_quality.png")
    return rows


# ---------------------------------------------------------------- 5. boundary quality
def tint(hexc, a):
    c = np.array(matplotlib.colors.to_rgb(hexc)); return tuple(1 - a * (1 - c))


def fig_boundary_quality():
    runs = [(n, load(n, "eval_hour.json")) for n in BOUNDARY_RUNS]
    runs = [(n, e["summary"]) for n, e in runs if e]
    if not runs:
        return print("skip boundary_quality")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharey="row")
    x = np.arange(len(runs)); W = 0.36
    for li in range(2):
        ar, at = axes[0, li], axes[1, li]
        for i, (n, s) in enumerate(runs):
            if li >= len(s["levels"]):
                continue
            L = s["levels"][li]; c = col(n)
            for off, v, solid in [(-W / 2, L["recall_action_events"], True), (W / 2, L["recall_light_events"], False)]:
                ar.bar(i + off, v, W * 0.92, color=c if solid else tint(c, 0.28), edgecolor=c,
                       hatch=None if solid else "////", linewidth=1.2)
                ar.text(i + off, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK, zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85))
            # recall a random boundary placement at the same rate would get (+-3-frame tolerance -> 7-frame window)
            p = 1 / L["frames_per_token"]; rnd = 1 - (1 - p) ** 7
            ar.plot([i - W - 0.04, i + W + 0.04], [rnd, rnd], color=INK, lw=1.6, solid_capstyle="butt", zorder=4)
            for off, v, solid in [(-W / 2, L["tok_per_min_idle"], False), (W / 2, L["tok_per_min_active"], True)]:
                at.bar(i + off, v, W * 0.92, color=c if solid else tint(c, 0.28), edgecolor=c,
                       hatch=None if solid else "////", linewidth=1.2)
                at.text(i + off, v * 1.12, f"{v:.0f}" if v >= 100 else f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
        for ax in (ar, at):
            ax.set_xticks(x, [lab(n).replace(" (pixel-change)", "\n(pixel-change)").replace("_videoonly", "\n_videoonly")
                              for n, _ in runs], fontsize=9, color=INK)
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="x", length=0)
        ar.set_title(f"level {li + 1}" + (" (top level → main network)" if li == 1 else ""), loc="left")
        ar.set_ylim(0, 1.15)
        at.set_yscale("log"); at.set_ylim(1, 1200)
    axes[0, 0].set_ylabel("event recall (boundary within ±3 frames)")
    axes[1, 0].set_ylabel("boundaries per minute (log)")
    axes[1, 0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    g = "#898781"
    h1 = [Patch(facecolor=g, edgecolor=g, label="action events (pick / place / start-stop)"),
          Patch(facecolor=tint(g, 0.28), edgecolor=g, hatch="////", label="exogenous light flicker (distractor)"),
          Line2D([], [], color=INK, lw=1.6, label="random boundaries at the same rate")]
    h2 = [Patch(facecolor=tint(g, 0.28), edgecolor=g, hatch="////", label="idle segments"),
          Patch(facecolor=g, edgecolor=g, label="active segments (robot moving or event nearby)")]
    axes[0, 1].legend(handles=h1, loc="upper right", fontsize=8.5)
    axes[1, 1].legend(handles=h2, loc="upper right", fontsize=8.5)
    header(fig, "Where do boundaries land?  The heuristic chases lighting; the learned chunker is only weakly event-aligned",
           "Held-out hour suite (eval.py). Top: recall of action vs exogenous events; black tick = recall of random "
           "boundaries at the same rate.\nBottom: boundary rate in idle vs active segments — an adaptive tokenizer "
           "should spend tokens where things happen.", y=1.0)
    fig.subplots_adjust(top=0.9, hspace=0.32, wspace=0.06)
    save(fig, "boundary_quality.png")


# ---------------------------------------------------------------- 6. training curves
def smooth(y, k=7):
    y = np.asarray(y, float)
    if len(y) < k: return y
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def fig_training_curves():
    runs = [(n, load(n, "train_log.json")) for n in TRAIN_RUNS]
    runs = [(n, l) for n, l in runs if l]
    if not runs:
        return print("skip training_curves")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for n, l in runs:
        st = np.array([x["step"] for x in l])
        kw = dict(color=col(n), ls=MODEL[n][4], lw=1.8)
        axes[0].plot(st, smooth([x["mse"] for x in l]), **kw)
        axes[1].plot(st, smooth([x["a_mse"] for x in l]), **kw)
        comp = np.array([x["comp"] for x in l])
        axes[2].plot(st, smooth(comp[:, -1], 9), **kw)
    t = ["(a) next-frame pixel MSE", "(b) action loss (query frames ×10)", "(c) top-level compression (frames/token)"]
    for ax, tt in zip(axes, t):
        ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_title(tt, loc="left")
    axes[0].set_ylim(top=0.01)
    axes[1].set_ylim(top=0.5)
    for ax in axes:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].set_ylim(0.8, 700)
    handles = [Line2D([], [], color=col(n), ls=MODEL[n][4], lw=2, label=lab(n)) for n, _ in runs]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), bbox_to_anchor=(0.5, -0.16))
    header(fig, "Training curves",
           "Logged every 50 steps (train.py), 7-point (compression: 9-point) moving average. Components shown instead of the total loss, "
           "which also includes a run-specific compression-ratio penalty.", y=1.04)
    fig.subplots_adjust(wspace=0.25)
    save(fig, "training_curves.png")


if __name__ == "__main__":
    fig_stream_compute()
    fig_kv_growth()
    fig_scale_projection()
    fig_compression_quality()
    fig_boundary_quality()
    fig_training_curves()
