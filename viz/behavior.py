"""README figures of learned tokenizer behaviour on the held-out hour stream (env 0).

  .venv/bin/python viz/behavior.py            # timeline.png, filmstrip.png, token_rate.png -> docs/figures/
  .venv/bin/python viz/stream_gif.py          # stream.gif

Inference results are cached in viz/.cache (delete it to recompute; ~3.5 GB GPU, B=1, a few seconds per run).
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from vt.env import (EV_GRASP, EV_RELEASE, EV_SPAWN, EV_CUE, EV_QUERY, EV_LIGHT, EV_MOVE_START, EV_MOVE_STOP,
                    ACTION_EVENTS)

plt = mpl_style()
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 2.5-minute window: cue at 31:06.5 answered by the query at 31:51.2, nine pick-and-places, four light
# toggles and a ~44 s idle stretch (chosen by scanning the hour for windows that contain all of these).
W0, W1 = 18400, 19900
MODELS = ["fixed_16x", "heur_16x", "hnet_16x", "hnet_16x_videoonly", "hnet_cap_r02"]
COLOR_NAMES = ["red", "green", "blue", "yellow"]
CHOSEN, HEART, STRIDE = BLUE, "#b9b7ae", INK2


def spans(mask):
    """[start, stop) runs of True."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def xt(t):
    return t / FPS  # seconds into the hour


def minute_axis(ax, t0, t1, step_s=15):
    ticks = np.arange(np.ceil(xt(t0) / step_s) * step_s, xt(t1) + 1e-6, step_s)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(s // 60)}:{int(s % 60):02d}" for s in ticks])


# ----------------------------------------------------------------------------------------------- timeline
def timeline(st, R):
    sl = slice(W0, W1)
    t = np.arange(W0, W1)
    ev = st["events"][ENV][sl]
    moving = moving_of(st)[sl]
    grip = st["action"][ENV, sl, 2] > 0.5
    light = st["light"][ENV][sl]
    mv_spans = [(W0 + a, W0 + b) for a, b in spans(moving)]

    heights = [2.3, 1.25] + [0.95] * len(MODELS)
    fig, axes = plt.subplots(len(heights), 1, figsize=(15, sum(heights) + 1.6), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.55})
    for ax in axes:
        for a, b in mv_spans:
            ax.axvspan(xt(a), xt(b), color="#f1f0ec", lw=0, zorder=0)
        ax.set_xlim(xt(W0), xt(W1))
        ax.tick_params(length=2)
    # ---- ground truth ----
    ax = axes[0]
    ax.set_ylim(0, 5.2); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    for a, b in mv_spans:
        ax.fill_between([xt(a), xt(b)], 3.0, 3.8, color="#d4d2ca", lw=0)
    for a, b in spans(grip):
        ax.fill_between([xt(W0 + a), xt(W0 + b)], 3.0, 3.8, color=INK2, lw=0)
    ax.text(xt(W0) - 1, 3.4, "robot", ha="right", va="center", fontsize=9, color=INK2)
    # light level as a step band
    dim = light < 0.9
    for a, b in spans(dim):
        ax.fill_between([xt(W0 + a), xt(W0 + b)], 1.9, 2.5, color="#ffd9a8", lw=0)
    ax.text(xt(W0) - 1, 2.2, "room light", ha="right", va="center", fontsize=9, color=INK2)
    for i in np.nonzero(ev & EV_LIGHT)[0]:
        ax.vlines(xt(W0 + i), 1.8, 2.6, color=ORANGE, lw=1.6)
    # action events
    for bit, y, mk in [(EV_GRASP, 0.95, "v"), (EV_RELEASE, 0.95, "^"), (EV_SPAWN, 0.95, "o")]:
        idx = np.nonzero(ev & bit)[0]
        ax.scatter(xt(W0 + idx), np.full(len(idx), y), marker=mk, s=26,
                   color=INK if bit != EV_SPAWN else "white", edgecolor=INK, lw=0.9, zorder=3)
    for bit in (EV_MOVE_START, EV_MOVE_STOP):
        idx = np.nonzero(ev & bit)[0]
        ax.vlines(xt(W0 + idx), 0.55, 1.35, color=MUTED, lw=0.8)
    ax.text(xt(W0) - 1, 0.95, "events", ha="right", va="center", fontsize=9, color=INK2)
    for bit, name in [(EV_CUE, "cue"), (EV_QUERY, "query")]:
        for i in np.nonzero(ev & bit)[0]:
            c = int(st["cue_col"][ENV, W0 + i])
            label = f"{name}: {COLOR_NAMES[c]}" if bit == EV_CUE else f"query (answer {COLOR_NAMES[st['query_answer'][ENV, W0 + i]]})"
            ax.vlines(xt(W0 + i), 0.4, 4.25, color=VIOLET, lw=1.6, zorder=2)
            ax.text(xt(W0 + i), 4.35, label, ha="center", va="bottom", fontsize=9.5, color=INK, weight="bold")
    # pair cue -> query with the gap
    for i in np.nonzero(ev & EV_QUERY)[0]:
        tq = W0 + i; tc = st["query_cue_t"][ENV, tq]
        if tc >= W0:
            ax.annotate("", xy=(xt(tq), 4.1), xytext=(xt(tc), 4.1),
                        arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.1))
            ax.text((xt(tq) + xt(tc)) / 2, 3.88, f"{(tq - tc) / FPS:.0f} s later", ha="center", va="top",
                    fontsize=8.5, color=VIOLET)
    ax.set_title("Ground truth", loc="left", fontsize=11, weight="bold", pad=4)
    leg = [Patch(color="#d4d2ca", label="robot moving"), Patch(color=INK2, label="holding object"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="object spawns"),
           Line2D([], [], marker="v", ls="", color=INK, label="grasp"),
           Line2D([], [], marker="^", ls="", color=INK, label="release"),
           Line2D([], [], color=MUTED, lw=0.8, label="motion start/stop"),
           Line2D([], [], color=ORANGE, lw=1.6, label="light toggle (exogenous)"),
           Patch(color="#ffd9a8", label="lights dimmed")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0, 1.13), ncol=8, fontsize=8.5,
              handlelength=1.4, columnspacing=1.2)

    # ---- hnet_16x L1 router probability ----
    ax = axes[1]
    p = R["hnet_16x"]["pr0"][sl]
    ax.plot(xt(t), p, color=BLUE, lw=0.7)
    ax.axhline(0.5, color=INK, lw=0.9, ls=(0, (4, 3)))
    lo, hi = np.nanmin(p), np.nanmax(p)
    pad = 0.1 * (hi - lo)
    idle_band = np.nanquantile(p[~busy_of(st)[sl]], [0.01, 0.99]) - 0.5
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(xt(W1) + 1, 0.5, "0.5 = boundary\nthreshold", va="center", fontsize=8.5, color=INK)
    ax.set_ylabel("p$_t$", rotation=0, ha="right", va="center", fontsize=10)
    ax.set_title("learned H-Net (16x) — level-1 router probability p$_t$ before heartbeat (y-axis zoomed to "
                 f"{lo:.2f}–{hi:.2f}; while idle, 98% of p$_t$ lie within {idle_band[0]:+.4f} / {idle_band[1]:+.4f} "
                 "of the threshold)", loc="left", fontsize=10, pad=3)
    ax.tick_params(axis="y", labelsize=8)

    # ---- boundary rasters ----
    for ax, n in zip(axes[2:], MODELS):
        d = R[n]
        ax.set_ylim(-0.1, 2.1); ax.set_yticks([0.45, 1.55]); ax.set_yticklabels(["L1", "L2"], fontsize=9)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        counts = []
        for i in np.nonzero(ev & (EV_CUE | EV_QUERY))[0]:
            ax.axvline(xt(W0 + i), color=VIOLET, lw=1.0, ls=(0, (2, 2)), alpha=0.8, zorder=1)
        for i in np.nonzero(ev & EV_LIGHT)[0]:
            ax.axvline(xt(W0 + i), color=ORANGE, lw=1.0, ls=(0, (2, 2)), alpha=0.8, zorder=1)
        for l in range(2):
            b, r = d[f"b{l}"][sl], d[f"r{l}"][sl]
            y0 = l * 1.1
            if n == "fixed_16x":
                ax.vlines(xt(t[b]), y0, y0 + 0.9, color=STRIDE, lw=0.7)
            else:
                ax.vlines(xt(t[b & ~r]), y0, y0 + 0.9, color=HEART, lw=1.1)
                ax.vlines(xt(t[b & r]), y0, y0 + 0.9, color=CHOSEN, lw=0.7)
            counts.append(int(b.sum()))
        ax.set_title(LABELS[n], loc="left", fontsize=10, weight="bold", pad=3)
        ax.text(xt(W1) + 1, 1.0, f"L2 {counts[1]} tok\nL1 {counts[0]} tok", va="center", fontsize=8.5,
                color=INK2, linespacing=1.6)
    leg = [Line2D([], [], color=CHOSEN, lw=2, label="boundary chosen by router / heuristic"),
           Line2D([], [], color=HEART, lw=2, label="heartbeat-forced only (every 64 / 256 frames)"),
           Line2D([], [], color=STRIDE, lw=2, label="fixed stride"),
           Patch(color="#f1f0ec", label="robot moving"),
           Line2D([], [], color=VIOLET, lw=1, ls=(0, (2, 2)), label="cue / query"),
           Line2D([], [], color=ORANGE, lw=1, ls=(0, (2, 2)), label="light toggle")]
    axes[2].legend(handles=leg, loc="lower right", bbox_to_anchor=(1, 1.25), ncol=6, fontsize=8.5,
                   handlelength=1.4)
    axes[-1].set_xlabel("time into the held-out hour (m:ss, 10 frames/s)")
    minute_axis(axes[-1], W0, W1)
    fig.suptitle(f"Where each tokenizer places boundaries — 2.5 min of the held-out stream ({W1 - W0} frames)",
                 x=0.07, ha="left", y=0.985, fontsize=13, weight="bold")
    fig.subplots_adjust(left=0.07, right=0.915, top=0.9, bottom=0.06)
    fig.savefig(os.path.join(OUT, "timeline.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- filmstrip
def event_tag(ev, t, tol=3):
    """Short label for the ground-truth event nearest to frame t (within +-tol)."""
    names = [(EV_CUE, "CUE"), (EV_QUERY, "QUERY"), (EV_GRASP, "grasp"), (EV_RELEASE, "release"),
             (EV_SPAWN, "spawn"), (EV_LIGHT, "LIGHT"), (EV_MOVE_START, "go"), (EV_MOVE_STOP, "stop")]
    best = None
    for dt in sorted(range(-tol, tol + 1), key=abs):
        if 0 <= t + dt < len(ev) and ev[t + dt]:
            for bit, nm in names:
                if ev[t + dt] & bit:
                    return nm
    return best


def filmstrip(st, R, cols=21):
    import matplotlib.patches as mp
    frames = render_frames(st, W0, W1)  # exactly what the model saw (sensor noise included)
    ev = st["events"][ENV]
    moving = moving_of(st)
    rows_per, blocks = {}, []
    for n in ["hnet_16x", "heur_16x"]:
        d = R[n]
        idx = np.nonzero(d["b1"][W0:W1])[0] + W0
        blocks.append((n, idx, d["r1"][idx]))
        rows_per[n] = int(np.ceil(len(idx) / cols))
    nrows = sum(rows_per.values())
    cw, ch = 0.72, 1.0  # inches per cell
    fig = plt.figure(figsize=(cols * cw + 0.6, nrows * ch + 0.62 * len(blocks) + 0.95))
    H = fig.get_figheight(); Wd = fig.get_figwidth()
    y = H - 0.72  # inches from bottom, cursor
    fig.text(0.3 / Wd, (H - 0.32) / H, "What each tokenizer chose to remember: every frame that emitted a level-2 "
             f"token in the same 2.5 min window ({fmt_t(W0)[:-2]}–{fmt_t(W1)[:-2]})", fontsize=12.5, weight="bold", va="center")
    fig.text(0.3 / Wd, (H - 0.62) / H, "Border: blue = chosen by router / heuristic, grey = heartbeat-forced only.  "
             "Label above = ground-truth event within ±3 frames (LIGHT = exogenous light toggle).  "
             "Timestamp below (m:ss.s into the hour).", fontsize=9, color=INK2, va="center")
    for n, idx, chosen in blocks:
        y -= 0.36
        n_ch = int(chosen.sum()); n_mv = int(moving[idx].sum())
        fig.text(0.3 / Wd, y / H, f"{LABELS[n]} — {len(idx)} tokens ({n_ch} chosen, {len(idx) - n_ch} heartbeat); "
                 f"{n_mv} while the robot is moving", fontsize=10.5, weight="bold", va="center")
        y -= 0.12
        for k, (t, c) in enumerate(zip(idx, chosen)):
            r, cidx = divmod(k, cols)
            x0 = 0.3 + cidx * cw
            yy = y - (r + 1) * ch
            ax = fig.add_axes([(x0 + 0.04) / Wd, (yy + 0.2) / H, (cw - 0.08) / Wd, (cw - 0.08) / H])
            ax.imshow(frames[t - W0].clip(0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(CHOSEN if c else HEART); s.set_linewidth(2.2 if c else 1.6)
            ax.text(0.5, -0.06, fmt_t(t), transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=INK2)
            tag = event_tag(ev, t)
            if tag:
                ax.text(0.5, 1.04, tag, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5,
                        color=ORANGE if tag == "LIGHT" else (VIOLET if tag in ("CUE", "QUERY") else INK),
                        weight="bold" if tag in ("CUE", "QUERY", "LIGHT") else "normal")
        y -= rows_per[n] * ch + 0.1
    fig.savefig(os.path.join(OUT, "filmstrip.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- token rate
def token_rate(st, R, win_s=60, min_idle_s=15):
    busy = busy_of(st)
    T = len(busy)
    k = win_s * FPS
    # idle stretches long enough to see at hour scale
    idle_sp = [(a, b) for a, b in spans(~busy) if b - a >= min_idle_s * FPS]
    w = np.hanning(k); w /= w.sum()  # smooth window: avoids the 1-vs-2 heartbeat aliasing of a box count
    series = [("heur_16x", ORANGE), ("hnet_16x", BLUE), ("hnet_cap_r02", AQUA)]
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.3, 1], wspace=0.16)
    ax = fig.add_subplot(gs[0])
    for a, b in idle_sp:
        ax.axvspan(a / FPS / 60, b / FPS / 60, color="#e4e2da", lw=0, zorder=0)
    tm = np.arange(T) / FPS / 60
    ends = {}
    for n, c in series:
        m = R[n]["b1"].astype(float)
        rate = np.convolve(m, w, mode="same") / np.convolve(np.ones(T), w, mode="same") * (60 * FPS)
        ax.plot(tm, rate, color=c, lw=1.4, label=LABELS[n])
        ends[n] = rate
    hb = 60 * FPS / 256
    ax.axhline(hb, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(60.4, hb + 1, f"heartbeat\nfloor\n{hb:.1f}/min", va="bottom", fontsize=8.5,
            color=INK2)
    ax.set_xlim(0, 60); ax.set_ylim(0, None)
    ax.set_xticks(np.arange(0, 61, 5))
    ax.set_xlabel("minutes into the held-out hour")
    ax.set_ylabel(f"level-2 tokens / min ({win_s} s Hann window)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=c, lw=2, label=LABELS[n] + (" (= heartbeat floor)" if n == "hnet_cap_r02" else ""))
                       for n, c in series]
              + [Patch(color="#e4e2da", label=f"idle stretch ≥ {min_idle_s} s (no motion, no event)")],
              loc="upper left", bbox_to_anchor=(0, 1.08), ncol=4, fontsize=9)
    ax.set_title("Level-2 token rate over one hour (env 0)", loc="left", fontsize=12, weight="bold", pad=26)

    # ---- idle vs active averages over all 6 held-out streams (eval_hour.json) ----
    ax2 = fig.add_subplot(gs[1])
    names = [n for n, _ in series]
    vals = []
    for n in names:
        L = json.load(open(f"runs/{n}/eval_hour.json"))["summary"]["levels"][-1]
        vals.append((L["tok_per_min_idle"], L["tok_per_min_active"]))
    y = np.arange(len(names))[::-1]
    for yi, (n, c), (vi, va) in zip(y, series, vals):
        ax2.barh(yi + 0.19, va, height=0.34, color=c, edgecolor="white", lw=1)
        ax2.barh(yi - 0.19, vi, height=0.34, color=c, alpha=0.35, edgecolor="white", lw=1)
        ax2.text(va + 1, yi + 0.19, f"active {va:.1f}", va="center", fontsize=8.5, color=INK)
        ax2.text(vi + 1, yi - 0.19, f"idle {vi:.1f}", va="center", fontsize=8.5, color=INK)
    short = {"heur_16x": "heuristic", "hnet_16x": "H-Net 16x", "hnet_cap_r02": "budget cap"}
    ax2.set_yticks(y); ax2.set_yticklabels([short[n] for n in names], fontsize=9.5)
    ax2.set_xlim(0, max(v for p in vals for v in p) * 1.45)
    ax2.set_xlabel("L2 tokens / min")
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("idle vs active (6 streams)", loc="left", fontsize=10.5, weight="bold", pad=26)
    fig.subplots_adjust(left=0.05, right=0.975, top=0.86, bottom=0.11)
    fig.savefig(os.path.join(OUT, "token_rate.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    st = stream()
    R = {n: run(n, st) for n in MODELS}
    which = sys.argv[1:] or ["timeline", "filmstrip", "token_rate"]
    for w in which:
        globals()[w](st, R)
        print("wrote", os.path.join(OUT, w + ".png"))
