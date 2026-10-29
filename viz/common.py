"""Shared loading / inference / styling for the README behaviour figures.

Runs each checkpoint over the held-out "hour" stream (simulate(6, 36000, seed=9999), env 0) exactly the way
eval.eval_stream does, and caches per-frame boundary masks, router probabilities, and the heartbeat/router
split to viz/.cache/<run>.npz so figure scripts can be re-run without the GPU.

Heartbeat vs router: Stage.boundaries ORs a heartbeat (floor(t/64) at L1, floor(t/256) at L2 changes) into the
router's (or heuristic's) boundaries. We wrap Stage.boundaries on the model instance (vt/ is untouched) to also
record what the router/heuristic alone would have chosen, so every emitted token is labelled
"router" (chooser fired, whether or not a heartbeat also fell there) or "heartbeat" (forced only).
"""
import os, sys, types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
CACHE = os.path.join(ROOT, "viz", ".cache")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)

FPS = 10  # simulator frame rate used throughout the repo (report.py: t/10 = seconds)
ENV, N_STEPS, SEED = 0, 36000, 9999

# ---- style (dataviz reference palette, light mode on white) ----
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
LABELS = {"fixed_16x": "fixed stride (16x)", "heur_16x": "pixel-change heuristic (16x)",
          "hnet_16x": "learned H-Net (16x)", "hnet_16x_videoonly": "learned, video-only (no actions)",
          "hnet_cap_r02": "learned + budget cap", "iso_1x": "dense (1 token / frame)"}


def mpl_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
        "legend.frameon": False,
    })
    return plt


def stream():
    from vt.env import simulate
    return simulate(6, N_STEPS, seed=SEED)


def _wrap_boundaries(stage):
    orig = stage.boundaries

    def boundaries(h, meta):
        hb = stage.heartbeat
        stage.heartbeat = 0
        p_raw, b_raw = orig(h, meta)  # chooser alone (router / heuristic / fixed stride)
        stage.heartbeat = hb
        p, b = orig(h, meta)
        stage._raw = (p_raw.detach(), b_raw.detach())
        return p, b

    stage.boundaries = boundaries


def run(name, st=None, force=False):
    """Returns dict with per-frame arrays (frame time, length T) for each level l:
    b{l} emitted token, r{l} chooser fired (router/heuristic), p{l} router prob after heartbeat max(p, 0.5),
    pr{l} raw router prob before the heartbeat (NaN where level l has no input
    position, i.e. for l>=1 at frames that are not L(l-1) tokens)."""
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    import torch
    from eval import load, encode_stream
    from vt.model import Stage
    st = stream() if st is None else st
    model, ck = load(name)
    stages, m = [], model.body
    while isinstance(m, Stage):
        _wrap_boundaries(m); stages.append(m); m = m.inner
    with torch.no_grad():
        x, score = encode_stream(model, st, ENV)
        act = torch.as_tensor(st["action"][ENV], device="cuda")[None]
        prev_act = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)
        ext = {i: score > th for i, th in enumerate(ck["thresholds"])} if ck["thresholds"] else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, infos = model(None, prev_act, ext=ext, x_lat=x)
    T = x.shape[1]
    out = {"score": score[0].float().cpu().numpy(), "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    frame_of = np.arange(T)
    for l, (inf, s) in enumerate(zip(infos, stages)):
        b = inf["b"][0].cpu().numpy(); p = inf["p"][0].float().cpu().numpy()
        praw, braw = s._raw[0][0].float().cpu().numpy(), s._raw[1][0].cpu().numpy()
        valid = inf["valid"][0].cpu().numpy()
        for k, v, fill in [("b", b & valid, False), ("r", braw & valid, False), ("p", p, np.nan), ("pr", praw, np.nan)]:
            a = np.full(T, fill, dtype=float if k.startswith("p") else bool)
            a[frame_of[valid]] = v[valid]
            out[f"{k}{l}"] = a
        order = inf["order"][0].cpu().numpy(); cv = inf["cvalid"][0].cpu().numpy()
        frame_of = frame_of[order[cv]]
    np.savez_compressed(path, **out)
    del model; torch.cuda.empty_cache()
    return out


def moving_of(st):
    return np.abs(st["action"][ENV, :, :2]).sum(1) > 0.2


def busy_of(st):
    """eval.py's 'active' definition: within 5 frames of robot motion or any event."""
    from eval import near
    ev = st["events"][ENV]
    return near(moving_of(st) | (ev > 0), np.arange(ev.shape[0]), tol=5)


def render_frames(st, t0, t1, noise=True):
    """Frames exactly as the model saw them (same generator seed as eval.encode_stream) for [t0, t1)."""
    import torch
    from vt.env import Renderer, flat
    r = Renderer("cuda")
    gen = torch.Generator(device="cuda").manual_seed(1234 + ENV)
    # encode_stream renders in 8192-frame chunks with one generator; replay it to get identical noise
    frames = []
    for c0 in range(0, t1, 8192):
        c1 = min(N_STEPS, c0 + 8192)
        f, clean = r(flat(st, ENV, c0, c1), gen=gen)
        if c1 > t0:
            g = f if noise else clean
            frames.append(g[max(0, t0 - c0):min(c1, t1) - c0].permute(0, 2, 3, 1).cpu().numpy())
        if c1 >= t1:
            break
    return np.concatenate(frames)


def fmt_t(t):
    s = t / FPS
    return f"{int(s // 60)}:{s % 60:04.1f}"
