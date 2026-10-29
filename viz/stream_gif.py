"""docs/figures/stream.gif: the held-out stream playing at 5x with live token counters.

Rows: dense (1 token / frame), learned H-Net 16x level 1 and level 2, and the pixel-change heuristic's level 2
for reference. A lamp flashes when a row emits a token; the strip to the right shows the last 20 s of tokens
(blue / orange = chosen by router / heuristic, grey = heartbeat-forced).

  .venv/bin/python viz/stream_gif.py
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from vt.env import EV_GRASP, EV_RELEASE, EV_SPAWN, EV_CUE, EV_QUERY, EV_LIGHT

T0, T1 = 18600, 19400        # 80 s: green cue, 45 s of idle, query -> recall, pick-and-place, blue cue
STEP, GIF_FPS = 2, 25        # 2 sim frames per gif frame at 25 fps = 50 frames/s = 5x real time (10 fps)
SCALE = 4
HIST = 200                   # raster history: 20 s
COLOR_NAMES = ["red", "green", "blue", "yellow"]


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def font(sz, bold=False):
    import matplotlib
    f = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", f), sz)


def labels_for(st, T0, T1):
    """Per-frame text: persistent state + a transient event caption held for 2 s."""
    ev = st["events"][ENV]; grip = st["action"][ENV, :, 2] > 0.5; moving = moving_of(st)
    state, caption = [], []
    cur, cur_until = "", -1
    for t in range(T0, T1):
        e = ev[t]
        if e & EV_CUE:
            cur, cur_until = f"CUE flashed: {COLOR_NAMES[st['cue_col'][ENV, t]]}  (remember it)", t + 25
        elif e & EV_QUERY:
            cur, cur_until = f"QUERY: go to the {COLOR_NAMES[st['query_answer'][ENV, t]]} bin  (cue was " \
                             f"{(t - st['query_cue_t'][ENV, t]) / FPS:.0f} s ago)", t + 35
        elif e & EV_LIGHT:
            cur, cur_until = "room light toggled  (exogenous)", t + 20
        elif e & EV_GRASP:
            cur, cur_until = "grasp", t + 15
        elif e & EV_RELEASE:
            cur, cur_until = "release into bin", t + 15
        elif e & EV_SPAWN:
            cur, cur_until = "new object appears", t + 15
        caption.append(cur if t <= cur_until else "")
        state.append("carrying" if grip[t] else ("moving" if moving[t] else "idle"))
    return state, caption


def main():
    st = stream()
    R = {n: run(n, st) for n in ["hnet_16x", "heur_16x"]}
    # shown without the 2% sensor noise the model sees (noise makes the GIF ~2x larger; it is barely visible)
    frames = render_frames(st, T0, T1, noise=False)
    rows = [("dense (1 token / frame)", np.ones(N_STEPS, bool), np.ones(N_STEPS, bool), INK2),
            ("H-Net 16x  level 1", R["hnet_16x"]["b0"], R["hnet_16x"]["r0"], BLUE),
            ("H-Net 16x  level 2", R["hnet_16x"]["b1"], R["hnet_16x"]["r1"], BLUE),
            ("heuristic  level 2", R["heur_16x"]["b1"], R["heur_16x"]["r1"], ORANGE)]
    state, caption = labels_for(st, T0, T1)

    VW = 64 * SCALE
    PAD = 16
    W, H = VW + 2 * PAD + 620, VW + 2 * PAD + 64
    f_s, f_m, f_b, f_l = font(13), font(15), font(15, True), font(24, True)
    bg = (255, 255, 255); ink = hex2rgb(INK); ink2 = hex2rgb(INK2); muted = hex2rgb(MUTED)
    heart = hex2rgb("#b9b7ae"); grid = hex2rgb(GRID); violet = hex2rgb(VIOLET)
    x_panel = VW + 2 * PAD + 8
    label_w, lamp_w, count_w = 205, 26, 64
    x_lamp = x_panel + label_w; x_cnt = x_lamp + lamp_w; x_strip = x_cnt + count_w + 10
    strip_w = W - PAD - x_strip
    row_y0, row_h = PAD + 40, 46
    imgs = []
    lit = np.zeros(len(rows))
    for gi, t in enumerate(range(T0, T1, STEP)):
        im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
        # video
        fr = (frames[t - T0].clip(0, 1) * 255).astype(np.uint8)
        im.paste(Image.fromarray(fr).resize((VW, VW), Image.NEAREST), (PAD, PAD))
        d.text((PAD, PAD + VW + 8), f"t = {fmt_t(t)}   ·   5x speed   ·   {state[t - T0]}", font=f_s, fill=ink2)
        cap = caption[t - T0]
        if cap:
            d.text((PAD, PAD + VW + 30), cap, font=f_b, fill=violet if cap.startswith(("CUE", "QUERY")) else ink)
        # panel header
        d.text((x_panel, PAD), "tokens emitted since start of clip", font=f_b, fill=ink)
        d.text((x_strip, PAD + 20), "last 20 s  →  now", font=f_s, fill=muted)
        for ri, (name, b, r, col) in enumerate(rows):
            y = row_y0 + ri * row_h
            c = hex2rgb(col)
            fired = b[t:t + STEP].any()
            lit[ri] = 1.0 if fired else lit[ri] * 0.45
            d.text((x_panel, y + 8), name, font=f_m, fill=ink)
            # lamp
            a = lit[ri]
            lc = tuple(int(255 * (1 - a) + c[i] * a) for i in range(3))
            d.ellipse([x_lamp, y + 8, x_lamp + 18, y + 26], fill=lc, outline=c, width=2)
            n = int(b[T0:t + STEP].sum())
            d.text((x_cnt, y + 4), f"{n}", font=f_l if ri else font(24), fill=ink if ri else ink2)
            # history raster
            d.rectangle([x_strip, y + 4, x_strip + strip_w, y + 32], fill=(249, 249, 247))
            h0 = t + STEP - HIST
            for k in np.nonzero(b[h0:t + STEP])[0]:
                x = x_strip + int(k / HIST * strip_w)
                d.line([x, y + 6, x, y + 30], fill=(c if r[h0 + k] else heart) if ri else muted, width=1)
        # compression line
        yb = row_y0 + len(rows) * row_h + 6
        nd = t + STEP - T0
        n1 = int(rows[1][1][T0:t + STEP].sum()); n2 = int(rows[2][1][T0:t + STEP].sum())
        n3 = int(rows[3][1][T0:t + STEP].sum())
        d.text((x_panel, yb), f"frames per token so far:  H-Net L1 {nd / max(n1, 1):.1f}  ·  H-Net L2 "
               f"{nd / max(n2, 1):.1f}  ·  heuristic L2 {nd / max(n3, 1):.1f}", font=f_s, fill=ink2)
        d.text((x_panel, yb + 22), "lamp = token emitted now;  grey ticks = heartbeat-forced (every 6.4 s / 25.6 s)",
               font=f_s, fill=muted)
        imgs.append(im)
    # shared palette keeps the file small and colours stable across frames
    pal = imgs[len(imgs) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    out = os.path.join(OUT, "stream.gif")
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / GIF_FPS), loop=0, optimize=True)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.2f} MB, {len(q)} frames, {len(q) / GIF_FPS:.1f} s")
    imgs[len(imgs) // 2].save(os.path.join(CACHE, "gif_mid.png"))
    for i in (34, 258):
        imgs[i].save(os.path.join(CACHE, f"gif_{i}.png"))


if __name__ == "__main__":
    main()
