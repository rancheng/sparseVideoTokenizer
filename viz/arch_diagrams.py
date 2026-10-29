"""Hand-written SVG architecture diagrams for the README / docs.

    .venv/bin/python viz/arch_diagrams.py        # writes docs/figures/{arch_pipeline,arch_streaming,hnet_mapping}.{svg,png}

Every figure has an explicit white background so it reads on GitHub in light and dark mode.
Text markup: "h_{t}" -> subscript, "x^{2}" -> superscript (rendered with <tspan dy>).
"""
import os
import re

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")
FONT = "DejaVu Sans, Verdana, Arial, Helvetica, sans-serif"
MONO = "DejaVu Sans Mono, Menlo, Consolas, monospace"

INK, MUTED, LINE = "#1f2937", "#5b6472", "#374151"
ROUTER, HEART, ACTION, RESID = "#6d28d9", "#d97706", "#047857", "#7b8494"
BAND = {"frame": ("#edf4fc", "#bcd3ee"), "l1ev": ("#f2effb", "#cfc6ee"), "event": ("#fdf0e4", "#efc9a5")}
BOX_FILL, BOX_STROKE = "#ffffff", "#4b5563"


_FONTS = {}


def text_width(s, size, weight="normal"):
    """Width in px of s in DejaVu Sans (the font the PNGs are rendered with; Verdana has near-identical metrics)."""
    from PIL import ImageFont
    key = (weight, round(size * 10))
    if key not in _FONTS:
        f = "DejaVuSans-Bold.ttf" if weight == "bold" else "DejaVuSans.ttf"
        _FONTS[key] = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/" + f, size * 10)
    return _FONTS[key].getlength(s) / 10


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h, label):
        self.w, self.h, self.parts = w, h, []
        self.label = label

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill=BOX_FILL, stroke=BOX_STROKE, sw=1.4, rx=8, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, weight="normal", anchor="start", fill=INK, family=FONT, italic=False, ls=None):
        toks = [t for t in re.split(r"(_\{[^}]*\}|\^\{[^}]*\})", s) if t]
        if anchor != "start" and len(toks) > 1:
            # renderers (cairosvg) anchor each <tspan> separately; resolve the anchor to an explicit start x
            wtot = sum(text_width(t[2:-1], size * 0.72, weight) if t[0] in "_^" and t[1:2] == "{" else
                       text_width(t, size, weight) for t in toks)
            x, anchor = x - (wtot / 2 if anchor == "middle" else wtot), "start"
        st = ' font-style="italic"' if italic else ""
        l = f' letter-spacing="{ls}"' if ls else ""
        out, shift = [], 0.0
        for tok in re.split(r"(_\{[^}]*\}|\^\{[^}]*\})", s):
            if not tok:
                continue
            if tok[0] in "_^" and tok[1:2] == "{":
                target = size * 0.32 if tok[0] == "_" else -size * 0.42
                out.append(f'<tspan dy="{target - shift:.2f}" font-size="{size * 0.72:.1f}">{esc(tok[2:-1])}</tspan>')
                shift = target
            else:
                dy = f' dy="{-shift:.2f}"' if shift else ""
                out.append(f'<tspan{dy}>{esc(tok)}</tspan>')
                shift = 0.0
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" fill="{fill}"{st}{l}>{"".join(out)}</text>')

    def line(self, pts, color=LINE, sw=1.6, dash=None, arrow=True, head=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#ah-{head or color[1:]})"' if arrow else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{d}{m}/>')

    def circle(self, x, y, r, fill="#fff", stroke=LINE, sw=1.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def plus(self, x, y, r=11, color=LINE):
        self.circle(x, y, r, "#fff", color)
        self.add(f'<path d="M{x - r + 4},{y} H{x + r - 4} M{x},{y - r + 4} V{y + r - 4}" stroke="{color}" stroke-width="1.6"/>')

    def diamond(self, x, y, r, fill, stroke):
        self.add(f'<polygon points="{x},{y - r} {x + r},{y} {x},{y + r} {x - r},{y}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')

    def svg(self):
        markers = []
        for c in {LINE, ROUTER, HEART, ACTION, RESID, MUTED}:
            markers.append(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                           f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" aria-label="{esc(self.label)}">\n<title>{esc(self.label)}</title>\n'
                f'<defs>{"".join(markers)}</defs>\n'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>\n' + "\n".join(self.parts) + "\n</svg>\n")


def box(s, x, y, w, h, title, lines=(), fill=BOX_FILL, stroke=BOX_STROKE, tsize=14, lsize=12.5, tcolor=INK):
    s.rect(x, y, w, h, fill, stroke)
    n = 1 + len(lines)
    lh = 19
    y0 = y + h / 2 - (n - 1) * lh / 2 + 5
    s.text(x + w / 2, y0, title, tsize, "bold", "middle", tcolor)
    for i, l in enumerate(lines):
        s.text(x + w / 2, y0 + (i + 1) * lh, l, lsize, "normal", "middle", MUTED if i else INK)


def save(s, name):
    import cairosvg
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p + ".svg", "w") as f:
        f.write(s.svg())
    cairosvg.svg2png(url=p + ".svg", write_to=p + ".png", output_width=s.w * 2, output_height=s.h * 2)
    print("wrote", p + ".svg", p + ".png")


# --------------------------------------------------------------------------------------------------------------
# 1. full pipeline
# --------------------------------------------------------------------------------------------------------------
LX, LW = 60, 280            # left spine boxes (encoder side); centre 200
LC = LX + LW / 2
MX, MW = 430, 400           # router (middle); centre 630
MC = MX + MW / 2
RX, RW = 900, 350           # right spine boxes (decoder side); centre 1075
RC = RX + RW / 2


def level_band(s, y0, lvl, d, d_in, hb, tier, header2):
    fill, stroke = BAND[tier]
    s.rect(20, y0, 1260, 334, fill, stroke, 1.2, 12)
    s.text(MC, y0 + 21, f"STAGE {lvl}  ·  d = {d}", 13, "bold", "middle", INK, ls="0.5")
    s.text(MC, y0 + 39, header2, 12.5, "normal", "middle", MUTED)
    uy, uh = y0 + 52, 84
    # row U: chunk | router | dechunk
    box(s, LX, uy, LW, uh, "chunk", [f"gather rows with b_{{t}} = 1 → z_{{k}}, P_{{k}}",
                                     f"up-proj {d}→{d_in} + Δt embedding"])
    box(s, MX, uy, MW, uh, "router", ["p_{t} = ½ (1 − cos(W_{q} h_{t}, W_{k} h_{t−1}))",
                                      f"b_{{t}} = [p_{{t}} ≥ ½]  or  heartbeat (every {hb} frames)"],
        stroke=ROUTER, tcolor=ROUTER)
    s.rect(RX, uy, RW, uh + 22, BOX_FILL, BOX_STROKE)
    ty = uy + 22
    s.text(RC, ty, "dechunk", 14, "bold", "middle")
    s.text(RC, ty + 19, f"down-proj {d_in}→{d}", 12.5, "normal", "middle")
    s.text(RC, ty + 38, "EMA:  z̄_{k} = P_{k} z_{k} + (1 − P_{k}) z̄_{k−1}", 12.5, "normal", "middle", MUTED)
    s.text(RC, ty + 57, "causal repeat to every position", 12.5, "normal", "middle", MUTED)
    s.text(RC, ty + 76, "× STE confidence (b_{t} ? p_{t} : 1 − p_{t})", 12.5, "normal", "middle", MUTED)
    # row L: encoder ; plus ; decoder
    ey = y0 + 182
    box(s, LX, ey, LW, 56, f"stage-{lvl} encoder", ["2× SWA block · W = 64 · local RoPE"])
    py = y0 + 210
    s.plus(RC, py)
    dy = y0 + 256
    box(s, RX, dy, RW, 56, f"stage-{lvl} decoder", ["2× SWA block · W = 64 · local RoPE"])
    # arrows inside band
    s.line([(LC, ey), (LC, uy + uh + 2)])                                       # enc -> chunk
    by = (ey + uy + uh) / 2 + 2
    s.line([(LC, by), (MC, by), (MC, uy + uh + 2)])                             # h_t -> router
    s.text(LC + 14, by - 6, "h_{t}", 12.5, "normal", "start", INK)
    s.line([(MX, uy + 42), (LX + LW + 2, uy + 42)], ROUTER, 1.6, "6 4")          # b -> chunk
    s.text((MX + LX + LW) / 2, uy + 35, "b_{t}", 12.5, "bold", "middle", ROUTER)
    s.line([(MX + MW, uy + 42), (RX - 2, uy + 42)], ROUTER, 1.6, "6 4")          # p,b -> dechunk
    s.text((MX + MW + RX) / 2, uy + 35, "p_{t}, b_{t}", 12.5, "bold", "middle", ROUTER)
    s.line([(LX + LW, py), (RC - 13, py)], RESID, 1.8)                           # residual
    s.text(MC, py - 8, "residual  Linear(h_{t})   (skip around the inner network)", 12.5, "normal", "middle", RESID)
    s.line([(RC, uy + uh + 22), (RC, py - 13)])                                  # dechunk -> plus
    s.line([(RC, py + 11), (RC, dy - 2)])                                        # plus -> decoder
    return dict(chunk_top=uy, dechunk_top=uy, enc_bot=ey + 56, dec_bot=dy + 56)


def fig_pipeline():
    W, H = 1300, 1200
    s = Svg(W, H, "VideoHNet pipeline: frame encoder, two H-Net chunking stages, a full-attention main network over "
                  "stage-2 events, dechunking back to frame rate, and per-frame policy and forward-model heads.")
    s.text(40, 38, "VideoHNet: a causal, action-conditioned 2-stage video H-Net", 21, "bold")
    s.text(40, 62, "Shapes for a T-frame clip, batch B; 64×64 RGB frames at 10 Hz. Stage 1 keeps ≈1 in 4 frames, "
                   "stage 2 ≈1 in 16–23 frames (learned, content-dependent).", 13, "normal", "start", MUTED)

    # main band
    my = 78
    s.rect(20, my, 1260, 124, *BAND["event"], 1.2, 12)
    s.text(MC, my + 22, "MAIN NETWORK  ·  d = 512  ·  expensive, global  ·  runs on stage-2 events only",
           13, "bold", "middle", INK, ls="0.5")
    s.rect(LX - 10, my + 34, 1200, 78, BOX_FILL, BOX_STROKE)
    s.text(LX + 10, my + 64, "8× full-causal attention blocks, d = 512", 15, "bold")
    s.text(LX + 10, my + 87, "NoPE: no positional encoding; time enters only via the Δt embedding", 12.5, "normal",
           "start", MUTED)
    kx = 760
    for i in range(14):
        s.rect(kx + i * 19, my + 50, 15, 15, "#f6c79b" if i < 13 else "#fff", HEART, 1.2, 2, dash=None if i < 13 else "3 2")
    s.text(kx + 14 * 19 + 6, my + 62, "+1", 12.5, "bold", "start", HEART)
    s.text(kx, my + 92, "KV cache = long-term memory: +1 entry per stage-2 event", 12.5, "normal", "start", INK)

    b2 = level_band(s, 212, 2, 384, 512, 256, "l1ev", "cheap, local  ·  runs on stage-1 events only (L_{1} ≈ T/4 positions)")
    b1 = level_band(s, 556, 1, 256, 384, 64, "frame", "cheap, local  ·  runs every frame (T positions)")

    # spines between levels
    s.line([(LC, b2["chunk_top"]), (LC, my + 113)])
    s.text(LC + 12, 236, "(B, L_{2} ≈ T/16–T/23, 512)", 12.5, "bold", "start", INK)
    s.line([(RC, my + 112), (RC, b2["dechunk_top"] - 2)])
    s.text(RC + 12, 236, "(B, L_{2}, 512)", 12.5, "bold", "start", INK)
    s.line([(LC, b1["chunk_top"]), (LC, b2["enc_bot"] + 2)])
    s.text(LC + 12, 520, "(B, L_{1} ≈ T/4, 384)", 12.5, "bold", "start", INK)
    s.line([(RC, b2["dec_bot"]), (RC, b1["dechunk_top"] - 2)])
    s.text(RC + 12, 584, "(B, L_{1}, 384)", 12.5, "bold", "start", INK)

    # frame I/O + heads band
    y3 = 900
    s.rect(20, y3, 1260, 190, *BAND["frame"], 1.2, 12)
    s.text(MC, y3 + 21, "FRAME ENCODER AND HEADS  ·  d = 256", 13, "bold", "middle", INK, ls="0.5")
    s.text(MC, y3 + 39, "cheap, local  ·  runs every frame", 12.5, "normal", "middle", MUTED)
    px, py = LC, y3 + 40
    s.plus(px, py)
    s.line([(px, py - 11), (px, b1["enc_bot"] + 2)])
    s.text(LC + 12, 856, "x_{t}  (B, T, 256)", 12.5, "bold", "start", INK)
    s.rect(262, py - 18, 136, 36, "#ecfdf5", ACTION)
    s.text(330, py + 5, "Linear(a_{t−1})", 12.5, "bold", "middle", ACTION)
    s.line([(262, py), (px + 13, py)], ACTION, 1.8)
    box(s, LX, y3 + 68, LW, 58, "FrameEncoder", ["4× [conv 4×4/2, GN, SiLU] → Linear"])
    s.line([(px, y3 + 68), (px, py + 13)])
    s.line([(px, y3 + 160), (px, y3 + 128)])
    s.text(px, y3 + 178, "o_{t}   (B, T, 3, 64, 64)", 12.5, "bold", "middle", INK)
    s.text(262 + 150, y3 + 178, "a_{t−1}: previous action", 12.5, "normal", "start", ACTION)

    # heads
    s.line([(RC, b1["dec_bot"]), (RC, y3 + 66)])
    s.text(RC + 12, 906, "h_{t}  (B, T, 256)", 12.5, "bold", "start", INK)
    hx = 745
    s.line([(RC, y3 + 48), (hx, y3 + 48), (hx, y3 + 66)])
    box(s, 600, y3 + 68, 290, 58, "policy head", ["RMSNorm → MLP → â_{t} = (vx, vy, grip)"])
    box(s, 930, y3 + 68, 290, 58, "forward model", ["MLP(h_{t} + Linear(a_{t})) → FrameDecoder"])
    s.line([(1272, y3 + 97), (1222, y3 + 97)], ACTION, 1.8)
    s.text(1250, y3 + 90, "a_{t}", 12.5, "bold", "middle", ACTION)
    s.line([(hx, y3 + 128), (hx, y3 + 158)])
    s.text(hx, y3 + 176, "â_{t}   (B, T, 3)", 12.5, "bold", "middle", INK)
    s.line([(RC, y3 + 128), (RC, y3 + 158)])
    s.text(RC, y3 + 176, "ô_{t+1}   (B, T, 3, 64, 64)", 12.5, "bold", "middle", INK)

    # legend + loss
    ly = 1118
    s.line([(40, ly), (80, ly)]); s.text(88, ly + 4, "activations", 12.5)
    s.line([(190, ly), (230, ly)], ROUTER, 1.6, "6 4"); s.text(238, ly + 4, "router signal", 12.5)
    s.line([(360, ly), (400, ly)], RESID, 1.8); s.text(408, ly + 4, "residual skip", 12.5)
    s.line([(530, ly), (570, ly)], ACTION, 1.8); s.text(578, ly + 4, "action input", 12.5)
    for i, (k, lab) in enumerate([("frame", "every frame"), ("l1ev", "stage-1 events"), ("event", "stage-2 events")]):
        x = 720 + i * 180
        s.rect(x, ly - 9, 26, 18, *BAND[k], 1.2, 4)
        s.text(x + 34, ly + 4, lab, 12.5)
    s.text(40, ly + 40, "Training loss  =  100 · MSE(ô_{t+1}, o_{t+1})  +  MSE(vx, vy)  +  0.1 · BCE(grip)  +  0.03 · Σ_{stages} "
                        "H-Net ratio loss   (or budget-cap mode: ratio loss only when over budget)", 12.5, "normal", "start", INK)
    s.text(40, ly + 62, "Everything is causal: router compares h_{t} with h_{t−1}, windowed attention looks back only, "
                        "dechunk at t uses the chunk that started at or before t.", 12.5, "normal", "start", MUTED)
    save(s, "arch_pipeline")


# --------------------------------------------------------------------------------------------------------------
# 2. streaming engine
# --------------------------------------------------------------------------------------------------------------
def fig_streaming():
    t0, n = 244, 24
    l1_router = [245, 248, 252, 259, 261, 265]
    l1_hb = [256]
    l2_router = [261]
    l2_hb = [256]
    l1 = sorted(l1_router + l1_hb)
    l2 = sorted(l2_router + l2_hb)
    assert set(l2) <= set(l1)

    W, H = 1300, 640
    s = Svg(W, H, "Event-driven streaming over 24 frames: frame encoder, stage 1 and heads run every frame; stage 2 runs "
                  "only on stage-1 boundaries; the main network steps and appends one KV-cache entry only on stage-2 "
                  "boundaries.")
    s.text(40, 38, "Streaming engine (vt/stream.py): one frame in, one action out", 21, "bold")
    s.text(40, 62, f"24 consecutive frames, t = {t0}…{t0 + n - 1} at 10 Hz (illustrative boundary pattern). "
                   "Filled cell = module runs at that frame; empty = skipped.", 13, "normal", "start", MUTED)

    cx0, cw = 330, 36
    colx = lambda t: cx0 + (t - t0) * cw + cw / 2
    rows = {"kv": 96, "main": 160, "l2b": 214, "l2": 236, "l1b": 290, "l1": 312, "frame": 372}
    ch = 34
    # heartbeat column highlight
    for t in l1_hb:
        x = colx(t) - cw / 2
        s.rect(x + 1, 84, cw - 2, 344, "#fff7ed", HEART, 1.0, 4, dash="4 3")
        s.text(colx(t), 446, "heartbeat", 12, "bold", "middle", HEART)
        s.text(colx(t), 462, f"t = {t}: multiple of 64 and of 256", 12, "normal", "middle", HEART)

    labels = [("kv", "main KV cache (entries)", "grows +1 per event"),
              ("main", "main network: 8 blocks, 1 step", "+ append 1 KV entry"),
              ("l2", "stage-2 encoder → router → decoder", "1 step each, W = 64 rings"),
              ("l1", "frame enc + stage-1 enc/router/dec", "+ policy / forward heads"),
              ("frame", "frame o_{t} and a_{t−1} in", "")]
    for k, a, b in labels:
        y = rows[k] + (ch / 2 if k != "kv" else 22) + (0 if b else 5)
        s.text(cx0 - 14, y - (2 if b else 0), a, 12.5, "bold", "end")
        if b:
            s.text(cx0 - 14, y + 15, b, 12, "normal", "end", MUTED)
    for k, a in [("l2b", "stage-2 boundary"), ("l1b", "stage-1 boundary")]:
        s.text(cx0 - 14, rows[k] + 15, a, 12, "normal", "end", MUTED, italic=True)

    for t in range(t0, t0 + n):
        x = colx(t) - cw / 2 + 3
        w = cw - 6
        # stage 1 + frame encoder: every frame
        s.rect(x, rows["l1"], w, ch, "#9cc3ea", "#4f86c6", 1.2, 4)
        # stage 2
        if t in l1:
            s.rect(x, rows["l2"], w, ch, "#b9abe6", "#6d5bc4", 1.2, 4)
        else:
            s.rect(x, rows["l2"], w, ch, "#ffffff", "#d4d0e4", 1.0, 4, dash="3 3")
        # main
        if t in l2:
            s.rect(x, rows["main"], w, ch, "#f4a96b", "#c2621a", 1.2, 4)
        else:
            s.rect(x, rows["main"], w, ch, "#ffffff", "#ead9c8", 1.0, 4, dash="3 3")
        # boundary markers
        for key, ev_r, ev_h in [("l1b", l1_router, l1_hb), ("l2b", l2_router, l2_hb)]:
            cy = rows[key] + 11
            if t in ev_h:
                s.diamond(colx(t), cy, 8, HEART, "#92400e")
            elif t in ev_r:
                s.circle(colx(t), cy, 6.5, ROUTER, "#4c1d95", 1.2)
        # frame thumbnail: table, block, gripper moving during the event
        fy = rows["frame"]
        s.rect(x, fy, w, 30, "#e5e7eb", "#9ca3af", 1.0, 3)
        s.rect(x, fy + 21, w, 9, "#cbd5e1", "none", 0, 0)
        gx = x + 6 + (min(max(t - 257, 0), 5) * 3.2)
        s.rect(x + w - 11, fy + 15, 7, 6, "#64748b", "none", 0, 1)
        s.add(f'<path d="M{gx:.1f},{fy} V{fy + 12 + (3 if 259 <= t <= 263 else 0)}" stroke="#334155" stroke-width="2.2"/>')
        s.text(colx(t), fy + 48, str(t), 11, "normal", "middle", MUTED)
    # connectors: boundary -> next level runs
    for t in l1:
        s.line([(colx(t), rows["l1"] - 1), (colx(t), rows["l1b"] + 20)], MUTED, 1.0, arrow=False)
        s.line([(colx(t), rows["l1b"] + 2), (colx(t), rows["l2"] + ch + 2)], MUTED, 1.2, head=MUTED[1:])
    for t in l2:
        s.line([(colx(t), rows["l2"] - 1), (colx(t), rows["l2b"] + 20)], MUTED, 1.0, arrow=False)
        s.line([(colx(t), rows["l2b"] + 2), (colx(t), rows["main"] + ch + 2)], MUTED, 1.2, head=MUTED[1:])

    # KV staircase
    ky, unit = rows["kv"], 12
    base = ky + 44
    pts, cnt = [], 0
    for t in range(t0, t0 + n):
        if t in l2:
            cnt += 1
        xl, xr = colx(t) - cw / 2, colx(t) + cw / 2
        pts += [(xl, base - 10 - cnt * unit), (xr, base - 10 - cnt * unit)]
    poly = [(colx(t0) - cw / 2, base)] + pts + [(colx(t0 + n - 1) + cw / 2, base)]
    s.add('<polygon points="' + " ".join(f"{x},{y}" for x, y in poly) + f'" fill="#fde3cb" stroke="{HEART}" stroke-width="1.4"/>')
    seg = [(t0, l2[0], "n"), (l2[0], l2[1], "n + 1"), (l2[1], t0 + n, "n + 2")]
    for i, (a, b, lab) in enumerate(seg):
        xm = (colx(a) - cw / 2 + colx(b) - cw / 2) / 2
        s.text(xm, base - 16 - i * unit, lab, 12, "bold", "middle", "#9a3412")

    # run counts
    rx = cx0 + n * cw + 22
    s.text(rx + 26, rows["kv"] + 12, "runs", 12.5, "bold", "middle", INK)
    for k, c in [("main", len(l2)), ("l2", len(l1)), ("l1", n)]:
        s.text(rx + 26, rows[k] + ch / 2 + 5, f"{c} / {n}", 13, "bold", "middle", INK)

    # legend
    ly = 500
    s.circle(48, ly, 6.5, ROUTER, "#4c1d95", 1.2); s.text(62, ly + 4, "router boundary (p_{t} ≥ ½)", 12.5)
    s.diamond(278, ly, 8, HEART, "#92400e"); s.text(292, ly + 4, "heartbeat-forced boundary (every 64 / 256 frames)", 12.5)
    s.rect(660, ly - 9, 24, 18, "#ffffff", "#c7c2da", 1.0, 3, dash="3 3")
    s.text(692, ly + 4, "skipped: the level below reuses its cached EMA state z̄", 12.5)
    s.line([(1100, ly + 8), (1100, ly - 8)], MUTED, 1.2, head=MUTED[1:]); s.text(1110, ly + 4, "boundary triggers", 12.5)
    s.rect(40, 528, 1220, 82, "#fffbeb", HEART, 1.2, 8)
    s.text(60, 556, "Main network runs at the event rate; only its KV cache grows with time.", 15, "bold", "start", "#9a3412")
    s.text(60, 580, "Outer stages keep fixed-size sliding-window KV rings (last 64 steps each), so per-frame cost and "
                    "state are constant; the frame-rate path is", 12.5, "normal", "start", INK)
    s.text(60, 598, "frame encoder → stage 1 → heads. The full-sequence forward pass gives the same outputs "
                    "(tests/test_stream.py).", 12.5, "normal", "start", INK)
    save(s, "arch_streaming")


# --------------------------------------------------------------------------------------------------------------
# 3. text H-Net -> video H-Net
# --------------------------------------------------------------------------------------------------------------
def fig_mapping():
    W, H = 1180, 760
    s = Svg(W, H, "Mapping from text H-Net to video H-Net, plus the five video-specific additions.")
    s.text(40, 38, "From text H-Net to video H-Net", 21, "bold")
    s.text(40, 62, "Same recipe (learned dynamic chunking inside the model, recursive stages, main network on the "
                   "coarsest tokens); different unit.", 13, "normal", "start", MUTED)
    c0, c1, c2 = 40, 250, 640
    w1, w2 = 330, 500
    hy = 92
    s.text(c0, hy, "ROLE", 12, "bold", "start", MUTED, ls="0.8")
    s.text(c1 + 12, hy, "TEXT H-NET (Cartesia)", 12, "bold", "start", MUTED, ls="0.8")
    s.text(c2 + 12, hy, "VIDEO H-NET (this repo)    [chunker name in vt/model.py]", 12, "bold", "start", MUTED, ls="0.8")
    rows = [
        ("unit of input", "bytes (UTF-8)", "frames: 64×64 RGB at 10 Hz, one 256-d vector each", None),
        ("fixed tokenizer", "BPE tokens", "fixed-stride chunks, one per N frames", "fixed"),
        ("heuristic chunker", "BLT entropy-threshold patches", "pixel-change keyframes (codec-style)", "external"),
        ("no compression", "byte-level Transformer", "frame-level Transformer", "all"),
        ("learned chunker", "router: cosine of adjacent positions + ratio loss", "same router + ratio loss, or budget-cap mode", "learned"),
        ("outer stages", "Mamba-2 encoder / decoder", "sliding-window attention, W = 64, local RoPE", None),
        ("objective", "next-byte prediction", "next frame given action, ô_{t+1} | h_{t}, a_{t};  plus policy â_{t} from h_{≤t}", None),
    ]
    rh, y = 44, hy + 14
    for i, (role, a, b, tag) in enumerate(rows):
        s.text(c0, y + 27, role, 13, "bold")
        s.rect(c1, y + 5, w1, rh - 10, "#f3f4f6", "#d1d5db", 1.0, 6)
        s.text(c1 + 12, y + 27, a, 12.5)
        s.line([(c1 + w1 + 8, y + rh / 2), (c2 - 8, y + rh / 2)], LINE, 1.6)
        s.rect(c2, y + 5, w2, rh - 10, "#edf4fc", "#bcd3ee", 1.0, 6)
        s.text(c2 + 12, y + 27, b, 12.5)
        if tag:
            tw = 9 + 7.8 * len(tag)
            s.rect(c2 + w2 - tw - 10, y + 12, tw, 20, "#ffffff", "#7ea6d6", 1.0, 4)
            s.text(c2 + w2 - 10 - tw / 2, y + 27, tag, 12, "normal", "middle", "#1d4f91", family=MONO)
        y += rh
    y += 26
    s.text(c0, y, "VIDEO-SPECIFIC ADDITIONS", 12, "bold", "start", ACTION, ls="0.8")
    adds = [
        ("random router init", "W_{q}, W_{k} start random, not at the identity. Adjacent frames are near-identical, so",
         "identity init gives p ≈ 0 and the router never fires."),
        ("heartbeat boundaries", "A boundary is forced every 64 (stage 1) / 256 (stage 2) frames of absolute time:",
         "bounded staleness for the control loop, and no collapse to zero tokens."),
        ("NoPE + Δt embedding", "The main network has no positions; a log₂-bucketed embedding of the gap since the",
         "previous chunk is added instead, so it extrapolates to hours and days."),
        ("action conditioning", "Linear(a_{t−1}) is added to every frame vector and the forward model gets a_{t}, so",
         "boundaries can separate self-caused visual change from external change."),
        ("strict causality", "The router compares h_{t} with h_{t−1} only; windowed attention and dechunking look back",
         "only. The same weights run frame by frame as a streaming policy."),
    ]
    y += 14
    for title, l1, l2 in adds:
        s.rect(c0, y, 1100, 50, "#f0fdf4", "#a7d7bd", 1.0, 6)
        s.text(c0 + 22, y + 30, "+", 18, "bold", "middle", ACTION)
        s.text(c0 + 40, y + 30, title, 13, "bold", "start", INK)
        s.text(c1 + 44, y + 21, l1, 12.5)
        s.text(c1 + 44, y + 39, l2, 12.5)
        y += 58
    save(s, "hnet_mapping")


if __name__ == "__main__":
    fig_pipeline()
    fig_streaming()
    fig_mapping()
