"""Evaluate sparse spatio-temporal tokenizers on held-out hour-long workcell streams (one causal pass per hour).

  spatial    cell events per frame (idle / active), recall of relevant cell changes, event rate on the noisy TV,
             fovea hit rate on decision-relevant cells (held glyph at the decision, rule symbol while shown)
  temporal   tokens per minute at each level, recall of primitive boundaries (L1) and task starts (L2) vs random
  task       sorting decision accuracy by age of the rule flash (needs fovea + memory)
  permanence next-cell prediction at the tunnel exit vs copy-last (needs the main network when the delay is long)
  necessity  the same metrics with the main network's contribution removed (--ablate_main)
  compute    front-end FLOPs per frame from the event counts, vs the dense patch model
"""
import argparse, json, math, os
import numpy as np
import torch
from vt.env2 import simulate, Renderer, flat, G, CELL, BINS, BELT_Y, CARRY, EV_EXIT, EV_REACH, RULE_C
from vt.model2 import SparseVideoModel, to_cells
from vt.stream2 import StreamingSparseModel
from eval import frame_masks, near

dev = "cuda"


def load(name):
    ck = torch.load(f"runs2/{name}/ckpt.pt", map_location=dev)
    m = SparseVideoModel(**ck["cfg"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck


@torch.no_grad()
def run_front(m, st, e, chunk=2048):
    N = st["action"].shape[1]
    r = Renderer(dev); gen = torch.Generator(device=dev).manual_seed(4321 + e)
    fs, bs, us, cells, state = [], [], [], [], None
    for t0 in range(0, N, chunk):
        fr, _ = r(flat(st, e, t0, min(N, t0 + chunk)), gen=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f, fi, state = m.front(fr[None], t0=t0, state=state)
        fs.append(f); bs.append(fi["b"]); us.append(fi["use"]); cells.append(fi["cell"].to(torch.bfloat16))
    return torch.cat(fs, 1), torch.cat(bs, 1)[0], torch.cat(us, 1)[0], torch.cat(cells, 1)[0]


@torch.no_grad()
def eval_stream(m, st, e, ablate=False):
    N = st["action"].shape[1]
    f, b, use, cell = run_front(m, st, e)
    act = torch.as_tensor(st["action"][e], device=dev)[None]
    prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)
    main = m.temporal.main
    orig = main.forward
    if ablate:
        def dead(x, meta):
            out, aux = orig(x, meta); main.last = (out * 0, *main.last[1:3], x * 0); return out * 0, aux
        main.forward = dead
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, infos = m.temporal(None, prev, x_lat=f)
        pol, z = m.temporal.heads(h, act)
        hd = m.heading(h)[0].float().cpu().numpy() if hasattr(m, "ang") else None
    main.forward = orig
    res = {}
    ph, ev = st["phase"][e], st["events"][e]
    pol_np = pol[0].float().cpu().numpy()
    # ---- task: decisions by rule age ----
    dec = np.nonzero(st["decision"][e])[0]
    recs = []
    for t in dec:
        dirs = BINS - st["agent"][e, t]; dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6
        pick, gl = int(np.argmax(dirs @ (hd[t] if hd is not None else pol_np[t, :2]))), int(st["item_gly"][e, t])
        recs.append({"age": int(t - st["rule_t"][e, t]), "ok": int(pick == st["bin"][e, t]), "glyph_ok": int(pick in (gl, 3 - gl)),
                     "side_ok": int(pick // 2 == st["bin"][e, t] // 2), "within_ok": int(pick % 2 == st["bin"][e, t] % 2),
                     "expert_ok": int(np.argmax(dirs @ st["action"][e, t, :2]) == st["bin"][e, t])})
    res["decisions"] = recs
    res["act_mse_moving"] = float(((pol_np[:, :2] - st["action"][e, :, :2]) ** 2)[np.abs(st["action"][e, :, :2]).sum(1) > 0.2].mean())
    if ablate:
        return res
    r = Renderer(dev)
    # ---- spatial ----
    bn = b.cpu().numpy(); un = use.cpu().numpy()
    busy = near((ph != 0) | (ev > 0), np.arange(N), tol=5)
    res["events_per_frame"] = float(bn.sum(1).mean()); res["hires_per_frame"] = float(un.sum(1).mean())
    res["events_per_frame_idle"] = float(bn[~busy].sum(1).mean()); res["events_per_frame_active"] = float(bn[busy].sum(1).mean())
    tvj = st["tv_cell"][e, 1] * G + st["tv_cell"][e, 0]
    res["tv_event_rate"] = float(bn[:, tvj].mean())
    # relevant changes: cells whose clean content changes, excluding the TV cell and light-toggle frames
    chg = np.zeros((N, G * G), bool)
    for s0 in range(0, N - 1, 4096):
        s1 = min(N, s0 + 4097)
        _, c = r(flat(st, e, s0, s1), noise=0)
        cc = to_cells(c[None])[0]
        chg[s0 + 1:s1] = ((cc[1:] - cc[:-1]).abs().mean((-1, -2, -3)) > 0.01).cpu().numpy()
    light = np.zeros(N, bool); light[1:] = st["light"][e, 1:] != st["light"][e, :-1]
    chg[light] = False; chg[:, tvj] = False
    # an event at t or t+1 covers a change at t (the router compares t with t-1)
    cov = bn.copy(); cov[:-1] |= bn[1:]
    res["relevant_change_recall"] = float(cov[chg].mean()) if chg.any() else float("nan")
    res["event_precision_relevant"] = float((chg | np.roll(chg, -1, 0))[bn].mean())
    # fovea: the item's cells at any frame from its tunnel exit to the decision (on the belt, then in the gripper),
    # and the rule cell while the rule symbol is shown
    def cell_of(x, y):
        return int(min(G - 1, max(0, y // CELL))) * G + int(min(G - 1, max(0, x // CELL)))
    exits = np.nonzero(ev & EV_EXIT)[0]
    hits = []
    for t in dec:
        t_ex = exits[exits <= t]
        t0_ = t_ex[-1] if len(t_ex) else max(0, t - 30)
        hit = False
        for u in range(t0_, t + 1):
            if st["belt_on"][e, u]:
                js = {cell_of(st["belt_x"][e, u] + dx, BELT_Y) for dx in (-4, 4)}
            else:
                x, y = st["agent"][e, u]; js = {cell_of(x + dx, y + 8) for dx in (-4, 4)}
            hit |= bool(un[u, list(js)].any())
        hits.append(hit)
    rj = RULE_C[1] * G + RULE_C[0]
    on = np.nonzero(st["rule_on"][e])[0]
    starts = on[np.r_[True, np.diff(on) > 1]] if len(on) else []
    rule_hits = [un[s:s + 5, rj].any() for s in starts]
    res["fovea_hit_glyph"] = float(np.mean(hits)) if hits else float("nan")
    res["fovea_hit_rule"] = float(np.mean(rule_hits)) if len(rule_hits) else float("nan")
    res["rule_event_on_flash"] = float(np.mean([bn[s:s + 2, rj].any() for s in starts])) if len(starts) else float("nan")
    res["hires_share_tv"] = float(un[:, tvj].sum() / max(1, un.sum()))
    # ---- temporal levels ----
    masks = frame_masks(infos, N)
    prim = np.nonzero(np.diff(ph) != 0)[0] + 1
    task = np.nonzero(ev & EV_REACH)[0]
    res["levels"] = []
    for l, mk in enumerate(masks):
        mn = mk.cpu().numpy(); rate = mn.mean()
        res["levels"].append({"frames_per_token": N / mn.sum(), "tok_per_min_idle": float(mn[~busy].mean() * 600),
                              "tok_per_min_active": float(mn[busy].mean() * 600),
                              "recall_primitive": float(near(mn, prim, 3).mean()), "recall_task": float(near(mn, task, 5).mean()),
                              "random_recall_3": float(1 - (1 - rate) ** 7), "random_recall_5": float(1 - (1 - rate) ** 11)})
    # ---- object permanence: next-cell prediction on the exit cells right after a tunnel exit ----
    exits = np.nonzero(ev & EV_EXIT)[0]
    sel = [(t + k, j) for t in exits for k in range(0, 8) for j in (G * 0 + 6, G * 0 + 7) if t + k + 1 < N]
    if sel:
        tt = torch.tensor([s[0] for s in sel], device=dev); jj = torch.tensor([s[1] for s in sel], device=dev)
        st_sub = flat(st, e, 0, N)
        o_now, c_now = r({k: v[tt.cpu().numpy()] if k != "tv_cell" else v[:len(tt)] for k, v in st_sub.items()})
        _, c_nxt = r({k: v[(tt + 1).cpu().numpy()] if k != "tv_cell" else v[:len(tt)] for k, v in st_sub.items()}, noise=0)
        cn = to_cells(c_nxt[:, None])[:, 0][torch.arange(len(tt)), jj]; cc = to_cells(c_now[:, None])[:, 0][torch.arange(len(tt)), jj]
        oc = to_cells(o_now[:, None])[:, 0][torch.arange(len(tt)), jj]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = m.cdec(z[0, tt], cell[tt, jj], jj, oc).float()
        mse = ((pred - cn) ** 2).mean().item(); cp = ((cc - cn) ** 2).mean().item()
        res["exit_psnr"] = 10 * math.log10(1 / mse); res["exit_psnr_copy"] = 10 * math.log10(1 / max(cp, 1e-10))
        res["exit_psnr_copy_noisy"] = 10 * math.log10(1 / max(((oc - cn) ** 2).mean().item(), 1e-10))  # what the model sees
        # the exit frame itself (k = 0): with --pop_exit only memory of the entry can predict it
        k0 = torch.tensor([s[0] in set(exits.tolist()) for s in sel], device=dev)
        res["exit0_psnr"] = 10 * math.log10(1 / ((pred - cn)[k0] ** 2).mean().item())
        res["exit0_psnr_copy"] = 10 * math.log10(1 / max(((cc - cn)[k0] ** 2).mean().item(), 1e-10))
        res["_exit_sel"] = (tt, jj, cn, oc, k0)
    # ---- compute (front end), from the event counts ----
    eng = StreamingSparseModel(m); c = eng.cost; J = G * G
    per = J * (c["cheap"] + c["router"]) + res["events_per_frame"] * (c["fovea"] + sum(c["block"]) + c["pool_row"]) + res["hires_per_frame"] * c["hires"] + c["pool"]
    res["front_mflops_per_frame"] = per / 1e6; res["dense_front_mflops_per_frame"] = eng.dense_flops_per_frame() / 1e6
    return res


@torch.no_grad()
def exit_psnr_ablated(m, st, e, sel):
    """Exit-cell prediction with the main network removed (same cells as the full run)."""
    N = st["action"].shape[1]
    f, b, use, cell = run_front(m, st, e)
    act = torch.as_tensor(st["action"][e], device=dev)[None]
    prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)
    main = m.temporal.main; orig = main.forward
    def dead(x, meta):
        out, aux = orig(x, meta); main.last = (out * 0, *main.last[1:3], x * 0); return out * 0, aux
    main.forward = dead
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, _ = m.temporal(None, prev, x_lat=f); _, z = m.temporal.heads(h, act)
        tt, jj, cn, oc, k0 = sel
        pred = m.cdec(z[0, tt], cell[tt, jj], jj, oc).float()
    main.forward = orig
    return 10 * math.log10(1 / ((pred - cn) ** 2).mean().item()), 10 * math.log10(1 / ((pred - cn)[k0] ** 2).mean().item())


def summarize(rs, abl):
    out = {}
    for k in ["events_per_frame", "events_per_frame_idle", "events_per_frame_active", "hires_per_frame", "tv_event_rate",
              "relevant_change_recall", "event_precision_relevant", "fovea_hit_glyph", "fovea_hit_rule", "rule_event_on_flash", "hires_share_tv", "exit_psnr", "exit0_psnr", "exit0_psnr_copy", "exit0_psnr_no_main",
              "exit_psnr_copy", "exit_psnr_copy_noisy", "exit_psnr_no_main", "act_mse_moving", "front_mflops_per_frame", "dense_front_mflops_per_frame"]:
        v = [r[k] for r in rs if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(np.mean(v)) if v else None
    out["levels"] = [{k: float(np.mean([r["levels"][l][k] for r in rs])) for k in rs[0]["levels"][l]} for l in range(len(rs[0]["levels"]))]
    for tag, R in [("decision", rs), ("decision_no_main", abl)]:
        d = [x for r in R for x in r["decisions"]]
        out[tag] = {f"{lo}-{hi}": {"acc": float(np.mean([x["ok"] for x in d if lo <= x["age"] < hi])) if any(lo <= x["age"] < hi for x in d) else None,
                                   "n": sum(lo <= x["age"] < hi for x in d)} for lo, hi in [(0, 600), (600, 1500), (1500, 3000), (3000, 10 ** 9)]}
        out[tag]["all"] = float(np.mean([x["ok"] for x in d])) if d else None
        out[tag]["glyph_read"] = float(np.mean([x["glyph_ok"] for x in d])) if d else None  # chance 0.5
        out[tag]["side_ok"] = float(np.mean([x["side_ok"] for x in d])) if d else None      # pair mode: rule part
        out[tag]["within_ok"] = float(np.mean([x["within_ok"] for x in d])) if d else None  # pair mode: glyph part
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--n_env", type=int, default=4)
    ap.add_argument("--delay", default="150,600")
    ap.add_argument("--tag", default="hour")
    ap.add_argument("--pop_exit", action="store_true")
    ap.add_argument("--rule_mode", default="mirror")
    ap.add_argument("--N", type=int, default=36000, help="stream length (frames)")
    a = ap.parse_args()
    st = simulate(a.n_env, a.N, seed=9999, delay=tuple(int(v) for v in a.delay.split(",")), pop_exit=a.pop_exit, rule_mode=a.rule_mode)
    for name in a.names:
        m, ck = load(name)
        rs, abl = [], []
        for e in range(a.n_env):
            r = eval_stream(m, st, e)
            if "_exit_sel" in r:
                r["exit_psnr_no_main"], r["exit0_psnr_no_main"] = exit_psnr_ablated(m, st, e, r.pop("_exit_sel"))
            rs.append(r); abl.append(eval_stream(m, st, e, ablate=True))
            torch.cuda.empty_cache()
        s = summarize(rs, abl)
        json.dump({"summary": s, "per_stream": rs}, open(f"runs2/{name}/eval_{a.tag}.json", "w"), default=float)
        print(name, json.dumps(s, default=float), flush=True)
