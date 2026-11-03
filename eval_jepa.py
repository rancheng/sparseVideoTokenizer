"""Where does each objective spend the sparse budget, and does the main network matter for latent prediction?

  budget   share of events and of high-resolution cells landing on each entity's cell (arm, item on the belt or held,
           rule symbol while shown, noisy TV), and each entity's own event / high-res rate while present
  latent   JEPA loss per horizon on held-out clips, with the main network's output zeroed (JEPA runs only)
  collapse per-dimension std of the targets across cells and frames, and the effective rank of the frame vectors
usage: python eval_jepa.py <run> [<run> ...] [--rule_mode pair --pop_exit]
"""
import argparse, json, math
import numpy as np
import torch
import torch.nn.functional as F
from eval2 import load, run_front, dev
from vt.env2 import simulate, G, CELL, BELT_Y, RULE_C, ClipSampler
from vt.jepa import JEPAHeads, HORIZONS


def entity_cells(st, e):
    """per frame, the grid cell (j) of each entity, -1 when absent"""
    N = st["agent"].shape[1]
    cj = lambda x, y: (np.clip((y // CELL).astype(int), 0, G - 1) * G + np.clip((x // CELL).astype(int), 0, G - 1))
    ag = st["agent"][e]
    arm = cj(ag[:, 0], ag[:, 1])
    item = np.where(st["held"][e], cj(ag[:, 0], ag[:, 1] + 8), np.where(st["belt_on"][e], cj(st["belt_x"][e], np.full(N, BELT_Y)), -1))
    rule = np.where(st["rule_on"][e], RULE_C[1] * G + RULE_C[0], -1)
    tv = np.full(N, st["tv_cell"][e][1] * G + st["tv_cell"][e][0])
    return {"arm": arm, "item": item, "rule": rule, "tv": tv}


@torch.no_grad()
def budget(m, st):
    tot = {k: np.zeros(4) for k in ["arm", "item", "rule", "tv"]}  # events on, hires on, frames present, (unused)
    n_ev = n_hr = 0
    for e in range(st["action"].shape[0]):
        f, b, use, cell = run_front(m, st, e)
        b, use = b.cpu().numpy(), use.cpu().numpy()
        n_ev += b.sum(); n_hr += use.sum()
        for k, jj in entity_cells(st, e).items():
            ok = jj >= 0; t = np.nonzero(ok)[0]
            tot[k][0] += b[t, jj[t]].sum(); tot[k][1] += use[t, jj[t]].sum(); tot[k][2] += len(t)
    out = {}
    for k, v in tot.items():
        out[k] = dict(event_share=v[0] / n_ev, hires_share=v[1] / n_hr, event_rate=v[0] / max(1, v[2]), hires_rate=v[1] / max(1, v[2]))
    out["other"] = dict(event_share=1 - sum(o["event_share"] for o in out.values()), hires_share=1 - sum(o["hires_share"] for o in out.values()))
    return out


@torch.no_grad()
def latent(m, ck, jm, sampler, n_batches=8):
    res = {f"j{k}": [] for k in HORIZONS}; res.update({f"j{k}_nomain": [] for k in HORIZONS})
    main = m.temporal.main; orig = main.forward
    for nb in range(n_batches):
        frames, clean, act, ev = sampler.sample(2)
        T = frames.shape[1] - 1
        prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :T - 1]], 1)
        for ablate in (False, True):
            if ablate:
                def dead(x, meta):
                    out, aux = orig(x, meta); main.last = (out * 0, *main.last[1:3], x * 0); return out * 0, aux
                main.forward = dead
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, fi, _ = m.front(frames[:, :T])
                h, _ = m.temporal(None, prev, x_lat=x)
            main.forward = orig
            g = torch.Generator(device=dev).manual_seed(nb)
            tf = torch.randint(64, T + 1, (2, 32), device=dev, generator=g)
            bix = torch.arange(2, device=dev)[:, None].expand_as(tf)
            Y = jm.targets(frames[bix, tf]); A = act[:, :T]
            S = torch.cat([torch.zeros_like(A[:, :1]), A.cumsum(1)], 1)
            for hi, k in enumerate(HORIZONS):
                t = tf - k; n = tf.numel(); J = G * G
                af = torch.cat([S[bix, tf] - S[bix, t], A[bix, t]], -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = jm.predict(h[bix, t].reshape(n, 1, -1).expand(-1, J, -1).reshape(n * J, -1), torch.arange(J, device=dev).repeat(n),
                                    torch.full((n * J,), hi, device=dev), af.reshape(n, 1, -1).expand(-1, J, -1).reshape(n * J, -1).to(torch.bfloat16),
                                    fi["cell"][bix, t].reshape(n * J, -1) if k == 1 else None)
                l = F.smooth_l1_loss(pr.float(), Y.reshape(n * J, -1)).item()
                res[f"j{k}{'_nomain' if ablate else ''}"].append(l)
            if not ablate:
                ystd = Y.float().std(dim=(0, 1, 2)).mean().item()
                xs = x.float().reshape(-1, x.shape[-1]); xs = xs - xs.mean(0)
                sv = torch.linalg.svdvals(xs); pk = sv / sv.sum(); erank = math.exp(-(pk * pk.clamp(min=1e-12).log()).sum().item())
                res.setdefault("y_std", []).append(ystd); res.setdefault("erank_f", []).append(erank)
    return {k: float(np.mean(v)) for k, v in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--rule_mode", default="pair")
    ap.add_argument("--pop_exit", action="store_true")
    a = ap.parse_args()
    st = simulate(4, 12000, seed=9999, rule_mode=a.rule_mode, pop_exit=a.pop_exit)
    val = ClipSampler(simulate(8, 6000, seed=7777, rule_mode=a.rule_mode, pop_exit=a.pop_exit), 2048, seed=3)
    for name in a.names:
        m, ck = load(name)
        r = {"budget": budget(m, st)}
        if ck.get("jepa"):
            jm = JEPAHeads(m.front, ck["cfg"]["temporal"]["dims"][0], 3).to(dev).eval()
            jm.load_state_dict(ck["jepa"])
            r["latent"] = latent(m, ck, jm, val)
        print(name, json.dumps(r, default=lambda o: round(float(o), 4)), flush=True)
        json.dump(r, open(f"runs2/{name}/eval_jepa.json", "w"), default=float)
