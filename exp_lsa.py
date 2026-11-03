"""Does Latent Sparse Attention work on the video memory? Cached hnet_v5 streams; the frame-rate memory read is replaced
by sparse selections at query frames, and the policy answers the cue->query task (chance 25%)."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from eval import load, encode_stream, dev
from vt.env import simulate, BINS
from vt.lsa import LSAIndex, block_select
from vt.model import time_bias, time_slopes

OUT = "runs_lsa"; os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def cache(name, tag, st, n_extra=256):
    """for each env: main's event tokens + outputs, and h before the read at query frames and random frames"""
    path = f"{OUT}/{tag}_{name}.pt"
    if os.path.exists(path):
        return torch.load(path)
    m, ck = load(name)
    out, rng = [], np.random.default_rng(0)
    for e in range(st["action"].shape[0]):
        x, _ = encode_stream(m, st, e)
        act = torch.as_tensor(st["action"][e], device=dev)[None]
        prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :-1]], 1)
        T = act.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xx = x + m.act_in(prev).to(x.dtype)
            idx = torch.arange(T, device=dev)[None]
            h, _ = m.body(xx, {"valid": torch.ones(1, T, dtype=torch.bool, device=dev), "time": idx, "idx": idx, "ext": {}})
            o, mt, mv, tok = m.main.last
            sel = mv[0]
            entry = m.read.entry(o, tok)[0][sel]
        qt = np.nonzero(st["query_answer"][e] >= 0)[0]
        rt = np.sort(rng.integers(T // 4, T, n_extra))
        fr = torch.as_tensor(np.concatenate([qt, rt]), device=dev)
        out.append(dict(entry=entry.float().cpu(), mt=mt[0][sel].cpu(), h=h[0, fr].float().cpu(), t=fr.cpu(), n_q=len(qt),
                        ans=torch.as_tensor(st["query_answer"][e][qt]), cue_t=torch.as_tensor(st["query_cue_t"][e][qt]),
                        agent=torch.as_tensor(st["agent"][e][fr.cpu().numpy()]), act=torch.as_tensor(st["action"][e][fr.cpu().numpy()])))
        print("cached env", e, "tokens", int(sel.sum()), "queries", len(qt), flush=True)
    torch.save(out, path)
    return out


class Read:
    """read projections of a model, evaluated on an arbitrary key subset per query"""

    def __init__(self, m):
        self.m, self.r = m, m.read
        self.h, self.dh = m.read.h, m.read.dh
        self.slopes = time_slopes(self.h, dev)

    @torch.no_grad()
    def prep(self, env):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            kv = self.r.kv(env["entry"].to(dev)).float()
            q = self.r.q(self.r.n(env["h"].to(dev))).float()
        L, Q = kv.shape[0], q.shape[0]
        k, v = kv.view(L, 2, self.h, self.dh).unbind(1)
        return q.view(Q, self.h, self.dh), k.transpose(0, 1), v.transpose(0, 1), env["mt"].to(dev), env["t"].to(dev)  # k, v (H,L,dh)

    @torch.no_grad()
    def attend(self, qi, ti, k, v, mt, sel=None):
        """one query (H,dh); sel: key indices (None = all before the query time) -> h update, attention (L,) head mean"""
        sel = (mt <= ti).nonzero()[:, 0] if sel is None else sel
        if len(sel) == 0:
            return None, sel, None
        dt = (ti - mt[sel]).float().clamp(min=0)
        lg = (qi[:, None] * k[:, sel]).sum(-1) / self.dh ** 0.5 - self.slopes[:, None] * dt[None]
        a = lg.softmax(-1)
        out = (a[..., None] * v[:, sel]).sum(1).view(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.r.o(out).float(), sel, a.mean(0)

    @torch.no_grad()
    def answer(self, h, up, agent):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pol = self.m.policy((h + up)[None]).float()[0]
        return pol


def score(rd, streams, selector, name=""):
    """selector(env_state, q_concat, t, query_id) -> (key indices, centroids scored). Returns accuracy etc."""
    bins = torch.as_tensor(BINS, device=dev)
    ok, rec, n_sc, n_at, mass, fid = [], [], [], [], [], []
    for env in streams:
        q, k, v, mt, t = rd.prep(env)
        hh = env["h"].to(dev); ag = env["agent"].to(dev)
        state = selector.prepare(env, q, k, v, mt) if selector else None
        for i in range(len(t)):
            ti = t[i]
            uf, allk, af = rd.attend(q[i], ti, k, v, mt)
            if selector:
                sel, sc = selector(state, q[i], k, ti, af, allk, rd, mt)
                us, sel, a = rd.attend(q[i], ti, k, v, mt, sel)
            else:
                us, sel, a, sc = uf, allk, af, len(allk)
            n_sc.append(sc); n_at.append(len(sel))
            if uf is None or us is None:
                continue
            mass.append(af[torch.isin(allk, sel)].sum().item())
            if i < env["n_q"]:
                d = bins - ag[i]; d = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
                pol = rd.answer(hh[i], us, ag[i])
                ok.append(int((d @ pol[:2]).argmax().item() == env["ans"][i].item()))
                c = env["cue_t"][i].item()
                need = ((mt >= c) & (mt < c + 32)).nonzero()[:, 0]
                rec.append(bool(torch.isin(need, sel).any()) if len(need) else False)
            else:
                fid.append(((us - uf) ** 2).sum().sqrt().item() / (uf.norm().item() + 1e-6))
    r = dict(name=name, acc=float(np.mean(ok)), n=len(ok), cue_recall=float(np.mean(rec)), mass=float(np.mean(mass)),
             attended=float(np.mean(n_at)), scored=float(np.mean(n_sc)), rel_err=float(np.mean(fid)))
    print(json.dumps({a: round(b, 3) if isinstance(b, float) else b for a, b in r.items()}), flush=True)
    return r


class Sel:
    """index space + search"""

    def __init__(self, kind, k, w, proj=None, n_cluster=32, n_bucket=4, recency=0.0, newest=False, tree=0, beam=None, n_pool=1, seed=0,
                 nmax=None):
        self.__dict__.update(kind=kind, k=k, w=w, proj=proj, n_cluster=n_cluster, n_bucket=n_bucket, recency=recency, newest=newest,
                             tree=tree, beam=beam, n_pool=n_pool, nmax=nmax)
        self.g = torch.Generator(device=dev).manual_seed(seed)

    def ik(self, k):
        x = k.transpose(0, 1).reshape(k.shape[1], -1)
        return x if self.proj is None else x @ self.proj[1]

    def iq(self, q):
        x = q.view(-1)
        return F.normalize(x if self.proj is None else x @ self.proj[0], dim=-1)

    def prepare(self, env, q, k, v, mt):
        if self.kind in ("lsa",):
            return LSAIndex(self.ik(k), mt, self.n_cluster, self.n_bucket, self.n_pool, self.tree, nmax=self.nmax)
        if self.kind in ("block", "token"):
            return F.normalize(self.ik(k), dim=-1)
        return None

    def __call__(self, st, qi, k, ti, af, allk, rd, mt):
        n = len(allk); w0 = max(0, n - self.w)
        win = torch.arange(w0, n, device=dev)
        if self.kind == "window":   # same budget, nothing long-range
            return torch.arange(max(0, n - self.w - self.k), n, device=dev), 0
        if self.kind == "random":
            if w0 == 0: return win, 0
            return torch.cat([torch.randperm(w0, device=dev, generator=self.g)[: self.k], win]), 0
        if self.kind == "oracle":   # the k keys full attention weighs most
            if w0 == 0: return win, 0
            return torch.cat([af[:w0].topk(min(self.k, w0)).indices, win]), n
        if self.kind == "token":
            if w0 == 0: return win, 0
            return torch.cat([(st[:w0] @ self.iq(qi)).topk(min(self.k, w0)).indices, win]), w0
        if self.kind == "block":
            return block_select(st, mt, self.iq(qi), ti, self.k, self.w, self.n_bucket)
        return st.select(self.iq(qi), ti, self.k, self.w, recency=self.recency, newest=self.newest, beam=self.beam)


def pca_proj(rd, streams, d=64):
    K, Q = [], []
    for env in streams:
        q, k, *_ = rd.prep(env)
        K.append(k.transpose(0, 1).reshape(k.shape[1], -1)); Q.append(q.view(q.shape[0], -1))
    pk = torch.pca_lowrank(torch.cat(K), q=d, center=False, niter=4)[2]
    pq = torch.pca_lowrank(torch.cat(Q), q=d, center=False, niter=4)[2]
    return pq, pk


def distill(rd, streams, d=64, steps=1500, bias=True, query_frac=0.0, lr=3e-3, seed=0):
    """forward KL to the head-mean attention (frozen model), scoring every key: D-LSA eq. 2. query_frac: share of training
    queries taken at the rare query frames instead of uniformly."""
    torch.manual_seed(seed)
    pq, pk = pca_proj(rd, streams, d)
    wq, wk = pq.clone().requires_grad_(), pk.clone().requires_grad_()
    tau = torch.tensor(0.1, device=dev).log().requires_grad_()
    opt = torch.optim.Adam([wq, wk, tau], lr=lr)
    data = [rd.prep(e) for e in streams]
    for s in range(steps):
        tl = 0.0
        e = int(torch.randint(len(data), ()))
        q, k, v, mt, t = data[e]
        nq = streams[e]["n_q"]
        with torch.no_grad():
            pick = torch.randint(nq, q.shape[0], (32,), device=dev)
            if query_frac > 0 and nq:
                rare = torch.rand(32, device=dev) < query_frac
                pick = torch.where(rare, torch.randint(0, nq, (32,), device=dev), pick)
            qp, tp = q[pick], t[pick]
            lg = (qp[:, :, None] * k[None]).sum(-1) / rd.dh ** 0.5
            mask = mt[None] > tp[:, None]
            dt = (tp[:, None] - mt[None]).float().clamp(min=0)
            if bias:
                lg = lg - rd.slopes[None, :, None] * dt[:, None]
            lg = lg.masked_fill(mask[:, None], float("-inf"))
            A = lg.softmax(-1).mean(1)
        ik = F.normalize(k.transpose(0, 1).reshape(k.shape[1], -1) @ wk, dim=-1)
        iq = F.normalize(qp.view(32, -1) @ wq, dim=-1)
        ls = (iq @ ik.T) / tau.exp()
        ls = ls.masked_fill(mask, float("-inf")).log_softmax(-1)
        loss = -(A * ls.masked_fill(mask, 0.0)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 300 == 0:
            print("distill", s, round(loss.item(), 3), flush=True)
    return wq.detach(), wk.detach()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hnet_v5")
    ap.add_argument("--part", default="budget")
    ap.add_argument("--n_env", type=int, default=12)
    a = ap.parse_args()
    # held-out test streams: gaps the model answers with full attention, plus a stream of stale cues in memory
    te = simulate(a.n_env, 36000, seed=8889, gap=(1500, 6000), cue_rate=1 / 300)
    tr = simulate(4, 36000, seed=4242, gap=(1500, 6000), cue_rate=1 / 300)
    test, train = cache(a.name, "te", te), cache(a.name, "tr", tr)
    del te, tr
    m, ck = load(a.name)
    rd = Read(m)
    if a.part == "native":  # a model trained with a sparse read: dense vs sparse at test time, short and long memory
        pool = test + train
        test = [dict(e, h=e["h"][:e["n_q"] + 64], t=e["t"][:e["n_q"] + 64], agent=e["agent"][:e["n_q"] + 64]) for e in test]
        res = []
        for Hh in (0, 7, 15):
            hist = []
            for i, env in enumerate(test):
                others = [pool[(i + 1 + j) % len(pool)] for j in range(Hh)]
                off = 36000 * Hh
                hist.append(dict(env, entry=torch.cat([o["entry"] for o in others] + [env["entry"]]),
                                 mt=torch.cat([o["mt"] + 36000 * j for j, o in enumerate(others)] + [env["mt"] + off]),
                                 t=env["t"] + off, cue_t=env["cue_t"] + off))
            print(f"== history {Hh} h", flush=True)
            for nm, sel in [("full", None), ("lsa k32", Sel("lsa", 32, 32, nmax=8)), ("lsa k8", Sel("lsa", 8, 32, nmax=8)),
                            ("lsa k32 rec0.05", Sel("lsa", 32, 32, nmax=8, recency=0.05)), ("lsa k8 rec0.05", Sel("lsa", 8, 32, nmax=8, recency=0.05))]:
                r = score(rd, hist, sel, nm); r["hours"] = Hh; res.append(r)
        json.dump(res, open(f"{OUT}/native_{a.name}.json", "w"), indent=1)
        raise SystemExit
    if a.part == "stale2":  # tree search with recency, tight budgets, and what content-only selection picks
        pool = test + train
        nf = lambda e: e["n_q"]
        test = [dict(e, h=e["h"][:nf(e)], t=e["t"][:nf(e)], agent=e["agent"][:nf(e)], act=e["act"][:nf(e)]) for e in test]
        res = []
        for Hh in (0, 7, 15):
            hist = []
            for i, env in enumerate(test):
                others = [pool[(i + 1 + j) % len(pool)] for j in range(Hh)]
                off = 36000 * Hh
                cues = [o["cue_t"] + 36000 * j for j, o in enumerate(others)] + [env["cue_t"] + off]
                hist.append(dict(env, entry=torch.cat([o["entry"] for o in others] + [env["entry"]]),
                                 mt=torch.cat([o["mt"] + 36000 * j for j, o in enumerate(others)] + [env["mt"] + off]),
                                 t=env["t"] + off, cue_t=env["cue_t"] + off, all_cues=torch.cat(cues)))
            print(f"== history {Hh} h", flush=True)
            for nm, sel in [("lsa-raw k8 rec0.05", Sel("lsa", 8, 32, recency=0.05)), ("lsa-raw k16 rec0.05", Sel("lsa", 16, 32, recency=0.05)),
                            ("tree k32 rec0.05", Sel("lsa", 32, 32, recency=0.05, tree=8, beam=16)),
                            ("tree k32 rec0.05 beam4", Sel("lsa", 32, 32, recency=0.05, tree=8, beam=4)),
                            ("lsa-raw k32", Sel("lsa", 32, 32)), ("lsa-raw k32 rec0.05", Sel("lsa", 32, 32, recency=0.05))]:
                r = score(rd, hist, sel, nm); r["hours"] = Hh
                # where do the long-range picks go: tokens right after the current cue, after an older cue, or elsewhere
                cur = old_ = tot = 0
                for env in hist:
                    q, k, v, mt, t = rd.prep(env)
                    stt = sel.prepare(env, q, k, v, mt)
                    for i in range(env["n_q"]):
                        allk = (mt <= t[i]).nonzero()[:, 0]
                        idx, _ = sel(stt, q[i], k, t[i], None, allk, rd, mt)
                        lr = idx[:-32] if len(idx) > 32 else idx[:0]
                        tt = mt[lr]
                        c = env["cue_t"][i].item()
                        near_c = (tt >= c) & (tt < c + 32)
                        ac = env["all_cues"].to(dev); ac = ac[ac < c]
                        near_o = ((tt[:, None] >= ac[None]) & (tt[:, None] < ac[None] + 32)).any(1) if len(ac) else torch.zeros_like(near_c)
                        cur += near_c.sum().item(); old_ += (near_o & ~near_c).sum().item(); tot += len(lr)
                r.update(pick_cur_cue=cur / max(1, tot), pick_old_cue=old_ / max(1, tot))
                print("   picks: current cue", round(r["pick_cur_cue"], 3), "older cues", round(r["pick_old_cue"], 3), flush=True)
                res.append(r)
        json.dump(res, open(f"{OUT}/stale2_{a.name}.json", "w"), indent=1)
        raise SystemExit
    if a.part == "stale":  # memory grows: H earlier hours (other streams, shifted back in time) hold stale cues
        pool = test + train
        nf = lambda e: e["n_q"] + 64     # queries + 64 random frames (fidelity)
        test = [dict(e, h=e["h"][:nf(e)], t=e["t"][:nf(e)], agent=e["agent"][:nf(e)], act=e["act"][:nf(e)]) for e in test]
        res = []
        for Hh in (0, 1, 3, 7, 15):
            hist = []
            for i, env in enumerate(test):
                others = [pool[(i + 1 + j) % len(pool)] for j in range(Hh)]
                off = 36000 * Hh
                ent = torch.cat([o["entry"] for o in others] + [env["entry"]])
                mt = torch.cat([o["mt"] + 36000 * j for j, o in enumerate(others)] + [env["mt"] + off])
                hist.append(dict(env, entry=ent, mt=mt, t=env["t"] + off, cue_t=env["cue_t"] + off))
            n_cue = np.mean([sum(o["n_q"] for o in pool[:Hh])]) if Hh else 0
            print(f"== history {Hh} h, tokens {np.mean([len(e['mt']) for e in hist]):.0f}, stale cues ~{n_cue:.0f}", flush=True)
            W = 32
            for nm, sel in [("full", None), ("window k32", Sel("window", 32, W)), ("block-raw k32", Sel("block", 32, W)),
                            ("lsa-raw k32", Sel("lsa", 32, W)), ("lsa-raw k64", Sel("lsa", 64, W)),
                            ("lsa-raw k32 newest", Sel("lsa", 32, W, newest=True)),
                            ("lsa-raw k32 rec0.02", Sel("lsa", 32, W, recency=0.02)), ("lsa-raw k32 rec0.05", Sel("lsa", 32, W, recency=0.05)),
                            ("lsa-raw k32 rec0.1", Sel("lsa", 32, W, recency=0.1)), ("lsa-raw k32 tree", Sel("lsa", 32, W, tree=8, beam=16))]:
                r = score(rd, hist, sel, nm); r["hours"] = Hh; res.append(r)
        json.dump(res, open(f"{OUT}/stale_{a.name}.json", "w"), indent=1)
        raise SystemExit
    print("tokens/stream", np.mean([len(e["mt"]) for e in test]), "queries", sum(e["n_q"] for e in test), flush=True)
    test = [dict(e, h=e["h"][:e["n_q"] + 64], t=e["t"][:e["n_q"] + 64], agent=e["agent"][:e["n_q"] + 64]) for e in test]
    res = [score(rd, test, None, "full")]
    pca = pca_proj(rd, train)
    d_u = distill(rd, train)
    d_q = distill(rd, train, query_frac=0.5)
    W = 32
    for k in (8, 16, 32, 64, 128):
        res += [score(rd, test, Sel("window", k, W), f"window k{k}"), score(rd, test, Sel("random", k, W), f"random k{k}"),
                score(rd, test, Sel("oracle", k, W), f"oracle k{k}"), score(rd, test, Sel("token", k, W, d_u), f"token-dist k{k}"),
                score(rd, test, Sel("block", k, W, d_u), f"block-dist k{k}"),
                score(rd, test, Sel("lsa", k, W, pca), f"lsa-pca k{k}"), score(rd, test, Sel("lsa", k, W, d_u), f"lsa-dist k{k}"),
                score(rd, test, Sel("lsa", k, W, d_q), f"lsa-distQ k{k}"), score(rd, test, Sel("lsa", k, W, None), f"lsa-raw k{k}")]
    json.dump(res, open(f"{OUT}/budget_{a.name}.json", "w"), indent=1)
