"""Causal, action-conditioned hierarchical video tokenizer (video H-Net).

frames o_t --conv--> x_t (+ emb(a_{t-1}))         one vector per frame ("byte")
  Stage 1: windowed causal encoder -> router -> chunk (~N1 frames / chunk)
    Stage 2: windowed causal encoder -> router -> chunk (~N2 chunks / chunk)
      Main network: full causal attention (NoPE + delta-t embedding) over events
    dechunk (EMA smoothing + causal repeat + STE confidence) + residual + decoder
  dechunk + residual + decoder
heads on h_t: policy a_t, and action-conditioned forward model o_{t+1} | h_t, a_t

Chunker types per stage: 'learned' (H-Net dynamic chunking), 'fixed' (stride N),
'external' (boundaries supplied, e.g. pixel-change heuristic), 'all' (no
compression -> isotropic frame transformer).
Everything is strictly causal so the same weights run as a streaming policy.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.w


def rope(x, pos):  # x (N,H,L,dh), pos (L,) local positions
    half = x.shape[-1] // 2
    freqs = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.float()[:, None] * freqs
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).type_as(x)


class Attention(nn.Module):
    """Causal self-attention. window=None -> full causal NoPE; else sliding window with local RoPE."""

    def __init__(self, d, window=None, dh=64):
        super().__init__()
        self.h, self.dh, self.window = d // dh, dh, window
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, bias_fn=None, block=2048):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        Wn = self.window
        if Wn is None and bias_fn is not None:  # causal attention with an additive (time-recency) bias
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            outs = [F.scaled_dot_product_attention(q[:, :, i0:i0 + block], k[:, :, :i0 + block], v[:, :, :i0 + block],
                                                   attn_mask=bias_fn(i0, min(T, i0 + block)).to(q.dtype))
                    for i0 in range(0, T, block)]
            out = torch.cat(outs, 2).transpose(1, 2)
        elif Wn is None:
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            out = out.transpose(1, 2)
        elif T <= Wn:
            pos = torch.arange(T, device=x.device)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            out = F.scaled_dot_product_attention(rope(q, pos), rope(k, pos), v, is_causal=True).transpose(1, 2)
        else:  # blocked sliding window: each block attends to [prev block, cur block] masked to window
            pad = (-T) % Wn
            if pad:
                q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
            nb = (T + pad) // Wn
            q, k, v = (t.view(B, nb, Wn, self.h, self.dh) for t in (q, k, v))
            kk = torch.cat([F.pad(k, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1], k], 2)
            vv = torch.cat([F.pad(v, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1], v], 2)
            i = torch.arange(Wn, device=x.device)[:, None]
            j = torch.arange(2 * Wn, device=x.device)[None] - Wn
            m = (j <= i) & (j > i - Wn)
            mask = m[None].expand(nb, Wn, 2 * Wn).clone()
            mask[0, :, :Wn] = False
            mask = mask[None, :, None].expand(B, nb, 1, Wn, 2 * Wn).reshape(B * nb, 1, Wn, 2 * Wn)
            qp = torch.arange(Wn, 2 * Wn, device=x.device); kp = torch.arange(2 * Wn, device=x.device)
            q = rope(q.reshape(B * nb, Wn, self.h, self.dh).transpose(1, 2), qp)
            kk = rope(kk.reshape(B * nb, 2 * Wn, self.h, self.dh).transpose(1, 2), kp)
            vv = vv.reshape(B * nb, 2 * Wn, self.h, self.dh).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            out = out.transpose(1, 2).reshape(B, nb * Wn, self.h, self.dh)[:, :T]
        return self.o(out.reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, window=None):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, window)
        hid = int(8 * d / 3) // 64 * 64
        self.w12 = nn.Linear(d, 2 * hid, bias=False)
        self.w3 = nn.Linear(hid, d, bias=False)

    def forward(self, x, bias_fn=None):
        x = x + self.attn(self.n1(x), bias_fn)
        a, b = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(a) * b)


class Stack(nn.Module):
    def __init__(self, d, n, window=None):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, window) for _ in range(n)])
        self.norm = RMSNorm(d)

    def forward(self, x, bias_fn=None):
        for b in self.blocks:
            x = b(x, bias_fn)
        return self.norm(x)


def linear_scan(a, b):
    """h_k = a_k * h_{k-1} + b_k (h_{-1}=0) along dim 1, Hillis-Steele log-depth scan. a (B,L,1), b (B,L,D)."""
    L, off = a.shape[1], 1
    while off < L:
        b = torch.cat([b[:, :off], b[:, off:] + a[:, off:] * b[:, :-off]], 1)
        a = torch.cat([a[:, :off], a[:, off:] * a[:, :-off]], 1)
        off *= 2
    return b


def dt_bucket(dt):  # log2 buckets of time gaps (frames), extrapolates to hours/days
    return torch.clamp(torch.floor(torch.log2(dt.float().clamp(min=1)) * 2), 0, 63).long()


def time_slopes(h, device):
    """ALiBi-style recency slopes per head, in 1/frames: horizons 16 frames .. ~2 h, plus one pure-content head."""
    return torch.tensor([1.0 / (16 * 4 ** i) for i in range(h - 1)] + [0.0], device=device)


def time_bias(tq, tk, slopes, kvalid=None, causal_idx=None):
    """Additive attention bias -slope_h * (t_q - t_k) on real frame times; -inf for the future / invalid keys.
    tq (B,Lq), tk (B,Lk) -> (B,H,Lq,Lk). causal_idx=(i0) masks by index (self-attention), else by time."""
    dt = (tq[:, :, None] - tk[:, None, :]).float()
    bias = -slopes[None, :, None, None] * dt.clamp(min=0)[:, None]
    if causal_idx is not None:
        i = torch.arange(tq.shape[1], device=tq.device)[:, None] + causal_idx
        j = torch.arange(tk.shape[1], device=tq.device)[None]
        bad = (j > i)[None].expand(tq.shape[0], -1, -1)
    else:
        bad = dt < 0
    if kvalid is not None:
        bad = bad | ~kvalid[:, None, :]
    return bias.masked_fill(bad[:, None], float("-inf"))


class MainNet(nn.Module):
    def __init__(self, d, n, alibi=False, pred=False):
        super().__init__()
        self.stack = Stack(d, n, window=None)
        self.alibi = alibi
        self.last = None
        # pred: the main network gets a loss at its own rate, as next-byte loss does at an H-Net boundary on text:
        # predict the next event token (stop-gradient, cosine) from its output. The dense per-frame loss is solved by
        # the local stages; this one needs whatever happened events ago (an item's colour when it leaves the tunnel).
        if pred:
            self.nxt = nn.Sequential(RMSNorm(d), nn.Linear(d, d))
        self.pred_loss = None

    def forward(self, x, meta):
        bias_fn = None
        if self.alibi:
            t, slopes, cache = meta["time"], time_slopes(self.stack.blocks[0].attn.h, x.device), {}

            def bias_fn(i0, i1):  # memoised: all layers share the same bias tensor
                if (i0, i1) not in cache:
                    cache[(i0, i1)] = time_bias(t[:, i0:i1], t[:, :i1], slopes, causal_idx=i0)
                return cache[(i0, i1)]
        out = self.stack(x, bias_fn)
        self.last = (out, meta["time"], meta["valid"], x)  # read by VideoHNet.mem_read (memory read at frame rate)
        if hasattr(self, "nxt") and self.training and x.shape[1] > 1:
            v = meta["valid"][:, 1:] & meta["valid"][:, :-1]
            cos = F.cosine_similarity(self.nxt(out[:, :-1]).float(), x[:, 1:].detach().float(), dim=-1)
            self.pred_loss = ((1 - cos) * v).sum() / v.sum().clamp(min=1)
        return out, []


class Stage(nn.Module):
    def __init__(self, d, d_in, n_enc, n_dec, window, chunker, N, inner, level, heartbeat=0, ratio_mode="target", rate_w=0.0,
                 router="cos", router_detach=False):
        super().__init__()
        self.chunker, self.N, self.level, self.heartbeat = chunker, N, level, heartbeat
        self.router_detach = router_detach
        self.ratio_mode, self.rate_w, self.router = ratio_mode, rate_w, router
        self.enc = Stack(d, n_enc, window)
        self.dec = Stack(d, n_dec, window)
        self.wq = nn.Linear(d, d, bias=False); self.wk = nn.Linear(d, d, bias=False)
        # router="cos": H-Net score p = (1 - cos)/2 with random wq, wk. On video this sits at p~0.5 for any input
        # (independent random projections are ~orthogonal) and the router never learns what a change is.
        # router="tied": H-Net's identity init (cosine measures real change) + a learned threshold/temperature,
        # p = sigmoid((1 - cos - beta) / tau), calibrated on the first batch so that p is sensitive at init.
        if router == "tied":
            with torch.no_grad():
                self.wq.weight.copy_(torch.eye(d)); self.wk.weight.copy_(torch.eye(d))
            self.beta = nn.Parameter(torch.zeros(()))
            self.log_tau = nn.Parameter(torch.zeros(()))
            self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        self.up = nn.Linear(d, d_in); self.down = nn.Linear(d_in, d)
        self.res = nn.Linear(d, d)
        self.dt_emb = nn.Embedding(64, d_in)
        self.inner = inner

    def boundaries(self, h, meta):
        B, T, _ = h.shape
        valid = meta["valid"]
        if self.chunker == "learned":
            # router_detach: the router reads the encoder state but cannot rewrite it. Otherwise the ratio loss can
            # reach the target rate by reshaping the encoder until consecutive states differ at random (seen at L2).
            hf = h.detach().float() if self.router_detach else h.float()
            q = F.normalize(self.wq(hf[:, 1:]), dim=-1); k = F.normalize(self.wk(hf[:, :-1]), dim=-1)
            dist = 1 - (q * k).sum(-1)
            if self.router == "tied":
                if self.training and not bool(self.calibrated):
                    with torch.no_grad():
                        dv = dist[valid[:, 1:]].float()
                        lo, md, hi = torch.quantile(dv, torch.tensor([0.5, 1 - 1 / self.N, 0.95], device=dv.device))
                        self.beta.copy_(md); self.log_tau.copy_(torch.log(((hi - lo) / 4).clamp(min=1e-5)))
                        self.calibrated.fill_(True)
                p = torch.sigmoid((dist - self.beta) / self.log_tau.exp().clamp(min=1e-5))
            else:
                p = (0.5 * dist).clamp(0, 1)
            p = torch.cat([torch.ones(B, 1, device=h.device), p], 1)
            b = p >= 0.5
        else:
            if self.chunker == "fixed":
                b = (meta["idx"] % self.N) == 0
            elif self.chunker == "external":
                b = meta["ext"][self.level].clone(); b[:, 0] = True
            else:  # 'all'
                b = torch.ones(B, T, dtype=torch.bool, device=h.device)
            p = b.float()
        if self.heartbeat and self.chunker in ("learned", "external"):
            # heartbeat: force a boundary whenever (absolute) time crosses a multiple of M frames -> bounded
            # staleness of the event stream for the policy and no collapse to zero tokens; parallel & causal.
            tm = meta["time"] // self.heartbeat
            hb = torch.cat([torch.ones_like(tm[:, :1], dtype=torch.bool), tm[:, 1:] != tm[:, :-1]], 1)
            p = torch.where(hb, torch.maximum(p, torch.full_like(p, 0.5)), p) if self.chunker == "learned" else p
            b = b | hb
        b = b & valid
        return p, b

    def forward(self, x, meta):
        h = self.enc(x)
        p, b = self.boundaries(h, meta)
        B, T, D = h.shape
        counts = b.sum(1)
        L = int(counts.max())
        order = torch.argsort((~b).to(torch.int8), dim=1, stable=True)[:, :L]  # boundary indices in time order
        cvalid = torch.arange(L, device=h.device)[None] < counts[:, None]
        z = torch.gather(h, 1, order[..., None].expand(B, L, D))
        P = torch.gather(p, 1, order)
        times = torch.gather(meta["time"], 1, order)
        dt = torch.diff(times, dim=1, prepend=times[:, :1])
        zi = self.up(z) + self.dt_emb(dt_bucket(dt)).to(z.dtype)
        inner_meta = {"valid": cvalid, "time": times, "idx": torch.arange(L, device=h.device).expand(B, L),
                      "ext": {k: torch.gather(v, 1, order) for k, v in meta["ext"].items()}}
        zo, aux = self.inner(zi, inner_meta)
        zo = self.down(zo)
        # dechunk: EMA smoothing over chunks (gives router a gradient), causal repeat, STE confidence
        Pm = torch.where(cvalid, P, torch.ones_like(P))[..., None]
        zbar = linear_scan((1 - Pm).to(zo.dtype), (Pm.to(zo.dtype) * zo))
        cidx = (torch.cumsum(b.long(), 1) - 1).clamp(min=0)
        up = torch.gather(zbar, 1, cidx[..., None].expand(B, T, D))
        conf = torch.where(b, p, 1 - p)
        ste = conf + (1 - conf).detach()
        y = self.dec(up * ste[..., None].to(up.dtype) + self.res(h))
        valid = meta["valid"]; vf = valid.float()
        Fr = (b.float() * vf).sum() / vf.sum(); G = (p * vf).sum() / vf.sum()
        N = self.N
        ratio = N / (N - 1) * ((N - 1) * Fr * G + (1 - Fr) * (1 - G)) if (self.chunker == "learned" and N > 1) else None
        if ratio is not None and self.ratio_mode == "cap":
            # budget cap: H-Net ratio loss only when over budget (F > 1/N; it then pushes G down), plus a small
            # per-token rate cost. Below the cap, tokens are spent only where prediction loss pays for them.
            ratio = ratio * (Fr > 1 / N).float() + self.rate_w * G
        info = {"level": self.level, "b": b, "p": p, "counts": counts, "valid": valid, "ratio_loss": ratio,
                "order": order, "cvalid": cvalid}
        return y, [info] + aux


class FrameEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        c = [32, 64, 128, 256]
        layers, cin = [], 3
        for co in c:
            layers += [nn.Conv2d(cin, co, 4, 2, 1), nn.GroupNorm(8, co), nn.SiLU()]
            cin = co
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(256 * 16, d)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class FrameDecoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 256 * 16)
        c = [256, 128, 64, 32]
        layers = []
        for ci, co in zip(c, c[1:] + [32]):
            layers += [nn.ConvTranspose2d(ci, co, 4, 2, 1), nn.GroupNorm(8, co), nn.SiLU()]
        layers += [nn.Conv2d(32, 3, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return torch.sigmoid(self.net(self.fc(z).view(-1, 256, 4, 4)))


class MemoryRead(nn.Module):
    """Cross-attention from every frame into the main network's outputs (one per top-level event) with keys
    restricted to events at or before the frame's time and an ALiBi recency bias on elapsed frames."""

    def __init__(self, d, dm, dh=64, src="out"):
        super().__init__()
        self.h, self.dh, self.src = dm // dh, dh, src
        self.n = RMSNorm(d)
        self.q = nn.Linear(d, dm, bias=False)
        # src="tok": each memory entry is the event token itself (main-network input) next to its main output.
        # Main outputs are trained to describe the present, so what an event contained gets overwritten there.
        if src == "tok":
            self.tn = RMSNorm(dm)
        self.kv = nn.Linear(2 * dm if src == "tok" else dm, 2 * dm, bias=False)
        self.o = nn.Linear(dm, d, bias=False)
        nn.init.zeros_(self.o.weight)  # starts as identity: the hierarchy's own output, then learns to read

    def entry(self, out, tok):  # (..., dm) main output and event token -> memory entry fed to kv
        return torch.cat([self.tn(tok).to(out.dtype), out], -1) if self.src == "tok" else out

    def forward(self, h, t, mem, mt, mv, block=4096):
        B, T, _ = h.shape
        L = mem.shape[1]
        q = self.q(self.n(h)).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(mem).view(B, L, 2, self.h, self.dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        slopes = time_slopes(self.h, h.device)
        outs = [F.scaled_dot_product_attention(q[:, :, i0:i0 + block], k, v,
                                               attn_mask=time_bias(t[:, i0:i0 + block], mt, slopes, kvalid=mv).to(q.dtype))
                for i0 in range(0, T, block)]
        return self.o(torch.cat(outs, 2).transpose(1, 2).reshape(B, T, -1))


class VideoHNet(nn.Module):
    def __init__(self, chunkers=("learned", "learned"), Ns=(4, 4), dims=(256, 384, 512), n_outer=2, n_main=8,
                 window=64, use_actions=True, heartbeats=(64, 256), ratio_mode="target", rate_w=0.0,
                 router="cos", alibi=False, mem_read=False, mem_src="out", router_detach=False, frame_io=True,
                 main_pred=False):
        super().__init__()
        self.use_actions, self.mem_read = use_actions, mem_read
        d0 = dims[0]
        if frame_io:  # 64x64 frame encoder/decoder; off when a separate front end supplies latents (vt/model2.py)
            self.fenc = FrameEncoder(d0)
            self.fdec = FrameDecoder(d0)
        self.act_in = nn.Linear(3, d0)
        self.act_fwd = nn.Linear(3, d0)
        net = MainNet(dims[len(chunkers)], n_main, alibi, main_pred)
        self.main = net
        for lvl in reversed(range(len(chunkers))):
            net = Stage(dims[lvl], dims[lvl + 1], n_outer, n_outer, window, chunkers[lvl], Ns[lvl], net, lvl,
                        heartbeats[lvl] if lvl < len(heartbeats) else 0, ratio_mode, rate_w, router, router_detach)
        self.body = net
        if mem_read:  # every frame reads the main network's memory (written only at events), causally by time
            dm = dims[len(chunkers)]
            self.read = MemoryRead(d0, dm, src=mem_src)
        self.policy = nn.Sequential(RMSNorm(d0), nn.Linear(d0, d0), nn.SiLU(), nn.Linear(d0, 3))
        self.fwd = nn.Sequential(RMSNorm(d0), nn.Linear(d0, d0), nn.SiLU(), nn.Linear(d0, d0))

    def encode_frames(self, frames, bs=4096):
        B, T = frames.shape[:2]
        f = frames.reshape(B * T, *frames.shape[2:])
        return torch.cat([self.fenc(f[i:i + bs]) for i in range(0, B * T, bs)]).view(B, T, -1)

    def forward(self, x_frames, prev_act, ext=None, x_lat=None):
        """x_frames (B,T,3,64,64) observations o_0..o_{T-1}; prev_act (B,T,3) = a_{t-1}."""
        x = self.encode_frames(x_frames) if x_lat is None else x_lat
        B, T, _ = x.shape
        if self.use_actions:
            x = x + self.act_in(prev_act).to(x.dtype)
        idx = torch.arange(T, device=x.device).expand(B, T)
        meta = {"valid": torch.ones(B, T, dtype=torch.bool, device=x.device), "time": idx, "idx": idx, "ext": ext or {}}
        h, infos = self.body(x, meta)
        if self.mem_read:
            out, mt, mv, tok = self.main.last
            h = h + self.read(h, idx, self.read.entry(out, tok), mt, mv)
        return h, infos

    def heads(self, h, act):
        """policy a_t from h_t; forward-model latent for o_{t+1} from (h_t, a_t)."""
        pol = self.policy(h)
        z = self.fwd(h + self.act_fwd(act).to(h.dtype))
        return pol, z
