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

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        Wn = self.window
        if Wn is None:
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

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        a, b = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(a) * b)


class Stack(nn.Module):
    def __init__(self, d, n, window=None):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, window) for _ in range(n)])
        self.norm = RMSNorm(d)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
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


class MainNet(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.stack = Stack(d, n, window=None)

    def forward(self, x, meta):
        return self.stack(x), []


class Stage(nn.Module):
    def __init__(self, d, d_in, n_enc, n_dec, window, chunker, N, inner, level, heartbeat=0, ratio_mode="target", rate_w=0.0):
        super().__init__()
        self.chunker, self.N, self.level, self.heartbeat = chunker, N, level, heartbeat
        self.ratio_mode, self.rate_w = ratio_mode, rate_w
        self.enc = Stack(d, n_enc, window)
        self.dec = Stack(d, n_dec, window)
        self.wq = nn.Linear(d, d, bias=False); self.wk = nn.Linear(d, d, bias=False)
        # NOTE: H-Net inits wq=wk=I. Adjacent video frames are near-identical, so that init gives p~0 (no
        # boundaries) and the router never wakes up. Random init starts at p~0.5 and the ratio loss compresses.
        self.up = nn.Linear(d, d_in); self.down = nn.Linear(d_in, d)
        self.res = nn.Linear(d, d)
        self.dt_emb = nn.Embedding(64, d_in)
        self.inner = inner

    def boundaries(self, h, meta):
        B, T, _ = h.shape
        valid = meta["valid"]
        if self.chunker == "learned":
            hf = h.float()
            q = F.normalize(self.wq(hf[:, 1:]), dim=-1); k = F.normalize(self.wk(hf[:, :-1]), dim=-1)
            p = (0.5 * (1 - (q * k).sum(-1))).clamp(0, 1)
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


class VideoHNet(nn.Module):
    def __init__(self, chunkers=("learned", "learned"), Ns=(4, 4), dims=(256, 384, 512), n_outer=2, n_main=8,
                 window=64, use_actions=True, heartbeats=(64, 256), ratio_mode="target", rate_w=0.0):
        super().__init__()
        self.use_actions = use_actions
        d0 = dims[0]
        self.fenc = FrameEncoder(d0)
        self.fdec = FrameDecoder(d0)
        self.act_in = nn.Linear(3, d0)
        self.act_fwd = nn.Linear(3, d0)
        net = MainNet(dims[len(chunkers)], n_main)
        for lvl in reversed(range(len(chunkers))):
            net = Stage(dims[lvl], dims[lvl + 1], n_outer, n_outer, window, chunkers[lvl], Ns[lvl], net, lvl,
                        heartbeats[lvl] if lvl < len(heartbeats) else 0, ratio_mode, rate_w)
        self.body = net
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
        return h, infos

    def heads(self, h, act):
        """policy a_t from h_t; forward-model latent for o_{t+1} from (h_t, a_t)."""
        pol = self.policy(h)
        z = self.fwd(h + self.act_fwd(act).to(h.dtype))
        return pol, z
