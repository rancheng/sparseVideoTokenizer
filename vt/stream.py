"""Event-driven streaming engine for a trained VideoHNet.

Runs the exact same weights one frame at a time:
  * every frame:            frame encoder + stage-1 encoder/decoder (sliding-window KV ring caches) + heads
  * on a level-1 boundary:  stage-2 encoder/decoder step
  * on a level-2 boundary:  ONE main-network step, appending one entry to its KV cache
Between boundaries, a level reuses its cached smoothed chunk state (zbar), which is exactly what dechunking
computes in training. The expensive network therefore runs at the *event* rate, not the frame rate, and the
only state that grows with time is the main network's KV cache (one entry per top-level event).

Every frame also does FLOP accounting (linear layers + attention), so dense vs. sparse cost can be compared.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from vt.model import Stage, MainNet, rope, dt_bucket, time_slopes


class WindowCache:
    """Last-W keys/values (un-roped; RoPE is applied with local positions, as in training)."""

    def __init__(self):
        self.k = self.v = None

    def push(self, k, v, W):  # k,v (H,1,dh)
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k[:, -(W - 1):], k], 1)
            self.v = torch.cat([self.v[:, -(W - 1):], v], 1)
        return self.k, self.v


class FullCache:
    """Append-only KV cache (NoPE main network) with capacity doubling."""

    def __init__(self):
        self.k = self.v = None; self.n = 0

    def push(self, k, v):
        if self.k is None or self.n == self.k.shape[1]:
            cap = max(1024, 2 * self.n)
            nk = k.new_empty(k.shape[0], cap, k.shape[2]); nv = torch.empty_like(nk)
            if self.n:
                nk[:, :self.n] = self.k[:, :self.n]; nv[:, :self.n] = self.v[:, :self.n]
            self.k, self.v = nk, nv
        self.k[:, self.n] = k[:, 0]; self.v[:, self.n] = v[:, 0]; self.n += 1
        return self.k[:, :self.n], self.v[:, :self.n]

    def nbytes(self):
        return 0 if self.k is None else 2 * self.n * self.k.shape[0] * self.k.shape[2] * self.k.element_size()


def lin_flops(mod):
    return 2 * sum(m.weight.numel() for m in mod.modules() if isinstance(m, torch.nn.Linear))


class StackRunner:
    def __init__(self, stack, full, alibi=False):
        self.stack, self.full, self.alibi = stack, full, alibi
        self.caches = [FullCache() if full else WindowCache() for _ in stack.blocks]
        self.block_flops = [lin_flops(b) for b in stack.blocks]
        self.tk = []  # key times (frames) for the time-recency bias
        if alibi:
            self.slopes = time_slopes(stack.blocks[0].attn.h, next(stack.parameters()).device)

    def step(self, x, flops, t=None):  # x (1,D)
        bias = None
        if self.alibi:
            self.tk.append(t)
            dt = (t - torch.tensor(self.tk, device=x.device, dtype=torch.float32)).clamp(min=0)
            bias = (-self.slopes[:, None, None] * dt[None, None])[None].to(x.dtype)  # (1,H,1,n)
        for blk, cache, bf in zip(self.stack.blocks, self.caches, self.block_flops):
            a = blk.attn
            q, k, v = a.qkv(blk.n1(x)).view(3, a.h, 1, a.dh).unbind(0)
            if self.full:
                K, V = cache.push(k, v)
                out = F.scaled_dot_product_attention(q[None], K[None], V[None], attn_mask=bias)
            else:
                K, V = cache.push(k, v, a.window)
                n = K.shape[1]; pos = torch.arange(n, device=x.device)
                out = F.scaled_dot_product_attention(rope(q[None], pos[-1:]), rope(K[None], pos), V[None])
            x = x + a.o(out.reshape(1, -1))
            u, g = blk.w12(blk.n2(x)).chunk(2, -1)
            x = x + blk.w3(F.silu(u) * g)
            flops[0] += bf + 4 * K.shape[1] * x.shape[-1]
        return self.stack.norm(x)

    def nbytes(self):
        return sum(c.nbytes() if isinstance(c, FullCache) else (0 if c.k is None else 2 * c.k.numel() * c.k.element_size())
                   for c in self.caches)


class LevelState:
    def __init__(self, S):
        self.S = S
        self.enc, self.dec = StackRunner(S.enc, False), StackRunner(S.dec, False)
        self.prev_h = None; self.prev_t = None; self.n = 0
        self.zbar = None; self.last_chunk_t = None; self.nchunks = 0
        self.misc_flops = lin_flops(S.wq) + lin_flops(S.wk) + lin_flops(S.up) + lin_flops(S.down) + lin_flops(S.res)


class StreamingVideoHNet:
    def __init__(self, model, thresholds=None):
        self.m = model.eval()
        self.th = thresholds
        self.levels = []
        net = model.body
        while isinstance(net, Stage):
            self.levels.append(LevelState(net)); net = net.inner
        self.main = StackRunner(net.stack, True, alibi=getattr(net, "alibi", False))
        self.read_k = self.read_v = None; self.read_t = []  # per-frame memory read cache (one entry per event)
        self.t = 0; self.prev_frame = None
        dev = next(model.parameters()).device
        with FlopCounterMode(display=False) as fc:  # constant per-frame cost of frame encoder + heads
            with torch.no_grad():
                if hasattr(model, "fenc"):  # absent when a sparse front end supplies the latents (vt/stream2.py)
                    model.fenc(torch.zeros(1, 3, 64, 64, device=dev))
                z = torch.zeros(1, model.act_in.out_features, device=dev)
                model.heads(z, torch.zeros(1, 3, device=dev))
        self.frame_flops = fc.get_total_flops() + lin_flops(model.act_in)
        self.flops = {"frame_enc+heads": 0, "level1": 0, "level2": 0, "main": 0}
        self.events = []  # (level, frame t) for every boundary

    def _level_step(self, l, x, t, score):
        L = self.levels[l]; S = L.S
        fl = [0]
        h = L.enc.step(x, fl)
        if S.chunker == "learned":
            if L.n == 0:
                p = torch.ones((), device=x.device)
            else:
                q = F.normalize(S.wq(h.float()), dim=-1); k = F.normalize(S.wk(L.prev_h.float()), dim=-1)
                dist = 1 - (q * k).sum()
                if getattr(S, "router", "cos") == "tied":
                    p = torch.sigmoid((dist - S.beta) / S.log_tau.exp().clamp(min=1e-5))
                else:
                    p = (0.5 * dist).clamp(0, 1)
            b = bool(p >= 0.5)
        else:
            if S.chunker == "fixed":
                b = L.n % S.N == 0
            elif S.chunker == "external":
                b = L.n == 0 or score > self.th[l]
            else:
                b = True
            p = torch.tensor(float(b), device=x.device)
        if S.heartbeat and S.chunker in ("learned", "external"):
            hb = L.n == 0 or (t // S.heartbeat) != (L.prev_t // S.heartbeat)
            if hb:
                b = True
                if S.chunker == "learned":
                    p = torch.maximum(p, torch.tensor(0.5, device=x.device))
        L.prev_h, L.prev_t = h, t; L.n += 1
        if b:
            dt = 0 if L.nchunks == 0 else t - L.last_chunk_t
            zi = S.up(h) + S.dt_emb(dt_bucket(torch.tensor([dt], device=x.device))).to(h.dtype)
            if l + 1 < len(self.levels):
                zo = self._level_step(l + 1, zi, t, score)
            else:
                mf = [0]; zo = self.main.step(zi, mf, t); self.flops["main"] += mf[0]
                if self.m.mem_read:  # write one memory entry per top-level event
                    r = self.m.read
                    kk, vv = r.kv(r.entry(zo, zi)).view(2, r.h, 1, r.dh).unbind(0)
                    self.read_k = kk if self.read_k is None else torch.cat([self.read_k, kk], 1)
                    self.read_v = vv if self.read_v is None else torch.cat([self.read_v, vv], 1)
                    self.read_t.append(t); self.flops["main"] += lin_flops(r.kv)
            zo = S.down(zo)
            p = p.to(zo.dtype)
            L.zbar = p * zo if L.zbar is None else p * zo + (1 - p) * L.zbar
            L.last_chunk_t = t; L.nchunks += 1
            self.events.append((l, t))
        y = L.dec.step(L.zbar + S.res(h), fl)
        self.flops[f"level{l+1}"] += fl[0] + L.misc_flops
        return y

    @torch.no_grad()
    def step(self, frame, prev_action, action=None):
        """frame (3,64,64) = o_t, prev_action (3,) = a_{t-1}. Returns policy output for a_t (and fwd latent)."""
        m = self.m
        score = 1e9 if self.prev_frame is None else (frame - self.prev_frame).abs().mean().item() if self.th else 0.0
        self.prev_frame = frame
        return self.step_latent(m.fenc(frame[None]), prev_action, action, score)

    @torch.no_grad()
    def step_latent(self, x, prev_action, action=None, score=0.0):
        """x (1,D) frame latent from any front end."""
        m = self.m
        if m.use_actions:
            x = x + m.act_in(prev_action[None]).to(x.dtype)
        h = self._level_step(0, x, self.t, score)
        if m.mem_read:  # every frame reads all events so far (time-causal, recency-biased)
            r = m.read
            q = r.q(r.n(h)).view(1, r.h, 1, r.dh)
            dt = (self.t - torch.tensor(self.read_t, device=h.device, dtype=torch.float32)).clamp(min=0)
            bias = (-time_slopes(r.h, h.device)[:, None, None] * dt[None, None])[None].to(q.dtype)
            out = F.scaled_dot_product_attention(q, self.read_k[None], self.read_v[None], attn_mask=bias)
            h = h + r.o(out.reshape(1, -1))
            self.flops["frame_enc+heads"] += lin_flops(r.q) + lin_flops(r.o) + 4 * self.read_k.shape[1] * r.h * r.dh
        pol, z = m.heads(h, (action if action is not None else torch.zeros(3, device=x.device))[None])
        self.flops["frame_enc+heads"] += self.frame_flops
        self.t += 1
        return pol[0], z[0], h[0]

    def state_bytes(self):
        lv = sum(L.enc.nbytes() + L.dec.nbytes() for L in self.levels)
        return {"windows": lv, "main_kv": self.main.nbytes(), "main_tokens": self.main.caches[0].n}
