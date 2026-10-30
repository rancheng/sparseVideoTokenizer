"""Sparse spatio-temporal front end: H-Net dynamic chunking applied per image cell, plus a learned fovea.

Per frame (128x128 = 8x8 cells of 16x16):
  1. cheap embedding of every cell from its 4x-downsampled pixels (48 numbers) -- the only always-on cost
  2. per-cell temporal router (H-Net: p = sigmoid((1 - cos(Wq e_t, Wk e_{t-1}) - beta) / tau), optionally + a learned
     relevance term) -> a cell is an EVENT if p >= 0.5, plus a staggered per-cell heartbeat
  3. learned fovea: the k_fov event cells with the highest learned score get a high-resolution conv encoding
  4. sparse spatial transformer: event cells are recomputed, attending to all cells; non-event cells keep their cached
     state from their last event (exactly what the streaming engine does, so training == inference)
  5. per-cell dechunk: H-Net EMA over the cell's events with weight p, causal repeat, straight-through confidence
  6. K latent queries pool the 64 cell states into one frame vector -> the temporal H-Net (vt/model.py) as before

Training emulates the sparse inference exactly (stale cells are gathered from their last event time) and computes the
high-resolution encoder only on event cells, so the expensive per-cell work is sparse at training time as well.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vt.model import RMSNorm, linear_scan, VideoHNet
from vt.env2 import G, CELL


def to_cells(frames):  # (B,T,3,H,W) -> (B,T,G*G,3,CELL,CELL), cell j = gy*G + gx
    B, T = frames.shape[:2]
    x = frames.reshape(B, T, 3, G, CELL, G, CELL).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(B, T, G * G, 3, CELL, CELL)


def last_index(b):
    """b (B,T,J) bool -> index of the latest t' <= t with b[t'] (b[:,0] must be all True)."""
    T = b.shape[1]
    t = torch.arange(T, device=b.device)[None, :, None].expand_as(b)
    return torch.cummax(torch.where(b, t, torch.zeros_like(t)), 1).values


def stale(x, idx):  # x (B,T,J,D), idx (B,T,J) -> x[b, idx[b,t,j], j]
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, -1, x.shape[-1]))


class SpatialBlock(nn.Module):
    """Pre-norm transformer block over the cells of one frame (bidirectional within the frame)."""

    def __init__(self, d, dh=32):
        super().__init__()
        self.h, self.dh = d // dh, dh
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.w12 = nn.Linear(d, 4 * d, bias=False); self.w3 = nn.Linear(2 * d, d, bias=False)

    def attn_kv(self, x):  # x (N,J,D) -> q,k,v (N,h,J,dh)
        N, J, D = x.shape
        return self.qkv(self.n1(x)).view(N, J, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)

    def finish(self, x, a):  # x (N,J',D), a (N,h,J',dh)
        x = x + self.o(a.transpose(1, 2).reshape(*x.shape))
        u, g = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(u) * g)

    def forward(self, x):
        q, k, v = self.attn_kv(x)
        return self.finish(x, F.scaled_dot_product_attention(q, k, v))


class HiRes(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU())
        self.fc = nn.Linear(64 * 16, d)

    def forward(self, c):  # (N,3,16,16)
        return self.fc(self.net(c).flatten(1))


class SparseFront(nn.Module):
    def __init__(self, d=128, dout=256, n_layers=2, router="tied", fovea="learned", N_cell=16, N_fov=4,
                 heartbeat=64, K=4, k_max=12, k_fov=2, fov_credit="ste", fov_pos=False, fov_explore=1.0):
        super().__init__()
        # fov_credit "ste": the fovea score learns from every loss through a straight-through gate, and only on the cells
        # it already picked. "task": the score regresses the first-order benefit of high resolution under the action
        # losses alone, measured on every event cell (counterfactual gate; train2.py), with Gumbel top-k exploration
        self.fov_credit, self.fov_pos, self.fov_explore = fov_credit, fov_pos, fov_explore
        self.force_fovea = False  # training warm-up only (see train2.py --fov_warm)
        self.k_fov = k_fov  # high-resolution cells per frame (learned fovea)
        self.k_max = k_max  # hard per-frame cap on routed events (heartbeats excluded): bounded compute and memory
        self.d, self.router, self.fovea, self.N_cell, self.N_fov, self.hb, self.K = d, router, fovea, N_cell, N_fov, heartbeat, K
        J = G * G
        self.cheap = nn.Linear(48, d)
        self.pos = nn.Parameter(torch.randn(J, d) * 0.02)
        self.wq = nn.Linear(d, d, bias=False); self.wk = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.wq.weight.copy_(torch.eye(d)); self.wk.weight.copy_(torch.eye(d))
        self.beta = nn.Parameter(torch.zeros(())); self.log_tau = nn.Parameter(torch.zeros(()))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        self.register_buffer("heur_th", torch.zeros(()))
        if router == "rel":  # learned relevance: lets the router veto change that does not help prediction
            self.rel = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
            nn.init.zeros_(self.rel[2].weight); nn.init.zeros_(self.rel[2].bias)
        self.hires = HiRes(d)
        self.fov = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.fov[2].weight); nn.init.zeros_(self.fov[2].bias)
        self.blocks = nn.ModuleList([SpatialBlock(d) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.lat = nn.Parameter(torch.randn(K, d) * 0.02)
        self.pool_kv = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(K * d, dout)

    # ---- per-frame pieces shared with the streaming engine ----
    def lowres(self, cells):  # (...,3,16,16) -> (...,48)
        return F.avg_pool2d(cells.reshape(-1, 3, CELL, CELL), 4).reshape(*cells.shape[:-3], 48)

    def route_p(self, c, c_prev):  # cheap content embeddings (..., d) -> p (...)
        # the router reads the embedding but does not write it: through the sharp temperature its ratio-loss gradient
        # was ~20x every task gradient and, after global clipping, starved the policy (measured: 7.4 vs 0.4)
        c, c_prev = c.detach(), c_prev.detach()
        q = F.normalize(self.wq(c.float()), dim=-1); k = F.normalize(self.wk(c_prev.float()), dim=-1)
        logit = (1 - (q * k).sum(-1) - self.beta) / self.log_tau.exp().clamp(min=1e-5)
        if self.router == "rel":
            logit = logit + self.rel(c.float()).squeeze(-1)
        return torch.sigmoid(logit)

    def fov_logit(self, c, j=None):  # fovea score from the (detached) content embedding, plus the cell position
        x = c.detach().float()
        if self.fov_pos:
            x = x + (self.pos if j is None else self.pos[j]).detach().float()
        return self.fov(x).squeeze(-1)

    def heartbeat(self, t, J, device):  # t (T,) absolute frame times -> (T,J) staggered forced updates
        j = torch.arange(J, device=device)
        return ((t[:, None] + 7 * j[None]) % self.hb) == 0

    def pool(self, s):  # s (N,J,d) -> (N,dout)
        N, J, d = s.shape
        k, v = self.pool_kv(s).chunk(2, -1)
        q = self.lat[None].expand(N, -1, -1)
        a = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        return self.out(a.reshape(N, -1))

    def forward(self, frames, t0=0, state=None):
        """frames (B,T,3,128,128). state: carried from a previous chunk (exact chunked processing of long streams).
        Returns frame vectors (B,T,dout), info, new state."""
        B, T = frames.shape[:2]
        J = G * G
        cells = to_cells(frames)
        low = self.lowres(cells)                               # (B,T,J,48)
        c = self.cheap(low)                                    # content embedding (router input)
        tt = torch.arange(T, device=frames.device) + t0
        hb = self.heartbeat(tt, J, frames.device)[None].expand(B, T, J)
        first = state is None
        c_prev = torch.cat([c[:, :1] if first else state["c"][:, None], c[:, :-1]], 1)
        if self.router in ("tied", "rel"):
            if self.training and not bool(self.calibrated):
                with torch.no_grad():
                    q = F.normalize(c[:, 1:].float(), dim=-1); k = F.normalize(c[:, :-1].float(), dim=-1)
                    dv = (1 - (q * k).sum(-1)).flatten()
                    dv = dv[torch.randperm(dv.numel(), device=dv.device)[:200000]]
                    lo, md, hi = torch.quantile(dv, torch.tensor([0.5, 1 - 1 / self.N_cell, 0.99], device=dv.device))
                    self.beta.copy_(md); self.log_tau.copy_(torch.log(((hi - md) / 4).clamp(min=1e-5)))
                    self.calibrated.fill_(True)
            p = self.route_p(c, c_prev)
            b = p >= 0.5
            score = p
        elif self.router == "heur":  # codec-style: mean |change| of the low-res cell above a calibrated threshold
            low_prev = torch.cat([low[:, :1] if first else state["low"][:, None], low[:, :-1]], 1)
            sc = (low - low_prev).abs().mean(-1)
            if self.training and not bool(self.calibrated):
                with torch.no_grad():
                    self.heur_th.copy_(torch.quantile(sc[:, 1:].flatten()[:1000000].float(), 1 - 1 / self.N_cell))
                    self.calibrated.fill_(True)
            b = sc > self.heur_th; p = b.float()
            score = sc
        else:  # dense: every cell every frame
            b = torch.ones(B, T, J, dtype=torch.bool, device=frames.device); p = b.float()
        if self.router != "dense" and self.k_max and self.k_max < J:
            b = b & (score >= torch.topk(score, self.k_max, dim=-1).values[..., -1:])
        if first:
            hb = hb.clone(); hb[:, 0] = True
        if self.router in ("tied", "rel"):
            p = torch.where(hb, torch.maximum(p, torch.full_like(p, 0.5)), p)
        b = b | hb
        # fovea gate (only matters on event cells)
        e = c + self.pos
        if self.fovea == "learned":
            # learned fovea = top-k ranking (mixture-of-depths style): the k_fov event cells with the highest score get
            # the high-resolution encoder. The score's gradient is how much those features helped (straight-through,
            # value 1); nothing pushes all scores down, the gate only has to rank. Fixed compute per frame.
            v = self.fov_logit(c)
            gs = torch.sigmoid(v)
            sv = v
            if self.fov_credit == "task" and self.training and self.fov_explore > 0:  # Gumbel top-k = sampled ranking
                sv = v - self.fov_explore * torch.log(-torch.log(torch.rand_like(v).clamp(1e-9, 1 - 1e-9)))
            if self.force_fovea:  # warm-up: every event cell, so the model learns to read fine detail first
                gh = b
            else:
                sc = torch.where(b, sv, torch.full_like(sv, -float("inf")))
                gh = b & (sc >= torch.topk(sc, self.k_fov, dim=-1).values[..., -1:])
            if self.fov_credit == "task":
                # counterfactual gate: in training every event cell is encoded at high resolution, the gate (value 0/1,
                # a leaf) keeps only the picked ones in the forward pass, and d(task loss)/d(gate) on the others says
                # what high resolution there would have bought. The forward pass equals inference exactly.
                g = gh.float().requires_grad_(self.training and torch.is_grad_enabled())
            else:
                g = 1 + gs - gs.detach()
        elif self.fovea == "all":
            gs = torch.ones(B, T, J, device=frames.device); gh = gs > 0; g = gs
        else:
            gs = torch.zeros(B, T, J, device=frames.device); gh = gs > 0; g = gs
        use = b & gh                    # cells whose high-resolution encoding enters the model (= inference cost)
        enc = b if (self.fovea == "learned" and self.fov_credit == "task" and g.requires_grad) else use
        x0 = e
        if enc.any():
            hr = torch.zeros(B, T, J, self.d, device=frames.device, dtype=e.dtype)
            ii = enc.nonzero(as_tuple=True)
            hr[ii] = self.hires(cells[ii]).to(e.dtype)
            x0 = e + g[..., None].to(e.dtype) * hr
        # virtual frame -1 carries the previous chunk's cached cell states
        if not first:
            b_ext = torch.cat([torch.ones_like(b[:, :1]), b], 1)
            idx = last_index(b_ext)
        else:
            idx = last_index(b)
        def fill(x, l):
            if not first:
                x = torch.cat([state["x"][l][:, None].to(x.dtype), x], 1)
            x = stale(x, idx)
            return x if first else x[:, 1:]
        x = fill(x0, 0)
        x_last = [x[:, -1].detach()]
        for l, blk in enumerate(self.blocks):
            x = blk(x.reshape(B * T, J, -1)).reshape(B, T, J, -1)
            x = fill(x, l + 1)
            x_last.append(x[:, -1].detach())
        s = self.norm(x)
        # per-cell H-Net dechunk over time: EMA with weight p at events, hold otherwise; STE confidence
        P = torch.where(b, p, torch.zeros_like(p))
        a = (1 - P).permute(0, 2, 1).reshape(B * J, T, 1)
        bb = (P[..., None] * s.float()).permute(0, 2, 1, 3).reshape(B * J, T, -1)
        if not first:  # prepend the carried smoothed state with a=0 so the scan continues it exactly
            a = torch.cat([torch.zeros_like(a[:, :1]), a], 1)
            bb = torch.cat([state["zbar"].float().reshape(B * J, 1, -1), bb], 1)
        zbar = linear_scan(a, bb)
        if not first:
            zbar = zbar[:, 1:]
        zbar = zbar.reshape(B, J, T, -1).permute(0, 2, 1, 3)
        conf = torch.where(b, p, 1 - p)
        cell = (zbar * (conf + (1 - conf).detach())[..., None]).to(s.dtype)
        f = self.pool(cell.reshape(B * T, J, -1)).reshape(B, T, -1)
        new_state = {"c": c[:, -1].detach(), "low": low[:, -1].detach(), "zbar": zbar[:, -1].detach(), "x": x_last}
        info = {"b": b, "p": p, "gate": gh, "gate_soft": gs, "use": use, "hb": hb, "cell": cell}
        if self.fovea == "learned":
            info.update(fov_v=v, fov_gate=g)
        return f, info, new_state


class CellDecoder(nn.Module):
    """Forward model per cell: o_{t+1}[cell j] from the temporal latent (h_t, a_t), the cell state, its position and
    the observed cell."""

    def __init__(self, d_cell, dz):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(G * G, dz) * 0.02)
        self.cin = nn.Linear(d_cell, dz)
        self.mlp = nn.Sequential(RMSNorm(dz), nn.Linear(dz, dz), nn.SiLU(), nn.Linear(dz, 64 * 16))
        self.logv = nn.Linear(dz, 1)  # predicted log-variance of the cell: unpredictable cells (noisy TV) learn a large one
        self.up = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
                                nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, z, cell, j, cur, with_var=False):
        """z (N,dz), cell (N,d_cell), j (N,), cur (N,3,16,16) observed cell at t -> o_{t+1} cell [, log-var (N,)].
        Residual coding: the prediction is the observed cell plus a predicted change, so static content is free and
        the loss concentrates on what changes (an item emerging from the tunnel needs memory)."""
        u = z + self.cin(cell) + self.pos[j]
        mean = cur.to(u.dtype) + self.up(self.mlp(u).view(-1, 64, 4, 4))
        return (mean, self.logv(u).squeeze(-1).float().clamp(-12, 2)) if with_var else mean


class SparseVideoModel(nn.Module):
    """Sparse front end -> temporal H-Net (frame-level chunking, main network, event-token memory) -> heads."""

    def __init__(self, front=None, temporal=None):
        super().__init__()
        front = dict(front or {}); temporal = dict(temporal or {})
        self.front = SparseFront(**front)
        temporal.setdefault("dims", (256, 384, 512))
        self.temporal = VideoHNet(frame_io=False, **temporal)
        self.cdec = CellDecoder(self.front.d, temporal["dims"][0])
        # heading as a classification over angle bins: MSE regression averages multimodal targets (an unread rule makes
        # the bin ambiguous) into a mean heading that carries no gradient for the glyph; cross-entropy does not
        self.n_ang = 64
        self.ang = nn.Sequential(RMSNorm(temporal["dims"][0]), nn.Linear(temporal["dims"][0], self.n_ang))

    def heading(self, h):  # most likely heading as a unit vector (..., 2)
        a = (self.ang(h).argmax(-1).float() + 0.5) * (2 * torch.pi / self.n_ang) - torch.pi
        return torch.stack([a.cos(), a.sin()], -1)

    def forward(self, frames, prev_act, t0=0, state=None):
        x, finfo, fstate = self.front(frames, t0, state)
        h, infos = self.temporal(None, prev_act, x_lat=x)
        return h, finfo, infos, fstate
