"""Event-driven streaming engine for SparseVideoModel: per-cell sparse front end + temporal H-Net.

Per frame:
  * every cell:     4x downsample, cheap embedding, router (the only always-on work)
  * event cells:    fovea gate; high-resolution encoder only where the gate is on; spatial transformer rows,
                    attending to the cached keys/values of all cells; EMA update of the cell state
  * any event:      re-pool the frame vector (pooling keys/values are cached per cell, recomputed only for events)
  * then the temporal engine (vt/stream.py): level 1 every frame, level 2 / main network only on their boundaries
Non-event cells cost nothing beyond the cheap embedding. Matches SparseVideoModel.forward (tests/test_stream2.py).
"""
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from vt.env2 import G, CELL
from vt.model2 import to_cells
from vt.stream import StreamingVideoHNet, lin_flops


class StreamingSparseModel:
    def __init__(self, model):
        self.m = model.eval()
        fr = model.front
        self.fr = fr
        self.J = G * G
        self.temporal = StreamingVideoHNet(model.temporal)
        self.t = 0
        self.c_prev = self.low_prev = None
        self.X = None          # per layer: (J,d) cell states (input of layer l), cached between events
        self.KV = None         # per layer: (h,J,dh) keys, values of every cell
        self.zbar = None
        self.pkv = None        # pooling keys/values per cell
        self.f = None
        dev = next(model.parameters()).device
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                fr.hires(torch.zeros(1, 3, CELL, CELL, device=dev))
        d = fr.d
        self.cost = {"cheap": 2 * 48 * d, "router": 2 * 2 * d * d + (lin_flops(fr.rel) if fr.router == "rel" else 0),
                     "fovea": lin_flops(fr.fov), "hires": fc.get_total_flops(),
                     "block": [lin_flops(b) + 4 * self.J * d for b in fr.blocks],
                     "pool_row": lin_flops(fr.pool_kv), "pool": 4 * fr.K * self.J * d + lin_flops(fr.out)}
        self.flops = {"cheap+router": 0, "fovea+hires": 0, "spatial": 0, "pool": 0}
        self.n_events = 0; self.n_hires = 0

    def dense_flops_per_frame(self):
        c = self.cost; J = self.J
        return J * (c["cheap"] + c["fovea"] + c["hires"] + sum(c["block"]) + c["pool_row"]) + c["pool"]

    @torch.no_grad()
    def step(self, frame, prev_action, action=None):
        fr, J, t = self.fr, self.J, self.t
        cells = to_cells(frame[None, None])[0, 0]                      # (J,3,16,16)
        low = fr.lowres(cells)
        c = fr.cheap(low)
        tt = torch.tensor([t], device=frame.device)
        hb = fr.heartbeat(tt, J, frame.device)[0]
        if fr.router in ("tied", "rel"):
            p = fr.route_p(c, c if self.c_prev is None else self.c_prev)
            b = p >= 0.5; score = p
        elif fr.router == "heur":
            sc = (low - (low if self.low_prev is None else self.low_prev)).abs().mean(-1)
            b = sc > fr.heur_th; p = b.float(); score = sc
        else:
            b = torch.ones(J, dtype=torch.bool, device=frame.device); p = b.float()
        if fr.router != "dense" and fr.k_max and fr.k_max < J:
            b = b & (score >= torch.topk(score, fr.k_max).values[-1])
        if t == 0:
            hb = torch.ones_like(hb)
        if fr.router in ("tied", "rel"):
            p = torch.where(hb, torch.maximum(p, torch.full_like(p, 0.5)), p)
        b = b | hb
        self.c_prev, self.low_prev = c, low
        self.flops["cheap+router"] += J * (self.cost["cheap"] + self.cost["router"])
        ev = b.nonzero()[:, 0]
        n = len(ev); self.n_events += n
        if n:
            e = c[ev] + fr.pos[ev]
            if fr.fovea == "learned":
                gs = torch.sigmoid(fr.fov(c[ev].float()).squeeze(-1))
                use = gs >= torch.topk(gs, min(fr.k_fov, n)).values[-1]
                self.flops["fovea+hires"] += n * self.cost["fovea"]
            else:
                use = torch.full((n,), fr.fovea == "all", device=frame.device)
            if use.any():
                e = e.clone(); e[use] = e[use] + fr.hires(cells[ev[use]]).to(e.dtype)
                self.flops["fovea+hires"] += int(use.sum()) * self.cost["hires"]; self.n_hires += int(use.sum())
            if self.X is None:
                self.X = [torch.zeros(J, fr.d, device=frame.device, dtype=e.dtype) for _ in range(len(fr.blocks) + 1)]
                self.KV = [None] * len(fr.blocks)
            self.X[0][ev] = e
            for l, blk in enumerate(fr.blocks):
                q, k, v = blk.attn_kv(self.X[l][ev][None])                  # (1,h,n,dh)
                if self.KV[l] is None:
                    self.KV[l] = [torch.zeros(blk.h, J, blk.dh, device=frame.device, dtype=k.dtype) for _ in range(2)]
                self.KV[l][0][:, ev] = k[0]; self.KV[l][1][:, ev] = v[0]
                a = F.scaled_dot_product_attention(q, self.KV[l][0][None], self.KV[l][1][None])
                self.X[l + 1][ev] = blk.finish(self.X[l][ev][None], a)[0]
                self.flops["spatial"] += n * self.cost["block"][l]
            s = fr.norm(self.X[-1][ev]).float()
            P = p[ev][:, None].float()
            if self.zbar is None:
                self.zbar = torch.zeros(J, fr.d, device=frame.device)
            self.zbar[ev] = P * s + (1 - P) * self.zbar[ev]
            kv = fr.pool_kv(self.zbar[ev].to(e.dtype))
            if self.pkv is None:
                self.pkv = torch.zeros(J, kv.shape[-1], device=frame.device, dtype=kv.dtype)
            self.pkv[ev] = kv
            k_, v_ = self.pkv.chunk(2, -1)
            a = F.scaled_dot_product_attention(fr.lat[None, None].to(k_.dtype), k_[None, None], v_[None, None])[0, 0]
            self.f = fr.out(a.reshape(1, -1))
            self.flops["pool"] += n * self.cost["pool_row"] + self.cost["pool"]
        self.t += 1
        return self.temporal.step_latent(self.f, prev_action, action)

    def summary(self):
        tf = self.temporal.flops
        return {"frames": self.t, "events_per_frame": self.n_events / max(1, self.t), "hires_per_frame": self.n_hires / max(1, self.t),
                "flops_per_frame": {**{k: v / self.t for k, v in self.flops.items()}, **{k: v / self.t for k, v in tf.items()}},
                "dense_front_flops_per_frame": self.dense_flops_per_frame()}
