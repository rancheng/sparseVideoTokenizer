"""Action-conditioned latent prediction (JEPA) heads for the sparse video model.

Targets: an EMA copy of the sparse front encodes the frame to be predicted, every cell at full resolution, within-frame
context (encode_dense), layer-normalised, no gradient. The predictor reads the temporal output h_t, the actions
a_t..a_{t+k-1}, the target cell's position and the horizon k. For k > 1 it does not see the target cell's own current
state, so whatever the cell will hold k frames later has to come through the token stream and the memory (the
shortcut that let the pixel decoder ignore h is gone). Alongside:
  inverse dynamics: a_t from the front's frame vectors f_t and f_{t+k} (action-free inputs): what the actions change
  future actions:   a_{t+k} from h_t: what will matter for behaviour
  surprise:         the one-step prediction error per cell, a target for the cell router (events where the latent
                    world model is wrong, not where pixels changed)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from vt.model import RMSNorm
from vt.env2 import G

HORIZONS = (1, 4, 16, 64)


class Res(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = RMSNorm(d); self.f = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        return x + self.f(self.n(x))


class JEPAHeads(nn.Module):
    def __init__(self, front, dz, act_dim, d=256, horizons=HORIZONS, ema=0.996):
        super().__init__()
        self.horizons, self.ema = horizons, ema
        self.target = copy.deepcopy(front).requires_grad_(False)
        dc = front.d
        self.h_in = nn.Linear(dz, d); self.cell_in = nn.Linear(dc, d); self.pos = nn.Parameter(torch.randn(G * G, d) * 0.02)
        self.hz = nn.Embedding(len(horizons), d); self.act_in = nn.Linear(2 * act_dim, d)
        self.pred = nn.Sequential(Res(d), Res(d), Res(d), RMSNorm(d), nn.Linear(d, dc))
        df = front.out.out_features
        self.inv = nn.Sequential(nn.Linear(2 * df + d, d), nn.SiLU(), Res(d), RMSNorm(d), nn.Linear(d, act_dim))
        self.fut = nn.Sequential(RMSNorm(dz + d), nn.Linear(dz + d, d), nn.SiLU(), Res(d), RMSNorm(d), nn.Linear(d, act_dim))

    @torch.no_grad()
    def update_target(self, front):
        for pt, po in zip(self.target.parameters(), front.parameters()):
            pt.lerp_(po.detach(), 1 - self.ema)
        for bt, bo in zip(self.target.buffers(), front.buffers()):
            bt.copy_(bo)

    @torch.no_grad()
    def targets(self, frames):  # (B,N,3,H,W) -> (B,N,J,dc), layer-normalised
        y = self.target.encode_dense(frames).float()
        return F.layer_norm(y, y.shape[-1:])

    def predict(self, h, j, hidx, act_feat, cell=None):
        """h (N,dz), j (N,) cell, hidx (N,) horizon index, act_feat (N,2A) [sum of a_t..a_{t+k-1}, a_t], cell (N,dc) or None"""
        x = self.h_in(h) + self.pos[j] + self.hz(hidx) + self.act_in(act_feat)
        if cell is not None:
            x = x + self.cell_in(cell) * (hidx == 0).to(x.dtype)[:, None]   # current cell state only for k = 1
        return self.pred(x)
