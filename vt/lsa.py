"""Latent Sparse Attention index (Kubicek et al. 2026, arXiv:2610.42147), small-scale research version.

Keys are clustered by content inside blocks of consecutive keys (Ward's method), once per block, and never change. A
query scores bucket centroids instead of keys and expands the best buckets into at most k key indices, next to a window
of the most recent keys (which covers the block still open). Two additions for streams, both off by default:
  recency: score - recency * log2(1 + elapsed frames of the bucket's newest key), so a stale bucket can lose to a fresh one
  newest:  expand each selected bucket newest-key first (a bucket of many old near-duplicates gives up its latest ones)
"""
import numpy as np
import torch
import torch.nn.functional as F


def ward(x, m, nmax=None):
    """Bottom-up Ward clustering with parallel reciprocal-best merges. x (n,d) -> list of m (or more) index arrays."""
    mem = [np.array([i]) for i in range(len(x))]
    s = x.astype(np.float64).copy()                # sum of each bucket's keys
    c = np.ones(len(x))                            # size of each bucket
    alive = np.arange(len(x))
    while len(alive) > m:
        mu = s[alive] / c[alive, None]
        d2 = ((mu[:, None] - mu[None]) ** 2).sum(-1)
        ca = c[alive]
        cost = ca[:, None] * ca[None] / (ca[:, None] + ca[None]) * d2
        np.fill_diagonal(cost, np.inf)
        if nmax is not None:
            cost[ca[:, None] + ca[None] > nmax] = np.inf
        best = cost.argmin(1)
        ok = np.isfinite(cost[np.arange(len(alive)), best])
        if not ok.any():
            break
        pairs = [(i, int(best[i])) for i in range(len(alive)) if ok[i] and best[int(best[i])] == i and i < best[i]]
        pairs.sort(key=lambda p: cost[p])
        keep = np.ones(len(alive), bool)
        for i, j in pairs[: len(alive) - m]:
            a, b = alive[i], alive[j]
            s[a] += s[b]; c[a] += c[b]; mem[a] = np.concatenate([mem[a], mem[b]])
            keep[j] = False
        alive = alive[keep]
    return [mem[a] for a in alive]


class LSAIndex:
    """keys (L,d) in stream order, times (L,) frames. Buckets of blocks that are complete at query time are searchable."""

    def __init__(self, keys, times, n_cluster=32, n_bucket=4, n_pool=1, tree=0, normalize=True, nmax=None, device="cuda"):
        self.dev, self.n_cluster, self.tree = device, n_cluster, tree
        k = F.normalize(keys.float(), dim=-1) if normalize else keys.float()
        self.times = times.to(device)
        L = (len(k) // n_cluster) * n_cluster
        kn = k.cpu().numpy()
        cents, mems, b_end, b_new = [], [], [], []
        for a in range(0, L, n_cluster):
            blk = kn[a:a + n_cluster]
            n = len(blk) // n_pool
            pool = np.stack([blk[i * n_pool:(i + 1) * n_pool].sum(0) for i in range(n)])
            pool = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-9)
            for g in ward(pool, max(1, n_cluster // n_bucket // 1 if n_pool == 1 else max(1, n // max(1, n_bucket // n_pool))), nmax):
                idx = np.concatenate([np.arange(i * n_pool, (i + 1) * n_pool) for i in g]) + a
                cents.append(kn[idx].sum(0)); mems.append(np.sort(idx))
                b_end.append(a + n_cluster - 1)
                b_new.append(idx.max())
        c = F.normalize(torch.as_tensor(np.stack(cents), device=device).float(), dim=-1) if cents else torch.zeros(0, k.shape[1], device=device)
        self.cent, self.mem = c, mems
        self.b_end = torch.as_tensor(b_end, device=device, dtype=torch.long)
        self.b_new_t = self.times[torch.as_tensor(b_new, device=device, dtype=torch.long)] if b_new else torch.zeros(0, device=device)
        self.nodes = None
        if tree:  # one level of node buckets over `tree` consecutive buckets' worth of Ward groups
            self._build_tree(tree)

    def _build_tree(self, fan):
        c = self.cent.cpu().numpy()
        nodes, kids = [], []
        for a in range(0, len(c), fan * 8):
            part = c[a:a + fan * 8]
            for g in ward(part, max(1, len(part) // fan)):
                kids.append(g + a); nodes.append(part[g].sum(0))
        self.nodes = F.normalize(torch.as_tensor(np.stack(nodes), device=self.dev).float(), dim=-1)
        self.kids = kids
        self.n_end = torch.as_tensor([int(self.b_end[g].max()) for g in kids], device=self.dev)
        self.n_new_t = torch.stack([self.b_new_t[torch.as_tensor(g, device=self.dev)].max() for g in kids])

    def select(self, q, t, k, n_window, recency=0.0, newest=False, beam=None):
        """q (d,) index query, t frame time. -> long tensor of key indices (window + up to k long-range), #centroids scored"""
        nv = int((self.times <= t).sum())                                 # keys that exist at time t
        w0 = max(0, nv - n_window)
        win = torch.arange(w0, nv, device=self.dev)
        elig = self.b_end < nv                                             # buckets of completed blocks (window keys skipped)
        cand = elig.nonzero()[:, 0]
        scored = 0
        if self.nodes is not None and beam:
            ne = (self.n_end < nv).nonzero()[:, 0]
            sn = self.nodes[ne] @ q; scored += len(ne)
            if recency:
                sn = sn - recency * torch.log2(1 + (t - self.n_new_t[ne]).float().clamp(min=0))
            top = ne[sn.topk(min(beam, len(ne))).indices].cpu().numpy() if len(ne) else []
            kids = [torch.as_tensor(self.kids[i], device=self.dev) for i in top]
            cand2 = torch.cat(kids) if kids else torch.zeros(0, dtype=torch.long, device=self.dev)
            openb = torch.ones(len(self.cent), dtype=torch.bool, device=self.dev)
            if len(ne):
                openb[torch.as_tensor(np.concatenate([self.kids[i] for i in ne.cpu().numpy()]), device=self.dev)] = False
            cand = torch.cat([cand2, (openb & elig).nonzero()[:, 0]])
            cand = cand[elig[cand]]
        if not len(cand) or k == 0:
            return win, scored
        s = self.cent[cand] @ q
        scored += len(cand)
        if recency:
            s = s - recency * torch.log2(1 + (t - self.b_new_t[cand]).float().clamp(min=0))
        order = cand[s.argsort(descending=True)].cpu().numpy()
        got, n = [], 0
        for b in order:
            idx = self.mem[b]
            idx = idx[idx < w0]
            if newest:
                idx = idx[::-1]
            idx = idx[: k - n]
            got.append(idx); n += len(idx)
            if n >= k:
                break
        lr = torch.as_tensor(np.concatenate(got).copy(), device=self.dev) if got else win[:0]
        return torch.cat([lr, win]), scored


def block_select(keys_n, times, q, t, k, n_window, b=4):
    """Block sparse attention baseline: scores every block of b consecutive keys by its summed key (flat search)."""
    nv = int((times <= t).sum()); w0 = max(0, nv - n_window)
    win = torch.arange(w0, nv, device=keys_n.device)
    nb = w0 // b
    if nb == 0 or k == 0:
        return win, 0
    blocks = F.normalize(keys_n[:nb * b].view(nb, b, -1).sum(1), dim=-1)
    top = (blocks @ q).topk(min(max(1, k // b), nb)).indices
    return torch.cat([(top[:, None] * b + torch.arange(b, device=top.device)).flatten(), win]), nb
