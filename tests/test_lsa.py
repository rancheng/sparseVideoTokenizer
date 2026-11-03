"""LSA index and native sparse memory read: selecting every entry must reproduce the dense read."""
import numpy as np
import torch
from vt.lsa import ward, LSAIndex
from vt.model import MemoryRead


def test_ward_partitions():
    x = np.random.default_rng(0).normal(size=(32, 16))
    g = ward(x, 8, nmax=8)
    assert len(g) == 8 and sorted(np.concatenate(g).tolist()) == list(range(32)) and max(map(len, g)) <= 8


def test_index_window_and_budget():
    keys, t = torch.randn(320, 16), torch.arange(320) * 10
    ix = LSAIndex(keys, t, n_cluster=32, n_bucket=4, device="cpu")
    idx, _ = ix.select(torch.nn.functional.normalize(torch.randn(16), dim=0), 3000, k=16, n_window=32)
    assert len(idx) == 48 and len(set(idx.tolist())) == 48 and (t[idx] <= 3000).all()


def test_sparse_read_matches_dense_when_reading_all():
    torch.manual_seed(0)
    r = MemoryRead(64, 128, src="tok").cuda()
    torch.nn.init.normal_(r.o.weight, std=0.02)
    T, L = 512, 96
    h = torch.randn(1, T, 64, device="cuda")
    t = torch.arange(T, device="cuda")[None]
    mt = torch.sort(torch.randint(0, T, (1, L), device="cuda")).values
    mv = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mem = torch.randn(1, L, 256, device="cuda")
    dense = r(h, t, mem, mt, mv)
    r.sparse = L
    sparse = r(h, t, mem, mt, mv)
    assert torch.allclose(dense, sparse, atol=1e-4), (dense - sparse).abs().max()
    r.sparse = 8
    assert not torch.allclose(dense, r(h, t, mem, mt, mv), atol=1e-4)
