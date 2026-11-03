"""Cost of one memory read per frame (streaming decode) as memory grows from 1 hour to 1 month of video.
Entries arrive at the hnet_v5 rate (2,342 per hour). Full: SDPA over every entry with the ALiBi bias. LSA flat: score
one centroid per bucket (4 entries), expand the best into k = 32 entries, attend to those + 32 recent. LSA tree: one more
level (32 buckets per node), beam 16. Index dim 64 (D-LSA projection) or 512 (raw keys)."""
import json, math, time, torch
import torch.nn.functional as F
dev = "cuda"; H, dh, PER_HOUR = 8, 64, 2342


def t_ms(fn, n=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1e3


res = []
for hours in (1, 24, 24 * 7, 24 * 30):
    L = PER_HOUR * hours
    try:
        k = torch.randn(1, H, L, dh, device=dev, dtype=torch.bfloat16); v = torch.randn_like(k)
        q = torch.randn(1, H, 1, dh, device=dev, dtype=torch.bfloat16)
        tk = torch.arange(L, device=dev).float() * (36000 / PER_HOUR)
        sl = torch.tensor([2 ** (-8 * (i + 1) / H) for i in range(H)], device=dev) * 1e-3

        def full():
            bias = (-sl[:, None] * (tk[-1] + 1 - tk)[None]).to(k.dtype)[None, :, None]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        r = dict(hours=hours, entries=L, kv_gb=2 * k.numel() * 2 / 1e9, full_ms=t_ms(full))
        for d_idx in (64, 512):
            nb = L // 4
            cent = F.normalize(torch.randn(nb, d_idx, device=dev, dtype=torch.bfloat16), dim=-1)
            iq = F.normalize(torch.randn(d_idx, device=dev, dtype=torch.bfloat16), dim=-1)
            node = F.normalize(torch.randn(max(1, nb // 32), d_idx, device=dev, dtype=torch.bfloat16), dim=-1)

            def attend(idx):
                kk, vv = k[:, :, idx], v[:, :, idx]
                bias = (-sl[:, None] * (tk[-1] + 1 - tk[idx])[None]).to(k.dtype)[None, :, None]
                return F.scaled_dot_product_attention(q, kk, vv, attn_mask=bias)

            def flat():
                top = (cent @ iq).topk(8).indices                      # 8 buckets x 4 = 32 long-range entries
                idx = torch.cat([(top[:, None] * 4 + torch.arange(4, device=dev)).flatten(), torch.arange(L - 32, L, device=dev)])
                return attend(idx)

            def tree():
                tn = (node @ iq).topk(min(16, len(node))).indices
                kids = (tn[:, None] * 32 + torch.arange(32, device=dev)).flatten().clamp(max=nb - 1)
                top = kids[(cent[kids] @ iq).topk(8).indices]
                idx = torch.cat([(top[:, None] * 4 + torch.arange(4, device=dev)).flatten(), torch.arange(L - 32, L, device=dev)])
                return attend(idx)
            r[f"flat{d_idx}_ms"] = t_ms(flat); r[f"tree{d_idx}_ms"] = t_ms(tree)
            r[f"index{d_idx}_gb"] = (cent.numel() + node.numel()) * 2 / 1e9
        del k, v
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        r = dict(hours=hours, entries=L, oom=True)
    print(json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in r.items()}), flush=True)
    res.append(r)
json.dump(res, open("runs_lsa/bench.json", "w"), indent=1)
