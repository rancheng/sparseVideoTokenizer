"""Real-scale latency of ONE main-network step vs. KV-cache length on the 4090 (random weights, bf16).

Main network ~0.7B params (24 layers, d=1536). Combined with the measured event rates, this projects
per-frame cost and memory for dense (main runs every frame) vs event-driven (main runs once per event)
streaming over 1 hour and 1 day.
"""
import json, time
import torch
from vt.model import Stack
from vt.stream import StackRunner

torch.manual_seed(0)
D, NL = 1536, 24
stack = Stack(D, NL, window=None).cuda().to(torch.bfloat16).eval()
params = sum(p.numel() for p in stack.parameters())
kv_per_tok = 2 * NL * D * 2
print(f"main params {params/1e6:.0f}M, KV/token {kv_per_tok/1e3:.0f} KB")
res = {"params": params, "kv_bytes_per_token": kv_per_tok, "points": []}
x = torch.randn(1, D, device="cuda", dtype=torch.bfloat16)
for n in [1_000, 4_000, 16_000, 36_000, 100_000]:
    try:
        run = StackRunner(stack, True)
        for c in run.caches:  # prefill with n random entries
            c.k = torch.randn(D // 64, n + 64, 64, device="cuda", dtype=torch.bfloat16)
            c.v = torch.randn_like(c.k); c.n = n
        with torch.no_grad():
            for _ in range(3):
                run.step(x, [0])
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20):
                run.step(x, [0])
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1e3
        pt = {"kv_tokens": n, "ms_per_step": ms, "kv_gb": n * kv_per_tok / 1e9, "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    except torch.OutOfMemoryError:
        pt = {"kv_tokens": n, "oom": True, "kv_gb": n * kv_per_tok / 1e9}
    print(pt); res["points"].append(pt)
    del run; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
json.dump(res, open("runs/bench_scale.json", "w"), indent=1)
