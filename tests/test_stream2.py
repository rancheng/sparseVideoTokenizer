"""The event-driven sparse engine (vt/stream2.py) must reproduce SparseVideoModel.forward frame by frame:
same cell events, same high-res cells, same temporal boundaries, same policy output."""
import sys, torch
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
from vt.env2 import simulate, Renderer, flat
from vt.model2 import SparseVideoModel
from vt.stream2 import StreamingSparseModel

torch.manual_seed(0)
T = 300
st = simulate(1, T, seed=5, delay=(20, 40))
frames, _ = Renderer()(flat(st, 0, 0, T))
act = torch.as_tensor(st["action"][0], device="cuda")
prev = torch.cat([torch.zeros_like(act[:1]), act[:-1]])
ok = True
for router, fovea in [("tied", "learned"), ("rel", "learned"), ("heur", "all"), ("dense", "none")]:
    torch.manual_seed(0)  # each config self-contained
    m = SparseVideoModel(dict(router=router, fovea=fovea), dict(router="tied", alibi=True, mem_read=True, mem_src="tok")).cuda().eval()
    with torch.no_grad():
        fr = m.front
        fr.beta.fill_(0.002); fr.log_tau.fill_(-7.0)                     # sparse-ish random router
        torch.nn.init.normal_(fr.fov[2].weight, std=0.5)                   # mixed fovea decisions
        if router == "rel":
            torch.nn.init.normal_(fr.rel[2].weight, std=0.5)
        fr.heur_th.fill_(0.01)
        torch.nn.init.normal_(m.temporal.read.o.weight, std=0.02)
        h, fi, ti, _ = m(frames[None], prev[None])
        pol_full, _ = m.temporal.heads(h, act[None])
        eng = StreamingSparseModel(m)
        pols = torch.stack([eng.step(frames[t], prev[t], act[t])[0] for t in range(T)])
    err = (pols - pol_full[0]).abs().max().item()
    s = eng.summary()
    ev_full = fi["b"][0].float().sum(-1).mean().item()
    print(f"{router:5s} {fovea:7s} events/frame full {ev_full:.2f} stream {s['events_per_frame']:.2f} | hires/frame full "
          f"{fi['use'][0].float().sum(-1).mean().item():.2f} stream {s['hires_per_frame']:.2f} | max|policy diff| {err:.2e} | "
          f"front FLOPs/frame {sum(v for k, v in s['flops_per_frame'].items() if k in ('cheap+router','fovea+hires','spatial','pool'))/1e6:.2f}M "
          f"(dense {s['dense_front_flops_per_frame']/1e6:.1f}M)")
    ok &= err < 1e-4 and abs(ev_full - s["events_per_frame"]) < 1e-6
print("sparse streaming equivalence", "OK" if ok else "FAILED")
