"""Streaming == full forward for the v2 model (tied router, time-ALiBi main network, per-frame memory read)."""
import sys, math, torch
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
sys.path.insert(0, ".")
from vt.env import simulate, Renderer, flat
from vt.model import VideoHNet
from vt.stream import StreamingVideoHNet
from eval import frame_masks

torch.manual_seed(0)
T = 1200
st = simulate(1, T, seed=4242)
frames, _ = Renderer()(flat(st, 0, 0, T), gen=torch.Generator(device="cuda").manual_seed(0))
act = torch.as_tensor(st["action"][0], device="cuda"); prev = torch.cat([torch.zeros_like(act[:1]), act[:-1]])
for kw in [dict(chunkers=("learned", "learned"), Ns=(4, 4), dims=(256, 384, 512), router="tied", alibi=True, mem_read=True),
           dict(chunkers=("learned", "learned"), Ns=(4, 4), dims=(256, 384, 512), router="tied", alibi=True, mem_read=True, mem_src="tok"),
           dict(chunkers=("all",), Ns=(1,), dims=(256, 512), alibi=True, mem_read=True)]:
    m = VideoHNet(**kw).cuda().eval()
    torch.nn.init.normal_(m.read.o.weight, std=0.02)
    net = m.body
    while hasattr(net, "inner"):  # stand-in calibration so the router fires on ~1/4 of frames
        if getattr(net, "router", "") == "tied":
            net.beta.data.fill_(2e-3); net.log_tau.data.fill_(math.log(1e-3))
        net = net.inner
    with torch.no_grad():
        h, infos = m(frames[None], prev[None])
        pol_full, _ = m.heads(h, act[None])
    masks = [x.cpu() for x in frame_masks(infos, T)]
    eng = StreamingVideoHNet(m)
    pols = torch.stack([eng.step(frames[t], prev[t], act[t])[0] for t in range(T)])
    for l, mk in enumerate(masks):
        sm = torch.zeros(T, dtype=torch.bool); sm[[t for (ll, t) in eng.events if ll == l]] = True
        print(kw["chunkers"], f"L{l+1}: full={int(mk.sum())} stream={int(sm.sum())} agree={(sm == mk).float().mean():.4f}")
    d = (pols - pol_full[0]).abs().max().item()
    print(kw["chunkers"], f"max |policy_stream - policy_full| = {d:.2e}")
    assert d < 1e-3
print("v2 streaming equivalence OK")
