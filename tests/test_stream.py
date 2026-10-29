"""Streaming engine == full-sequence forward (fp32), for every trained chunker type."""
import sys, torch
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
sys.path.insert(0, ".")
from vt.env import simulate, Renderer, flat
from vt.stream import StreamingVideoHNet
from eval import load, frame_masks

names = sys.argv[1:] or ["hnet_16x", "fixed_16x", "heur_16x", "iso_1x"]
T = 1500
st = simulate(1, T, seed=4242)
frames, _ = Renderer()(flat(st, 0, 0, T), gen=torch.Generator(device="cuda").manual_seed(0))
act = torch.as_tensor(st["action"][0], device="cuda")
prev = torch.cat([torch.zeros_like(act[:1]), act[:-1]])
for n in names:
    model, ck = load(n)
    model.float()
    with torch.no_grad():
        ext = None
        if ck["thresholds"]:
            sc = torch.cat([torch.tensor([1e9], device="cuda"), (frames[1:] - frames[:-1]).abs().mean((1, 2, 3))])
            ext = {i: (sc > th)[None] for i, th in enumerate(ck["thresholds"])}
        h, infos = model(frames[None], prev[None], ext=ext)
        pol_full, _ = model.heads(h, act[None])
    masks = [m.cpu() for m in frame_masks(infos, T)]
    eng = StreamingVideoHNet(model, ck["thresholds"])
    pols = torch.stack([eng.step(frames[t], prev[t], act[t])[0] for t in range(T)])
    for l, m in enumerate(masks):
        sm = torch.zeros(T, dtype=torch.bool); sm[[t for (ll, t) in eng.events if ll == l]] = True
        agree = (sm == m).float().mean().item()
        print(f"{n} L{l+1}: full={int(m.sum())} stream={int(sm.sum())} boundary agreement={agree:.4f}")
    d = (pols - pol_full[0]).abs().max().item()
    print(f"{n}: max |policy_stream - policy_full| = {d:.2e}")
