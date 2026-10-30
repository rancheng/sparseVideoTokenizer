"""Streaming-safety check: outputs at time t must not depend on anything after t (frames or actions)."""
import sys, torch
sys.path.insert(0, ".")
from vt.model import VideoHNet

torch.manual_seed(0)
for chunkers, Ns, dims, kw in [(("learned", "learned"), (4, 4), (256, 384, 512), {}),
                               (("fixed", "fixed"), (4, 4), (256, 384, 512), {}), (("all",), (1,), (256, 512), {}),
                               (("learned", "learned"), (4, 4), (256, 384, 512), dict(router="tied", alibi=True, mem_read=True)),
                               (("learned", "learned"), (4, 4), (256, 384, 512), dict(router="tied", alibi=True, mem_read=True, mem_src="tok")),
                               (("all",), (1,), (256, 512), dict(alibi=True, mem_read=True))]:
    m = VideoHNet(chunkers=chunkers, Ns=Ns, dims=dims, **kw).cuda().eval()
    if kw.get("mem_read"):
        torch.nn.init.normal_(m.read.o.weight, std=0.02)  # zero-init would hide leaks through the read
    T, cut = 300, 170
    x = torch.randn(2, T, 256, device="cuda"); a = torch.randn(2, T, 3, device="cuda")
    with torch.no_grad():
        h1, _ = m(None, a, x_lat=x)
        x2, a2 = x.clone(), a.clone(); x2[:, cut:] += torch.randn_like(x2[:, cut:]); a2[:, cut:] = 5
        h2, _ = m(None, a2, x_lat=x2)
    past = (h1[:, :cut] - h2[:, :cut]).abs().max().item(); fut = (h1[:, cut:] - h2[:, cut:]).abs().max().item()
    print(chunkers, kw, f"max|dh| past={past:.2e} future={fut:.2e}")
    assert past < 1e-4 and fut > 1e-3
print("causality OK")
