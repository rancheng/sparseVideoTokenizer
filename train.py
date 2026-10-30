"""Train a (hierarchical) causal video tokenizer + policy/forward heads on the synthetic robot stream."""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from vt.env import simulate, ClipSampler, EV_QUERY
from vt.model import VideoHNet

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--chunkers", default="learned,learned")
p.add_argument("--Ns", default="4,4")
p.add_argument("--dims", default="256,384,512")
p.add_argument("--n_main", type=int, default=8)
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--B", type=int, default=4)
p.add_argument("--T", type=int, default=2048)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--alpha", type=float, default=0.03)
p.add_argument("--w_pix", type=float, default=100.0)
p.add_argument("--no_actions", action="store_true", help="tokenizer does not see a_{t-1}")
p.add_argument("--detach_policy", action="store_true", help="policy head is a probe: no action gradient into tokenizer")
p.add_argument("--max_gap", type=int, default=1500)
p.add_argument("--min_gap", type=int, default=100)
p.add_argument("--curriculum", default="", help="memory-gap curriculum 'step:min:max,...' e.g. '0:5:60,1500:30:400'")
p.add_argument("--cue_rate", type=float, default=1 / 150, help="denser memory episodes than test (1/600)")
p.add_argument("--w_query", type=float, default=10.0, help="loss weight on query-onset (decision) frames")
p.add_argument("--init", default=None)
p.add_argument("--warmup", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_dec", type=int, default=1536)
p.add_argument("--heartbeats", default="64,256")
p.add_argument("--ratio_mode", default="target", choices=["target", "cap"])
p.add_argument("--act_drop", type=float, default=0.0, help="drop a_{t-1} input per frame (anti-copycat)")
p.add_argument("--router", default="cos", choices=["cos", "tied"])
p.add_argument("--alibi", action="store_true", help="time-recency bias in the main network")
p.add_argument("--mem_read", action="store_true", help="per-frame cross-attention into main-network memory")
p.add_argument("--mem_src", default="out", choices=["out", "tok"], help="memory entries: main outputs, or event token + output")
p.add_argument("--freeze_fenc", action="store_true", help="frozen frame encoder (latents without activations): long clips fit")
p.add_argument("--router_detach", action="store_true", help="no gradient from the router into the encoder state")
p.add_argument("--mask_unanswerable", action="store_true", help="drop query frames whose cue is outside the clip")
p.add_argument("--rate_w", type=float, default=0.0, help="per-token rate cost (cap mode)")
p.add_argument("--router_lr", type=float, default=0.3, help="lr multiplier for router wq/wk")
args = p.parse_args()

torch.manual_seed(args.seed)
out_dir = f"runs/{args.name}"; os.makedirs(out_dir, exist_ok=True)
chunkers = args.chunkers.split(","); Ns = [int(n) for n in args.Ns.split(",")]
dims = [int(d) for d in args.dims.split(",")][: len(chunkers) + 1]
cfg = dict(chunkers=chunkers, Ns=Ns, dims=dims, n_main=args.n_main, use_actions=not args.no_actions,
           heartbeats=[int(h) for h in args.heartbeats.split(",")], ratio_mode=args.ratio_mode, rate_w=args.rate_w,
           router=args.router, alibi=args.alibi, mem_read=args.mem_read, mem_src=args.mem_src, router_detach=args.router_detach)
model = VideoHNet(**cfg).cuda()
if args.init:
    model.load_state_dict(torch.load(args.init)["model"])
if args.freeze_fenc:
    model.fenc.requires_grad_(False)
print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", cfg)

def make_sampler(gmin, gmax, k=0):
    st = simulate(64, 60000, seed=1000 + args.seed + 7919 * k, gap=(gmin, gmax), cue_rate=args.cue_rate)
    return ClipSampler(st, args.T, seed=args.seed + k)


curr = [tuple(int(v) for v in c.split(":")) for c in args.curriculum.split(",")] if args.curriculum else [(0, args.min_gap, args.max_gap)]
sampler = make_sampler(*curr[0][1:])


def pixel_score(frames):  # mean |o_t - o_{t-1}| : the "keyframe / entropy" style heuristic signal (causal)
    d = (frames[:, 1:] - frames[:, :-1]).abs().mean((2, 3, 4))
    return torch.cat([torch.full_like(d[:, :1], 1e9), d], 1)


thresholds = None
if "external" in chunkers:  # calibrate global thresholds to hit the target compression on average
    sc = torch.cat([pixel_score(sampler.sample(4)[0][:, :-1]).flatten() for _ in range(8)])
    cum = 1
    thresholds = []
    for N in Ns:
        cum *= N
        thresholds.append(torch.quantile(sc[sc < 1e8].float(), 1 - 1 / cum).item())
    print("heuristic thresholds", thresholds)

rp = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and n.endswith(("wq.weight", "wk.weight"))]
op = [p_ for n, p_ in model.named_parameters() if p_.requires_grad and not n.endswith(("wq.weight", "wk.weight"))]
opt = torch.optim.AdamW([{"params": op}, {"params": rp, "lr": args.lr * args.router_lr}], lr=args.lr,
                        betas=(0.9, 0.95), weight_decay=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, *_: min(1, (s + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps)))))


def make_ext(frames):
    if thresholds is None:
        return None
    sc = pixel_score(frames)
    return {i: sc > th for i, th in enumerate(thresholds)}


def step_loss(model, frames, act, ext, ev):
    T = frames.shape[1] - 1
    obs, nxt = frames[:, :T], frames[:, 1:]
    prev_act = torch.cat([torch.zeros_like(act[:, :1]), act[:, : T - 1]], 1)
    if args.act_drop > 0:
        prev_act = prev_act * (torch.rand_like(prev_act[..., :1]) > args.act_drop)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if args.freeze_fenc:
            with torch.no_grad():
                x_lat = model.encode_frames(obs)
            h, infos = model(None, prev_act, ext=ext, x_lat=x_lat)
        else:
            h, infos = model(obs, prev_act, ext=ext)
        hp = h.detach() if args.detach_policy else h
        pol, _ = model.heads(hp, act[:, :T])
        _, z = model.heads(h, act[:, :T])
        zf = z.reshape(-1, z.shape[-1])
        sel = torch.randperm(zf.shape[0], device=zf.device)[: args.n_dec]  # decode a random subset of frames (memory)
        pred = model.fdec(zf[sel])
    mse = F.mse_loss(pred.float(), nxt.reshape(-1, *nxt.shape[2:])[sel])
    wq = 1 + (args.w_query - 1) * ((ev[:, :T] & EV_QUERY) > 0).float()
    if args.mask_unanswerable:
        wq = torch.where(sampler.last_qbad[:, :T], torch.zeros_like(wq), wq)
    qok = sampler.last_qok[:, :T]
    mem_acc = None
    if qok.any():  # running memory accuracy on answerable query frames (direction sign agreement)
        mem_acc = ((pol[..., :2].float().sign() == act[:, :T, :2].sign()).all(-1)[qok]).float().mean().item()
    a_mse = (wq[..., None] * (pol[..., :2].float() - act[:, :T, :2]) ** 2).mean()
    g_bce = F.binary_cross_entropy_with_logits(pol[..., 2].float(), act[:, :T, 2])
    ratio = sum(i["ratio_loss"] for i in infos if i["ratio_loss"] is not None) if any(i["ratio_loss"] is not None for i in infos) else torch.zeros((), device=h.device)
    loss = args.w_pix * mse + a_mse + 0.1 * g_bce + args.alpha * ratio
    comp = [(infos[0]["valid"].float().sum() / i["b"].float().sum()).item() for i in infos]  # cumulative frames/token
    return loss, dict(loss=loss.item(), mse=mse.item(), a_mse=a_mse.item(), g_bce=g_bce.item(), ratio=float(ratio), comp=comp,
                      mem_acc=mem_acc)


log = []
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
for s in range(args.steps):
    for k, (cs, gmin, gmax) in enumerate(curr):
        if k and s == cs:
            sampler = make_sampler(gmin, gmax, k); print(f"curriculum: step {s} -> gaps {gmin}-{gmax}", flush=True)
    frames, clean, act, ev = sampler.sample(args.B)
    ext = make_ext(frames[:, :-1])
    loss, m = step_loss(model, frames, act, ext, ev)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if s % 50 == 0 or s == args.steps - 1:
        m.update(step=s, time=time.time() - t0, gn=gn.item(), mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        log.append(m)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
torch.save({"model": model.state_dict(), "cfg": cfg, "thresholds": thresholds, "args": vars(args)}, f"{out_dir}/ckpt.pt")
json.dump(log, open(f"{out_dir}/train_log.json", "w"))
print("done", time.time() - t0)
