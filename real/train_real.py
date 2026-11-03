"""Train the sparse spatio-temporal tokenizer (vt/model2.py) on real robot video (LeRobot datasets, real/prep_lerobot.py).

Losses: per-cell forward model (o_{t+1} from the temporal latent, the action and the cell state; beta-NLL, half the
sampled cells are cells that change), behaviour cloning of the normalised action, H-Net ratio losses, and the
task-credit fovea (high resolution goes where it lowers the action loss). Clips are cut from each dataset's training
episodes laid end to end, so a clip can span an episode reset, as a robot's stream would.
"""
import argparse, json, math, os, queue, threading, time
import numpy as np
import torch
import torch.nn.functional as F
from vt.model2 import SparseVideoModel, to_cells
from vt.env2 import G

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--data", default="/home/ubuntu/data/robot_video/prep")
p.add_argument("--sets", default="all", help="comma-separated dataset tags, or 'all' / 'fixed' (fixed cameras only)")
p.add_argument("--router", default="tied", choices=["tied", "rel", "heur", "dense"])
p.add_argument("--fovea", default="learned", choices=["learned", "all", "none"])
p.add_argument("--size", default="base", choices=["small", "base", "large"])
p.add_argument("--N_cell", type=int, default=16)
p.add_argument("--k_fov", type=int, default=2)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--B", type=int, default=8)
p.add_argument("--T", type=int, default=256)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=300)
p.add_argument("--alpha", type=float, default=0.03)
p.add_argument("--alpha_s", type=float, default=0.05)
p.add_argument("--w_pix", type=float, default=1.0)
p.add_argument("--w_act", type=float, default=1.0)
p.add_argument("--w_fov", type=float, default=0.1)
p.add_argument("--n_dec", type=int, default=2048)
p.add_argument("--act_drop", type=float, default=0.5, help="drop the previous action input (vision has to carry the policy)")
p.add_argument("--fov_warm", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

SIZES = {"small": dict(d=96, dims=(192, 256, 384), n_main=4), "base": dict(d=128, dims=(256, 384, 512), n_main=8),
         "large": dict(d=192, dims=(384, 512, 768), n_main=12)}
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
out_dir = f"runs_real/{args.name}"; os.makedirs(out_dir, exist_ok=True)

tags = sorted(d for d in os.listdir(args.data) if os.path.exists(f"{args.data}/{d}/meta.json"))
if args.sets == "fixed":
    tags = [t for t in tags if "__" not in t]
elif args.sets != "all":
    tags = args.sets.split(",")
ALL = sorted(d for d in os.listdir(args.data) if os.path.exists(f"{args.data}/{d}/meta.json"))  # embedding ids stay fixed
sets = []
for t in tags:
    m = json.load(open(f"{args.data}/{t}/meta.json"))
    end = m["starts"][m["test_from"]]
    sets.append(dict(tag=t, dom=ALL.index(t), meta=m, end=end, frames=np.load(f"{args.data}/{t}/frames.npy", mmap_mode="r"),
                     acts=np.load(f"{args.data}/{t}/actions.npy"), adim=m["act_dim"]))
print("datasets", [(s["tag"], s["end"]) for s in sets], flush=True)
A = sets[0]["acts"].shape[1]

sz = SIZES[args.size]
front = dict(d=sz["d"], dout=sz["dims"][0], router=args.router, fovea=args.fovea, N_cell=args.N_cell, k_fov=args.k_fov,
             fov_credit="task", fov_pos=True)
temporal = dict(chunkers=["learned", "learned"], Ns=[4, 4], dims=list(sz["dims"]), n_main=sz["n_main"], router="tied",
                alibi=True, mem_read=True, mem_src="tok", act_dim=A)
cfg = dict(front=front, temporal=temporal, n_dom=len(ALL), doms=ALL)
model = SparseVideoModel(front, temporal, n_dom=len(ALL)).cuda()
print(f"params {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", cfg, flush=True)

rp = [q for n, q in model.named_parameters() if n.endswith(("wq.weight", "wk.weight"))]
sp = [q for n, q in model.named_parameters() if n.endswith(("beta", "log_tau"))]
op = [q for n, q in model.named_parameters() if not n.endswith(("wq.weight", "wk.weight", "beta", "log_tau"))]
opt = torch.optim.AdamW([{"params": op}, {"params": rp, "lr": args.lr * 0.3}, {"params": sp, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, *_: min(1, (s + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps)))))


def sample():
    """B clips of T+1 frames; datasets drawn uniformly (every robot counts the same)"""
    fr = np.empty((args.B, args.T + 1, 128, 128, 3), np.uint8)
    ac = np.zeros((args.B, args.T + 1, A), np.float32); mk = np.zeros((args.B, A), np.float32); dom = np.zeros(args.B, np.int64)
    for i in range(args.B):
        s = sets[rng.integers(len(sets))]
        t0 = rng.integers(0, s["end"] - args.T - 1)
        fr[i] = s["frames"][t0:t0 + args.T + 1]; ac[i] = s["acts"][t0:t0 + args.T + 1]
        mk[i, :s["adim"]] = 1; dom[i] = s["dom"]
    return fr, ac, mk, dom


q = queue.Queue(maxsize=6)
def loader():
    while True:
        q.put([torch.from_numpy(x).pin_memory() for x in sample()])
for _ in range(3):
    threading.Thread(target=loader, daemon=True).start()


def hnet_ratio(Fr, Gm, N):
    return N / (N - 1) * ((N - 1) * Fr * Gm + (1 - Fr) * (1 - Gm))


def step_loss(fr, act, mask, dom):
    B, T = fr.shape[0], fr.shape[1] - 1
    frames = fr.permute(0, 1, 4, 2, 3).float().div_(255)
    obs = frames[:, :T]
    prev = torch.cat([torch.zeros_like(act[:, :1]), act[:, :T - 1]], 1)
    prev = prev * (torch.rand_like(prev[..., :1]) > args.act_drop)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, fi, ti, _ = model(obs, prev, dom=dom)
        pol, z = model.temporal.heads(h, act[:, :T])
        cc, cn = to_cells(frames[:, :T]), to_cells(frames[:, 1:])
        chg = (cn - cc).abs().mean((-1, -2, -3)) > 0.02
        n = args.n_dec // 2
        fc = chg.flatten().nonzero()[:, 0]
        pick = torch.cat([fc[torch.randint(0, len(fc), (n,), device=h.device)] if len(fc) else fc,
                          torch.randint(0, B * T * G * G, (n,), device=h.device)])
        bi, tt, ji = pick // (T * G * G), (pick // (G * G)) % T, pick % (G * G)
        pred, logv = model.cdec(z[bi, tt], fi["cell"][bi, tt, ji], ji, cc[bi, tt, ji], with_var=True)
    tgt = cn[bi, tt, ji]
    err = ((pred.float() - tgt) ** 2).mean((1, 2, 3))
    nll = (torch.exp(logv.detach()) ** 0.5 * 0.5 * (err * torch.exp(-logv) + logv)).mean()
    a_err = (((pol.float() - act[:, :T]) ** 2) * mask[:, None]).sum(-1) / mask.sum(-1)[:, None]
    a_mse = a_err.mean()
    ratio_t = sum(i["ratio_loss"] for i in ti if i["ratio_loss"] is not None)
    ratio_s = torch.zeros((), device=h.device)
    if args.router in ("tied", "rel"):
        ratio_s = hnet_ratio(fi["b"].float().mean(), fi["p"].float().mean(), args.N_cell)
    loss = args.w_pix * nll + args.w_act * a_mse + args.alpha * ratio_t + args.alpha_s * ratio_s
    fov_loss = torch.zeros((), device=h.device)
    if args.fovea == "learned" and "fov_gate" in fi and fi["fov_gate"].requires_grad:
        gr, = torch.autograd.grad(a_mse, fi["fov_gate"], retain_graph=True, allow_unused=True)
        ben = -gr.float() if gr is not None else torch.zeros_like(fi["fov_gate"])
        evm = fi["b"]
        ben = (ben / (ben[evm].pow(2).mean().sqrt() + 1e-12)).clamp(-5, 5)
        fov_loss = (fi["fov_v"].float() - ben)[evm].pow(2).mean()
        loss = loss + args.w_fov * fov_loss
    with torch.no_grad():
        mse = err.mean().item(); copy = ((cc[bi, tt, ji] - tgt) ** 2).mean().item()
    return loss, dict(loss=loss.item(), nll=nll.item(), mse=mse, mse_copy=copy, a_mse=a_mse.item(), ratio_t=float(ratio_t),
                      ratio_s=float(ratio_s), fov_loss=float(fov_loss), ev_rate=fi["b"].float().mean().item(),
                      hires_rate=fi["use"].float().mean().item(),
                      comp=[(ti[0]["valid"].float().sum() / i["b"].float().sum()).item() for i in ti])


log, t0 = [], time.time()
torch.cuda.reset_peak_memory_stats()
for s in range(args.steps):
    model.front.force_fovea = s < args.fov_warm
    fr, ac, mk, dom = [x.cuda(non_blocking=True) for x in q.get()]
    loss, m = step_loss(fr, ac, mk, dom)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if s % 50 == 0 or s == args.steps - 1:
        m.update(step=s, time=time.time() - t0, gn=gn.item(), mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        log.append(m)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
torch.save({"model": model.state_dict(), "cfg": cfg, "args": vars(args)}, f"{out_dir}/ckpt.pt")
json.dump(log, open(f"{out_dir}/train_log.json", "w"))
print("done", time.time() - t0)
