"""Train the sparse spatio-temporal tokenizer (vt/model2.py) on the workcell v2 stream (vt/env2.py)."""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from vt.env2 import simulate, ClipSampler, G, CARRY, BINS, EV_EXIT
from vt.model2 import SparseVideoModel, to_cells
from vt.jepa import JEPAHeads, HORIZONS

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--router", default="tied", choices=["tied", "rel", "heur", "dense"], help="per-cell temporal router")
p.add_argument("--fovea", default="learned", choices=["learned", "all", "none"])
p.add_argument("--N_cell", type=int, default=16, help="target: 1 in N_cell cells updates per frame")
p.add_argument("--k_max", type=int, default=12, help="hard cap on routed cell events per frame")
p.add_argument("--k_fov", type=int, default=2, help="learned fovea: high-resolution cells per frame (top-k of events)")
p.add_argument("--chunkers", default="learned,learned")
p.add_argument("--Ns", default="4,4")
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--B", type=int, default=2)
p.add_argument("--T", type=int, default=2048)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=300)
p.add_argument("--alpha", type=float, default=0.03, help="temporal ratio loss weight")
p.add_argument("--fov_credit", default="ste", choices=["ste", "task"], help="fovea score: straight-through on picked "
               "cells from every loss (ste), or regress the counterfactual action-loss benefit on every event cell (task)")
p.add_argument("--fov_pos", action="store_true", help="fovea score also sees the cell position")
p.add_argument("--fov_explore", type=float, default=1.0, help="task credit: Gumbel temperature of the training-time top-k")
p.add_argument("--w_fov", type=float, default=0.1, help="task credit: weight of the fovea value regression")
p.add_argument("--alpha_s", type=float, default=0.05, help="spatial (cell router, fovea) ratio loss weight")
p.add_argument("--w_pix", type=float, default=1.0, help="weight on the per-cell forward-model NLL")
p.add_argument("--w_dec", type=float, default=1.0, help="weight of the decision-frame action loss (own mean)")
p.add_argument("--n_dec", type=int, default=2048, help="cells decoded per step (half of them changing cells)")
p.add_argument("--pop_exit", action="store_true", help="env: items pop out of the tunnel in one frame (memory-only exit)")
p.add_argument("--rule_mode", default="mirror", choices=["mirror", "pair"], help="env: glyph x rule -> bin mapping")
p.add_argument("--w_main", type=float, default=0.0, help="weight of the main network's next-event-token loss (0 = off)")
p.add_argument("--main_detach", action="store_true", help="main loss trains the main network only (detached tokens)")
p.add_argument("--n_exit", type=int, default=0, help="extra decoded cells per step at tunnel-exit frames (exit cells)")
p.add_argument("--delay", default="150,600", help="tunnel transit time range (frames)")
p.add_argument("--n_env", type=int, default=64)
p.add_argument("--act_drop", type=float, default=0.3)
p.add_argument("--init", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--focus", type=float, default=0.0, help="fraction of clips centred on a decision frame")
p.add_argument("--fov_warm", type=int, default=0, help="steps with full resolution on every event cell before the "
               "learned top-k fovea takes over")
p.add_argument("--focus_curr", type=float, default=None, help="decision focus during the fixed-rule stage")
p.add_argument("--stages", default="", help="curriculum 'step:T:B:focus:rule,...', rule in fixed/short/normal; "
               "overrides T, B, focus and rule_curr (e.g. glyph on short decision clips, then rule memory on longer clips)")
p.add_argument("--freeze_front", action="store_true", help="frozen sparse front, run in chunks without activations: "
               "the temporal model trains on long clips (rule flashes and exits far apart inside one clip)")
p.add_argument("--chunk", type=int, default=2048, help="frozen front: frames rendered and encoded per piece")
p.add_argument("--obj", default="pix", choices=["pix", "jepa"], help="pix: per-cell next-frame decoder; jepa: action-"
               "conditioned latent prediction at several horizons + inverse dynamics + future actions + surprise router")
p.add_argument("--w_jepa", type=float, default=1.0)
p.add_argument("--w_inv", type=float, default=0.5)
p.add_argument("--w_fut", type=float, default=0.5)
p.add_argument("--w_surp", type=float, default=0.1, help="cell router: BCE towards the cells with the largest one-step latent error")
p.add_argument("--n_tf", type=int, default=32, help="jepa: target frames per clip")
p.add_argument("--rule_curr", type=int, default=0, help="first N steps with a fixed rule (glyph only), then rule flashes")
args = p.parse_args()

torch.manual_seed(args.seed)
out_dir = f"runs2/{args.name}"; os.makedirs(out_dir, exist_ok=True)
chunkers = args.chunkers.split(","); Ns = [int(n) for n in args.Ns.split(",")]
front = dict(router=args.router, fovea=args.fovea, N_cell=args.N_cell, k_fov=args.k_fov, k_max=args.k_max,
             fov_credit=args.fov_credit, fov_pos=args.fov_pos, fov_explore=args.fov_explore)
temporal = dict(chunkers=chunkers, Ns=Ns, dims=[256, 384, 512][: len(chunkers) + 1] if chunkers != ["all"] else [256, 512],
                router="tied", alibi=True, mem_read=True, mem_src="tok", main_pred=args.w_main > 0, main_pred_detach=args.main_detach)
cfg = dict(front=front, temporal=temporal)
model = SparseVideoModel(front, temporal).cuda()
if args.init:
    model.load_state_dict(torch.load(args.init)["model"])
if args.freeze_front:
    model.front.requires_grad_(False); model.front.eval()
jm = JEPAHeads(model.front, temporal["dims"][0], 3).cuda() if args.obj == "jepa" else None
print(f"params {sum(q.numel() for q in model.parameters())/1e6:.1f}M", cfg, flush=True)

delay = tuple(int(v) for v in args.delay.split(","))
st = simulate(args.n_env, 30000, seed=1000 + args.seed, delay=delay, pop_exit=args.pop_exit, rule_mode=args.rule_mode)
sampler = ClipSampler(st, args.T, seed=args.seed)
main_sampler = sampler
if args.rule_curr:
    sampler = ClipSampler(simulate(args.n_env, 30000, seed=2000 + args.seed, delay=delay, rule_fixed=0, pop_exit=args.pop_exit, rule_mode=args.rule_mode), args.T, seed=args.seed, rule_fixed=True)
BINS_T = torch.as_tensor(BINS, device="cuda")

rp = [q for n, q in model.named_parameters() if n.endswith(("wq.weight", "wk.weight"))]
op = [q for n, q in model.named_parameters() if not n.endswith(("wq.weight", "wk.weight")) and not n.endswith(("beta", "log_tau"))]
if jm is not None:
    op += [q for q in jm.parameters() if q.requires_grad]
sp = [q for n, q in model.named_parameters() if n.endswith(("beta", "log_tau"))]  # no weight decay on thresholds
opt = torch.optim.AdamW([{"params": op}, {"params": rp, "lr": args.lr * 0.3}, {"params": sp, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, *_: min(1, (s + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps)))))


def hnet_ratio(Fr, Gm, N):
    return N / (N - 1) * ((N - 1) * Fr * Gm + (1 - Fr) * (1 - Gm))


@torch.no_grad()
def front_long(B, T, ev):
    """frozen front over a long clip, piece by piece (state carried): pooled frame vectors for the temporal model, the
    event/fovea masks, and the cells picked for the forward model (same mix as below: changing, uniform, exits)"""
    fs, bs, us, picks, state = [], [], [], [], None
    for a in range(0, T, args.chunk):
        e = min(T, a + args.chunk); L = e - a
        fr, cl = sampler.render_range(a, e + 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f, fi, state = model.front(fr[:, :L], t0=a, state=state)
        fs.append(f); bs.append(fi["b"]); us.append(fi["use"])
        cc, cn, co = to_cells(cl[:, :L]), to_cells(cl[:, 1:]), to_cells(fr[:, :L])
        chg = ((cn - cc).abs().mean((-1, -2, -3)) > 0.01).flatten().nonzero()[:, 0]
        n = max(1, args.n_dec // 2 * L // T)
        pk = [torch.randint(0, B * L * G * G, (n,), device="cuda")]
        if len(chg):
            pk.append(chg[torch.randint(0, len(chg), (n,), device="cuda")])
        if args.n_exit:
            ex = ((ev[:, a:e] & EV_EXIT) > 0).flatten().nonzero()[:, 0]
            if len(ex):
                ne = max(1, args.n_exit * L // T)
                pe = ex[torch.randint(0, len(ex), (ne,), device="cuda")]
                pk.append(pe * (G * G) + torch.tensor([6, G + 6], device="cuda")[torch.randint(0, 2, (ne,), device="cuda")])
        pick = torch.cat(pk)
        bi, tl, ji = pick // (L * G * G), (pick // (G * G)) % L, pick % (G * G)
        picks.append((bi, tl + a, ji, fi["cell"][bi, tl, ji], co[bi, tl, ji], cn[bi, tl, ji]))
    cat = lambda i: torch.cat([q[i] for q in picks])
    return torch.cat(fs, 1), dict(b=torch.cat(bs, 1), use=torch.cat(us, 1)), [cat(i) for i in range(6)]


def jepa_loss(frames, x, h, fi, act):
    """frames (B,T+1,...) observed frames, x (B,T,df) front frame vectors, h (B,T,dz) temporal output"""
    B, T = h.shape[:2]; J = G * G; dev = h.device
    tf = torch.randint(1, T + 1, (B, args.n_tf), device=dev)                     # frames to predict
    bix = torch.arange(B, device=dev)[:, None].expand_as(tf)
    Y = jm.targets(frames[bix, tf])                                               # (B,n,J,dc)
    S = torch.cat([torch.zeros_like(act[:, :1]), act.cumsum(1)], 1)              # S[t] = sum a_<t
    losses, err1 = {}, None
    for hi, k in enumerate(HORIZONS):
        t = tf - k; ok = t >= 0; tc = t.clamp(min=0)
        af = torch.cat([S[bix, tf] - S[bix, tc], act[bix, tc]], -1)              # [sum a_t..a_{t'-1}, a_t]
        n = B * args.n_tf
        hh = h[bix, tc].reshape(n, 1, -1).expand(-1, J, -1).reshape(n * J, -1)
        jj = torch.arange(J, device=dev).repeat(n)
        cell = fi["cell"][bix, tc].reshape(n * J, -1) if k == 1 else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = jm.predict(hh, jj, torch.full_like(jj, hi), af.reshape(n, 1, -1).expand(-1, J, -1).reshape(n * J, -1).to(hh.dtype), cell)
        e = F.smooth_l1_loss(pr.float(), Y.reshape(n * J, -1), reduction="none").mean(-1).view(B, args.n_tf, J)
        losses[k] = (e * ok[..., None]).sum() / (ok.sum() * J).clamp(min=1)
        if k == 1:
            err1 = torch.where(ok[..., None], e.detach(), torch.zeros_like(e))
    l_jepa = sum(losses.values()) / len(losses)
    # surprise router: the top 1/N_cell cells by one-step latent error are the cells that should have been events
    l_surp = torch.zeros((), device=dev)
    if args.router in ("tied", "rel"):
        th = torch.quantile(err1.flatten().float(), 1 - 1 / args.N_cell)
        p_ = fi["p"][bix, tf.clamp(max=T - 1)].float().clamp(1e-4, 1 - 1e-4)
        l_surp = F.binary_cross_entropy(p_, (err1 >= th).float())
    # inverse dynamics on action-free frame vectors, and future actions from h
    k = torch.randint(1, 17, (B, args.n_tf), device=dev); t0 = torch.randint(0, T, (B, args.n_tf), device=dev)
    t1 = (t0 + k).clamp(max=T - 1); k = t1 - t0; okk = k > 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inv = jm.inv(torch.cat([x[bix, t0], x[bix, t1], jm.hz((k > 3).long() + (k > 15).long())], -1))   # gap bucket
    l_inv = (((inv.float() - act[bix, t0]) ** 2).mean(-1) * okk).sum() / okk.sum().clamp(min=1)
    kf = torch.tensor([4, 16, 64], device=dev)[torch.randint(0, 3, (B, args.n_tf), device=dev)]
    tt = torch.randint(0, T, (B, args.n_tf), device=dev); ok2 = tt + kf < T
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fu = jm.fut(torch.cat([h[bix, tt], jm.hz(1 + (kf > 4).long() + (kf > 16).long())], -1))
    l_fut = (((fu.float() - act[bix, (tt + kf).clamp(max=T - 1)]) ** 2).mean(-1) * ok2).sum() / ok2.sum().clamp(min=1)
    with torch.no_grad():
        ystd = Y.float().std(dim=(0, 1, 2)).mean().item()
    return l_jepa, l_surp, l_inv, l_fut, {f"j{k}": v.item() for k, v in losses.items()} | {"y_std": ystd}


def step_loss(frames, clean, act, ev):
    B, T = act.shape[0], act.shape[1] - 1
    prev_act = torch.cat([torch.zeros_like(act[:, :1]), act[:, : T - 1]], 1)
    if args.act_drop > 0:
        prev_act = prev_act * (torch.rand_like(prev_act[..., :1]) > args.act_drop)
    if args.freeze_front:
        f, fi, (bi, ti_, ji, cel, cur, tgt) = front_long(B, T, ev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, ti = model.temporal(None, prev_act, x_lat=f)
            pol, z = model.temporal.heads(h, act[:, :T])
            pred, logv = model.cdec(z[bi, ti_], cel, ji, cur, with_var=True)
    else:
      obs = frames[:, :T]
      with torch.autocast("cuda", dtype=torch.bfloat16):
        x_f, fi, _ = model.front(obs)
        h, ti = model.temporal(None, prev_act, x_lat=x_f)
        pol, z = model.temporal.heads(h, act[:, :T])
        # forward model on sampled cells: half among cells that change at t+1, half uniform
        cc, cn = to_cells(clean[:, :T]), to_cells(clean[:, 1:])
        chg = (cn - cc).abs().mean((-1, -2, -3)) > 0.01                   # (B,T,J)
        n = args.n_dec // 2
        flat_chg = chg.flatten().nonzero()[:, 0]
        pick_c = flat_chg[torch.randint(0, max(1, len(flat_chg)), (n,), device=h.device)] if len(flat_chg) else torch.zeros(0, dtype=torch.long, device=h.device)
        pick_u = torch.randint(0, B * T * G * G, (n,), device=h.device)
        pick = torch.cat([pick_c, pick_u])
        if args.n_exit:  # the exit frame: the two cells the item pops into
            ex = ((ev[:, :T] & EV_EXIT) > 0).flatten().nonzero()[:, 0]
            if len(ex):
                pe = ex[torch.randint(0, len(ex), (args.n_exit,), device=h.device)]
                je = torch.tensor([6, G + 6], device=h.device)[torch.randint(0, 2, (args.n_exit,), device=h.device)]
                pick = torch.cat([pick, pe * (G * G) + je])
        bi, ti_, ji = pick // (T * G * G), (pick // (G * G)) % T, pick % (G * G)
        cur = to_cells(obs)[bi, ti_, ji]
        pred, logv = model.cdec(z[bi, ti_], fi["cell"][bi, ti_, ji], ji, cur, with_var=True)
      tgt = cn[bi, ti_, ji]
    err = ((pred.float() - tgt) ** 2).mean((1, 2, 3))
    mse = err.mean()
    # heteroscedastic Gaussian NLL per cell: unpredictable cells (noisy TV) get a large variance and stop dominating
    # beta-NLL (beta=0.5, Seitzer et al. 2022): weighting by the detached variance stops the model from giving up on
    # hard-but-predictable cells (tunnel exits) by declaring them noisy
    nll = (torch.exp(logv.detach()) ** 0.5 * 0.5 * (err * torch.exp(-logv) + logv)).mean()
    # policy: carry frames are label noise when the rule flash is outside the clip
    phase = sampler.last_phase[:, :T]; carry = phase == CARRY
    ok = ~carry | sampler.last_rule_in[:, :T]
    dec = sampler.last_decision[:, :T] & sampler.last_rule_in[:, :T]
    a_err = ((pol[..., :2].float() - act[:, :T, :2]) ** 2).sum(-1)
    a_mse = (ok.float() * a_err).mean()
    # decisions get their own normalised term: averaged into ~4000 frames per clip they would carry no signal
    # heading classification on moving frames (own term on decisions)
    av = act[:, :T, :2]; moving = av.norm(dim=-1) > 0.2
    nb = model.n_ang
    ang_t = ((torch.atan2(av[..., 1], av[..., 0]) + torch.pi) / (2 * torch.pi) * nb).long().clamp(0, nb - 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.ang(h).float()
    ce = F.cross_entropy(logits.reshape(-1, nb), ang_t.reshape(-1), reduction="none").view(B, T)
    m_ok = moving & ok
    a_ang = ce[m_ok].mean() if m_ok.any() else torch.zeros((), device=h.device)
    a_dec = ce[dec].mean() if dec.any() else torch.zeros((), device=h.device)
    g_bce = F.binary_cross_entropy_with_logits(pol[..., 2].float(), act[:, :T, 2])
    # ratio losses: temporal stages + spatial cell router + fovea
    ratio_t = sum(i["ratio_loss"] for i in ti if i["ratio_loss"] is not None) if any(i["ratio_loss"] is not None for i in ti) else torch.zeros((), device=h.device)
    ratio_s = torch.zeros((), device=h.device)
    b, pr = fi["b"], fi.get("p")
    if args.router in ("tied", "rel") and not args.freeze_front:
        ratio_s = ratio_s + hnet_ratio(b.float().mean(), pr.float().mean(), args.N_cell)
    if args.fovea == "learned" and args.fov_credit == "ste" and not args.freeze_front and fi["use"].any() and not model.front.force_fovea:
        ratio_s = ratio_s + fi["gate_soft"][fi["use"]].float().mean()  # small pull on selected scores: useless picks rotate out
    task = a_mse + 0.1 * a_ang + args.w_dec * 0.1 * a_dec + 0.1 * g_bce
    jl = {}
    if jm is not None:
        l_jepa, l_surp, l_inv, l_fut, jl = jepa_loss(frames, x_f, h, fi, act[:, :T])
        task = task + args.w_inv * l_inv + args.w_fut * l_fut     # what the fovea is credited for: behaviour, not pixels
        jl.update(l_jepa=l_jepa.item(), l_surp=l_surp.item(), l_inv=l_inv.item(), l_fut=l_fut.item())
    loss = args.w_pix * nll + task + args.alpha * ratio_t + args.alpha_s * ratio_s
    if jm is not None:
        loss = loss + args.w_jepa * l_jepa + args.w_surp * l_surp
    main_l = model.temporal.main.pred_loss if args.w_main > 0 else None
    if main_l is not None:
        loss = loss + args.w_main * main_l
    fov_loss = torch.zeros((), device=h.device)
    if args.fovea == "learned" and args.fov_credit == "task" and not args.freeze_front:
        # value of high resolution per event cell = -d(action losses)/d(gate), first order, pixels excluded: the noisy TV
        # helps next-frame pixels but not actions. RMS-normalised per batch; the score regresses its conditional mean
        gate, ev_m = fi["fov_gate"], b
        gr, = torch.autograd.grad(task, gate, retain_graph=True, allow_unused=True)
        ben = -gr.float() if gr is not None else torch.zeros_like(gate)
        ben = (ben / (ben[ev_m].pow(2).mean().sqrt() + 1e-12)).clamp(-5, 5)
        fov_loss = (fi["fov_v"].float() - ben)[ev_m].pow(2).mean()
        loss = loss + args.w_fov * fov_loss
    with torch.no_grad():
        dirs = BINS_T[None, None] - sampler.last_agent[:, :T, None]                      # (B,T,4,2)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-6)
        pick = (dirs * model.heading(h.detach()).float()[..., None, :]).sum(-1).argmax(-1)
        dec_acc = (pick == sampler.last_bin[:, :T])[dec].float().mean().item() if dec.any() else None
        tv = sampler.last_tv; jt = tv[:, 1] * G + tv[:, 0]
        tv_ev = b[torch.arange(B), :, jt].float().mean().item()
        tv_hr = fi["use"][torch.arange(B), :, jt].float().mean().item()
        comp = [(ti[0]["valid"].float().sum() / i["b"].float().sum()).item() for i in ti]
    return loss, dict(loss=loss.item(), mse=mse.item(), nll=nll.item(), a_mse=a_mse.item(), a_dec=a_dec.item(), a_ang=a_ang.item(), g_bce=g_bce.item(), ratio_t=float(ratio_t),
                      ratio_s=float(ratio_s), fov_loss=float(fov_loss), main_l=float(main_l) if main_l is not None else 0.0, ev_rate=b.float().mean().item(), hires_rate=fi["use"].float().mean().item(),
                      tv_ev=tv_ev, tv_hr=tv_hr, dec_acc=dec_acc, comp=comp, **jl)


log = []
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
RULES = {"fixed": dict(rule_fixed=0), "short": dict(rule_period=(300, 1200)), "normal": {}}
stages = [(int(a), int(b), int(c), float(d), e) for a, b, c, d, e in (x.split(":") for x in args.stages.split(","))] if args.stages else []
B, focus = args.B, args.focus
STEP = [0]
for s in range(args.steps):
    STEP[0] = s
    model.front.force_fovea = s < args.fov_warm
    if not stages:
        focus = args.focus_curr if (args.focus_curr is not None and s < args.rule_curr) else args.focus
        if args.rule_curr and s == args.rule_curr:
            sampler = main_sampler; print(f"curriculum: step {s} -> rule flashes", flush=True)
    for (s0, T_, B_, f_, r_) in stages:
        if s == s0:
            st_ = simulate(args.n_env, 30000, seed=1000 + args.seed + s0, delay=delay, pop_exit=args.pop_exit, rule_mode=args.rule_mode, **RULES[r_])
            sampler = ClipSampler(st_, T_, seed=args.seed + s0, rule_fixed=(r_ == "fixed"))
            B, focus = B_, f_
            print(f"stage: step {s} -> T {T_} B {B_} focus {f_} rule {r_}", flush=True)
    frames, clean, act, ev = sampler.sample(B, focus=focus, render=not args.freeze_front)
    loss, m = step_loss(frames, clean, act, ev)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if jm is not None:
        jm.update_target(model.front)
    if s % 50 == 0 or s == args.steps - 1:
        m.update(step=s, time=time.time() - t0, gn=gn.item(), mem_gb=torch.cuda.max_memory_allocated() / 1e9)
        log.append(m)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
torch.save({"model": model.state_dict(), "cfg": cfg, "args": vars(args), "jepa": jm.state_dict() if jm is not None else None}, f"{out_dir}/ckpt.pt")
json.dump(log, open(f"{out_dir}/train_log.json", "w"))
print("done", time.time() - t0)
