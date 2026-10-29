"""Synthetic long-horizon tabletop robot stream with ground-truth events.

Designed to stress exactly what a robot-policy video tokenizer must do:
  * long idle stretches (nothing happens -> should cost ~0 tokens)
  * action-caused events (grasp / release / motion onset) -> must be tokenized
  * non-action events (light toggles) and predictable nuisances (blinking LED,
    sensor noise) -> a pixel-change heuristic wastes tokens on these
  * a memory task: a colour cue is flashed, and minutes (or hours) later a query
    flash asks the robot to go to the bin of that colour. The correct action at
    the query frame depends ONLY on the far past -> tests long context.

Simulation is vectorised over envs in numpy (cheap state only); frames are
rendered on the GPU on demand, so hour/day long streams cost ~nothing to store.
"""
import numpy as np
import torch

H = W = 64
BINS = np.array([[6, 6], [57, 6], [6, 57], [57, 57]], dtype=np.float32)  # x, y
HOME = np.array([32.0, 34.0], dtype=np.float32)
COLORS = torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.3, 0.9], [0.9, 0.85, 0.2]])
SPEED = 1.5  # px / frame at |v| = 1
IDLE, REACH, CARRY, RETURN, RECALL = range(5)
# event bits
EV_GRASP, EV_RELEASE, EV_SPAWN, EV_CUE, EV_QUERY, EV_LIGHT, EV_MOVE_START, EV_MOVE_STOP = (1 << i for i in range(8))
ACTION_EVENTS = EV_GRASP | EV_RELEASE | EV_SPAWN | EV_CUE | EV_QUERY | EV_MOVE_START | EV_MOVE_STOP


def simulate(n_env, n_steps, seed=0, gap=(100, 1500), cue_rate=1 / 600, light_rate=1 / 400,
             idle_mix=((0.70, 5, 40), (0.25, 50, 300), (0.05, 300, 1500))):
    """Returns dict of per-frame state arrays shaped (n_env, n_steps, ...)."""
    rng = np.random.default_rng(seed)
    E = n_env
    pos = np.tile(HOME, (E, 1)) + rng.uniform(-3, 3, (E, 2)).astype(np.float32)
    grip = np.zeros(E, np.float32)
    mode = np.full(E, IDLE)
    idle_left = rng.integers(5, 40, E)
    obj_on = np.zeros(E, bool)
    obj_pos = np.zeros((E, 2), np.float32)
    obj_col = np.zeros(E, np.int64)
    light = np.ones(E, np.float32)
    cue_pending = np.zeros(E, bool)
    cue_col = np.zeros(E, np.int64)
    cue_show = np.zeros(E, np.int64)
    cue_time = np.zeros(E, np.int64)
    query_at = np.zeros(E, np.int64)
    query_show = np.zeros(E, np.int64)
    recall_col = np.zeros(E, np.int64)
    led_phase = rng.integers(0, 16, E)
    moving_prev = np.zeros(E, bool)

    out = {k: np.zeros((E, n_steps) + s, d) for k, s, d in [
        ("agent", (2,), np.float32), ("grip", (), np.float32), ("obj_on", (), bool), ("obj_pos", (2,), np.float32),
        ("obj_col", (), np.int64), ("light", (), np.float32), ("led", (), bool), ("cue_on", (), bool),
        ("cue_col", (), np.int64), ("query_on", (), bool), ("action", (3,), np.float32), ("events", (), np.int64),
        ("query_answer", (), np.int64), ("query_cue_t", (), np.int64)]}
    out["query_answer"][:] = -1
    out["query_cue_t"][:] = -1

    def sample_idle(n):
        comp = rng.choice(len(idle_mix), n, p=[m[0] for m in idle_mix])
        lo = np.array([idle_mix[c][1] for c in comp]); hi = np.array([idle_mix[c][2] for c in comp])
        return rng.integers(lo, hi)

    for t in range(n_steps):
        ev = np.zeros(E, np.int64)
        # ---- exogenous events (not caused by the robot) ----
        tog = rng.random(E) < light_rate
        light = np.where(tog, np.where(light > 0.9, 0.7, 1.0), light).astype(np.float32)
        ev |= np.where(tog, EV_LIGHT, 0)
        new_cue = (~cue_pending) & (rng.random(E) < cue_rate)
        if new_cue.any():
            cue_pending |= new_cue
            cue_col = np.where(new_cue, rng.integers(0, 4, E), cue_col)
            cue_show = np.where(new_cue, 4, cue_show)
            cue_time = np.where(new_cue, t, cue_time)
            query_at = np.where(new_cue, t + rng.integers(gap[0], gap[1], E), query_at)
            ev |= np.where(new_cue, EV_CUE, 0)
        # ---- expert policy ----
        target = np.tile(HOME, (E, 1))
        idle = mode == IDLE
        fire_q = idle & cue_pending & (t >= query_at)
        if fire_q.any():
            mode = np.where(fire_q, RECALL, mode)
            recall_col = np.where(fire_q, cue_col, recall_col)
            query_show = np.where(fire_q, 3, query_show)
            ev |= np.where(fire_q, EV_QUERY, 0)
            out["query_answer"][fire_q, t] = cue_col[fire_q]
            out["query_cue_t"][fire_q, t] = cue_time[fire_q]
            cue_pending &= ~fire_q
        idle = mode == IDLE
        idle_left = np.where(idle, idle_left - 1, idle_left)
        wake = idle & (idle_left <= 0)
        spawn = wake & ~obj_on
        if spawn.any():
            obj_on |= spawn
            obj_pos = np.where(spawn[:, None], rng.uniform(16, 48, (E, 2)).astype(np.float32), obj_pos)
            obj_col = np.where(spawn, rng.integers(0, 4, E), obj_col)
            idle_left = np.where(spawn, rng.integers(5, 15, E), idle_left)
            ev |= np.where(spawn, EV_SPAWN, 0)
        go = wake & ~spawn & obj_on
        mode = np.where(go, REACH, mode)

        # ---- record observation o_t (state before the robot acts at t) ----
        out["agent"][:, t] = pos; out["grip"][:, t] = grip
        out["obj_on"][:, t] = obj_on; out["obj_pos"][:, t] = obj_pos; out["obj_col"][:, t] = obj_col
        out["light"][:, t] = light; out["led"][:, t] = ((t + led_phase) // 8) % 2 == 0
        out["cue_on"][:, t] = cue_show > 0; out["cue_col"][:, t] = cue_col
        out["query_on"][:, t] = query_show > 0

        target = np.where((mode == REACH)[:, None], obj_pos, target)
        target = np.where((mode == CARRY)[:, None], BINS[obj_col], target)
        target = np.where((mode == RECALL)[:, None], BINS[recall_col], target)
        delta = target - pos
        dist = np.linalg.norm(delta, axis=1)
        v = np.clip(delta / (SPEED * 2.0), -1, 1)
        v = np.where((mode == IDLE)[:, None], 0.0, v)
        active = mode != IDLE
        v = v + np.where(active[:, None], rng.normal(0, 0.05, (E, 2)), 0.0)
        v = np.clip(v, -1, 1).astype(np.float32)

        # transitions (decided on current obs, applied with this action)
        grasp = (mode == REACH) & (dist < 1.5)
        release = (mode == CARRY) & (dist < 2.0)
        recall_done = (mode == RECALL) & (dist < 2.0)
        home = (mode == RETURN) & (dist < 1.5)
        grip = np.where(grasp, 1.0, np.where(release, 0.0, grip)).astype(np.float32)
        ev |= np.where(grasp, EV_GRASP, 0) | np.where(release, EV_RELEASE, 0)
        obj_on &= ~release
        mode = np.where(grasp, CARRY, mode)
        mode = np.where(release | recall_done, RETURN, mode)
        mode = np.where(home, IDLE, mode)
        if home.any():
            idle_left = np.where(home, sample_idle(E), idle_left)
        moving = np.abs(v).sum(1) > 0.2
        ev |= np.where(moving & ~moving_prev, EV_MOVE_START, 0) | np.where(~moving & moving_prev, EV_MOVE_STOP, 0)
        moving_prev = moving

        # ---- record a_t ----
        out["action"][:, t, :2] = v; out["action"][:, t, 2] = grip
        out["events"][:, t] = ev
        # ---- apply dynamics ----
        pos = np.clip(pos + v * SPEED, 3, 60).astype(np.float32)
        obj_pos = np.where(((mode == CARRY) | grasp)[:, None] & obj_on[:, None], pos, obj_pos)
        cue_show = np.maximum(cue_show - 1, 0)
        query_show = np.maximum(query_show - 1, 0)
    return out


class Renderer:
    """Renders state arrays to (N,3,64,64) float frames on GPU."""

    def __init__(self, device="cuda"):
        self.dev = device
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        self.xs, self.ys = xs.float() + 0.5, ys.float() + 0.5
        self.colors = COLORS.to(device)
        bg = torch.full((3, H, W), 0.45, device=device)
        for i, (bx, by) in enumerate(BINS):
            m = ((self.xs - bx).abs() < 5) & ((self.ys - by).abs() < 5)
            bg[:, m] = self.colors[i][:, None] * 0.6
        self.bg = bg

    def sq(self, cx, cy, r):
        return ((self.xs - cx[:, None, None]).abs() < r) & ((self.ys - cy[:, None, None]).abs() < r)

    @torch.no_grad()
    def __call__(self, st, noise=0.02, gen=None):
        g = lambda k: torch.as_tensor(st[k], device=self.dev)
        agent, grip, obj_on, obj_pos, obj_col = g("agent"), g("grip"), g("obj_on"), g("obj_pos"), g("obj_col")
        light, led, cue_on, cue_col, query_on = g("light"), g("led"), g("cue_on"), g("cue_col"), g("query_on")
        N = agent.shape[0]
        img = self.bg.expand(N, 3, H, W).clone()
        m = self.sq(obj_pos[:, 0], obj_pos[:, 1], 3.0) & obj_on[:, None, None]
        img = torch.where(m[:, None], self.colors[obj_col][:, :, None, None], img)
        d2 = (self.xs - agent[:, 0, None, None]) ** 2 + (self.ys - agent[:, 1, None, None]) ** 2
        m = d2 < 3.5 ** 2
        acol = torch.where(grip[:, None] > 0.5, torch.tensor([1.0, 0.55, 0.1], device=self.dev),
                           torch.tensor([0.95, 0.95, 0.95], device=self.dev))
        img = torch.where(m[:, None], acol[:, :, None, None], img)
        z = torch.zeros(N, device=self.dev)
        m = self.sq(z + 32, z + 2.5, 1.5) & led[:, None, None]
        img = torch.where(m[:, None], torch.tensor([1.0, 0.1, 0.1], device=self.dev)[None, :, None, None], img)
        m = self.sq(z + 32, z + 60, 3.0) & cue_on[:, None, None]
        img = torch.where(m[:, None], self.colors[cue_col][:, :, None, None], img)
        border = (self.xs < 2) | (self.xs > W - 2) | (self.ys < 2) | (self.ys > H - 2)
        m = border[None] & query_on[:, None, None]
        img = torch.where(m[:, None], torch.ones(1, 3, 1, 1, device=self.dev), img)
        img = img * light[:, None, None, None]
        clean = img
        if noise > 0:
            img = (img + noise * torch.randn(img.shape, device=self.dev, generator=gen)).clamp(0, 1)
        return img, clean


def flat(st, idx_env, t0, t1):
    """Slice env idx_env, frames [t0,t1) -> dict of flat arrays."""
    return {k: v[idx_env, t0:t1] for k, v in st.items()}


class ClipSampler:
    """Samples (B,T) clips from pre-simulated streams, rendered on GPU."""

    def __init__(self, st, T, device="cuda", seed=0):
        self.st, self.T = st, T
        self.E, self.N = st["action"].shape[:2]
        self.r = Renderer(device)
        self.rng = np.random.default_rng(seed)
        self.dev = device

    def sample(self, B):
        T = self.T
        envs = self.rng.integers(0, self.E, B)
        t0s = self.rng.integers(0, self.N - T - 1, B)
        idx_t = t0s[:, None] + np.arange(T + 1)[None]
        sub = {k: v[envs[:, None], idx_t].reshape((B * (T + 1),) + v.shape[2:]) for k, v in self.st.items()}
        frames, clean = self.r(sub)
        frames = frames.view(B, T + 1, 3, H, W)
        clean = clean.view(B, T + 1, 3, H, W)
        act = torch.as_tensor(sub["action"], device=self.dev).view(B, T + 1, 3)
        ev = torch.as_tensor(sub["events"], device=self.dev).view(B, T + 1)
        return frames, clean, act, ev
