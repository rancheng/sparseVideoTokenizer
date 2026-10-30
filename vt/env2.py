"""Workcell v2: a 128x128 robot stream built so that each tokenizer assumption can be tested on its own.

What is in the scene, and which assumption it tests:
  * conveyor + opaque tunnel   object permanence. An item enters the tunnel and exits D frames later; its colour and
                               glyph at the exit can only be predicted from the entry, D frames back. With D beyond the
                               local windows this makes the MAIN network necessary for the dense forward-model loss
                               (H1). D is a knob: short D is solvable locally (control).
  * glyph-coded sorting        each item carries a glyph (solid / h-stripes / v-stripes / checker, all with the same
                               mean colour), so it is invisible at low resolution. The bin depends on the glyph, so the
                               policy needs high-resolution tokens at the right place (fovea, H4). Colour is a distractor.
  * rule display               a small symbol flashed for 5 frames sets the glyph->bin mapping (identity or mirrored)
                               until the next flash, minutes later. Every transport decision depends on it, so
                               long-range credit is dense (H3), and the symbol also needs the fovea.
  * noisy TV                   one grid cell (random position per env) shows unpredictable, irrelevant noise. A pure
                               change detector must fire on it every frame; a relevance-aware router can learn not to (H6).
  * static textured background, global light toggles, sensor noise, long idle stretches: spatial and temporal sparsity.
  * ground truth               phase per frame (idle / reach / carry / return), event bits, item episode ids, the rule,
                               the glyph, the target bin, tunnel exits, TV cell. Primitive boundaries (phase changes)
                               test level 1; task starts (reach onset) test level 2 (H2).

The simulation is vectorised over envs in numpy (cheap state only); frames are rendered on the GPU on demand.
The frame is an 8x8 grid of 16x16 cells, the unit the sparse front end works with.
"""
import numpy as np
import torch

H = W = 128
CELL = 16
G = H // CELL
HOME = np.array([64.0, 76.0], dtype=np.float32)
BINS = np.array([[16, 118], [48, 118], [80, 118], [112, 118]], dtype=np.float32)
BELT_Y, TUN_X0, TUN_X1, PICK_X = 13.0, 24.0, 104.0, 116.0
RULE_C = (7, 4)  # grid cell (gx, gy) of the rule display
COLORS = torch.tensor([[0.9, 0.25, 0.2], [0.25, 0.8, 0.25], [0.25, 0.35, 0.95], [0.95, 0.85, 0.2]])
SPEED = 2.5  # px / frame at |v| = 1
IDLE, REACH, CARRY, RETURN = range(4)
EV_GRASP, EV_RELEASE, EV_SPAWN, EV_EXIT, EV_RULE, EV_LIGHT, EV_REACH, EV_HOME = (1 << i for i in range(8))
ACTION_EVENTS = EV_GRASP | EV_RELEASE | EV_REACH | EV_HOME


def target_bin(glyph, rule):
    return np.where(rule == 0, glyph, 3 - glyph)


def simulate(n_env, n_steps, seed=0, delay=(150, 600), rule_period=(600, 2400), light_rate=1 / 900, rule_fixed=None,
             idle_mix=((0.6, 10, 60), (0.3, 60, 300), (0.1, 300, 1200))):
    """Returns dict of per-frame arrays shaped (n_env, n_steps, ...). delay = tunnel transit time range in frames."""
    rng = np.random.default_rng(seed)
    E = n_env
    pos = np.tile(HOME, (E, 1)).astype(np.float32)
    grip = np.zeros(E, np.float32)
    mode = np.full(E, IDLE)
    react = np.zeros(E, np.int64)
    belt_on = np.zeros(E, bool); belt_x = np.zeros(E, np.float32); belt_v = np.ones(E, np.float32)
    item_col = np.zeros(E, np.int64); item_gly = np.zeros(E, np.int64)
    held = np.zeros(E, bool)
    spawn_in = rng.integers(5, 60, E)
    rule = rng.integers(0, 2, E); rule_show = np.full(E, 5); rule_t = np.zeros(E, np.int64)
    rule_next = rng.integers(*rule_period, E)
    if rule_fixed is not None:  # curriculum stage: one known rule, never flashed again (only the glyph matters)
        rule[:] = rule_fixed; rule_next[:] = n_steps + 1
    light = np.ones(E, np.float32)
    tv_cell = np.stack([rng.integers(0, 3, E), rng.integers(2, 6, E)], 1)  # left half, away from belt and bins
    tv_code = rng.integers(0, 2 ** 31, E)
    episode = np.zeros(E, np.int64)

    out = {k: np.zeros((E, n_steps) + s, d) for k, s, d in [
        ("agent", (2,), np.float32), ("grip", (), np.float32), ("belt_on", (), bool), ("belt_x", (), np.float32),
        ("item_col", (), np.int64), ("item_gly", (), np.int64), ("held", (), bool), ("rule", (), np.int64),
        ("rule_on", (), bool), ("rule_t", (), np.int64), ("light", (), np.float32), ("tv_code", (), np.int64),
        ("action", (3,), np.float32), ("events", (), np.int64), ("phase", (), np.int64), ("episode", (), np.int64),
        ("bin", (), np.int64), ("decision", (), bool)]}
    out["tv_cell"] = tv_cell
    out["env_id"] = np.broadcast_to(np.arange(E)[:, None], (E, n_steps)).copy()
    out["bin"][:] = -1

    def sample_idle(n):
        comp = rng.choice(len(idle_mix), n, p=[m[0] for m in idle_mix])
        lo = np.array([idle_mix[c][1] for c in comp]); hi = np.array([idle_mix[c][2] for c in comp])
        return rng.integers(lo, hi)

    for t in range(n_steps):
        ev = np.zeros(E, np.int64)
        # ---- exogenous: light, rule flashes, TV, conveyor ----
        tog = rng.random(E) < light_rate
        light = np.where(tog, np.where(light > 0.9, 0.75, 1.0), light).astype(np.float32)
        ev |= np.where(tog, EV_LIGHT, 0)
        flip = t >= rule_next
        rule = np.where(flip, 1 - rule, rule); rule_show = np.where(flip, 5, rule_show)
        rule_t = np.where(flip, t, rule_t); rule_next = np.where(flip, t + rng.integers(*rule_period, E), rule_next)
        ev |= np.where(flip | (t == 0), EV_RULE, 0)
        tv_code = np.where(rng.random(E) < 0.5, rng.integers(0, 2 ** 31, E), tv_code)
        spawn_in = np.where(~belt_on & ~held & (mode == IDLE), spawn_in - 1, spawn_in)
        spawn = ~belt_on & ~held & (mode == IDLE) & (spawn_in <= 0)
        if spawn.any():
            belt_on |= spawn; belt_x = np.where(spawn, 0.0, belt_x).astype(np.float32)
            d = rng.integers(delay[0], delay[1] + 1, E)
            belt_v = np.where(spawn, (TUN_X1 - TUN_X0) / d, belt_v).astype(np.float32)
            item_col = np.where(spawn, rng.integers(0, 4, E), item_col)
            item_gly = np.where(spawn, rng.integers(0, 4, E), item_gly)
            ev |= np.where(spawn, EV_SPAWN, 0)
        at_pick = belt_on & (belt_x >= PICK_X)
        # ---- expert policy ----
        idle = mode == IDLE
        react = np.where(idle & at_pick & (react == 0), rng.integers(3, 15, E), react)
        go = idle & at_pick & (react == 1)
        react = np.where(idle & at_pick & (react > 1), react - 1, react)
        if go.any():
            mode = np.where(go, REACH, mode); react = np.where(go, 0, react); episode += go
            ev |= np.where(go, EV_REACH, 0)
        out["agent"][:, t] = pos; out["grip"][:, t] = grip
        out["belt_on"][:, t] = belt_on; out["belt_x"][:, t] = belt_x
        out["item_col"][:, t] = item_col; out["item_gly"][:, t] = item_gly; out["held"][:, t] = held
        out["rule"][:, t] = rule; out["rule_on"][:, t] = rule_show > 0; out["rule_t"][:, t] = rule_t
        out["light"][:, t] = light; out["tv_code"][:, t] = tv_code
        out["phase"][:, t] = mode; out["episode"][:, t] = episode

        b = target_bin(item_gly, rule)
        target = np.tile(HOME, (E, 1))
        target = np.where((mode == REACH)[:, None], np.stack([np.full(E, PICK_X), np.full(E, BELT_Y + 6)], 1), target)
        target = np.where((mode == CARRY)[:, None], BINS[b] - np.array([0, 8.0]), target)
        delta = target - pos
        dist = np.linalg.norm(delta, axis=1)
        # unit direction times a slow-down near the target: the heading at a decision frame identifies the bin
        v = delta / np.maximum(dist, 1e-6)[:, None] * np.clip(dist / (SPEED * 2.0), 0, 1)[:, None]
        v = np.where((mode == IDLE)[:, None], 0.0, v)
        v = v + np.where((mode != IDLE)[:, None], rng.normal(0, 0.05, (E, 2)), 0.0)
        v = np.clip(v, -1, 1).astype(np.float32)
        grasp = (mode == REACH) & (dist < 2.0)
        release = (mode == CARRY) & (dist < 2.5)
        home = (mode == RETURN) & (dist < 2.0)
        grip = np.where(grasp, 1.0, np.where(release, 0.0, grip)).astype(np.float32)
        ev |= np.where(grasp, EV_GRASP, 0) | np.where(release, EV_RELEASE, 0) | np.where(home, EV_HOME, 0)
        out["bin"][:, t] = np.where(mode == CARRY, b, -1)
        held = np.where(grasp, True, np.where(release, False, held))
        belt_on &= ~grasp
        mode = np.where(grasp, CARRY, mode)
        mode = np.where(release, RETURN, mode)
        mode = np.where(home, IDLE, mode)
        spawn_in = np.where(home, sample_idle(E), spawn_in)
        out["action"][:, t, :2] = v; out["action"][:, t, 2] = grip
        # ---- dynamics ----
        pos = np.clip(pos + v * SPEED, 6, 122).astype(np.float32)
        nx = np.where(belt_on & (belt_x < PICK_X), np.minimum(belt_x + belt_v, PICK_X), belt_x)
        ev |= np.where(belt_on & (belt_x < TUN_X1) & (nx >= TUN_X1), EV_EXIT, 0)
        belt_x = nx.astype(np.float32)
        rule_show = np.maximum(rule_show - 1, 0)
        out["events"][:, t] = ev
    ph = out["phase"]  # decision = first carry frame: its action must already head to the glyph/rule bin
    out["decision"][:, 1:] = (ph[:, 1:] == CARRY) & (ph[:, :-1] == REACH)
    return out


def _hash(*xs):  # deterministic pseudo-random in [0,1) from integer tensors (for the TV)
    h = torch.zeros_like(xs[0], dtype=torch.float32)
    for i, x in enumerate(xs):
        h = h + (x.float() % 9973.0) * (12.9898 + 7.233 * i)
    return torch.frac(torch.sin(h) * 43758.5453)


class Renderer:
    """Renders state arrays to (N,3,128,128) frames on GPU. Backgrounds are per-env static textures."""

    def __init__(self, device="cuda", n_bg=4096):
        self.dev = device
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        self.xs, self.ys = xs.float() + 0.5, ys.float() + 0.5
        self.xi, self.yi = xs, ys
        self.colors = COLORS.to(device)
        self.n_bg = n_bg
        self._bg = {}

    def bg(self, env_id):  # static textured table, one per env (low-frequency blobs + fine weave)
        uniq, inv = torch.unique(env_id, return_inverse=True)
        for e in uniq.tolist():
            if e not in self._bg:
                g = torch.Generator(device=self.dev).manual_seed(int(e) * 7 + 1)
                low = torch.rand(1, 3, 8, 8, device=self.dev, generator=g)
                low = torch.nn.functional.interpolate(low, size=(H, W), mode="bicubic", align_corners=False)[0]
                weave = 0.04 * ((self.xi // 2 + self.yi // 2) % 2).float()
                base = 0.35 + 0.15 * low + weave
                for k, (bx, by) in enumerate(BINS):  # bins: dark frames, identical (position is the identity)
                    m = ((self.xs - bx).abs() < 8) & ((self.ys - by).abs() < 8) & ~(((self.xs - bx).abs() < 6) & ((self.ys - by).abs() < 6))
                    base[:, m] = 0.12
                belt = (self.ys - BELT_Y).abs() < 5
                base[:, belt] = 0.22
                self._bg[e] = base.clamp(0, 1)
        return torch.stack([self._bg[e] for e in uniq.tolist()])[inv]

    def glyph_square(self, img, cx, cy, col, gly, half, on):
        """Square of colour col with pattern gly (0 solid, 1 h-stripes, 2 v-stripes, 3 checker), equal mean colour."""
        dx, dy = self.xs - cx[:, None, None], self.ys - cy[:, None, None]
        m = (dx.abs() < half) & (dy.abs() < half) & on[:, None, None]
        u = torch.floor(dx + half).long() % 2; v = torch.floor(dy + half).long() % 2
        g = gly[:, None, None]
        hi = torch.where(g == 1, v == 0, torch.where(g == 2, u == 0, (u + v) % 2 == 0))
        f = torch.where(g == 0, torch.full_like(dx, 0.7), torch.where(hi, 1.0, 0.4))
        c = self.colors[col][:, :, None, None] * f[:, None]
        return torch.where(m[:, None], c, img)

    @torch.no_grad()
    def __call__(self, st, noise=0.02, gen=None):
        g = lambda k: torch.as_tensor(np.ascontiguousarray(st[k]), device=self.dev)
        agent, grip, belt_on, belt_x = g("agent"), g("grip"), g("belt_on"), g("belt_x")
        col, gly, held, rule, rule_on = g("item_col"), g("item_gly"), g("held"), g("rule"), g("rule_on")
        light, tv_code, env_id, tv_cell = g("light"), g("tv_code"), g("env_id"), g("tv_cell")
        N = agent.shape[0]
        img = self.bg(env_id).clone()
        # item on the belt, then the opaque tunnel on top of it
        img = self.glyph_square(img, belt_x, torch.full_like(belt_x, BELT_Y), col, gly, 4.0, belt_on)
        tun = (self.xs >= TUN_X0) & (self.xs < TUN_X1) & ((self.ys - BELT_Y).abs() < 9)
        img = torch.where(tun[None, None], torch.tensor([0.42, 0.3, 0.2], device=self.dev)[None, :, None, None], img)
        # robot + held item (below the gripper, glyph visible)
        img = self.glyph_square(img, agent[:, 0], agent[:, 1] + 8, col, gly, 4.0, held)
        d2 = (self.xs - agent[:, 0, None, None]) ** 2 + (self.ys - agent[:, 1, None, None]) ** 2
        acol = torch.where(grip[:, None] > 0.5, torch.tensor([1.0, 0.55, 0.1], device=self.dev),
                           torch.tensor([0.95, 0.95, 0.95], device=self.dev))
        img = torch.where((d2 < 25)[:, None], acol[:, :, None, None], img)
        # rule display: h-stripes = rule 0, v-stripes = rule 1 (same mean: needs high resolution)
        rc = torch.full((N,), RULE_C[0] * CELL + 8.0, device=self.dev); ry = torch.full((N,), RULE_C[1] * CELL + 8.0, device=self.dev)
        img = self.glyph_square(img, rc, ry, torch.full_like(rule, 3), rule + 1, 5.0, rule_on)
        # noisy TV: 4x4 blocks of random colour inside one cell, redrawn when tv_code changes
        tx0, ty0 = tv_cell[:, 0] * CELL, tv_cell[:, 1] * CELL
        inx = (self.xi[None] >= tx0[:, None, None] + 1) & (self.xi[None] < tx0[:, None, None] + 15)
        iny = (self.yi[None] >= ty0[:, None, None] + 1) & (self.yi[None] < ty0[:, None, None] + 15)
        bx = (self.xi[None] - tx0[:, None, None]) // 4; by = (self.yi[None] - ty0[:, None, None]) // 4
        tvc = torch.stack([_hash(tv_code[:, None, None], bx, by, torch.full_like(bx, k)) for k in range(3)], 1)
        img = torch.where((inx & iny)[:, None], tvc, img)
        img = img * light[:, None, None, None]
        clean = img
        if noise > 0:
            img = (img + noise * torch.randn(img.shape, device=self.dev, generator=gen)).clamp(0, 1)
        return img, clean


def flat(st, idx_env, t0, t1):
    out = {k: v[idx_env, t0:t1] for k, v in st.items() if k != "tv_cell"}
    out["tv_cell"] = np.broadcast_to(st["tv_cell"][idx_env], (t1 - t0, 2))
    return out


class ClipSampler:
    """(B,T) clips from pre-simulated streams, rendered on GPU. Marks decision frames whose rule flash is in the clip."""

    def __init__(self, st, T, device="cuda", seed=0, rule_fixed=False):
        self.st, self.T, self.rule_fixed = st, T, rule_fixed
        self.E, self.N = st["action"].shape[:2]
        self.r = Renderer(device)
        self.rng = np.random.default_rng(seed)
        self.dev = device

    def sample(self, B, focus=0.0):
        """focus: fraction of clips placed so that a decision frame lies in their second half (rare-event replay)."""
        T = self.T
        envs = self.rng.integers(0, self.E, B)
        t0s = self.rng.integers(0, self.N - T - 1, B)
        if focus > 0:
            if not hasattr(self, "_dec"):
                self._dec = [np.nonzero(self.st["decision"][e])[0] for e in range(self.E)]
            for i in range(B):
                d = self._dec[envs[i]]
                d = d[(d >= T) & (d < self.N - 2)]
                if len(d) and self.rng.random() < focus:
                    t0s[i] = min(self.N - T - 2, self.rng.choice(d) - self.rng.integers(T // 2, T))
        idx_t = t0s[:, None] + np.arange(T + 1)[None]
        sub = {k: v[envs[:, None], idx_t].reshape((B * (T + 1),) + v.shape[2:]) for k, v in self.st.items() if k != "tv_cell"}
        sub["tv_cell"] = np.repeat(self.st["tv_cell"][envs], T + 1, 0)
        frames, clean = self.r(sub)
        frames = frames.view(B, T + 1, 3, H, W); clean = clean.view(B, T + 1, 3, H, W)
        tt = lambda k: torch.as_tensor(sub[k], device=self.dev).view(B, T + 1, *sub[k].shape[1:])
        act, ev = tt("action"), tt("events")
        # a carry decision is answerable when the latest rule flash lies inside the clip (or the rule is fixed)
        rule_in = tt("rule_t") >= torch.as_tensor(t0s, device=self.dev)[:, None]
        if self.rule_fixed:
            rule_in = torch.ones_like(rule_in)
        self.last_bin, self.last_rule_in = tt("bin"), rule_in
        self.last_decision = tt("decision")
        self.last_phase = tt("phase")
        self.last_agent = tt("agent")
        self.last_tv = torch.as_tensor(self.st["tv_cell"][envs], device=self.dev)
        return frames, clean, act, ev
