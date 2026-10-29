"""Export real boundary/event data for the blog's interactive demos (a 10-min window of held-out env 0)."""
import json, sys
import numpy as np
from viz.common import run, stream
from vt.env import EV_GRASP, EV_RELEASE, EV_SPAWN, EV_CUE, EV_QUERY, EV_LIGHT, EV_MOVE_START, EV_MOVE_STOP

T0, T1 = 18400, 24400  # 30:40 -> 40:40 into the held-out hour
st = stream()
sl = slice(T0, T1)
act = st["action"][0, sl]
moving = np.abs(act[:, :2]).sum(1) > 0.2
ev = st["events"][0, sl]

def runs(mask):
    m = np.concatenate([[0], mask.astype(int), [0]]); d = np.diff(m)
    return [[int(a), int(b)] for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]

events = []
for bit, name in [(EV_SPAWN, "spawn"), (EV_GRASP, "grasp"), (EV_RELEASE, "release"), (EV_CUE, "cue"), (EV_QUERY, "query"), (EV_LIGHT, "light")]:
    for i in np.nonzero(ev & bit)[0]:
        e = {"t": int(i), "type": name}
        if name in ("cue", "query"):
            e["color"] = int(st["cue_col"][0, T0 + i]) if name == "cue" else int(st["query_answer"][0, T0 + i])
        events.append(e)
out = {"t0": T0, "n": T1 - T0, "fps": 10, "moving": runs(moving), "dim": runs(st["light"][0, sl] < 0.9),
       "holding": runs(st["grip"][0, sl] > 0.5), "events": sorted(events, key=lambda e: e["t"]), "tok": {}}
for name in ["fixed_16x", "heur_16x", "hnet_16x", "hnet_16x_videoonly", "hnet_cap_r02"]:
    r = run(name)
    rec = {}
    for l in (0, 1):
        b, rr = r[f"b{l}"][sl], r[f"r{l}"][sl]
        rec[f"L{l+1}"] = [int(i) for i in np.nonzero(b)[0]]
        rec[f"L{l+1}_hb"] = [int(i) for i in np.nonzero(b & ~rr)[0]]  # heartbeat-only tokens
    if name == "hnet_16x":
        rec["p1"] = [round(float(x), 4) for x in r["pr0"][sl]]
    out["tok"][name] = rec
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
print({k: (len(v["L1"]), len(v["L2"])) for k, v in out["tok"].items()}, "events", len(events))
