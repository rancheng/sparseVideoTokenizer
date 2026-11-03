"""LeRobot v3 datasets -> 128x128 uint8 frames + normalised actions, about 10 fps, one camera.

Reads meta/episodes, data/*.parquet and the first video file of the chosen camera (episodes stored in other files are
skipped). Short side scaled to 128, centre crop. Actions: per-dataset z-score, padded to A_MAX dims with a mask.
The last 10% of episodes (at least 3) are held out for evaluation.
usage: python real/prep_lerobot.py <raw_dir> <out_dir> <datasets.txt>
"""
import glob, json, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
import pandas as pd

S, A_MAX, MAX_FRAMES = 128, 14, 30000


def prep(args):
    raw, out, name, cam, kind = args
    tag = name if kind == "fixed" else f"{name}__{kind}"
    od = os.path.join(out, tag)
    if os.path.exists(os.path.join(od, "meta.json")):
        return tag, "cached"
    info = json.load(open(f"{raw}/{name}/meta/info.json"))
    fps = info["fps"]; stride = max(1, round(fps / 10))
    ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{raw}/{name}/meta/episodes/*/*.parquet"))])
    ep = ep[(ep[f"videos/{cam}/file_index"] == 0) & (ep[f"videos/{cam}/chunk_index"] == 0)].sort_values("episode_index")
    data = pd.concat([pd.read_parquet(f, columns=["action", "episode_index", "frame_index"]) for f in sorted(glob.glob(f"{raw}/{name}/data/*/*.parquet"))])
    vid = f"{raw}/{name}/videos/{cam}/chunk-000/file-000.mp4"
    # decode the whole file once (frame k <-> timestamp k / fps)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", vid, "-vf", f"scale={S}:{S}:force_original_aspect_ratio=increase,crop={S}:{S}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    buf = np.frombuffer(p.stdout.read(), np.uint8); p.wait()
    vf = buf.reshape(-1, S, S, 3)
    frames, acts, starts, eps = [], [], [], []
    n = 0
    for _, r in ep.iterrows():
        e = int(r["episode_index"]); L = int(r["length"])
        f0 = int(round(r[f"videos/{cam}/from_timestamp"] * fps))
        a = np.stack(data[data.episode_index == e].sort_values("frame_index")["action"].to_numpy())
        keep = np.arange(0, min(L, len(a), len(vf) - f0), stride)
        if len(keep) < 16:
            continue
        frames.append(vf[f0 + keep]); acts.append(a[keep].astype(np.float32)); starts.append(n); eps.append(e)
        n += len(keep)
        if n >= MAX_FRAMES:
            break
    fr = np.concatenate(frames); ac = np.concatenate(acts)
    ad = ac.shape[1]
    mu, sd = ac.mean(0), ac.std(0) + 1e-6
    acn = np.zeros((len(ac), A_MAX), np.float32); acn[:, :min(ad, A_MAX)] = ((ac - mu) / sd)[:, :A_MAX]
    n_te = max(3, len(starts) // 10)
    os.makedirs(od, exist_ok=True)
    np.save(f"{od}/frames.npy", fr); np.save(f"{od}/actions.npy", acn)
    meta = dict(name=name, cam=cam, kind=kind, fps=fps, stride=stride, fps_out=fps / stride, act_dim=min(ad, A_MAX),
                n_frames=int(len(fr)), episodes=eps, starts=starts + [int(len(fr))], test_from=len(starts) - n_te)
    json.dump(meta, open(f"{od}/meta.json", "w"))
    return tag, f"{len(fr)} frames, {len(starts)} episodes (test {n_te}), {fps}->{fps / stride:.0f} fps, act {ad}"


if __name__ == "__main__":
    raw, out, lst = sys.argv[1:4]
    jobs = [(raw, out, *l.split()) for l in open(lst) if l.strip()]
    jobs = [j for j in jobs if os.path.exists(f"{raw}/{j[2]}/videos/{j[3]}/chunk-000/file-000.mp4")]
    with Pool(int(os.environ.get("PREP_WORKERS", 6))) as pool:   # each worker decodes one whole video file into RAM
        for tag, msg in pool.imap_unordered(prep, jobs):
            print(tag, msg, flush=True)
