"""ADR-511: "measure, then fix" -- python -m iw3.mismatch_plan_cli

Step 1 of a two-pass workflow. Reads a FINISHED 3D movie (Half Top-Bottom or Half SBS) and its
Pre-Plan scene list ('..._scene_plan.txt', written next to every Pre-Plan output), measures the real
left/right mismatch on MOVING areas (where decker sees characters doubled / out of focus) every
`--every` frames, scores every scene, and writes a per-scene 3D Strength plan (JSON). Step 2 is a
normal conversion of the same movie with `--scene-strength-plan <json>`: each scene's 3D Strength is
multiplied by its factor (held for the whole scene, sharing Scene Safety's cap).

Why: pre-render predictors (ADR-508/510) could not flag Mario's 16:49 workshop with depth reset (2.2x
D2's moving mismatch); measuring the real 3D finds it directly. The worst scenes (this movie's top
15% by measured moving mismatch) get a cut that ramps up to --max-cut at the top 2%.
CPU only; decodes once, sequentially, only the sampled frame pairs."""
import argparse
import json
import re
import subprocess
import sys
from os import path

import numpy as np

ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
FFMPEG = path.join(ROOT, "ffmpeg.exe") if path.exists(path.join(ROOT, "ffmpeg.exe")) else "ffmpeg"
FFPROBE = path.join(ROOT, "ffprobe.exe") if path.exists(path.join(ROOT, "ffprobe.exe")) else "ffprobe"
WS = 960


def read_scene_plan(plan_path):
    """[(index, start_seconds, length_seconds)] from a Pre-Plan '..._scene_plan.txt'."""
    rows = []
    for line in open(plan_path, encoding="utf-8"):
        m = re.match(r"^\s*(\d+)\s+(\d+):(\d\d):(\d\d\.\d)\s+([\d.]+)s\s", line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4)),
                         float(m.group(5))))
    return rows


def pair_scores(L0, R0, L1):
    """(moving mismatch, whole mismatch) for one frame pair of the left eye (L0 -> L1) and its right
    eye R0. Mismatch = mean |L - right eye warped onto the left| (Farneback flow); 'moving' = only
    pixels that move between L0 and L1. moving is NaN when < 1500 pixels move."""
    import cv2
    H, W = L0.shape
    keep = np.ones((H, W), bool)
    keep[:8] = keep[-8:] = False
    keep[:, :16] = keep[:, -16:] = False
    st = cv2.calcOpticalFlowFarneback(L0, R0, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    err = np.abs(L0.astype(np.float32) - cv2.remap(R0, gx + st[..., 0], gy + st[..., 1],
                                                    cv2.INTER_LINEAR).astype(np.float32))
    mot = cv2.calcOpticalFlowFarneback(L0, L1, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    moving = keep & (np.hypot(mot[..., 0], mot[..., 1]) > 1.5)
    mm = float(err[moving].mean()) if moving.sum() >= 1500 else float("nan")
    return mm, float(err[keep].mean())


def split_eyes(frame, layout):
    import cv2
    h, w = frame.shape
    if layout == "half_tb":
        a, b = frame[: h // 2], frame[h // 2: (h // 2) * 2]
        return cv2.resize(a, (w, (h // 2) * 2)), cv2.resize(b, (w, (h // 2) * 2))
    a, b = frame[:, : w // 2], frame[:, w // 2: (w // 2) * 2]
    return cv2.resize(a, ((w // 2) * 2, h)), cv2.resize(b, ((w // 2) * 2, h))


def scene_scores(samples, scenes):
    """samples: [(time, moving, whole)] -> per scene (moving mean, whole mean, n)."""
    out = []
    for idx, start, dur in scenes:
        seg = [(m, w) for t, m, w in samples if start <= t < start + dur]
        mv = [m for m, _ in seg if m == m]
        out.append((float(np.mean(mv)) if mv else None, float(np.mean([w for _, w in seg])) if seg else None, len(seg)))
    return out


def plan_factors(moving, max_cut, low_pct=85, high_pct=98):
    vals = np.array([v for v in moving if v is not None])
    if len(vals) < 10:
        return [1.0] * len(moving), None, None
    lo, hi = float(np.percentile(vals, low_pct)), float(np.percentile(vals, high_pct))
    out = []
    for v in moving:
        if v is None:
            out.append(1.0)
            continue
        t = min(max((v - lo) / max(hi - lo, 1e-9), 0.0), 1.0)
        out.append(round(1.0 - max_cut * t, 4))
    return out, lo, hi


def measure_movie(video, scenes, layout="half_tb", every=12, log=print):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate", "-of", "csv=p=0", video], capture_output=True, text=True)
    w, h, rate = r.stdout.strip().split(",")[:3]
    num, den = rate.split("/")
    fps = float(num) / float(den)
    H = int(round(int(h) * WS / int(w) / 2)) * 2
    vf = f"select='lt(mod(n\\,{every})\\,2)',scale={WS}:{H},format=gray"
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", video, "-vf", vf, "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    samples, k, size = [], 0, WS * H
    while True:
        a = p.stdout.read(size)
        b = p.stdout.read(size)
        if len(b) < size:
            break
        f0 = np.frombuffer(a, np.uint8).reshape(H, WS)
        f1 = np.frombuffer(b, np.uint8).reshape(H, WS)
        L0, R0 = split_eyes(f0, layout)
        L1, _ = split_eyes(f1, layout)
        mm, wm = pair_scores(L0, R0, L1)
        samples.append((k * every / fps, mm, wm))
        k += 1
        if k % 1000 == 0:
            log(f"  measured {k} frame pairs ({k * every / fps / 60:.1f} min of movie)")
    p.wait()
    return samples


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m iw3.mismatch_plan_cli", description=__doc__.split("\n")[2])
    ap.add_argument("--input", "-i", required=True, help="finished 3D movie (Half TB or Half SBS)")
    ap.add_argument("--scene-plan", required=True, help="the Pre-Plan '..._scene_plan.txt' of that conversion")
    ap.add_argument("--output", "-o", required=True, help="per-scene 3D Strength plan to write (.json)")
    ap.add_argument("--layout", choices=["half_tb", "half_sbs"], default="half_tb")
    ap.add_argument("--max-cut", type=float, default=0.35, help="largest 3D Strength cut for the worst scenes")
    ap.add_argument("--every", type=int, default=12, help="sample one frame pair every N frames (12 = 0.5 s)")
    ap.add_argument("--start-pct", type=float, default=85,
                    help="cutting starts at this percentile of the movie's scenes (85 = worst 15%%; 75 = worst 25%%)")
    ap.add_argument("--full-pct", type=float, default=98,
                    help="the full --max-cut applies from this percentile up (98 = worst 2%%; 90 = worst 10%%)")
    ap.add_argument("--from-plan", default=None,
                    help="re-plan from an earlier plan's saved measurements (no decoding); --input/--scene-plan ignored")
    a = ap.parse_args(argv)
    if a.from_plan:
        old = json.load(open(a.from_plan, encoding="utf-8"))
        scenes = [(s["i"], s["start"], s["dur"]) for s in old["scenes"]]
        per = [(s["moving"], s["whole"], s["samples"]) for s in old["scenes"]]
        a.input, a.scene_plan = old.get("source", "?"), old.get("scene_plan", "?")
    else:
        scenes = read_scene_plan(a.scene_plan)
        if not scenes:
            print("ERROR: no scenes found in the scene plan", file=sys.stderr)
            return 1
        print(f"[mismatch plan] {len(scenes)} scenes; measuring {path.basename(a.input)} ...", file=sys.stderr, flush=True)
        samples = measure_movie(a.input, scenes, a.layout, a.every, log=lambda s: print(s, file=sys.stderr, flush=True))
        per = scene_scores(samples, scenes)
    factors, lo, hi = plan_factors([m for m, _, _ in per], a.max_cut, low_pct=a.start_pct, high_pct=a.full_pct)
    out = {"version": 1, "source": path.basename(a.input), "scene_plan": path.basename(a.scene_plan),
           "max_cut": a.max_cut, "levels": [lo, hi], "scenes": [
               {"i": idx, "start": round(st, 2), "dur": dur, "moving": m, "whole": wv, "samples": n, "factor": f}
               for (idx, st, dur), (m, wv, n), f in zip(scenes, per, factors)]}
    with open(a.output, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=0)
    cut = [s for s in out["scenes"] if s["factor"] < 1.0]
    print(f"[mismatch plan] wrote {a.output}: {len(cut)} of {len(scenes)} scenes get a cut "
          f"({sum(1 for s in cut if s['factor'] <= 1 - a.max_cut + 1e-6)} the full {a.max_cut:.0%}); "
          f"levels {lo} .. {hi}", file=sys.stderr)
    for s in sorted(cut, key=lambda s: s["factor"])[:15]:
        print(f"  scene {s['i']:4} {int(s['start']//60)}:{s['start']%60:04.1f}  moving mismatch {s['moving']:.2f}"
              f"  -> 3D Strength x{s['factor']:.2f}", file=sys.stderr)
    return 0


def _self_test():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x_scene_plan.txt")
        open(p, "w", encoding="utf-8").write("scene start length\n    0   0:00:00.0     2.0s   3.7  Wide\n"
                                              "    1   0:00:02.0     3.0s   3.8  Mid\n")
        assert read_scene_plan(p) == [(0, 0.0, 2.0), (1, 2.0, 3.0)]
    sc = scene_scores([(0.5, 1.0, 2.0), (1.0, float("nan"), 2.0), (2.5, 3.0, 4.0)], [(0, 0.0, 2.0), (1, 2.0, 3.0)])
    assert sc[0][0] == 1.0 and sc[0][2] == 2 and sc[1][0] == 3.0
    f, lo, hi = plan_factors([i / 10 for i in range(100)], 0.35)
    assert f[0] == 1.0 and abs(f[-1] - 0.65) < 1e-9 and lo < hi
    rng = np.random.default_rng(0)
    L = (rng.random((120, 200)) * 255).astype(np.uint8)
    R = np.roll(L, 3, axis=1)                                    # a clean, consistent stereo pair
    L1 = np.roll(L, 6, axis=0)                                   # everything moves
    mm, wm = pair_scores(L, R, L1)
    assert mm == mm and wm < 40, (mm, wm)
    a, b = split_eyes(np.vstack([L, R]), "half_tb")
    assert a.shape == (240, 200) and b.shape == (240, 200)
    print("mismatch_plan_cli self-test: PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)
    sys.exit(main())
