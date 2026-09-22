r"""
Auto 3D Strength -- per-scene divergence, chosen from how close the shot is.

Why iw3 needs it
----------------
iw3 normalises every frame's depth to 0..1 before warping, so a mountain range
gets exactly the same depth budget as a face filling the frame. A real stereo
camera does the opposite: distant scenes have almost no disparity, close-ups a
lot. One fixed 3D Strength is therefore always wrong for somebody -- a
landscape turns into a miniature diorama at the strength a close-up needs, and
a close-up looks flat at the strength a landscape can take.

How close is the shot?
----------------------
Not answerable from iw3's depth: the normalisation above throws the absolute
scale away, and the best depth-only estimate tried (share of far pixels plus
the ground-plane gradient) ranked 32 hand-labelled photos at only 0.59. What a
person uses is meaning -- a face filling the frame IS a close-up -- so this
asks CLIP, zero-shot, which of six shot scales the frame matches -- each
described by several phrasings, averaged ("an extreme wide shot of a vast
landscape", "a distant panoramic view" ... "a macro photo") -- and averages
their closeness values by probability, over three crops that together cover
the whole frame. On the same photos (16:9 crops) that ranks at 0.945, mean
error 0.074 on a 0..1 scale; a single phrasing per scale and one centre crop
managed 0.86 / 0.131, with twice the frame-to-frame noise.

Only CLIP's image half runs (ViT-B/32, 176 MB in fp16, well under a
millisecond a frame on a GPU). The text half was run once, ahead of time, and
its seven embeddings ship in autodiv_prompts.pt. The weights are downloaded on
first use from open_clip's GitHub release, checked against their SHA-256, cut
down to the image half and cached in <nunif>\nt_auto3d\cache\. Without them it falls
back to the depth-only estimate and says so.

From closeness to strength
--------------------------
    strength = min + (max - min) * s ** gamma

with gamma chosen so that s = 0.5 (a medium shot) lands exactly on iw3's own
3D Strength box. So Min / Max bound it, and 3D Strength is "what an ordinary
shot gets". 2 / 16 with 3D Strength 6: wide shot ~2.8, medium 6, close-up
~11.4, extreme close-up 16. Measured on a test video built from public photos:
skyline 2.4, aerial view 2.8, street 2.7, interior 4.1, man at a stall 5.6,
pumpkins 11.7, flower macro 15.2.

Over time
---------
  cuts    decided at the start of each scene and held until the next cut
  smooth  follows the framing slowly and never jumps, not even at cuts
  hybrid  jumps at cuts, then follows slowly within the shot

Cuts come from iw3's own scene detection when it is on. When it is off, a cut
is a frame whose CLIP embedding is unlike the previous one (cosine < 0.70):
frames of one shot measured >= 0.87 even when moved or zoomed 10%, different
shots <= 0.54.

How it is applied
-----------------
No iw3 file is edited. `iw3.utils.apply_divergence` is wrapped: it works out a
strength per frame and calls the original once per run of equal values with
`args.divergence` set to it, so every method -- forward_inpaint, mlbw, row_flow
-- gets it. The CLI gains --auto-divergence, --auto-divergence-mode,
--divergence-min, --divergence-max and --auto-divergence-overlay (debug: the
strength used, written into each frame's corner); the iw3 window gains the
same controls under 3D Strength, saved with the rest of its settings.

NT_AUTODIV_DISABLE=1 turns all of it off, including the controls.
"""
from __future__ import annotations

import collections
import copy
import math
import os
import sys
import threading
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["install", "ensure_weights", "strength_curve", "MODES"]

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(HERE)                               # <nunif>\nt_auto3d
CACHE_DIR = path.join(ROOT, "cache")
PROMPTS_FILE = path.join(HERE, "autodiv_prompts.pt")
VISUAL_FILE = "clip_vitb32_laion400m_visual_fp16.pth"

MODES = ("hybrid", "cuts", "smooth")
DEFAULT_MIN, DEFAULT_MAX, DEFAULT_MODE = 2.0, 16.0, "hybrid"

CUT_SIMILARITY = 0.70      # below this, consecutive frames are different shots
STEP = 0.1                 # strengths are rounded to this, so runs batch together
# Averaging CLIP's probabilities pulls the extremes in: on 32 hand-labelled
# photos (16:9 crops, as video is) raw scores only spanned 0.04-0.75. This
# linear stretch -- two numbers fitted to those labels -- puts them back on
# 0..1 (mean error 0.133 -> 0.124) without changing their order.
CALIB = (1.39, -0.15)

_installed = False
_lock = threading.Lock()


def _say(msg):
    print(f"iw3 auto strength: {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# CLIP ViT-B/32, image half only -- same layout and key names as open_clip
# --------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.Module()
        self.attn.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.attn.in_proj_bias = nn.Parameter(torch.empty(3 * width))
        self.attn.out_proj = nn.Linear(width, width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(width, width * 4)
        self.mlp.c_proj = nn.Linear(width * 4, width)

    def forward(self, x):
        B, N, C = x.shape
        h = self.ln_1(x)
        q, k, v = F.linear(h, self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, N, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        x = x + self.attn.out_proj(a)
        h = self.mlp.c_fc(self.ln_2(x))
        h = h * torch.sigmoid(1.702 * h)                    # QuickGELU
        return x + self.mlp.c_proj(h)


class ClipVisual(nn.Module):
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, width=768, layers=12, heads=12, patch=32, size=224, dim=512):
        super().__init__()
        self.size = size
        self.conv1 = nn.Conv2d(3, width, patch, patch, bias=False)
        self.class_embedding = nn.Parameter(torch.empty(width))
        self.positional_embedding = nn.Parameter(torch.empty((size // patch) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(_Block(width, heads) for _ in range(layers))
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.empty(width, dim))

    def preprocess(self, rgb):
        """rgb 0..1, BCHW, any size -> CLIP input: shorter side to 224, centre
        crop, normalised. Same steps as open_clip's own transform."""
        H, W = rgb.shape[-2:]
        s = self.size / min(H, W)
        h, w = max(self.size, round(H * s)), max(self.size, round(W * s))
        x = F.interpolate(rgb.float(), size=(h, w), mode="bicubic", antialias=True,
                          align_corners=False).clamp(0, 1)
        top, left = (h - self.size) // 2, (w - self.size) // 2
        x = x[..., top:top + self.size, left:left + self.size]
        mean = torch.tensor(self.MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, x):
        x = self.conv1(x.to(self.conv1.weight.dtype)).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for blk in self.transformer.resblocks:
            x = blk(x)
        x = self.ln_post(x[:, 0])
        return x @ self.proj


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
def _prompts():
    return torch.load(PROMPTS_FILE, map_location="cpu", weights_only=True)


def weights_path():
    return path.join(CACHE_DIR, VISUAL_FILE)


def ensure_weights(progress=True):
    """Make sure the image half of CLIP is cached. Downloads the open_clip
    release once (605 MB), verifies its SHA-256 prefix, keeps only the image
    half in fp16 (176 MB) and deletes the rest. Returns the cached path, or ""
    if it could not be had (offline, disk full...)."""
    dst = weights_path()
    if path.isfile(dst):
        return dst
    meta = _prompts()
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path.join(CACHE_DIR, "clip_download.tmp")
    try:
        _say(f"downloading the shot-framing model once (605 MB) to {CACHE_DIR}")
        from torch.hub import download_url_to_file
        download_url_to_file(meta["url"], tmp, hash_prefix=meta["sha256_prefix"],
                             progress=progress)
        sd = torch.load(tmp, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        vis = {}
        for k, v in sd.items():
            k = k[len("module."):] if k.startswith("module.") else k
            if k.startswith("visual."):
                vis[k[len("visual."):]] = v.half()
        if not vis:
            raise RuntimeError("the download has no image model in it")
        torch.save(vis, dst + ".part")
        os.replace(dst + ".part", dst)
        return dst
    except Exception as e:                                          # noqa: BLE001
        _say(f"could not get the shot-framing model ({type(e).__name__}: {e}); "
             f"using the rougher depth-only estimate instead")
        return ""
    finally:
        for f in (tmp, dst + ".part"):
            try:
                os.remove(f)
            except OSError:
                pass


# --------------------------------------------------------------------------
# closeness 0..1 (0 = extreme wide, 1 = extreme close-up)
# --------------------------------------------------------------------------
class Estimator:
    def __init__(self, device):
        self.device = device
        self.model = None
        meta = _prompts()
        self.text = meta["text"].to(device)
        self.values = meta["values"].to(device)
        self.temperature = float(meta["temperature"])
        self.calib = tuple(meta.get("calib", CALIB))
        self.three_crops = meta.get("views", "center") == "3crop"
        wp = ensure_weights()
        if wp:
            try:
                m = ClipVisual()
                m.load_state_dict(torch.load(wp, map_location="cpu", weights_only=True))
                dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
                self.model = m.eval().to(device=device, dtype=dtype)
            except Exception as e:                                  # noqa: BLE001
                _say(f"the shot-framing model would not load ({type(e).__name__}: {e}); "
                     f"using the depth-only estimate")
                self.model = None

    def _views(self, rgb):
        """Left, centre and right squares of the frame (one if it is square).
        A centre crop of a 16:9 frame drops 22% off each side, and whatever
        moves across that line changes the score; averaging three crops that
        cover the whole frame halved the frame-to-frame noise (closeness std
        0.022 -> 0.012 on 12 panned/zoomed copies of each test photo)."""
        size = self.model.size
        H, W = rgb.shape[-2:]
        s = size / min(H, W)
        h, w = max(size, round(H * s)), max(size, round(W * s))
        x = F.interpolate(rgb.float(), size=(h, w), mode="bicubic", antialias=True,
                          align_corners=False).clamp(0, 1)
        if self.three_crops and w > size:
            offs = [(0, 0), (0, (w - size) // 2), (0, w - size)]
        elif self.three_crops and h > size:
            offs = [(0, 0), ((h - size) // 2, 0), (h - size, 0)]
        else:
            offs = [((h - size) // 2, (w - size) // 2)]
        mean = torch.tensor(ClipVisual.MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(ClipVisual.STD, device=x.device).view(1, 3, 1, 1)
        return [(x[..., t:t + size, l:l + size] - mean) / std for t, l in offs]

    @torch.inference_mode()
    def __call__(self, rgb, depth):
        """-> (closeness[B], embedding[B, 512] or None)"""
        if self.model is None:
            return depth_closeness(depth), None
        views = self._views(rgb.to(self.device))
        B = views[0].shape[0]
        e = self.model(torch.cat(views, 0).to(self.model.conv1.weight.dtype)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.view(len(views), B, -1).mean(0)
        e = e / e.norm(dim=-1, keepdim=True)
        p = (self.temperature * e @ self.text.T).softmax(dim=-1)
        return (self.calib[0] * (p @ self.values) + self.calib[1]).clamp(0, 1), e


def depth_closeness(depth):
    """Fallback without CLIP: mean depth (1 = near) and the ground-plane cue
    (bottom near, top far). Rank 0.59 against hand labels -- rough."""
    d = depth.float()
    H = d.shape[-2]
    mean = d.mean(dim=(-1, -2, -3))
    tb = d[..., 2 * H // 3:, :].mean(dim=(-1, -2, -3)) - d[..., :H // 3, :].mean(dim=(-1, -2, -3))
    return (0.958 * mean - 0.526 * tb + 0.282).clamp(0, 1)


# --------------------------------------------------------------------------
# closeness -> strength, and the behaviour over time
# --------------------------------------------------------------------------
def strength_curve(s, lo, typical, hi):
    """lo at s=0, hi at s=1, `typical` at s=0.5 (a medium shot)."""
    lo, hi = min(lo, hi), max(lo, hi)
    if hi - lo < 1e-6:
        return lo
    t = min(max((typical - lo) / (hi - lo), 0.02), 0.98)
    gamma = math.log(t) / math.log(0.5)
    return lo + (hi - lo) * min(max(float(s), 0.0), 1.0) ** gamma


# How settled the strength is. Per frame (so at 24-30 fps):
#   settle    frames after a cut over which the scene's value is averaged
#   drift     how fast it follows the framing within a shot (hybrid, smooth)
#   deadband  how far (in strength) the target must move before it follows
STABILITY = {
    "low":       dict(settle=6,  drift=0.05,  deadband=0.25),
    "medium":    dict(settle=12, drift=0.02,  deadband=0.5),
    "high":      dict(settle=24, drift=0.008, deadband=0.8),
    "very high": dict(settle=36, drift=0.003, deadband=1.2),
}
DEFAULT_STABILITY = "medium"


def _stability(name):
    name = (name or DEFAULT_STABILITY).replace("-", " ").replace("_", " ").strip().lower()
    return STABILITY.get(name, STABILITY[DEFAULT_STABILITY])


class Tracker:
    """Per-frame closeness in, per-frame strength out.

    After a cut (cuts / hybrid) the scene's value is the running mean of the
    frames seen since the cut, for `settle` frames -- the first frame alone is
    a noisy guess. Then cuts holds it; hybrid lets it drift slowly with the
    framing; smooth drifts always and never jumps. On top of all of that, the
    applied strength only moves once the target is `deadband` away -- during
    settling too -- and then glides to it, so small wobbles never reach the
    picture and a higher stability always means fewer visible changes."""

    def __init__(self, mode, stability=DEFAULT_STABILITY):
        self.mode = mode if mode in MODES else DEFAULT_MODE
        self.p = _stability(stability)
        self.reset()

    def reset(self):
        self.s = None             # the closeness currently in use
        self.n = 0                # frames seen in this scene
        self.out = None           # the strength currently applied
        self.moving = False
        self.prev_emb = None
        self.cut_pending = True   # the next frame starts a scene

    def run(self, scores, embs, reset_pts, own_cuts, curve):
        out = []
        p = self.p
        for i, sc in enumerate(scores):
            cut = self.cut_pending
            self.cut_pending = False
            if own_cuts and embs is not None and self.prev_emb is not None:
                cut = cut or float(embs[i] @ self.prev_emb) < CUT_SIMILARITY
            if embs is not None:
                self.prev_emb = embs[i]
            if reset_pts is not None and i < len(reset_pts) and reset_pts[i]:
                self.cut_pending = True      # iw3 resets AFTER this frame

            sc = float(sc)
            if self.s is None:
                self.s, self.n, cut = sc, 1, True
            elif cut and self.mode != "smooth":
                self.s, self.n = sc, 1
            else:
                self.n += 1
                if self.mode != "smooth" and self.n <= p["settle"]:
                    self.s += (sc - self.s) / self.n            # running mean of the scene so far
                elif self.mode != "cuts":
                    self.s += (sc - self.s) * p["drift"]

            target = curve(self.s)
            if self.out is None or (cut and self.mode != "smooth"):
                self.out, self.moving = target, False           # a cut hides the change
            else:
                gap = target - self.out
                if not self.moving and abs(gap) >= p["deadband"]:
                    self.moving = True
                if self.moving:
                    self.out += gap * 0.1                       # glide, ~10 frames
                    if abs(target - self.out) < 0.05:
                        self.out, self.moving = target, False
            out.append(self.out)
        return out


class _RunState:
    """Lives in args.state, so every run starts fresh."""

    def __init__(self, args):
        self.estimator = Estimator(args.state.get("device", "cpu")
                                   if isinstance(args.state, dict) else "cpu")
        self.tracker = Tracker(getattr(args, "auto_divergence_mode", DEFAULT_MODE),
                               getattr(args, "auto_divergence_stability", DEFAULT_STABILITY))
        self.last_depth = None    # the depth tensor of the last call, held so its id stays unique
        self.last_divs = None
        self.repeat = False       # this call is iw3's alpha pass over the same frames
        # strengths of frames handed to iw3 but not yet given back: the inpaint
        # methods hold frames in a queue and return them several calls later,
        # so the overlay must label each frame with the strength it was warped
        # with, not the one being worked out now
        self.pending = collections.deque()
        self.announced = False
        self.verbose = bool(os.environ.get("NT_INPAINT_VERBOSE"))

    def divergences(self, im, depth, args, reset_pts, same_obj=None):
        # iw3 warps the alpha channel with a second call on the SAME depth
        # tensor; that call must get the same strengths and must not advance
        # anything. Identified by object identity -- NOT by comparing values:
        # on a static stretch of video two different batches have identical
        # depth, and a value match wrongly skipped them (label gone, and the
        # labels behind it shifted late).
        obj = depth if same_obj is None else same_obj
        if obj is self.last_depth and self.last_divs is not None:
            self.repeat = True
            return self.last_divs
        self.repeat = False
        scores, embs = self.estimator(im, depth)
        scores = [float(s) for s in scores]
        own_cuts = not getattr(args, "scene_detect", False)
        lo = float(getattr(args, "divergence_min", DEFAULT_MIN))
        hi = float(getattr(args, "divergence_max", DEFAULT_MAX))
        typical = float(args.divergence)
        used = self.tracker.run(scores, embs, reset_pts, own_cuts,
                                lambda c: strength_curve(c, lo, typical, hi))
        divs = [round(d / STEP) * STEP for d in used]
        if self.verbose or not self.announced:
            self.announced = True
            _say(f"{self.tracker.mode}, {lo:g}-{hi:g} around {typical:g}"
                 + ("" if self.estimator.model is not None else " (depth-only estimate)")
                 + f"; now {divs[-1]:.1f}")
        self.last_depth, self.last_divs = obj, divs
        log = os.environ.get("NT_AUTODIV_LOG")
        if log:
            try:
                with open(log, "a", encoding="utf-8") as f:
                    for raw, d in zip(scores, divs):
                        self.logged = getattr(self, "logged", 0) + 1
                        f.write(f"{self.logged},{raw:.4f},{self.tracker.s:.4f},{d:.1f}\n")
            except OSError:
                pass
        return divs


_ACTIVE = []                # the run state in use, so an iw3 queue reset can reach it


def _overlay_on(args):
    return bool(getattr(args, "auto_divergence_overlay", False)
                or os.environ.get("NT_AUTODIV_OVERLAY"))


_labels = {}


def _label(text, h):
    """White text on black, as a 0..1 alpha mask (H, W), h pixels tall. Cached."""
    key = (text, h)
    m = _labels.get(key)
    if m is None:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.load_default(size=h)          # Pillow >= 10.1
            x0, y0, x1, y1 = font.getbbox(text)
            im = Image.new("L", (x1 + h // 2, y1 + h // 3), 0)
            ImageDraw.Draw(im).text((h // 4, h // 8), text, fill=255, font=font)
            m = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
        except TypeError:                                   # older Pillow: bitmap font, scaled up
            font = ImageFont.load_default()
            x0, y0, x1, y1 = font.getbbox(text)
            im = Image.new("L", (x1 + 4, y1 + 3), 0)
            ImageDraw.Draw(im).text((2, 1), text, fill=255, font=font)
            m = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
            scale = max(1, round(h / max(1, y1)))
            m = m.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        _labels[key] = m
    return m


def _stamp(frames, values):
    """Write each frame's strength into its top-left corner. frames: BCHW or CHW."""
    if frames is None or not torch.is_tensor(frames) or not values:
        return frames
    single = frames.ndim == 3
    x = frames.unsqueeze(0) if single else frames
    x = x.clone()                     # never draw into a tensor iw3 still owns
    H, W = x.shape[-2:]
    h = max(12, H // 22)
    y0 = x0 = h // 3
    for i in range(x.shape[0]):
        v = values[min(i, len(values) - 1)]
        m = _label(f"3D {v:.1f}", h).to(device=x.device, dtype=x.dtype)
        mh, mw = min(m.shape[0], H - y0), min(m.shape[1], W - x0)
        if mh <= 0 or mw <= 0:
            continue
        m = m[:mh, :mw]
        r = x[i, :, y0:y0 + mh, x0:x0 + mw]
        r.copy_(r * 0.3 * (1 - m) + m)      # darkened box, white text
    return x[0] if single else x


def _state(args):
    st = args.state.get("nt_autodiv") if isinstance(args.state, dict) else None
    if st is None:
        st = _RunState(args)
        if isinstance(args.state, dict):
            args.state["nt_autodiv"] = st
    _ACTIVE[:] = [st]
    return st


def _enabled(args):
    return bool(getattr(args, "auto_divergence", False)) and isinstance(
        getattr(args, "state", None), dict)


# --------------------------------------------------------------------------
# patches
# --------------------------------------------------------------------------
def _cat(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _patch_utils(U):
    if getattr(U.apply_divergence, "_nt_autodiv", False):
        return
    orig = U.apply_divergence

    def apply_divergence(depth, im, args, side_model, reset_pts=None):
        if not _enabled(args) or not torch.is_tensor(depth) or not torch.is_tensor(im):
            return orig(depth, im, args, side_model, reset_pts=reset_pts)
        with _lock:
            st = _state(args)
            batch = depth.ndim == 4
            d4, i4 = (depth, im) if batch else (depth.unsqueeze(0), im.unsqueeze(0))
            divs = st.divergences(i4, d4, args, reset_pts, same_obj=depth)
        overlay = _overlay_on(args) and not st.repeat

        def run(d, x, value, count, rp):
            a = copy.copy(args)                # args.state is shared, as it should be
            a.divergence = value
            left, right = orig(d, x, a, side_model, reset_pts=rp)
            if overlay:
                st.pending.extend([value] * count)
                if left is not None:
                    n_out = 1 if left.ndim == 3 else left.shape[0]
                    vals = [st.pending.popleft() if st.pending else value for _ in range(n_out)]
                    left, right = _stamp(left, vals), _stamp(right, vals)
            return left, right

        if not batch:
            return run(depth, im, divs[0], 1, reset_pts)
        lefts, rights = [], []
        i, n = 0, len(divs)
        while i < n:
            j = i + 1
            while j < n and divs[j] == divs[i]:
                j += 1
            rp = None if reset_pts is None else list(reset_pts[i:j])
            if i == 0 and j == n:
                return run(depth, im, divs[i], n, rp)
            left, right = run(depth[i:j], im[i:j], divs[i], j - i, rp)
            lefts.append(left)
            rights.append(right)
            i = j
        return _cat(lefts), _cat(rights)

    apply_divergence._nt_autodiv = True
    U.apply_divergence = apply_divergence

    # each video and each image starts fresh
    def fresh(fn, independent=False):
        def wrapper(*a, **kw):
            args = kw.get("args")
            if args is None:
                args = next((x for x in a if hasattr(x, "divergence") and hasattr(x, "state")), None)
            if args is not None and _enabled(args):
                st = args.state.get("nt_autodiv")
                if st is not None:
                    st.tracker.reset()
                    st.last_depth = None
                    st.pending.clear()
            return fn(*a, **kw)
        wrapper._nt_autodiv = True
        wrapper.__wrapped__ = fn
        return wrapper

    for name in ("process_video", "process_config_video", "process_image"):
        fn = getattr(U, name, None)
        if fn is not None and not getattr(fn, "_nt_autodiv", False):
            setattr(U, name, fresh(fn))

    # iw3 empties the inpaint frame queue with side_model.reset() (e.g. after
    # its output-size probe) without returning those frames; drop their labels
    try:
        import iw3.base_inpaint as BI
        if not getattr(BI.BaseInpaint.reset, "_nt_autodiv", False):
            orig_reset = BI.BaseInpaint.reset

            def reset(self, *a, **kw):
                for st in _ACTIVE:
                    st.pending.clear()
                return orig_reset(self, *a, **kw)

            reset._nt_autodiv = True
            BI.BaseInpaint.reset = reset
    except Exception:                                              # noqa: BLE001
        pass

    orig_parser = U.create_parser

    def create_parser(*a, **kw):
        p = orig_parser(*a, **kw)
        g = p.add_argument_group("auto 3D strength")
        g.add_argument("--auto-divergence", action="store_true",
                       help="choose the 3D strength per scene from how close the shot is; "
                            "--divergence becomes what a medium shot gets")
        g.add_argument("--auto-divergence-mode", choices=MODES, default=DEFAULT_MODE,
                       help="cuts: decided per scene and held; smooth: follows slowly, never "
                            "jumps; hybrid: jumps at cuts, follows slowly within a shot")
        g.add_argument("--divergence-min", type=float, default=DEFAULT_MIN,
                       help="auto 3D strength: the least a very wide shot gets")
        g.add_argument("--divergence-max", type=float, default=DEFAULT_MAX,
                       help="auto 3D strength: the most an extreme close-up gets")
        g.add_argument("--auto-divergence-stability", default=DEFAULT_STABILITY,
                       choices=["low", "medium", "high", "very-high"],
                       help="how settled the strength is: low follows the framing quickly, "
                            "very-high barely moves within a shot")
        g.add_argument("--auto-divergence-overlay", action="store_true",
                       help="debug: write the strength used into the top-left of every frame")
        return p

    create_parser._nt_autodiv = True
    U.create_parser = create_parser


# ---- the iw3 window ------------------------------------------------------
# iw3's launchers start the window with `python -m iw3.gui`, which runs gui.py as
# __main__ -- it is never imported under the name iw3.gui, so nothing can wait
# for that name. What it does import by name is nunif.gui, for the helpers that
# save and restore every control. The window's first call to
# persistent_manager_register_all is the last step of building it: layout done,
# settings not yet restored. The new controls go in right there, so their saved
# values come back with everything else.
def _is_iw3_window(w):
    return (hasattr(w, "grp_stereo") and hasattr(w, "cbo_divergence")
            and hasattr(w, "parse_args") and hasattr(w, "get_editable_comboboxes"))


def _add_controls(self, T, EditableComboBox):
    import wx
    box = self.grp_stereo
    self.chk_nt_auto_div = wx.CheckBox(box, label=T("Auto 3D Strength"), name="chk_nt_auto_div")
    self.chk_nt_auto_div.SetToolTip(
        "Choose the 3D Strength per scene from how close the shot is:\n"
        "wide shots and landscapes get less, close-ups get more.\n"
        "3D Strength above becomes what an ordinary (medium) shot gets.\n\n"
        "hybrid: changes at cuts, then follows the framing slowly\n"
        "cuts: decided at each cut and held until the next\n"
        "smooth: follows the framing slowly, never jumps\n\n"
        "Turn on Scene Boundary Detection for the cleanest cuts; without it,\n"
        "cuts are detected from the picture.")
    self.cbo_nt_auto_div_mode = wx.ComboBox(box, choices=list(MODES), name="cbo_nt_auto_div_mode")
    self.cbo_nt_auto_div_mode.SetEditable(False)
    self.cbo_nt_auto_div_mode.SetSelection(0)
    self.lbl_nt_div_range = wx.StaticText(box, label=T("Auto Range (min, max)"))
    self.cbo_nt_div_min = EditableComboBox(box, choices=["1.0", "2.0", "3.0", "4.0"],
                                           name="cbo_nt_div_min")
    self.cbo_nt_div_min.SetValue(f"{DEFAULT_MIN:.1f}")
    self.cbo_nt_div_min.SetToolTip("The least a very wide shot or landscape gets")
    self.cbo_nt_div_max = EditableComboBox(box, choices=["8.0", "10.0", "12.0", "16.0", "20.0"],
                                           name="cbo_nt_div_max")
    self.cbo_nt_div_max.SetValue(f"{DEFAULT_MAX:.1f}")
    self.cbo_nt_div_max.SetToolTip("The most an extreme close-up gets")
    self.lbl_nt_auto_div_stab = wx.StaticText(box, label=T("Auto Stability"))
    self.cbo_nt_auto_div_stab = wx.ComboBox(box, choices=list(STABILITY),
                                            name="cbo_nt_auto_div_stab")
    self.cbo_nt_auto_div_stab.SetEditable(False)
    self.cbo_nt_auto_div_stab.SetSelection(list(STABILITY).index(DEFAULT_STABILITY))
    self.cbo_nt_auto_div_stab.SetToolTip(
        "How settled the strength is.\n"
        "low: follows the framing quickly, reacts to small changes\n"
        "medium: settles over ~half a second after a cut, then ignores\n"
        "        changes under 0.5\n"
        "high / very high: settles over 1-1.5 s and barely moves within a shot")
    self.chk_nt_auto_div_overlay = wx.CheckBox(box, label=T("Show strength on video (debug)"),
                                               name="chk_nt_auto_div_overlay")
    self.chk_nt_auto_div_overlay.SetToolTip(
        "Writes the 3D Strength used for each frame into its top-left corner,\n"
        "in both eyes, so you can see what Auto 3D Strength decided.\n"
        "For testing - it is burned into the output video.")

    # four rows directly under 3D Strength (and the warning line under it)
    grid = self.cbo_divergence.GetContainingSizer()
    row = grid.GetItemPosition(self.cbo_divergence).GetRow() + 2
    items = [(it, it.GetPos()) for it in grid.GetChildren()]
    for it, pos in sorted(items, key=lambda t: -t[1].GetRow()):
        if pos.GetRow() >= row:
            target = it.GetWindow() or it.GetSizer()
            grid.SetItemPosition(target, wx.GBPosition(pos.GetRow() + 4, pos.GetCol()))
    grid.Add(self.chk_nt_auto_div, (row, 0), flag=wx.ALIGN_CENTER_VERTICAL)
    grid.Add(self.cbo_nt_auto_div_mode, (row, 1), (1, 2), flag=wx.EXPAND)
    grid.Add(self.lbl_nt_div_range, (row + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
    grid.Add(self.cbo_nt_div_min, (row + 1, 1), flag=wx.EXPAND)
    grid.Add(self.cbo_nt_div_max, (row + 1, 2), flag=wx.EXPAND)
    grid.Add(self.lbl_nt_auto_div_stab, (row + 2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
    grid.Add(self.cbo_nt_auto_div_stab, (row + 2, 1), (1, 2), flag=wx.EXPAND)
    grid.Add(self.chk_nt_auto_div_overlay, (row + 3, 1), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
    self._nt_T = T
    self.chk_nt_auto_div.Bind(wx.EVT_CHECKBOX, lambda e: _update(self))
    self._nt_autodiv_ready = True


def _update(self):
    if not getattr(self, "_nt_autodiv_ready", False):
        return
    on = self.chk_nt_auto_div.IsChecked()
    for c in (self.cbo_nt_auto_div_mode, self.lbl_nt_div_range,
              self.cbo_nt_div_min, self.cbo_nt_div_max, self.lbl_nt_auto_div_stab,
              self.cbo_nt_auto_div_stab, self.chk_nt_auto_div_overlay):
        c.Enable(on)
    T = getattr(self, "_nt_T", lambda s: s)
    self.lbl_divergence.SetLabel(T("3D Strength") + (" (typical)" if on else ""))
    try:
        self.Layout()
    except Exception:                                              # noqa: BLE001
        pass


def _patch_window_class(MF):
    if getattr(MF, "_nt_autodiv_patched", False):
        return
    MF._nt_autodiv_patched = True

    orig_combos = MF.get_editable_comboboxes

    def get_editable_comboboxes(self):
        extra = [c for c in (getattr(self, "cbo_nt_div_min", None),
                             getattr(self, "cbo_nt_div_max", None)) if c is not None]
        return list(orig_combos(self)) + extra

    MF.get_editable_comboboxes = get_editable_comboboxes

    if hasattr(MF, "update_controls"):
        orig_update = MF.update_controls

        def update_controls(self, *a, **kw):
            r = orig_update(self, *a, **kw)
            _update(self)
            return r

        MF.update_controls = update_controls

    orig_parse = MF.parse_args

    def parse_args(self, *a, **kw):
        ready = getattr(self, "_nt_autodiv_ready", False)
        on = ready and self.chk_nt_auto_div.IsChecked()
        if on:
            T = getattr(self, "_nt_T", lambda s: s)
            for c, label in ((self.cbo_nt_div_min, "Auto Range (min)"),
                             (self.cbo_nt_div_max, "Auto Range (max)")):
                try:
                    ok = 0.0 <= float(c.GetValue()) <= 100.0
                except ValueError:
                    ok = False
                if not ok:
                    self.show_validation_error_message(T(label), 0.0, 100.0)
                    return None
        args = orig_parse(self, *a, **kw)
        if args is None or not ready:
            return args
        args.auto_divergence = bool(on)
        args.auto_divergence_mode = self.cbo_nt_auto_div_mode.GetValue() or DEFAULT_MODE
        args.divergence_min = float(self.cbo_nt_div_min.GetValue())
        args.divergence_max = float(self.cbo_nt_div_max.GetValue())
        args.auto_divergence_overlay = bool(on) and self.chk_nt_auto_div_overlay.IsChecked()
        args.auto_divergence_stability = (self.cbo_nt_auto_div_stab.GetValue()
                                          or DEFAULT_STABILITY).replace(" ", "-")
        return args

    MF.parse_args = parse_args


def _patch_nunif_gui(NG):
    orig = getattr(NG, "persistent_manager_register_all", None)
    if orig is None or getattr(orig, "_nt_autodiv", False):
        return

    def persistent_manager_register_all(manager, window):
        if _is_iw3_window(window) and not getattr(window, "_nt_autodiv_ready", False):
            try:
                mod = sys.modules.get(type(window).__module__)
                T = getattr(mod, "T", lambda s: s)
                _patch_window_class(type(window))
                _add_controls(window, T, NG.EditableComboBox)
            except Exception as e:                                  # noqa: BLE001
                window._nt_autodiv_ready = False
                _say(f"could not add the controls to the iw3 window "
                     f"({type(e).__name__}: {e}); the CLI options still work")
        return orig(manager, window)

    persistent_manager_register_all._nt_autodiv = True
    NG.persistent_manager_register_all = persistent_manager_register_all


def _after_import(name, fn):
    """Run fn(module) once `name` has been imported -- now, or whenever it is."""
    if name in sys.modules:
        fn(sys.modules[name])
        return

    class _Loader:
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                fn(module)
            except Exception as e:                                  # noqa: BLE001
                _say(f"could not extend {name} ({type(e).__name__}: {e})")

        def __getattr__(self, attr):
            return getattr(self._inner, attr)

    class _Finder:
        def find_spec(self, fullname, p=None, target=None):
            if fullname != name:
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                return None
            spec = None                      # rest of meta_path, so other hooks still fire
            for finder in list(sys.meta_path):
                find = getattr(finder, "find_spec", None)
                if find is not None:
                    spec = find(fullname, p, target)
                    if spec is not None:
                        break
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())


def install() -> bool:
    """Patch iw3 in place. Safe to call more than once."""
    global _installed
    if _installed:
        return True
    if os.environ.get("NT_AUTODIV_DISABLE"):
        return False
    import iw3.utils as U
    _patch_utils(U)
    try:
        _after_import("nunif.gui", _patch_nunif_gui)
    except Exception as e:                                          # noqa: BLE001
        _say(f"the iw3 window will not show the controls ({type(e).__name__}: {e})")
    _installed = True
    return True
