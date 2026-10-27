"""ADR-389: `clip_shot` Convergence Plane mode.

Real request: a convergence mode that judges close-up vs. wide framing using CLIP
(studied -- not imported -- from nt_auto3d/nt_autostrength/autodiv.py's `ClipVisual`,
which does the same "CLIP image tower, zero-shot classify the shot scale" idea for
Auto 3D Strength's --divergence). Core iw3 must never import from nt_auto3d (see
convergence_tracker.py's own module docstring) -- the CLIP ViT-B/32 image tower below
is reimplemented independently, and its weights are cached under this project's own
`iw3/hub_dir.py` HUB_MODEL_DIR, never nt_auto3d's CACHE_DIR.

Deliberate simplification vs. autodiv.py's own version: autodiv ships a small
precomputed prompts file (text embeddings for a handful of "close-up"/"wide shot"
text prompts, computed once offline against CLIP's TEXT tower). Reproducing that here
would mean also reimplementing CLIP's text transformer AND its BPE tokenizer (a ~1.3MB
vocab file this task has no reason to newly bundle) just to re-derive two class
embeddings. Instead, `clip_shot` is "image-only" in a stronger sense: the two class
anchors ("close-up" vs. "wide shot") are themselves CLIP embeddings of two small,
deterministic, PROCEDURALLY generated calibration images (a single large centered soft
blob standing in for "one subject filling the frame", vs. a fine repeating grid
standing in for "a busy/distant wide scene") computed once at model-load time with the
exact same image tower used on real frames -- not real photos, not text, no new
bundled assets. This keeps the classification genuinely zero-shot (no training on
labeled data) and still driven by CLIP's real learned visual features, while staying
fully self-contained inside iw3/. See docs/ai/AI_DECISIONS.md ADR-389 for the exact
weights URL/cache path and the bias-mapping formula.

Falls back gracefully (behaves exactly like plain depth_percentile, i.e. no bias at
all) if the CLIP weights can't be downloaded or loaded -- same spirit as
face_convergence_estimator.py's missing-mediapipe/opencv message.
"""
import math
import os
import sys
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nunif.device import create_device, autocast
from .hub_dir import HUB_MODEL_DIR
from .convergence_tracker import SceneHoldTracker
from .convergence_estimator import ConvergenceEstimator

# Real, public open_clip ViT-B/32 checkpoint (LAION's own release) -- a plain state
# dict (optionally wrapped in {"state_dict": ...}, optionally "module."-prefixed from
# DataParallel training), same key layout as OpenAI's original CLIP ("visual.*" for the
# image tower). Only the "visual.*" half is kept (~176MB in fp16) -- the much larger
# text tower is discarded immediately after download, same approach as autodiv.py's
# own ensure_weights(). NOTE: this exact URL has not been exercised by a live download
# in this sandboxed dev session (no internet/GPU here) -- if it ever 404s/changes, the
# except branch below degrades cleanly to plain depth_percentile rather than crashing;
# fix by updating _WEIGHTS_URL alone, nothing else needs to change.
_WEIGHTS_URL = "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin"
_WEIGHTS_FILE = "clip_vitb32_visual_fp16.pth"

# softmax temperature over the 2-way (close/wide) cosine-similarity affinity -- CLIP's
# own logit_scale is typically ~100 over its full [-1, 1] cosine range; this is tamer
# since our "classes" are synthetic calibration images, not real labeled photos, so the
# similarity gap between them is smaller and a very high temperature would saturate to
# a near-binary 0/1 output on almost any input.
_TEMPERATURE = 20.0
_CALIB_SIZE = 224


def _say(msg):
    print(f"iw3 clip_shot: {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# CLIP ViT-B/32, image tower only -- reimplemented independently from nt_auto3d's
# ClipVisual (studied as a reference, not imported); same standard open_clip/OpenAI
# CLIP key layout so the public checkpoint above loads directly.
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
        h = h * torch.sigmoid(1.702 * h)  # QuickGELU
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
        """rgb 0..1, BCHW, any size -> CLIP input: shorter side to `size`, centre crop,
        normalized."""
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


def _weights_path():
    return path.join(HUB_MODEL_DIR, _WEIGHTS_FILE)


def _ensure_clip_visual_weights(progress=True):
    """Downloads the public open_clip ViT-B/32 checkpoint once, keeps only its image
    tower in fp16 under iw3's own HUB_MODEL_DIR cache, deletes the rest. Returns the
    cached path, or "" if it could not be had (offline, URL changed, disk full...)."""
    dst = _weights_path()
    if path.isfile(dst):
        return dst
    os.makedirs(HUB_MODEL_DIR, exist_ok=True)
    tmp = dst + ".download.tmp"
    tmp_sd = dst + ".sd.tmp"
    try:
        _say(f"downloading the CLIP image encoder once (~600MB) to {HUB_MODEL_DIR}")
        from torch.hub import download_url_to_file
        download_url_to_file(_WEIGHTS_URL, tmp, progress=progress)
        sd = torch.load(tmp, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        vis = {}
        for k, v in sd.items():
            k = k[len("module."):] if k.startswith("module.") else k
            if k.startswith("visual."):
                vis[k[len("visual."):]] = v.half()
        if not vis:
            raise RuntimeError("the download has no image encoder in it")
        torch.save(vis, tmp_sd)
        os.replace(tmp_sd, dst)
        return dst
    except Exception as e:                                           # noqa: BLE001
        _say(f"could not get the CLIP image encoder ({type(e).__name__}: {e}); "
             f"clip_shot will behave like plain depth_percentile (no close-up bias)")
        return ""
    finally:
        for f in (tmp, tmp_sd):
            try:
                os.remove(f)
            except OSError:
                pass


def _close_pattern(size):
    """Deterministic calibration image standing in for "one subject filling the
    frame" (close-up): a single large, soft, centered blob."""
    y = torch.linspace(-1.0, 1.0, size)
    x = torch.linspace(-1.0, 1.0, size)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    r = (gx ** 2 + gy ** 2).sqrt()
    blob = (1.0 - (r / 0.8)).clamp(0, 1)
    return blob.unsqueeze(0).expand(3, size, size).unsqueeze(0).contiguous()


def _wide_pattern(size):
    """Deterministic calibration image standing in for "a busy/distant wide scene":
    a fine repeating grid with no single dominant region."""
    y = torch.linspace(-1.0, 1.0, size)
    x = torch.linspace(-1.0, 1.0, size)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.sin(gx * 8 * math.pi) * torch.sin(gy * 8 * math.pi)
    img = (grid + 1.0) / 2.0
    return img.unsqueeze(0).expand(3, size, size).unsqueeze(0).contiguous()


class ClipShotEstimator():
    """
    Biases the depth_percentile mechanism (ConvergenceEstimator._quantile_position()
    over the full frame) using a CLIP-judged close-up/wide "closeness" score. See
    module docstring for the exact mapping and the fallback behavior.
    """

    # effective_pos = clamp(convergence_bias - closeness * _BIAS_STRENGTH, 0, 1):
    # closeness=0 (wide, or CLIP unavailable) -> effective_pos == convergence_bias,
    # unchanged/neutral. closeness=1 (extreme close-up) -> effective_pos shifted toward
    # 0 by 0.25, biasing toward more foreground pop. 0.25 is a deliberately moderate,
    # documented constant -- not exposed as a new CLI flag (CS-PROJECT-001: Convergence
    # Bias is already the right control for "how strong," this just decides how much of
    # it close-ups get automatically on top of whatever the user already set).
    _BIAS_STRENGTH = 0.25

    def __init__(self, convergence, device_id, enable_ema=False, decay=0.9, scene_hold=False, cut_smooth=0):
        self.convergence = convergence
        self.enable_ema = enable_ema
        self.decay = decay
        self.scene_hold = scene_hold
        self.cut_smooth = cut_smooth
        self.convergence_ema = None
        self.tracker = SceneHoldTracker(decay, cut_smooth)
        self.device = create_device(device_id)
        self.model = None
        self.anchor_close = None
        self.anchor_wide = None

        wp = _ensure_clip_visual_weights()
        if wp:
            try:
                m = ClipVisual()
                m.load_state_dict(torch.load(wp, map_location="cpu", weights_only=True))
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                self.model = m.eval().to(device=self.device, dtype=dtype)
                with torch.inference_mode():
                    self.anchor_close = self._embed(_close_pattern(_CALIB_SIZE).to(self.device))[0]
                    self.anchor_wide = self._embed(_wide_pattern(_CALIB_SIZE).to(self.device))[0]
            except Exception as e:                                   # noqa: BLE001
                _say(f"the CLIP image encoder would not load ({type(e).__name__}: {e}); "
                     f"clip_shot will behave like plain depth_percentile (no close-up bias)")
                self.model = None
                self.anchor_close = None
                self.anchor_wide = None

    def reset(self, enable_ema=None, decay=None, scene_hold=None, cut_smooth=None, **_ignored):
        if enable_ema is not None:
            self.enable_ema = enable_ema
        if decay is not None:
            self.decay = decay
        if scene_hold is not None:
            self.scene_hold = scene_hold
        if cut_smooth is not None:
            self.cut_smooth = cut_smooth
        self.tracker.set_decay(self.decay)
        self.tracker.set_cut_smooth(self.cut_smooth)
        self.tracker.reset()
        self.convergence_ema = None

    def _embed(self, rgb):
        x = self.model.preprocess(rgb)
        e = self.model(x).float()
        return e / e.norm(dim=-1, keepdim=True)

    def _closeness(self, rgb, depth):
        if self.model is None:
            return torch.zeros(rgb.shape[0], device=depth.device)
        with torch.inference_mode(), autocast(self.device):
            emb = self._embed(rgb.to(self.device))
            sim_close = emb @ self.anchor_close
            sim_wide = emb @ self.anchor_wide
            logits = torch.stack([sim_close, sim_wide], dim=-1) * _TEMPERATURE
            closeness = logits.softmax(dim=-1)[:, 0]
        return closeness.to(depth.device).float()

    def __call__(self, rgb, depth, reset_pts=None):
        B = depth.shape[0]
        reset_pts = reset_pts if reset_pts is not None else [False] * B
        closeness = self._closeness(rgb, depth)

        results = []
        for i in range(B):
            eff_pos = self.convergence - closeness[i].item() * self._BIAS_STRENGTH
            eff_pos = max(0.0, min(1.0, eff_pos))
            d = depth[i].flatten()
            p = ConvergenceEstimator._quantile_position(
                d, eff_pos, depth.device, depth.dtype).reshape(1, 1, 1).clamp(0, 1)

            if self.enable_ema:
                if self.scene_hold:
                    out = self.tracker.update(p.item())
                    p = torch.full_like(p, out)
                    if reset_pts[i]:
                        self.tracker.mark_cut()
                else:
                    if self.convergence_ema is None:
                        self.convergence_ema = p.clone()
                    else:
                        self.convergence_ema = self.decay * self.convergence_ema + (1. - self.decay) * p
                    p = self.convergence_ema.clone()
                    if reset_pts[i]:
                        self.convergence_ema = None

            results.append(p)

        return torch.stack(results, dim=0)
