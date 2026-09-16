import os
import sys
from os import path
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from nunif.device import create_device, autocast, device_is_mps, device_is_xpu  # noqa
from .dilation import dilate_edge, edge_dilation_is_enabled
from .base_depth_model import BaseDepthModel, HUB_MODEL_DIR


# ADR-155: Metric3D v2 (https://github.com/YvanYin/Metric3D, BSD-2-Clause).
# Unlike every other depth model in this file's siblings, this one is NOT loaded
# through a nagadomi-maintained torch.hub fork -- nagadomi has never forked this
# repo, and only nagadomi's own GitHub org is used by the existing pattern (see
# zoedepth_model.py / depth_anything_v3_model.py / depth_pro_model.py, all of
# which torch.hub.load from "nagadomi/..._iw3"). Rather than stand up and publish
# a brand new public GitHub repo just to mirror that convention for one model,
# the (small, ~260KB, pure Python, no compiled extensions) upstream `mono/`
# package is vendored directly into iw3/thirdparty/metric3d/ instead, matching
# this project's own "self-contained, no extra network dependency" distribution
# philosophy. LICENSE (BSD-2-Clause, permissive, redistribution-with-notice is
# explicitly allowed) is kept alongside it in that same directory.
#
# One real bug in the vendored copy was patched by hand: mono/utils/comm.py
# unconditionally did `from mmcv.utils import collect_env as collect_base_env`
# for a function (collect_env()) that is dead code -- commented out immediately
# below that same import. mmcv is a heavy, often compile-from-source dependency
# this project does not otherwise need; the only two REAL mmcv usages elsewhere
# in this vendored copy (hubconf-equivalent Config loading, ViT_DINO_reg.py's own
# Config import) already have their own try/except fallback to the much lighter
# `mmengine` package built in by the original authors. The comm.py import was
# simply missing that same fallback for a function nothing calls -- wrapped it in
# the same try/except pattern (assigns None on failure) rather than installing
# mmcv. Confirmed via live test: model loads and runs correctly with mmengine
# alone, mmcv never installed.
THIRDPARTY_ROOT = path.join(path.dirname(__file__), "thirdparty", "metric3d")

# Real Hugging Face checkpoint URLs and config paths, copied from the upstream
# repo's own hubconf.py MODEL_TYPE dict (JUGGHM/Metric3D HF repo) -- confirmed
# correct for all 5 variants by reading that file directly, not guessed.
MODEL_TYPE = {
    "Metric3D_ConvNeXt_Tiny": {
        "cfg_rel_path": path.join("mono", "configs", "HourglassDecoder", "convtiny.0.3_150.py"),
        "ckpt_url": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth",
        "input_size": (544, 1216),
    },
    "Metric3D_ConvNeXt_Large": {
        "cfg_rel_path": path.join("mono", "configs", "HourglassDecoder", "convlarge.0.3_150.py"),
        "ckpt_url": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth",
        "input_size": (544, 1216),
    },
    "Metric3D_ViT_Small": {
        "cfg_rel_path": path.join("mono", "configs", "HourglassDecoder", "vit.raft5.small.py"),
        "ckpt_url": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth",
        "input_size": (616, 1064),
    },
    "Metric3D_ViT_Large": {
        "cfg_rel_path": path.join("mono", "configs", "HourglassDecoder", "vit.raft5.large.py"),
        "ckpt_url": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth",
        "input_size": (616, 1064),
    },
    "Metric3D_ViT_Giant2": {
        "cfg_rel_path": path.join("mono", "configs", "HourglassDecoder", "vit.raft5.giant2.py"),
        "ckpt_url": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth",
        "input_size": (616, 1064),
    },
}
MODEL_FILES = {
    name: path.join(HUB_MODEL_DIR, "checkpoints", path.basename(cfg["ckpt_url"]))
    for name, cfg in MODEL_TYPE.items()
}


def _load_metric3d_model(model_name):
    sys.path.insert(0, THIRDPARTY_ROOT)
    try:
        from mmengine import Config
        from mono.model.monodepth_model import get_configured_monodepth_model
        cfg_file = path.join(THIRDPARTY_ROOT, MODEL_TYPE[model_name]["cfg_rel_path"])
        cfg = Config.fromfile(cfg_file)
        model = get_configured_monodepth_model(cfg)
    finally:
        sys.path.remove(THIRDPARTY_ROOT)

    ckpt_url = MODEL_TYPE[model_name]["ckpt_url"]
    checkpoint = torch.hub.load_state_dict_from_url(ckpt_url, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[iw3] Metric3D '{model_name}': load_state_dict missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def batch_preprocess(x, input_size):
    # x: BCHW float32 0-1. Tensor-native reimplementation of the upstream
    # hubconf.py's numpy/cv2 preprocessing (keep-aspect-ratio resize to fit
    # `input_size`, then pad with the ImageNet mean so the padded region
    # normalizes to exactly zero -- same result, GPU-resident throughout).
    B, C, height, width = x.shape
    scale = min(input_size[0] / height, input_size[1] / width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    antialias = not (device_is_mps(x.device) or device_is_xpu(x.device))
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=antialias)
    x = x.clamp(0, 1)

    pad_h = input_size[0] - new_h
    pad_w = input_size[1] - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    mean = torch.tensor([123.675, 116.28, 103.53], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1) / 255.0
    std = torch.tensor([58.395, 57.12, 57.375], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1) / 255.0

    padded = mean.expand(B, C, input_size[0], input_size[1]).clone()
    padded[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = x
    x = (padded - mean) / std

    return x, (pad_top, pad_bottom, pad_left, pad_right)


def _desaturate_clamp_outliers(pred_depth, min_val, max_val, tail_fraction=0.01, gap_ratio_threshold=3.0):
    """Metric3D's decode head regresses depth then hard-clamps it to
    [min_val, max_val] (RAFTDepthNormalDPTDecoder5.py / HourGlassDecoder.py's own
    .clamp(), e.g. (0.1, 200) meters for the ViT-RAFT5 family, (0.3, 150) for the
    ConvNeXt family) -- a real, confirmed architectural bound. Pure-black
    letterbox-bar content (no real visual signal) makes the model jump to
    exactly this bound there -- confirmed on a real movie frame. But (also
    confirmed live, on that same real frame) it is NOT a clean, isolated jump:
    the ViT's own receptive field blends that black-only signal into nearby
    real-content patches too, producing a smooth multi-row DECAY from the hard
    clamp value down to a normal value at the boundary -- e.g. one real
    measured case went 200 -> 98 -> 37 -> 31 -> 25 -> 21 -> 14 -> 8 (back to
    normal) over 7 rows. An earlier version of this function only matched the
    exact clamp value and missed that whole decaying halo, leaving the
    normalization range still badly blown out (confirmed: min stayed at -111
    after that version "fixed" a real -200 case). This version instead asks a
    purely statistical question with no knowledge of the model's specific
    clamp values needed at all: is there a small minority of pixels sitting far
    beyond where the bulk of the frame's own values live? `min_val`/`max_val`
    are still accepted (and read from the live model in batch_infer(), never
    hardcoded) only as documentation of why this problem exists, not used in
    the check itself, since this generalizes correctly regardless of exactly
    where a given checkpoint's own clamp bound sits.

    Method: find a robust "normal" range via a modest tail fraction (default
    1%, via kthvalue rather than torch.quantile -- exact, and has no GPU
    element-count ceiling to worry about at 4K+ resolutions) instead of the
    frame's true min/max. Only when the TRUE extreme sits dramatically (default
    3x) farther beyond that normal range than the normal range's own width does
    the true extreme get clamped in to the normal range's edge -- a real,
    well-behaved frame's true min/max sit close to its own 1st/99th percentile
    by construction (this ratio would be small), so this is a safe no-op for
    ordinary content and only engages on a genuine, dramatic, isolated-tail
    situation like the one that motivated it. Winsorizes toward the normal
    range's own edge -- never deletes a pixel or invents a value, and every
    pixel already within the normal range is provably untouched (clamp is
    idempotent there)."""
    for i in range(pred_depth.shape[0]):
        flat = pred_depth[i].reshape(-1)
        n = flat.numel()
        k = max(1, int(n * tail_fraction))
        if k * 2 >= n:
            continue
        lo_normal = torch.kthvalue(flat, k).values
        hi_normal = torch.kthvalue(flat, n - k + 1).values
        true_min = flat.amin()
        true_max = flat.amax()
        normal_range = (hi_normal - lo_normal).clamp_min(1e-6)

        clamp_lo = lo_normal.item() if (lo_normal - true_min) > normal_range * gap_ratio_threshold else None
        clamp_hi = hi_normal.item() if (true_max - hi_normal) > normal_range * gap_ratio_threshold else None
        if clamp_lo is not None or clamp_hi is not None:
            pred_depth[i] = pred_depth[i].clamp(min=clamp_lo, max=clamp_hi)
    return pred_depth


@torch.inference_mode()
def batch_infer(model, im, flip_aug=True, low_vram=False, enable_amp=False,
                output_device="cpu", device=None, edge_dilation=0, **kwargs):
    device = device if device is not None else model.device
    batch = False
    if torch.is_tensor(im):
        assert im.ndim == 3 or im.ndim == 4
        if im.ndim == 3:
            im = im.unsqueeze(0)
        else:
            batch = True
        x = im.to(device)
    else:
        # PIL
        x = TF.to_tensor(im).unsqueeze(0).to(device)

    orig_h, orig_w = x.shape[2], x.shape[3]
    # Real, confirmed architectural bounds the decode head clamps its own output
    # to (see _desaturate_clamp_outliers) -- read from the model itself so this
    # never drifts out of sync with either family's config. Missing/unexpected
    # attribute path -> both stay None -> the desaturation call below becomes a
    # guaranteed no-op (unchanged prior behavior) rather than risk a crash.
    try:
        _clamp_min_val = model.depth_model.decoder.min_val
        _clamp_max_val = model.depth_model.decoder.max_val
    except AttributeError:
        _clamp_min_val = _clamp_max_val = None

    def _run(x_in):
        x_in, (pad_top, pad_bottom, pad_left, pad_right) = batch_preprocess(x_in, model.input_size)
        with autocast(device=x_in.device, enabled=enable_amp):
            pred_depth, _confidence, _output_dict = model.inference({"input": x_in})
        pred_depth = torch.nan_to_num(pred_depth.float())
        h, w = pred_depth.shape[-2], pred_depth.shape[-1]
        pred_depth = pred_depth[:, :, pad_top:h - pad_bottom, pad_left:w - pad_right]
        # Fix BEFORE the resize below, not after: resizing first would bilinear-
        # blend the exact clamp-boundary value into a wider band of neighboring
        # pixels, spreading the contamination and making it no longer exactly
        # detectable at the known clamp value.
        if _clamp_min_val is not None:
            pred_depth = _desaturate_clamp_outliers(pred_depth, _clamp_min_val, _clamp_max_val)
        pred_depth = F.interpolate(pred_depth, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return pred_depth

    if not low_vram:
        if flip_aug:
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        out = _run(x)
    else:
        x_org = x
        out = _run(x)
        if flip_aug:
            x = torch.flip(x_org, dims=[3])
            out2 = _run(x)
            out = torch.cat([out, out2], dim=0)

    # NOTE: raw model output is literal (canonical-space) distance -- smaller
    # value = nearer. Every other model in this codebase returns depth in the
    # opposite ("near=large value", disparity-like) orientation by the time
    # infer() returns it (see zoedepth_model.py's own identical unconditional
    # negation for the same reason: its raw metric_depth output has the same
    # near=small polarity as this one). Mirrored here so downstream code
    # (mapper.py's div_* metric mappers, EMA min/max normalization) sees the
    # same convention regardless of which model produced the depth map.
    #
    # The upstream hubconf.py additionally multiplies by a "de-canonical"
    # focal-length-derived scale to turn this into real-world meters -- that
    # step is deliberately NOT reproduced here. It is a positive scalar
    # multiply, and every consumer of this model's output (minmax_normalize_chw
    # -> EMAMinMaxScaler) immediately min/max-normalizes the depth map into a
    # fixed 0-1 range per frame/segment, which is invariant to any positive
    # scalar multiply of its input -- so the omitted step provably cannot change
    # the final result. It also sidesteps needing a real camera focal length,
    # which is never available for arbitrary source video anyway. Confirmed
    # visually: a real movie frame test without this step already produced a
    # clean, correctly-ordered depth map (foreground/midground/background all
    # separated properly).
    if edge_dilation_is_enabled(edge_dilation):
        out = dilate_edge(-out, edge_dilation)
    else:
        out = -out

    if flip_aug:
        if batch:
            n = out.shape[0] // 2
            z = torch.empty((n, *out.shape[1:]), device=out.device)
            for i in range(n):
                z[i] = (out[i] + torch.flip(out[i + n], dims=[2])) * 0.5
        else:
            z = (out[0:1] + torch.flip(out[1:2], dims=[3])) * 0.5
    else:
        z = out
    if not batch:
        assert z.shape[0] == 1
        z = z.squeeze(0)

    z = z.to(output_device)

    return z


class Metric3DModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)

    def load_model(self, model_type, resolution=None, device=None):
        model = _load_metric3d_model(model_type)
        model.input_size = MODEL_TYPE[model_type]["input_size"]
        model.device = device
        return model

    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0, **kwargs):
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).to(self.device)
        return batch_infer(
            self.model, x, flip_aug=tta, low_vram=low_vram,
            enable_amp=enable_amp,
            output_device=x.device,
            device=x.device,
            edge_dilation=edge_dilation)

    @classmethod
    def get_name(cls):
        return "Metric3D"

    @classmethod
    def has_checkpoint_file(cls, model_type):
        return cls.supported(model_type) and path.exists(MODEL_FILES[model_type])

    @classmethod
    def supported(cls, model_type):
        return model_type in MODEL_FILES

    @classmethod
    def get_model_path(cls, model_type):
        return MODEL_FILES[model_type]

    def is_metric(self):
        return True

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return True

    @classmethod
    def force_update(cls):
        # Not hub-based (see module docstring) -- nothing to force-reclone. A
        # corrupted checkpoint is already handled generically by
        # BaseDepthModel.load()'s own delete-and-retry logic, and re-running
        # will simply re-download it fresh (torch.hub.load_state_dict_from_url).
        pass

    def infer_raw(self, *args, **kwargs):
        return batch_infer(self.model, *args, **kwargs)


if __name__ == "__main__":
    Metric3DModel("Metric3D_ViT_Small")
