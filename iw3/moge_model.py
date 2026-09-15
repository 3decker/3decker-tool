import sys
from os import path
import torch
from torchvision.transforms import functional as TF
from huggingface_hub import try_to_load_from_cache
from nunif.device import create_device, device_is_mps, device_is_xpu  # noqa
from .dilation import dilate_edge, edge_dilation_is_enabled
from .base_depth_model import BaseDepthModel, HUB_MODEL_DIR


# ADR-156: MoGe-3 (https://github.com/microsoft/MoGe, MIT license, Microsoft
# Research). Same vendoring rationale as Metric3D (ADR-155) -- no nagadomi hub
# fork exists for this model, so the small (~260KB after trimming train/test/
# scripts/gradio-demo-only code) `moge/` package is vendored directly into
# iw3/thirdparty/moge/ instead of standing up a new external GitHub repo.
#
# UNLIKE Metric3D, MoGe-3's real, non-vendorable dependency is `flex_gemm`
# (https://github.com/JeffreyXiang/FlexGEMM) -- a package with a genuine
# compiled CUDA extension (real .cu source, needs the CUDA Toolkit's nvcc to
# build). This machine does NOT have the CUDA Toolkit installed (confirmed:
# no nvcc.exe anywhere under Program Files) -- but FlexGEMM has its own real,
# working fallback (mono/kernels/__init__.py: `try: from . import cuda /
# except ImportError: pkg_config.USE_CUDA_EXTENSION = False`) to pure-Triton
# kernels, and that fallback was confirmed live on this exact machine: `pip
# install` completed cleanly producing a pure-Python wheel (no compiled
# extension inside, confirmed: cannot `from flex_gemm.kernels import cuda`),
# and a real sparse 3D convolution (forward AND backward, via
# flex_gemm.ops.spconv.sparse_submanifold_conv3d) ran correctly through the
# Triton path in under 2.5 seconds including first-call JIT compilation.
# flex_gemm/utils3d_moge/pipeline/trimesh/moderngl/glcontext are real PyPI/git
# dependencies (not vendored -- they're genuine installable packages, not tiny
# pure-Python source trees) -- see requirements.txt.
#
# `moge`'s own package (the one being vendored) is deliberately installed
# WITHOUT its declared `gradio>=6.0`/`starlette` dependencies -- those exist
# only for MoGe's own bundled web demo app (`moge/scripts/app.py`, NOT
# vendored here), which this integration never uses; pulling in a full Gradio
# web server for zero benefit was judged not worth the bloat.
THIRDPARTY_ROOT = path.join(path.dirname(__file__), "thirdparty", "moge")

MODEL_TYPE = {
    "MoGe3_ViT_L": "Ruicheng/moge-3-vitl",
    "MoGe3_ViT_G": "Ruicheng/moge-3-vitg",
}
# Routed into this project's own model directory (matching every other depth
# model here) rather than the default ~/.cache/huggingface -- passed as
# `cache_dir` to both from_pretrained() (download) and try_to_load_from_cache()
# (existence check), so both agree on the same real location.
HF_CACHE_DIR = path.join(HUB_MODEL_DIR, "huggingface_moge")


def _load_moge_model(model_name, device):
    sys.path.insert(0, THIRDPARTY_ROOT)
    try:
        # Aliased to avoid shadowing this module's own MoGeModel(BaseDepthModel)
        # class defined below -- these are two unrelated classes that happen to
        # share a name (the vendored upstream model class vs. this file's iw3
        # integration wrapper).
        from moge.model.v3 import MoGeModel as VendoredMoGeModel
        model = VendoredMoGeModel.from_pretrained(MODEL_TYPE[model_name], cache_dir=HF_CACHE_DIR)
    finally:
        sys.path.remove(THIRDPARTY_ROOT)
    model = model.to(device).eval()
    return model


@torch.inference_mode()
def batch_infer(model, im, flip_aug=False, low_vram=False, enable_amp=True,
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

    def _run(x_in):
        # apply_mask=False: MoGe's infer() writes literal `inf` into `depth`
        # for pixels it flags low-confidence (e.g. sky) when apply_mask=True --
        # left on, that `inf` would poison the EMA min/max scaler downstream
        # for the whole frame the instant one such pixel appears (confirmed by
        # reading infer()'s own source: `depth = torch.where(mask_binary,
        # depth, torch.inf)`). Disabling it keeps a real, dense, ordinary
        # number everywhere -- more useful for stereo conversion than punching
        # literal holes in the depth map, and the same tradeoff DA3 already
        # makes for its own sky pixels (blended toward a max distance instead
        # of left undefined). resolution_level stays at its own default (9,
        # the highest-quality setting) -- see the class docstring for why
        # iw3's Depth Resolution field is disabled for this model instead of
        # threaded through here.
        out = model.infer(x_in, apply_mask=False, use_fp16=enable_amp)
        depth = torch.nan_to_num(out["depth"].float())
        return depth.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

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

    # NOTE: raw `depth` is literal distance (smaller = nearer) -- confirmed
    # live: a real movie frame test showed the nearest foreground figures at
    # the smallest raw values. Every other model in this codebase returns the
    # opposite ("near = large value") convention by the time infer() returns
    # (see zoedepth_model.py / metric3d_model.py's identical unconditional
    # negation for the same reason) -- mirrored here for the same reason.
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


class MoGeModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)

    def load_model(self, model_type, resolution=None, device=None):
        # NOTE: `resolution` (iw3's Depth Resolution field) is deliberately
        # ignored -- MoGe's own resolution knob is a discrete 0-9
        # `resolution_level` (a token-budget, not a pixel target; output size
        # always matches input size regardless), which doesn't map cleanly
        # onto iw3's pixel-based field. cbo_resolution is disabled for this
        # model in gui.py (grouped with DEPTH_PRO_MODELS/METRIC3D_MODELS),
        # same as Metric3D, rather than building a lossy/confusing mapping
        # between two different kinds of knob.
        return _load_moge_model(model_type, device)

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
        return "MoGe3"

    @classmethod
    def has_checkpoint_file(cls, model_type):
        if not cls.supported(model_type):
            return False
        result = try_to_load_from_cache(
            repo_id=MODEL_TYPE[model_type], filename="model.pt",
            cache_dir=HF_CACHE_DIR, repo_type="model")
        return isinstance(result, str)

    @classmethod
    def supported(cls, model_type):
        return model_type in MODEL_TYPE

    @classmethod
    def get_model_path(cls, model_type):
        # Best-effort: resolves to the real cached blob path once downloaded
        # (used by BaseDepthModel.load()'s corrupted-file cleanup path on
        # error). Before the first successful download there is no real file
        # to point to yet -- fall back to a location under this model's own
        # cache dir so callers get a sensible, non-crashing path either way.
        result = try_to_load_from_cache(
            repo_id=MODEL_TYPE[model_type], filename="model.pt",
            cache_dir=HF_CACHE_DIR, repo_type="model")
        if isinstance(result, str):
            return result
        return path.join(HF_CACHE_DIR, model_type, "model.pt")

    def is_metric(self):
        # MoGe-3-vitl/-vitg both report real metric-scale output (unlike
        # Metric3D, this does not need an assumed/default camera focal length
        # -- MoGe recovers focal+shift directly from its own predicted point
        # map every call, confirmed by reading infer()'s recover_focal_shift
        # call in the vendored source).
        return True

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return True

    @classmethod
    def force_update(cls):
        # Not hub-based (see module docstring) -- nothing to force-reclone.
        # A corrupted checkpoint is already handled generically by
        # BaseDepthModel.load()'s own delete-and-retry logic, and re-running
        # will simply re-download it fresh via huggingface_hub.
        pass

    def infer_raw(self, *args, **kwargs):
        return batch_infer(self.model, *args, **kwargs)


if __name__ == "__main__":
    MoGeModel("MoGe3_ViT_L")
