"""Support for externally-pretrained super-resolution models (RealESRGAN, BSRGAN --
see docs/ai/AI_DECISIONS.md for where the weights came from) that don't fit this
project's own noise-level / scale2x-vs-scale4x model-slot conventions used by the
Waifu2x class in utils.py.

Deliberately kept as a SEPARATE, parallel path rather than folded into the Waifu2x
class itself: that class's `render`/`convert`/`load_model`/`_model_offset` methods
all hardcode the same fixed 5-way method dispatch ("scale"/"scale4x"/"noise"/
"noise_scale"/"noise_scale4x") in several places, and extending that cleanly would
mean touching all of them. Instead, ExternalSRModel below duck-types just the two
methods process_images()/process_video() (in ui_utils.py) actually call on their
`ctx` object -- .convert() and .compile() -- so this addition can never risk
changing Waifu2x's own already-working behavior for its own model families."""
import os
import torch
import torch.nn.functional as F
from os import path
from nunif.models import create_model
from nunif.utils.render import tiled_render
from nunif.transforms.tta import tta_merge, tta_split
from nunif.utils.ui import HiddenPrints
from . import models  # noqa: registers waifu2x.rrdbnet / waifu2x.srvgg_net_compact
from .models._rrdbnet import remap_bsrgan_state_dict
from .model_dir import MODEL_DIR


EXTERNAL_SR_MODEL_DIR = path.join(MODEL_DIR, "external_sr")

# method name -> (nunif model name, constructor kwargs, weight filename, optional
# state-dict key remap function). Each entry verified to load its real weight file
# with strict=True (exact key match) -- see the test that accompanied this addition.
EXTERNAL_SR_MODELS = {
    "realesrgan_x4": dict(
        arch="waifu2x.rrdbnet", kwargs=dict(scale=4, num_block=23),
        weight_file="RealESRGAN_x4plus.pth", remap=None,
        description="Real-ESRGAN x4plus -- general-purpose photo/video upscaling, 4x"),
    "realesrgan_x2": dict(
        arch="waifu2x.rrdbnet", kwargs=dict(scale=2, num_block=23),
        weight_file="RealESRGAN_x2plus.pth", remap=None,
        description="Real-ESRGAN x2plus -- general-purpose photo/video upscaling, 2x"),
    "realesrgan_general_x4": dict(
        arch="waifu2x.srvgg_net_compact", kwargs=dict(num_conv=32, scale=4),
        weight_file="realesr-general-x4v3.pth", remap=None,
        description="Real-ESRGAN general x4v3 -- lighter/faster, 4x"),
    "bsrgan_x4": dict(
        arch="waifu2x.rrdbnet", kwargs=dict(scale=4, num_block=23),
        weight_file="BSRGAN.pth", remap=remap_bsrgan_state_dict,
        description="BSRGAN -- blind super-resolution, 4x"),
    "bsrgan_x2": dict(
        arch="waifu2x.rrdbnet", kwargs=dict(scale=2, num_block=23, pixel_unshuffle_input=False),
        weight_file="BSRGANx2.pth", remap=remap_bsrgan_state_dict,
        description="BSRGAN -- blind super-resolution, 2x"),
}


ONNX_SR_METHOD_PREFIX = "onnx:"
ONNX_SR_SUBDIR = "onnx"


def is_external_sr_method(method):
    return method in EXTERNAL_SR_MODELS or method.startswith(ONNX_SR_METHOD_PREFIX)


def available_onnx_sr_methods(model_dir=None):
    """Auto-discovers any *.onnx file dropped in
    <model_dir>/onnx/ (default waifu2x/pretrained_models/external_sr/onnx/) --
    doesn't validate them (that only happens, cheaply, at actual load time via one
    real probe inference -- see ONNXSRModel), just lists what's there by filename
    so the caller can offer them as choices without needing onnxruntime installed
    just to list files."""
    model_dir = model_dir or EXTERNAL_SR_MODEL_DIR
    onnx_dir = path.join(model_dir, ONNX_SR_SUBDIR)
    if not path.isdir(onnx_dir):
        return []
    return [f"{ONNX_SR_METHOD_PREFIX}{path.splitext(f)[0]}"
            for f in sorted(os.listdir(onnx_dir)) if f.lower().endswith(".onnx")]


def available_external_sr_methods(model_dir=None):
    """Only lists methods whose weight file is actually present -- these are large
    (5-64MB+) external downloads, not bundled with the app, so absence is the normal
    case unless the user has explicitly added them (see docs/ai/AI_DECISIONS.md)."""
    model_dir = model_dir or EXTERNAL_SR_MODEL_DIR
    fixed = [name for name, spec in EXTERNAL_SR_MODELS.items()
             if path.exists(path.join(model_dir, spec["weight_file"]))]
    return fixed + available_onnx_sr_methods(model_dir)


def _load_state_dict(weight_path):
    with HiddenPrints():
        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    elif isinstance(sd, dict) and "params" in sd:
        sd = sd["params"]
    return sd


class ExternalSRModel():
    """Thin adapter around one externally-pretrained model, matching just the
    interface process_images()/process_video() need from Waifu2x.convert()."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.is_half = False

    def to(self, device):
        self.device = device
        self.model = self.model.to(device).eval()
        return self

    def half(self):
        self.is_half = True
        self.model = self.model.half()
        return self

    def float(self):
        self.is_half = False
        self.model = self.model.float()
        return self

    def compile(self):
        # Not supported for these architectures yet -- a safe no-op (matches how
        # Waifu2x's own compile() already skips models where can_compile() is False,
        # this just always takes that path for external models).
        pass

    def render(self, x, tile_size=None, batch_size=None, enable_amp=False):
        return tiled_render(x, self.model, tile_size=tile_size, batch_size=batch_size,
                             enable_amp=enable_amp)

    def convert(self, x, alpha, method, noise_level,
                tile_size=None, batch_size=None,
                tta=False, enable_amp=False, output_device="cpu"):
        # `method`/`noise_level` accepted only for interface compatibility with
        # Waifu2x.convert() (process_images/process_video call it positionally) --
        # this class only ever has the one loaded model, selected at construction.
        assert not torch.is_grad_enabled()
        assert x.shape[0] == 3
        x = x.to(self.device)
        if tta:
            rgb = tta_merge([
                self.render(xx, tile_size, batch_size, enable_amp)
                for xx in tta_split(x)])
        else:
            rgb = self.render(x, tile_size, batch_size, enable_amp)
        rgb = rgb.to(output_device)
        if alpha is not None:
            # Simple nearest/bilinear alpha upscale rather than running alpha
            # through the SR model too -- matches Waifu2x's own fallback behavior
            # when no dedicated scale model is available for the alpha channel.
            alpha = alpha.to(self.device)
            blank_alpha = torch.equal(alpha, torch.ones(alpha.shape, device=alpha.device, dtype=alpha.dtype))
            mode = "nearest" if blank_alpha else "bilinear"
            alpha = F.interpolate(alpha.unsqueeze(0), scale_factor=self.model.i2i_scale,
                                  mode=mode).squeeze(0)
            alpha = alpha.to(output_device)
        return rgb, alpha


def load_onnx_sr_model(method, device, model_dir=None):
    assert method.startswith(ONNX_SR_METHOD_PREFIX)
    model_dir = model_dir or EXTERNAL_SR_MODEL_DIR
    model_name = method[len(ONNX_SR_METHOD_PREFIX):]
    onnx_path = path.join(model_dir, ONNX_SR_SUBDIR, f"{model_name}.onnx")
    if not path.exists(onnx_path):
        raise FileNotFoundError(f"{onnx_path} not found.")
    try:
        import onnxruntime  # noqa: import check only -- real error surfaces from create_model below
    except ImportError as e:
        raise ImportError(
            "ONNX SR model support requires the optional 'onnxruntime' (or "
            "'onnxruntime-gpu') package, which is not installed. See "
            "requirements-torch-cu126.txt / docs/ai/AI_DECISIONS.md.") from e
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    model = create_model("waifu2x.onnx_sr", onnx_path=onnx_path, providers=providers)
    model = model.to(device).eval()
    return ExternalSRModel(model, device)


def load_external_sr_model(method, device, gpus=None, model_dir=None):
    if method.startswith(ONNX_SR_METHOD_PREFIX):
        return load_onnx_sr_model(method, device, model_dir=model_dir)
    if method not in EXTERNAL_SR_MODELS:
        raise ValueError(f"Unknown external SR method: {method}")
    spec = EXTERNAL_SR_MODELS[method]
    model_dir = model_dir or EXTERNAL_SR_MODEL_DIR
    weight_path = path.join(model_dir, spec["weight_file"])
    if not path.exists(weight_path):
        raise FileNotFoundError(
            f"{spec['weight_file']} not found in {model_dir} -- this is a large external "
            f"model file not bundled with the app; see docs/ai/AI_DECISIONS.md for where "
            f"to get it.")
    model = create_model(spec["arch"], **spec["kwargs"])
    sd = _load_state_dict(weight_path)
    if spec["remap"] is not None:
        sd = spec["remap"](sd)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    return ExternalSRModel(model, device)
