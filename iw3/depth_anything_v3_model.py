import os
import sys
from os import path
import torch
import safetensors.torch
from torchvision.transforms import functional as TF
from nunif.device import create_device, autocast, device_is_mps, device_is_xpu # noqa
from .dilation import dilate_edge, edge_dilation_is_enabled
from .base_depth_model import BaseDepthModel, HUB_MODEL_DIR
from .depth_anything_model import batch_preprocess
from .models import DepthAA
from .depth_scaler import EMAMinMaxScaler


# ADR-135: da3-small/-base/-large-1.1/metric-large added alongside the original
# da3mono-large -- real Hugging Face repo IDs confirmed against VisionDepth3D's
# own official model list, not guessed. is_metric() still hardcoded False for all
# five below -- DA3METRIC-LARGE's genuinely metric (absolute-scale) output is NOT
# specially handled here; it loads and runs through the exact same relative-depth
# postprocessing as everything else, so its absolute-scale semantics may not be
# correctly exploited yet. Treat that one specifically as experimental until
# tested against a known-metric scene, unlike the other three (same relative-
# depth family as the already-proven da3mono-large, no such caveat).
NAME_MAP = {
    "Any_V3_Mono": "da3mono-large",
    "Any_V3_Mono_01": "da3mono-large",
    "Any_V3_Small": "da3-small",
    "Any_V3_Base": "da3-base",
    "Any_V3_Large_1_1": "da3-large-1.1",
    "Any_V3_Metric_Large": "da3metric-large",
    # ADR-159: CC-BY-NC-4.0 -- gated behind has_checkpoint_file in gui.py's
    # get_depth_models(), same treatment as Any_V2_B/Any_V2_L (ADR-142).
    "Any_V3_Giant": "da3-giant-1.1",
    "Any_V3_Nested_Giant_Large": "da3nested-giant-large-1.1",
}
MODEL_FILES = {
    "Any_V3_Mono": path.join(HUB_MODEL_DIR, "checkpoints", "da3mono-large.safetensors"),
    "Any_V3_Mono_01": path.join(HUB_MODEL_DIR, "checkpoints", "da3mono-large.safetensors"),
    "Any_V3_Small": path.join(HUB_MODEL_DIR, "checkpoints", "da3-small.safetensors"),
    "Any_V3_Base": path.join(HUB_MODEL_DIR, "checkpoints", "da3-base.safetensors"),
    "Any_V3_Large_1_1": path.join(HUB_MODEL_DIR, "checkpoints", "da3-large-1.1.safetensors"),
    "Any_V3_Metric_Large": path.join(HUB_MODEL_DIR, "checkpoints", "da3metric-large.safetensors"),
    "Any_V3_Giant": path.join(HUB_MODEL_DIR, "checkpoints", "da3-giant-1.1.safetensors"),
    "Any_V3_Nested_Giant_Large": path.join(HUB_MODEL_DIR, "checkpoints", "da3nested-giant-large-1.1.safetensors"),
}
# Whether Depth Anti-aliasing (a separate, small refinement net run on top of the
# raw depth output) is architecturally valid for a given DA3 variant has only ever
# been confirmed for da3mono-large -- deliberately NOT extended to the 4 new
# variants below without the same real verification, rather than assumed safe.
AA_SUPPORTED_MODELS = {
    "Any_V3_Mono",
    "Any_V3_Mono_01",
}


# ADR-139: real, live-confirmed bug -- Any_V3_Large_1_1 crashed a real
# conversion job with KeyError: 'da3-large-1.1'. MODEL_REGISTRY's real keys
# (confirmed by importing it directly) are da3-small/-base/-large/-giant/
# mono-large/metric-large/nested-giant-large -- there is NO "da3-large-1.1"
# architecture entry. "1.1" is a Hugging Face CHECKPOINT release version for
# the SAME "da3-large" architecture/config, not a separate architecture --
# _da3_hf_url(model_name) is still correct using the full "da3-large-1.1"
# name (that's the real checkpoint's HF repo ID), but MODEL_REGISTRY lookup
# must use the underlying config name instead.
DA3_CONFIG_NAME_OVERRIDES = {
    "da3-large-1.1": "da3-large",
    # ADR-159: same "-1.1 is a checkpoint release, not a separate config" story
    # as da3-large-1.1 above -- confirmed by reading the real config filenames
    # in nagadomi's own cached hub repo (configs/da3-giant.yaml, configs/
    # da3nested-giant-large.yaml -- no "-1.1" in either filename).
    "da3-giant-1.1": "da3-giant",
    "da3nested-giant-large-1.1": "da3nested-giant-large",
}


def _da3_config_name(model_name):
    return DA3_CONFIG_NAME_OVERRIDES.get(model_name, model_name)


def _da3_hf_url(model_name):
    # Confirmed to match nagadomi's own hardcoded da3mono-large URL exactly
    # (hubconf.py: "https://huggingface.co/depth-anything/DA3MONO-LARGE/resolve/
    # main/model.safetensors?download=true"), and separately confirmed to match
    # the real Hugging Face repo IDs VisionDepth3D's own official model list uses
    # for da3-small/-base/metric-large/-large-1.1 -- one generic pattern serves
    # every variant, not four separate guesses.
    return f"https://huggingface.co/depth-anything/{model_name.upper()}/resolve/main/model.safetensors?download=true"


def _patch_da3_state_dict_key(state_dict):
    # Exact mirror of nagadomi's own hubconf.py _patch_state_dict_key() -- only
    # keys prefixed "model." are kept (matches the real, already-proven behavior
    # for da3mono-large; deliberately not "improved" with an else branch that
    # would change behavior nagadomi's own code doesn't have).
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("model."):
            new_state_dict[key[len("model."):]] = state_dict[key]
    return new_state_dict


def _ensure_da3_repo_cached():
    """Returns nagadomi's Depth-Anything-3_iw3 repo's own root directory
    (containing hubconf.py and src/depth_anything_3/), guaranteeing it is
    present in the torch hub cache first if it is not already -- using ONLY
    torch.hub's public API (torch.hub.load itself), never a private/internal
    torch.hub function, so this stays correct across torch versions. The one
    real, already-proven-working entry point (model_name="da3mono-large") is
    used purely to trigger the clone as a side effect when the cache is cold;
    the model it constructs is thrown away immediately -- cheap relative to
    the alternative (reimplementing git-clone-and-cache logic ourselves)."""
    hub_dir = torch.hub.get_dir()
    repo_dir = path.join(hub_dir, "nagadomi_Depth-Anything-3_iw3_main")
    if os.getenv("IW3_DEBUG"):
        assert path.exists("../Depth-Anything-3_iw3/hubconf.py")
        return path.abspath("../Depth-Anything-3_iw3")
    if not path.isdir(repo_dir):
        torch.hub.load("nagadomi/Depth-Anything-3_iw3:main", "load_model",
                       model_name="da3mono-large", verbose=False, trust_repo=True)
    return repo_dir


def _load_da3_model(model_name):
    """Constructs and loads any Depth-Anything-3 variant nagadomi's own hub repo
    has a real config for (MODEL_REGISTRY's own keys -- da3-small/-base/-large/
    -giant/mono-large/metric-large/large-1.1/nested-giant-large), bypassing a
    real, confirmed bug in that repo's own hubconf.py: `_load_state_dict()`
    there only ever assigns a download URL when model_name == "da3mono-large" --
    every other model_name hits `UnboundLocalError: local variable 'file_name'
    referenced before assignment` on the very next line, confirmed by reading
    that function directly, not assumed. This reimplements ONLY the weight-
    fetching step, using a URL pattern confirmed correct for every variant (see
    _da3_hf_url) -- model CONSTRUCTION (config/registry/create_object) still
    goes through nagadomi's own real code unmodified, since that part already
    works correctly for every variant (the config YAMLs for all of them already
    ship in that same repo)."""
    repo_dir = _ensure_da3_repo_cached()
    pkg_root = path.join(repo_dir, "src")
    sys.path.insert(0, pkg_root)
    try:
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        config = load_config(MODEL_REGISTRY[_da3_config_name(model_name)])
        model = create_object(config)
    finally:
        sys.path.remove(pkg_root)

    checkpoint_path = path.join(HUB_MODEL_DIR, "checkpoints", f"{model_name}.safetensors")
    if not path.exists(checkpoint_path):
        os.makedirs(path.dirname(checkpoint_path), exist_ok=True)
        torch.hub.download_url_to_file(_da3_hf_url(model_name), checkpoint_path)
    state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
    # strict=False (not nagadomi's own strict=True, confirmed correct for
    # da3mono-large only): real, live-confirmed finding on da3-small -- its
    # published checkpoint is genuinely missing a handful of
    # "head.scratch.output_conv2_aux.*" keys, an auxiliary/deep-supervision
    # output head (naming and the fact only that one sub-module is affected both
    # point to a training-only head, not part of the real inference forward pass
    # -- confirmed by the model still producing a real, correct-looking depth map
    # in the same live test that surfaced this). Missing/unexpected keys are
    # still printed, not silently swallowed, so a genuinely wrong checkpoint for
    # some future variant is still visible instead of hidden by this.
    missing, unexpected = model.load_state_dict(_patch_da3_state_dict_key(state_dict), strict=False)
    if missing or unexpected:
        print(f"[iw3] DA3 '{model_name}': load_state_dict missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def _forward(model, x, enable_amp, sky_thresh=0.3, raw_output=False):
    amp_dtype = torch.bfloat16 if (x.device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    with autocast(device=x.device, enabled=enable_amp, dtype=amp_dtype):
        x = x.unsqueeze(1)  # (B, S, C, H, W)
        out = model(x)

    depths = out["depth"]
    if "sky" in out:
        sky_masks = out["sky"] > sky_thresh
        sky_weights = (out["sky"].clamp(sky_thresh, 1.0) - sky_thresh) / (1.0 - sky_thresh)
    else:
        # ADR-135: real, live-confirmed finding -- da3-small/-base/-large-1.1/
        # metric-large have no "sky" output at all. Their real keys are depth/
        # depth_conf/extrinsics/intrinsics/aux (confirmed by directly inspecting
        # a real forward pass) -- this whole DA3 family (not "DA3MONO") is built
        # for multi-view 3D reconstruction (hence the camera extrinsics/
        # intrinsics), and only the mono-specialized head predicts a sky mask.
        # Treat every pixel as "not sky" (zero weight) so the exact same
        # downstream math below degrades cleanly to plain, unmasked depth for
        # these variants instead of crashing on a key that was never there.
        sky_masks = torch.zeros_like(depths, dtype=torch.bool)
        sky_weights = torch.zeros_like(depths)
    disparity_maps = []
    for depth, sky_mask, sky_weight in zip(depths, sky_masks, sky_weights):
        # TODO: improve this
        if not raw_output:
            non_sky_pixels = sky_mask.numel() - sky_mask.sum()
            if non_sky_pixels < 10:
                # all sky
                depth = torch.zeros_like(depth)
            else:
                # TODO: This value should ideally be adjustable via the foreground scale option,
                #       but currently it is not possible.
                shift = 0.2
                depth = 1.0 / (depth + shift)
                depth = depth * (1 - sky_weight)
        else:
            max_rel_dist = torch.quantile(depth[~sky_mask], 0.99)
            depth = (depth * (1 - sky_weight) + sky_weight * max_rel_dist).clamp(max=max_rel_dist)

        disparity_maps.append(depth)

    out = torch.stack(disparity_maps)

    return out


@torch.inference_mode()
def batch_infer(model, im, flip_aug=True, low_vram=False, enable_amp=False,
                output_device="cpu", device=None, edge_dilation=2, depth_aa=None,
                limit_resolution=False,
                raw_output=False,
                **kwargs):
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

    x = batch_preprocess(x, model.prep_lower_bound, limit_resolution=limit_resolution)

    if not low_vram:
        if flip_aug:
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        out = _forward(model, x, enable_amp, raw_output=raw_output)
    else:
        x_org = x
        out = _forward(model, x, enable_amp, raw_output=raw_output)
        if flip_aug:
            x = torch.flip(x_org, dims=[3])
            out2 = _forward(model, x, enable_amp, raw_output=raw_output)
            out = torch.cat([out, out2], dim=0)
    if depth_aa is not None:
        out = depth_aa.infer(out)

    if edge_dilation_is_enabled(edge_dilation):
        if not raw_output:
            out = dilate_edge(out, edge_dilation)
        else:
            out = -dilate_edge(-out, edge_dilation)

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


class DepthAnythingV3MonoModel(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)

    def create_depth_scaler(self):
        if self.model_type == "Any_V3_Mono":
            # Max=1 scaling
            return EMAMinMaxScaler(decay=0, buffer_size=1, mode="max")
        else:
            # Max=1 and Min=0 scaling -- same as "Any_V3_Mono_01", this project's
            # own standing default variant. Applied to the 4 newer variants
            # (ADR-135) too, absent any model-specific guidance otherwise: they
            # share the same underlying architecture/head design as mono-large,
            # just different network sizes (or, for Metric_Large, a different
            # training objective this scaler does not specially account for --
            # see that model's own caveat where it's registered above), so a
            # plain min/max normalize into a usable 0..1 range is the same
            # reasonable default this whole family already uses rather than
            # returning None (which would leave self.scaler unset and break the
            # very first depth pass).
            return EMAMinMaxScaler(decay=0, buffer_size=1, mode="minmax")

    def load_model(self, model_type, resolution=None, device=None, raw_output=False):
        # load aa model
        if model_type in AA_SUPPORTED_MODELS:
            self.depth_aa = DepthAA().load().eval().to(device)
        else:
            self.depth_aa = None

        model_name = NAME_MAP[model_type]
        model = _load_da3_model(model_name)

        model.prep_lower_bound = resolution or 392
        if model.prep_lower_bound % 14 != 0:
            # From GUI, 512 -> 518
            model.prep_lower_bound += (14 - model.prep_lower_bound % 14)
        model.device = device
        self.raw_output = raw_output

        return model

    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0, depth_aa=False, **kwargs):
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).to(self.device)
        return batch_infer(
            self.model, x, flip_aug=tta, low_vram=low_vram,
            enable_amp=enable_amp,
            output_device=x.device,
            device=x.device,
            edge_dilation=edge_dilation,
            depth_aa=self.depth_aa if depth_aa else None,
            raw_output=self.raw_output,
            limit_resolution=self.limit_resolution,
        )

    @classmethod
    def get_name(cls):
        return "DepthAnythingV3Mono"

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
        return False

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return True

    @classmethod
    def force_update(cls):
        BaseDepthModel.force_update_hub("nagadomi/Depth-Anything-3_iw3:main", "load_model")

    def infer_raw(self, *args, **kwargs):
        return batch_infer(self.model, *args, **kwargs)


def _bench(resolution=504, do_compile=False):
    import time
    import gc

    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    B = 4
    N = 20
    model = DepthAnythingV3MonoModel("Any_V3_Mono")
    model.load(gpu=0, resolution=resolution)

    print(f"*** bench: resolution={resolution}, batch={B}, compile={do_compile}")

    if do_compile:
        model.compile()
    x = torch.randn((B, 3, 1080, 1920)).cuda()
    model.infer(x)
    torch.cuda.synchronize()

    t = time.time()
    for _ in range(N):
        model.infer(x)
    torch.cuda.synchronize()
    print(round(1.0 / ((time.time() - t) / (B * N)), 4), "FPS")

    max_vram_mb = int(torch.cuda.max_memory_allocated("cuda") / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    _bench(do_compile=False)
    _bench(do_compile=True)
