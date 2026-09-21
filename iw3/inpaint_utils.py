import sys
from os import path

import yaml

from nunif.models import load_model
from nunif.utils.home_dir import ensure_home_dir
from nunif.utils.ui import TorchHubDir

from .hub_dir import HUB_MODEL_DIR


def pth_url(filename):
    return "https://github.com/nagadomi/nunif/releases/download/0.0.0/" + filename


MASK_MLBW_L2_D1_URL = pth_url("iw3_mask_mlbw_l2_d1_20250903.pth")
INPAINT_CONFIG_FILE = path.join(ensure_home_dir("iw3"), "inpaint_models.yml")
INPAINT_MODEL_DEFAULT = "light_inpaint_v1"

# Single source of truth for the 3 optional Aether inpaint models, written to
# INPAINT_CONFIG_FILE by ensure_optional_inpaint_models_registered() below.
# Called from both setup.ps1 (fresh installs) and update-3decker.bat (ADR-137:
# existing installs that set up before these 3 models existed never got this
# file written otherwise, since setup.ps1 only ever runs once, and
# update-3decker.bat never touched it -- "Install Update Now" alone could
# never make these appear for anyone who had already completed setup before
# this feature shipped).
OPTIONAL_INPAINT_MODELS_YAML = """\
# Optional extra inpaint models, written by setup.ps1 / update-3decker.bat on
# first install or first update after this feature shipped. Safe to edit or
# delete -- see inpaint_models.yml.sample in windows_package/ for the full
# format. Note: real A/B testing on this project found Video_Large_Aether
# looks WORSE than the default light_inpaint_v1 -- the two Medium variants are
# untested. Treat light_inpaint_v1 as the recommended default; these three are
# optional extras to experiment with, not proven upgrades.
Video_Large_Aether:
  video: https://github.com/3decker/3decker-tool/releases/download/inpaint-models-v1/video_inpaint_v1_large-aether.pth

Video_Medium_Aether:
  video: https://github.com/3decker/3decker-tool/releases/download/inpaint-models-v1/video_inpaint_v1_medium-aether.pth

Video_Medium_Aether_v2:
  video: https://github.com/3decker/3decker-tool/releases/download/inpaint-models-v1/video_inpaint_v1_medium_aether_20260222.pth
"""


# Rowan's trained inpainting model (see nt_inpaint/README.md). Kept apart from the
# Aether block so it can be appended to an existing inpaint_models.yml.
ROWAN_INPAINT_MODEL_NAME = "Rowan_High-Depth_Inpaint_e594-Medium"
ROWAN_INPAINT_MODEL_YAML = """\

# Rowan's High-Depth inpainting model (needs the nt_inpaint extras that ship with
# 3DECKER; downloaded on first use, ~83 MB). Best at divergence 8 and above.
Rowan_High-Depth_Inpaint_e594-Medium:
  video: https://github.com/3decker/3decker-tool/releases/download/inpaint-models-v1/Rowan_High-Depth_Inpaint_e594-Medium.pth
"""


def ensure_optional_inpaint_models_registered():
    """Make sure INPAINT_CONFIG_FILE lists the optional models. Idempotent, safe
    to call on every setup/update run. A new file gets the Aether models and
    Rowan's model; an existing file is never rewritten, but Rowan's model is
    appended to it if no entry of that name is there yet (this keeps a user's own
    hand-edited customization intact)."""
    if not path.exists(INPAINT_CONFIG_FILE):
        with open(INPAINT_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(OPTIONAL_INPAINT_MODELS_YAML + ROWAN_INPAINT_MODEL_YAML)
        print(f"Wrote {INPAINT_CONFIG_FILE} (4 optional inpaint models registered, not yet downloaded).")
        return True
    with open(INPAINT_CONFIG_FILE, encoding="utf-8") as f:
        text = f.read()
    if any(line.startswith(ROWAN_INPAINT_MODEL_NAME + ":") for line in text.splitlines()):
        print(f"{INPAINT_CONFIG_FILE} already lists {ROWAN_INPAINT_MODEL_NAME} -- leaving it alone.")
        return False
    with open(INPAINT_CONFIG_FILE, "a", encoding="utf-8") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write(ROWAN_INPAINT_MODEL_YAML)
    print(f"Added {ROWAN_INPAINT_MODEL_NAME} to {INPAINT_CONFIG_FILE}.")
    return True

def _resolve_path(path_or_url):
    if not path_or_url:
        return path_or_url

    if path_or_url.lower().startswith(("http://", "https://")):
        return path_or_url

    if path.isabs(path_or_url):
        return path_or_url

    # Relative Path
    repository_root = ensure_home_dir(None)
    return path.normpath(path.join(repository_root, path_or_url))


def _load_inpaint_model_list():
    inpaint_models = {
        INPAINT_MODEL_DEFAULT: {
            "video": pth_url("iw3_light_video_inpaint_v1_20250919.pth"),
            "image": pth_url("iw3_light_inpaint_v1_20250919.pth"),
        }
    }
    if path.exists(INPAINT_CONFIG_FILE):
        with open(INPAINT_CONFIG_FILE, encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"{INPAINT_CONFIG_FILE}: Error: {e}", file=sys.stderr)
                config = None

        if isinstance(config, dict):
            for name, value in config.items():
                if not isinstance(value, dict):
                    continue
                if name in inpaint_models:
                    continue

                inpaint_models[name] = {
                    "video": _resolve_path(value.get("video")),
                    "image": _resolve_path(value.get("image")),
                }

                # Use the default model when the video or image path is not defined
                if inpaint_models[name]["video"] is None:
                    inpaint_models[name]["video"] = inpaint_models[INPAINT_MODEL_DEFAULT]["video"]
                if inpaint_models[name]["image"] is None:
                    inpaint_models[name]["image"] = inpaint_models[INPAINT_MODEL_DEFAULT]["image"]

    return inpaint_models


INPAINT_MODELS = _load_inpaint_model_list()


def _load_inpaint_checkpoint(name, which, device_id):
    """load_model() for an inpaint checkpoint, but a failed DOWNLOAD becomes a plain-language message. The three optional
    "Aether" models are registered with download addresses on the project's GitHub releases; if that file was never
    uploaded (or the PC is offline) the raw "HTTP Error 404: Not Found" gave no hint what to do."""
    import urllib.error
    location = INPAINT_MODELS[name][which]
    try:
        return load_model(location, device_ids=[device_id], weights_only=True)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"The inpainting model '{name}' could not be downloaded (the server answered: {e.code} {e.reason}). "
            f"Its file is not available at: {location}\n\n"
            f"What to do: choose the '{INPAINT_MODEL_DEFAULT}' inpainting model instead (it is the recommended "
            f"one), or copy this model's .pth file into iw3/pretrained_models/hub/checkpoints/ yourself.") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"The inpainting model '{name}' could not be downloaded: {e.reason}. Check the internet connection "
            f"and try again, or choose the '{INPAINT_MODEL_DEFAULT}' inpainting model (it is downloaded once "
            f"during installation).") from e


def load_image_inpaint_model(name, device_id):
    with TorchHubDir(HUB_MODEL_DIR):
        if name is None:
            name = INPAINT_MODEL_DEFAULT
        if name not in INPAINT_MODELS:
            raise ValueError(f"inpaint model `{name}` is not defined")
        model, _ = _load_inpaint_checkpoint(name, "image", device_id)
        return model.eval()


def load_video_inpaint_model(name, device_id):
    with TorchHubDir(HUB_MODEL_DIR):
        if name is None:
            name = INPAINT_MODEL_DEFAULT
        if name not in INPAINT_MODELS:
            raise ValueError(f"inpaint model `{name}` is not defined")
        model, _ = _load_inpaint_checkpoint(name, "video", device_id)
        return model.eval()


def load_mask_mlbw(device_id):
    with TorchHubDir(HUB_MODEL_DIR):
        model, _ = load_model(MASK_MLBW_L2_D1_URL, device_ids=[device_id], weights_only=True)
        model.delta_output = True
        return model.eval()


class CompileContext:
    def __init__(self, base_model):
        self.base_model = base_model

    def __enter__(self):
        self.base_model.compile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.base_model.clear_compiled_model()
        return False
