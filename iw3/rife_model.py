"""Optional RIFE (Practical-RIFE, https://github.com/hzwer/Practical-RIFE) frame
interpolation support -- see docs/ai/AI_DECISIONS.md ADR-029 for the license
verification (repo code AND pretrained weight downloads are both MIT, confirmed
directly against the upstream LICENSE file and README on 2026-09-07) and the
overall design rationale, and ADR-029's amendment note for the model/ package
bug fixed here (2026-09-08).

Weights are NOT bundled with this app -- fetched on demand into RIFE_MODEL_DIR
the first time a tier is actually used, mirroring waifu2x's external_sr.py
pattern (EXTERNAL_SR_MODEL_DIR / available_external_sr_methods()): an optional,
large, per-tier third-party download checked by presence rather than force-
installed for every user.

Each official Practical-RIFE release archive bundles BOTH the pretrained weight
file (flownet.pkl) AND the exact matching Python model definition it was trained
against (RIFE_HDv3.py + friends -- these differ across versions and are NOT part
of the git repository itself, only shipped inside each version's release
archive). Rather than vendoring one version's architecture code into this repo
(which would silently go stale/wrong the moment a different tier's internal
architecture differs), this module dynamically imports the official,
version-matched RIFE_HDv3.Model class from the fetched directory at inference
time -- the same "load model code from a fetched location instead of
reimplementing it in this repo" approach already used for the ZoeDepth/
Depth-Anything hub models (see iw3/base_depth_model.py's force_update_hub ->
torch.hub.load).

RIFE_HDv3.py/IFNet_HDv3.py also import a SECOND, separate package: `model`
(`from model.warplayer import warp`, `from model.loss import *`). Verified
directly against the live upstream repo on 2026-09-08: unlike train_log/'s
per-version *.py files, `model/` (warplayer.py, loss.py, pytorch_msssim/) is
checked into the Practical-RIFE git repository itself and is NOT included in
the per-tier Google Drive weight archive -- the official README's own setup
instructions ("Download a model from the model list and put *.py and
flownet.pkl on train_log/") only ever populate train_log/, assuming the user
already has a full clone of the repo (with its pre-existing model/ directory)
alongside it. Since this project fetches train_log/ standalone with no
accompanying repo clone, `model/` is fetched separately here (from the repo's
zip archive) and placed as a sibling of train_log/ so `from model.warplayer
import warp` resolves as a top-level import -- see
_RifeModelPackageDownloader/_ensure_rife_model_package below."""
import importlib
import os
import shutil
import sys
from os import path

from nunif.utils.downloader import ArchiveDownloader
from nunif.utils.home_dir import ensure_home_dir
from nunif.logger import logger


RIFE_MODEL_DIR = path.join(ensure_home_dir("iw3"), "pretrained_models", "rife")

# Google Drive file IDs verified directly against the official Practical-RIFE
# README (https://github.com/hzwer/Practical-RIFE/blob/main/README.md) on
# 2026-09-07 -- see ADR-029. "uc?export=download" is Google Drive's standard
# direct-download URL form. Large files can still return an HTML virus-scan
# interstitial instead of the real archive; download_rife_model() below detects
# that (the interstitial isn't a valid zip / contains no .pkl) and raises a clear,
# actionable error rather than silently saving garbage -- see _RifeModelDownloader.
RIFE_TIERS = {
    "rife_425": dict(
        label="RIFE 4.25 (recommended, higher quality)",
        drive_id="1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg",
        dir_name="4.25",
    ),
    "rife_425_lite": dict(
        label="RIFE 4.25 Lite (faster, lower compute cost)",
        drive_id="1zlKblGuKNatulJNFf5jdB-emp9AqGK05",
        dir_name="4.25_lite",
    ),
}
DEFAULT_RIFE_MODEL = "rife_425"

# The shared `model/` package (warplayer.py, loss.py, pytorch_msssim/) that
# RIFE_HDv3.py/IFNet_HDv3.py import -- see module docstring. Unlike the
# per-tier weights, this is identical across tiers/versions and lives in the
# main Practical-RIFE git repository itself, not in a release/Drive archive.
RIFE_REPO_ARCHIVE_URL = "https://github.com/hzwer/Practical-RIFE/archive/refs/heads/main.zip"


def _drive_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def get_rife_dir(tier):
    if tier not in RIFE_TIERS:
        raise ValueError(f"Unknown RIFE tier: {tier}")
    return path.join(RIFE_MODEL_DIR, RIFE_TIERS[tier]["dir_name"])


def _find_train_log_dir(root):
    """The official release archive's internal layout has varied across versions
    (sometimes a top-level train_log/, sometimes the files directly at the
    archive root) -- search for whichever directory actually holds the .pkl
    weight file instead of assuming one fixed path."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(f.lower().endswith(".pkl") for f in filenames):
            return dirpath
    return None


def rife_model_available(tier):
    train_log_dir = path.join(get_rife_dir(tier), "train_log")
    return path.isdir(train_log_dir) and any(
        f.lower().endswith(".pkl") for f in os.listdir(train_log_dir))


def _model_package_available(tier):
    return path.isfile(path.join(get_rife_dir(tier), "model", "warplayer.py"))


class _RifeModelDownloader(ArchiveDownloader):
    def __init__(self, tier, **kwargs):
        spec = RIFE_TIERS[tier]
        super().__init__(_drive_url(spec["drive_id"]), name=f"RIFE {spec['label']}", format="zip", **kwargs)
        self.tier = tier

    def handle(self, src):
        found = _find_train_log_dir(src)
        if found is None:
            raise RuntimeError("downloaded archive did not contain a .pkl weight file")
        dst = path.join(get_rife_dir(self.tier), "train_log")
        os.makedirs(path.dirname(dst), exist_ok=True)
        if path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(found, dst)


def _find_model_package_dir(root):
    """Search for the checked-into-the-repo `model/` directory (identified by
    containing warplayer.py) inside an extracted Practical-RIFE repo archive --
    same search-by-content approach as _find_train_log_dir, robust to the
    archive's top-level folder name (e.g. 'Practical-RIFE-main/model')."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if path.basename(dirpath) == "model" and "warplayer.py" in filenames:
            return dirpath
    return None


class _RifeModelPackageDownloader(ArchiveDownloader):
    """Fetches the shared `model/` package from the main Practical-RIFE repo
    (see module docstring) and places it as a sibling of a tier's train_log/ so
    RIFE_HDv3.py's `from model.warplayer import warp` resolves as a top-level
    import."""
    def __init__(self, tier, **kwargs):
        super().__init__(RIFE_REPO_ARCHIVE_URL, name="RIFE model package", format="zip", **kwargs)
        self.tier = tier

    def handle(self, src):
        found = _find_model_package_dir(src)
        if found is None:
            raise RuntimeError("downloaded Practical-RIFE repo archive did not contain a model/warplayer.py file")
        dst = path.join(get_rife_dir(self.tier), "model")
        os.makedirs(path.dirname(dst), exist_ok=True)
        if path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(found, dst)


def _ensure_rife_model_package(tier, show_progress=True):
    """Ensures tier's model/ package (see module docstring) is present,
    independent of whether train_log/ was already downloaded by an older
    version of this app that predates this fix."""
    if _model_package_available(tier):
        return
    logger.debug(f"RIFE: downloading model package for {tier}")
    downloader = _RifeModelPackageDownloader(tier)
    try:
        downloader.run(show_progress=show_progress)
    except Exception as e:
        raise RuntimeError(
            f"RIFE {tier}: automatic download of the 'model' package (from the official "
            f"Practical-RIFE GitHub repo) failed ({type(e).__name__}: {e}). Download "
            f"{RIFE_REPO_ARCHIVE_URL} manually and place its 'model' folder (containing "
            f"warplayer.py) into: {path.join(get_rife_dir(tier), 'model')}"
        ) from e
    if not _model_package_available(tier):
        raise RuntimeError(f"RIFE {tier}: model package download completed but warplayer.py was not found afterward")


def download_rife_model(tier, show_progress=True):
    """Fetches tier's weights on demand (see module docstring). Never leaves a
    partial/corrupt train_log/ behind on failure -- _RifeModelDownloader.handle()
    only replaces the destination after a valid .pkl was actually found in the
    downloaded archive."""
    if tier not in RIFE_TIERS:
        raise ValueError(f"Unknown RIFE tier: {tier}")
    if rife_model_available(tier):
        return get_rife_dir(tier)
    logger.debug(f"RIFE: downloading {tier}")
    downloader = _RifeModelDownloader(tier)
    try:
        downloader.run(show_progress=show_progress)
    except Exception as e:
        raise RuntimeError(
            f"RIFE {tier}: automatic download failed ({type(e).__name__}: {e}). Google Drive "
            f"sometimes serves a virus-scan warning page instead of the real file for larger "
            f"downloads instead of a direct file. Download it manually from the official "
            f"Practical-RIFE README (https://github.com/hzwer/Practical-RIFE#model-list) and "
            f"place the extracted 'train_log' folder (containing flownet.pkl) into: "
            f"{path.join(get_rife_dir(tier), 'train_log')}"
        ) from e
    if not rife_model_available(tier):
        raise RuntimeError(f"RIFE {tier}: download completed but no weight file was found afterward")
    return get_rife_dir(tier)


def ensure_rife_model(tier, show_progress=True):
    if not rife_model_available(tier):
        download_rife_model(tier, show_progress=show_progress)
    _ensure_rife_model_package(tier, show_progress=show_progress)
    return path.join(get_rife_dir(tier), "train_log")


def load_rife_model(tier, device):
    """Dynamically imports the official RIFE_HDv3.Model class from the fetched,
    version-matched train_log/ directory (see module docstring) and loads it onto
    `device`. Returns the loaded, eval()-mode model instance."""
    import torch

    train_log_dir = ensure_rife_model(tier)
    train_log_parent = path.dirname(train_log_dir)
    if train_log_parent not in sys.path:
        sys.path.insert(0, train_log_parent)
    # Force a fresh import in case a different tier's train_log was previously
    # imported under the same "train_log" module name in this same process --
    # different tiers are NOT interchangeable (different architectures/weights).
    for mod_name in list(sys.modules):
        if mod_name == "train_log" or mod_name.startswith("train_log."):
            del sys.modules[mod_name]

    if device.type == "cuda":
        # The official RIFE_HDv3.Model.device() resolves torch.device("cuda")
        # internally with no explicit index argument -- set_device is the only
        # way to make that resolve to the caller's chosen GPU in a multi-GPU setup.
        torch.cuda.set_device(device)

    rife_hd = importlib.import_module("train_log.RIFE_HDv3")
    model = rife_hd.Model()
    model.load_model(train_log_dir, -1)
    model.eval()
    model.device()
    return model


def pad_to_valid_size(img, scale=1.0):
    """Matches the official inference_video.py padding convention exactly:
    tmp = max(128, int(128 / scale)); pad up to a multiple of tmp."""
    import torch.nn.functional as F

    _n, _c, h, w = img.shape
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    return F.pad(img, padding), h, w


def interpolate_frame(model, img0, img1, timestep=0.5, scale=1.0):
    """img0/img1: NCHW float tensors, 0-1 range, same H/W, same device as model.
    Returns the single interpolated frame at the given `timestep` (0.0=img0,
    1.0=img1, 0.5=midpoint -- the previous, only-ever-used default), cropped back
    to the original size -- matches the official inference_video.py two-frame
    call convention (RIFE >=3.9 models take an explicit timestep; older ones only
    support the fixed midpoint via a 2-arg call, so any other timestep against
    one of those is a real caller error, not silently rounded to 0.5)."""
    padded0, h, w = pad_to_valid_size(img0, scale)
    padded1, _h, _w = pad_to_valid_size(img1, scale)
    if getattr(model, "version", 0) >= 3.9:
        middle = model.inference(padded0, padded1, timestep, scale)
    else:
        if timestep != 0.5:
            raise ValueError(
                f"RIFE model version {getattr(model, 'version', 0)} does not support arbitrary "
                f"timesteps (only the fixed midpoint, timestep=0.5) -- got timestep={timestep}")
        middle = model.inference(padded0, padded1, scale)
    return middle[:, :, :h, :w]
