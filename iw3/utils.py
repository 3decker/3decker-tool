import sys
import traceback
import os
import csv
import subprocess
from os import path
from datetime import datetime
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import threading
import math
from tqdm import tqdm
from PIL import Image
import contextlib
from nunif.initializer import gc_collect
from nunif.utils.image_loader import ImageLoader
from nunif.utils.pil_io import load_image_simple
import nunif.utils.shot_boundary_detection as SBD
from nunif.models import compile_model
import nunif.utils.video as VU
from nunif.utils.video.hdr_metadata import get_hdr_metadata
from nunif.utils.video.metadata import parse_time
from nunif.utils.ui import is_image, is_video, is_text, is_output_dir, make_parent_dir, list_subdir, TorchHubDir
from nunif.utils.ticket_lock import TicketLock
from nunif.utils.autocrop import AutoCrop, AutoCropDummy
from nunif.device import create_device, device_is_cuda, mps_is_available, xpu_is_available
from nunif.models.data_parallel import DeviceSwitchInference
from . import export_config
from .dilation import dilate_edge, edge_dilation_is_enabled
from .forward_warp import apply_divergence_forward_warp, SPLAT_BLEND_TEMPERATURE
from .anaglyph import apply_anaglyph_redcyan
from .mapper import get_mapper, resolve_mapper_name, MAPPER_ALL
from .depth_model_factory import create_depth_model
from .base_depth_model import BaseDepthModel
from .hub_dir import HUB_MODEL_DIR
from .equirectangular import equirectangular_projection
from .backward_warp import (
    apply_divergence_grid_sample,
    apply_divergence_monobw,
    apply_divergence_nn_LR,
)
from .stereo_model_factory import create_stereo_model
from .inpaint_utils import INPAINT_MODELS
from .convergence_estimator import ConvergenceEstimator
from .face_convergence_estimator import FaceConvergenceEstimator
from . import depth_effects as DE
from . import scene_boundary_cache as SceneBoundaryCache


ROW_FLOW_V2_MAX_DIVERGENCE = 2.5
ROW_FLOW_V3_MAX_DIVERGENCE = 5.0
ROW_FLOW_V2_AUTO_STEP_DIVERGENCE = 2.0
ROW_FLOW_V3_AUTO_STEP_DIVERGENCE = 4.0
IMAGE_IO_QUEUE_MAX = 100


def _find_dovi_tool():
    import shutil
    found = shutil.which("dovi_tool") or shutil.which("dovi_tool.exe")
    if found:
        return found
    here = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    for name in ("dovi_tool.exe", "dovi_tool"):
        candidate = path.join(here, name)
        if path.exists(candidate):
            return candidate
    return None


def _get_ffmpeg_bin():
    import shutil
    # Check nunif-windows root first (same directory as dovi_tool.exe)
    here = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    for name in ("ffmpeg.exe", "ffmpeg"):
        candidate = path.join(here, name)
        if path.exists(candidate):
            return candidate
    # PyAV bundled binary (older distributions)
    try:
        import av
        bundled = path.join(path.dirname(av.__file__), "ffmpeg.exe")
        if path.exists(bundled):
            return bundled
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


# Selectable "Target Resolution" values for the stereo-aware waifu2x upscale path
# (ADR-040) -- the FINAL PACKED (both eyes) frame width the user wants to land on.
# "auto" (default, off) means: don't compute a target at all, keep today's plain
# whole-frame _run_waifu2x_upscale behavior (fixed 2x/4x per the Method combo).
WAIFU2X_TARGET_PACKED_WIDTH = {"4k": 3840, "8k": 7680}


def _invoke_waifu2x_cli(input_path, output_path, method, noise_level, style, nunif_dir, log_prefix="[iw3]"):
    """The actual waifu2x.cli subprocess invocation, factored out of
    _run_waifu2x_upscale() so it is the single place this call is ever built --
    both the plain whole-frame upscale path (_run_waifu2x_upscale) and the
    per-eye stereo-aware path (waifu2x_upscale_stereo_cli.py, ADR-040) call this
    same function, so they can never drift apart on how waifu2x itself is
    invoked. Uses sys.executable (the same bundled Python already running this
    process) with -m, so there's no separate interpreter path to discover the
    way ffmpeg/dovi_tool need -- only the working directory needs to be pointed
    at the nunif/ package root explicitly, since a subprocess does not inherit
    this process's in-memory sys.path.

    Returns True on verified success (subprocess exit 0 AND the output file
    actually exists), False otherwise -- logs why in both failure cases itself.
    Verifies the output file actually exists rather than trusting the
    subprocess's exit code alone (see the HDR extraction silent-failure lesson
    in docs/ai/AI_KNOWN_ISSUES.md)."""
    cmd = [sys.executable, "-m", "waifu2x.cli",
           "-i", str(input_path), "-o", str(output_path),
           "-m", method, "-n", str(int(noise_level)),
           "--style", style, "-y"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=nunif_dir)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace").strip()
        print(f"{log_prefix} waifu2x upscale failed: {msg[:300]}", file=sys.stderr)
        return False
    if not path.exists(output_path):
        print(f"{log_prefix} waifu2x upscale exited 0 but produced no output file", file=sys.stderr)
        return False
    return True


def _run_waifu2x_upscale(output_path, args):
    """Optionally invokes waifu2x's own CLI as a subprocess against a just-finished
    iw3 output, when the user explicitly opted in (GUI: "Upscale with waifu2x after
    conversion" / --waifu2x-upscale). Runs as a genuinely separate subprocess rather
    than an in-process import+call so waifu2x's own model loading never competes for
    the same GPU memory iw3's depth/stereo models were just using in this same
    process -- matches this project's existing convention of treating heavyweight
    tools as subprocesses (see CODING_STANDARDS.md CS-SUBPROCESS-001).

    Returns the upscaled file's path on success, or None (having already logged why)
    on failure -- a failure here must never be mistaken for the conversion itself
    having failed, since output_path is untouched either way.

    This is the plain whole-frame path -- see _run_waifu2x_upscale_stereo (ADR-040)
    for the per-eye split/upscale/smooth path used instead when the output is a
    packed two-eye stereo video AND a Target Resolution (4K/8K) is selected
    (_should_use_stereo_upscale decides which one runs; this function is otherwise
    completely unchanged from before that feature existed)."""
    if not getattr(args, "waifu2x_upscale", False):
        return None
    method = getattr(args, "waifu2x_method", None) or "noise_scale2x"
    noise_level = getattr(args, "waifu2x_noise_level", None)
    if noise_level is None:
        noise_level = 1
    style = getattr(args, "waifu2x_style", None) or "photo"
    nunif_dir = path.dirname(path.dirname(path.abspath(__file__)))

    base, ext = path.splitext(str(output_path))
    upscaled_path = f"{base}_w2x{ext}"

    _notify_stage(args, STAGE_WAIFU2X_UPSCALE)
    print(f"[iw3] Upscaling finished output with waifu2x ({method}, noise={noise_level}, "
          f"style={style})...", file=sys.stderr)
    if not _invoke_waifu2x_cli(output_path, upscaled_path, method, noise_level, style, nunif_dir):
        return None
    print(f"[iw3] waifu2x upscale done: {upscaled_path}", file=sys.stderr)
    return upscaled_path


def _run_rife_interpolation(output_path, args):
    """Optionally invokes RIFE (iw3.rife_cli, a thin wrapper around the
    Practical-RIFE model -- see docs/ai/AI_DECISIONS.md ADR-029) as a subprocess
    against a just-finished iw3 output, when the user explicitly opted in (GUI:
    "Interpolate frames with RIFE after conversion" / --rife-interpolate).
    Follows the exact same pattern as _run_waifu2x_upscale() above: a genuinely
    separate subprocess (sys.executable -m iw3.rife_cli) launched only AFTER the
    conversion's own output file is fully written, so RIFE's model never competes
    for GPU memory with iw3's depth/stereo models in this same process, and a
    separate '<name>_rife<ext>' output file so a RIFE failure can never be
    mistaken for the conversion itself having failed -- the original output is
    always left untouched either way.

    RIFE interpolates the FINAL PACKED stereo frame (both eyes already combined),
    not each eye separately -- see rife_cli.py's module docstring for why, and
    the accepted packed-eye-seam tradeoff this implies.

    Mutually exclusive with --preserve-dowi (enforced earlier, in
    set_state_args() -- there is no way to assign correct DV/HDR10+ metadata to
    RIFE's synthetic in-between frames), so this function does not need to
    re-check that here.

    args.rife_target_fps (if set) takes priority over args.rife_multiplier --
    both are forwarded to rife_cli.py as-is, which re-validates the mutual
    exclusion and the "target must exceed source fps" rule itself (see
    docs/ai/AI_DECISIONS.md ADR-049); args.rife_multiplier defaults to the
    original 2x behavior when neither is set.

    Returns the interpolated file's path on success, or None (having already
    logged why) on failure -- verifies the output file actually exists rather
    than trusting the subprocess's exit code alone (see the HDR extraction
    silent-failure lesson in docs/ai/AI_KNOWN_ISSUES.md)."""
    if not getattr(args, "rife_interpolate", False):
        return None
    rife_model = getattr(args, "rife_model", None) or "rife_425"
    rife_multiplier = getattr(args, "rife_multiplier", None)
    rife_target_fps = getattr(args, "rife_target_fps", None)
    nunif_dir = path.dirname(path.dirname(path.abspath(__file__)))

    base, ext = path.splitext(str(output_path))
    interpolated_path = f"{base}_rife{ext}"
    cmd = [sys.executable, "-m", "iw3.rife_cli",
           "-i", str(output_path), "-o", interpolated_path,
           "--rife-model", rife_model]
    if rife_target_fps is not None:
        cmd += ["--rife-target-fps", str(rife_target_fps)]
    elif rife_multiplier is not None:
        cmd += ["--rife-multiplier", str(rife_multiplier)]

    _notify_stage(args, STAGE_RIFE_INTERPOLATE)
    print(f"[iw3] Interpolating finished output with RIFE ({rife_model})...", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=nunif_dir)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace").strip()
        print(f"[iw3] RIFE interpolation failed: {msg[:300]}", file=sys.stderr)
        return None
    if not path.exists(interpolated_path):
        print("[iw3] RIFE interpolation exited 0 but produced no output file", file=sys.stderr)
        return None
    print(f"[iw3] RIFE interpolation done: {interpolated_path}", file=sys.stderr)
    return interpolated_path


def _extract_dovi_rpu(input_path, rpu_path, ffmpeg_bin, dovi_bin, tmp_hevc):
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(input_path), "-c:v", "copy", "-an", "-f", "hevc", str(tmp_hevc)],
        check=True, capture_output=True,
    )
    subprocess.run(
        [dovi_bin, "extract-rpu", "-i", str(tmp_hevc), "-o", str(rpu_path)],
        check=True, capture_output=True,
    )


def _inject_dovi_rpu(output_path, rpu_path, ffmpeg_bin, dovi_bin, tmp_dir):
    hevc_out = path.join(tmp_dir, "_iw3_out.hevc")
    hevc_dv = path.join(tmp_dir, "_iw3_out_dv.hevc")
    final_tmp = path.splitext(output_path)[0] + ".dv_inject" + path.splitext(output_path)[1]
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(output_path), "-c:v", "copy", "-an", "-f", "hevc", hevc_out],
            check=True, capture_output=True,
        )
        subprocess.run(
            [dovi_bin, "inject-rpu", "-i", hevc_out, "-r", str(rpu_path), "-o", hevc_dv],
            check=True, capture_output=True,
        )
        subprocess.run(
            [ffmpeg_bin, "-y",
             "-i", str(output_path),
             "-i", hevc_dv,
             "-map", "1:v", "-map", "0:a?",
             "-c", "copy", "-copyts", final_tmp],
            check=True, capture_output=True,
        )
        os.replace(final_tmp, output_path)
    finally:
        for f in (hevc_out, hevc_dv):
            if path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def _find_mkvmerge():
    import shutil
    found = shutil.which("mkvmerge") or shutil.which("mkvmerge.exe")
    if found:
        return found
    # nunif-windows root, e.g. <root>\mkvtoolnix\mkvmerge.exe (same layout as ffmpeg.exe/dovi_tool.exe)
    here = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    for candidate in (
        path.join(here, "mkvtoolnix", "mkvmerge.exe"),
        path.join(here, "mkvmerge.exe"),
    ):
        if path.exists(candidate):
            return candidate
    for prog_dir in (r"C:\Program Files\MKVToolNix", r"C:\Program Files (x86)\MKVToolNix"):
        candidate = path.join(prog_dir, "mkvmerge.exe")
        if path.exists(candidate):
            return candidate
    return None


def _find_mkvpropedit():
    """Same resolution strategy as _find_mkvmerge() -- mkvpropedit ships alongside
    mkvmerge in the same bundled MKVToolNix folder (verified present at
    <root>/mkvtoolnix/mkvpropedit.exe)."""
    import shutil
    found = shutil.which("mkvpropedit") or shutil.which("mkvpropedit.exe")
    if found:
        return found
    here = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    for candidate in (
        path.join(here, "mkvtoolnix", "mkvpropedit.exe"),
        path.join(here, "mkvpropedit.exe"),
    ):
        if path.exists(candidate):
            return candidate
    for prog_dir in (r"C:\Program Files\MKVToolNix", r"C:\Program Files (x86)\MKVToolNix"):
        candidate = path.join(prog_dir, "mkvpropedit.exe")
        if path.exists(candidate):
            return candidate
    return None


# Matroska StereoMode values actually used here (see matroska.org/technical/elements.html,
# "StereoMode" under video track elements). "first" means which eye's view occupies the
# first (left, or top for top-bottom) half of the packed frame -- confirmed against the
# real spec text, not assumed. Only the 3 values iw3 can ever correctly produce are
# defined; the other 12 spec values (checkboard/row/column interleaved, anaglyph, laced)
# do not correspond to anything iw3 outputs and are intentionally not used.
STEREO_MODE_SBS_LEFT_FIRST = 1    # side by side, left eye's view in the left half
STEREO_MODE_TB_LEFT_FIRST = 3     # top - bottom, left eye's view on top
STEREO_MODE_SBS_RIGHT_FIRST = 11  # side by side, right eye's view in the left half


def _resolve_stereo_mode_value(args):
    """Maps iw3's resolved output stereo format to the Matroska StereoMode value that
    correctly describes how iw3 actually packs the two eye views into the frame, or
    None if the format is not a two-eye stereo pair StereoMode can describe at all.

    Eye placement verified directly from postprocess_image()'s actual frame-assembly
    code above (tensors are CHW -- dim=1 is height, dim=2 is width):
      - Normal SBS (half_sbs / the default Full SBS / vr180 "VR90"): the final `else`
        branch does `torch.cat([left_eye, right_eye], dim=2)` -> left eye occupies the
        LEFT half.
      - Cross Eyed: `torch.cat([right_eye, left_eye], dim=2)` (explicitly commented
        "# Reverse SideBySide" in that code) -> right eye occupies the LEFT half.
      - TB (full or half): `torch.cat([left_eye, right_eye], dim=1)` -> left eye
        occupies the TOP half.

    RGB-D / Half RGB-D store a depth channel, not a second eye view -- never a real
    stereo pair. Anaglyph is already a single merged image correct on any player
    without tagging. Debug Depth is not a real stereo output. All four always return
    None here; callers must skip tagging entirely rather than pick a nearest value.
    """
    if getattr(args, "rgbd", False) or getattr(args, "half_rgbd", False):
        return None
    if getattr(args, "anaglyph", None):
        return None
    if getattr(args, "debug_depth", False):
        return None
    if getattr(args, "cross_eyed", False):
        return STEREO_MODE_SBS_RIGHT_FIRST
    if getattr(args, "tb", False) or getattr(args, "half_tb", False):
        return STEREO_MODE_TB_LEFT_FIRST
    # half_sbs, vr180 ("VR90" in the GUI), and the unflagged default (Full SBS) all pack
    # via the same dim=2 cat in postprocess_image's final `else` branch.
    return STEREO_MODE_SBS_LEFT_FIRST


def _resolve_stereo_split_axis(args):
    """Maps the same resolved StereoMode value _resolve_stereo_mode_value already
    computes from args to which physical axis the two eyes are packed along, for
    the stereo-aware waifu2x upscale path (ADR-040): "sbs" (vertical seam -- both
    SBS StereoMode values map here, since a pure geometric split doesn't care
    which eye's VIEW occupies which half, only where the seam is -- see
    split_stereo_frame), "tb" (horizontal seam), or None when the resolved format
    isn't a two-eye pair at all (RGB-D, Half RGB-D, Anaglyph, Debug Depth -- same
    exclusions _resolve_stereo_mode_value already applies, for the same reason).

    Deliberately reuses _resolve_stereo_mode_value (real args from the actual run
    that produced the file) rather than the filename-suffix-tag detection pattern
    used by the STANDALONE retroactive tools (subtitle_mux_cli.py/
    stereo_mode_tag_cli.py) -- those tools duplicate suffix parsing because they
    run later, without the original args, against an arbitrary file. This call
    site is different: it always runs immediately after the SAME run's own args
    produced output_path (see _run_waifu2x_upscale_stereo's callers), so reusing
    the already-shared, already-correct resolver is strictly more reliable than
    re-deriving the same answer by guessing a filename suffix back apart."""
    value = _resolve_stereo_mode_value(args)
    if value in (STEREO_MODE_SBS_LEFT_FIRST, STEREO_MODE_SBS_RIGHT_FIRST):
        return "sbs"
    if value == STEREO_MODE_TB_LEFT_FIRST:
        return "tb"
    return None


def split_stereo_frame(frame, axis):
    """Splits a packed two-eye frame (any array with H,W as its first two axes --
    HWC or plain HxW, dtype-agnostic) into its two physical halves along the
    packing axis: "sbs" (vertical seam -> left half, right half) or "tb"
    (horizontal seam -> top half, bottom half). Pure slicing -- no eye-identity
    semantics (which eye's VIEW occupies which half, e.g. reversed for
    Cross-Eyed, is resolved separately by _resolve_stereo_split_axis's caller and
    is irrelevant to a pure geometric split/rejoin). An odd packed dimension
    gives the remainder pixel to the second half, so join_stereo_frame always
    reconstructs the exact original frame with no pixel loss."""
    if axis == "sbs":
        mid = frame.shape[1] // 2
        return frame[:, :mid, ...], frame[:, mid:, ...]
    elif axis == "tb":
        mid = frame.shape[0] // 2
        return frame[:mid, ...], frame[mid:, ...]
    raise ValueError(f"unknown stereo split axis: {axis!r}")


def join_stereo_frame(a, b, axis):
    """Inverse of split_stereo_frame -- concatenates the two eye halves back along
    the same axis. Round-trips split_stereo_frame pixel-for-pixel when neither
    half was resized in between (see the regression test in
    waifu2x_upscale_stereo_cli.py's --self-test)."""
    if axis == "sbs":
        return np.concatenate([a, b], axis=1)
    elif axis == "tb":
        return np.concatenate([a, b], axis=0)
    raise ValueError(f"unknown stereo split axis: {axis!r}")


def compute_stereo_upscale_plan(src_packed_w, src_packed_h, axis, target_packed_w):
    """Works out the concrete per-eye upscale plan needed to bring a packed
    two-eye source video up to a target FINAL PACKED width (e.g. 7680 for 8K),
    computed from the source's own real packed resolution -- never assumes a
    fixed 2x/4x always lands exactly on the requested target (ADR-040).

    target_packed_h is derived by preserving the SOURCE's own packed aspect
    ratio (not hardcoded to 16:9), so a non-16:9 source still lands on a
    correctly-proportioned final packed frame instead of a wrong fixed height.
    A useful consequence of always preserving the source's own aspect ratio
    this way: the needed per-eye scale factor always comes out UNIFORM (same
    on width and height, up to integer rounding) regardless of source aspect
    ratio or squeeze (Half-SBS/TB vs Full-SBS/TB) -- there is never a real case
    where width and height need genuinely different scale factors, so no
    aspect-distorting anisotropic upscale is ever required.

    scale_tier is the smaller of waifu2x's two available upscale factors (2x or
    4x -- the only tiers any bundled/external SR method here actually offers,
    see waifu2x/ui_utils.py's _method_type fixed_choices) that brings the
    upscaled eye to AT LEAST the target eye resolution in both dimensions; the
    caller (waifu2x_upscale_stereo_cli.py) resizes down to the exact target
    after upscaling, since a real SR model's fixed multiplier will essentially
    never land on an exact requested pixel count by itself."""
    if src_packed_w <= 0 or src_packed_h <= 0:
        raise ValueError(f"invalid source packed resolution: {src_packed_w}x{src_packed_h}")
    if axis not in ("sbs", "tb"):
        raise ValueError(f"unknown stereo split axis: {axis!r}")
    if target_packed_w <= 0:
        raise ValueError(f"invalid target packed width: {target_packed_w}")

    target_packed_h = round(target_packed_w * src_packed_h / src_packed_w)
    if axis == "sbs":
        src_eye_w, src_eye_h = src_packed_w // 2, src_packed_h
        target_eye_w, target_eye_h = target_packed_w // 2, target_packed_h
    else:  # tb
        src_eye_w, src_eye_h = src_packed_w, src_packed_h // 2
        target_eye_w, target_eye_h = target_packed_w, target_packed_h // 2

    needed_scale = max(target_eye_w / src_eye_w, target_eye_h / src_eye_h)
    scale_tier = 2 if needed_scale <= 2.0 else 4

    return {
        "target_packed_w": target_packed_w,
        "target_packed_h": target_packed_h,
        "src_eye_w": src_eye_w,
        "src_eye_h": src_eye_h,
        "target_eye_w": target_eye_w,
        "target_eye_h": target_eye_h,
        "needed_scale": needed_scale,
        "scale_tier": scale_tier,
    }


def resolve_waifu2x_method_for_tier(method, scale_tier):
    """Rewrites a user-chosen waifu2x --method string (e.g. "noise_scale2x") to use
    the given scale_tier (2 or 4, from compute_stereo_upscale_plan) instead,
    preserving whichever family (noise_scale / scale / realesrgan / bsrgan / an
    "onnx:<file>" custom model -- see waifu2x/ui_utils.py's _method_type) the
    user actually picked. Only the trailing 2x/4x multiplier is swapped, since
    that's the only part the scale-tier decision is actually about."""
    for suffix in ("4x", "2x"):
        if method.endswith(suffix):
            return f"{method[:-len(suffix)]}{scale_tier}x"
    # No recognizable 2x/4x suffix (a bare "noise"/"scale", or a custom onnx:
    # model with a fixed baked-in scale) -- nothing to rewrite; the final resize
    # step in waifu2x_upscale_stereo_cli.py still lands on the exact target.
    return method


def _should_use_stereo_upscale(args):
    """True only when the user opted into BOTH waifu2x upscaling AND a real
    (non-"auto") Target Resolution, AND the resolved output format is an actual
    two-eye packed stereo pair (_resolve_stereo_split_axis) -- e.g. RGB-D, Half
    RGB-D, Anaglyph, and Debug Depth output never qualify, same exclusions
    _resolve_stereo_mode_value already applies (not a two-eye pair to split at
    all). When this is False, the caller uses the existing plain whole-frame
    _run_waifu2x_upscale completely unchanged -- this is the single
    off-by-default gate for the entire per-eye split/smooth feature (ADR-040)."""
    if not getattr(args, "waifu2x_upscale", False):
        return False
    target = getattr(args, "waifu2x_upscale_target", None) or "auto"
    if target not in WAIFU2X_TARGET_PACKED_WIDTH:
        return False
    return _resolve_stereo_split_axis(args) is not None


def _run_waifu2x_upscale_stereo(output_path, args):
    """Per-eye, stereo-aware alternative to _run_waifu2x_upscale (ADR-040), used
    only when _should_use_stereo_upscale(args) is True: splits the packed output
    into its two independent eye videos at the correct axis, upscales EACH EYE
    INDEPENDENTLY with waifu2x (via the same _invoke_waifu2x_cli() subprocess
    call _run_waifu2x_upscale uses -- never duplicated), smooths each eye's own
    upscaled frame sequence independently with RGBTemporalStabilizer (never
    blending information across eyes), resizes to the exact computed per-eye
    target resolution, and recombines into the final packed output at the
    requested Target Resolution (4K/8K).

    Runs as a genuinely separate subprocess (iw3.waifu2x_upscale_stereo_cli),
    exactly mirroring _run_waifu2x_upscale/_run_rife_interpolation's own
    subprocess pattern -- waifu2x's model must never share GPU memory with
    iw3's depth/stereo models still resident in this process.

    Returns the upscaled file's path on success, or None (having already
    logged why) on failure -- output_path itself is never touched either way."""
    if not _should_use_stereo_upscale(args):
        return None
    axis = _resolve_stereo_split_axis(args)
    target_key = getattr(args, "waifu2x_upscale_target", None) or "auto"
    target_packed_w = WAIFU2X_TARGET_PACKED_WIDTH[target_key]
    method = getattr(args, "waifu2x_method", None) or "noise_scale2x"
    noise_level = getattr(args, "waifu2x_noise_level", None)
    if noise_level is None:
        noise_level = 1
    style = getattr(args, "waifu2x_style", None) or "photo"
    crf = getattr(args, "crf", None)
    crf = str(crf) if crf is not None else "20"
    preset = getattr(args, "preset", None) or "medium"
    nunif_dir = path.dirname(path.dirname(path.abspath(__file__)))

    # target_key ("4k"/"8k") is folded into the derivative filename itself
    # (_w2x4k / _w2x8k) so it is actually distinguishable from the plain
    # whole-frame _run_waifu2x_upscale path's "_w2x" output -- see
    # docs/ai/AI_DECISIONS.md ADR-050. Before this, both paths produced the
    # identical "<name>_w2x<ext>" filename, contradicting the comment in
    # _build_iw3_comment_metadata that claims "the derivative file's own
    # '_w2x' suffix already marks it".
    base, ext = path.splitext(str(output_path))
    upscaled_path = f"{base}_w2x{target_key}{ext}"

    _notify_stage(args, STAGE_WAIFU2X_UPSCALE)
    cmd = [sys.executable, "-m", "iw3.waifu2x_upscale_stereo_cli",
           "-i", str(output_path), "-o", upscaled_path,
           "--split-axis", axis,
           "--target-packed-width", str(target_packed_w),
           "--waifu2x-method", method,
           "--waifu2x-noise-level", str(int(noise_level)),
           "--waifu2x-style", style,
           "--crf", crf,
           "--preset", str(preset)]
    gpu = getattr(args, "gpu", None)
    if isinstance(gpu, (list, tuple)) and len(gpu) > 0:
        cmd += ["--gpu", str(gpu[0])]
    elif isinstance(gpu, int):
        cmd += ["--gpu", str(gpu)]

    print(f"[iw3] Stereo-aware upscaling finished output with waifu2x (split={axis}, "
          f"target={target_key}, method={method}, noise={noise_level}, style={style})...",
          file=sys.stderr)
    try:
        subprocess.run(cmd, check=True, capture_output=True, cwd=nunif_dir)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace").strip()
        print(f"[iw3] stereo-aware waifu2x upscale failed: {msg[:300]}", file=sys.stderr)
        return None
    if not path.exists(upscaled_path):
        print("[iw3] stereo-aware waifu2x upscale exited 0 but produced no output file", file=sys.stderr)
        return None
    print(f"[iw3] stereo-aware waifu2x upscale done: {upscaled_path}", file=sys.stderr)
    return upscaled_path


def _apply_stereo_mode_tag(output_path, args, mkvpropedit_bin=None):
    """Tags an already-produced MKV's video track with the Matroska StereoMode
    property (see _resolve_stereo_mode_value's docstring for the verified eye-order
    mapping) via mkvpropedit, so 3D-aware players/TVs (VLC, Kodi, compatible smart
    TVs) auto-detect the packing/eye-order and switch into 3D display automatically,
    instead of the viewer having to tell their player it's 3D by hand. An in-place
    edit of the container's metadata only -- mkvpropedit never re-encodes or re-muxes
    the actual audio/video streams, so this is fast even on a large finished file.

    A no-op (returns False, no side effects) when: the setting isn't enabled, the
    output isn't an .mkv (StereoMode is a Matroska-only property), the resolved
    format has no valid StereoMode value (RGB-D/Half RGB-D/Anaglyph/Debug Depth --
    see _resolve_stereo_mode_value), the file doesn't exist, or mkvpropedit isn't
    found (printed clearly rather than silently skipped in that specific case, since
    it means the feature was requested but genuinely cannot run).

    Follows CS-SUBPROCESS-001: subprocess.run with a list of args (never shell=True),
    wrapped in try/except so a failure here is printed and never mistaken for the
    conversion job itself having failed -- the file that was already successfully
    produced is left exactly as it was if tagging fails.
    """
    if not getattr(args, "stereo_mode_tag", False):
        return False
    if not output_path or path.splitext(str(output_path))[1].lower() != ".mkv":
        return False
    if not path.exists(output_path):
        return False
    stereo_value = _resolve_stereo_mode_value(args)
    if stereo_value is None:
        print("[iw3] --stereo-mode-tag has no effect on this Stereo Format -- RGB-D, Half "
              "RGB-D, Anaglyph, and Debug Depth are not a two-eye stereo pair Matroska's "
              "StereoMode can describe, so tagging is skipped.", file=sys.stderr)
        return False
    mkvpropedit_bin = mkvpropedit_bin or _find_mkvpropedit()
    if not mkvpropedit_bin:
        print("[iw3] --stereo-mode-tag was requested but mkvpropedit was not found -- "
              "skipping MKV StereoMode tagging.", file=sys.stderr)
        return False
    try:
        subprocess.run(
            [mkvpropedit_bin, str(output_path), "--edit", "track:v1",
             "--set", f"stereo-mode={stereo_value}"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[iw3] StereoMode tagging failed: {e.stderr.decode(errors='replace').strip()}",
              file=sys.stderr)
        return False
    print(f"[iw3] Tagged {path.basename(str(output_path))} as MKV StereoMode {stereo_value}.",
          file=sys.stderr)
    if getattr(args, "vr180", False):
        print("[iw3] Note: StereoMode only tells a player the eye packing/order -- it does "
              "NOT carry spherical/180-degree projection metadata, so this tag alone will "
              "not make a generic 3D-aware player or TV display VR90 output correctly as VR "
              "(see docs/ai/AI_DECISIONS.md ADR-033). VR headset apps still rely on the "
              "existing '_180x180_LR' filename convention to recognize this as VR180 "
              "content.", file=sys.stderr)
    return True


def _find_hdr10plus_tool():
    import shutil
    found = shutil.which("hdr10plus_tool") or shutil.which("hdr10plus_tool.exe")
    if found:
        return found
    here = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    for name in ("hdr10plus_tool.exe", "hdr10plus_tool"):
        candidate = path.join(here, name)
        if path.exists(candidate):
            return candidate
    return None


def _find_ffprobe():
    here = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    for name in ("ffprobe.exe", "ffprobe"):
        candidate = path.join(here, name)
        if path.exists(candidate):
            return candidate
    import shutil
    return shutil.which("ffprobe") or "ffprobe"


def _detect_hdr_types(input_path, ffprobe_bin):
    """Return which HDR metadata types are present in the source video stream."""
    import json as _json
    result = {"dv": False, "hdr10plus": False}
    try:
        proc = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-print_format", "json",
             "-show_streams", str(input_path)],
            capture_output=True, text=True, timeout=60,
        )
        data = _json.loads(proc.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            if stream.get("codec_tag_string", "") in ("dvh1", "dvhe", "dav1"):
                result["dv"] = True
            for sd in stream.get("side_data_list", []):
                sdt = sd.get("side_data_type", "")
                if "DOVI" in sdt or "Dolby" in sdt:
                    result["dv"] = True
                if "HDR Dynamic" in sdt or "SMPTE2094" in sdt:
                    result["hdr10plus"] = True
    except Exception:
        pass
    return result


def _detect_pq_or_hlg(input_path, ffprobe_bin):
    """True if the video stream's transfer characteristic is PQ (HDR10/HDR10+/most Dolby
    Vision) or HLG — the two curves that need real tone-mapping to look correct as SDR,
    as opposed to a plain SDR source that doesn't need any conversion at all."""
    import json as _json
    try:
        proc = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_streams", str(input_path)],
            capture_output=True, text=True, timeout=60,
        )
        data = _json.loads(proc.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            if stream.get("color_transfer", "") in ("smpte2084", "arib-std-b67"):
                return True
    except Exception:
        pass
    return False


def _tonemap_hdr_to_sdr(input_filename, args):
    """Pre-converts a PQ/HLG HDR source to a clean SDR 10-bit intermediate file using
    ffmpeg's zscale+tonemap filters. This can't be done inside iw3's own (PyAV-based)
    per-frame filter graph -- that build of PyAV doesn't have the zscale filter, and the
    filter it does have (colorspace) has no notion of the PQ/HLG curve at all, so it can't
    correctly linearize HDR. The standalone bundled ffmpeg.exe does have zscale, so this
    runs as a one-time pass through that before iw3's own pipeline starts.

    Returns (path_to_actually_process, temp_file_to_clean_up_or_None).
    """
    if not getattr(args, "hdr_to_sdr", False):
        return input_filename, None

    ffprobe_bin = _find_ffprobe()
    if not _detect_pq_or_hlg(input_filename, ffprobe_bin):
        print("[hdr-to-sdr] source isn't tagged as PQ/HLG HDR -- nothing to convert, "
              "processing it as-is.", file=sys.stderr)
        return input_filename, None

    if getattr(args, "preserve_dowi", False):
        # Preserving dynamic HDR/DV metadata on a deliberately-tonemapped SDR output is
        # contradictory -- the metadata would describe a HDR grade that no longer exists.
        print("[hdr-to-sdr] Preserve Dolby Vision is incompatible with converting to SDR; "
              "disabling it for this run.", file=sys.stderr)
        args.preserve_dowi = False

    ffmpeg_bin = _get_ffmpeg_bin()
    out_dir = path.dirname(path.abspath(input_filename))
    tmp_sdr = path.join(
        out_dir, "_iw3_hdr_to_sdr_" + path.splitext(path.basename(input_filename))[0] + ".mkv")

    trim_args = []
    if getattr(args, "start_time", None):
        trim_args += ["-ss", str(parse_time(args.start_time))]
    if getattr(args, "end_time", None):
        trim_args += ["-to", str(parse_time(args.end_time))]

    print("[hdr-to-sdr] converting HDR source to SDR (10-bit retained) before processing "
          "-- this adds one extra encoding pass.", file=sys.stderr)
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", *trim_args, "-i", str(input_filename),
             "-vf", "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                    "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p10le",
             "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-crf", "12",
             "-c:a", "copy", tmp_sdr],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[hdr-to-sdr] conversion failed, processing the original HDR source instead: "
              f"{e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
        return input_filename, None

    if trim_args:
        # The intermediate file already covers exactly [start_time, end_time]; clear those
        # so the rest of the pipeline doesn't try to trim an already-trimmed file again.
        args.start_time = None
        args.end_time = None

    return tmp_sdr, tmp_sdr


def _denoise_preprocess(input_filename, args):
    """Pre-denoises the source using the standalone bundled ffmpeg.exe's real hqdn3d filter,
    producing a clean intermediate file before iw3's own pipeline starts. This can't be done
    as a per-frame filter inside iw3's own pipeline -- PyAV's bundled libavfilter doesn't have
    hqdn3d, and the GPU tensor-frame decode path (used whenever HWAccel=cuda) only supports a
    small fixed set of geometric filters with no denoiser at all -- so neither internal path
    can run a denoiser reliably. The standalone ffmpeg.exe does have hqdn3d, so this runs as a
    one-time pass through that, the same way HDR-to-SDR tonemap does.

    Returns (path_to_actually_process, temp_file_to_clean_up_or_None).
    """
    if not getattr(args, "denoise", False):
        return input_filename, None

    ffmpeg_bin = _get_ffmpeg_bin()
    out_dir = path.dirname(path.abspath(input_filename))
    tmp_denoised = path.join(
        out_dir, "_iw3_denoise_" + path.splitext(path.basename(input_filename))[0] + ".mkv")

    trim_args = []
    if getattr(args, "start_time", None):
        trim_args += ["-ss", str(parse_time(args.start_time))]
    if getattr(args, "end_time", None):
        trim_args += ["-to", str(parse_time(args.end_time))]

    # Get source duration for progress percentage
    try:
        import av as _av
        with _av.open(str(input_filename), metadata_errors="ignore") as _c:
            _dur = float(_c.duration) / 1000000.0 if _c.duration else None
        start_sec = parse_time(getattr(args, "start_time", None)) if getattr(args, "start_time", None) else 0.0
        end_sec = parse_time(getattr(args, "end_time", None)) if getattr(args, "end_time", None) else _dur
        total_sec = (end_sec - start_sec) if (end_sec is not None) else _dur
    except Exception:
        total_sec = None

    tqdm_fn = (getattr(args, "state", None) or {}).get("tqdm_fn") or tqdm
    stop_event = (getattr(args, "state", None) or {}).get("stop_event")
    desc = f"{path.basename(input_filename)}: Denoise"
    pbar = tqdm_fn(desc=desc, total=100, ncols=len(desc) + 60)

    print("[denoise] pre-denoising source with hqdn3d before processing -- this adds one "
          "extra encoding pass.", file=sys.stderr)
    try:
        # Use -progress pipe:2 so ffmpeg emits machine-readable key=value progress
        # lines on stderr (e.g. out_time_ms=12345) that we can parse to drive the
        # tqdm bar without relying on the carriage-return terminal output format.
        proc = subprocess.Popen(
            [ffmpeg_bin, "-y", *trim_args, "-i", str(input_filename),
             "-vf", "hqdn3d=1:1:6:6",
             "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-crf", "12",
             "-c:a", "copy", "-progress", "pipe:2", tmp_denoised],
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        last_pct = 0
        for line in proc.stderr:
            if stop_event is not None and stop_event.is_set():
                proc.terminate()
                break
            line = line.strip()
            if line.startswith("out_time_ms=") and total_sec and total_sec > 0:
                try:
                    us = int(line.split("=", 1)[1])  # value is microseconds despite the name
                    current_sec = us / 1000000.0 - start_sec
                    pct = min(int(current_sec / total_sec * 100), 100)
                    if pct > last_pct:
                        pbar.update(pct - last_pct)
                        last_pct = pct
                except (ValueError, IndexError):
                    pass
        proc.wait()
        pbar.update(100 - last_pct)
        pbar.close()
        if proc.returncode not in (0, None) and not (stop_event is not None and stop_event.is_set()):
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
    except subprocess.CalledProcessError as e:
        pbar.close()
        print(f"[denoise] pre-denoise failed, processing the original source instead.",
              file=sys.stderr)
        return input_filename, None
    except Exception as e:
        pbar.close()
        print(f"[denoise] pre-denoise error: {e}", file=sys.stderr)
        return input_filename, None

    if trim_args:
        # The intermediate file already covers exactly [start_time, end_time]; clear those
        # so the rest of the pipeline doesn't try to trim an already-trimmed file again.
        args.start_time = None
        args.end_time = None

    return tmp_denoised, tmp_denoised


_HEVC_ENCODER_NAMES = {"libx265", "hevc_nvenc", "hevc_qsv", "hevc_amf"}


def _inject_hdr_rpu(output_path, rpu_path, hdr10plus_json, ffmpeg_bin, dovi_bin, hdr10plus_bin, tmp_dir,
                     configured_video_codec=None):
    """Phase 1 of DV/HDR10+ injection: extract the output's video-only HEVC stream and
    inject RPU and/or HDR10+ into it. Returns (injected_hevc_path, fps_str, reason) --
    reason is None on success, or a human-readable string explaining why injection was
    declined/failed on failure (injected_hevc_path is None in that case). The caller is
    responsible for remuxing the result back into the container via
    _remux_injected_hevc, and that call is what deletes the returned file.

    `reason` exists specifically so the caller can put the real cause into
    iw3_step_timing.log -- this previously only ever printed to stderr, which is
    invisible in the GUI's windowless (pythonw) process, so a failure here looked
    like an unexplained instant no-op with no way to diagnose it after the fact.

    `configured_video_codec` (e.g. args.video_codec, pass-through from whatever
    Final Render was actually told to encode with) lets the codec check be trusted
    instead of re-read from disk -- see the two-tier logic below for why this
    matters, found necessary by real testing on 2026-09-06."""
    import json as _json
    import time as _time

    def _wait_for_file_stable(p, checks=3, interval=2.0, max_wait=30.0):
        """Waits until the file's size stops changing across `checks` consecutive
        samples `interval` seconds apart -- a quick sanity check, NOT the main
        defense (see the much longer ffprobe retry budget below): a real MKV job
        was observed to fail ffprobe's stream check with "no usable video stream"
        (not a crash, not a codec mismatch -- ffprobe ran fine and just found
        nothing) for several real minutes after Final Render reported done and the
        file's size had already stopped changing -- confirmed by manually re-running
        the exact same check ~5 minutes later, which then worked first try. So file
        SIZE stabilizing is necessary but not sufficient; something (most likely
        antivirus real-time scanning a large freshly-written video file on Windows,
        based on everything else already ruled out for this project) can still hold
        the file in a not-fully-readable state well after its size looks final.
        Returns True if size stabilized within max_wait, False otherwise (caller
        proceeds anyway rather than waiting forever -- this is a head start against
        the common case, not a guarantee)."""
        start = _time.monotonic()
        last_size = -1
        stable = 0
        while _time.monotonic() - start < max_wait:
            try:
                size = os.path.getsize(p)
            except OSError:
                size = -1
            if size == last_size and size > 0:
                stable += 1
                if stable >= checks:
                    return True
            else:
                stable = 0
            last_size = size
            _time.sleep(interval)
        return False

    ffprobe_bin = ffmpeg_bin.replace("ffmpeg", "ffprobe").replace("ffmpeg.exe", "ffprobe.exe")
    codec = None
    fps_str = "30fps"
    last_error = None
    trusted_hevc = configured_video_codec in _HEVC_ENCODER_NAMES

    if trusted_hevc:
        # Trust what Final Render was actually configured to encode with, instead of
        # re-reading the codec back from the file it just wrote. Real testing on
        # 2026-09-06 found ffprobe reporting "no usable video stream" on a file that
        # (a) was confirmed genuinely HEVC by a completely separate process reading
        # it minutes later, and (b) STILL failed this same check from inside the
        # SAME process after a full ~5-minute retry budget was exhausted -- pointing
        # at something process-local (write buffering not yet visible to a re-open
        # from the same process, not a generic external lock a longer wait would
        # clear) rather than a simply slow external lock. Re-reading the codec back
        # at all is unnecessary anyway: we already know what we told the encoder to
        # produce. Still makes ONE quick attempt to read back the real frame rate
        # for accurate remuxing, but a failure there is NOT fatal -- falls back to
        # the "30fps" default rather than blocking injection entirely on it.
        codec = "hevc"
        try:
            probe = subprocess.run(
                [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_streams", str(output_path)],
                capture_output=True, text=True, timeout=10,
            )
            streams = _json.loads(probe.stdout).get("streams", [])
            vid = next((s for s in streams if s.get("codec_type") == "video"), None)
            if vid:
                r = vid.get("r_frame_rate", "30/1")
                try:
                    num, den = r.split("/")
                    fps_str = f"{int(num)/int(den):.6f}fps"
                except Exception:
                    fps_str = f"{r}fps"
        except Exception:
            pass  # fps_str stays at the safe default -- codec is already trusted
    else:
        # Unknown/non-HEVC-family configured codec, or the caller didn't pass one --
        # can't trust a setting we don't recognize, so fall back to the full
        # wait-and-retry verification against the actual file.
        _wait_for_file_stable(str(output_path))
        attempts = 20
        for attempt in range(attempts):
            try:
                probe = subprocess.run(
                    [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_streams", str(output_path)],
                    capture_output=True, text=True, timeout=30,
                )
                streams = _json.loads(probe.stdout).get("streams", [])
                vid = next((s for s in streams if s.get("codec_type") == "video"), None)
                if vid:
                    codec = vid.get("codec_name")
                    r = vid.get("r_frame_rate", "30/1")
                    try:
                        num, den = r.split("/")
                        fps_str = f"{int(num)/int(den):.6f}fps"
                    except Exception:
                        fps_str = f"{r}fps"
                if not codec:
                    last_error = f"ffprobe returned no usable video stream (stderr: {probe.stderr.strip()[:200]})"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
            if codec:
                break
            if attempt < attempts - 1:
                _time.sleep(15)
        if codec != "hevc":
            reason = (f"could not confirm HEVC output after {attempt + 1} attempt(s) over ~{15 * attempt}s "
                      f"(plus an earlier file-stability wait) "
                      f"(codec detected: {codec!r}{'; last error: ' + last_error if last_error else ''}) -- "
                      f"DV/HDR10+ injection requires HEVC output (use --video-codec libx265)")
            print(f"--preserve-dowi: {reason}.", file=sys.stderr)
            return None, None, reason

    hevc_out = path.join(tmp_dir, "_iw3_out.hevc")
    hevc_dv = path.join(tmp_dir, "_iw3_out_dv.hevc")
    hevc_h10p = path.join(tmp_dir, "_iw3_out_h10p.hevc")
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(output_path), "-c:v", "copy", "-an", "-f", "hevc", hevc_out],
        check=True, capture_output=True,
    )
    current = hevc_out
    try:
        if rpu_path and path.exists(str(rpu_path)):
            subprocess.run(
                [dovi_bin, "inject-rpu", "-i", current, "-r", str(rpu_path), "-o", hevc_dv],
                check=True, capture_output=True,
            )
            current = hevc_dv

        if hdr10plus_json and path.exists(str(hdr10plus_json)):
            subprocess.run(
                [hdr10plus_bin, "inject", "-i", current, "-j", str(hdr10plus_json), "-o", hevc_h10p],
                check=True, capture_output=True,
            )
            current = hevc_h10p
    finally:
        # clean up whichever intermediate(s) did NOT end up being the one returned
        for f in (hevc_out, hevc_dv, hevc_h10p):
            if f != current and path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
    return current, fps_str, None


def _remux_injected_hevc(output_path, injected_hevc_path, fps_str, ffmpeg_bin, tmp_dir):
    """Phase 2 of DV/HDR10+ injection: remux an already-injected HEVC stream (from
    _inject_hdr_rpu) back into output_path's container in place of its original video
    stream. Deletes injected_hevc_path (and its own temp file) when done, regardless
    of success or failure."""
    final_tmp = path.splitext(output_path)[0] + ".hdr_inject" + path.splitext(output_path)[1]
    try:
        ext = path.splitext(output_path)[1].lower()
        mkvmerge_bin = _find_mkvmerge() if ext == ".mkv" else None
        if mkvmerge_bin:
            # mkvmerge correctly handles raw HEVC timestamps
            subprocess.run(
                [mkvmerge_bin, "-o", final_tmp,
                 "--default-duration", f"0:{fps_str}",
                 injected_hevc_path,
                 "--no-video", str(output_path)],
                check=True, capture_output=True,
            )
        else:
            subprocess.run(
                [ffmpeg_bin, "-y",
                 "-i", str(output_path),
                 "-i", injected_hevc_path,
                 "-map", "1:v", "-map", "0:a?",
                 "-c", "copy", "-copyts", final_tmp],
                check=True, capture_output=True,
            )
        os.replace(final_tmp, output_path)
    finally:
        for f in (injected_hevc_path, final_tmp):
            if f and path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def _inject_hdr_metadata(output_path, rpu_path, hdr10plus_json, ffmpeg_bin, dovi_bin, hdr10plus_bin, tmp_dir,
                          configured_video_codec=None):
    """Inject DV RPU and/or HDR10+ into the output HEVC, then remux back into the
    container. Convenience wrapper combining _inject_hdr_rpu + _remux_injected_hevc in
    one call, for callers (a normal conversion) that don't need the two phases
    reported as separate progress steps the way Dual-Pass Depth Blend does."""
    # _inject_hdr_rpu already prints its own reason to stderr on failure -- nothing
    # extra needed here, just don't crash on the 3rd (reason) element it now returns.
    injected_hevc, fps_str, _reason = _inject_hdr_rpu(output_path, rpu_path, hdr10plus_json,
                                                       ffmpeg_bin, dovi_bin, hdr10plus_bin, tmp_dir,
                                                       configured_video_codec=configured_video_codec)
    if injected_hevc is None:
        return
    _remux_injected_hevc(output_path, injected_hevc, fps_str, ffmpeg_bin, tmp_dir)


_UPGRADE_PIX_FMT_MAP = {
    10: {
        "yuv420p":  "yuv420p10le",
        "yuvj420p": "yuv420p10le",
        "yuv422p":  "yuv422p10le",
        "yuv444p":  "yuv444p10le",
        "yuvj444p": "yuv444p10le",
        "gbrp":     "gbrp10le",
        "rgb24":    "gbrp10le",
    },
    12: {
        "yuv420p":    "yuv420p12le",
        "yuvj420p":   "yuv420p12le",
        "yuv420p10le": "yuv420p12le",
        "yuv422p":    "yuv422p12le",
        "yuv422p10le": "yuv422p12le",
        "yuv444p":    "yuv444p12le",
        "yuvj444p":   "yuv444p12le",
        "yuv444p10le": "yuv444p12le",
        "gbrp":       "gbrp12le",
        "gbrp10le":   "gbrp12le",
        "rgb24":      "gbrp12le",
    },
}

# NVENC/QSV hardware encoder chips can only encode 8-bit or 10-bit video — there is no
# such thing as 12-bit hardware encoding on this kind of chip, on any GPU generation
# including current ones. Asking avcodec_open2 for a 12-bit format on these codecs doesn't
# get tone-mapped down like the CLI ffmpeg tool does for other pix_fmt mismatches; it just
# fails outright. So for these codecs specifically, cap "Bit Depth Upgrade: 12" down to 10.
#
# Separately, only plain yuv420p/yuv420p10le get auto-translated to NVENC's native
# nv12/p010le elsewhere in this project (see configure_video_codec() in
# nunif/utils/video/color_transform.py). There is no equivalent translation for a 10-bit
# yuv444p/yuv422p on NVENC/QSV, so requesting the upgrade on those would also crash — fall
# back to the plain 8-bit version of those instead.
_HW_ENCODER_NO_12BIT = {"h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv"}
_12BIT_TO_10BIT_PIX_FMT = {
    "yuv420p12le": "yuv420p10le",
    "yuv422p12le": "yuv422p10le",
    "yuv444p12le": "yuv444p10le",
    "gbrp12le":    "gbrp10le",
}
_HW_ENCODER_NO_HIGHBIT_TRANSLATION = {
    "yuv422p10le": "yuv422p",
    "yuv444p10le": "yuv444p",
}


def _clamp_pix_fmt_for_codec(pix_fmt, video_codec):
    if video_codec in _HW_ENCODER_NO_12BIT:
        pix_fmt = _12BIT_TO_10BIT_PIX_FMT.get(pix_fmt, pix_fmt)
        pix_fmt = _HW_ENCODER_NO_HIGHBIT_TRANSLATION.get(pix_fmt, pix_fmt)
    return pix_fmt


def _upgrade_pix_fmt(pix_fmt, target_bits):
    return _UPGRADE_PIX_FMT_MAP.get(target_bits, {}).get(pix_fmt, pix_fmt)


def print_exception(filename):
    e_type, e, tb = sys.exc_info()
    message = getattr(e, "message", str(e))
    print(f"Error: {filename}: {message}", file=sys.stderr)
    traceback.print_tb(tb, file=sys.stderr)


def chunks(array, n):
    for i in range(0, len(array), n):
        yield array[i:i + n]


def to_pil_image(x):
    # x is already clipped to 0-1
    assert x.dtype in {torch.float32, torch.float16}
    x = TF.to_pil_image((x * 255).round().to(torch.uint8).cpu())
    return x


def apply_rgbd(im, depth, mapper):
    height, width = im.shape[-2:]
    left_eye = im
    if mapper is not None:
        depth = get_mapper(mapper)(depth)

    if depth.ndim == 3:
        right_eye = F.interpolate(depth.unsqueeze(0), (height, width),
                                  mode="bicubic", antialias=True).squeeze(0)
    else:
        right_eye = F.interpolate(depth, (height, width),
                                  mode="bicubic", antialias=True)

    right_eye = right_eye.expand_as(left_eye)
    return left_eye, right_eye


# Filename suffix for VR Player's video format detection
# LRF: full left-right 3D video
FULL_SBS_SUFFIX = "_LRF_Full_SBS"
HALF_SBS_SUFFIX = "_LR"
FULL_TB_SUFFIX = "_TBF_fulltb"
HALF_TB_SUFFIX = "_TB"
CROSS_EYED_SUFFIX = "_RLF_cross"
RGBD_SUFFIX = "_RGBD"  # TODO
HALF_RGBD_SUFFIX = "_HRGBD"  # TODO

VR180_SUFFIX = "_180x180_LR"
ANAGLYPH_SUFFIX = "_redcyan"
DEBUG_SUFFIX = "_debug"

# SMB Invalid characters
# Linux SMB replaces file names with random strings if they contain these invalid characters
# So need to remove these for the filenaming rules.
SMB_INVALID_CHARS = '\\/:*?"<>|'


def _scene_auto_ema_active(args, ema_normalize, video=True):
    """True when --scene-batch-auto-ema governs a REGULAR (non---scene-batch)
    conversion's EMA settings -- i.e. it's on, --scene-detect is also on (the only
    way it has scene boundaries to key off of), and this isn't one of --scene-batch's
    own per-scene file conversions. --scene-batch already sets args.ema_decay/
    args.ema_buffer to that SPECIFIC scene's real overridden values before calling
    process_video() per scene (see scene_batch._process_scenes), so its own per-file
    output already tags a truthful, non-misleading fixed number -- excluded here via
    `not args.scene_batch` so that existing behavior is untouched. Used by both
    make_output_filename/_build_iw3_comment_metadata (a misleading fixed ema_decay/
    ema_buffer would otherwise be tagged, since a regular conversion never mutates
    those for tagging purposes -- see process_video_full) and, with the max-fps-
    adjusted ema_normalize local, by process_video_full itself to decide whether to
    precompute a per-scene EMA schedule at all."""
    return bool(
        video and ema_normalize and getattr(args, "scene_detect", False)
        and getattr(args, "scene_batch_auto_ema", False) and not getattr(args, "scene_batch", False)
    )


def make_output_filename(input_filename, args, video=False):
    basename = path.splitext(path.basename(input_filename))[0]
    basename = basename.translate({ord(c): ord("_") for c in SMB_INVALID_CHARS})
    if args.vr180:
        auto_detect_suffix = VR180_SUFFIX
    elif args.half_sbs:
        auto_detect_suffix = HALF_SBS_SUFFIX
    elif args.tb:
        auto_detect_suffix = FULL_TB_SUFFIX
    elif args.half_tb:
        auto_detect_suffix = HALF_TB_SUFFIX
    elif args.cross_eyed:
        auto_detect_suffix = CROSS_EYED_SUFFIX
    elif args.anaglyph:
        auto_detect_suffix = ANAGLYPH_SUFFIX + f"_{args.anaglyph}"
    elif args.rgbd:
        auto_detect_suffix = RGBD_SUFFIX
    elif args.half_rgbd:
        auto_detect_suffix = HALF_RGBD_SUFFIX
    elif args.debug_depth:
        auto_detect_suffix = DEBUG_SUFFIX
    else:
        auto_detect_suffix = FULL_SBS_SUFFIX

    def to_deciaml(f, scale, zfill=0):
        s = str(int(f * scale))
        if zfill:
            s = s.zfill(zfill)
        return s

    if args.metadata == "filename":
        if args.resolution:
            resolution = f"{args.resolution}_"
        else:
            resolution = ""
        if args.tta:
            tta = "TTA_"
        else:
            tta = ""
        if args.depth_aa:
            daa = "AA_"
        else:
            daa = ""
        if args.ema_normalize and video:
            if _scene_auto_ema_active(args, args.ema_normalize, video):
                auto_model = getattr(args, "scene_batch_auto_ema_model", None) or "3DECKER VDA_L"
                ema = f"_autoema{auto_model}"
            else:
                ema_ma = "ma" if getattr(args, "ema_motion_adaptive", False) else ""
                ema = f"_ema{to_deciaml(args.ema_decay, 100, 2)}b{args.ema_buffer}{ema_ma}"
        else:
            ema = ""
        if isinstance(args.edge_dilation, (list, tuple)):
            edge_dilation = "x".join([str(v) for v in args.edge_dilation])
        else:
            edge_dilation = args.edge_dilation
        if args.convergence_mode != "constant":
            convergence_name = "ac"
            convergence_smoothing = f"cs{to_deciaml(getattr(args, 'convergence_smoothing', 0.9), 100, 2)}"
        else:
            convergence_name = "c"
            convergence_smoothing = ""
        if video and args.video_codec == "libopenh264":
            bitrate = f"_br{args.video_bitrate}"
        elif video:
            bitrate = f"_crf{args.crf}"
        else:
            bitrate = ""

        if getattr(args, "background_pop", 0.0) > 0:
            bp = f"_bp{args.background_pop}"
            if getattr(args, "background_pop_coverage", 0.15) != 0.15:
                bp += f"c{to_deciaml(args.background_pop_coverage, 100, 2)}"
        else:
            bp = ""
        if getattr(args, "foreground_divergence", None) is not None:
            fd = f"_fd{to_deciaml(args.foreground_divergence, 10, 2)}"
        else:
            fd = ""
        if getattr(args, "background_divergence", None) is not None:
            bd = f"_bd{to_deciaml(args.background_divergence, 10, 2)}"
        else:
            bd = ""
        if getattr(args, "depth_refine", False):
            drefine = "_dr"
            dr_strength = getattr(args, "depth_refine_strength", 1.0) or 1.0
            if dr_strength != 1.0:
                drefine += f"{to_deciaml(dr_strength, 100, 2)}"
        else:
            drefine = ""
        if getattr(args, "temporal_stabilize", False) and video:
            ts_strength = getattr(args, "temporal_stabilize_strength", 0.7) or 0.7
            tstab = f"_ts{to_deciaml(ts_strength, 100, 2)}"
            ts_max_shift = getattr(args, "temporal_stabilize_max_shift_velocity", None)
            if ts_max_shift is not None:
                tstab += f"ms{to_deciaml(ts_max_shift, 100, 2)}"
            ts_flat_boost = getattr(args, "temporal_stabilize_flat_region_boost", 0.0) or 0.0
            if ts_flat_boost != 0.0:
                tstab += f"fb{to_deciaml(ts_flat_boost, 100, 2)}"
            ts_edge_protect = getattr(args, "temporal_stabilize_edge_protection", 0.0) or 0.0
            if ts_edge_protect != 0.0:
                tstab += f"ep{to_deciaml(ts_edge_protect, 100, 2)}"
        else:
            tstab = ""
        if getattr(args, "depth_blend", False):
            db_strength = getattr(args, "depth_blend_strength", 1.0) or 1.0
            db_region = getattr(args, "depth_blend_region", None) or "detail"
            if db_region == "detail":
                db_region_tag = "detail"
            else:
                db_percent = getattr(args, "depth_blend_region_percent", 25.0) or 25.0
                db_region_tag = f"{'fg' if db_region == 'foreground' else 'bg'}{int(db_percent)}"
            dblend = (f"_db{getattr(args, 'depth_blend_model', None) or 'VDA_L'}"
                      f"s{to_deciaml(db_strength, 100, 2)}{db_region_tag}")
            # Feather/Bilateral/CLAHE are optional post-processing on top of the blend
            # above -- each only appears in the name when actually enabled/non-default,
            # same convention as everything else here, so the common case (all off)
            # doesn't grow the filename at all.
            db_feather = int(getattr(args, "depth_blend_feather_blur", 0) or 0)
            if db_feather > 0:
                dblend += f"f{db_feather}"
            if getattr(args, "depth_blend_bilateral", False):
                bi_d = int(getattr(args, "depth_blend_bilateral_d", 12) or 12)
                bi_c = getattr(args, "depth_blend_bilateral_sigma_color", 75.0) or 75.0
                bi_s = getattr(args, "depth_blend_bilateral_sigma_space", 75.0) or 75.0
                dblend += f"bi{bi_d}c{int(bi_c)}s{int(bi_s)}"
            if getattr(args, "depth_blend_clahe", False):
                cl_clip = getattr(args, "depth_blend_clahe_clip", 2.0) or 2.0
                cl_tile = int(getattr(args, "depth_blend_clahe_tile", 8) or 8)
                dblend += f"cl{to_deciaml(cl_clip, 10, 1)}t{cl_tile}"
            if getattr(args, "depth_blend_align", False):
                dblend += "algn"
                db_align_decay = getattr(args, "depth_blend_align_decay", 0.9) or 0.9
                if db_align_decay != 0.9:
                    dblend += f"d{to_deciaml(db_align_decay, 100, 2)}"
            db_edge_supp = getattr(args, "depth_blend_edge_suppression", 0.5) or 0.5
            if db_edge_supp != 0.5:
                dblend += f"es{to_deciaml(db_edge_supp, 100, 2)}"
            if getattr(args, "depth_blend_edge_hard_cutoff", False):
                dblend += "hc"
        else:
            dblend = ""

        # Inpaint-specific settings only mean anything for inpaint stereo methods, and
        # only worth naming when they differ from the effective default -- otherwise
        # every single inpaint-method filename would carry the same fixed boilerplate.
        inpaint_methods = {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}
        if args.method in inpaint_methods:
            inpaint_model_val = getattr(args, "inpaint_model", None) or "light_inpaint_v1"
            im_tag = f"_im{inpaint_model_val}" if inpaint_model_val != "light_inpaint_v1" else ""

            overlap = getattr(args, "inpaint_overlap_frames", None) or [3, 3]
            overlap = list(overlap) if isinstance(overlap, (list, tuple)) else [overlap, overlap]
            iof_tag = f"_iof{overlap[0]}x{overlap[-1]}" if overlap != [3, 3] else ""

            mask_inner = getattr(args, "mask_inner_dilation", 0) or 0
            mask_outer = getattr(args, "mask_outer_dilation", 0) or 0
            imd_tag = f"_imd{mask_inner}x{mask_outer}" if (mask_inner or mask_outer) else ""

            max_w = getattr(args, "inpaint_max_width", None)
            imw_tag = f"_imw{int(max_w)}" if max_w else ""
        else:
            im_tag = iof_tag = imd_tag = imw_tag = ""

        # Splat blend temperature only means anything for forward_splat_fill, and only
        # worth naming when it differs from that method's original fixed behavior --
        # same "only shown when non-default" convention as everything else here.
        if args.method == "forward_splat_fill":
            splat_temp = getattr(args, "splat_blend_temperature", SPLAT_BLEND_TEMPERATURE)
            splat_temp = SPLAT_BLEND_TEMPERATURE if splat_temp is None else splat_temp
            spt_tag = f"_spt{to_deciaml(splat_temp, 10)}" if splat_temp != SPLAT_BLEND_TEMPERATURE else ""
        else:
            spt_tag = ""

        stereo_w = getattr(args, "stereo_width", None)
        sw_tag = f"_sw{int(stereo_w)}" if stereo_w else ""

        sbd_tag = "_sbd" if getattr(args, "scene_detect", False) and video else ""
        psb_tag = "_psb" if getattr(args, "preserve_screen_border", False) else ""

        edge_repair_strength = getattr(args, "edge_repair_strength", 0.0) or 0.0
        er_tag = f"_er{to_deciaml(edge_repair_strength, 100, 2)}" if edge_repair_strength > 0.0 else ""

        if getattr(args, "sharpen", False):
            sharpen_strength = getattr(args, "sharpen_strength", None)
            sharpen_strength = 0.5 if sharpen_strength is None else sharpen_strength
        else:
            sharpen_strength = 0.0
        sharp_tag = f"_sharp{to_deciaml(sharpen_strength, 100, 2)}" if sharpen_strength > 0.0 else ""

        if getattr(args, "rife_interpolate", False):
            rife_model_val = getattr(args, "rife_model", None) or "rife_425"
            rife_tag = "_rife"
            if rife_model_val != "rife_425":
                rife_tag += rife_model_val[len("rife_"):] if rife_model_val.startswith("rife_") else rife_model_val
            # Multiplier/target-fps suffix, same "only when non-default" convention --
            # --rife-target-fps (when given) always wins over --rife-multiplier (see
            # ADR-049/set_state_args()'s mutual-exclusion check), and the plain
            # default (2x, neither flag set) adds nothing, exactly matching this
            # feature's original hardcoded-doubling filename output.
            rife_target_fps_val = getattr(args, "rife_target_fps", None)
            rife_multiplier_val = getattr(args, "rife_multiplier", None)
            if rife_target_fps_val is not None:
                rife_tag += f"{rife_target_fps_val:g}".replace(".", "p") + "fps"
            elif rife_multiplier_val is not None and rife_multiplier_val != 2:
                rife_tag += f"{rife_multiplier_val}x"
        else:
            rife_tag = ""

        # Only shown when the tag will actually be MEANINGFUL to apply: the setting is on,
        # this is a real video going to an .mkv container, and the resolved format is one
        # StereoMode can describe (see _resolve_stereo_mode_value) -- e.g. never appears for
        # RGB-D/Half RGB-D/Anaglyph/Debug Depth even if the flag is set, and never for a
        # non-mkv container. Mirrors every other tag's "only when non-default/non-no-op"
        # convention (depth_refine/rife_interpolate above).
        if (getattr(args, "stereo_mode_tag", False) and video
                and getattr(args, "video_extension", None) == ".mkv"
                and _resolve_stereo_mode_value(args) is not None):
            smtag = "_smtag"
        else:
            smtag = ""

        metadata = (f"_{args.depth_model}_{resolution}{tta}{daa}{args.method}_"
                    f"d{to_deciaml(args.divergence, 10, 2)}{fd}{bd}_{convergence_name}{to_deciaml(args.convergence, 10, 2)}"
                    f"{convergence_smoothing}_"
                    f"di{edge_dilation}_fs{args.foreground_scale}_fp{args.foreground_pop}{bp}_"
                    f"ipd{to_deciaml(args.ipd_offset, 1)}{ema}{drefine}{tstab}{dblend}"
                    f"{im_tag}{iof_tag}{imd_tag}{imw_tag}{spt_tag}{sw_tag}{sbd_tag}{psb_tag}{er_tag}{sharp_tag}{rife_tag}{smtag}{bitrate}")
    else:
        metadata = ""

    return basename + metadata + auto_detect_suffix + (args.video_extension if video else get_image_ext(args.format))


def _build_iw3_comment_metadata(args, video=True):
    """Builds the iw3_* embedded COMMENT metadata string, mirroring make_output_filename's
    own tags (same fields, same conditions) so renaming a file never loses the settings
    it was made with. Shared by every code path that writes a final video file
    (process_video_full, and process_config_video -- Depth Blend's Final Render) so
    they can never drift out of sync with each other. Returns None if there is nothing
    to record (should not happen in practice, since core settings are unconditional)."""
    inpaint_methods = {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}
    comment_parts = []

    comment_parts.append(f"iw3_depth_model={args.depth_model}")
    if args.resolution:
        comment_parts.append(f"iw3_resolution={args.resolution}")
    if args.tta:
        comment_parts.append("iw3_tta=1")
    if args.depth_aa:
        comment_parts.append("iw3_depth_aa=1")
    comment_parts.append(f"iw3_method={args.method}")
    comment_parts.append(f"iw3_divergence={args.divergence}")
    if getattr(args, "foreground_divergence", None) is not None:
        comment_parts.append(f"iw3_foreground_divergence={args.foreground_divergence}")
    if getattr(args, "background_divergence", None) is not None:
        comment_parts.append(f"iw3_background_divergence={args.background_divergence}")
    comment_parts.append(f"iw3_convergence={args.convergence}")
    if args.convergence_mode != "constant":
        comment_parts.append(f"iw3_convergence_mode={args.convergence_mode}")
        comment_parts.append(
            f"iw3_convergence_smoothing={getattr(args, 'convergence_smoothing', 0.9)}")
    if isinstance(args.edge_dilation, (list, tuple)):
        comment_parts.append(f"iw3_edge_dilation={'x'.join(str(v) for v in args.edge_dilation)}")
    else:
        comment_parts.append(f"iw3_edge_dilation={args.edge_dilation}")
    comment_parts.append(f"iw3_foreground_scale={args.foreground_scale}")
    comment_parts.append(f"iw3_foreground_pop={args.foreground_pop}")
    if getattr(args, "background_pop", 0.0) > 0:
        comment_parts.append(f"iw3_background_pop={args.background_pop}")
        if getattr(args, "background_pop_coverage", 0.15) != 0.15:
            comment_parts.append(f"iw3_background_pop_coverage={args.background_pop_coverage}")
    comment_parts.append(f"iw3_ipd_offset={args.ipd_offset}")
    if video:
        if args.video_codec == "libopenh264":
            comment_parts.append(f"iw3_bitrate={args.video_bitrate}")
        else:
            comment_parts.append(f"iw3_crf={args.crf}")

    if args.method in inpaint_methods:
        comment_parts.append(f"iw3_inpaint_model={args.inpaint_model or 'light_inpaint_v1'}")
        overlap = getattr(args, "inpaint_overlap_frames", None) or [3, 3]
        overlap = list(overlap) if isinstance(overlap, (list, tuple)) else [overlap, overlap]
        comment_parts.append(f"iw3_inpaint_overlap_frames={overlap[0]}x{overlap[-1]}")
        mask_inner = getattr(args, "mask_inner_dilation", 0) or 0
        mask_outer = getattr(args, "mask_outer_dilation", 0) or 0
        if mask_inner or mask_outer:
            comment_parts.append(f"iw3_mask_dilation={mask_inner}x{mask_outer}")
        max_w = getattr(args, "inpaint_max_width", None)
        if max_w:
            comment_parts.append(f"iw3_inpaint_max_width={int(max_w)}")
    if args.method == "forward_splat_fill":
        splat_temp = getattr(args, "splat_blend_temperature", SPLAT_BLEND_TEMPERATURE)
        splat_temp = SPLAT_BLEND_TEMPERATURE if splat_temp is None else splat_temp
        if splat_temp != SPLAT_BLEND_TEMPERATURE:
            comment_parts.append(f"iw3_splat_blend_temperature={splat_temp}")
    if getattr(args, "depth_blend", False):
        db_model = getattr(args, "depth_blend_model", None) or "VDA_L"
        db_strength = getattr(args, "depth_blend_strength", 1.0) or 1.0
        db_region = getattr(args, "depth_blend_region", None) or "detail"
        db_region_desc = db_region
        if db_region != "detail":
            db_percent = getattr(args, "depth_blend_region_percent", 25.0) or 25.0
            db_region_desc = f"{db_region}{int(db_percent)}"
        comment_parts.append(
            f"iw3_depth_blend_primary_model={args.depth_model} "
            f"iw3_depth_blend_model={db_model} "
            f"iw3_depth_blend_strength={db_strength} "
            f"iw3_depth_blend_region={db_region_desc}"
        )
        db_feather = int(getattr(args, "depth_blend_feather_blur", 0) or 0)
        if db_feather > 0:
            comment_parts.append(f"iw3_depth_blend_feather_blur={db_feather}")
        if getattr(args, "depth_blend_bilateral", False):
            comment_parts.append(
                f"iw3_depth_blend_bilateral_d={int(getattr(args, 'depth_blend_bilateral_d', 12) or 12)} "
                f"iw3_depth_blend_bilateral_sigma_color="
                f"{getattr(args, 'depth_blend_bilateral_sigma_color', 75.0) or 75.0} "
                f"iw3_depth_blend_bilateral_sigma_space="
                f"{getattr(args, 'depth_blend_bilateral_sigma_space', 75.0) or 75.0}"
            )
        if getattr(args, "depth_blend_clahe", False):
            comment_parts.append(
                f"iw3_depth_blend_clahe_clip={getattr(args, 'depth_blend_clahe_clip', 2.0) or 2.0} "
                f"iw3_depth_blend_clahe_tile={int(getattr(args, 'depth_blend_clahe_tile', 8) or 8)}"
            )
        if getattr(args, "depth_blend_align", False):
            comment_parts.append("iw3_depth_blend_align=1")
            db_align_decay = getattr(args, "depth_blend_align_decay", 0.9) or 0.9
            if db_align_decay != 0.9:
                comment_parts.append(f"iw3_depth_blend_align_decay={db_align_decay}")
        db_edge_supp = getattr(args, "depth_blend_edge_suppression", 0.5) or 0.5
        if db_edge_supp != 0.5:
            comment_parts.append(f"iw3_depth_blend_edge_suppression={db_edge_supp}")
        if getattr(args, "depth_blend_edge_hard_cutoff", False):
            comment_parts.append("iw3_depth_blend_edge_hard_cutoff=1")
    if getattr(args, "depth_refine", False):
        comment_parts.append("iw3_depth_refine=1")
        dr_strength = getattr(args, "depth_refine_strength", 1.0) or 1.0
        if dr_strength != 1.0:
            comment_parts.append(f"iw3_depth_refine_strength={dr_strength}")
    if getattr(args, "temporal_stabilize", False) and video:
        ts_strength = getattr(args, "temporal_stabilize_strength", 0.7) or 0.7
        comment_parts.append(f"iw3_temporal_stabilize_strength={ts_strength}")
        ts_max_shift = getattr(args, "temporal_stabilize_max_shift_velocity", None)
        if ts_max_shift is not None:
            comment_parts.append(f"iw3_temporal_stabilize_max_shift_velocity={ts_max_shift}")
        ts_flat_boost = getattr(args, "temporal_stabilize_flat_region_boost", 0.0) or 0.0
        if ts_flat_boost != 0.0:
            comment_parts.append(f"iw3_temporal_stabilize_flat_region_boost={ts_flat_boost}")
        ts_edge_protect = getattr(args, "temporal_stabilize_edge_protection", 0.0) or 0.0
        if ts_edge_protect != 0.0:
            comment_parts.append(f"iw3_temporal_stabilize_edge_protection={ts_edge_protect}")
    if args.ema_normalize and video:
        if _scene_auto_ema_active(args, args.ema_normalize, video):
            auto_model = getattr(args, "scene_batch_auto_ema_model", None) or "3DECKER VDA_L"
            comment_parts.append(f"iw3_scene_auto_ema=1 iw3_scene_auto_ema_model={auto_model}")
        else:
            ma = "1" if getattr(args, "ema_motion_adaptive", False) else "0"
            comment_parts.append(
                f"iw3_ema_decay={args.ema_decay} iw3_ema_buffer={args.ema_buffer} "
                f"iw3_ema_motion_adaptive={ma}"
            )
    stereo_w = getattr(args, "stereo_width", None)
    if stereo_w:
        comment_parts.append(f"iw3_stereo_width={int(stereo_w)}")
    if getattr(args, "scene_detect", False) and video:
        comment_parts.append("iw3_scene_detect=1")
    if getattr(args, "preserve_screen_border", False):
        comment_parts.append("iw3_preserve_screen_border=1")
    if getattr(args, "edge_repair_strength", 0.0):
        comment_parts.append(f"iw3_edge_repair_strength={args.edge_repair_strength}")
    if getattr(args, "sharpen", False):
        sharpen_strength = getattr(args, "sharpen_strength", None)
        if sharpen_strength is None:
            sharpen_strength = 0.5
        if sharpen_strength > 0.0:
            comment_parts.append(f"iw3_sharpen_strength={sharpen_strength}")
    if getattr(args, "rife_interpolate", False):
        rife_model_val = getattr(args, "rife_model", None) or "rife_425"
        comment_parts.append(f"iw3_rife_interpolate=1 iw3_rife_model={rife_model_val}")
        rife_target_fps_val = getattr(args, "rife_target_fps", None)
        rife_multiplier_val = getattr(args, "rife_multiplier", None)
        if rife_target_fps_val is not None:
            comment_parts.append(f"iw3_rife_target_fps={rife_target_fps_val}")
        elif rife_multiplier_val is not None and rife_multiplier_val != 2:
            comment_parts.append(f"iw3_rife_multiplier={rife_multiplier_val}")
    if (getattr(args, "stereo_mode_tag", False) and video
            and getattr(args, "video_extension", None) == ".mkv"
            and _resolve_stereo_mode_value(args) is not None):
        comment_parts.append(f"iw3_stereo_mode_tag={_resolve_stereo_mode_value(args)}")
    if getattr(args, "waifu2x_upscale", False):
        # Recorded here as provenance even though the upscale itself produces a
        # SEPARATE "<name>_w2x<ext>" file (see _run_waifu2x_upscale) rather than
        # modifying this file -- not added to make_output_filename's tag scheme for
        # the same reason: the derivative file's own "_w2x" suffix already marks it,
        # and duplicating that onto the original file's name would just be confusing
        # about which file is which.
        w2x_method = getattr(args, "waifu2x_method", None) or "noise_scale2x"
        w2x_noise = getattr(args, "waifu2x_noise_level", None)
        w2x_noise = 1 if w2x_noise is None else w2x_noise
        w2x_style = getattr(args, "waifu2x_style", None) or "photo"
        comment_parts.append(
            f"iw3_waifu2x_upscale_requested=1 iw3_waifu2x_method={w2x_method} "
            f"iw3_waifu2x_noise_level={int(w2x_noise)} iw3_waifu2x_style={w2x_style}"
        )
        w2x_target = getattr(args, "waifu2x_upscale_target", None) or "auto"
        if w2x_target in WAIFU2X_TARGET_PACKED_WIDTH:
            comment_parts.append(f"iw3_waifu2x_upscale_target={w2x_target}")

    return " ".join(comment_parts) if comment_parts else None


# Job-level stage names, shared with iw3/gui.py (imported from there, not
# re-typed) so the GUI's precomputed "Step X of N" stage list and the actual
# _notify_stage() calls below can never drift apart on spelling.
# ADR-052 amendment (2026-09-08): four more real stages a regular conversion can go
# through were added -- STAGE_SCENE_DETECT/STAGE_AUTOCROP/STAGE_HDR_EXTRACT/
# STAGE_AUDIO_EXTRACT -- so "Step X of N" reflects the true number of phases for
# whatever combination of features a given run has enabled, not just the 4 stages
# originally wired in. Listed here in the real order process_video_full/
# process_video_with_resume run them.
STAGE_SCENE_DETECT = "Scene Boundary Detection"
STAGE_AUTOCROP = "AutoCrop Analysis"
STAGE_HDR_EXTRACT = "HDR/DV RPU Extraction"
STAGE_AUDIO_EXTRACT = "Audio Extraction"
STAGE_DEPTH_STEREO = "Depth & Stereo Conversion"
STAGE_WAIFU2X_UPSCALE = "Upscaling with waifu2x"
STAGE_RIFE_INTERPOLATE = "RIFE Frame Interpolation"
STAGE_HDR_REINJECT = "HDR/Dolby Vision Reinjection"


def _notify_stage(args, name):
    """Posts a job-level stage-change notification, if something is listening
    (args.state["stage_fn"], wired only by iw3/gui.py -- see docs/ai/AI_DECISIONS.md
    for the progress-display ADR). No-op for CLI use and for any other caller that
    never set stage_fn.

    Exists because waifu2x upscaling, RIFE interpolation, and HDR/DV reinjection run
    as blocking subprocess.run(..., capture_output=True) calls (CS-SUBPROCESS-001)
    AFTER the tqdm-tracked depth/stereo encode has already finished -- unlike that
    encode (which reports live per-frame progress via tqdm_fn), these steps produced
    no signal the GUI could show at all, so a multi-minute RIFE pass looked identical
    to the job being finished or hung. This is purely a display signal: it computes
    nothing and changes no processing behavior, matching every other tqdm_fn call
    site in this module."""
    stage_fn = (getattr(args, "state", None) or {}).get("stage_fn")
    if stage_fn is not None:
        try:
            stage_fn(name)
        except Exception:
            pass


def _progress_title(basename, args):
    """Builds the text shown on a processing progress bar: the file name, which depth
    model is actually running right now, and -- when set (Dual-Pass Depth Blend sets this
    once per pass) -- which numbered step of a multi-step job this is. Without this, a
    multi-pass job like Depth Blend showed only the filename and frame count on the bar
    itself; which pass/model was active was only ever printed as a one-line banner that
    scrolls out of view the moment the bar starts updating in place."""
    model_name = getattr(args, "depth_model", None)
    step_label = (getattr(args, "state", None) or {}).get("progress_step_label")
    if step_label and model_name:
        return f"{basename} [{step_label}: {model_name}]"
    elif model_name:
        return f"{basename} [{model_name}]"
    else:
        return basename


def make_video_codec_option(args, input_path=None):
    if args.video_codec in {"libx264", "libx265", "hevc_nvenc", "h264_nvenc"}:
        options = {"preset": args.preset, "crf": str(args.crf)}

        if args.tune:
            options["tune"] = ",".join(set(args.tune))

        if args.profile_level:
            options["level"] = str(int(float(args.profile_level) * 10))

        if args.video_codec == "libx265":
            x265_params = ["log-level=warning", "high-tier=enabled"]
            if args.profile_level:
                x265_params.append(f"level-idc={int(float(args.profile_level) * 10)}")

            if (input_path is not None and args.colorspace in {"auto", "bt2020-tv", "bt2020-pq-tv"}):
                hdr_metadata = get_hdr_metadata(input_path)
                x265_params += hdr_metadata.to_x265_params()

            options["x265-params"] = ":".join(x265_params)
            # print(options)
        elif args.video_codec == "libx264":
            # TODO:
            # if args.tb or args.half_tb:
            #    options["x264-params"] = "frame-packing=4"
            if args.half_sbs:
                options["x264-params"] = "frame-packing=3"
        elif args.video_codec in {"hevc_nvenc", "h264_nvenc"}:
            options["rc"] = "constqp"
            options["qp"] = str(args.crf)
            if torch.cuda.is_available() and args.gpu[0] >= 0:
                options["gpu"] = str(args.gpu[0])
    elif args.video_codec in {"h264_qsv", "hevc_qsv"}:
        options = {
            "preset": args.preset,
            "crf": str(args.crf),
            "global_quality": str(args.crf)
        }
    elif args.video_codec == "libopenh264":
        # NOTE: It seems libopenh264 does not support most options.
        options = {"b": args.video_bitrate}
    else:
        options = {}

    return options


def get_image_ext(format):
    if format == "png":
        return ".png"
    elif format == "webp":
        return ".webp"
    elif format == "jpeg":
        return ".jpg"
    else:
        raise NotImplementedError(format)


def save_image(im, output_filename, format="png", png_info=None):
    if format == "png":
        options = {
            "compress_level": 6,
            "pnginfo": png_info,
        }
    elif format == "webp":
        options = {
            "quality": 95,
            "method": 4,
            "lossless": True
        }
    elif format == "jpeg":
        options = {
            "quality": 95,
            "subsampling": "4:2:0",
        }
    else:
        raise NotImplementedError(format)

    im.save(output_filename, format=format, **options)


def preprocess_image(x, args):
    if args.rotate_left:
        x = torch.rot90(x, 1, (-2, -1))
    elif args.rotate_right:
        x = torch.rot90(x, 3, (-2, -1))

    h, w = x.shape[-2:]
    new_w, new_h = w, h
    if args.max_output_height is not None and new_h > args.max_output_height:
        new_w = int(args.max_output_height / new_h * new_w)
        new_h = args.max_output_height
        # only apply max height
    if new_w != w or new_h != h:
        new_h -= new_h % 2
        new_w -= new_w % 2
        if x.ndim == 3:
            x = F.interpolate(x.unsqueeze(0), (new_h, new_w),
                              mode="bicubic", antialias=True, align_corners=True).squeeze(0)
        elif x.ndim == 4:
            x = F.interpolate(x, (new_h, new_w),
                              mode="bicubic", antialias=True, align_corners=True)

        x = torch.clamp(x, 0, 1)

    return x


def add_preprocess_vf(vf_org, args) -> str:
    vf = []

    # NOTE: Denoise is handled as a one-time ffmpeg pre-pass in _denoise_preprocess(), not
    # as a per-frame filter here -- PyAV's bundled libavfilter doesn't have hqdn3d, and the
    # GPU tensor-frame decode path (used whenever HWAccel=cuda) only supports a small fixed
    # set of filters (scale/crop/bob/transpose/lut3d/setparams) that doesn't include any
    # denoiser at all, so no per-frame filter choice here works in every decode mode.

    # Rotation
    if getattr(args, "rotate_left", False):
        vf.append("transpose=2")
    elif getattr(args, "rotate_right", False):
        vf.append("transpose=1")

    # Max height scaling and even-dimension fix
    max_h = getattr(args, "max_output_height", None)
    if max_h is not None:
        target_h = (max_h // 2) * 2
        vf.append(f"scale=if(gt(ih\\,{max_h})\\,-2\\,iw):if(gt(ih\\,{max_h})\\,{target_h}\\,ih):flags=bicubic")

    if vf:
        if vf_org:
            vf_added = vf_org + "," + ",".join(vf)
        else:
            vf_added = ",".join(vf)
        return vf_added
    else:
        return vf_org


def apply_divergence(depth, im, args, side_model, reset_pts=None):
    batch = True
    if depth.ndim != 4:
        # CHW
        depth = depth.unsqueeze(0)
        im = im.unsqueeze(0)
        batch = False
    else:
        # BCHW
        pass

    if args.state["convergence_model"] is not None:
        convergence = args.state["convergence_model"](im, depth, reset_pts=reset_pts)
        mapper_fn = get_mapper(args.mapper)
        convergence = mapper_fn(convergence)
        depth = mapper_fn(depth)
    else:
        convergence = args.convergence
        depth = get_mapper(args.mapper)(depth)

    # Depth pop effects
    foreground_pop = getattr(args, "foreground_pop", 0.0)
    if foreground_pop > 0:
        depth = DE.apply_foreground_pop(depth, foreground_pop)
    foreground_divergence = getattr(args, "foreground_divergence", None)
    if foreground_divergence is not None:
        depth = DE.apply_foreground_divergence(depth, convergence, args.divergence, foreground_divergence)
    background_pop = getattr(args, "background_pop", 0.0)
    if background_pop > 0:
        background_pop_coverage = getattr(args, "background_pop_coverage", 0.15)
        depth = DE.apply_background_pop(depth, background_pop, threshold_percentile=background_pop_coverage)
    background_divergence = getattr(args, "background_divergence", None)
    if background_divergence is not None:
        depth = DE.apply_background_divergence(depth, convergence, args.divergence, background_divergence)

    if args.method == "NULL":
        left_eye, right_eye = im.clone(), im.clone()
        if not batch:
            left_eye = left_eye.squeeze(0)
            right_eye = right_eye.squeeze(0)
    elif args.method in {"grid_sample", "backward"}:
        left_eye, right_eye = apply_divergence_grid_sample(
            im, depth,
            args.divergence, convergence=convergence,
            synthetic_view=args.synthetic_view)
    elif args.method == "monobw":
        left_eye, right_eye = apply_divergence_monobw(
            side_model,
            im,
            depth,
            divergence=args.divergence,
            convergence=convergence,
            synthetic_view=args.synthetic_view,
            preserve_screen_border=args.preserve_screen_border,
        )
    elif args.method in {"forward", "forward_fill", "forward_splat_fill"}:
        left_eye, right_eye = apply_divergence_forward_warp(
            im, depth,
            args.divergence, convergence=convergence,
            method=args.method, synthetic_view=args.synthetic_view, width_base=False,
            splat_blend_temperature=getattr(args, "splat_blend_temperature", SPLAT_BLEND_TEMPERATURE) or SPLAT_BLEND_TEMPERATURE)
    elif args.method in {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}:
        left_eyes = []
        right_eyes = []
        reset_pts = reset_pts if reset_pts is not None else [False] * depth.shape[0]
        for i in range(depth.shape[0]):
            conv_i = convergence[i:i + 1] if torch.is_tensor(convergence) else convergence
            left_eye, right_eye = side_model.infer(
                im[i:i + 1], depth[i:i + 1],
                divergence=args.divergence,
                convergence=conv_i,
                preserve_screen_border=args.preserve_screen_border,
                synthetic_view=args.synthetic_view,
                inner_dilation=args.mask_inner_dilation,
                outer_dilation=args.mask_outer_dilation,
                max_width=args.inpaint_max_width,
                enable_amp=not args.disable_amp,
            )
            if left_eye is not None:
                left_eyes.append(left_eye)
                right_eyes.append(right_eye)
            if reset_pts[i]:
                left_eye, right_eye = side_model.flush(enable_amp=not args.disable_amp)
                if left_eye is not None:
                    left_eyes.append(left_eye)
                    right_eyes.append(right_eye)

        if left_eyes:
            if len(left_eyes) == 1:
                left_eye = left_eyes[0]
                right_eye = right_eyes[0]
            else:
                left_eye = torch.cat(left_eyes, dim=0)
                right_eye = torch.cat(right_eyes, dim=0)
        else:
            left_eye = right_eye = None
    else:
        # row_flow*, mlbw*
        if args.stereo_width is not None:
            # NOTE: use src aspect ratio instead of depth aspect ratio
            H, W = im.shape[2:]
            stereo_width = min(W, args.stereo_width)
            if depth.shape[3] != stereo_width:
                new_w = stereo_width
                new_h = int(H * (stereo_width / W))
                depth = F.interpolate(depth, size=(new_h, new_w),
                                      mode="bilinear", align_corners=True, antialias=True)
                depth = torch.clamp(depth, 0, 1)
        left_eye, right_eye = apply_divergence_nn_LR(
            side_model, im, depth,
            args.divergence, convergence, args.warp_steps,
            synthetic_view=args.synthetic_view,
            preserve_screen_border=args.preserve_screen_border,
            enable_amp=not args.disable_amp,
        )

    edge_repair_strength = getattr(args, "edge_repair_strength", 0.0)
    if edge_repair_strength > 0.0:
        left_eye, right_eye = DE.repair_stereo_edges(
            left_eye, right_eye, depth, strength=edge_repair_strength)

    # Runs AFTER Edge Repair, deliberately (see DE.apply_sharpen's own docstring):
    # sharpening the frame BEFORE Edge Repair cleans up hairline fringing/ghosting
    # would exaggerate exactly the artifact Edge Repair is about to smooth away,
    # working against it instead of alongside it.
    if getattr(args, "sharpen", False):
        sharpen_strength = getattr(args, "sharpen_strength", None)
        if sharpen_strength is None:
            sharpen_strength = 0.5
        if sharpen_strength > 0.0:
            left_eye, right_eye = DE.apply_sharpen(
                left_eye, right_eye, strength=sharpen_strength)

    if not batch:
        if left_eye is not None:
            left_eye = left_eye.squeeze(0)
        if right_eye is not None:
            right_eye = right_eye.squeeze(0)

    return left_eye, right_eye


def postprocess_padding(left_eye, right_eye, pad, pad_mode):
    assert pad_mode in {"tblr", "tb", "lr", "16:9", "top"}
    if pad_mode in {"tblr", "tb", "lr"}:
        pad_h = pad_w = 0
        if "tb" in pad_mode:
            pad_h = round(left_eye.shape[1] * pad) // 2
        if "lr" in pad_mode:
            pad_w = round(left_eye.shape[2] * pad) // 2
        left_eye = TF.pad(left_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
        right_eye = TF.pad(right_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
    elif pad_mode == "top":
        pad_top = round(left_eye.shape[1] * pad)
        left_eye = TF.pad(left_eye, (0, pad_top, 0, 0), padding_mode="constant")
        right_eye = TF.pad(right_eye, (0, pad_top, 0, 0), padding_mode="constant")
    elif pad_mode == "16:9":
        # fit to 16:9
        # pad size is ignored
        eps = 1e-3
        target_ratio = 16 / 9
        height, width = left_eye.shape[1:]
        current_ratio = width / height
        if abs(target_ratio - current_ratio) > eps:
            pad_h = pad_w = 0
            if current_ratio > target_ratio:
                # pad top-bottom
                target_height = round(width / target_ratio)
                pad_h = (target_height - height) // 2
            else:
                # pad left-right
                target_width = round(height * target_ratio)
                pad_w = (target_width - width) // 2
            left_eye = TF.pad(left_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
            right_eye = TF.pad(right_eye, (pad_w, pad_h, pad_w, pad_h), padding_mode="constant")
    return left_eye, right_eye


def postprocess_image(left_eye, right_eye, args):
    # CHW
    ipd_pad = int(abs(args.ipd_offset) * 0.01 * max(left_eye.shape[-2:]))
    ipd_pad -= ipd_pad % 2
    if ipd_pad > 0 and not (args.rgbd or args.half_rgbd):
        pad_o, pad_i = (ipd_pad * 2, ipd_pad) if args.ipd_offset > 0 else (ipd_pad, ipd_pad * 2)
        left_eye = TF.pad(left_eye, (pad_o, 0, pad_i, 0), padding_mode="constant")
        right_eye = TF.pad(right_eye, (pad_i, 0, pad_o, 0), padding_mode="constant")

    if args.pad is not None or args.pad_mode == "16:9":
        left_eye, right_eye = postprocess_padding(left_eye, right_eye, pad=args.pad, pad_mode=args.pad_mode)
    if args.vr180:
        left_eye = equirectangular_projection(left_eye, device=left_eye.device)
        right_eye = equirectangular_projection(right_eye, device=right_eye.device)
    elif args.half_sbs or args.half_rgbd:
        left_eye = TF.resize(left_eye, (left_eye.shape[1], left_eye.shape[2] // 2),
                             interpolation=InterpolationMode.BICUBIC, antialias=True)
        right_eye = TF.resize(right_eye, (right_eye.shape[1], right_eye.shape[2] // 2),
                              interpolation=InterpolationMode.BICUBIC, antialias=True)
    elif args.half_tb:
        left_eye = TF.resize(left_eye, (left_eye.shape[1] // 2, left_eye.shape[2]),
                             interpolation=InterpolationMode.BICUBIC, antialias=True)
        right_eye = TF.resize(right_eye, (right_eye.shape[1] // 2, right_eye.shape[2]),
                              interpolation=InterpolationMode.BICUBIC, antialias=True)

    if args.anaglyph is not None:
        # Anaglyph
        sbs = apply_anaglyph_redcyan(left_eye, right_eye, args.anaglyph)
    elif args.tb or args.half_tb:
        # TopBottom
        sbs = torch.cat([left_eye, right_eye], dim=1)
        sbs = torch.clamp(sbs, 0., 1.)
    elif args.cross_eyed:
        # Reverse SideBySide
        sbs = torch.cat([right_eye, left_eye], dim=2)
        sbs = torch.clamp(sbs, 0., 1.)
    else:
        # SideBySide or RGBD
        sbs = torch.cat([left_eye, right_eye], dim=2)
        sbs = torch.clamp(sbs, 0., 1.)

    h, w = sbs.shape[1:]
    new_w, new_h = w, h
    if args.max_output_height is not None and new_h > args.max_output_height:
        if args.keep_aspect_ratio:
            new_w = int(args.max_output_height / new_h * new_w)
        new_h = args.max_output_height
    if args.max_output_width is not None and new_w > args.max_output_width:
        if args.keep_aspect_ratio:
            new_h = int(args.max_output_width / new_w * new_h)
        new_w = args.max_output_width
    if new_w != w or new_h != h:
        new_h -= new_h % 2
        new_w -= new_w % 2
        sbs = TF.resize(sbs, (new_h, new_w),
                        interpolation=InterpolationMode.BICUBIC, antialias=True)
        sbs = torch.clamp(sbs, 0, 1)
    return sbs


def debug_depth_image(depth, args):
    depth = depth.float()
    mean_depth, std_depth = depth.mean().item(), depth.std().item()
    depth2 = get_mapper(args.mapper)(depth)
    out = torch.cat([depth, depth2], dim=2).cpu()
    out = out.repeat((3, 1, 1))
    # gc = ImageDraw.Draw(out)
    # gc.text((16, 16), (f"min={round(float(depth_min), 4)}\n"
    #                    f"max={round(float(depth_max), 4)}\n"
    #                    f"mean={round(float(mean_depth), 4)}\n"
    #                    f"std={round(float(std_depth), 4)}"), "gray")

    return out


def process_image(x, args, depth_model, side_model, skip_autocrop=None, autocrop_uncrop=False):
    assert depth_model.get_ema_buffer_size() == 1

    if args.autocrop is None or skip_autocrop:
        autocrop = AutoCropDummy()
    else:
        autocrop = AutoCrop.from_image(x, mode=args.autocrop, uncrop_enabled=autocrop_uncrop)

    with torch.inference_mode():
        x = preprocess_image(x, args)
        x = autocrop.crop(x)
        depth = depth_model.infer(x, tta=args.tta, low_vram=args.low_vram,
                                  enable_amp=not args.disable_amp,
                                  edge_dilation=args.edge_dilation,
                                  depth_aa=args.depth_aa)
        depth = depth_model.minmax_normalize_chw(depth, rgb=x)

        if args.debug_depth:
            return debug_depth_image(depth, args)
        elif args.rgbd or args.half_rgbd:
            left_eye, right_eye = apply_rgbd(x, depth, mapper=args.mapper)
            left_eye = autocrop.uncrop(left_eye)
            right_eye = autocrop.uncrop(right_eye)
            sbs = postprocess_image(left_eye, right_eye, args)
            return sbs
        else:
            while True:
                left_eye, right_eye = apply_divergence(depth, x, args, side_model)
                if left_eye is not None:
                    break
            if left_eye.ndim == 4:
                # NOTE: side_model is video inpaint model.
                #       This may be called from test_callback
                assert 0, "No longer reaching this block"
                left_eye = autocrop.uncrop(left_eye[0])
                right_eye = autocrop.uncrop(right_eye[0])
                sbs = postprocess_image(left_eye, right_eye, args)
            else:
                left_eye = autocrop.uncrop(left_eye)
                right_eye = autocrop.uncrop(right_eye)
                sbs = postprocess_image(left_eye, right_eye, args)
            return sbs


def process_images(files, output_dir, args, depth_model, side_model, title=None):
    # disable ema minmax for each process
    depth_model.disable_ema()
    if side_model is not None and hasattr(side_model, "set_mode"):
        side_model.set_mode("image")
        side_model.reset()
    if args.state["convergence_model"] is not None:
        args.state["convergence_model"].reset(enable_ema=False)

    os.makedirs(output_dir, exist_ok=True)

    if args.resume:
        # skip existing output files
        remaining_files = []
        existing_files = []
        for fn in files:
            output_filename = path.join(
                output_dir,
                make_output_filename(path.basename(fn), args, video=False))
            if not path.exists(output_filename):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)
        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        files = remaining_files

    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb", "exif_transpose": not args.disable_exif_transpose})
    futures = []
    tqdm_fn = args.state["tqdm_fn"] or tqdm
    pbar = tqdm_fn(ncols=80, total=len(files), desc=title)
    stop_event = args.state["stop_event"]
    suspend_event = args.state["suspend_event"]

    max_workers = max(args.max_workers, 8)
    with PoolExecutor(max_workers=max_workers) as pool:
        for im, meta in loader:
            filename = meta["filename"]
            output_filename = path.join(
                output_dir,
                make_output_filename(filename, args, video=False))
            if im is None:
                pbar.update(1)
                continue
            im = TF.to_tensor(im).to(args.state["device"])
            output = process_image(im, args, depth_model, side_model)
            output = to_pil_image(output)
            f = pool.submit(save_image, output, output_filename, format=args.format)
            #  f.result() # for debug
            futures.append(f)
            pbar.update(1)
            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break
            if len(futures) > IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures = []
        for f in futures:
            f.result()
    pbar.close()


# video callbacks

def bind_single_frame_callback(depth_model, side_model, segment_pts, args, scene_ema_settings=None,
                               scene_ema_report=None):
    src_queue = []
    frame_cpu_offload = depth_model.get_ema_buffer_size() > 1

    def _postprocess(depths, flush):
        for depth in depths:
            x, pts = src_queue.pop(0)
            reset_pts = [pts in segment_pts]
            if isinstance(x, VU.OffloadFrame):
                x = x.load(device=args.state["device"])
            if args.debug_depth:
                out = debug_depth_image(depth, args)
            elif args.rgbd or args.half_rgbd:
                left_eye, right_eye = apply_rgbd(x, depth, mapper=args.mapper)
                out = postprocess_image(left_eye, right_eye, args)
            else:
                left_eye, right_eye = apply_divergence(depth, x, args, side_model, reset_pts=reset_pts)
                if left_eye is not None:
                    if left_eye.ndim == 3:
                        out = postprocess_image(left_eye, right_eye, args)
                    else:
                        out = [postprocess_image(left, right, args) for left, right in zip(left_eye, right_eye)]
                else:
                    out = None

            if not isinstance(out, list):
                if out is not None:
                    out = [out]
                else:
                    out = []

            if pts in segment_pts and args.debug_depth:
                for o in out:
                    # debug red line
                    o[0, 0:8, :] = 1.0

            for o in out:
                yield o

        if flush and hasattr(side_model, "flush"):
            left_eye, right_eye = side_model.flush(enable_amp=not args.disable_amp)
            if left_eye is not None:
                for left, right in zip(left_eye, right_eye):
                    out = postprocess_image(left, right, args)
                    yield out

    @torch.inference_mode()
    def _frame_callback(frame):
        if frame is None:
            # flush
            yield from _postprocess(depth_model.flush_minmax_normalize(), flush=True)
            return

        pix_dtype = VU.get_source_dtype(frame)
        x = VU.to_tensor(frame, device=args.state["device"])
        if frame_cpu_offload:
            # cpu buffer
            src_queue.append((VU.OffloadFrame(x, dtype=pix_dtype), frame.pts))
        else:
            # gpu buffer
            src_queue.append((x, frame.pts))

        depth = depth_model.infer(x, tta=args.tta, low_vram=args.low_vram,
                                  enable_amp=not args.disable_amp,
                                  edge_dilation=args.edge_dilation,
                                  depth_aa=args.depth_aa)
        depth = depth_model.minmax_normalize_chw(depth, rgb=x)
        depths = [depth] if depth is not None else []
        flush = frame.pts in segment_pts
        if flush:
            depths += depth_model.flush_minmax_normalize()
            depth_model.reset_state()
            # --scene-batch-auto-ema on the regular path (see compute_scene_ema_schedule):
            # the scaler was just cleared above -- re-arm it with the UPCOMING scene's
            # own Buffer/Decay instead of leaving the fixed --ema-decay/--ema-buffer in
            # place for the rest of the run. No-op (scene_ema_settings stays empty) unless
            # the feature is actually enabled.
            if scene_ema_settings:
                update = scene_ema_settings.get(frame.pts)
                if update is not None:
                    depth_model.enable_ema(decay=update[1], buffer_size=update[0],
                                           motion_adaptive=getattr(args, "ema_motion_adaptive", False))
                    if scene_ema_report:
                        row = scene_ema_report.get(frame.pts)
                        if row is not None:
                            _log_scene_ema_row(row)

        yield from _postprocess(depths, flush=flush)

    return _frame_callback


def bind_batch_frame_callback(depth_model, side_model, segment_pts, args, scene_ema_settings=None,
                              scene_ema_report=None):
    depth_lock = threading.RLock()
    sbs_lock = threading.RLock()
    enqueue_ticket_lock = TicketLock()
    dequeue_ticket_lock = TicketLock()
    streams = threading.local()
    src_queue = []
    frame_cpu_offload = depth_model.get_ema_buffer_size() > 1
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)

    def _postprocess(depth_batch, reset_ema, ema_updates, dequeue_ticket_id, flush, device):
        # Reorder threads
        with dequeue_ticket_lock(dequeue_ticket_id):
            with depth_lock:
                if flush:
                    depth_list = depth_model.flush_minmax_normalize()
                else:
                    depth_list = depth_model.minmax_normalize(depth_batch, reset_ema=reset_ema,
                                                               ema_updates=ema_updates)

            for depths in chunks(depth_list, args.batch_size):
                if isinstance(depths, list):
                    depths = torch.stack([depth.to(device) for depth in depths])
                else:
                    depths = depths.to(device)
                if frame_cpu_offload:
                    x_srcs = []
                    pts = []
                    for _ in range(len(depths)):
                        x_src, t = src_queue.pop(0)
                        x_srcs.append(x_src.load(device=device))
                        pts.append(t)
                    x_srcs = torch.stack(x_srcs)
                else:
                    x_srcs, pts = src_queue.pop(0)
                reset_pts = [t in segment_pts for t in pts]

                with sbs_lock:  # TODO: unclear whether this is actually needed
                    if args.rgbd or args.half_rgbd:
                        left_eyes, right_eyes = apply_rgbd(x_srcs, depths, mapper=args.mapper)
                    else:
                        if args.method in {"forward_fill", "forward", "forward_splat_fill"}:
                            # ordered_index_copy() flips the *global* torch.use_deterministic_algorithms
                            # flag, so no other thread may run CUDA depth inference while this executes.
                            # depth_lock alone is sufficient for that: _batch_infer() already holds
                            # depth_lock for its entire depth_model.infer() call, and this whole
                            # _postprocess() call is already the only one running at a time (it holds
                            # dequeue_ticket_lock's ticket for its full duration, from the top of this
                            # function). Do NOT also raw-acquire enqueue_ticket_lock/dequeue_ticket_lock
                            # here: _batch_infer() acquires enqueue_ticket_lock THEN dequeue_ticket_lock
                            # (to mint a dequeue ticket while holding its enqueue turn), while this
                            # function already holds dequeue_ticket_lock from entry -- grabbing
                            # enqueue_ticket_lock here too is the reverse order and is a real deadlock
                            # (confirmed via py-spy: see docs/ai/AI_DECISIONS.md ADR-067).
                            with depth_lock:
                                left_eyes, right_eyes = apply_divergence(depths, x_srcs, args, side_model, reset_pts=reset_pts)
                        else:
                            left_eyes, right_eyes = apply_divergence(depths, x_srcs, args, side_model, reset_pts=reset_pts)

                for i in range(left_eyes.shape[0]):
                    yield postprocess_image(left_eyes[i], right_eyes[i], args)

    def _batch_infer(x, pts, flush, enqueue_ticket_id):
        # Reorder threads
        with enqueue_ticket_lock(enqueue_ticket_id):
            dequeue_ticket_id = dequeue_ticket_lock.new_ticket()
            if not flush:
                if frame_cpu_offload:
                    pix_dtype = torch.uint16 if use_16bit else torch.uint8
                    for i in range(len(pts)):
                        src_queue.append((VU.OffloadFrame(x[i], dtype=pix_dtype), pts[i]))
                else:
                    src_queue.append((x, pts))

        if flush:
            return None, dequeue_ticket_id
        else:
            with depth_lock:
                depth_batch = depth_model.infer(x, tta=args.tta, low_vram=args.low_vram,
                                                enable_amp=not args.disable_amp,
                                                edge_dilation=args.edge_dilation,
                                                depth_aa=args.depth_aa)
            return depth_batch, dequeue_ticket_id

    @torch.inference_mode()
    def _cuda_stream_wrapper(preprocess_args):
        x, pts, flush, enqueue_ticket_id = preprocess_args
        if flush:
            device = args.state["device"]
            reset_ema = None
            ema_updates = None
            depth_batch, dequeue_ticket_id = _batch_infer(
                None, None, flush=flush, enqueue_ticket_id=enqueue_ticket_id)
            # Return a generator directly to avoid out-of-memory errors during flush.
            # Processing is performed on the main thread.
            return _postprocess(
                depth_batch, reset_ema, ema_updates,
                dequeue_ticket_id=dequeue_ticket_id,
                flush=flush,
                device=device
            )
        else:
            device = x.device
            reset_ema = [t in segment_pts for t in pts]
            # --scene-batch-auto-ema on the regular path: pairs each True reset flag
            # above with the UPCOMING scene's own (ema_buffer, ema_decay), looked up
            # once upfront in process_video_full (compute_scene_ema_schedule) -- None
            # for every index when the feature is off, an exact no-op downstream.
            ema_updates = ([scene_ema_settings.get(t) if r else None for t, r in zip(pts, reset_ema)]
                          if scene_ema_settings else None)
            if scene_ema_report:
                for t, r in zip(pts, reset_ema):
                    if r:
                        row = scene_ema_report.get(t)
                        if row is not None:
                            _log_scene_ema_row(row)
            if args.cuda_stream and device_is_cuda(x.device):
                device_name = str(device)
                if not hasattr(streams, device_name):
                    setattr(streams, device_name, torch.cuda.Stream(device=x.device))
                stream = getattr(streams, device_name)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.device(x.device), torch.cuda.stream(stream):
                    depth_batch, dequeue_ticket_id = _batch_infer(
                        x, pts, flush=flush, enqueue_ticket_id=enqueue_ticket_id)
                    stream.synchronize()
            else:
                depth_batch, dequeue_ticket_id = _batch_infer(
                    x, pts, flush=flush, enqueue_ticket_id=enqueue_ticket_id)

            results = _postprocess(
                depth_batch, reset_ema, ema_updates,
                dequeue_ticket_id=dequeue_ticket_id,
                flush=flush,
                device=device
            )
            # Run the generator in the worker thread and return the result.
            return [frame for frame in results]

    def _preprocess(x, pts, flush):
        enqueue_ticket_id = enqueue_ticket_lock.new_ticket()
        return (x, pts, flush, enqueue_ticket_id)

    return _cuda_stream_wrapper, _preprocess


def bind_vda_frame_callback(depth_model, side_model, segment_pts, args, scene_ema_settings=None,
                            scene_ema_report=None):
    src_queue = []
    batch_queue = []
    pts_queue = []
    pix_dtype = None
    pix_max = None

    depth_model.reset()

    def _postprocess(depth_list, flush=False):
        if args.debug_depth:
            for depth in depth_list:
                out = debug_depth_image(depth, args)
                _, pts = src_queue.pop(0)
                if pts in segment_pts:
                    out[0, 0:8, :] = 1.0
                yield out
        else:
            for depths in chunks(depth_list, args.batch_size):
                depths = torch.stack(depths)
                x_pts = [src_queue.pop(0) for _ in range(len(depths))]
                reset_pts = [pts in segment_pts for _, pts in x_pts]
                x_srcs = torch.stack([x.load(device=args.state["device"]) for x, _ in x_pts])
                if args.rgbd or args.half_rgbd:
                    left_eyes, right_eyes = apply_rgbd(x_srcs, depths, mapper=args.mapper)
                else:
                    left_eyes, right_eyes = apply_divergence(depths, x_srcs, args, side_model, reset_pts=reset_pts)
                if left_eyes is not None:
                    for i in range(left_eyes.shape[0]):
                        yield postprocess_image(left_eyes[i], right_eyes[i], args)

        if flush and hasattr(side_model, "flush"):
            left_eyes, right_eyes = side_model.flush(enable_amp=not args.disable_amp)
            if left_eyes is not None:
                for left_eye, right_eye in zip(left_eyes, right_eyes):
                    yield postprocess_image(left_eye, right_eye, args)

    def _batch_infer():
        assert pix_max is not None and pix_dtype is not None
        for i in range(len(batch_queue)):
            src_queue.append((VU.OffloadFrame(batch_queue[i], dtype=pix_dtype), pts_queue[i]))

        if scene_ema_report:
            for t in pts_queue:
                if t in segment_pts:
                    row = scene_ema_report.get(t)
                    if row is not None:
                        _log_scene_ema_row(row)

        x = torch.stack(batch_queue)
        depth_list = depth_model.infer_with_normalize(
            x, pts_queue, segment_pts,
            enable_amp=not args.disable_amp,
            edge_dilation=args.edge_dilation,
            depth_aa=args.depth_aa,
            tta=args.tta,
            ema_updates=scene_ema_settings or None)

        pts_queue.clear()
        batch_queue.clear()

        yield from _postprocess(depth_list)

    @torch.inference_mode()
    def frame_callback(frame):
        nonlocal pix_dtype, pix_max

        if frame is None:
            # flush
            if batch_queue:
                yield from _batch_infer()
            depth_list = depth_model.flush_with_normalize(
                enable_amp=not args.disable_amp,
                edge_dilation=args.edge_dilation,
                depth_aa=args.depth_aa)
            yield from _postprocess(depth_list, flush=True)
            return

        if pix_dtype is None:
            pix_dtype = VU.get_source_dtype(frame)
            pix_max = torch.iinfo(pix_dtype).max

        x = VU.to_tensor(frame, device=args.state["device"])
        batch_queue.append(x)
        pts_queue.append(frame.pts)

        if len(batch_queue) == args.batch_size:
            yield from _batch_infer()

    return frame_callback


def try_compile_context(side_model, enabled):
    if enabled and side_model is not None and hasattr(side_model, "compile_context"):
        return side_model.compile_context()
    else:
        return contextlib.nullcontext()


def try_load_scene_cache(video_path, args, max_fps=None):
    if max_fps is None:
        max_fps = args.max_fps
    if args.scene_cache_file:
        segment_pts = SceneBoundaryCache.try_load_cache_with_filename(
            args.scene_cache_file,
            video_path,
            max_fps=max_fps,
            start_time=args.start_time,
            end_time=args.end_time
        )
    else:
        segment_pts = SceneBoundaryCache.try_load_cache(
            video_path,
            max_fps=max_fps,
            start_time=args.start_time,
            end_time=args.end_time,
            cache_dir=args.scene_cache_dir,
        )
    return segment_pts


def save_scene_cache(video_path, segment_pts, args, start_time=None, end_time=None, max_fps=None):
    if start_time is None:
        start_time = args.start_time
    if end_time is None:
        end_time = args.end_time
    if max_fps is None:
        max_fps = args.max_fps
    if args.scene_cache_file:
        SceneBoundaryCache.save_cache_with_filename(
            args.scene_cache_file,
            video_path,
            segment_pts,
            max_fps=max_fps,
            start_time=start_time,
            end_time=end_time
        )
    else:
        SceneBoundaryCache.save_cache(
            video_path,
            segment_pts,
            max_fps=max_fps,
            start_time=start_time,
            end_time=end_time,
            cache_dir=args.scene_cache_dir,
        )


def should_save_scene_cache(disable_scene_cache, is_preview, stop_event):
    """
    A scene-boundary scan's result must only ever be written to the cache if BOTH:
      1. it's a real (non-preview-quality) scan -- unchanged from the original
         "don't let a fast/low-fps preview scan overwrite the real cache with
         lower-quality results" guard, and
      2. it actually finished scanning the full requested start_time/end_time range.

    (2) matters because SBD.detect_boundary() returns an incomplete result (in
    practice, an empty set -- see nunif.utils.shot_boundary_detection.detect_boundary,
    which returns set() the moment it notices `stop_event` was set, discarding
    whatever partial results it had accumulated) the instant a scan is interrupted
    partway through, e.g. the user clicking Cancel/Stop mid-scan. Without this check,
    the caller would still save that incomplete/empty result to the cache tagged with
    the FULL originally-requested start_time/end_time -- so a later real request
    covering that same range would pass is_within_range() and silently trust the
    incomplete data (e.g. "zero scene cuts in this whole range") as if the scan had
    actually completed. Called BEFORE any stop_event-triggered early return, so the
    caller must check `stop_event` again afterward to actually stop.
    """
    if disable_scene_cache or is_preview:
        return False
    if stop_event is not None and stop_event.is_set():
        return False
    return True


def get_cached_scene_range(video_path, args, max_fps=None):
    if max_fps is None:
        max_fps = args.max_fps
    if args.scene_cache_file:
        return SceneBoundaryCache.get_cached_range_with_filename(args.scene_cache_file)
    else:
        return SceneBoundaryCache.get_cached_range(
            video_path, max_fps=max_fps, cache_dir=args.scene_cache_dir)


def widen_scene_scan_range(video_path, args, max_fps=None):
    """
    If a scene cache already exists for this video but doesn't cover the requested
    start/end range, widen the scan to the union of the existing cached range and the
    requested range, so the cache only ever grows and previously detected scene
    boundaries (from testing a different clip of the same movie) are never discarded.
    """
    scan_start_time, scan_end_time = args.start_time, args.end_time
    cached_range = get_cached_scene_range(video_path, args, max_fps=max_fps)
    if cached_range is not None:
        cached_start, cached_end = cached_range
        query_start = SceneBoundaryCache.time_to_sec(args.start_time, 0)
        query_end = SceneBoundaryCache.time_to_sec(args.end_time, float("inf"))
        data_start = SceneBoundaryCache.time_to_sec(cached_start, 0)
        data_end = SceneBoundaryCache.time_to_sec(cached_end, float("inf"))
        union_start = min(query_start, data_start)
        union_end = max(query_end, data_end)
        scan_start_time = str(union_start)
        scan_end_time = None if union_end == float("inf") else str(union_end)
    return scan_start_time, scan_end_time


def _resolve_auto_ema_table(model_name):
    """The real (override-aware) EMA-by-duration Buffer/Decay table for `model_name` --
    reuses --scene-batch's own ADR-057 user-editable table (scene_batch.EMA_OVERRIDES_PATH)
    so --scene-batch-auto-ema on a regular (non---scene-batch) conversion is governed by
    the exact same, possibly hand-edited, values as --scene-batch's. Imported lazily (not
    at module load time) since scene_batch.py itself imports from this module at call
    time, and a module-level import here would create a circular import."""
    from . import scene_batch
    base_table = scene_batch.EMA_BY_DURATION_TABLES.get(model_name, scene_batch.EMA_BY_DURATION_VDA_L)
    return scene_batch._table_from_override(model_name, base_table) or base_table


def _scene_ema_schedule_rows(segment_pts, args, native_fps, range_start, range_end):
    """The real per-scene lookup shared by compute_scene_ema_schedule (applies the
    schedule at each reset point -- ADR-060) and the Auto EMA report/live-log
    (records what was actually applied -- ADR-062). One row per scene, in order:
    {"scene_index" (0-based), "start_sec", "end_sec", "settings"} where "settings"
    is (ema_buffer, ema_decay) or None if the table had no valid match for that
    scene's duration; every row but the first also carries "pts", the exact
    boundary pts (reset_pts/segment_pts membership elsewhere in this file) at
    which that scene starts. See compute_scene_ema_schedule for the meaning of
    `range_start`/`range_end`."""
    from .scene_batch import resolve_scene_scan_fps, _overrides_for_scene

    table = _resolve_auto_ema_table(getattr(args, "scene_batch_auto_ema_model", None) or "3DECKER VDA_L")
    scan_fps = resolve_scene_scan_fps(native_fps, args.max_fps)
    sorted_pts = sorted(segment_pts)

    def _lookup(scene_index, scene_start, scene_end):
        overrides = _overrides_for_scene(table, scene_index, scene_start, max(scene_end - scene_start, 0.0))
        buffer, decay = overrides.get("ema_buffer"), overrides.get("ema_decay")
        return (buffer, decay) if buffer is not None and decay is not None else None

    if not sorted_pts:
        return [{"scene_index": 0, "start_sec": range_start, "end_sec": range_end,
                 "settings": _lookup(0, range_start, range_end)}]

    first_cut_time = sorted_pts[0] / scan_fps
    rows = [{"scene_index": 0, "start_sec": range_start, "end_sec": first_cut_time,
             "settings": _lookup(0, range_start, first_cut_time)}]

    for i, pts in enumerate(sorted_pts):
        start_sec = pts / scan_fps
        end_sec = sorted_pts[i + 1] / scan_fps if i + 1 < len(sorted_pts) else range_end
        rows.append({"scene_index": i + 1, "start_sec": start_sec, "end_sec": end_sec,
                     "settings": _lookup(i + 1, start_sec, end_sec), "pts": pts})

    return rows


def compute_scene_ema_schedule(segment_pts, args, native_fps, range_start, range_end):
    """
    Precomputes --scene-batch-auto-ema's Buffer/Decay for every scene of a regular
    (non---scene-batch) --scene-detect run, from the COMPLETE, upfront `segment_pts`
    set -- every scene's real duration is known before frame processing starts, the
    same way --scene-batch already knows it for its own, separate per-file pipeline
    (see scene_batch._overrides_for_scene, which this reuses directly).

    `range_start`/`range_end` are the ABSOLUTE (from the true start of the source
    video) seconds of the portion actually being converted (--start-time/--end-time,
    or the full source when unset). This matters because FPSFilter's pts (the same
    units `segment_pts` values and resolve_scene_scan_fps are already expressed in,
    see ADR-059) are absolute source-timeline positions, not relative to any trim --
    so scene 0 starts at range_start, not at 0. `range_end` also stands in for "the
    very last scene has no next boundary" -- that scene's duration is measured out to
    range_end (the end of the portion being converted, i.e. the remaining video
    length from its own start).

    Returns (first_scene_settings, boundary_settings):
      - first_scene_settings: (ema_buffer, ema_decay) for the scene starting at
        range_start (applied once, before frame processing begins, overriding the
        fixed --ema-decay/--ema-buffer), or None if the table has no valid match.
      - boundary_settings: {pts: (ema_buffer, ema_decay)} for the scene that STARTS
        at each detected cut, keyed by the exact pts value already used for
        reset_pts/segment_pts membership elsewhere in this file.
    """
    rows = _scene_ema_schedule_rows(segment_pts, args, native_fps, range_start, range_end)
    first_scene_settings = rows[0]["settings"]
    boundary_settings = {r["pts"]: r["settings"] for r in rows[1:] if r["settings"] is not None}
    return first_scene_settings, boundary_settings


def _log_scene_ema_row(row):
    """Live text visibility for Auto EMA by Scene Length on the regular
    (non---scene-batch) path (ADR-062): prints the real per-scene Buffer/Decay AS
    that scene's schedule is actually applied during the run, mirroring
    scene_batch._process_scenes's own "[scene-batch] converting scene N/M: ..."
    per-scene print convention -- this project's established style for a live,
    per-scene status line. `row` is one entry from _scene_ema_schedule_rows, with
    "settings" already confirmed non-None by the caller. Scene numbering here is
    1-based (matching scene_batch's own live "scene N/M" line); the saved CSV
    report uses the 0-based "scene_index" instead, matching scene_manifest.csv's
    own column."""
    buffer, decay = row["settings"]
    start_sec, end_sec = row["start_sec"], row["end_sec"]
    print(f"[auto-ema] scene {row['scene_index'] + 1}: {start_sec:.1f}s-{end_sec:.1f}s "
          f"({end_sec - start_sec:.1f}s) -> buffer={buffer} decay={decay:.3f}", file=sys.stderr)


def _write_scene_ema_report(report_path, rows):
    """Sidecar CSV (ADR-062) recording exactly which EMA Buffer/Decay Auto EMA by
    Scene Length actually applied to each scene of a regular (non---scene-batch)
    conversion -- the after-the-fact equivalent of --scene-batch's own
    scene_manifest.csv (see _write_scene_manifest) for the pipeline that had no
    saved record at all before this. Written once, after the whole run has already
    finished successfully (unlike scene_manifest.csv, nothing reads this
    incrementally mid-run), via temp-name + os.replace so a reader can never
    observe a partially-written file (CS-IO-001)."""
    fieldnames = ["scene_index", "start_time_sec", "duration_sec", "ema_buffer", "ema_decay"]
    tmp_path = report_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, report_path)
    return report_path


def _scene_ema_report_summary(applied_rows):
    """(scene_count, distinct_count) for Auto EMA by Scene Length's end-of-run
    summary line (ADR-062) -- `applied_rows` is the subset of
    _scene_ema_schedule_rows' output whose "settings" resolved to a real
    (ema_buffer, ema_decay) pair. distinct_count counts unique (ema_buffer,
    ema_decay) pairs actually used, not unique scenes."""
    return len(applied_rows), len({r["settings"] for r in applied_rows})


def _fmt_hms_report(sec):
    """m:ss.mmm (or h:mm:ss.mmm past one hour) -- shared time format for the Auto
    EMA report's HTML and plain-text siblings (ADR-074/ADR-075)."""
    sec = float(sec)
    sign = "-" if sec < 0 else ""
    sec = abs(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h >= 1:
        return f"{sign}{int(h)}:{int(m):02d}:{s:06.3f}"
    return f"{sign}{int(m)}:{s:06.3f}"


_EMA_REPORT_HEADERS = ["Scene", "Start Time", "Duration (s)", "EMA Buffer", "EMA Decay"]


def _scene_ema_report_group_parity(rows):
    """Assigns each row a 0/1 "group" number that flips every time consecutive
    rows' (ema_buffer, ema_decay) pair changes -- shared by the HTML and TXT
    report writers so scenes using identical smoothing settings are visually
    banded together instead of a plain alternating-row stripe that carries no
    real information (ADR-077)."""
    parity = 0
    prev_key = None
    out = []
    for r in rows:
        key = (r["ema_buffer"], r["ema_decay"])
        if key != prev_key:
            parity ^= 1
            prev_key = key
        out.append(parity)
    return out


def _write_scene_ema_report_txt(txt_path, rows, scene_count, distinct_count, source_name):
    """Plain-text sibling of _write_scene_ema_report's CSV (ADR-075/ADR-077),
    readable in Notepad or any plain text viewer with no HTML rendering: a
    fixed-width table, column widths sized to the actual data so it stays
    aligned regardless of row count. Deliberately mirrors the HTML sibling's
    title/summary line and column headers word-for-word, and marks the same
    same-Buffer/Decay row groups the HTML shades (via a leading marker
    column, plain text's only real equivalent to a background tint) so
    switching between the two formats feels like the same report, not two
    different ones. Same temp-name + os.replace pattern (CS-IO-001) as the
    CSV/HTML siblings."""
    headers = _EMA_REPORT_HEADERS
    groups = _scene_ema_report_group_parity(rows)
    cells = [
        [
            str(r["scene_index"]),
            _fmt_hms_report(r["start_time_sec"]),
            f"{float(r['duration_sec']):.3f}",
            str(r["ema_buffer"]),
            str(r["ema_decay"]),
        ]
        for r in rows
    ]
    widths = [max(len(headers[i]), max((len(row[i]) for row in cells), default=0)) for i in range(5)]

    def fmt_row(values, marker=" "):
        # Scene (column 0) left-justified, the four numeric columns right-justified.
        parts = [values[0].ljust(widths[0])]
        parts += [values[i].rjust(widths[i]) for i in range(1, 5)]
        return marker + " " + "  ".join(parts)

    lines = [
        f"Auto EMA by Scene Length -- {source_name}",
        f"{scene_count} scenes -- {distinct_count} distinct Buffer/Decay values used",
        "",
        fmt_row(headers),
        "  " + "  ".join("-" * w for w in widths),
    ]
    lines += [fmt_row(row, "|" if groups[i] else " ") for i, row in enumerate(cells)]

    tmp_path = txt_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp_path, txt_path)
    return txt_path


def _write_scene_ema_report_html(html_path, rows, scene_count, distinct_count, source_name):
    """Human-readable sibling of _write_scene_ema_report's plain CSV (ADR-074,
    revised ADR-077): a single self-contained HTML file (no network/CDN
    dependency -- this is an offline desktop tool, so every font is a system
    stack, nothing loaded remotely) with the same rows as a real, sortable,
    scrollable table, so the numbers are readable without opening the CSV in
    a separate spreadsheet program. Rows sharing the same (Buffer, Decay) are
    banded together (see _scene_ema_report_group_parity) instead of a plain
    even/odd stripe, so runs of scenes using identical smoothing settings are
    visible at a glance -- the same grouping the TXT sibling marks with a `|`
    column, so the two formats read as one report. Written after the CSV,
    same temp-name + os.replace pattern (CS-IO-001) so a reader never sees a
    half-written file."""
    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;"))

    fmt_hms = _fmt_hms_report
    groups = _scene_ema_report_group_parity(rows)

    row_html = []
    for i, r in enumerate(rows):
        cls = "g1" if groups[i] else "g0"
        row_html.append(
            f'<tr class="{cls}">'
            f"<td>{esc(r['scene_index'])}</td>"
            f"<td>{esc(fmt_hms(r['start_time_sec']))}</td>"
            f"<td>{esc(f'{float(r['duration_sec']):.3f}')}</td>"
            f"<td>{esc(r['ema_buffer'])}</td>"
            f"<td>{esc(r['ema_decay'])}</td>"
            "</tr>"
        )

    header_cells = "".join(f"<th data-n>{esc(h)}</th>" for h in _EMA_REPORT_HEADERS)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auto EMA by Scene Length Report</title>
<style>
  :root {{
    --bg:#faf9f7; --surface:#ffffff; --fg:#2a2622; --fg-dim:#847c70; --border:#e2ddd4;
    --head-bg:#f2efe9; --row-a:#ffffff; --row-b:#f6f3ee; --accent:#93714c; --accent-fg:#fff9f2;
    --shadow:0 1px 2px rgba(40,32,20,.06), 0 6px 16px -8px rgba(40,32,20,.12);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#17140f; --surface:#1d1a15; --fg:#ece6da; --fg-dim:#9c9284; --border:#3a352c;
      --head-bg:#252019; --row-a:#1d1a15; --row-b:#232019; --accent:#d3ac7a; --accent-fg:#211a10;
      --shadow:0 1px 2px rgba(0,0,0,.3), 0 10px 24px -10px rgba(0,0,0,.5);
    }}
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:28px 20px; background:var(--bg); color:var(--fg);
    font:14px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif;
  }}
  .card {{
    max-width:900px; margin:0 auto; background:var(--surface); border:1px solid var(--border);
    border-radius:12px; box-shadow:var(--shadow); overflow:hidden;
  }}
  header {{ padding:18px 22px; border-bottom:1px solid var(--border); }}
  h1 {{ font-size:17px; font-weight:600; margin:0 0 8px; text-wrap:balance; }}
  .stats {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .stat {{
    font:600 12px/1 -apple-system, "Segoe UI", system-ui, sans-serif; color:var(--accent);
    background:var(--head-bg); border:1px solid var(--border); border-radius:999px; padding:5px 11px;
  }}
  .hint {{ margin-top:8px; color:var(--fg-dim); font-size:12px; }}
  .wrap {{ max-height:calc(100vh - 170px); overflow:auto; }}
  table {{
    border-collapse:collapse; width:100%;
    font:13.5px/1.3 ui-monospace, "Cascadia Mono", "Segoe UI Mono", "SFMono-Regular", Consolas, monospace;
    font-variant-numeric:tabular-nums;
  }}
  th, td {{ padding:7px 14px; text-align:right; border-bottom:1px solid var(--border); white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{
    position:sticky; top:0; background:var(--head-bg); cursor:pointer; user-select:none;
    font:600 11px/1 -apple-system, "Segoe UI", system-ui, sans-serif; letter-spacing:.04em;
    text-transform:uppercase; color:var(--fg-dim); box-shadow:0 1px 0 var(--border);
  }}
  th:hover {{ color:var(--accent); }}
  tr.g0 td {{ background:var(--row-a); }}
  tr.g1 td {{ background:var(--row-b); }}
  tbody tr:hover td {{ background:var(--head-bg); }}
  th.sorted {{ color:var(--accent); }}
  th.sorted::after {{ content:" \\25BE"; }}
  th.sorted.asc::after {{ content:" \\25B4"; }}
</style></head>
<body>
<div class="card">
<header>
  <h1>Auto EMA by Scene Length &mdash; {esc(source_name)}</h1>
  <div class="stats">
    <span class="stat">{scene_count} scenes</span>
    <span class="stat">{distinct_count} distinct Buffer/Decay values</span>
  </div>
  <div class="hint">Click a column header to sort. Shaded bands mark consecutive scenes sharing the same Buffer/Decay.</div>
</header>
<div class="wrap">
<table id="t">
<thead><tr>{header_cells}</tr></thead>
<tbody>
{"".join(row_html)}
</tbody>
</table>
</div>
</div>
<script>
(function() {{
  var table = document.getElementById("t");
  var ths = table.querySelectorAll("th");
  ths.forEach(function(th, idx) {{
    th.addEventListener("click", function() {{
      var asc = !(th.classList.contains("sorted") && th.classList.contains("asc"));
      ths.forEach(function(h) {{ h.classList.remove("sorted", "asc"); }});
      th.classList.add("sorted"); if (asc) th.classList.add("asc");
      var tbody = table.querySelector("tbody");
      var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
      var isNumeric = th.hasAttribute("data-n") && idx !== 1;
      rows.sort(function(a, b) {{
        var av = a.children[idx].textContent, bv = b.children[idx].textContent;
        if (isNumeric) {{ av = parseFloat(av); bv = parseFloat(bv); }}
        if (av < bv) return asc ? -1 : 1;
        if (av > bv) return asc ? 1 : -1;
        return 0;
      }});
      rows.forEach(function(r, i) {{
        r.className = (i % 2) ? "g1" : "g0";
        tbody.appendChild(r);
      }});
    }});
  }});
}})();
</script>
</body></html>"""

    tmp_path = html_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp_path, html_path)
    return html_path


def process_video_full(input_filename, output_path, args, depth_model, side_model):
    is_preview = getattr(args, "preview", False)
    scene_cache_max_fps = args.max_fps  # capture before --preview clamps it, so cache key stays stable
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)
    is_video_depth_anything = depth_model.get_name() == "VideoDepthAnything"
    is_video_depth_anything_streaming = depth_model.get_name() == "VideoDepthAnythingStreaming"
    is_inpaint_model = args.method in {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}
    if (getattr(args, "temporal_stabilize", False) and not is_video_depth_anything
            and not (args.low_vram or args.debug_depth or is_video_depth_anything_streaming or is_inpaint_model)):
        warnings.warn("--temporal-stabilize only takes effect on the single-frame processing path "
                       "(used automatically for --low-vram, --debug-depth, VDA streaming models, and "
                       "inpaint stereo methods) -- it will have no effect on this run.")
    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if ema_normalize:
        depth_model.enable_ema(decay=args.ema_decay, buffer_size=args.ema_buffer,
                                motion_adaptive=getattr(args, "ema_motion_adaptive", False))

    if (
            args.compile and
            side_model is not None and
            not isinstance(side_model, DeviceSwitchInference) and
            not hasattr(side_model, "compile_context")
    ):
        try:
            # compile_model() (nunif/models/utils.py) already catches real compile
            # failures on its own (ADR-068 amendment) and enables torch._dynamo's
            # suppress_errors fallback so a failure deferred to the first real
            # forward() call doesn't crash this conversion either -- this try/except
            # is defense-in-depth at this real pipeline entry point, matching the
            # same "not a change to what gets compiled, only to what can possibly
            # crash the run" precedent ADR-068 established at the GUI checkbox site.
            side_model = compile_model(side_model, device=args.state["device"])
        except Exception as e:
            warnings.warn(f"torch.compile failed for the stereo model -- "
                           f"continuing without it: {e.__class__.__name__}: {e}")

    if getattr(args, "preview", False):
        import copy as _copy
        args = _copy.copy(args)
        args.max_fps = min(args.max_fps, 1.0)
        if args.limit_resolution is None or args.limit_resolution > 256:
            args.limit_resolution = 256
        if getattr(args, "end_time", None) is None:
            args.end_time = 60.0

    output_parent_dir = path.basename(output_path)
    input_parent_dir = path.basename(path.dirname(input_filename))
    if is_output_dir(output_path) or (output_parent_dir != "" and output_parent_dir == input_parent_dir):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(path.basename(input_filename), args, video=True))
    else:
        output_filename = output_path

    if getattr(args, "preview", False):
        base, ext = path.splitext(output_filename)
        output_filename = base + "_preview" + ext

    if (
            # --resume and already processed
            (args.resume and path.exists(output_filename)) or
            # --skip-error and already terminated with an error
            (args.skip_error and path.exists(VU.make_error_file_path(output_filename)))
    ):
        return  # skip

    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    make_parent_dir(output_filename)
    if args.scene_detect or args.scene_detect_only:
        _notify_stage(args, STAGE_SCENE_DETECT)
        segment_pts = None
        scan_start_time, scan_end_time = args.start_time, args.end_time
        if not args.disable_scene_cache:
            # Look up the cache under the real (pre-preview-clamp) max_fps, so Quick Preview
            # reuses the same cache as a full run instead of always missing and rescanning.
            segment_pts = try_load_scene_cache(input_filename, args, max_fps=scene_cache_max_fps)
            if segment_pts is None and not is_preview:
                scan_start_time, scan_end_time = widen_scene_scan_range(
                    input_filename, args, max_fps=scene_cache_max_fps)

        if segment_pts is None:
            with TorchHubDir(HUB_MODEL_DIR):
                segment_pts = SBD.detect_boundary(
                    input_filename,
                    max_fps=args.max_fps,
                    device=args.state["device"],
                    hwaccel=args.hwaccel,
                    disable_software_fallback=args.disable_software_fallback,
                    start_time=scan_start_time,
                    end_time=scan_end_time,
                    stop_event=args.state["stop_event"],
                    suspend_event=args.state["suspend_event"],
                    tqdm_fn=args.state["tqdm_fn"],
                    tqdm_title=f"{path.basename(input_filename)}: Scene Boundary Detection",
                )
                # Don't let a fast/low-fps preview scan, or a scan that got cancelled
                # partway through, overwrite the real cache with lower-quality or
                # incomplete results saved under the full requested range (see
                # should_save_scene_cache). Checked BEFORE the stop_event early-return
                # below, not after -- saving first and checking stop_event second would
                # let a cancelled scan's incomplete result get cached as if complete.
                if should_save_scene_cache(args.disable_scene_cache, is_preview, args.state["stop_event"]):
                    save_scene_cache(input_filename, segment_pts, args,
                                      start_time=scan_start_time, end_time=scan_end_time,
                                      max_fps=scene_cache_max_fps)
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    return
            gc_collect()
    else:
        segment_pts = set()
    if args.scene_detect_only:
        return

    # --scene-batch-auto-ema on the regular (non---scene-batch) path: segment_pts is
    # now the COMPLETE, upfront list of every cut in the whole processed range, so
    # every scene's real duration can be computed before frame processing starts (see
    # _scene_ema_schedule_rows). scene_ema_boundary_settings maps each boundary pts
    # to the (ema_buffer, ema_decay) for the scene that STARTS there, applied at each
    # reset_ema/reset_pts point in bind_single_frame_callback/bind_batch_frame_callback/
    # bind_vda_frame_callback below; the very first scene (no boundary of its own) is
    # applied here by re-calling enable_ema(), immediately overriding the fixed
    # --ema-decay/--ema-buffer set above. Off by default -- stays {}/[] (an exact
    # no-op downstream) unless --scene-batch-auto-ema and --scene-detect are both on.
    # scene_ema_report_rows/scene_ema_report_by_pts (ADR-062) are the same schedule
    # kept around purely for user-visible reporting: a live "[auto-ema] scene N: ..."
    # log line as each scene's settings are actually applied (see _log_scene_ema_row,
    # passed into the binders below), plus a saved "<output>.auto_ema_report.csv"
    # sidecar and an end-of-run summary line written once processing finishes (see
    # the bottom of this function).
    scene_ema_boundary_settings = {}
    scene_ema_report_rows = []
    scene_ema_report_by_pts = {}
    if _scene_auto_ema_active(args, ema_normalize):
        native_fps = None
        metadata_duration = None
        try:
            metadata = VU.VideoMetadata.from_file(input_filename)
            native_fps = float(metadata.get_fps())
            metadata_duration = metadata.guess_duration(to_int=False)
        except Exception as e:
            warnings.warn(f"--scene-batch-auto-ema: could not read '{input_filename}' metadata "
                          f"({e}) -- falling back to the fixed --ema-decay/--ema-buffer for this file.")
        if native_fps:
            range_start = SceneBoundaryCache.time_to_sec(args.start_time, 0.0)
            range_end = SceneBoundaryCache.time_to_sec(args.end_time, None)
            if range_end is None:
                range_end = metadata_duration if metadata_duration and metadata_duration > 0 else range_start
            elif metadata_duration and metadata_duration > 0:
                range_end = min(range_end, metadata_duration)
            scene_ema_report_rows = _scene_ema_schedule_rows(
                segment_pts, args, native_fps, range_start, range_end)
            first_scene_settings = scene_ema_report_rows[0]["settings"]
            scene_ema_boundary_settings = {r["pts"]: r["settings"] for r in scene_ema_report_rows[1:]
                                           if r["settings"] is not None}
            scene_ema_report_by_pts = {r["pts"]: r for r in scene_ema_report_rows[1:]
                                       if r["settings"] is not None}
            if first_scene_settings is not None:
                depth_model.enable_ema(decay=first_scene_settings[1], buffer_size=first_scene_settings[0],
                                       motion_adaptive=getattr(args, "ema_motion_adaptive", False))
                _log_scene_ema_row(scene_ema_report_rows[0])

    if args.autocrop is not None:
        _notify_stage(args, STAGE_AUTOCROP)
        crop = AutoCrop.from_video_file(
            input_filename,
            mode=args.autocrop,
            uncrop_enabled=False,
            vf=args.vf,
            hwaccel=args.hwaccel,
            disable_software_fallback=args.disable_software_fallback,
            device=args.state["device"],
            batch_size=args.batch_size,
            stop_event=args.state["stop_event"],
            suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
            tqdm_title=f"{path.basename(input_filename)}: AutoCrop Analysis",
        ).get_crop()
        if crop is not None:
            crop_filter = f"crop=x={crop[0]}:y={crop[1]}:w={crop[2]}:h={crop[3]}"
            video_filter = args.vf + f",{crop_filter}" if args.vf else crop_filter
        else:
            video_filter = args.vf
    else:
        video_filter = args.vf

    # Integrate preprocess_image() logic into vf
    video_filter = add_preprocess_vf(video_filter, args)

    # HDR metadata extraction (DV RPU + HDR10+) before encoding
    _hdr_ffmpeg_bin = None
    _hdr_dovi_bin = None
    _hdr_hdr10plus_bin = None
    _hdr_rpu_path = None
    _hdr_h10p_path = None

    if getattr(args, "preserve_dowi", False):
        _hdr_ffmpeg_bin = _get_ffmpeg_bin()
        _hdr_out_dir = path.dirname(path.abspath(output_filename))
        hdr_types = _detect_hdr_types(input_filename, _find_ffprobe())

        if not hdr_types["dv"] and not hdr_types["hdr10plus"]:
            print("--preserve-dowi: no DV or HDR10+ detected in source, skipping extraction.",
                  file=sys.stderr)
        else:
            _notify_stage(args, STAGE_HDR_EXTRACT)
            _hdr_tmp_hevc = path.join(_hdr_out_dir, "_iw3_src.hevc")
            # Match whatever time range is ACTUALLY being converted -- without this, a
            # full-movie run "worked" only by coincidence (its extraction naturally
            # covered the whole file, matching a whole-file output), while any
            # --start-time/--end-time clip extracted DV/HDR10+ metadata for the ENTIRE
            # source but tried to inject it onto an output covering only a fraction of
            # it, a mismatch severe enough that injection failed or was silently skipped.
            _hdr_trim_args = []
            if getattr(args, "start_time", None):
                _hdr_trim_args += ["-ss", str(parse_time(args.start_time))]
            if getattr(args, "end_time", None):
                _hdr_trim_args += ["-to", str(parse_time(args.end_time))]
            try:
                subprocess.run(
                    [_hdr_ffmpeg_bin, "-y", *_hdr_trim_args, "-i", str(input_filename),
                     "-c:v", "copy", "-an", "-f", "hevc", _hdr_tmp_hevc],
                    check=True, capture_output=True,
                )
                if hdr_types["dv"]:
                    _hdr_dovi_bin = _find_dovi_tool()
                    if _hdr_dovi_bin:
                        _hdr_rpu_path = path.join(_hdr_out_dir, "_iw3_rpu.bin")
                        try:
                            subprocess.run(
                                [_hdr_dovi_bin, "extract-rpu", "-i", _hdr_tmp_hevc, "-o", _hdr_rpu_path],
                                check=True, capture_output=True,
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"--preserve-dowi: DV RPU extraction failed: "
                                  f"{e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
                            _hdr_rpu_path = None
                        except OSError as e:
                            # CS-SUBPROCESS-001: subprocess.run() itself raises a bare
                            # OSError (not CalledProcessError) if the child process
                            # never starts at all (e.g. the OS refuses to spawn it) --
                            # distinct from a process that ran and exited non-zero.
                            # Never previously caught here, so this optional
                            # (--preserve-dowi) step could crash a whole real
                            # conversion job on a transient OS-level spawn failure
                            # instead of just skipping DV preservation for this run.
                            print(f"--preserve-dowi: DV RPU extraction failed to start "
                                  f"({e.__class__.__name__}: {e}).", file=sys.stderr)
                            _hdr_rpu_path = None

                if hdr_types["hdr10plus"]:
                    _hdr_hdr10plus_bin = _find_hdr10plus_tool()
                    if _hdr_hdr10plus_bin:
                        _hdr_h10p_path = path.join(_hdr_out_dir, "_iw3_hdr10plus.json")
                        try:
                            subprocess.run(
                                [_hdr_hdr10plus_bin, "extract", "-i", _hdr_tmp_hevc, "-o", _hdr_h10p_path],
                                check=True, capture_output=True,
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"--preserve-dowi: HDR10+ extraction failed: "
                                  f"{e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
                            _hdr_h10p_path = None
                        except OSError as e:
                            # CS-SUBPROCESS-001: see the matching DV RPU comment above.
                            print(f"--preserve-dowi: HDR10+ extraction failed to start "
                                  f"({e.__class__.__name__}: {e}).", file=sys.stderr)
                            _hdr_h10p_path = None

            except subprocess.CalledProcessError as e:
                print(f"--preserve-dowi: source HEVC extraction failed: "
                      f"{e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
            except OSError as e:
                # CS-SUBPROCESS-001: subprocess.run() raises a bare OSError (not
                # CalledProcessError) when the OS itself fails to spawn the child
                # process (e.g. a transient handle/resource ceiling) rather than when
                # the process runs and exits non-zero -- previously uncaught here, so
                # it escaped this whole --preserve-dowi block as a raw crash instead of
                # just skipping DV/HDR10+ preservation for this run. Matches the
                # "an optional acceleration/preservation path must never crash a real
                # conversion" precedent from ADR-070/ADR-071.
                print(f"--preserve-dowi: source HEVC extraction failed to start "
                      f"({e.__class__.__name__}: {e}).", file=sys.stderr)
            finally:
                if path.exists(_hdr_tmp_hevc):
                    try:
                        os.remove(_hdr_tmp_hevc)
                    except Exception:
                        pass

    def config_callback(metadata):
        fps = metadata.get_fps()
        if float(fps) > args.max_fps:
            fps = args.max_fps
        pix_fmt = args.pix_fmt
        upgrade = getattr(args, "upgrade_pix_fmt", None)
        if upgrade and not metadata.use_16bit:
            pix_fmt = _upgrade_pix_fmt(pix_fmt, upgrade)
            pix_fmt = _clamp_pix_fmt_for_codec(pix_fmt, args.video_codec)

        extra_meta = {}
        comment = _build_iw3_comment_metadata(args)
        if comment:
            extra_meta["comment"] = comment

        return VU.VideoOutputConfig(
            fps=fps,
            container_format=args.video_format,
            video_codec=args.video_codec,
            pix_fmt=pix_fmt,
            colorspace=args.colorspace,
            options=make_video_codec_option(args, input_filename),
            container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
            metadata=extra_meta,
        )

    # ADR-072: process_video_full() (this function) never announced
    # STAGE_DEPTH_STEREO -- the only other _notify_stage(STAGE_DEPTH_STEREO) call
    # site is in the separate, simpler process_video() function (used for
    # keyframe/per-file batch mode), which this function does not go through. Without
    # this call, the GUI's "Step k/N: <stage>" title/status bar stayed frozen on
    # whatever stage last fired -- STAGE_HDR_EXTRACT when --preserve-dowi is set, or
    # STAGE_AUTOCROP/STAGE_SCENE_DETECT otherwise -- for the ENTIRE real depth/stereo
    # encode below (often the majority of a multi-hour job), mislabeling any crash
    # that actually happens inside VU.process_video() (e.g. the real input-container
    # hwaccel-open failure ADR-071 addresses) as if it were still happening in an
    # earlier, already-finished stage. See docs/ai/AI_DECISIONS.md ADR-072.
    _notify_stage(args, STAGE_DEPTH_STEREO)

    if is_video_depth_anything:
        with depth_model.compile_context(enabled=args.compile), try_compile_context(side_model, enabled=args.compile):
            VU.process_video(
                input_filename, output_filename,
                config_callback=config_callback,
                frame_callback=bind_vda_frame_callback(
                    depth_model=depth_model,
                    side_model=side_model,
                    segment_pts=segment_pts,
                    args=args,
                    scene_ema_settings=scene_ema_boundary_settings,
                    scene_ema_report=scene_ema_report_by_pts,
                ),
                vf=video_filter,
                stop_event=args.state["stop_event"],
                suspend_event=args.state["suspend_event"],
                tqdm_fn=args.state["tqdm_fn"],
                title=_progress_title(path.basename(input_filename), args),
                start_time=args.start_time,
                end_time=args.end_time,
                device=args.state["device"],
                hwaccel=args.hwaccel,
                disable_software_fallback=args.disable_software_fallback,
            )

    elif args.low_vram or args.debug_depth or is_video_depth_anything_streaming or is_inpaint_model:
        with depth_model.compile_context(enabled=args.compile), try_compile_context(side_model, enabled=args.compile):
            VU.process_video(
                input_filename, output_filename,
                config_callback=config_callback,
                frame_callback=bind_single_frame_callback(
                    depth_model=depth_model,
                    side_model=side_model,
                    segment_pts=segment_pts,
                    args=args,
                    scene_ema_settings=scene_ema_boundary_settings,
                    scene_ema_report=scene_ema_report_by_pts,
                ),
                vf=video_filter,
                stop_event=args.state["stop_event"],
                suspend_event=args.state["suspend_event"],
                tqdm_fn=args.state["tqdm_fn"],
                title=_progress_title(path.basename(input_filename), args),
                start_time=args.start_time,
                end_time=args.end_time,
                device=args.state["device"],
                hwaccel=args.hwaccel,
                disable_software_fallback=args.disable_software_fallback,
            )
    else:
        extra_queue = 1 if len(args.state["devices"]) == 1 else 0
        minibatch_size = args.batch_size // 2 or 1 if args.tta else args.batch_size

        frame_callback, preprocess_callback = bind_batch_frame_callback(
            depth_model=depth_model,
            side_model=side_model,
            segment_pts=segment_pts,
            args=args,
            scene_ema_settings=scene_ema_boundary_settings,
            scene_ema_report=scene_ema_report_by_pts,
        )
        frame_callback = VU.FrameCallbackPool(
            frame_callback=frame_callback,
            preprocess_callback=preprocess_callback,
            batch_size=minibatch_size,
            device=args.state["devices"],
            max_workers=args.max_workers,
            max_batch_queue=args.max_workers + extra_queue,
            require_pts=True,
            require_flush=True,
            use_16bit=use_16bit,
        )
        try:
            with depth_model.compile_context(enabled=args.compile):
                VU.process_video(
                    input_filename, output_filename,
                    config_callback=config_callback,
                    frame_callback=frame_callback,
                    vf=video_filter,
                    stop_event=args.state["stop_event"],
                    suspend_event=args.state["suspend_event"],
                    tqdm_fn=args.state["tqdm_fn"],
                    title=_progress_title(path.basename(input_filename), args),
                    start_time=args.start_time,
                    end_time=args.end_time,
                    device=args.state["device"],
                    hwaccel=args.hwaccel,
                    disable_software_fallback=args.disable_software_fallback,
                )
        finally:
            frame_callback.shutdown()

    # HDR metadata injection (DV RPU + HDR10+) after encoding
    if (_hdr_rpu_path or _hdr_h10p_path) and path.exists(output_filename):
        _notify_stage(args, STAGE_HDR_REINJECT)
        try:
            _inject_hdr_metadata(
                output_filename,
                _hdr_rpu_path, _hdr_h10p_path,
                _hdr_ffmpeg_bin, _hdr_dovi_bin, _hdr_hdr10plus_bin,
                path.dirname(path.abspath(output_filename)),
                configured_video_codec=args.video_codec,
            )
        except Exception as e:
            print(f"--preserve-dowi: HDR metadata injection failed: {e}", file=sys.stderr)
        finally:
            for f in filter(None, [_hdr_rpu_path, _hdr_h10p_path]):
                try:
                    os.remove(f)
                except Exception:
                    pass

    # MKV StereoMode tagging (opt-in, final touch-up after everything else, including
    # any HDR reinjection above, has already finished) -- see ADR-033 / _apply_stereo_mode_tag.
    if path.exists(output_filename):
        _apply_stereo_mode_tag(output_filename, args)

    # Auto EMA by Scene Length report (ADR-062): a saved, after-the-fact record of
    # exactly which Buffer/Decay this run actually applied to each scene, matching
    # --scene-batch's own scene_manifest.csv in spirit -- the regular (non---
    # scene-batch) path had no equivalent before this. Off by default: nothing is
    # written unless the feature genuinely ran (scene_ema_report_rows non-empty,
    # i.e. --scene-batch-auto-ema and --scene-detect were both actually on) and the
    # run actually produced its final output.
    applied_ema_rows = [r for r in scene_ema_report_rows if r["settings"] is not None]
    if applied_ema_rows and path.exists(output_filename):
        report_path = output_filename + ".auto_ema_report.csv"
        report_rows = [
            {
                "scene_index": r["scene_index"],
                "start_time_sec": f"{r['start_sec']:.3f}",
                "duration_sec": f"{r['end_sec'] - r['start_sec']:.3f}",
                "ema_buffer": r["settings"][0],
                "ema_decay": r["settings"][1],
            }
            for r in applied_ema_rows
        ]
        _write_scene_ema_report(report_path, report_rows)
        scene_count, distinct_count = _scene_ema_report_summary(applied_ema_rows)
        # Human-readable sibling of the CSV above (ADR-074) -- a real, sortable
        # HTML table, so these numbers are readable without opening a separate
        # spreadsheet program. Best-effort only: never let a report-writing
        # problem fail an otherwise-successful conversion job.
        html_path = output_filename + ".auto_ema_report.html"
        try:
            _write_scene_ema_report_html(html_path, report_rows, scene_count, distinct_count,
                                          path.basename(output_filename))
        except Exception as e:
            print(f"[auto-ema] could not write HTML report ({e.__class__.__name__}: {e}) -- "
                  f"the CSV at {report_path} is still complete.", file=sys.stderr)
        # Plain-text sibling (ADR-075) -- readable in Notepad, no HTML rendering needed.
        # Same best-effort try/except as the HTML sibling above.
        txt_path = output_filename + ".auto_ema_report.txt"
        try:
            _write_scene_ema_report_txt(txt_path, report_rows, scene_count, distinct_count,
                                         path.basename(output_filename))
        except Exception as e:
            print(f"[auto-ema] could not write TXT report ({e.__class__.__name__}: {e}) -- "
                  f"the CSV at {report_path} is still complete.", file=sys.stderr)
        print(f"[auto-ema] Auto EMA by Scene Length: {scene_count} scenes, "
              f"{distinct_count} distinct Buffer/Decay values used (see {report_path}, "
              f"{html_path}, and {txt_path})",
              file=sys.stderr)


def _probe_video_duration(path_str):
    """
    Determine the true, actually-decodable duration of a (possibly not cleanly finalized,
    e.g. after a crash) video file by scanning its packets directly, rather than trusting
    container-level duration metadata that may be missing or stale if the file wasn't
    closed properly.
    """
    import av as _av
    try:
        with _av.open(str(path_str)) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            last_pts = None
            for packet in container.demux(stream):
                if packet.pts is not None:
                    last_pts = packet.pts
            if last_pts is None:
                return None
            frame_dur = float(1.0 / (stream.average_rate or 24))
            return float(last_pts * stream.time_base) + frame_dur
    except Exception:
        return None


def process_video_with_resume(input_filename, output_path, args, depth_model, side_model):
    import json
    import copy
    import av as _av

    if not getattr(args, "auto_resume", False):
        process_video_full(input_filename, output_path, args, depth_model, side_model)
        return

    # Resolve final output filename (mirrors process_video_full logic)
    output_parent_dir = path.basename(output_path)
    input_parent_dir = path.basename(path.dirname(input_filename))
    if is_output_dir(output_path) or (output_parent_dir != "" and output_parent_dir == input_parent_dir):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(output_path, make_output_filename(path.basename(input_filename), args, video=True))
    else:
        output_filename = output_path

    if args.resume and path.exists(output_filename):
        return

    try:
        with _av.open(str(input_filename)) as c:
            duration = float(c.duration) / 1000000.0 if c.duration else None
    except Exception:
        duration = None

    effective_start = parse_time(args.start_time) if getattr(args, "start_time", None) else 0.0
    effective_end = parse_time(args.end_time) if getattr(args, "end_time", None) else (duration or 0.0)
    if duration:
        effective_end = min(effective_end, duration)

    # Short clips: just run a single normal pass, no point wrapping this in the
    # segment/checkpoint machinery below.
    if not duration or (effective_end - effective_start) <= 60.0:
        process_video_full(input_filename, output_path, args, depth_model, side_model)
        return

    ext = path.splitext(output_filename)[1]
    base = path.splitext(output_filename)[0]
    checkpoint_path = output_filename + ".iw3resume"

    # Settings that change the actual video stream's format (codec, pixel format, frame
    # size/layout) can't safely differ between segments of the same job — the final merge
    # just stream-copies pieces together, which requires them to all match. Settings that
    # only affect quality/tuning (3D Strength, Convergence, Depth Resolution, EMA, the
    # depth model itself, etc.) are fine to change between segments — e.g. via Cancel,
    # adjusting settings, then Start again with the same output path.
    fingerprint = {
        "video_codec": getattr(args, "video_codec", None),
        "pix_fmt": getattr(args, "pix_fmt", None),
        "upgrade_pix_fmt": getattr(args, "upgrade_pix_fmt", None),
        "vr180": getattr(args, "vr180", False),
        "half_sbs": getattr(args, "half_sbs", False),
        "tb": getattr(args, "tb", False),
        "half_tb": getattr(args, "half_tb", False),
        "cross_eyed": getattr(args, "cross_eyed", False),
        "rgbd": getattr(args, "rgbd", False),
        "half_rgbd": getattr(args, "half_rgbd", False),
        "anaglyph": getattr(args, "anaglyph", None),
        "max_output_width": getattr(args, "max_output_width", None),
        "max_output_height": getattr(args, "max_output_height", None),
        "keep_aspect_ratio": getattr(args, "keep_aspect_ratio", False),
    }

    # Load any segments left over from a previous interrupted run, and re-verify each
    # one's *actual* coverage by scanning the file itself — never trust a stale recorded
    # position, since the file's real content is the only ground truth after a crash.
    segments = []
    saved_fingerprint = None
    if path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            saved = data.get("segments", [])
            saved_fingerprint = data.get("fingerprint")
            for seg in saved:
                seg_file = seg.get("file")
                seg_start = seg.get("start")
                if seg_file is None or seg_start is None or not path.exists(seg_file):
                    continue
                actual_dur = _probe_video_duration(seg_file)
                if actual_dur is None or actual_dur <= 0.5:
                    continue
                segments.append({"start": seg_start, "end": seg_start + actual_dur, "file": seg_file})
        except Exception:
            segments = []
            saved_fingerprint = None

    if segments and saved_fingerprint is not None and saved_fingerprint != fingerprint:
        changed = {k: (saved_fingerprint.get(k), v) for k, v in fingerprint.items()
                  if saved_fingerprint.get(k) != v}
        print(f"[auto-resume] refusing to continue: this changed since the in-progress pieces "
              f"were made, and would produce a broken/unplayable merge: {changed}. "
              f"Revert that setting to resume normally, or use a different output filename "
              f"to start a separate fresh job.", file=sys.stderr)
        return

    covered_end = max([s["end"] for s in segments], default=effective_start)

    # Recover from a HARD crash/power-loss, not just a graceful stop: if the process died
    # mid-segment, that segment never got renamed from its "_tmp_..." working name to its
    # final name, and never made it into the checkpoint at all. Look for that leftover
    # temp file and salvage whatever of it is actually readable, instead of discarding it
    # and re-encoding that whole stretch from scratch.
    while True:
        seg_index = len(segments)
        seg_file = f"{base}_resume_seg_{seg_index:04d}{ext}"
        orphaned_tmp = path.join(path.dirname(seg_file), "_tmp_" + path.basename(seg_file))
        if not path.exists(orphaned_tmp):
            break
        actual_dur = _probe_video_duration(orphaned_tmp)
        if actual_dur is None or actual_dur <= 0.5:
            # Nothing usable was salvageable (e.g. killed before any frame was flushed).
            try:
                os.remove(orphaned_tmp)
            except Exception:
                pass
            break
        try:
            os.replace(orphaned_tmp, seg_file)
        except Exception:
            break
        segments.append({"start": covered_end, "end": covered_end + actual_dur, "file": seg_file})
        print(f"[auto-resume] recovered {actual_dur:.1f}s from a leftover in-progress file "
              f"(likely from a crash or power loss) instead of re-encoding it.", file=sys.stderr)
        covered_end += actual_dur

    if segments:
        with open(checkpoint_path, "w") as f:
            json.dump({"segments": [{"start": s["start"], "file": s["file"]} for s in segments],
                      "fingerprint": fingerprint}, f)
        print(f"[auto-resume] resuming continuously from {covered_end:.1f}s "
              f"({len(segments)} segment(s) already on disk).", file=sys.stderr)

    # Process the rest of the video in ONE continuous pass — no scheduled chunk
    # boundaries. A new segment is only ever created here because we're picking up after
    # an interruption, not on a fixed timer, so a run that completes without incident
    # produces exactly one continuous file with zero internal seams.
    # Keep creating segments until the whole range is actually covered, or we're
    # genuinely interrupted. A single segment occasionally falls a little short of
    # effective_end even without being interrupted (e.g. decoder/EOF quirks near the
    # very end of a long file) — looping here means the job still finishes in this same
    # run instead of silently stopping short and reporting "Finished" on an incomplete file.
    interrupted = False
    stall_guard = 0
    while covered_end < effective_end - 0.5:
        seg_index = len(segments)
        seg_file = f"{base}_resume_seg_{seg_index:04d}{ext}"
        seg_args = copy.copy(args)
        seg_args.start_time = covered_end
        seg_args.end_time = effective_end
        seg_args.yes = True
        seg_args.resume = False
        seg_args.auto_resume = False
        seg_args.preserve_dowi = False  # DV/HDR10+ handled once at the end, over the whole range
        seg_args.stereo_mode_tag = False  # StereoMode tagging handled once at the end too

        process_video_full(input_filename, seg_file, seg_args, depth_model, side_model)

        actual_dur = _probe_video_duration(seg_file)
        new_covered_end = covered_end
        if actual_dur is not None and actual_dur > 0.5:
            segments.append({"start": covered_end, "end": covered_end + actual_dur, "file": seg_file})
            new_covered_end = covered_end + actual_dur

        with open(checkpoint_path, "w") as f:
            json.dump({"segments": [{"start": s["start"], "file": s["file"]} for s in segments],
                      "fingerprint": fingerprint}, f)

        if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
            # Stopped (cancel button or interruption) — checkpoint above already records
            # everything completed so far; next run picks up exactly here.
            interrupted = True
            break

        if new_covered_end <= covered_end + 0.5:
            # This attempt made no real progress. This usually means the source file's
            # own container metadata overstates its real duration (common with some
            # WEB-DL releases — the video/audio streams genuinely have no more frames
            # past this point even though the container header claims a longer runtime).
            # Treat this as having reached the real end of the file rather than retrying
            # forever or abandoning the job short of a target that can never be reached.
            stall_guard += 1
            if stall_guard >= 2:
                print(f"[auto-resume] source file has no more decodable frames past "
                      f"{covered_end:.1f}s (container metadata claims {effective_end:.1f}s) — "
                      f"treating {covered_end:.1f}s as the real end and finishing up.",
                      file=sys.stderr)
                break
        else:
            stall_guard = 0
        covered_end = new_covered_end

    if interrupted or not segments:
        return

    # Reaching here means either the full requested range was covered, or the source
    # genuinely has no more frames to give (see stall handling above) — either way,
    # what's in `segments` is as complete as this job is ever going to get, so merge it.

    segment_files = [s["file"] for s in segments]

    # All segments together now cover the full range — join them into the final output.
    # NOTE: each segment re-encodes its own short audio slice independently, and every
    # independent AAC encode adds its own small priming delay. Stitching those segment-local
    # audio tracks together (as a naive "-c copy" concat would) makes the delay grow by one
    # increment at every segment boundary. To avoid that, concat video-only here and mux in a
    # single audio track extracted once from the original source below.
    ffmpeg_bin = _get_ffmpeg_bin()
    concat_list = base + "_resume_concat.txt"
    video_only_filename = base + "_resume_video_only" + ext

    # Prefer mkvmerge to splice the segments together. Each segment was encoded by its own
    # fresh encoder instance, so the raw bitstreams each carry their own SPS/PPS headers;
    # ffmpeg's concat demuxer joins them at the byte level, which can make some players
    # briefly hitch at the seam. mkvmerge's "+" append is built specifically for splicing
    # encoded segments back together without that hiccup. With this continuous design, a
    # successful run produces only one segment, so there is usually nothing to splice at all.
    mkvmerge_bin = _find_mkvmerge() if ext.lower() == ".mkv" and len(segment_files) > 1 else None
    concat_ok = len(segment_files) == 1
    if concat_ok:
        os.replace(segment_files[0], video_only_filename)
    elif mkvmerge_bin:
        mkvmerge_args = [mkvmerge_bin, "-o", video_only_filename, "--no-audio", segment_files[0]]
        for cf in segment_files[1:]:
            mkvmerge_args += ["+", cf]
        result = subprocess.run(mkvmerge_args, capture_output=True)
        # mkvmerge exit code: 0 = ok, 1 = warnings (output still produced), 2 = error
        if result.returncode in (0, 1) and path.exists(video_only_filename):
            concat_ok = True
        else:
            print(f"[auto-resume] mkvmerge splice failed, falling back to ffmpeg concat: "
                  f"{result.stderr.decode(errors='replace').strip()}", file=sys.stderr)

    if not concat_ok:
        with open(concat_list, "w", encoding="utf-8") as f:
            for cf in segment_files:
                # ffmpeg concat demuxer uses single quotes as the path delimiter; a literal
                # single quote in the path (e.g. "Joe's Apartment") must be escaped as '\'' or
                # ffmpeg truncates the path right there and fails with "No such file or directory".
                f.write("file '{}'\n".format(cf.replace("'", "'\\''")))

        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_list, "-map", "0:v", "-c", "copy", "-an", video_only_filename],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"[auto-resume] concat failed: {e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
            return
        finally:
            if path.exists(concat_list):
                try:
                    os.remove(concat_list)
                except Exception:
                    pass

    for cf in segment_files:
        if path.exists(cf):
            try:
                os.remove(cf)
            except Exception:
                pass
    if path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass

    # Extract audio once, directly from the original source, over the full effective
    # range — a single clean encode instead of N stitched chunk-local encodes.
    _notify_stage(args, STAGE_AUDIO_EXTRACT)
    audio_tmp = base + "_resume_audio.m4a"
    has_audio = False
    try:
        has_audio = VU.export_audio(
            input_filename, audio_tmp,
            start_time=effective_start, end_time=effective_end,
            title=f"{path.basename(input_filename)} Audio",
            stop_event=args.state["stop_event"], suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
        )
    except Exception as e:
        print(f"[auto-resume] audio export failed: {e}", file=sys.stderr)
        has_audio = False

    try:
        if has_audio:
            # NOTE: -copyts is required here -- ffmpeg normally resets each input's start time
            # to 0 when muxing multiple separate input files together, which would silently
            # discard any deliberate leading gap export_audio() preserved for sources whose
            # audio elementary stream doesn't truly start at the video's pts=0 (see the
            # frame.pts handling in export_audio()).
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", video_only_filename, "-i", audio_tmp,
                 "-map", "0:v", "-map", "1:a", "-c", "copy", "-copyts", "-shortest", output_filename],
                check=True, capture_output=True,
            )
        else:
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", video_only_filename, "-map", "0:v", "-c", "copy", output_filename],
                check=True, capture_output=True,
            )
    except subprocess.CalledProcessError as e:
        print(f"[auto-resume] final mux failed: {e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
        return
    finally:
        for f in (video_only_filename, audio_tmp):
            if path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    # HDR metadata injection on the final concatenated output
    if getattr(args, "preserve_dowi", False) and path.exists(output_filename):
        fb = _get_ffmpeg_bin()
        out_dir = path.dirname(path.abspath(output_filename))
        hdr_types = _detect_hdr_types(input_filename, _find_ffprobe())
        rpu_path = path.join(out_dir, "_iw3_rpu.bin") if hdr_types["dv"] else None
        h10p_path = path.join(out_dir, "_iw3_hdr10plus.json") if hdr_types["hdr10plus"] else None
        dovi_b = _find_dovi_tool() if hdr_types["dv"] else None
        h10p_b = _find_hdr10plus_tool() if hdr_types["hdr10plus"] else None
        tmp_hevc = path.join(out_dir, "_iw3_src.hevc")

        if (rpu_path and dovi_b) or (h10p_path and h10p_b):
            try:
                # Limit extraction to the same [effective_start, effective_end] window that
                # was actually processed (e.g. when --start-time/--end-time trims the video),
                # so the RPU's frame order lines up with the output instead of starting from
                # frame 0 of the whole original file.
                trim_args = []
                if effective_start > 0:
                    trim_args += ["-ss", str(effective_start)]
                if duration and effective_end < duration:
                    trim_args += ["-to", str(effective_end)]
                subprocess.run(
                    [fb, "-y", *trim_args, "-i", str(input_filename),
                     "-c:v", "copy", "-an", "-f", "hevc", tmp_hevc],
                    check=True, capture_output=True,
                )
                if rpu_path and dovi_b:
                    subprocess.run(
                        [dovi_b, "extract-rpu", "-i", tmp_hevc, "-o", rpu_path],
                        check=True, capture_output=True,
                    )
                if h10p_path and h10p_b:
                    subprocess.run(
                        [h10p_b, "extract", "-i", tmp_hevc, "-o", h10p_path],
                        check=True, capture_output=True,
                    )
            except subprocess.CalledProcessError as e:
                print(f"--preserve-dowi: HDR extraction failed: "
                      f"{e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
                rpu_path = h10p_path = None
            finally:
                if path.exists(tmp_hevc):
                    try:
                        os.remove(tmp_hevc)
                    except Exception:
                        pass

            if rpu_path or h10p_path:
                _notify_stage(args, STAGE_HDR_REINJECT)
                try:
                    _inject_hdr_metadata(output_filename, rpu_path, h10p_path,
                                         fb, dovi_b, h10p_b, out_dir,
                                         configured_video_codec=args.video_codec)
                except Exception as e:
                    print(f"--preserve-dowi: HDR injection failed: {e}", file=sys.stderr)
                finally:
                    for f in filter(None, [rpu_path, h10p_path]):
                        try:
                            os.remove(f)
                        except Exception:
                            pass

    # MKV StereoMode tagging on the final concatenated output -- once here, over the
    # whole range, same reasoning as the HDR block above (see ADR-033).
    if path.exists(output_filename):
        _apply_stereo_mode_tag(output_filename, args)


def process_video_keyframes(input_filename, output_path, args, depth_model, side_model):
    assert depth_model.get_name() not in {"VideoDepthAnything", "VideoDepthAnythingStreaming"}

    if is_output_dir(output_path):
        os.makedirs(output_path, exist_ok=True)
        output_filename = path.join(
            output_path,
            make_output_filename(path.basename(input_filename), args, video=True))
    else:
        output_filename = output_path

    output_dir = path.join(path.dirname(output_filename), path.splitext(path.basename(output_filename))[0])
    if output_dir.endswith("_LRF"):
        output_dir = output_dir[:-4]
    os.makedirs(output_dir, exist_ok=True)

    max_workers = max(args.max_workers, 8)
    with PoolExecutor(max_workers=max_workers) as pool:
        futures = []

        def frame_callback(frame):
            if frame is None:
                return

            x = VU.to_tensor(frame, device=args.state["device"])
            output = process_image(x, args, depth_model, side_model)
            output = to_pil_image(output)
            output_filename = path.join(
                output_dir,
                path.basename(output_dir) + "_" + str(frame.pts).zfill(8) + FULL_SBS_SUFFIX + get_image_ext(args.format))
            f = pool.submit(save_image, output, output_filename, format=args.format)
            futures.append(f)
            if len(futures) > IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures.clear()

        VU.hook_frame(
            input_filename,
            frame_callback=frame_callback,
            keyframe_only=True,
            min_interval_sec=args.keyframe_interval,
            vf=args.vf,
            stop_event=args.state["stop_event"],
            suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
            title=_progress_title(path.basename(input_filename), args),
            start_time=args.start_time,
            end_time=args.end_time,
            device=args.state["device"],
            hwaccel=args.hwaccel,
            disable_software_fallback=args.disable_software_fallback,
        )
        for f in futures:
            f.result()


def process_video(input_filename, output_path, args, depth_model, side_model):
    # disable ema minmax for each process
    depth_model.reset()
    depth_model.disable_ema()

    # Explicit stage announcement (not just relying on GUI startup default) because
    # this function is called once per file in directory/batch mode -- without this,
    # a second file's Depth & Stereo Conversion pass would start while the GUI's
    # stage indicator was still left on "RIFE Frame Interpolation"/"Upscaling with
    # waifu2x" from the previous file.
    _notify_stage(args, STAGE_DEPTH_STEREO)

    input_filename, hdr_tmp_file = _tonemap_hdr_to_sdr(input_filename, args)
    input_filename, denoise_tmp_file = _denoise_preprocess(input_filename, args)
    try:
        if args.keyframe:
            if side_model is not None and hasattr(side_model, "set_mode"):
                side_model.set_mode("image")
                side_model.reset()
            if args.state["convergence_model"] is not None:
                args.state["convergence_model"].reset(enable_ema=False)

            process_video_keyframes(input_filename, output_path, args, depth_model, side_model)
        else:
            if side_model is not None and hasattr(side_model, "set_mode"):
                side_model.set_mode("video")
                side_model.reset()
            if args.state["convergence_model"] is not None:
                args.state["convergence_model"].reset(enable_ema=True)

            process_video_with_resume(input_filename, output_path, args, depth_model, side_model)
    finally:
        for tmp_file in (hdr_tmp_file, denoise_tmp_file):
            if tmp_file and path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    # Only on a genuine, uncancelled completion -- the functions above return early on
    # cancellation without raising, so a cancelled job would otherwise still reach here.
    stop_event = args.state.get("stop_event") if getattr(args, "state", None) else None
    if not (stop_event is not None and stop_event.is_set()):
        if _should_use_stereo_upscale(args):
            _run_waifu2x_upscale_stereo(output_path, args)
        else:
            _run_waifu2x_upscale(output_path, args)
        _run_rife_interpolation(output_path, args)


def export_images(input_path, output_dir, args, title=None):
    if path.isdir(input_path):
        files = ImageLoader.listdir(input_path)
        src_rgb_dir = path.normpath(path.abspath(input_path))
    else:
        assert is_image(input_path)
        files = [input_path]
        src_rgb_dir = path.normpath(path.abspath(path.dirname(input_path)))

    if not files:
        # no image files
        return

    if args.export_disparity:
        mapper = "none"
        edge_dilation = args.edge_dilation
        skip_edge_dilation = True
        skip_mapper = True
    else:
        mapper = args.mapper
        edge_dilation = 0
        skip_edge_dilation = False
        skip_mapper = False

    config = export_config.ExportConfig(
        type=export_config.IMAGE_TYPE,
        fps=1,
        mapper=mapper,
        skip_mapper=skip_mapper,
        skip_edge_dilation=skip_edge_dilation,
        user_data={
            "export_options": {
                "depth_model": args.depth_model,
                "export_disparity": args.export_disparity,
                "mapper": args.mapper,
                "edge_dilation": args.edge_dilation,
                "ema_normalize": False,
            }
        }
    )
    config.audio_file = None
    if args.export_depth_only:
        config.rgb_dir = src_rgb_dir
        rgb_dir = src_rgb_dir
    else:
        rgb_dir = path.join(output_dir, config.rgb_dir)
    depth_dir = path.join(output_dir, config.depth_dir)
    config_file = path.join(output_dir, export_config.FILENAME)

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    depth_model = args.state["depth_model"]
    depth_model.disable_ema()

    if args.resume:
        # skip existing depth files
        remaining_files = []
        existing_files = []
        for fn in files:
            basename = path.splitext(path.basename(fn))[0] + ".png"
            depth_file = path.join(depth_dir, basename)
            if not path.exists(depth_file):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)

        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        files = remaining_files

    loader = ImageLoader(
        files=files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb", "exif_transpose": not args.disable_exif_transpose})
    futures = []
    tqdm_fn = args.state["tqdm_fn"] or tqdm
    pbar = tqdm_fn(ncols=80, total=len(files), desc=title or "Images")
    stop_event = args.state["stop_event"]
    suspend_event = args.state["suspend_event"]

    max_workers = max(args.max_workers, 8)
    with PoolExecutor(max_workers=max_workers) as pool, torch.inference_mode():
        for im, meta in loader:
            basename = path.splitext(path.basename(meta["filename"]))[0] + ".png"
            rgb_file = path.join(rgb_dir, basename)
            depth_file = path.join(depth_dir, basename)
            if im is None:
                pbar.update(1)
                continue
            im = TF.to_tensor(im).to(args.state["device"])
            im = preprocess_image(im, args)
            depth = depth_model.infer(im, tta=args.tta, low_vram=args.low_vram,
                                      enable_amp=not args.disable_amp,
                                      edge_dilation=edge_dilation,
                                      depth_aa=args.depth_aa)
            depth = depth_model.minmax_normalize_chw(depth, rgb=im)
            if args.export_disparity:
                depth = get_mapper(args.mapper)(depth)
            if args.export_depth_fit:
                depth = F.interpolate(depth.unsqueeze(0), size=(im.shape[1], im.shape[2]),
                                      mode="bilinear", antialias=True, align_corners=True).squeeze(0)
            futures.append(pool.submit(depth_model.save_normalized_depth, depth, depth_file))

            if not args.export_depth_only:
                futures.append(pool.submit(save_image, to_pil_image(im), rgb_file))

            pbar.update(1)
            if suspend_event is not None:
                suspend_event.wait()
            if stop_event is not None and stop_event.is_set():
                break
            if len(futures) > IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures = []

        for f in futures:
            f.result()
    pbar.close()
    config.save(config_file)


def get_resume_seq(depth_dir, rgb_dir):
    last_seq = -1
    depth_files = sorted([path.basename(fn) for fn in ImageLoader.listdir(depth_dir)])
    if rgb_dir:
        rgb_files = sorted([path.basename(fn) for fn in ImageLoader.listdir(rgb_dir)])
        if rgb_files and depth_files:
            last_seq = int(path.splitext(min(rgb_files[-1], depth_files[-1]))[0], 10)
    else:
        if depth_files:
            last_seq = int(path.splitext(depth_files[-1])[0], 10)

    return last_seq


# export callbacks

_MOTION_ADAPTIVE_LOG_PATH = path.join(path.dirname(__file__), "..", "logs", "iw3_motion_adaptive_log.txt")


def _log_motion_activity(basename, entries):
    """Answers "when was Motion-Adaptive Smoothing actually used, how much, and on
    which part of the video" -- a real question with no other way to check, since the
    effective decay is computed and used per-frame but otherwise never surfaced. One
    line per segment (a scene, or the final tail of the export); frame numbers match
    the exported PNG filenames directly."""
    if not entries:
        return
    os.makedirs(path.dirname(_MOTION_ADAPTIVE_LOG_PATH), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_MOTION_ADAPTIVE_LOG_PATH, "a", encoding="utf-8") as f:
        for e in entries:
            pct_eased = 100.0 * e["eased_frames"] / e["frames"] if e["frames"] else 0.0
            f.write(
                f"{ts} | {basename} | frames {e['start_frame']:08d}-{e['end_frame']:08d} "
                f"({e['frames']} frames) | base_decay={e['base_decay']:.2f} "
                f"min_effective_decay={e['min_effective_decay']:.3f} "
                f"avg_motion_score={e['avg_motion_score']:.3f} "
                f"eased_on={pct_eased:.0f}% of frames\n"
            )


def bind_export_single_frame_callback(depth_model, segment_pts, rgb_dir, depth_dir, pool, args, basename=None):
    batch_queue = []
    pts_queue = []
    src_queue = []
    futures = []
    batch_size = 1 if args.low_vram else args.batch_size

    if args.export_disparity:
        edge_dilation = args.edge_dilation
    else:
        edge_dilation = 0

    def _postprocess(depths):
        for depth in depths:
            x, x_shape, pts = src_queue.pop(0)

            if args.export_disparity:
                depth = get_mapper(args.mapper)(depth)
            if args.export_depth_fit:
                depth = TF.resize(depth, size=(x_shape[-2], x_shape[-1]),
                                  interpolation=InterpolationMode.BILINEAR, antialias=True)
            seq = str(pts).zfill(8)
            futures.append(
                pool.submit(depth_model.save_normalized_depth, depth, path.join(depth_dir, f"{seq}.png"))
            )
            if not args.export_depth_only:
                assert x is not None
                im = Image.fromarray(x.cpu_buffer().permute(1, 2, 0).numpy())
                futures.append(
                    pool.submit(save_image, im, path.join(rgb_dir, f"{seq}.png"))
                )

            if len(futures) >= IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures.clear()

        return None

    def _batch_infer():
        x = torch.stack(batch_queue)
        reset_ema = [t in segment_pts for t in pts_queue]
        depth_batch = depth_model.infer(
            x,
            tta=args.tta,
            low_vram=args.low_vram,
            enable_amp=not args.disable_amp,
            edge_dilation=edge_dilation,
            depth_aa=args.depth_aa)
        depth_list = depth_model.minmax_normalize(depth_batch, reset_ema=reset_ema)
        if args.ema_motion_adaptive:
            _log_motion_activity(basename, depth_model.pop_motion_activity_log())

        pts_queue.clear()
        batch_queue.clear()

        return _postprocess(depth_list)

    @torch.inference_mode()
    def _frame_callback(frame):
        if frame is None:
            # flush
            if batch_queue:
                _batch_infer()
            result = _postprocess(depth_model.flush_minmax_normalize())
            if args.ema_motion_adaptive:
                _log_motion_activity(basename, depth_model.pop_motion_activity_log())
            return result

        x = VU.to_tensor(frame, device=args.state["device"])
        batch_queue.append(x)
        pts_queue.append(frame.pts)
        if args.export_depth_only:
            src_queue.append((None, x.shape, frame.pts))
        else:
            src_queue.append((VU.OffloadFrame(x, dtype=torch.uint8), x.shape, frame.pts))

        if len(batch_queue) == batch_size:
            _batch_infer()

    return _frame_callback


def bind_export_vda_frame_callback(depth_model, segment_pts, rgb_dir, depth_dir, pool, args):
    src_queue = []
    batch_queue = []
    pts_queue = []
    futures = []

    depth_model.reset()
    if args.export_disparity:
        edge_dilation = args.edge_dilation
    else:
        edge_dilation = 0

    def _postprocess(depth_list):
        for depth in depth_list:
            x, x_shape, pts = src_queue.pop(0)

            if args.export_disparity:
                depth = get_mapper(args.mapper)(depth)
            if args.export_depth_fit:
                depth = TF.resize(depth, size=(x_shape[-2], x_shape[-1]),
                                  interpolation=InterpolationMode.BILINEAR, antialias=True)
            seq = str(pts).zfill(8)
            futures.append(
                pool.submit(depth_model.save_normalized_depth, depth, path.join(depth_dir, f"{seq}.png"))
            )
            if not args.export_depth_only:
                assert x is not None
                im = Image.fromarray(x.cpu_buffer().permute(1, 2, 0).numpy())
                futures.append(
                    pool.submit(save_image, im, path.join(rgb_dir, f"{seq}.png"))
                )

            if len(futures) >= IMAGE_IO_QUEUE_MAX:
                for f in futures:
                    f.result()
                futures.clear()

        return None

    def _batch_infer():
        x = torch.stack(batch_queue)

        depth_list = depth_model.infer_with_normalize(
            x, pts_queue, segment_pts,
            enable_amp=not args.disable_amp,
            edge_dilation=edge_dilation,
            depth_aa=args.depth_aa,
            tta=args.tta)

        pts_queue.clear()
        batch_queue.clear()

        return _postprocess(depth_list)

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            # flush
            if batch_queue:
                _batch_infer()
            depth_list = depth_model.flush_with_normalize(
                enable_amp=not args.disable_amp,
                edge_dilation=edge_dilation,
                depth_aa=args.depth_aa)
            _postprocess(depth_list)
        else:
            x = VU.to_tensor(frame, device=args.state["device"])
            batch_queue.append(x)
            pts_queue.append(frame.pts)
            if args.export_depth_only:
                src_queue.append((None, x.shape, frame.pts))
            else:
                src_queue.append((VU.OffloadFrame(x, dtype=torch.uint8), x.shape, frame.pts))

            if len(batch_queue) == args.batch_size:
                _batch_infer()

    return frame_callback


def export_video(input_filename, output_dir, args, title=None):
    basename = path.splitext(path.basename(input_filename))[0]
    title = title or _progress_title(path.basename(input_filename), args)
    if args.export_disparity:
        mapper = "none"
        skip_edge_dilation = True
        skip_mapper = True
    else:
        mapper = args.mapper
        skip_edge_dilation = False
        skip_mapper = False
    config = export_config.ExportConfig(
        type=export_config.VIDEO_TYPE,
        basename=basename,
        mapper=mapper,
        skip_mapper=skip_mapper,
        skip_edge_dilation=skip_edge_dilation,
        user_data={
            "export_options": {
                "depth_model": args.depth_model,
                "export_disparity": args.export_disparity,
                "export_depth_only": args.export_depth_only,
                "mapper": args.mapper,
                "edge_dilation": args.edge_dilation,
                "depth_aa": args.depth_aa,
                "max_fps": args.max_fps,
                "ema_normalize": args.ema_normalize,
                "ema_decay": args.ema_decay,
                "ema_buffer": args.ema_buffer,
                "scene_detect": args.scene_detect,
            }
        }
    )
    # NOTE: Windows does not allow creating folders with trailing spaces. basename.strip()
    output_dir = path.join(output_dir, basename.strip())
    rgb_dir = path.join(output_dir, config.rgb_dir)
    depth_dir = path.join(output_dir, config.depth_dir)
    audio_file = path.join(output_dir, config.audio_file)
    config_file = path.join(output_dir, export_config.FILENAME)

    if not args.resume and (not args.yes and path.exists(config_file)):
        y = input(f"File '{config_file}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    if args.export_depth_only:
        rgb_dir = None
        config.rgb_dir = None
    else:
        os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    if args.scene_detect or args.scene_detect_only:
        segment_pts = None
        scan_start_time, scan_end_time = args.start_time, args.end_time
        if not args.disable_scene_cache:
            segment_pts = try_load_scene_cache(input_filename, args)
            if segment_pts is None:
                scan_start_time, scan_end_time = widen_scene_scan_range(input_filename, args)
        if segment_pts is None:
            with TorchHubDir(HUB_MODEL_DIR):
                segment_pts = SBD.detect_boundary(
                    input_filename,
                    max_fps=args.max_fps,
                    device=args.state["device"],
                    hwaccel=args.hwaccel,
                    disable_software_fallback=args.disable_software_fallback,
                    start_time=scan_start_time,
                    end_time=scan_end_time,
                    stop_event=args.state["stop_event"],
                    suspend_event=args.state["suspend_event"],
                    tqdm_fn=args.state["tqdm_fn"],
                    tqdm_title=f"{path.basename(input_filename)}: Scene Boundary Detection",
                )
                # See should_save_scene_cache: must not cache a scan that got cancelled
                # partway through under metadata claiming the full requested range, and
                # must be checked before (not after) the stop_event early-return below.
                if should_save_scene_cache(args.disable_scene_cache, False, args.state["stop_event"]):
                    save_scene_cache(input_filename, segment_pts, args,
                                      start_time=scan_start_time, end_time=scan_end_time)
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    return
            gc_collect()
    else:
        segment_pts = set()
    if args.scene_detect_only:
        return

    if args.autocrop is not None:
        crop = AutoCrop.from_video_file(
            input_filename,
            mode=args.autocrop,
            uncrop_enabled=False,
            vf=args.vf,
            hwaccel=args.hwaccel,
            disable_software_fallback=args.disable_software_fallback,
            device=args.state["device"],
            batch_size=args.batch_size,
            stop_event=args.state["stop_event"],
            suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
            tqdm_title=f"{path.basename(input_filename)}: AutoCrop Analysis",
        ).get_crop()
        crop_filter = f"crop=x={crop[0]}:y={crop[1]}:w={crop[2]}:h={crop[3]}" if crop is not None else None
        base_vf = (args.vf + f",{crop_filter}") if (crop_filter and args.vf) else (crop_filter or args.vf)
    else:
        base_vf = args.vf

    config.user_data["scene_boundary"] = ",".join([str(pts).zfill(8) for pts in sorted(list(segment_pts))])

    if path.exists(audio_file):
        # Already there -- whether from a genuine resume or because a caller (Dual-
        # Pass Depth Blend, extracting audio as its own visible step) already pulled
        # it out ahead of time. Re-extracting identical audio for the same input/range
        # would only waste time, never produce a different result, so simple
        # existence is enough here regardless of resume state.
        has_audio = True
    else:
        if args.export_depth_only:
            has_audio = False
        else:
            has_audio = VU.export_audio(input_filename, audio_file,
                                        start_time=args.start_time, end_time=args.end_time,
                                        title=f"{title} Audio",
                                        stop_event=args.state["stop_event"], suspend_event=args.state["suspend_event"],
                                        tqdm_fn=args.state["tqdm_fn"])
    if not has_audio:
        config.audio_file = None

    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
        return

    # Integrate preprocess_image() logic into vf
    video_filter = add_preprocess_vf(base_vf, args)

    def config_callback(metadata):
        fps = metadata.get_fps()
        if float(fps) > args.max_fps:
            fps = args.max_fps
        config.fps = fps  # update fps
        pix_fmt = args.pix_fmt
        upgrade = getattr(args, "upgrade_pix_fmt", None)
        if upgrade and not metadata.use_16bit:
            pix_fmt = _upgrade_pix_fmt(pix_fmt, upgrade)
            pix_fmt = _clamp_pix_fmt_for_codec(pix_fmt, args.video_codec)

        def state_update_callback(c):
            config.output_colorspace = c.output_colorspace
            config.output_color_primaries = c.output_color_primaries
            config.output_color_trc = c.output_color_trc
            config.source_color_range = c.source_color_range

        video_output_config = VU.VideoOutputConfig(fps=fps, pix_fmt=pix_fmt, colorspace=args.colorspace)
        video_output_config.state_updated = state_update_callback

        return video_output_config

    depth_model = args.state["depth_model"]
    depth_model.reset()
    depth_model.disable_ema()
    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if ema_normalize:
        depth_model.enable_ema(decay=args.ema_decay, buffer_size=args.ema_buffer,
                                motion_adaptive=getattr(args, "ema_motion_adaptive", False))

    max_workers = max(args.max_workers, 8)
    with depth_model.compile_context(enabled=args.compile), PoolExecutor(max_workers=max_workers) as pool:
        if args.state["depth_model"].get_name() == "VideoDepthAnything":
            frame_callback = bind_export_vda_frame_callback(
                depth_model=depth_model,
                segment_pts=segment_pts,
                rgb_dir=rgb_dir,
                depth_dir=depth_dir,
                pool=pool,
                args=args,
            )
        else:
            frame_callback = bind_export_single_frame_callback(
                depth_model=depth_model,
                segment_pts=segment_pts,
                rgb_dir=rgb_dir,
                depth_dir=depth_dir,
                pool=pool,
                args=args,
                basename=basename,
            )

        VU.hook_frame(
            input_filename,
            config_callback=config_callback,
            frame_callback=frame_callback,
            vf=video_filter,
            stop_event=args.state["stop_event"],
            suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
            title=title,
            start_time=args.start_time,
            end_time=args.end_time,
            device=args.state["device"],
            hwaccel=args.hwaccel,
            disable_software_fallback=args.disable_software_fallback
        )
    config.save(config_file)


def process_config_video(config, args, side_model):
    base_dir = path.dirname(args.input)
    rgb_dir, depth_dir, audio_file = config.resolve_paths(base_dir)
    if side_model is not None and hasattr(side_model, "set_mode"):
        side_model.set_mode("video")
        side_model.reset()
    if args.state["convergence_model"] is not None:
        args.state["convergence_model"].reset(enable_ema=True)

    if is_output_dir(args.output):
        os.makedirs(args.output, exist_ok=True)
        basename = config.basename or path.basename(base_dir)
        output_filename = path.join(
            args.output,
            make_output_filename(basename, args, video=True))
    else:
        output_filename = args.output
    make_parent_dir(output_filename)

    if args.resume and path.exists(output_filename):
        return
    if not args.yes and path.exists(output_filename):
        y = input(f"File '{output_filename}' already exists. Overwrite? [y/N]").lower()
        if y not in {"y", "ye", "yes"}:
            return

    rgb_files = ImageLoader.listdir(rgb_dir)
    depth_files = ImageLoader.listdir(depth_dir)
    if len(rgb_files) != len(depth_files):
        raise ValueError(f"No match rgb_files={len(rgb_files)} and depth_files={len(depth_files)}")
    if len(rgb_files) == 0:
        raise ValueError(f"{rgb_dir} is empty")

    if "scene_boundary" in config.user_data and isinstance(config.user_data["scene_boundary"], str):
        segment_pts = set(config.user_data["scene_boundary"].split(","))
    else:
        segment_pts = set()

    rgb_loader = ImageLoader(
        files=rgb_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    depth_loader = ImageLoader(
        files=depth_files,
        load_func=BaseDepthModel.load_depth)
    sbs_lock = threading.Lock()

    @torch.inference_mode()
    def batch_callback(x, depths, reset_pts, test=False):
        if not config.skip_edge_dilation and edge_dilation_is_enabled(args.edge_dilation):
            # apply --edge-dilation
            depths = -dilate_edge(-depths, args.edge_dilation)
        with sbs_lock:
            if test:
                assert x.shape[0] == 1
                while True:
                    left_eyes, right_eyes = apply_divergence(depths, x, args, side_model, reset_pts=reset_pts)
                    if left_eyes is not None:
                        break
                left_eyes = left_eyes[0:1]
                right_eyes = right_eyes[0:1]
            else:
                left_eyes, right_eyes = apply_divergence(depths, x, args, side_model, reset_pts=reset_pts)

        if left_eyes is not None:
            for i in range(left_eyes.shape[0]):
                yield postprocess_image(left_eyes[i], right_eyes[i], args)

    def test_output_size(rgb_file, depth_file):
        rgb = load_image_simple(rgb_file, color="rgb")[0]
        depth = BaseDepthModel.load_depth(depth_file)[0].to(args.state["device"])
        rgb = TF.to_tensor(rgb).to(args.state["device"])
        frame = next(batch_callback(rgb.unsqueeze(0), depth.unsqueeze(0), [False], test=True))
        if side_model is not None and hasattr(side_model, "reset"):
            side_model.reset()
        if args.state["convergence_model"] is not None:
            args.state["convergence_model"].reset()

        return frame.shape[-2:]

    minibatch_size = args.batch_size // 2 or 1 if args.tta else args.batch_size

    def generator():
        rgb_batch = []
        depth_batch = []
        reset_pts_batch = []
        for rgb, depth in zip(rgb_loader, depth_loader):
            rgb = TF.to_tensor(rgb[0])
            rgb_batch.append(rgb)
            depth_batch.append(depth[0])
            depth_basename = path.splitext(path.basename(depth[1]["filename"]))[0]
            reset_pts_batch.append(depth_basename in segment_pts)
            if len(rgb_batch) == minibatch_size:
                yield from batch_callback(torch.stack(rgb_batch).to(args.state["device"]),
                                          torch.stack(depth_batch).to(args.state["device"]),
                                          reset_pts_batch)
                rgb_batch.clear()
                depth_batch.clear()
                reset_pts_batch.clear()

        if rgb_batch:
            yield from batch_callback(torch.stack(rgb_batch).to(args.state["device"]),
                                      torch.stack(depth_batch).to(args.state["device"]),
                                      reset_pts_batch)
            rgb_batch.clear()
            depth_batch.clear()
            reset_pts_batch.clear()

    output_height, output_width = test_output_size(rgb_files[0], depth_files[0])
    if side_model is not None and hasattr(side_model, "reset"):
        side_model.reset()

    extra_meta = {}
    comment = _build_iw3_comment_metadata(args)
    if comment:
        extra_meta["comment"] = comment

    video_config = VU.VideoOutputConfig(
        fps=config.fps,  # use config.fps, ignore args.max_fps
        container_format=args.video_format,
        video_codec=args.video_codec,
        pix_fmt=args.pix_fmt,
        colorspace=args.colorspace,
        options=make_video_codec_option(args),
        container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
        output_width=output_width,
        output_height=output_height,
        metadata=extra_meta,
    )
    video_config.output_colorspace = config.output_colorspace
    video_config.output_color_trc = config.output_color_trc
    video_config.output_color_primaries = config.output_color_primaries
    video_config.source_color_range = config.source_color_range

    original_mapper = args.mapper
    try:
        if config.skip_mapper:
            # force use "none" mapper
            args.mapper = "none"
        else:
            if config.mapper is not None:
                # use specified mapper from config
                # NOTE: override args
                args.mapper = config.mapper
            else:
                # when config.mapper is not defined, use args.mapper
                # TODO: It can be still disputable
                pass
        VU.generate_video(
            output_filename,
            generator,
            config=video_config,
            audio_file=audio_file,
            title=_progress_title(path.basename(base_dir), args),
            total_frames=len(rgb_files),
            stop_event=args.state["stop_event"],
            suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
            device=args.state["device"],
        )
    finally:
        args.mapper = original_mapper


def process_config_images(config, args, side_model):
    base_dir = path.dirname(args.input)
    rgb_dir, depth_dir, _ = config.resolve_paths(base_dir)
    if side_model is not None and hasattr(side_model, "set_mode"):
        side_model.set_mode("image")
        side_model.reset()
    if args.state["convergence_model"] is not None:
        args.state["convergence_model"].reset(enable_ema=False)

    def fix_rgb_depth_pair(rgb_files, depth_files):
        rgb_db = {path.splitext(path.basename(fn))[0]: fn for fn in rgb_files}
        depth_db = {path.splitext(path.basename(fn))[0]: fn for fn in depth_files}
        and_keys = sorted(list(rgb_db.keys() & depth_db.keys()))
        rgb_files = [rgb_db[key] for key in and_keys if key in rgb_db]
        depth_files = [depth_db[key] for key in and_keys if key in depth_db]
        return rgb_files, depth_files

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    rgb_files = ImageLoader.listdir(rgb_dir)
    depth_files = ImageLoader.listdir(depth_dir)

    if args.resume:
        # skip existing output files
        remaining_files = []
        existing_files = []
        for fn in rgb_files:
            output_filename = path.join(
                output_dir,
                make_output_filename(path.basename(fn), args, video=False))
            if not path.exists(output_filename):
                remaining_files.append(fn)
            else:
                existing_files.append(fn)
        if existing_files:
            # The last file may be corrupt, so process it again
            remaining_files.insert(0, existing_files[0])
        rgb_files = remaining_files

    rgb_files, depth_files = fix_rgb_depth_pair(rgb_files, depth_files)

    if len(rgb_files) != len(depth_files):
        raise ValueError(f"No match rgb_files={len(rgb_files)} and depth_files={len(depth_files)}")
    if len(rgb_files) == 0:
        raise ValueError(f"{rgb_dir} is empty")

    rgb_loader = ImageLoader(
        files=rgb_files,
        load_func=load_image_simple,
        load_func_kwargs={"color": "rgb"})
    depth_loader = ImageLoader(
        files=depth_files,
        load_func=BaseDepthModel.load_depth)

    original_mapper = args.mapper
    try:
        if config.skip_mapper:
            args.mapper = "none"
        else:
            if config.mapper is not None:
                args.mapper = config.mapper
            else:
                pass
        with PoolExecutor(max_workers=4) as pool:
            tqdm_fn = args.state["tqdm_fn"] or tqdm
            pbar = tqdm_fn(ncols=80, total=len(rgb_files), desc="Images")
            stop_event = args.state["stop_event"]
            suspend_event = args.state["suspend_event"]
            futures = []
            for (rgb, rgb_meta), (depth, depth_meta) in zip(rgb_loader, depth_loader):
                rgb_filename = path.splitext(path.basename(rgb_meta["filename"]))[0]
                depth_filename = path.splitext(path.basename(depth_meta["filename"]))[0]
                if rgb_filename != depth_filename:
                    raise ValueError(f"No match {rgb_filename} and {depth_filename}")
                rgb = TF.to_tensor(rgb).to(args.state["device"])
                depth = depth.to(args.state["device"])
                if not config.skip_edge_dilation and edge_dilation_is_enabled(args.edge_dilation):
                    depth = -dilate_edge(-depth.unsqueeze(0), args.edge_dilation).squeeze(0)

                left_eye, right_eye = apply_divergence(depth, rgb, args, side_model)
                sbs = postprocess_image(left_eye, right_eye, args)
                sbs = to_pil_image(sbs)

                output_filename = path.join(
                    output_dir,
                    make_output_filename(rgb_filename, args, video=False))
                f = pool.submit(save_image, sbs, output_filename, format=args.format)
                futures.append(f)
                pbar.update(1)
                if suspend_event is not None:
                    suspend_event.wait()
                if stop_event is not None and stop_event.is_set():
                    break
                if len(futures) > IMAGE_IO_QUEUE_MAX:
                    for f in futures:
                        f.result()
                    futures = []
            for f in futures:
                f.result()
            pbar.close()
    finally:
        args.mapper = original_mapper


def create_parser(required_true=True):
    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, other):
            return self.start <= other <= self.end

        def __repr__(self):
            return f"{self.start} <= value <= {self.end}"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if torch.cuda.is_available() or mps_is_available() or xpu_is_available():
        default_gpu = 0
    else:
        default_gpu = -1

    parser.add_argument("--input", "-i", type=str, required=required_true,
                        help="input file or directory")
    parser.add_argument("--output", "-o", type=str, required=required_true,
                        help="output file or directory")
    parser.add_argument("--gpu", "-g", type=int, nargs="+", default=[default_gpu],
                        help="GPU device id. -1 for CPU")
    parser.add_argument("--compile", action="store_true", help="compile model if possible")
    parser.add_argument("--method", type=str, default="row_flow",
                        choices=["grid_sample", "backward",
                                 "monobw", "monobw_inpaint",
                                 "forward", "forward_fill", "forward_splat_fill", "forward_inpaint",
                                 "mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s",
                                 "mask_mlbw_l2", "mlbw_l2_inpaint",
                                 "row_flow", "row_flow_sym",
                                 "row_flow_v3", "row_flow_v3_sym",
                                 "row_flow_v2",
                                 "NULL"],
                        help="left-right divergence method")
    parser.add_argument("--splat-blend-temperature", type=float, default=SPLAT_BLEND_TEMPERATURE,
                        help=("only used by --method forward_splat_fill: how sharply that method's "
                              "depth-weighted collision blend favors the nearer of two colliding source "
                              "pixels. Higher = sharper cutoff, closer to forward_fill's old hard-overwrite "
                              "behavior (whichever pixel is nearer wins almost completely); lower = "
                              "smoother/more even blending between competing pixels. Default (50.0) matches "
                              "this method's original fixed behavior exactly. New setting, not yet tuned "
                              "against real footage -- start at the default and only adjust it if "
                              "forward_splat_fill's results look wrong at that default."))
    parser.add_argument("--synthetic-view", type=str, default="both", choices=["both", "right", "left"],
                        help=("the side that generates synthetic view."
                              "when `right`, the left view will be the original input image/frame"
                              " and only the right will be synthesized."))
    parser.add_argument("--preserve-screen-border", action="store_true",
                        help=("force set screen border parallax to zero"))
    parser.add_argument("--divergence", "-d", type=float, default=2.0,
                        help=("strength of 3D effect. 0-2 is reasonable value"))
    parser.add_argument("--warp-steps", type=int, help=("warp steps for row_flow_v3"))
    parser.add_argument("--convergence", "-c", type=float, default=0.5,
                        help=("(normalized) distance of convergence plane(screen position). 0-1 is reasonable value"))
    parser.add_argument("--convergence-mode", type=str, choices=["constant", "sod_v1", "face_detect"], default="constant",
                        help=("auto convergence mode"))
    parser.add_argument("--convergence-smoothing", type=float, default=0.9,
                        help=("EMA decay for auto convergence modes (sod_v1/face_detect). "
                              "Higher = smoother but slower to react. Lower = more aggressive/dynamic. "
                              "0 = no smoothing"))
    parser.add_argument("--update", action="store_true",
                        help="force update midas models from torch hub")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="process all subdirectories")
    parser.add_argument("--resume", action="store_true",
                        help="skip processing when the output file already exists")
    parser.add_argument("--skip-error", action="store_true",
                        help="continue processing even if an error occurs for a specific file during batch processing.")
    parser.add_argument("--batch-size", type=int, default=2, choices=[Range(1, 64)],
                        help="batch size. ignored when --low-vram")
    parser.add_argument("--max-fps", type=float, default=30,
                        help="max framerate for video. output fps = min(fps, --max-fps)")
    parser.add_argument("--profile-level", type=str, help="h264 profile level")
    parser.add_argument("--crf", type=int, default=20,
                        help="constant quality value for video. smaller value is higher quality")
    parser.add_argument("--video-bitrate", type=str, default="8M",
                        help="bitrate option for libopenh264")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                                 "medium", "slow", "slower", "veryslow", "placebo",
                                 "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                        help="encoder preset option for video")
    parser.add_argument("--tune", type=str, nargs="+", default=[],
                        # libx264: film, animation, grain, stillimage, psnr, fastdecode, zerolatency
                        # libx265: grain, animation, psnr, fastdecode, zerolatency
                        # NVENC (h264_nvenc/hevc_nvenc): hq, uhq (hevc_nvenc only), ll, ull, lossless
                        # kept in sync with TUNE_LIBX264/TUNE_LIBX265/TUNE_NVENC_HEVC in
                        # nunif/gui/video_encoding_box.py, the GUI's Tune dropdown
                        choices=["film", "animation", "grain", "stillimage", "psnr",
                                 "fastdecode", "zerolatency",
                                 "hq", "uhq", "ll", "ull", "lossless"],
                        help="encoder tunings option for video")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="overwrite output files")
    parser.add_argument("--pad", type=float, help="pad_size = round(width * pad) // 2")
    parser.add_argument("--pad-mode", type=str, default="tblr", choices=["tblr", "tb", "lr", "16:9", "top"], help="padding mode")
    parser.add_argument("--depth-model", type=str, default="ZoeD_Any_N",
                        choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK",
                                 "Any_S", "Any_B", "Any_L",
                                 "ZoeD_Any_N", "ZoeD_Any_K",
                                 "Any_V2_S", "Any_V2_B", "Any_V2_L",
                                 "Any_V2_N", "Any_V2_K",
                                 "Any_V2_N_S", "Any_V2_N_B", "Any_V2_N_L",
                                 "Any_V2_K_S", "Any_V2_K_B", "Any_V2_K_L",
                                 "Distill_Any_S", "Distill_Any_B", "Distill_Any_L",
                                 "Any_V3_Mono", "Any_V3_Mono_01",
                                 "DepthPro", "DepthPro_S",
                                 "VDA_S", "VDA_B", "VDA_L",
                                 "VDA_Metric", "VDA_Metric_S", "VDA_Metric_B", "VDA_Metric_L",
                                 "VDA_Stream_S", "VDA_Stream_B", "VDA_Stream_L",
                                 "VDA_Stream_Metric_S", "VDA_Stream_Metric_B", "VDA_Stream_Metric_L",
                                 "NULL",
                                 ],
                        help="depth model name")
    parser.add_argument("--remove-bg", action="store_true",
                        help="remove background depth, not recommended for video (DELETED)")
    parser.add_argument("--bg-model", type=str, default="u2net_human_seg",
                        help="rembg model type")
    parser.add_argument("--rotate-left", action="store_true",
                        help="Rotate 90 degrees to the left(counterclockwise)")
    parser.add_argument("--disable-exif-transpose", action="store_true",
                        help="Disable EXIF orientation transpose")
    parser.add_argument("--rotate-right", action="store_true",
                        help="Rotate 90 degrees to the right(clockwise)")
    parser.add_argument("--low-vram", action="store_true",
                        help="disable batch processing for low memory GPU")
    parser.add_argument("--pause-frees-vram", action="store_true",
                        help=("when Suspend/Resume is used (GUI only), also move every loaded model "
                              "(depth model, stereo/side model, SOD_v1 auto-convergence model) off the "
                              "GPU and release the freed VRAM back to the OS while paused, then reload "
                              "them on Resume. Off by default: pausing keeps models resident on the GPU "
                              "exactly as before, for an instant Resume. Turning this on frees VRAM while "
                              "paused (so other GPU work can run) at the cost of a few seconds of extra "
                              "reload time on Resume. Has no effect with multi-GPU (--gpu with more than "
                              "one id): those models are already replicated across every configured "
                              "device and are left resident."))
    parser.add_argument("--keyframe", action="store_true",
                        help="process only keyframe as image")
    parser.add_argument("--keyframe-interval", type=float, default=4.0,
                        help="keyframe minimum interval (sec)")
    parser.add_argument("--vf", type=str, default="",
                        help="video filter options for ffmpeg.")
    parser.add_argument("--debug-depth", action="store_true",
                        help="debug output normalized depthmap, info and preprocessed depth")
    parser.add_argument("--export", action="store_true", help="export depth, frame, audio")
    parser.add_argument("--export-disparity", action="store_true",
                        help=("export dispary instead of depth. "
                              "this means applying --mapper and --foreground-scale."))
    parser.add_argument("--export-depth-only", action="store_true",
                        help=("output only depth image and omits rgb image"))
    parser.add_argument("--export-depth-fit", action="store_true",
                        help=("fit depth image size to rgb image"))
    parser.add_argument("--mapper", type=str,
                        choices=MAPPER_ALL,
                        help=("(re-)mapper function for depth. "
                              "if auto, div_6 for ZoeDepth model, none for DepthAnything/DepthPro model. "
                              "directly using this option is not recommended. "
                              "use --foreground-scale instead."))
    parser.add_argument("--foreground-pop", type=float, default=0.0,
                        help="push the nearest pixels even further toward the audience (0.0=off, 0.5=medium, 1.0=strong)")
    parser.add_argument("--foreground-divergence", type=float, default=None,
                        help="use a separate effective Divergence value for the nearest 15%% of pixels only "
                             "(same units/range as --divergence). unset = disabled, uses the same Divergence as "
                             "the rest of the scene")
    parser.add_argument("--background-pop", type=float, default=0.0,
                        help="push the farthest pixels even further away from the audience (0.0=off, 0.5=medium, 1.0=strong)")
    parser.add_argument("--background-pop-coverage", type=float, default=0.15,
                        help="how much of the scene Background Pop treats as \"background\" (0.0-1.0, e.g. "
                             "0.25 = farthest 25%%). Default 0.15 = farthest 15%%. Larger values affect more of "
                             "the scene but move the transition line further into the midground")
    parser.add_argument("--background-divergence", type=float, default=None,
                        help="use a separate effective Divergence value for the farthest 15%% of pixels only "
                             "(same units/range as --divergence). unset = disabled, uses the same Divergence as "
                             "the rest of the scene")
    parser.add_argument("--edge-repair-strength", type=float, default=0.0,
                        help=("final cleanup pass on the RENDERED stereo output (after whichever stereo "
                              "method made it), gently smoothing only a thin band right around real depth "
                              "edges to reduce hairline fringing/ghosting residue. 0.0=off (default, zero "
                              "cost), 1.0=strongest. Cannot affect flat regions or areas with no depth edge."))
    parser.add_argument("--sharpen", action="store_true",
                        help=("final unsharp-mask sharpening pass on the RENDERED stereo output (after "
                              "whichever stereo method made it, and after --edge-repair-strength if that is "
                              "also set). Edge-aware: boosted at real local detail/edges, tapered to near "
                              "zero in flat or grain-only areas, so it does not aggressively amplify film "
                              "grain/sensor noise the way a plain unsharp mask would. Off by default. Use "
                              "--sharpen-strength to control the amount."))
    parser.add_argument("--sharpen-strength", type=float, default=0.5,
                        help=("how strong the --sharpen unsharp-mask pass is (0.0=no effect, 1.0=strongest). "
                              "Default 0.5. Has no effect unless --sharpen is also set."))
    parser.add_argument("--rife-interpolate", action="store_true",
                        help=("after conversion finishes, run the finished PACKED stereo output (both eyes "
                              "already combined, e.g. Half-SBS) through RIFE frame interpolation as a "
                              "separate post-processing step, raising the effective frame rate (2x by "
                              "default -- see --rife-multiplier/--rife-target-fps for 3x/4x or an exact "
                              "target fps). Written to a separate '<name>_rife<ext>' file -- the original "
                              "conversion output is never modified. Cannot be combined with --preserve-dowi "
                              "(no correct DV/HDR10+ metadata can be assigned to RIFE's synthetic "
                              "in-between frames)."))
    parser.add_argument("--rife-model", type=str, default="rife_425",
                        choices=["rife_425", "rife_425_lite"],
                        help=("which RIFE model tier to use for --rife-interpolate. rife_425 (default) is "
                              "the recommended full-quality model; rife_425_lite is a lower-compute-cost "
                              "variant. Weights are downloaded on first use if not already present."))
    parser.add_argument("--rife-multiplier", type=int, default=None, choices=[2, 3, 4],
                        help=("frame-rate multiplier for --rife-interpolate: inserts N-1 evenly-spaced "
                              "synthetic frames between every real frame pair (N=2 doubles the frame rate, "
                              "N=3/N=4 triple/quadruple it at proportionally higher processing cost). "
                              "Mutually exclusive with --rife-target-fps. Defaults to 2 (the original "
                              "doubling behavior) when neither is given."))
    parser.add_argument("--rife-target-fps", type=float, default=None,
                        help=("interpolate --rife-interpolate's output to this exact frame rate instead of "
                              "a simple multiplier (e.g. 60 to go from a 24fps source to 60fps). Must be "
                              "higher than the source's own frame rate. Mutually exclusive with "
                              "--rife-multiplier."))
    parser.add_argument("--waifu2x-upscale", action="store_true",
                        help=("after conversion finishes, run the finished output through waifu2x (a "
                              "separate, dedicated AI upscaler bundled with this app) as one extra step. "
                              "Written to a separate '<name>_w2x<ext>' file -- the original conversion "
                              "output is never modified. See --waifu2x-method/--waifu2x-noise-level/"
                              "--waifu2x-style for the plain whole-frame upscale's own settings (GUI-only "
                              "controls historically; use waifu2x.cli directly for full control from the "
                              "CLI). --waifu2x-upscale-target additionally selects a stereo-aware per-eye "
                              "upscale path on a packed 3D video output."))
    parser.add_argument("--waifu2x-upscale-target", type=str, default="auto",
                        choices=["auto", "4k", "8k"],
                        help=("Only takes effect together with --waifu2x-upscale on a packed two-eye "
                              "stereo video output (Half/Full SBS, Half/Full TB, Cross-Eyed, VR180). "
                              "'auto' (default) leaves --waifu2x-upscale's plain whole-frame behavior "
                              "completely unchanged (fixed 2x/4x per --waifu2x-method). '4k'/'8k' instead "
                              "run a stereo-aware path (iw3.waifu2x_upscale_stereo_cli): split the packed "
                              "frame into its two independent eye images, upscale each eye independently, "
                              "apply RGB temporal smoothing to each eye's own frame sequence to reduce "
                              "flicker at very high output resolutions, then recombine at the requested "
                              "FINAL PACKED width (3840 for 4k, 7680 for 8k) -- the actual per-eye scale "
                              "factor needed is computed from your source's real resolution, not assumed."))
    parser.add_argument("--foreground-scale", type=float, choices=[Range(-3.0, 3.0)], default=0,
                        help="foreground scaling level. 0 is disabled")
    parser.add_argument("--mapper-type", type=str, choices=["div", "mul", "shift"], default=None,
                        help="mapper type for foreground scaling level")
    parser.add_argument("--vr180", action="store_true",
                        help="output in VR180 format")
    parser.add_argument("--half-sbs", action="store_true",
                        help="output in Half SBS")
    parser.add_argument("--tb", action="store_true", help="output in Full TopBottom")
    parser.add_argument("--half-tb", action="store_true", help="output in Half TopBottom")

    parser.add_argument("--anaglyph", type=str, nargs="?", default=None, const="dubois",
                        choices=["color", "gray", "half-color", "wimmer", "wimmer2", "dubois", "dubois2"],
                        help="output in anaglyph 3d")
    parser.add_argument("--cross-eyed", action="store_true", help="output for cross-eyed viewing")
    parser.add_argument("--rgbd", action="store_true", help="output in RGBD")
    parser.add_argument("--half-rgbd", action="store_true", help="output in Half RGBD")

    parser.add_argument("--pix-fmt", type=str, default="yuv420p", choices=["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp", "gbrp10le", "gbrp16le"],
                        help="pixel format (video only)")
    parser.add_argument("--tta", action="store_true",
                        help="Use flip augmentation on depth model")
    parser.add_argument("--disable-amp", action="store_true",
                        help="disable AMP for some special reason")
    parser.add_argument("--cuda-stream", action="store_true",
                        help="use multi cuda stream for each thread/device")
    parser.add_argument("--max-output-width", type=int,
                        help="limit output width for cardboard players")
    parser.add_argument("--max-output-height", type=int,
                        help="limit output height for cardboard players")
    parser.add_argument("--keep-aspect-ratio", action="store_true",
                        help="keep aspect ratio when resizing")
    parser.add_argument("--start-time", type=str,
                        help="set the start time offset for video. hh:mm:ss or mm:ss format")
    parser.add_argument("--end-time", type=str,
                        help="set the end time offset for video. hh:mm:ss or mm:ss format")
    parser.add_argument("--resolution", type=int,
                        help="input resolution(small side) for depth model")
    parser.add_argument("--limit-resolution", action="store_true",
                        help=("if the source resolution is lower than --resolution, "
                              "the depth resolution will be limited to the source resolution."))
    parser.add_argument("--stereo-width", type=int,
                        help="input width for row_flow_v3/row_flow_v2 model")
    parser.add_argument("--ipd-offset", type=float, default=0,
                        help="IPD Offset (width scale %%). 0-10 is reasonable value for Full SBS")
    parser.add_argument("--ema-normalize", action="store_true",
                        help="use min/max moving average to normalize video depth")
    parser.add_argument("--ema-decay", type=float, default=0.75,
                        help="parameter for ema-normalize (0-1). large value makes it smoother")
    parser.add_argument("--ema-buffer", type=int, default=30, help="TODO")
    parser.add_argument("--ema-motion-adaptive", action="store_true",
                        help=("make --ema-decay automatically ease off during fast motion instead of "
                              "using one fixed smoothing strength for the whole clip. Reduces motion "
                              "smearing/lag on fast scenes while keeping full smoothing on calm ones. "
                              "Never smooths MORE than --ema-decay itself, only less -- a safe layer on "
                              "top of your existing EMA settings, not a replacement for them."))
    parser.add_argument("--scene-detect", action="store_true",
                        help=("splitting a scene using shot boundary detection. "
                              "ema and other states will be reset at the boundary of the scene"))
    parser.add_argument("--disable-scene-cache", action="store_true",
                        help="disable --scene-detect cache")
    parser.add_argument("--scene-cache-file", type=str,
                        help="force specify cache file for --scene-detect")
    parser.add_argument("--scene-cache-dir", type=str,
                        help="specify cache directory for --scene-detect")
    parser.add_argument("--scene-detect-only", action="store_true",
                        help="run only --scene-detect and skip the subsequent video processing")

    parser.add_argument("--scene-batch", action="store_true",
                        help=("fully automated whole-movie pipeline: remove letterbox bars once, "
                              "detect scene cuts once, split into per-scene clips with exact "
                              "keyframe-accurate boundaries, convert each scene independently "
                              "(true fresh start per scene, not a shared running state), then "
                              "join the results back into one seamless video and reattach the "
                              "original audio. Input must be a single video file."))
    parser.add_argument("--scene-batch-crop", type=str, default=None,
                        help=("explicit crop for --scene-batch, as WxH or WxH:X:Y (X/Y default to "
                              "centered). If omitted, letterbox bars are auto-detected once from "
                              "the whole movie."))
    parser.add_argument("--scene-settings", type=str, default=None,
                        help=("JSON file for --scene-batch giving per-scene setting overrides. "
                              "A list of rules, each combining an optional position match -- either "
                              '{"start_scene": 0, "end_scene": 40, ...} (scene-index range, end '
                              'exclusive) or {"start_time": 0, "end_time": 600, ...} (seconds) -- '
                              "with an optional duration match, "
                              '{"min_duration": 2, "max_duration": 3, ...} (seconds, end exclusive), '
                              "e.g. to give every short 2-3 second scene its own EMA settings "
                              'regardless of where it falls in the movie. Both end with "overrides": '
                              '{"divergence": 3.5}. Rules are applied in order; later matching rules '
                              "override earlier ones; position and duration matches on the same rule "
                              "must both be satisfied for it to apply."))
    parser.add_argument("--scene-batch-auto-ema", action="store_true",
                        help=("automatically pick EMA Decay/Buffer per scene based on that scene's own "
                              "length, using a built-in table (one bucket per whole second, 0-15s+). "
                              "Works two ways: with --scene-batch, tuned for its independent-scene "
                              "processing (short scenes get a smaller Buffer so the smoothing actually "
                              "finishes settling within the scene, instead of a Buffer sized for one long "
                              "continuous shot), applied before --scene-settings so anything that file "
                              "sets explicitly (EMA included) still wins; without --scene-batch, requires "
                              "plain --scene-detect instead (there are no scene boundaries to key off of "
                              "otherwise) and re-picks EMA Decay/Buffer at every detected cut within the "
                              "same continuous video, overriding the fixed --ema-decay/--ema-buffer for "
                              "each scene as it starts. Which table is used depends on "
                              "--scene-batch-auto-ema-model."))
    from .scene_batch import EMA_BY_DURATION_TABLES
    parser.add_argument("--scene-batch-auto-ema-model", type=str, default="3DECKER VDA_L",
                        choices=list(EMA_BY_DURATION_TABLES.keys()),
                        help=("which built-in EMA-by-duration table --scene-batch-auto-ema uses, "
                              "matched to the Depth Model in use. 3DECKER VDA_L: a real video depth model "
                              "with its own frame-to-frame memory, needs only light smoothing on top. "
                              "3DECKER Any_V3_Mono_01: a stills-only model with no frame-to-frame memory of "
                              "its own (prone to visible 'depth breathing' without help), so this table "
                              "uses double VDA_L's Buffer at every scene length with a correspondingly "
                              "higher Decay to compensate. The remaining names (Nagadomi_Reference, "
                              "GEMINI AI, ChatGPT, Grok, Fast Action, Medium Magical, Drama Slow Paced) are "
                              "the same tables offered in the GUI's Auto EMA by Scene Length dropdown -- "
                              "this list is read from the same EMA_BY_DURATION_TABLES source of truth in "
                              "scene_batch.py so the CLI and GUI never drift apart again."))
    parser.add_argument("--scene-batch-variant", type=str, default=None,
                        help=("optional name for --scene-batch. Reuses the shared, already-done work "
                              "from a prior run of the SAME movie (Dolby Vision RPU extraction, the "
                              "crop+keyframe pass, and the split scene clips) but writes its own "
                              "converted scenes and final output under this name, so trying different "
                              "settings (e.g. a different EMA table) never touches or overwrites an "
                              "earlier attempt. Output becomes <name>_<variant>.<ext>."))

    parser.add_argument("--depth-blend", action="store_true",
                        help=("blend depth from a second model into --depth-model's output, favoring the "
                              "second model specifically in areas with dense fine visual detail (foliage, "
                              "hair, close-up texture) where a single-frame model like Any_V3_Mono_01 "
                              "often struggles. Runs as three full passes over the clip -- export depth "
                              "with the primary model, export depth with the secondary model, then blend "
                              "and render -- fully releasing each model before the next loads, so the two "
                              "models are never in GPU memory at the same time. Costs real extra time (~3x "
                              "a normal pass) and real extra disk space (a full rgb+depth frame dump, "
                              "twice) in a '<output>.depth_blend_work' folder next to your output, safe to "
                              "delete once you're happy with the result. Requires a single video file "
                              "input; not compatible with --scene-batch."))
    parser.add_argument("--depth-blend-model", type=str, default="VDA_L",
                        help="secondary depth model for --depth-blend (default: VDA_L)")
    parser.add_argument("--depth-blend-strength", type=float, default=1.0,
                        help=("how strongly to favor --depth-blend-model in the selected region "
                              "(0-1). 1.0 = fully trust the secondary model there, 0.0 = ignore it "
                              "entirely (same as not using --depth-blend)."))
    parser.add_argument("--depth-blend-region", type=str, default="detail",
                        choices=["detail", "foreground", "background"],
                        help=("what decides WHERE --depth-blend-model gets blended in. 'detail' "
                              "(default): wherever the image shows dense fine visual detail (foliage, "
                              "hair, texture) -- the original foliage/close-up fix. 'foreground': the "
                              "nearest --depth-blend-region-percent%% of the scene by depth, regardless "
                              "of visual detail. 'background': the farthest --depth-blend-region-percent%% "
                              "instead. Foreground/background use iw3's own near/far convention (higher "
                              "depth value = nearer), the same one --foreground-pop and "
                              "--foreground-divergence already use."))
    parser.add_argument("--depth-blend-region-percent", type=float, default=25.0,
                        help=("for --depth-blend-region foreground/background: what percent of the scene "
                              "(by depth) to blend the secondary model into, e.g. 25 = nearest (or "
                              "farthest) 25%% of the scene. Ignored for --depth-blend-region detail."))

    parser.add_argument("--autocrop", type=str.upper, default=None,
                        choices=["BLACK_TB", "BLACK", "FLAT_TB", "FLAT"],
                        help=("autocrop mode. automatically removes black bars. "
                              "BLACK_TB: Removes only the top and bottom black bars. "
                              "BLACK: Automatically removes black bars from all sides. "
                              "FLAT_TB: Removes only the top and bottom flat-color borders."
                              "FLAT: Removes flat-color borders. "
                              ))

    parser.add_argument("--edge-dilation", type=int, nargs="+", default=[2, 1],
                        help="loop count of edge dilation. <x> <y> or <xy>")

    parser.add_argument("--inpaint-model", type=str, default=None, choices=list(INPAINT_MODELS.keys()),
                        help="inpaint model name defined in iw3/inpaint_models.yml")
    parser.add_argument("--mask-inner-dilation", type=int, default=0,
                        help="loop count of inner mask dilation")
    parser.add_argument("--mask-outer-dilation", type=int, default=0,
                        help="loop count of outer mask dilation")
    parser.add_argument("--inpaint-max-width", type=int, default=None,
                        help="max width of inpaint result")
    parser.add_argument("--inpaint-overlap-frames", type=int, nargs="+", default=None,
                        help="overlap/padding frames for video inpaint model. <frames> or <pre frames> <post frames>")

    parser.add_argument("--depth-aa", action="store_true",
                        help="apply depth antialiasing. ignored for unsupported models")
    parser.add_argument("--depth-refine", action="store_true",
                        help=("clean up each depth frame's own internal noise using edge-preserving "
                              "(bilateral) smoothing, within that single frame -- a different axis from EMA "
                              "smoothing, which works ACROSS frames over time. Cheap: no extra passes, no "
                              "new dependencies, doesn't affect how many depth models are used."))
    parser.add_argument("--depth-refine-strength", type=float, default=1.0,
                        help=("how strong the --depth-refine bilateral cleanup is. 1.0 (default) matches "
                              "this feature's original fixed behavior exactly; higher pushes the smoothing "
                              "further, lower pulls back. Has no effect unless --depth-refine is also set."))
    parser.add_argument("--temporal-stabilize", action="store_true",
                        help=("approximates a video-aware model's (VDA_L) per-pixel stability for a "
                              "single-frame model (e.g. Any_V3_Mono_01): tracks real motion via optical "
                              "flow on the RGB image and blends each frame's depth with the PREVIOUS "
                              "frame's depth warped to match where content actually moved to, reducing a "
                              "specific object's depth flickering that EMA's overall-range smoothing can't "
                              "touch. Only takes effect on the single-frame processing path (used "
                              "automatically for --low-vram, --debug-depth, VDA streaming models, and "
                              "inpaint stereo methods e.g. mlbw_l2_inpaint) -- has no effect on the batched "
                              "path other models use."))
    parser.add_argument("--temporal-stabilize-strength", type=float, default=0.7,
                        help=("how strongly to trust the motion-warped previous frame vs the fresh "
                              "per-frame depth (0-1) for --temporal-stabilize. Automatically tapers down "
                              "during fast/unreliable motion regardless of this setting."))
    parser.add_argument("--max-workers", type=int, default=0, choices=[0, 1, 2, 3, 4, 8, 16],
                        help="max inference worker threads for video processing. 0 is disabled")
    parser.add_argument("--video-format", "-vf", type=str, default="mp4", choices=["mp4", "mkv", "avi"],
                        help="video container format")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "webp", "jpeg"],
                        help="output image format")
    parser.add_argument("--video-codec", "-vc", type=str, default=None, help="video codec")
    parser.add_argument("--hwaccel", type=str, default=None,
                        choices=VU.HW_DEVICES,
                        help="hardware accelerator for the video decoder")
    parser.add_argument("--disable-software-fallback", action="store_true",
                        help="disable software fallback for hardware hwaccel")

    parser.add_argument("--metadata", type=str, nargs="?", default=None, const="filename", choices=["filename"],
                        help="Add metadata")
    parser.add_argument("--find-param", type=str, nargs="+",
                        choices=["divergence", "convergence", "foreground-scale", "ipd-offset"],
                        help="outputs results for various parameter combinations")

    parser.add_argument("--colorspace", type=str, default="auto",
                        choices=["unspecified", "auto",
                                 "bt709", "bt709-pc", "bt709-tv",
                                 "bt601", "bt601-pc", "bt601-tv",
                                 "bt2020-tv", "bt2020-pq-tv"],
                        help="video colorspace")
    parser.add_argument("--preserve-dowi", action="store_true",
                        help="preserve Dolby Vision RPU metadata in HEVC output (requires dovi_tool)")
    parser.add_argument("--stereo-mode-tag", action="store_true",
                        help="tag the finished .mkv's video track with the Matroska StereoMode "
                             "property (via mkvpropedit) so 3D-aware players/TVs (VLC, Kodi, "
                             "compatible smart TVs) auto-detect the 3D packing/eye-order instead of "
                             "the viewer having to select it manually. Only applies to .mkv output "
                             "and only to genuine two-eye stereo layouts (SBS/TB/Cross Eyed/VR90) -- "
                             "RGB-D, Half RGB-D, and Anaglyph are skipped with a printed note (no "
                             "correct or useful StereoMode value exists for them). See "
                             "docs/ai/AI_DECISIONS.md ADR-033.")
    parser.add_argument("--hdr-to-sdr", action="store_true",
                        help="tone-map a PQ/HLG HDR source down to SDR (10-bit retained) before conversion")
    parser.add_argument("--auto-resume", action="store_true",
                        help="split video into chunks and resume from checkpoint if interrupted")
    parser.add_argument("--resume-chunk-duration", type=int, default=300, metavar="SECONDS",
                        help="chunk duration in seconds for --auto-resume (default: 300)")
    parser.add_argument("--upgrade-pix-fmt", type=int, default=None, choices=[10, 12],
                        help="upgrade 8-bit source to 10-bit or 12-bit output pixel format")
    parser.add_argument("--denoise", action="store_true",
                        help="apply temporal denoising (hqdn3d via a one-time ffmpeg pre-pass) before depth "
                             "estimation to reduce film grain artifacts")
    parser.add_argument("--preview", action="store_true",
                        help="generate a quick 1fps/256p preview of the first 60 seconds to check 3D settings")
    # Deprecated
    parser.add_argument("--zoed-batch-size", type=int,
                        help="Deprecated. Use --batch-size instead")
    parser.add_argument("--zoed-height", type=int,
                        help="Deprecated. Use --resolution instead")

    return parser


def calc_auto_warp_steps(method, divergence, synthetic_view):
    divergence = divergence if synthetic_view == "both" else divergence * 2
    if method == "row_flow_v2" and divergence > ROW_FLOW_V2_MAX_DIVERGENCE:
        return math.ceil(divergence / ROW_FLOW_V2_AUTO_STEP_DIVERGENCE)
    if method in {"row_flow", "row_flow_v3"} and divergence > ROW_FLOW_V3_MAX_DIVERGENCE:
        return math.ceil(divergence / ROW_FLOW_V3_AUTO_STEP_DIVERGENCE)

    return None


def _release_pause_vram(args):
    """--pause-frees-vram (ADR-038): moves every loaded model this run actually put
    on the GPU to CPU and empties the CUDA caching allocator, so VRAM is genuinely
    released back to the OS while paused. Reads args.state lazily (called from
    inside _PauseVramSuspendEvent.wait(), long after set_state_args() built the
    initial state dict, and after iw3_main() has since added "side_model" to it),
    so it always sees whichever model is currently the live one -- including
    Dual-Pass Depth Blend's own primary/secondary swap of args.state["depth_model"],
    which this never fights with: only one is ever resident at a time either way."""
    state = args.state
    depth_model = state.get("depth_model")
    if depth_model is not None and depth_model.loaded():
        depth_model.move_to("cpu")
    side_model = state.get("side_model")
    if side_model is not None and not isinstance(side_model, DeviceSwitchInference) and hasattr(side_model, "to"):
        side_model.to("cpu")
    convergence_model = state.get("convergence_model")
    if convergence_model is not None and getattr(convergence_model, "model", None) is not None:
        convergence_model.model = convergence_model.model.to("cpu")
    gc_collect()


def _reload_pause_vram(args):
    """The Resume half of _release_pause_vram() -- moves everything back to its
    real device (args.state["device"], the single-GPU device this run was
    configured with; multi-GPU models were skipped on release and need no
    reload) before processing continues."""
    state = args.state
    device = state.get("device")
    depth_model = state.get("depth_model")
    if depth_model is not None and depth_model.loaded():
        depth_model.move_to(depth_model.device)
    side_model = state.get("side_model")
    if side_model is not None and not isinstance(side_model, DeviceSwitchInference) and hasattr(side_model, "to"):
        side_model.to(device)
    convergence_model = state.get("convergence_model")
    if convergence_model is not None and getattr(convergence_model, "model", None) is not None:
        convergence_model.model = convergence_model.model.to(convergence_model.device)


class _PauseVramSuspendEvent():
    """Wraps a real threading.Event so every existing call site that already does
    `suspend_event.wait()` across this codebase (iw3/utils.py, depth_blend.py,
    scene_batch.py, nunif/utils/video/processor.py) gets real VRAM release for
    free while paused, without any of those call sites changing. Delegates
    is_set()/set()/clear() straight to the real event, so the GUI's Suspend/
    Resume button (which only ever calls those three) and Cancel/close handling
    are completely unaffected. Only ever constructed by set_state_args() when
    args.pause_frees_vram is True (ADR-038) -- when the setting is off, the plain
    threading.Event is stored exactly as before.

    _released tracks whether THIS wrapper has already moved things to CPU for the
    pause currently in progress, guarded by a lock so multiple threads calling
    wait() during the same pause (e.g. the image-save pool and a scene-boundary
    scan) release/reload exactly once, not once per caller."""

    def __init__(self, event, args):
        self._event = event
        self._args = args
        self._lock = threading.Lock()
        self._released = False

    def is_set(self):
        return self._event.is_set()

    def set(self):
        self._event.set()

    def clear(self):
        self._event.clear()

    def wait(self, timeout=None):
        if not self._event.is_set():
            with self._lock:
                if not self._released and not self._event.is_set():
                    self._released = True
                    _release_pause_vram(self._args)
        result = self._event.wait(timeout)
        if self._event.is_set():
            with self._lock:
                if self._released:
                    self._released = False
                    _reload_pause_vram(self._args)
        return result


def set_state_args(args, stop_event=None, tqdm_fn=None, depth_model=None, suspend_event=None, stage_fn=None):
    if depth_model is None:
        depth_model = create_depth_model(args.depth_model)
    depth_model.enable_refine(getattr(args, "depth_refine", False),
                               strength=getattr(args, "depth_refine_strength", 1.0) or 1.0)
    if getattr(args, "temporal_stabilize", False):
        depth_model.enable_temporal_stabilize(
            strength=getattr(args, "temporal_stabilize_strength", 0.7) or 0.7,
            max_shift_velocity=getattr(args, "temporal_stabilize_max_shift_velocity", None),
            flat_region_boost=getattr(args, "temporal_stabilize_flat_region_boost", 0.0) or 0.0,
            edge_protection=getattr(args, "temporal_stabilize_edge_protection", 0.0) or 0.0)
    else:
        depth_model.disable_temporal_stabilize()

    convergence_model = None
    if args.convergence_mode == "sod_v1":
        convergence_model = ConvergenceEstimator(args.convergence, device_id=args.gpu[0],
                                                 decay=getattr(args, "convergence_smoothing", 0.9),
                                                 compile=args.compile)
    elif args.convergence_mode == "face_detect":
        try:
            convergence_model = FaceConvergenceEstimator(decay=getattr(args, "convergence_smoothing", 0.9))
        except Exception as e:
            raise RuntimeError(
                f"face_detect convergence mode failed to initialize.\n"
                f"If mediapipe is missing, run: pip install mediapipe absl-py protobuf attrs flatbuffers\n"
                f"Error: {type(e).__name__}: {e}"
            ) from e

    if args.export_disparity:
        args.export = True
    if args.export_depth_only and not args.export:
        raise ValueError("--export-depth-only must be specified together with --export or --export-disparity")
    if args.export_depth_fit and not args.export:
        raise ValueError("--export-depth-fit must be specified together with --export or --export-disparity")

    if getattr(args, "rife_interpolate", False) and getattr(args, "preserve_dowi", False):
        raise ValueError(
            "--rife-interpolate and --preserve-dowi cannot be used together: RIFE inserts "
            "synthetic in-between frames that have no correct Dolby Vision/HDR10+ per-frame "
            "metadata to assign, so there is currently no way to preserve DV/HDR10+ through "
            "frame interpolation. Disable one of the two options and try again.")

    if getattr(args, "rife_multiplier", None) is not None and getattr(args, "rife_target_fps", None) is not None:
        raise ValueError(
            "--rife-multiplier and --rife-target-fps cannot be used together: specify at most "
            "one of the two (see docs/ai/AI_DECISIONS.md ADR-049). --rife-target-fps is a more "
            "precise override of --rife-multiplier's simple N-times behavior.")

    if depth_model.get_name() == "VideoDepthAnything":
        if not args.ema_normalize:
            warnings.warn("--ema-normalize is highly recommended for VideoDepthAnything")
        if not args.scene_detect:
            warnings.warn("--scene-detect is highly recommended for VideoDepthAnything")

    if is_video(args.output):
        # replace --video-format when filename is specified
        ext = path.splitext(args.output)[-1]
        if ext == ".mp4":
            args.video_format = "mp4"
        elif ext == ".mkv":
            args.video_format = "mkv"
        elif ext == ".avi":
            args.video_format = "avi"

    args.video_extension = "." + args.video_format
    if args.video_codec is None:
        args.video_codec = VU.get_default_video_codec(args.video_format)

    if not args.profile_level or args.profile_level == "auto":
        args.profile_level = None

    # deprecated options
    if args.zoed_batch_size is not None:
        args.batch_size = args.zoed_batch_size
        warnings.warn("--zoed-batch-size is deprecated. Use --batch-size instead")
    if args.zoed_height is not None:
        args.resolution = args.zoed_height
        warnings.warn("--zoed-height is deprecated. Use --resolution instead")
    if args.remove_bg:
        warnings.warn("--remove-bg is deleted")

    if suspend_event is not None and getattr(args, "pause_frees_vram", False):
        # ADR-038: opt-in only -- wraps the real Event so every existing
        # `suspend_event.wait()` call site gets VRAM release for free while
        # paused. When the setting is off, the plain threading.Event is stored
        # below exactly as before this feature existed.
        suspend_event = _PauseVramSuspendEvent(suspend_event, args)

    args.state = {
        "stop_event": stop_event,
        "suspend_event": suspend_event,
        "tqdm_fn": tqdm_fn,
        "stage_fn": stage_fn,
        "depth_model": depth_model,
        "convergence_model": convergence_model,
        "device": create_device(args.gpu),
        "devices": [create_device(gpu_id) for gpu_id in args.gpu],
    }

    gc_collect()

    return args


def export_main(args):
    if is_text(args.input):
        raise NotImplementedError("--export with text format input is not supported")

    depth_model = args.state["depth_model"]

    if path.isdir(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not args.recursive:
            if depth_model.is_image_supported():
                export_images(args.input, args.output, args)
                gc_collect()
            if depth_model.is_video_supported():
                for video_file in VU.list_videos(args.input):
                    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                        return args
                    export_video(video_file, args.output, args)
                    gc_collect()
        else:
            subdirs = list_subdir(args.input, include_root=True, excludes=args.output)
            for input_dir in subdirs:
                output_dir = path.normpath(path.join(args.output, path.relpath(input_dir, start=args.input)))
                if depth_model.is_image_supported():
                    export_images(input_dir, output_dir, args, title=path.relpath(input_dir, args.input))
                    gc_collect()
                if depth_model.is_video_supported():
                    for video_file in VU.list_videos(input_dir):
                        if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                            return args
                        export_video(video_file, output_dir, args)
                        gc_collect()

    elif is_image(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not depth_model.is_image_supported():
            raise ValueError(f"{args.depth_model} does not support image input")
        export_images(args.input, args.output, args)
    elif is_video(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if not depth_model.is_video_supported():
            raise ValueError(f"{args.depth_model} not support video input")
        export_video(args.input, args.output, args)
    else:
        raise ValueError("Unrecognized file type")


def is_yaml(filename):
    return path.splitext(filename)[-1].lower() in {".yaml", ".yml"}


def iw3_main(args):
    assert not (args.rotate_left and args.rotate_right)
    assert sum([1 for flag in (args.half_sbs, args.vr180, args.anaglyph, args.tb, args.half_tb, args.cross_eyed, args.half_rgbd, args.rgbd) if flag]) < 2

    if len(args.gpu) > 1 and len(args.gpu) > args.max_workers:
        # For GPU round-robin on thread pool
        args.max_workers = len(args.gpu)

    if args.warp_steps is None:
        args.warp_steps = calc_auto_warp_steps(method=args.method, divergence=args.divergence,
                                               synthetic_view=args.synthetic_view)

    if path.normpath(args.input) == path.normpath(args.output):
        raise ValueError("input and output must be different file")

    if args.export and is_yaml(args.input):
        raise ValueError("YAML file input does not support --export")

    if getattr(args, "scene_batch_auto_ema", False) and not args.scene_batch and not args.scene_detect:
        raise ValueError("--scene-batch-auto-ema requires either --scene-batch or --scene-detect -- "
                         "there are no scene boundaries to key off of otherwise.")

    if args.tune and args.video_codec == "libx265":
        if len(args.tune) != 1:
            raise ValueError("libx265 does not support multiple --tune options.\n"
                             f"tune={','.join(args.tune)}")
        if args.tune[0] in {"film", "stillimage"}:
            raise ValueError(f"libx265 does not support --tune {args.tune[0]}\n"
                             "available options: grain,animation,psnr,zerolatency,fastdecode")

    assert args.state["depth_model"] is not None
    depth_model = args.state["depth_model"]
    if args.update:
        depth_model.force_update()

    if getattr(args, "depth_blend", False) and not is_yaml(args.input):
        # NOTE: depth_blend's own final render step re-enters iw3_main with a YAML
        # config as input (its "resume from exported depth" path) and deliberately
        # keeps args.depth_blend=True so the filename/metadata tagging above still
        # fires -- the is_yaml() check here is what stops that from being mistaken
        # for a fresh "start a new Depth Blend job" request and recursing.
        if path.isdir(args.input):
            raise ValueError("--depth-blend requires a single video file as input")
        if args.scene_batch:
            raise ValueError("--depth-blend cannot be combined with --scene-batch")
        from .depth_blend import run_depth_blend
        run_depth_blend(args, depth_model, None)
        return args

    if not is_yaml(args.input):
        if not depth_model.loaded():
            depth_model.load(gpu=args.gpu, resolution=args.resolution, limit_resolution=args.limit_resolution)

        is_metric = depth_model.is_metric()
        args.mapper = resolve_mapper_name(mapper=args.mapper, foreground_scale=args.foreground_scale,
                                          metric_depth=is_metric,
                                          mapper_type=args.mapper_type)
    else:
        depth_model = None
        # specified args.mapper never used in process_config_*
        args.mapper = "none"

    if args.export:
        export_main(args)
        return args

    side_model = create_stereo_model(
        args.method,
        divergence=args.divergence * (2.0 if args.synthetic_view in {"right", "left"} else 1.0),
        device_id=args.gpu[0],
        inpaint_model=args.inpaint_model,
        overlap_frames=args.inpaint_overlap_frames,
    )
    if (
            side_model is not None
            and len(args.gpu) > 1
            and args.method not in {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}
    ):
        side_model = DeviceSwitchInference(side_model, device_ids=args.gpu)

    # ADR-038 (--pause-frees-vram): side_model is otherwise only ever a local
    # variable threaded through the call chain by hand -- stash it in args.state
    # so a pause anywhere downstream can find and move it, same as depth_model.
    args.state["side_model"] = side_model

    if args.find_param:
        assert is_image(args.input) and (path.isdir(args.output) or not path.exists(args.output))
        find_param(args, depth_model, side_model)
        return args

    if args.scene_batch:
        if path.isdir(args.input) or is_yaml(args.input):
            raise ValueError("--scene-batch requires a single video file as input")
        from .scene_batch import run_scene_batch
        run_scene_batch(args, depth_model, side_model)
        return args

    if path.isdir(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if args.scene_cache_file is not None:
            raise ValueError("--scene-cache-file cannot be used in batch processing."
                             " Use --scene-cache-dir instead.")

        if not args.recursive:
            if depth_model.is_image_supported():
                image_files = ImageLoader.listdir(args.input)
                process_images(image_files, args.output, args, depth_model, side_model, title="Images")
                gc_collect()
            if depth_model.is_video_supported():
                for video_file in VU.list_videos(args.input):
                    if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                        return args
                    try:
                        process_video(video_file, args.output, args, depth_model, side_model)
                    except KeyboardInterrupt:
                        raise
                    except: # noqa
                        if not args.skip_error:
                            print(f"Error: {video_file}", file=sys.stderr)
                            raise
                        print_exception(video_file)
                    gc_collect()
        else:
            subdirs = list_subdir(args.input, include_root=True, excludes=args.output)
            for input_dir in subdirs:
                output_dir = path.normpath(path.join(args.output, path.relpath(input_dir, start=args.input)))
                if depth_model.is_image_supported():
                    image_files = ImageLoader.listdir(input_dir)
                    if image_files:
                        process_images(image_files, output_dir, args, depth_model, side_model,
                                       title=path.relpath(input_dir, args.input))
                        gc_collect()
                if depth_model.is_video_supported():
                    for video_file in VU.list_videos(input_dir):
                        if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                            return args
                        try:
                            process_video(video_file, output_dir, args, depth_model, side_model)
                        except KeyboardInterrupt:
                            raise
                        except: # noqa
                            if not args.skip_error:
                                print(f"Error: {video_file}", file=sys.stderr)
                                raise
                            print_exception(video_file)
                        gc_collect()

    elif is_yaml(args.input):
        config = export_config.ExportConfig.load(args.input)
        if config.type == export_config.VIDEO_TYPE:
            process_config_video(config, args, side_model)
        if config.type == export_config.IMAGE_TYPE:
            process_config_images(config, args, side_model)
    elif is_text(args.input):
        if not is_output_dir(args.output):
            raise ValueError("-o must be a directory")
        if args.scene_cache_file is not None:
            raise ValueError("--scene-cache-file cannot be used in batch processing."
                             " Use --scene-cache-dir instead.")

        files = []
        with open(args.input, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line.startswith("#"):
                    files.append(line.strip())
        if depth_model.is_image_supported():
            image_files = [f for f in files if is_image(f)]
            process_images(image_files, args.output, args, depth_model, side_model, title="Images")
        if depth_model.is_video_supported():
            video_files = [f for f in files if is_video(f)]
            for video_file in video_files:
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    return args
                process_video(video_file, args.output, args, depth_model, side_model)
                gc_collect()
    elif is_video(args.input):
        if not depth_model.is_video_supported():
            raise ValueError(f"{args.depth_model} does not support video input")
        process_video(args.input, args.output, args, depth_model, side_model)
    elif is_image(args.input):
        if not depth_model.is_image_supported():
            raise ValueError(f"{args.depth_model} does not support image input")
        if side_model is not None and hasattr(side_model, "set_mode"):
            side_model.set_mode("image")
            side_model.reset()
        if args.state["convergence_model"] is not None:
            args.state["convergence_model"].reset(enable_ema=False)

        if is_output_dir(args.output):
            os.makedirs(args.output, exist_ok=True)
            output_filename = path.join(
                args.output,
                make_output_filename(args.input, args, video=False))
        else:
            output_filename = args.output
        im, _ = load_image_simple(args.input, color="rgb", exif_transpose=not args.disable_exif_transpose)
        im = TF.to_tensor(im).to(args.state["device"])
        output = process_image(im, args, depth_model, side_model)
        output = to_pil_image(output)
        make_parent_dir(output_filename)
        output.save(output_filename)
    else:
        raise ValueError("Unrecognized file type")

    return args


def find_param(args, depth_model, side_model):
    im, _ = load_image_simple(args.input, color="rgb")
    if im is None:
        raise RuntimeError(f"{args.input} cannot be loadded")
    im = TF.to_tensor(im).to(args.state["device"])

    args.metadata = "filename"
    os.makedirs(args.output, exist_ok=True)
    if args.method == "forward_fill":
        divergence_cond = range(1, 10 + 1) if "divergence" in args.find_param else [args.divergence]
        convergence_cond = np.arange(-2, 2, 0.25) if "convergence" in args.find_param else [args.convergence]
    else:
        max_divegence = 10 if args.method.startswith("mlbw_") else 5
        divergence_cond = range(1, max_divegence + 1) if "divergence" in args.find_param else [args.divergence]
        convergence_cond = np.arange(0, 1, 0.25) if "convergence" in args.find_param else [args.convergence]

    foreground_scale_cond = range(0, 3 + 1) if "foreground-scale" in args.find_param else [args.foreground_scale]
    ipd_offset_cond = range(0, 5 + 1) if "ipd-offset" in args.find_param else [args.ipd_offset]
    is_metric = depth_model.is_metric()

    params = []
    for divergence in divergence_cond:
        for convergence in convergence_cond:
            for foreground_scale in foreground_scale_cond:
                for ipd_offset in ipd_offset_cond:
                    params.append((divergence, convergence, foreground_scale, ipd_offset))

    for divergence, convergence, foreground_scale, ipd_offset in tqdm(params, ncols=80):
        args.divergence = float(divergence)
        args.convergence = float(convergence)
        args.ipd_offset = ipd_offset
        args.foreground_scale = foreground_scale
        args.mapper = resolve_mapper_name(mapper=None, foreground_scale=args.foreground_scale,
                                          metric_depth=is_metric,
                                          mapper_type=args.mapper_type)

        output_filename = path.join(
            args.output,
            make_output_filename("param.png", args, video=False))
        output = process_image(im, args, depth_model, side_model)
        output = to_pil_image(output)
        output.save(output_filename)
