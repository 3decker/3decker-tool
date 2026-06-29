import sys
import traceback
import os
import subprocess
from os import path
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
from .forward_warp import apply_divergence_forward_warp
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

    print("[denoise] pre-denoising source with hqdn3d before processing -- this adds one "
          "extra encoding pass.", file=sys.stderr)
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", *trim_args, "-i", str(input_filename),
             "-vf", "hqdn3d=1:1:6:6",
             "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-crf", "12",
             "-c:a", "copy", tmp_denoised],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[denoise] pre-denoise failed, processing the original source instead: "
              f"{e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
        return input_filename, None

    if trim_args:
        # The intermediate file already covers exactly [start_time, end_time]; clear those
        # so the rest of the pipeline doesn't try to trim an already-trimmed file again.
        args.start_time = None
        args.end_time = None

    return tmp_denoised, tmp_denoised


def _inject_hdr_metadata(output_path, rpu_path, hdr10plus_json, ffmpeg_bin, dovi_bin, hdr10plus_bin, tmp_dir):
    """Inject DV RPU and/or HDR10+ into the output HEVC, then remux back into the container."""
    import json as _json
    ffprobe_bin = ffmpeg_bin.replace("ffmpeg", "ffprobe").replace("ffmpeg.exe", "ffprobe.exe")
    # Verify the output video is HEVC and get its frame rate
    codec = None
    fps_str = "30fps"
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
    except Exception:
        pass
    if codec != "hevc":
        print(f"--preserve-dowi: output codec is '{codec}', not hevc — "
              f"DV/HDR10+ injection requires HEVC output (use --video-codec libx265).", file=sys.stderr)
        return

    hevc_out = path.join(tmp_dir, "_iw3_out.hevc")
    hevc_dv = path.join(tmp_dir, "_iw3_out_dv.hevc")
    hevc_h10p = path.join(tmp_dir, "_iw3_out_h10p.hevc")
    final_tmp = path.splitext(output_path)[0] + ".hdr_inject" + path.splitext(output_path)[1]
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(output_path), "-c:v", "copy", "-an", "-f", "hevc", hevc_out],
            check=True, capture_output=True,
        )
        current = hevc_out

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

        ext = path.splitext(output_path)[1].lower()
        mkvmerge_bin = _find_mkvmerge() if ext == ".mkv" else None
        if mkvmerge_bin:
            # mkvmerge correctly handles raw HEVC timestamps
            subprocess.run(
                [mkvmerge_bin, "-o", final_tmp,
                 "--default-duration", f"0:{fps_str}",
                 current,
                 "--no-video", str(output_path)],
                check=True, capture_output=True,
            )
        else:
            subprocess.run(
                [ffmpeg_bin, "-y",
                 "-i", str(output_path),
                 "-i", current,
                 "-map", "1:v", "-map", "0:a?",
                 "-c", "copy", "-copyts", final_tmp],
                check=True, capture_output=True,
            )
        os.replace(final_tmp, output_path)
    finally:
        for f in (hevc_out, hevc_dv, hevc_h10p, final_tmp):
            if path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


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
        if args.ema_normalize and video:
            ema = f"_ema{to_deciaml(args.ema_decay, 100, 2)}b{args.ema_buffer}"
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

        metadata = (f"_{args.depth_model}_{resolution}{tta}{args.method}_"
                    f"d{to_deciaml(args.divergence, 10, 2)}_{convergence_name}{to_deciaml(args.convergence, 10, 2)}"
                    f"{convergence_smoothing}_"
                    f"di{edge_dilation}_fs{args.foreground_scale}_fp{args.foreground_pop}_"
                    f"ipd{to_deciaml(args.ipd_offset, 1)}{ema}{bitrate}")
    else:
        metadata = ""

    return basename + metadata + auto_detect_suffix + (args.video_extension if video else get_image_ext(args.format))


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
    elif args.method in {"forward", "forward_fill"}:
        left_eye, right_eye = apply_divergence_forward_warp(
            im, depth,
            args.divergence, convergence=convergence,
            method=args.method, synthetic_view=args.synthetic_view, width_base=False)
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
        depth = depth_model.minmax_normalize_chw(depth)

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

def bind_single_frame_callback(depth_model, side_model, segment_pts, args):
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
        depth = depth_model.minmax_normalize_chw(depth)
        depths = [depth] if depth is not None else []
        flush = frame.pts in segment_pts
        if flush:
            depths += depth_model.flush_minmax_normalize()
            depth_model.reset_state()

        yield from _postprocess(depths, flush=flush)

    return _frame_callback


def bind_batch_frame_callback(depth_model, side_model, segment_pts, args):
    depth_lock = threading.RLock()
    sbs_lock = threading.RLock()
    enqueue_ticket_lock = TicketLock()
    dequeue_ticket_lock = TicketLock()
    streams = threading.local()
    src_queue = []
    frame_cpu_offload = depth_model.get_ema_buffer_size() > 1
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)

    def _postprocess(depth_batch, reset_ema, dequeue_ticket_id, flush, device):
        # Reorder threads
        with dequeue_ticket_lock(dequeue_ticket_id):
            with depth_lock:
                if flush:
                    depth_list = depth_model.flush_minmax_normalize()
                else:
                    depth_list = depth_model.minmax_normalize(depth_batch, reset_ema=reset_ema)

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
                        if args.method in {"forward_fill", "forward"}:
                            # lock all threads (sbs_lock -> ticket_lock -> depth_lock order)
                            with enqueue_ticket_lock, dequeue_ticket_lock, depth_lock:
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
            depth_batch, dequeue_ticket_id = _batch_infer(
                None, None, flush=flush, enqueue_ticket_id=enqueue_ticket_id)
            # Return a generator directly to avoid out-of-memory errors during flush.
            # Processing is performed on the main thread.
            return _postprocess(
                depth_batch, reset_ema,
                dequeue_ticket_id=dequeue_ticket_id,
                flush=flush,
                device=device
            )
        else:
            device = x.device
            reset_ema = [t in segment_pts for t in pts]
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
                depth_batch, reset_ema,
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


def bind_vda_frame_callback(depth_model, side_model, segment_pts, args):
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

        x = torch.stack(batch_queue)
        depth_list = depth_model.infer_with_normalize(
            x, pts_queue, segment_pts,
            enable_amp=not args.disable_amp,
            edge_dilation=args.edge_dilation,
            depth_aa=args.depth_aa,
            tta=args.tta)

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


def process_video_full(input_filename, output_path, args, depth_model, side_model):
    is_preview = getattr(args, "preview", False)
    scene_cache_max_fps = args.max_fps  # capture before --preview clamps it, so cache key stays stable
    use_16bit = VU.pix_fmt_requires_16bit(args.pix_fmt)
    is_video_depth_anything = depth_model.get_name() == "VideoDepthAnything"
    is_video_depth_anything_streaming = depth_model.get_name() == "VideoDepthAnythingStreaming"
    is_inpaint_model = args.method in {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}
    ema_normalize = args.ema_normalize and args.max_fps >= 15
    if ema_normalize:
        depth_model.enable_ema(decay=args.ema_decay, buffer_size=args.ema_buffer)

    if (
            args.compile and
            side_model is not None and
            not isinstance(side_model, DeviceSwitchInference) and
            not hasattr(side_model, "compile_context")
    ):
        side_model = compile_model(side_model, device=args.state["device"])

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
                # Don't let a fast/low-fps preview scan overwrite the real cache with
                # lower-quality results.
                if not args.disable_scene_cache and not is_preview:
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
            _hdr_tmp_hevc = path.join(_hdr_out_dir, "_iw3_src.hevc")
            try:
                subprocess.run(
                    [_hdr_ffmpeg_bin, "-y", "-i", str(input_filename),
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

            except subprocess.CalledProcessError as e:
                print(f"--preserve-dowi: source HEVC extraction failed: "
                      f"{e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
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

        return VU.VideoOutputConfig(
            fps=fps,
            container_format=args.video_format,
            video_codec=args.video_codec,
            pix_fmt=pix_fmt,
            colorspace=args.colorspace,
            options=make_video_codec_option(args, input_filename),
            container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},
        )

    if is_video_depth_anything:
        with depth_model.compile_context(enabled=args.compile), try_compile_context(side_model, enabled=args.compile):
            VU.process_video(
                input_filename, output_filename,
                config_callback=config_callback,
                frame_callback=bind_vda_frame_callback(
                    depth_model=depth_model,
                    side_model=side_model,
                    segment_pts=segment_pts,
                    args=args
                ),
                vf=video_filter,
                stop_event=args.state["stop_event"],
                suspend_event=args.state["suspend_event"],
                tqdm_fn=args.state["tqdm_fn"],
                title=path.basename(input_filename),
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
                ),
                vf=video_filter,
                stop_event=args.state["stop_event"],
                suspend_event=args.state["suspend_event"],
                tqdm_fn=args.state["tqdm_fn"],
                title=path.basename(input_filename),
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
            args=args
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
                    title=path.basename(input_filename),
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
        try:
            _inject_hdr_metadata(
                output_filename,
                _hdr_rpu_path, _hdr_h10p_path,
                _hdr_ffmpeg_bin, _hdr_dovi_bin, _hdr_hdr10plus_bin,
                path.dirname(path.abspath(output_filename)),
            )
        except Exception as e:
            print(f"--preserve-dowi: HDR metadata injection failed: {e}", file=sys.stderr)
        finally:
            for f in filter(None, [_hdr_rpu_path, _hdr_h10p_path]):
                try:
                    os.remove(f)
                except Exception:
                    pass


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
                try:
                    _inject_hdr_metadata(output_filename, rpu_path, h10p_path,
                                         fb, dovi_b, h10p_b, out_dir)
                except Exception as e:
                    print(f"--preserve-dowi: HDR injection failed: {e}", file=sys.stderr)
                finally:
                    for f in filter(None, [rpu_path, h10p_path]):
                        try:
                            os.remove(f)
                        except Exception:
                            pass


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
            title=path.basename(input_filename),
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
            depth = depth_model.minmax_normalize_chw(depth)
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


def bind_export_single_frame_callback(depth_model, segment_pts, rgb_dir, depth_dir, pool, args):
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

        pts_queue.clear()
        batch_queue.clear()

        return _postprocess(depth_list)

    @torch.inference_mode()
    def _frame_callback(frame):
        if frame is None:
            # flush
            if batch_queue:
                _batch_infer()
            return _postprocess(depth_model.flush_minmax_normalize())

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
    title = title or path.basename(input_filename)
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
                if not args.disable_scene_cache:
                    save_scene_cache(input_filename, segment_pts, args,
                                      start_time=scan_start_time, end_time=scan_end_time)
                if args.state["stop_event"] is not None and args.state["stop_event"].is_set():
                    return
            gc_collect()
    else:
        segment_pts = set()
    if args.scene_detect_only:
        return

    # TODO: AutoCrop

    config.user_data["scene_boundary"] = ",".join([str(pts).zfill(8) for pts in sorted(list(segment_pts))])

    if args.resume:
        resume_seq = get_resume_seq(depth_dir, rgb_dir) - args.batch_size
    else:
        resume_seq = -1

    if resume_seq > 0 and path.exists(audio_file):
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
    video_filter = add_preprocess_vf(args.vf, args)

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
        depth_model.enable_ema(decay=args.ema_decay, buffer_size=args.ema_buffer)

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
            title=path.basename(base_dir),
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
                                 "forward", "forward_fill", "forward_inpaint",
                                 "mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s",
                                 "mask_mlbw_l2", "mlbw_l2_inpaint",
                                 "row_flow", "row_flow_sym",
                                 "row_flow_v3", "row_flow_v3_sym",
                                 "row_flow_v2",
                                 "NULL"],
                        help="left-right divergence method")
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
                        choices=["film", "animation", "grain", "stillimage", "psnr",
                                 "fastdecode", "zerolatency"],
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


def set_state_args(args, stop_event=None, tqdm_fn=None, depth_model=None, suspend_event=None):
    if depth_model is None:
        depth_model = create_depth_model(args.depth_model)

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

    args.state = {
        "stop_event": stop_event,
        "suspend_event": suspend_event,
        "tqdm_fn": tqdm_fn,
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

    if args.find_param:
        assert is_image(args.input) and (path.isdir(args.output) or not path.exists(args.output))
        find_param(args, depth_model, side_model)
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
