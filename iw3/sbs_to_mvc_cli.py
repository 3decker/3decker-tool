"""python -m iw3.sbs_to_mvc_cli -- turn a side-by-side / top-bottom 3D video into a
real 3D Blu-ray (MVC) .iso that a 3D Blu-ray player or PowerDVD can play.

Pipeline (ADR-182 UPDATE 5/6; each step tried on real clips):
  1. ffmpeg decodes the input, cuts it into the left and right eye, scales each eye
     to 1920x1080 (keeping the picture's real shape, black bars if needed) and pipes
     the two eyes side by side as raw video straight into FRIMEncode -- no big raw
     temp file ever touches the disk.
  2. FRIMEncode (freeware: videofan3d's FRIM 1.31, built on Intel's Media
     SDK) encodes the MVC pair with its SOFTWARE encoder (-sw). Its hardware mode
     ("-hw") fails on current Intel graphics ("undeveloped feature") and there is no
     other free MVC encoder. Output: a base-view and a dependent-view H.264 stream.
  3. tsMuxeR (same bundled tool as the 3D Blu-ray import) muxes both streams, plus any
     Blu-ray-compatible audio (AC-3/DTS/TrueHD/PCM as-is, anything else converted to
     AC-3) and PGS subtitles from the input, into a 3D Blu-ray .iso.

3D Blu-ray only allows 1920x1080 at 23.976/24 fps, so other frame rates are refused
(changing the speed of a movie silently would be worse than saying no).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from os import path

from .mvc_extract_cli import AUTOCROP_MODES, Cancelled, detect_eye_crop, list_tracks
from .utils import _find_tsmuxer, _get_ffmpeg_bin

LAYOUTS = ("full_sbs", "half_sbs", "full_tb", "half_tb")
_BD_FPS = {"23.976": "24000/1001", "24": "24/1"}
_BD_AUDIO = {"A_AC3", "A_EAC3", "A_DTS", "A_TRUEHD", "A_LPCM", "A_MLP"}
FRIM_URL = "https://drive.google.com/uc?export=download&id=1lumXLd74U-E2k195bzfETbHgFcHcT4sH"
FRIM_SHA256 = "76689784495D53B34889F0EA67C9DB6B9750925DB9D1147F8FD9159E111C0778"
# Extra places to fetch the same file from if the author's link stops working. Empty on purpose:
# FRIM has no explicit redistribution permission, so none is hosted by this project. Anything added
# here (or set in the FRIM_MIRROR_URL environment variable) is still only accepted if its SHA-256
# matches FRIM_SHA256, so a mirror can never substitute a different file.
FRIM_MIRRORS = ()


def _root():
    return path.dirname(path.dirname(path.dirname(path.abspath(__file__))))


def find_frim():
    found = shutil.which("FRIMEncode64") or shutil.which("FRIMEncode64.exe")
    if found:
        return found
    candidate = path.join(_root(), "frim", "FRIMEncode64.exe")
    return candidate if path.exists(candidate) else None


_FRIM_FILES = ("FRIMEncode64.exe", "libmfxsw64.dll")


def install_frim(root=None, package_dir=None):
    """Installs the two files the software MVC encoder needs into <root>\\frim.

    Preferred: copy them from nunif\\windows_package\\frim\\ in this repo (hosted with the FRIM
    author's permission, so an install never depends on their download link surviving).
    Fallback if that folder is missing: download FRIM 1.31 from the author's page (or a mirror)
    and accept it only if its SHA-256 matches FRIM_SHA256."""
    root = root or _root()
    dest = path.join(root, "frim")
    if all(path.exists(path.join(dest, n)) for n in _FRIM_FILES):
        return "already installed"
    package_dir = package_dir or path.join(root, "nunif", "windows_package", "frim")
    if all(path.exists(path.join(package_dir, n)) for n in _FRIM_FILES):
        os.makedirs(dest, exist_ok=True)
        for name in _FRIM_FILES:
            shutil.copy2(path.join(package_dir, name), path.join(dest, name))
        return "installed"
    urls = [FRIM_URL] + list(FRIM_MIRRORS)
    if os.environ.get("FRIM_MIRROR_URL"):
        urls.append(os.environ["FRIM_MIRROR_URL"])
    data, errors = None, []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                candidate = resp.read()
        except Exception as e:
            errors.append(f"{url}: {e}")
            continue
        if hashlib.sha256(candidate).hexdigest().upper() != FRIM_SHA256:
            errors.append(f"{url}: downloaded file did not match the expected checksum")
            continue
        data = candidate
        break
    if data is None:
        raise RuntimeError("could not download FRIM from any address:\n  " + "\n  ".join(errors))
    tmp = path.join(root, "tmp", "frim_download")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    archive = path.join(tmp, "frim.rar")
    with open(archive, "wb") as f:
        f.write(data)
    # Windows' own tar (libarchive) reads RAR
    r = subprocess.run(["tar", "-xf", archive, "-C", tmp], capture_output=True, text=True)
    src = path.join(tmp, "x64")
    if r.returncode != 0 or not path.exists(path.join(src, "FRIMEncode64.exe")):
        raise RuntimeError(f"could not unpack FRIM: {(r.stderr or r.stdout).strip()[-300:]}")
    os.makedirs(dest, exist_ok=True)
    for name in ("FRIMEncode64.exe", "libmfxsw64.dll"):
        shutil.copy2(path.join(src, name), path.join(dest, name))
    shutil.rmtree(tmp, ignore_errors=True)
    return "installed"


def _ffprobe_bin():
    ffmpeg = _get_ffmpeg_bin()
    cand = path.join(path.dirname(ffmpeg), "ffprobe.exe") if ffmpeg else None
    if cand and path.exists(cand):
        return cand
    return shutil.which("ffprobe")


def probe_video(input_path):
    """(width, height, fps_string, duration_seconds, is_hdr) of the first video stream."""
    ffprobe = _ffprobe_bin()
    if ffprobe is None:
        raise RuntimeError("ffprobe not found")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,color_transfer:format=duration", "-of", "json", input_path],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"could not read {input_path}: {out.stderr.strip()[-300:]}")
    info = json.loads(out.stdout)
    if not info.get("streams"):
        raise RuntimeError("no video stream found in the input")
    s = info["streams"][0]
    duration = float(info.get("format", {}).get("duration") or 0)
    hdr = s.get("color_transfer") in ("smpte2084", "arib-std-b67")
    return int(s["width"]), int(s["height"]), s["r_frame_rate"], duration, hdr


def bd_frame_rate(rate_string):
    """Maps ffprobe's r_frame_rate to (tsMuxeR/FRIM fps text, frim fraction) or raises."""
    num, den = (rate_string.split("/") + ["1"])[:2]
    fps = float(num) / float(den)
    if abs(fps - 23.976) < 0.02:
        return "23.976", "24000/1001"
    if abs(fps - 24.0) < 0.01:
        return "24", "24/1"
    raise ValueError(f"3D Blu-ray only allows 23.976 or 24 frames per second, but this video is "
                     f"{fps:.3f}. Convert its frame rate first (this tool will not silently change "
                     f"the speed of your movie).")


_BD_FPS_CANDIDATES = (("23.976", "24000/1001", 24000 / 1001), ("24", "24/1", 24.0))


def _nearest_bd_fps(source_fps):
    """Whichever of 23.976/24 is numerically closer to source_fps -> (fps_text, fps_frac, fps_float)."""
    return min(_BD_FPS_CANDIDATES, key=lambda c: abs(c[2] - source_fps))


def _atempo_chain(tempo):
    """ffmpeg's atempo filter only accepts 0.5-2.0 per instance; chain several stages for a
    factor outside that range (e.g. a 60fps source re-timed down to 24fps, tempo=0.4)."""
    stages, t = [], tempo
    while t < 0.5 or t > 2.0:
        stage = 0.5 if t < 0.5 else 2.0
        stages.append(stage)
        t /= stage
    stages.append(t)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def _parse_ffmpeg_out_time(s):
    """Parses ffmpeg -progress's out_time=HH:MM:SS.ffffff field into seconds."""
    h, m, sec = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _intermediate_video_codec_args():
    """hevc_nvenc (GPU) when this machine has it, same convention already used elsewhere in this
    project (iw3.utils._hdr_upscale_codec/_sdr_upscale_codec: probe via PyAV, confirm CUDA too) and
    the same rc/qp quality-control flags make_video_codec_option() already uses for NVENC (NVENC
    ignores -crf; constqp+qp is the correct equivalent). Falls back to CPU libx264 -- these are
    temporary intermediate files re-encoded again by FRIM right afterward, so "faster" (not
    "medium") costs no real quality (CRF/QP already fixes the quality target) while cutting real
    wall-clock time noticeably, which matters since neither of these pre-processing steps has a
    natural frame-count-based progress the way FRIM's own step does."""
    try:
        import av
        av.codec.Codec("hevc_nvenc", "w")
        import torch
        if torch.cuda.is_available():
            return ["-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", "16"]
    except Exception:
        pass
    return ["-c:v", "libx264", "-crf", "16", "-preset", "faster"]


def _run_ffmpeg_stage(cmd, out_path, stage, total_out_seconds, stop_event, progress_cb, error_prefix):
    """Runs an ffmpeg command that already includes -progress pipe:1 (the caller builds `cmd`),
    streaming real progress_cb(stage, done_seconds, total_seconds) updates parsed from its
    out_time= lines as it runs -- shared by every long-running pre-processing pass this module
    needs (retime_to_bd_fps, tonemap_hdr_to_sdr_for_bd) instead of duplicating the same
    subprocess/thread/cancel plumbing in each one. Real user report that drove this: the original
    version of retime_to_bd_fps() called progress_cb once before starting and nothing again until
    it finished or failed, so a user watching an indeterminate spinner for 20+ minutes on a real
    movie reasonably assumed the app had hung. Raises Cancelled() on stop_event, or
    RuntimeError(error_prefix + ffmpeg's own stderr tail) on failure."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err_tail = []

    def _read_stderr():
        for line in proc.stderr:
            err_tail.append(line)
            del err_tail[:-20]

    def _read_stdout():
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time=") and progress_cb and total_out_seconds:
                try:
                    done = min(_parse_ffmpeg_out_time(line.split("=", 1)[1]), total_out_seconds)
                except ValueError:
                    continue
                progress_cb(stage, done, total_out_seconds)

    stderr_reader = threading.Thread(target=_read_stderr, daemon=True)
    stdout_reader = threading.Thread(target=_read_stdout, daemon=True)
    stderr_reader.start()
    stdout_reader.start()
    while proc.poll() is None:
        if stop_event is not None and stop_event.is_set():
            proc.kill()
            raise Cancelled()
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
    stderr_reader.join()
    stdout_reader.join()
    if proc.returncode != 0 or not (path.exists(out_path) and path.getsize(out_path) > 0):
        raise RuntimeError(error_prefix + "".join(err_tail)[-600:])
    if progress_cb and total_out_seconds:
        progress_cb(stage, total_out_seconds, total_out_seconds)


def retime_to_bd_fps(input_path, out_path, source_rate_string, ffmpeg_bin, duration=None,
                      stop_event=None, progress_cb=None):
    """Genuinely re-times input_path -- video, audio (pitch-preserved) and subtitles together --
    to whichever of 23.976/24 fps is numerically closer to its real rate, so the result passes
    bd_frame_rate(). This is a real speed change (like the classic PAL/NTSC conversion), not a
    frame-rate relabel: relabeling alone would just duplicate/drop frames (judder) instead of
    retiming them. Returns the percent speed change applied (negative = slower)."""
    num, den = (source_rate_string.split("/") + ["1"])[:2]
    source_fps = float(num) / float(den)
    fps_text, fps_frac, target_fps = _nearest_bd_fps(source_fps)
    tempo = target_fps / source_fps      # audio speed factor: <1 slower, >1 faster
    stretch = 1.0 / tempo                # video/subtitle timestamp multiplier (same direction)
    total_out_seconds = duration * stretch if duration else None

    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
           "-itsscale:s", f"{stretch:.6f}", "-i", input_path, "-map", "0",
           "-vf", f"setpts={stretch:.6f}*PTS", "-filter:a", _atempo_chain(tempo),
           "-r", fps_frac] + _intermediate_video_codec_args() + [
           "-c:a", "aac", "-b:a", "384k", "-c:s", "copy", out_path]
    _run_ffmpeg_stage(cmd, out_path, "retime", total_out_seconds, stop_event, progress_cb,
                      "could not re-time the video to a 3D Blu-ray-legal frame rate:\n")
    return (tempo - 1.0) * 100.0


# Same zscale+tonemap filter chain as iw3.utils._tonemap_hdr_to_sdr (proven, already used by
# iw3's own main pipeline) -- linearize the PQ/HLG curve, tonemap (Hable operator) into BT.709,
# then re-apply the BT.709 transfer curve for a normal SDR signal. That function's own output
# stays 10-bit (its consumer, iw3's depth pipeline, has no reason to throw bit depth away).
#
# Two real fixes on top of that base chain (confirmed against community-documented working
# ffmpeg HDR-to-SDR pipelines, not guessed): (1) `dither=error_diffusion` on the final zscale
# call -- without it, the actual bit-depth-reducing step (still 16+ bits of internal precision
# down to 8, or even 10) is a bare truncation with no noise-shaping, which is what produces
# visible banding in skies/gradients/dark scenes, exactly the real complaint this was built to
# fix. zscale negotiates its own output format with the adjacent `format=` filter that follows
# it, so `dither=` on zscale actually governs that conversion -- the trailing `format=` filter
# is still required (zscale has no format option of its own), just no longer undithered.
# (2) `sidedata=delete` at the very end strips leftover per-frame HDR side-data (mastering
# display / content-light-level metadata, Dolby Vision RPU) that tonemapping the PIXELS alone
# does not remove -- without it, a player or tool that reads frame side-data rather than just
# the stream-level color_transfer tag this module's own probe_video() checks could still treat
# the output as HDR and apply its own (now-double) tonemap, washing the picture out.
def _hdr_to_sdr_filter(bit_depth):
    fmt = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    return (f"zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            f"tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv:dither=error_diffusion,"
            f"format={fmt},sidedata=delete")


def tonemap_hdr_to_sdr(input_path, out_path, ffmpeg_bin, codec_args, bit_depth=8, audio_args=None,
                       duration=None, stop_event=None, progress_cb=None, stage="tonemap"):
    """General HDR (PQ/HLG -- HDR10, HDR10+, or Dolby Vision's base layer) to SDR conversion,
    parameterized by the caller's own choice of output codec/quality (codec_args) and bit depth
    (8 or 10 -- 10-bit SDR is a real, legitimate choice for general use, it just isn't a legal
    3D Blu-ray input; see tonemap_hdr_to_sdr_for_bd for that specific, hard-capped-at-8-bit
    case). This is a real, one-way change to the picture (the HDR grade is genuinely gone
    afterward, tone-mapped down to a normal range), not just a metadata strip."""
    cmd = ([ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
            "-i", input_path, "-map", "0", "-vf", _hdr_to_sdr_filter(bit_depth)] + codec_args +
           (audio_args or ["-c:a", "copy"]) + ["-c:s", "copy", out_path])
    _run_ffmpeg_stage(cmd, out_path, stage, duration, stop_event, progress_cb,
                      "could not convert the HDR video to SDR:\n")


def tonemap_hdr_to_sdr_for_bd(input_path, out_path, ffmpeg_bin, duration=None,
                               stop_event=None, progress_cb=None):
    """Converts an HDR (PQ/HLG, e.g. HDR10 or Dolby Vision) source to plain 8-bit SDR, since
    3D Blu-ray (MVC) cannot carry HDR or Dolby Vision at all -- there is no combination of the
    classic MVC/H.264 3D Blu-ray format with HDR10/HDR10+, and Dolby Vision's own 3D-capable
    profile (Profile 20, MV-HEVC) is a completely different, modern codec that today only the
    Apple Vision Pro can play -- no 3D Blu-ray player, PowerDVD, or TV supports it, and no
    physical Blu-ray disc has ever shipped with it. 8-bit specifically because BD-ROM (including
    3D/MVC) is hard-capped at 8-bit H.264 High Profile -- there is no 10-bit extension to that
    format, unlike UHD Blu-ray (which supports neither MVC nor 3D at all)."""
    tonemap_hdr_to_sdr(input_path, out_path, ffmpeg_bin, _intermediate_video_codec_args(), bit_depth=8,
                       audio_args=["-c:a", "copy"], duration=duration, stop_event=stop_event,
                       progress_cb=progress_cb, stage="tonemap")


def guess_layout(width, height):
    """A starting suggestion only -- the user picks the real layout."""
    if height > width:
        return "full_tb" if height >= 2000 else "half_tb"
    if width >= height * 2.6:
        return "full_sbs"
    return "half_sbs"


def eye_only_filter(layout):
    """ffmpeg -vf that keeps just the LEFT eye of a packed frame (used to look for black bars)."""
    return "crop=iw/2:ih:0:0" if layout in ("full_sbs", "half_sbs") else "crop=iw:ih/2:0:0"


def eye_filter(layout, width, height, crop=None):
    """ffmpeg -vf chain: cut both eyes out of the input frame, scale each to fit
    1920x1080 keeping its true picture shape (half layouts store a squeezed picture),
    pad with black, and put them side by side (3840x1080 raw for FRIM's -sbs 2).
    With `crop` = (x, y, w, h) (in one eye's own pixels) the black bars are cut off each eye
    first. The disc frame is still 1920x1080, so the bars are put back around the fitted
    picture -- auto-crop mainly tidies uneven or noisy edges, it cannot shrink a Blu-ray frame."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {LAYOUTS}")
    if layout in ("full_sbs", "half_sbs"):
        eye_w, eye_h = width // 2, height
        crop_l, crop_r = f"crop={eye_w}:{eye_h}:0:0", f"crop={eye_w}:{eye_h}:{eye_w}:0"
    else:
        eye_w, eye_h = width, height // 2
        crop_l, crop_r = f"crop={eye_w}:{eye_h}:0:0", f"crop={eye_w}:{eye_h}:0:{eye_h}"
    if crop is not None:
        cx, cy, cw, ch = crop
        extra = f",crop={cw}:{ch}:{cx}:{cy}"
        crop_l, crop_r = crop_l + extra, crop_r + extra
        eye_w, eye_h = cw, ch
    if layout == "half_sbs":
        disp_w, disp_h = eye_w * 2, eye_h
    elif layout == "half_tb":
        disp_w, disp_h = eye_w, eye_h * 2
    else:
        disp_w, disp_h = eye_w, eye_h
    scale = min(1920 / disp_w, 1080 / disp_h)
    w = min(1920, max(2, round(disp_w * scale / 2) * 2))
    h = min(1080, max(2, round(disp_h * scale / 2) * 2))
    fit = f"scale={w}:{h}:flags=lanczos,pad=1920:1080:{(1920 - w) // 2}:{(1080 - h) // 2},setsar=1"
    return f"split[a][b];[a]{crop_l},{fit}[l];[b]{crop_r},{fit}[r];[l][r]hstack"


# A Blu-ray only holds picture subtitles (PGS). Text subtitles (SRT/ASS/... -- what most MKV files carry) are
# turned into PGS by tsMuxeR itself, which draws the text with a font; the font must have the language's
# letters, so the non-Latin languages get their own Windows font (everything else: Arial).
_SUB_FONT_BY_LANG = {
    "chi": "Microsoft YaHei", "zho": "Microsoft YaHei", "jpn": "Yu Gothic", "kor": "Malgun Gothic",
    "tha": "Leelawadee UI", "hin": "Nirmala UI", "ben": "Nirmala UI", "tam": "Nirmala UI", "tel": "Nirmala UI",
    "ara": "Segoe UI", "heb": "Segoe UI", "per": "Segoe UI", "fas": "Segoe UI", "urd": "Segoe UI",
}
_MAX_BD_SUBTITLES = 32          # the Blu-ray limit for picture-subtitle streams
_SUB_STYLE = "font-size=65, font-color=0xffffffff, bottom-offset=24, font-border=5, text-align=center"


def _text_sub_meta(srt_path, lang, fps_text, width, height):
    font = _SUB_FONT_BY_LANG.get((lang or "").lower(), "Arial")
    lang_part = f", lang={lang}" if lang else ""
    return (f'S_TEXT/UTF8, "{srt_path.replace(chr(92), "/")}", font-name="{font}", {_SUB_STYLE}, '
            f"video-width={width}, video-height={height}, fps={fps_text}{lang_part}")


def _plan_audio_subs(input_path, tsmuxer_bin, work_dir, ffmpeg_bin, include_av, fps_text="23.976",
                     width=1920, height=1080):
    """Returns (meta_lines, notes). Compatible audio goes in as-is, anything else is
    converted to AC-3 (a Blu-ray-legal format) first; PGS subtitles go in as-is; text subtitles
    (SRT/ASS/...) are extracted to .srt and rendered into Blu-ray subtitles by tsMuxeR (up to the disc's
    limit of 32); only bitmap formats a Blu-ray cannot hold (e.g. VobSub) are skipped, with a note."""
    lines, notes = [], []
    if not include_av:
        return lines, notes
    src = path.abspath(input_path).replace(chr(92), "/")
    audio_index = 0
    subtitle_index = 0          # position among ALL subtitle streams, same order ffmpeg's 0:s:N uses
    subtitle_count = 0
    for t in list_tracks(input_path, tsmuxer_bin):
        codec = t["codec"]
        lang = f", lang={t['lang']}" if t["lang"] else ""
        if codec.startswith("A_"):
            if codec in _BD_AUDIO:
                lines.append(f"{codec}, {src}, track={t['id']}{lang}")
            else:
                out = path.join(work_dir, f"audio_{audio_index}.ac3")
                converted = False
                # AC-3 holds up to 5.1 channels; retry as 5.1 if the source has more (7.1)
                for extra in ([], ["-ac", "6"]):
                    r = subprocess.run([ffmpeg_bin, "-y", "-v", "error", "-i", input_path, "-map",
                                        f"0:a:{audio_index}", "-vn", "-c:a", "ac3", "-b:a", "640k"] + extra + [out],
                                       capture_output=True, text=True)
                    if r.returncode == 0 and path.exists(out):
                        converted = True
                        break
                if converted:
                    lines.append(f"A_AC3, {out.replace(chr(92), '/')}{lang}")
                    notes.append(f"audio track {audio_index + 1} ({codec}) converted to AC-3")
                else:
                    notes.append(f"audio track {audio_index + 1} ({codec}) skipped: could not convert it")
            audio_index += 1
        elif codec.startswith("S_"):
            label = f"subtitle track {subtitle_index + 1} ({t['lang'] or 'unknown language'}, {codec})"
            if subtitle_count >= _MAX_BD_SUBTITLES:
                notes.append(f"{label} skipped: a Blu-ray holds at most {_MAX_BD_SUBTITLES} subtitle tracks")
            elif codec == "S_HDMV/PGS":
                lines.append(f"{codec}, {src}, track={t['id']}{lang}")
                subtitle_count += 1
            elif codec.startswith("S_TEXT"):
                out = path.join(work_dir, f"subtitle_{subtitle_index}.srt")
                r = subprocess.run([ffmpeg_bin, "-y", "-v", "error", "-i", input_path, "-map",
                                    f"0:s:{subtitle_index}", "-c:s", "srt", out],
                                   capture_output=True, text=True)
                if r.returncode == 0 and path.exists(out) and path.getsize(out) > 0:
                    lines.append(_text_sub_meta(out, t["lang"], fps_text, width, height))
                    subtitle_count += 1
                    notes.append(f"{label} converted to a Blu-ray picture subtitle")
                else:
                    notes.append(f"{label} skipped: could not extract its text")
            else:
                notes.append(f"{label} skipped: this subtitle format cannot be put on a Blu-ray")
            subtitle_index += 1
    return lines, notes


def convert(input_path, output_iso, layout="full_sbs", bitrate_mbps=20.0, swap_eyes=False,
            include_av=True, work_dir=None, cut_seconds=None, keep_temp=False,
            stop_event=None, progress_cb=None, autocrop=None, fix_frame_rate=False,
            convert_hdr_to_sdr=False):
    """Full job. Returns the number of frames encoded. progress_cb(stage, done, total),
    stage in {"tonemap", "retime", "autocrop", "encode", "mux"}. fix_frame_rate: if the source
    isn't 23.976/24fps, re-time the whole movie (video+audio+subtitles) to the nearer one instead
    of refusing -- opt-in only, since it's a real (if usually tiny) speed change.
    convert_hdr_to_sdr: if the source is HDR (HDR10/Dolby Vision/HLG), tone-map it to plain SDR
    instead of refusing -- opt-in only, since the HDR grade is genuinely gone afterward (3D
    Blu-ray/MVC cannot carry HDR at all, so there is no way to keep it either way)."""
    frim = find_frim()
    tsmuxer = _find_tsmuxer()
    ffmpeg = _get_ffmpeg_bin()
    if frim is None:
        raise RuntimeError("FRIMEncode not found -- run `python -m iw3.install_mvc_tools`")
    if tsmuxer is None:
        raise RuntimeError("tsMuxeR not found -- run `python -m iw3.install_mvc_tools`")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    if not path.exists(input_path):
        raise RuntimeError(f"input file not found: {input_path}")
    if path.splitext(output_iso)[1].lower() != ".iso":
        raise ValueError("the output must be an .iso file")
    if not 2 <= bitrate_mbps <= 40:
        raise ValueError("bitrate must be between 2 and 40 Mbps (3D Blu-ray allows about 40 combined)")

    width, height, rate, duration, hdr = probe_video(input_path)
    if hdr and not convert_hdr_to_sdr:
        raise RuntimeError("HDR video is not supported for 3D Blu-ray here -- convert it to SDR first")
    if layout in ("full_sbs", "half_sbs") and (width < 2 or width % 2):
        raise ValueError("a side-by-side video needs an even width")

    out_dir = path.dirname(path.abspath(output_iso))
    os.makedirs(out_dir, exist_ok=True)
    stem = path.splitext(path.basename(output_iso))[0]
    work_dir = work_dir or path.join(out_dir, f"_sbs2mvc_work_{stem}")
    created_work = not path.isdir(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    tonemapped_path = None
    if hdr:
        if progress_cb:
            progress_cb("tonemap", 0, 0)
        tonemapped_path = path.join(work_dir, "tonemapped_sdr.mkv")
        tonemap_hdr_to_sdr_for_bd(input_path, tonemapped_path, ffmpeg, duration=duration,
                                  stop_event=stop_event, progress_cb=progress_cb)
        print("[sbs2mvc] note: source is HDR (HDR10/Dolby Vision/HLG); converted to SDR before "
              "continuing -- 3D Blu-ray cannot carry HDR at all, so the HDR grade is genuinely "
              "gone in this file", file=sys.stderr)
        input_path = tonemapped_path
        width, height, rate, duration, hdr = probe_video(input_path)

    retimed_path = None
    try:
        fps_text, fps_frac = bd_frame_rate(rate)
    except ValueError:
        if not fix_frame_rate:
            if created_work:
                # rmtree, not rmdir: tonemapped_sdr.mkv may already be sitting in here if
                # convert_hdr_to_sdr ran above, so the directory isn't necessarily empty.
                shutil.rmtree(work_dir, ignore_errors=True)
            raise
        if progress_cb:
            progress_cb("retime", 0, 0)
        retimed_path = path.join(work_dir, "retimed_input.mkv")
        percent = retime_to_bd_fps(input_path, retimed_path, rate, ffmpeg, duration=duration,
                                   stop_event=stop_event, progress_cb=progress_cb)
        print(f"[sbs2mvc] note: source frame rate is not 3D Blu-ray-legal; re-timed the whole "
              f"movie (picture, sound and subtitles together, {percent:+.2f}% speed) to match",
              file=sys.stderr)
        input_path = retimed_path
        width, height, rate, duration, hdr = probe_video(input_path)
        fps_text, fps_frac = bd_frame_rate(rate)

    if duration <= 0:
        raise RuntimeError("could not read the video's length")
    seconds = min(duration, cut_seconds) if cut_seconds else duration
    total_frames = int(seconds * float(fps_text))

    crop = None
    if autocrop:
        if progress_cb:
            progress_cb("autocrop", 0, 1)
        crop = detect_eye_crop(input_path, autocrop, vf=eye_only_filter(layout))
        if crop is None:
            print("[sbs2mvc] auto-crop: no black bars found; nothing cropped", file=sys.stderr)
        else:
            print(f"[sbs2mvc] auto-crop: each eye cut to x={crop[0]} y={crop[1]} {crop[2]}x{crop[3]}",
                  file=sys.stderr)

    # streams ~ bitrate*length*1.5 (dependent view is smaller than the base view), then the
    # ISO is about the same again
    est_streams = bitrate_mbps * 1e6 / 8 * seconds * 1.5
    free = shutil.disk_usage(work_dir).free
    if free < est_streams * 2.2:
        if created_work:
            # rmtree, not rmdir: a re-timed intermediate (retimed_input.mkv) may already be
            # sitting in here if fix_frame_rate ran above, so the directory isn't empty anymore.
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"not enough free disk space: need about {est_streams * 2.2 / 1e9:.0f} GB "
                           f"(temporary streams plus the ISO), only {free / 1e9:.0f} GB free")

    base_es, dep_es = path.join(work_dir, "base.264"), path.join(work_dir, "dep.264")
    meta_path = path.join(work_dir, "mux.meta")
    ffmpeg_log = path.join(work_dir, "ffmpeg.log")
    procs, ok = [], False
    try:
        vf = eye_filter(layout, width, height, crop)
        ff_cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if cut_seconds:
            ff_cmd += ["-t", str(cut_seconds)]
        ff_cmd += ["-i", input_path, "-an", "-sn", "-vf", vf, "-pix_fmt", "yuv420p",
                   "-r", fps_frac, "-f", "rawvideo", "-"]
        target = int(bitrate_mbps * 1000)
        frim_cmd = [frim, "-i", "-", "-o:mvc", base_es, dep_es, "-viewoutput", "-sbs", "2",
                    "-w", "1920", "-h", "1080", "-f", fps_frac, "-profile", "high", "-level", "4.1",
                    "-vbr", str(target), str(int(target * 1.25)), "-sw"]
        if swap_eyes:
            frim_cmd.append("-swaplr")

        with open(ffmpeg_log, "wb") as ff_err:
            ff = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=ff_err)
            procs.append(ff)
            fr = subprocess.Popen(frim_cmd, stdin=ff.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            procs.append(fr)
            ff.stdout.close()
            tail = []

            def _read():
                buf = b""
                while True:
                    chunk = fr.stdout.read(256)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        m = re.search(rb"[\r\n]", buf)
                        if not m:
                            break
                        line, buf = buf[:m.start()], buf[m.end():]
                        tail.append(line.decode(errors="replace"))
                        del tail[:-20]
                        fm = re.search(rb"Frame number:\s*(\d+)", line)
                        if fm and progress_cb:
                            progress_cb("encode", min(int(fm.group(1)), total_frames), total_frames)

            reader = threading.Thread(target=_read, daemon=True)
            reader.start()
            while fr.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    for p in procs:
                        p.kill()
                    raise Cancelled()
                try:
                    fr.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            reader.join()
            ff.wait()

        if fr.returncode != 0 or not (path.exists(base_es) and path.getsize(base_es) > 0
                                       and path.exists(dep_es) and path.getsize(dep_es) > 0):
            with open(ffmpeg_log, "rb") as f:
                ff_msg = f.read()[-600:].decode(errors="replace")
            raise RuntimeError("the MVC encode failed:\n" + "\n".join(tail[-8:]) + ("\nffmpeg: " + ff_msg if ff_msg.strip() else ""))

        av_lines, notes = ([], [])
        if include_av:
            av_lines, notes = _plan_audio_subs(input_path, tsmuxer, work_dir, ffmpeg, True, fps_text=fps_text)
        for n in notes:
            print(f"[sbs2mvc] note: {n}", file=sys.stderr)

        fwd = lambda p: p.replace(chr(92), "/")  # noqa: E731
        meta = [f"MUXOPT --blu-ray --auto-chapters=10",
                f"V_MPEG4/ISO/AVC, {fwd(base_es)}, fps={fps_text}, insertSEI, contSPS",
                f"V_MPEG4/ISO/MVC, {fwd(dep_es)}, fps={fps_text}, insertSEI, contSPS"] + av_lines
        with open(meta_path, "w", encoding="utf-8", newline="\n") as f:  # no BOM: tsMuxeR rejects one
            f.write("\n".join(meta) + "\n")

        if progress_cb:
            progress_cb("mux", 0, 100)
        mux = subprocess.Popen([tsmuxer, meta_path, output_iso], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
        procs.append(mux)
        mux_tail = []
        for line in mux.stdout:
            mux_tail.append(line.rstrip())
            del mux_tail[:-12]
            m = re.search(r"(\d+(?:\.\d+)?)%", line)
            if m and progress_cb:
                progress_cb("mux", float(m.group(1)), 100)
            if stop_event is not None and stop_event.is_set():
                mux.kill()
                raise Cancelled()
        mux.wait()
        if mux.returncode != 0:
            raise RuntimeError("tsMuxeR failed:\n" + "\n".join(mux_tail))
        ok = True
        return total_frames
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        if not ok:
            try:
                os.remove(output_iso)
            except OSError:
                pass
        if not keep_temp:
            if created_work:
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                for leftover in (base_es, dep_es, meta_path, ffmpeg_log, retimed_path, tonemapped_path):
                    if leftover is None:
                        continue
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True, help="the 3D video (side-by-side or top-bottom)")
    parser.add_argument("--output", "-o", required=True, help="the 3D Blu-ray .iso to write")
    parser.add_argument("--layout", choices=LAYOUTS, default=None,
                        help="how the eyes are stored in the input (default: guessed from its size)")
    parser.add_argument("--bitrate", type=float, default=20.0,
                        help="target Mbps per view (default 20; 3D Blu-ray allows about 40 combined)")
    parser.add_argument("--swap-eyes", action="store_true", help="the input is right-eye-first (cross-eyed)")
    parser.add_argument("--autocrop", type=str.upper, default=None, choices=AUTOCROP_MODES,
                        help="remove black bars from each eye before fitting: BLACK = all sides, BLACK_TB = "
                             "top/bottom only (FLAT / FLAT_TB for flat-colour borders)")
    parser.add_argument("--no-audio-subs", action="store_true", help="video only")
    parser.add_argument("--fix-frame-rate", action="store_true",
                        help="if the source isn't 23.976/24fps, re-time the whole movie (picture, sound "
                             "and subtitles together) to the nearer one instead of refusing -- this is a "
                             "real, if usually small, speed change, so it is opt-in, never automatic")
    parser.add_argument("--convert-hdr-to-sdr", action="store_true",
                        help="if the source is HDR (HDR10/Dolby Vision/HLG), tone-map it to plain SDR "
                             "instead of refusing -- 3D Blu-ray cannot carry HDR at all, so this is a real, "
                             "one-way loss of the HDR grade, so it is opt-in, never automatic")
    parser.add_argument("--cut-seconds", type=float, default=None, help="only convert the first N seconds")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--gui-progress", action="store_true",
                        help="print 'IW3_MVC_PROGRESS <stage> <done> <total>' lines to stdout")
    args = parser.parse_args()

    if args.gui_progress:
        def show(stage, done, total):
            print(f"IW3_MVC_PROGRESS {stage} {done} {total}", flush=True)
    else:
        def show(stage, done, total):
            print(f"\r[sbs2mvc] {stage}: {done}/{total}      ", end="", file=sys.stderr, flush=True)
    try:
        layout = args.layout
        if layout is None:
            w, h, *_ = probe_video(args.input)
            layout = guess_layout(w, h)
            print(f"[sbs2mvc] layout not given; guessed {layout} from {w}x{h}", file=sys.stderr)
        frames = convert(args.input, args.output, layout=layout, bitrate_mbps=args.bitrate,
                         swap_eyes=args.swap_eyes, include_av=not args.no_audio_subs,
                         work_dir=args.work_dir, cut_seconds=args.cut_seconds, keep_temp=args.keep_temp,
                         progress_cb=show, autocrop=args.autocrop, fix_frame_rate=args.fix_frame_rate,
                         convert_hdr_to_sdr=args.convert_hdr_to_sdr)
    except Cancelled:
        print("\n[sbs2mvc] cancelled", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    print(f"\n[sbs2mvc] done: {args.output} ({frames} frames)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
