"""python -m iw3.mvc_extract_cli -- standalone 3D Blu-ray MVC extraction tool.

Pulls the REAL stereoscopic video out of a real 3D Blu-ray disc (mounted ISO
or an already-extracted disc folder), producing a standard side-by-side video
file iw3 (or any normal player) can treat as ordinary input.

Real 3D Blu-ray discs store video in TWO places that are easy to confuse:
  BDMV/STREAM/<N>.m2ts       -- the 2D-compatible base view ONLY (for players
                                 that don't understand 3D). NOT what this tool
                                 wants -- confirmed directly on a real disc
                                 that this file alone has zero MVC data.
  BDMV/STREAM/SSIF/<N>.ssif  -- the REAL interleaved base+dependent stream.
                                 THIS is the file this tool needs.

Pipeline (each step verified against a real disc, not just synthetic test
streams -- see docs/ai/AI_DECISIONS.md ADR-182):
  1. tsMuxeR (bundled, github.com/justdan96/tsMuxer, Apache-2.0) demuxes the
     .ssif with --demux, once for each view's codec name (V_MPEG4/ISO/AVC for
     the base view, V_MPEG4/ISO/MVC for the dependent view), from the SAME
     track ID -- producing two separate elementary streams.
  2. This tool re-interleaves them access-unit by access-unit (base AU N,
     then dependent AU N, for every N) into one combined MVC bitstream --
     tsMuxeR has no documented way to produce this combined form directly.
  3. edge264-mvc (bundled, github.com/jens-duttke/edge264-mvc, BSD-3-Clause)
     decodes the combined stream with -O (side-by-side output) and -k
     (REQUIRED -- real 3D Blu-rays carry a type-24 NAL per access unit that
     edge264 deliberately reports as unsupported and halts on without -k,
     confirmed directly: omitting -k stopped decoding before ever reaching
     the dependent view, at bit-identical structural correctness otherwise).
  4. edge264-mvc's raw Y4M output is piped DIRECTLY into ffmpeg for encoding
     -- never written to disk as raw video. A real 15-second test produced a
     2.2GB raw Y4M file; a full ~90-minute movie at that same rate would be
     roughly 800GB, which is not practical. Piping keeps disk usage to just
     the final compressed output.
"""
import argparse
import os
import re
import subprocess
import sys
import threading
from os import path

# ADR-253: same fix as sbs_to_mvc_cli.py -- this module also runs as its own
# separate `python -m iw3.mvc_extract_cli` process (the "3D Blu-ray Import"
# standalone tool) and makes several of its own subprocess calls (tsMuxeR demux,
# edge264-mvc decode piped into ffmpeg, mkvmerge mux) that would each flash their
# own console window without this patch, which only iw3/gui.py's own process
# imported before now.
import nunif.gui.subprocess_patch  # noqa

from .utils import _find_tsmuxer, _find_edge264_mvc, _get_ffmpeg_bin, _find_mkvmerge


def _find_video_track(mpls_or_m2ts_path, tsmuxer_bin):
    """Runs tsMuxeR's track-detection mode (single positional argument, no
    output) against a real disc's .mpls playlist -- NOT a plain .m2ts clip,
    which never shows the MVC track at all, confirmed directly on a real
    disc -- and returns (avc_track_id, mvc_track_id) or (None, None) if this
    title has no MVC track (a genuinely 2D-only disc, or the wrong playlist)."""
    result = subprocess.run([tsmuxer_bin, str(mpls_or_m2ts_path)], capture_output=True, text=True)
    avc_track = mvc_track = None
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Track ID:"):
            track_id = int(line.split(":")[1].strip())
            # the Stream ID line follows 2 lines later
            for j in range(i, min(i + 4, len(lines))):
                if "V_MPEG4/ISO/MVC" in lines[j]:
                    mvc_track = track_id
                elif "V_MPEG4/ISO/AVC" in lines[j]:
                    avc_track = track_id
    return avc_track, mvc_track


_CHUNK_SIZE = 64 * 1024 * 1024  # 64MB
_OVERLAP = 8  # a start code (3 bytes) + NAL header (1 byte) never needs more lookback than this


def _find_au_boundaries(path, delimiter_types):
    """Scans an elementary stream FILE (not an in-memory buffer) in bounded
    64MB chunks for access-unit start offsets, each at one of the given
    delimiter NAL types -- confirmed directly against real extracted streams
    (not assumed from spec alone): base-view access units start with a
    type-9 (AUD) NAL; dependent-view access units start with a type-24 NAL.
    Returns a list of (start, end) byte-offset pairs -- integers only, no
    file data is retained here.

    ADR-182: two earlier versions of this function both used too much memory
    on a real ~90-minute movie (14GB base + 9.5GB dependent view): reading
    each whole file into a plain bytes object pushed a real 63GB-RAM machine
    down to ~2GB available; switching to mmap.mmap() was better in theory
    (file-backed pages the OS can discard without a pagefile write) but its
    reported working set still climbed to 12GB+ with available memory
    dropping to ~300MB before being killed as too risky to trust unattended
    -- confirmed via live Get-Counter monitoring both times, not assumed.
    This version reads fixed 64MB chunks via plain seek+read, keeping only
    the current chunk (plus an 8-byte overlap to catch a start code split
    across a chunk boundary) in memory at any time -- a small, predictable,
    genuinely bounded footprint regardless of file size."""
    starts = []
    with open(path, "rb") as f:
        file_offset = 0
        carry = b""
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            buf = carry + chunk
            buf_base_offset = file_offset - len(carry)

            idx = 0
            while True:
                idx = buf.find(b"\x00\x00\x01", idx)
                if idx == -1 or idx + 3 >= len(buf):
                    break
                nal_type = buf[idx + 3] & 0x1F
                if nal_type in delimiter_types:
                    starts.append(buf_base_offset + idx)
                idx += 3

            file_offset += len(chunk)
            carry = buf[-_OVERLAP:] if len(buf) >= _OVERLAP else buf

    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()

    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else file_size
        bounds.append((s, e))
    return bounds


def interleave_mvc(base_path, dependent_path, out_path):
    """Recombines a separately-demuxed base (AVC) and dependent (MVC) view
    elementary stream into one combined bitstream a real MVC decoder can
    read. Returns (base_au_count, dependent_au_count, written_au_count).

    Bounded memory (see _find_au_boundaries' own docstring for why): reads
    and writes exactly one access unit's worth of data (typically well under
    1MB, confirmed against real extracted streams) at a time via seek+read,
    never the whole file."""
    base_bounds = _find_au_boundaries(base_path, delimiter_types={9})
    dep_bounds = _find_au_boundaries(dependent_path, delimiter_types={24, 9})

    n = min(len(base_bounds), len(dep_bounds))
    with open(base_path, "rb") as bf, open(dependent_path, "rb") as df, open(out_path, "wb") as out:
        for i in range(n):
            bs, be = base_bounds[i]
            ds, de = dep_bounds[i]
            bf.seek(bs)
            out.write(bf.read(be - bs))
            df.seek(ds)
            out.write(df.read(de - ds))

    return len(base_bounds), len(dep_bounds), n


LAYOUTS = ("full_sbs", "half_sbs", "full_tb", "half_tb",
           "full_sbs_4k", "half_sbs_4k", "full_tb_4k", "half_tb_4k", "frame_packed")
CODECS = ("hevc_nvenc", "libx265", "libx264")

# Only libx264 can write the H.264 Frame Packing SEI that lets a 3D TV/player
# auto-detect the layout (ADR-181). x265 defines the SEI type but exposes no way
# to set it; NVENC has no equivalent at all.
_SEI_TYPE = {"half_sbs": 3, "half_tb": 4, "half_sbs_4k": 3, "half_tb_4k": 4, "frame_packed": 4}

_COLOR_TAG = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"

# 4K layouts: each 1080p eye is ENLARGED (Lanczos -- no new detail; use the Upscale tool for AI
# upscaling) to the per-eye size below, then the eyes are stacked. name -> (eye_w, eye_h, stack filter)
_LAYOUTS_4K = {
    "full_sbs_4k": (3840, 2160, "hstack"),   # 7680x2160
    "half_sbs_4k": (1920, 2160, "hstack"),   # 3840x2160, each eye squeezed to half width
    "full_tb_4k": (3840, 2160, "vstack"),    # 3840x4320
    "half_tb_4k": (3840, 1080, "vstack"),    # 3840x2160, each eye squeezed to half height
}

_SPLIT_EYES = "split[a][b];[a]crop=iw/2:ih:0:0[l];[b]crop=iw/2:ih:iw/2:0[r];[l][r]vstack"


AUTOCROP_MODES = ("BLACK", "BLACK_TB", "FLAT", "FLAT_TB")


def detect_eye_crop(video_path, mode, vf=""):
    """Runs iw3's own AutoCrop analysis (the same detector as the main conversion's Auto Crop) on
    `video_path` and returns (x, y, w, h) of the picture inside ONE eye, or None when there
    is nothing to remove. `vf` can isolate one eye when the video is a packed 3D frame.

    iw3's helper samples KEYFRAMES only; a short or sparse-keyframe video can deliver none and
    then silently report "no bars". So if the keyframe pass sees nothing, sample ordinary frames."""
    import torch
    import nunif.utils.video as VU
    from nunif.utils.autocrop import AutoCrop, AutoCropDetector

    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def quiet_tqdm(**kwargs):  # keep progress-bar noise out of the tool's log
        return tqdm(**{**kwargs, "disable": True})

    def analyze(keyframe_only):
        model = AutoCropDetector(mode=mode.upper(), mod=2)
        size = [0, 0]

        def on_batch(x):
            size[0] = max(size[0], x.shape[-2])
            size[1] = max(size[1], x.shape[-1])
            model.update(x)

        pool = VU.FrameCallbackPool(on_batch, batch_size=2, device=device, max_workers=0)
        VU.sample_frames(video_path, pool, num_samples=40, keyframe_only=keyframe_only, vf=vf,
                         device=device, title="AutoCrop Analysis", tqdm_fn=quiet_tqdm)
        return model, size

    model, (height, width) = analyze(True)
    if model.frame_count == 0:
        model, (height, width) = analyze(False)
    if model.frame_count == 0:
        return None
    slice_h, slice_w = model.get_crop()
    return AutoCrop.calc_crop(slice_h, slice_w, height, width)


def _even(v):
    return max(2, int(round(v / 2)) * 2)


def _cropped_layout_filter(layout, crop):
    """Same layouts as layout_filter(), but each eye is first cut down to the picture
    rectangle `crop` = (x, y, w, h) (black bars removed, same rectangle for both eyes).
    The 4K layouts keep the picture's true shape: the width is fixed by the layout and the
    height follows from the cropped picture's aspect ratio (a bar-free 2.39:1 eye becomes
    3840x1608, not a stretched 3840x2160)."""
    x, y, w, h = crop
    is_4k = layout.endswith("_4k")
    geometry = layout[:-3] if is_4k else layout
    if geometry == "frame_packed":
        geometry = "full_tb"
    if is_4k:
        nat_h = _even(3840 * h / w)
        sizes = {"full_sbs": (3840, nat_h), "half_sbs": (1920, nat_h),
                 "full_tb": (3840, nat_h), "half_tb": (3840, _even(nat_h / 2))}
    else:
        sizes = {"full_sbs": (w, h), "half_sbs": (_even(w / 2), h),
                 "full_tb": (w, h), "half_tb": (w, _even(h / 2))}
    eye_w, eye_h = sizes[geometry]
    stack = "hstack" if geometry.endswith("sbs") else "vstack"
    scale = "" if (eye_w, eye_h) == (w, h) else f",scale={eye_w}:{eye_h}:flags=lanczos"
    return (f"split[a][b];[a]crop=iw/2:ih:0:0,crop={w}:{h}:{x}:{y}{scale}[l];"
            f"[b]crop=iw/2:ih:iw/2:0,crop={w}:{h}:{x}:{y}{scale}[r];[l][r]{stack}")


def layout_filter(layout, crop=None):
    """ffmpeg -vf chain turning edge264's full-width side-by-side frame (left eye |
    right eye, each at full 1920x1080) into the requested layout. None = no change.
    With `crop` (x, y, w, h) each eye is cut to that picture rectangle first."""
    if crop is not None:
        if layout not in LAYOUTS:
            raise ValueError(f"unknown layout {layout!r}; choose from {LAYOUTS}")
        return _cropped_layout_filter(layout, crop)
    if layout == "full_sbs":
        return None
    if layout == "half_sbs":
        return "scale=iw/2:ih:flags=lanczos"
    if layout in ("full_tb", "frame_packed"):
        return _SPLIT_EYES
    if layout == "half_tb":
        return _SPLIT_EYES + ",scale=iw:ih/2:flags=lanczos"
    if layout in _LAYOUTS_4K:
        w, h, stack = _LAYOUTS_4K[layout]
        return (f"split[a][b];[a]crop=iw/2:ih:0:0,scale={w}:{h}:flags=lanczos[l];"
                f"[b]crop=iw/2:ih:iw/2:0,scale={w}:{h}:flags=lanczos[r];[l][r]{stack}")
    raise ValueError(f"unknown layout {layout!r}; choose from {LAYOUTS}")


def encoder_args(codec, quality, layout):
    """ffmpeg output-side video arguments for the chosen codec/quality/layout.
    `quality` is CRF for the software encoders and the equivalent constant-quality
    (-cq) level for NVENC -- lower is better quality and a bigger file."""
    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec!r}; choose from {CODECS}")
    if codec == "hevc_nvenc":
        # -maxrate/-bufsize are REQUIRED for -cq to work below ~20: without them NVENC silently caps the
        # bitrate near 19 Mbps, so Quality 8, 14 and 18 all produced the same file (measured on a lossless
        # 15 s sample: cq14 34.6 MB capped vs 66.9 MB with the ceiling raised). 100 Mbps is only a safety
        # ceiling; the Quality number decides the real bitrate.
        args = ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0",
                "-maxrate", "100M", "-bufsize", "200M"]
    elif codec == "libx265":
        args = ["-c:v", "libx265", "-crf", str(quality)]
    else:
        args = ["-c:v", "libx264", "-crf", str(quality)]
        if layout in _SEI_TYPE:
            args += ["-x264-params", f"frame-packing={_SEI_TYPE[layout]}"]
    # The Rec.709 tags are applied by _COLOR_TAG in the filter chain, NOT with ffmpeg's
    # -colorspace/-color_* output flags: those make ffmpeg CONVERT the picture from an
    # assumed BT.601 to BT.709 (measured: luma only 37.9 dB against a bit-exact decode),
    # while setparams only labels it (bit-exact, verified).
    args += ["-pix_fmt", "yuv420p"]
    return args


class Cancelled(Exception):
    pass


def _remove_stale_temp(*paths):
    """Deletes each path if it exists, retrying briefly on a locked-file error before
    giving up silently. Used both to clear out a reused work_dir before a new demux
    starts and for normal end-of-job cleanup.

    Real gap this closes: a Cancelled/force-killed job (taskkill /T /F from the GUI's
    Cancel button) never runs this module's own `finally` cleanup at all -- a killed
    process can't run its own cleanup code -- so a large (~15-25 GB) partial demux can
    be left behind in work_dir. Since work_dir is named only from the output filename
    (not a timestamp or the job's settings), a later retry with the same output name
    silently reuses that same dirty folder: either it runs low on disk space partway
    through, or -- if a killed child process (tsMuxeR/edge264/ffmpeg) hadn't yet fully
    released a file handle at the moment taskkill returned, a real possible race, not
    guaranteed instant on Windows -- the new tsMuxeR demux fails outright trying to
    overwrite a file still locked by the OS. Both look like a random, unrelated
    failure to whatever the user happened to also change when retrying (a real user
    report described it as seemingly tied to video codec choice, which the actual
    demux step never even reads). The retry loop below specifically targets that
    locked-file race window, which is normally milliseconds, not the disk-space case
    (nothing to retry there -- the preflight check ahead of every real call site is
    what actually guards against that)."""
    import time as _time
    for p in paths:
        for attempt in range(5):
            try:
                os.remove(p)
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt == 4:
                    break
                _time.sleep(0.2)


def _stream_interleaved(stdin, base_path, base_bounds, dep_path, dep_bounds, n, stop_event, errors):
    """Writes base AU i then dependent AU i, for every i, straight into the decoder's
    stdin -- the combined ~24GB stream never exists on disk or in memory."""
    try:
        with open(base_path, "rb") as bf, open(dep_path, "rb") as df:
            for i in range(n):
                if stop_event is not None and stop_event.is_set():
                    return
                bs, be = base_bounds[i]
                ds, de = dep_bounds[i]
                bf.seek(bs)
                stdin.write(bf.read(be - bs))
                df.seek(ds)
                stdin.write(df.read(de - ds))
    except (BrokenPipeError, OSError) as e:
        errors.append(e)
    finally:
        try:
            stdin.close()
        except OSError:
            pass


def _read_ffmpeg_progress(stream, tail, total_frames, progress_cb):
    """Reads ffmpeg's stderr (progress lines end in \\r, not \\n), keeps the last
    few KB for error reporting, and reports the running frame number."""
    buf = b""
    while True:
        chunk = stream.read(512)
        if not chunk:
            break
        buf += chunk
        while True:
            m = re.search(rb"[\r\n]", buf)
            if not m:
                break
            line, buf = buf[:m.start()], buf[m.end():]
            tail.append(line)
            del tail[:-40]
            fm = re.search(rb"frame=\s*(\d+)", line)
            if fm and progress_cb is not None:
                progress_cb("encode", int(fm.group(1)), total_frames)


def _restore_disc_av(ssif_path, video_only, output_path, cut_start, cut_end, progress_cb):
    """Adds every audio and subtitle track of the disc's normal .m2ts clip (which sits
    one folder above SSIF/ and, unlike the video-only demux, carries them) to the
    encoded video, using the same engine as the main pipeline's Restore Audio &
    Subtitles option. If that clip is missing or the mux fails, the video-only file
    is kept as the result and a warning is printed -- never lose the encode."""
    from types import SimpleNamespace
    from . import av_restore_cli

    stem = path.splitext(path.basename(ssif_path))[0]
    m2ts = path.join(path.dirname(path.dirname(path.abspath(ssif_path))), stem + ".m2ts")
    if not path.exists(m2ts):
        print(f"[mvc-extract] WARNING: {m2ts} not found; output has no audio/subtitles", file=sys.stderr)
        os.replace(video_only, output_path)
        return
    if progress_cb:
        progress_cb("restore", 0, 1)

    def secs(t):
        return None if t is None else str(t).rstrip("s")

    start = secs(cut_start)
    args = SimpleNamespace(input=video_only, source=m2ts, output=output_path,
                           source_start_time=None if start in (None, "0") else start,
                           source_end_time=secs(cut_end))
    if av_restore_cli.run(args) != 0:
        print("[mvc-extract] WARNING: audio/subtitle restore failed; keeping the video-only result",
              file=sys.stderr)
        os.replace(video_only, output_path)
    if progress_cb:
        progress_cb("restore", 1, 1)


def extract_and_decode(ssif_path, avc_track, mvc_track, cut_start, cut_end, work_dir, output_path,
                        video_codec="hevc_nvenc", quality=18, layout="full_sbs",
                        restore_av=True, keep_temp=False, stop_event=None, progress_cb=None,
                        autocrop=None):
    """Full pipeline: tsMuxeR demux -> interleave (streamed) -> edge264-mvc decode -> ffmpeg encode.

    Two things never touch disk: the combined MVC stream (interleaved on the fly
    straight into the decoder's stdin) and the raw decoded video (edge264's stdout
    is piped directly into ffmpeg; a full movie would be ~800GB raw). Only the two
    demuxed elementary streams (~24GB for a full film) are temporary files, and
    they are deleted afterwards unless keep_temp is set.

    Feeding the decoder through stdin (not letting it memory-map a file) matters
    twice: edge264's own Windows file-size call wrapped at 4GB (see ADR-182 UPDATE
    3), and a 25GB mapped view drove the machine low on memory.

    progress_cb(stage, done, total) is called with stage in
    {"demux", "scan", "encode"}; stop_event (threading.Event) cancels cleanly."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {LAYOUTS}")
    if layout in _SEI_TYPE and video_codec != "libx264":
        if layout == "frame_packed":
            print(f"[mvc-extract] frame_packed needs libx264 (only it can write the Frame Packing SEI); "
                  f"switching from {video_codec}", file=sys.stderr)
            video_codec = "libx264"
        # half_sbs/half_tb are still valid with other codecs -- just no auto-detect flag

    tsmuxer_bin = _find_tsmuxer()
    edge264_bin = _find_edge264_mvc()
    ffmpeg_bin = _get_ffmpeg_bin()
    if tsmuxer_bin is None:
        raise RuntimeError("tsMuxeR not found -- see docs/ai/AI_DECISIONS.md ADR-182")
    if edge264_bin is None:
        raise RuntimeError("edge264-mvc not found -- see docs/ai/AI_DECISIONS.md ADR-182")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found")

    if restore_av and path.splitext(output_path)[1].lower() != ".mkv":
        raise ValueError("restoring the disc's audio/subtitles needs an .mkv output path")
    video_only = (path.splitext(output_path)[0] + ".video_only.mkv") if restore_av else output_path
    crop = None
    if autocrop:
        # analyse the disc's own 2D-compatible clip (BDMV/STREAM/<N>.m2ts = the left-eye picture);
        # the packed 3D stream is only decoded later
        stem_name = path.splitext(path.basename(ssif_path))[0]
        m2ts = path.join(path.dirname(path.dirname(path.abspath(ssif_path))), stem_name + ".m2ts")
        if path.exists(m2ts):
            if progress_cb:
                progress_cb("autocrop", 0, 1)
            crop = detect_eye_crop(m2ts, autocrop)
            if crop is None:
                print("[mvc-extract] auto-crop: no black bars found; nothing cropped", file=sys.stderr)
            else:
                print(f"[mvc-extract] auto-crop: each eye cut to x={crop[0]} y={crop[1]} {crop[2]}x{crop[3]}",
                      file=sys.stderr)
        else:
            print(f"[mvc-extract] WARNING: {m2ts} not found; auto-crop skipped", file=sys.stderr)

    os.makedirs(work_dir, exist_ok=True)
    meta_path = path.join(work_dir, "_mvc_extract.meta")
    base_es = path.join(work_dir, f"{path.splitext(path.basename(ssif_path))[0]}.track_{avc_track}.264")
    dep_es = path.join(work_dir, f"{path.splitext(path.basename(ssif_path))[0]}.track_{mvc_track}.mvc")
    edge_log = path.join(work_dir, "_edge264_stderr.log")
    procs = []
    # Clear out anything a previous, force-killed attempt at this same output path left
    # behind before starting a new demux into the same work_dir -- see
    # _remove_stale_temp()'s own docstring for the real bug this closes.
    _remove_stale_temp(base_es, dep_es, edge_log)
    try:
        cut_opts = f" --cut-start={cut_start} --cut-end={cut_end}" if cut_end else f" --cut-start={cut_start}"
        ssif_meta = path.abspath(ssif_path).replace(chr(92), "/")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"MUXOPT --demux{cut_opts}\n")
            f.write(f"V_MPEG4/ISO/AVC, {ssif_meta}, track={avc_track}\n")
            f.write(f"V_MPEG4/ISO/MVC, {ssif_meta}, track={mvc_track}\n")

        if progress_cb:
            progress_cb("demux", 0, 1)
        demux = subprocess.Popen([tsmuxer_bin, meta_path, work_dir],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(demux)
        out_lines = []
        for line in demux.stdout:
            out_lines.append(line)
            m = re.search(r"(\d+(?:\.\d+)?)%", line)
            if m and progress_cb:
                progress_cb("demux", float(m.group(1)), 100)
            if stop_event is not None and stop_event.is_set():
                demux.kill()
                raise Cancelled()
        demux.wait()
        if demux.returncode != 0:
            raise RuntimeError(f"tsMuxeR demux failed: {''.join(out_lines)[-2000:]}")

        if progress_cb:
            progress_cb("scan", 0, 1)
        base_bounds = _find_au_boundaries(base_es, delimiter_types={9})
        dep_bounds = _find_au_boundaries(dep_es, delimiter_types={24, 9})
        n = min(len(base_bounds), len(dep_bounds))
        if len(base_bounds) != len(dep_bounds):
            print(f"[mvc-extract] WARNING: base AUs={len(base_bounds)} != dependent AUs={len(dep_bounds)}; "
                  f"using the first {n}", file=sys.stderr)
        print(f"[mvc-extract] base AUs={len(base_bounds)} dependent AUs={len(dep_bounds)} using={n}",
              file=sys.stderr)

        vf = layout_filter(layout, crop)
        ffmpeg_args = [ffmpeg_bin, "-y", "-hide_banner", "-i", "-",
                       "-vf", f"{vf},{_COLOR_TAG}" if vf else _COLOR_TAG]
        ffmpeg_args += encoder_args(video_codec, quality, layout) + [video_only]

        # edge264 reads "-" (stdin) and writes Y4M to stdout; ffmpeg reads that pipe.
        # The parent closes its copy of edge264's stdout so a dead ffmpeg is noticed.
        with open(edge_log, "wb") as edge_err:
            edge264_proc = subprocess.Popen([edge264_bin, "-", "-O", "-k", "-y"],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=edge_err)
            procs.append(edge264_proc)
            ffmpeg_proc = subprocess.Popen(ffmpeg_args, stdin=edge264_proc.stdout,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            procs.append(ffmpeg_proc)
            edge264_proc.stdout.close()

            feed_errors = []
            feeder = threading.Thread(target=_stream_interleaved, daemon=True,
                                       args=(edge264_proc.stdin, base_es, base_bounds, dep_es, dep_bounds,
                                             n, stop_event, feed_errors))
            tail = []
            reader = threading.Thread(target=_read_ffmpeg_progress, daemon=True,
                                       args=(ffmpeg_proc.stderr, tail, n, progress_cb))
            feeder.start()
            reader.start()
            while ffmpeg_proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    for p in (ffmpeg_proc, edge264_proc):
                        p.kill()
                    raise Cancelled()
                try:
                    ffmpeg_proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            feeder.join()
            reader.join()
            edge264_proc.wait()

        ffmpeg_tail = b"\n".join(tail).decode(errors="replace")
        if ffmpeg_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed:\n{ffmpeg_tail[-3000:]}")
        frames = 0
        for line in reversed(tail):
            fm = re.search(rb"frame=\s*(\d+)", line)
            if fm:
                frames = int(fm.group(1))
                break
        # edge264 exits 1 after decoding every frame of a demuxed stream (no formal
        # end-of-stream NAL). Judge success by frames delivered, not its exit code.
        if edge264_proc.returncode not in (0, 1) or frames < n - 2:
            with open(edge_log, "rb") as f:
                edge_msg = f.read()[-2000:].decode(errors="replace")
            raise RuntimeError(f"decode incomplete: {frames} of {n} frames encoded "
                               f"(edge264 exit {edge264_proc.returncode}, feeder errors {feed_errors}):\n{edge_msg}")
        if restore_av:
            _restore_disc_av(ssif_path, video_only, output_path, cut_start, cut_end, progress_cb)
        return frames
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        if not keep_temp:
            _remove_stale_temp(base_es, dep_es, meta_path, edge_log, *((video_only,) if restore_av else ()))


ISO_LAYOUT = "bd3d_iso"


def list_tracks(ssif_path, tsmuxer_bin):
    """Every track tsMuxeR sees in the .ssif, as dicts: id, codec (e.g. V_MPEG4/ISO/MVC,
    A_AC3, S_HDMV/PGS), stream_type (tsMuxeR's own free-text label, e.g. "TRUE-HD" for a
    TrueHD/Atmos track muxed under the A_AC3 codec -- see ADR-262) and lang (3-letter
    code or "")."""
    out = subprocess.run([tsmuxer_bin, str(ssif_path)], capture_output=True, text=True).stdout
    tracks, cur = [], None
    for line in out.splitlines():
        if line.startswith("Track ID:"):
            cur = {"id": int(line.split(":", 1)[1].strip()), "codec": "", "stream_type": "", "lang": ""}
            tracks.append(cur)
        elif cur is not None and line.startswith("Stream type:"):
            cur["stream_type"] = line.split(":", 1)[1].strip()
        elif cur is not None and line.startswith("Stream ID:"):
            cur["codec"] = line.split(":", 1)[1].strip()
        elif cur is not None and line.startswith("Stream lang:"):
            cur["lang"] = line.split(":", 1)[1].strip()
    return tracks


def mux_bd3d_iso(ssif_path, out_iso, include_av=True, stop_event=None, progress_cb=None, cut_end=None):
    """Lossless copy: writes the disc's own base+dependent 3D video (untouched, no
    re-encode) plus, if include_av, every audio and subtitle track, into a new 3D
    Blu-ray .iso that a 3D Blu-ray player or PowerDVD can play (tsMuxeR builds the
    BDMV structure incl. the SSIF file -- only when the output is an .iso, per its
    docs). Chapters are added every 10 minutes. Returns the number of tracks written."""
    tsmuxer_bin = _find_tsmuxer()
    if tsmuxer_bin is None:
        raise RuntimeError("tsMuxeR not found -- run `python -m iw3.install_mvc_tools`")
    if path.splitext(out_iso)[1].lower() != ".iso":
        raise ValueError("a 3D Blu-ray copy must be saved as an .iso file")
    tracks = list_tracks(ssif_path, tsmuxer_bin)
    if not any(t["codec"] == "V_MPEG4/ISO/MVC" for t in tracks):
        raise RuntimeError("no 3D (MVC) video track found in this disc's main movie")
    wanted = [t for t in tracks if t["codec"].startswith("V_") or (include_av and t["codec"][:2] in ("A_", "S_"))]
    # ADR-258 found tsMuxeR does correctly DETECT every track on a real disc (ruled out
    # in ADR-261's real end-to-end test); ADR-262 found the REAL bug, one level deeper:
    # for a TrueHD/Atmos track (tsMuxeR labels it Stream type "TRUE-HD" even though its
    # muxing codec is the generic A_AC3), tsMuxeR's --blu-ray disc-AUTHORING step itself
    # corrupts the audio data while regenerating a brand-new BDMV/CLPI structure -- real,
    # decoded-audio evidence: ffmpeg decoding 30s of the produced track threw dozens of
    # "Invalid nonrestart_substr" errors and only yielded 4.56s of actual audio (heard
    # by the reporting user as fast/"chipmunk" playback), while the SAME 30s decoded
    # perfectly from the untouched source disc. This is why extract_and_decode()/
    # mux_lossless_mvc_mkv() never hit this: both pull audio via av_restore_cli.py's
    # ffmpeg-based stream copy from the plain .m2ts instead, never routing it through
    # tsMuxeR's disc-authoring engine at all. tsMuxeR's own --help documents a per-track
    # "down-to-ac3" option specifically for TRUE-HD tracks ("Filter out HD part") --
    # confirmed by direct testing (real disc, real decode) to produce a clean, fully
    # decodable 384Kbps 5.1 AC3 core with zero errors. Applied automatically below for
    # any TRUE-HD track: real, working AC3-core audio beats a silently corrupted
    # "lossless" TrueHD/Atmos track in every case, but it IS a real quality tradeoff
    # (lossless video is unaffected -- this only downgrades that one audio track), so
    # it's printed clearly, not silently swapped.
    print(f"[mvc-extract] tsMuxeR detected {len(tracks)} track(s) on this disc:", file=sys.stderr)
    for t in tracks:
        note = f" (lang={t['lang']})" if t["lang"] else ""
        print(f"[mvc-extract]   track {t['id']}: {t['codec']}{note}", file=sys.stderr)
    if include_av and not any(t["codec"].startswith("A_") for t in tracks):
        print("[mvc-extract] WARNING: tsMuxeR reported ZERO audio tracks on this disc -- if you know "
             "this disc has audio, please report this with the track list printed just above.",
             file=sys.stderr)
    ssif_meta = path.abspath(ssif_path).replace(chr(92), "/")
    cut = f" --cut-start=0s --cut-end={cut_end}" if cut_end else ""
    lines = [f"MUXOPT --blu-ray --auto-chapters=10{cut}"]
    for t in wanted:
        extra = f", lang={t['lang']}" if t["lang"] else ""
        if t.get("stream_type") == "TRUE-HD":
            extra += ", down-to-ac3"
            print(f"[mvc-extract]   NOTE: track {t['id']} is TrueHD/Atmos -- tsMuxeR's disc-authoring "
                 "step corrupts the full HD extension when rebuilding a new disc structure (ADR-262), "
                 "so only its AC3 core (384Kbps 5.1) is included -- real, working audio instead of a "
                 "silently broken \"lossless\" track.", file=sys.stderr)
        lines.append(f"{t['codec']}, {ssif_meta}, track={t['id']}{extra}")
    os.makedirs(path.dirname(path.abspath(out_iso)), exist_ok=True)
    meta_path = path.splitext(path.abspath(out_iso))[0] + ".mux.meta"
    # Clear out a stale meta file (and any partial .iso) a previous force-killed
    # attempt at this exact output path left behind -- see _remove_stale_temp()'s
    # own docstring for the real bug this closes.
    _remove_stale_temp(meta_path, out_iso)
    # no BOM: tsMuxeR rejects a UTF-8 byte-order mark on the first line
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    proc = None
    ok = False
    try:
        proc = subprocess.Popen([tsmuxer_bin, meta_path, out_iso], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in proc.stdout:
            tail.append(line.rstrip())
            del tail[:-15]
            m = re.search(r"(\d+(?:\.\d+)?)%", line)
            if m and progress_cb:
                progress_cb("mux", float(m.group(1)), 100)
            if stop_event is not None and stop_event.is_set():
                proc.kill()
                raise Cancelled()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("tsMuxeR failed:\n" + "\n".join(tail))
        ok = True
        return len(wanted)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        _remove_stale_temp(meta_path, *((out_iso,) if not ok else ()))


MVC_MKV_LAYOUT = "mvc_mkv"


def mux_lossless_mvc_mkv(ssif_path, avc_track, mvc_track, out_mkv, include_av=True, cut_start="0s",
                         cut_end=None, work_dir=None, keep_temp=False, stop_event=None, progress_cb=None):
    """Lossless copy, matching what MakeMKV/CloneBD produce from a real 3D Blu-ray:
    the disc's own base+dependent MVC video, re-interleaved into ONE real combined
    MVC elementary stream (the exact same reconstruction extract_and_decode() builds
    before decoding it -- see interleave_mvc()'s docstring for why tsMuxeR alone
    can't produce this form), muxed directly into a plain .mkv with NO decode and NO
    re-encode. A real MVC-aware player's own H.264 decoder (not this tool) does the
    actual stereo reconstruction at playback time -- exactly like a disc ripped with
    MakeMKV/CloneBD's own "MVC" option, for a library that (unlike this project's own
    default SBS/TB outputs) plays real MVC files directly.

    Real user request: someone already using this tool's Lossless 3D Blu-ray ISO
    layout was separately re-ripping that ISO with MakeMKV/CloneBD themselves just to
    get an MVC .mkv for their library, then having to manually re-mux it with a
    source file's audio because the ISO (before ADR-239) had none. This layout does
    that whole job directly, in one step, video AND audio/subtitles together.

    NOT verified end to end against a real disc in a real MVC-capable player this
    session (no 3D Blu-ray source was available to test with). What IS confirmed
    directly: mkvmerge accepts a plain H.264 elementary stream as an ordinary file
    argument (not stdin -- tested directly, mkvmerge has no stdin support at all,
    contrary to the first assumption) and produces a working container from it, using
    the exact `--default-duration <TID>:<fps>` idiom already proven elsewhere in this
    codebase (see nunif/utils/video/processor.py's _remux_injected_hevc, which mixes
    a raw elementary stream with pre-existing audio the same way). What is NOT
    confirmed: whether mkvmerge also carries the MVC extension NAL units (subset SPS
    type 15, coded slice extension type 20) through untouched rather than stripping
    or choking on them, since only a real disc's genuine MVC bitstream contains them
    -- flagged here and to the user; needs a real disc and a real MVC-capable player
    (e.g. Kodi with MVC-aware hardware decode) to fully confirm.

    progress_cb(stage, done, total), stage in {"demux", "interleave", "mux",
    "restore"}; stop_event (threading.Event) cancels cleanly."""
    tsmuxer_bin = _find_tsmuxer()
    mkvmerge_bin = _find_mkvmerge()
    if tsmuxer_bin is None:
        raise RuntimeError("tsMuxeR not found -- run `python -m iw3.install_mvc_tools`")
    if mkvmerge_bin is None:
        raise RuntimeError("mkvmerge not found -- it ships in this project's own mkvtoolnix/ folder")
    if path.splitext(out_mkv)[1].lower() != ".mkv":
        raise ValueError("the lossless MVC output must be saved as an .mkv file")

    from .sbs_to_mvc_cli import probe_video
    stem = path.splitext(path.basename(ssif_path))[0]
    # the disc's normal 2D-compatible clip sits one folder above SSIF/ (see
    # _restore_disc_av's own docstring) -- unlike the raw demuxed elementary streams,
    # it has real container-level timing this tool can trust for the combined
    # stream's frame rate (a 3D Blu-ray's dependent view always shares the base
    # view's frame rate, so probing either clip gives the same real answer).
    m2ts = path.join(path.dirname(path.dirname(path.abspath(ssif_path))), stem + ".m2ts")
    if not path.exists(m2ts):
        raise RuntimeError(f"{m2ts} not found -- can't determine the disc's real frame rate")
    _, _, rate, _, _ = probe_video(m2ts)
    try:
        num, den = rate.split("/")
        fps_str = f"{int(num) / int(den):.6f}fps"
    except (ValueError, ZeroDivisionError):
        fps_str = f"{rate}fps"

    stem_out = path.splitext(path.basename(out_mkv))[0]
    work_dir = work_dir or path.join(path.dirname(path.abspath(out_mkv)), f"_mvc_work_{stem_out}")
    os.makedirs(work_dir, exist_ok=True)
    meta_path = path.join(work_dir, "_mvc_extract.meta")
    base_es = path.join(work_dir, f"{stem}.track_{avc_track}.264")
    dep_es = path.join(work_dir, f"{stem}.track_{mvc_track}.mvc")
    combined_es = path.join(work_dir, f"{stem}.combined_mvc.264")
    video_only = (path.splitext(out_mkv)[0] + ".video_only.mkv") if include_av else out_mkv

    procs = []
    # Clear out anything a previous, force-killed attempt at this same output path left
    # behind before starting a new demux into the same work_dir -- see
    # _remove_stale_temp()'s own docstring for the real bug this closes.
    _remove_stale_temp(base_es, dep_es, combined_es)
    try:
        cut_opts = f" --cut-start={cut_start} --cut-end={cut_end}" if cut_end else f" --cut-start={cut_start}"
        ssif_meta = path.abspath(ssif_path).replace(chr(92), "/")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"MUXOPT --demux{cut_opts}\n")
            f.write(f"V_MPEG4/ISO/AVC, {ssif_meta}, track={avc_track}\n")
            f.write(f"V_MPEG4/ISO/MVC, {ssif_meta}, track={mvc_track}\n")

        if progress_cb:
            progress_cb("demux", 0, 1)
        demux = subprocess.Popen([tsmuxer_bin, meta_path, work_dir],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(demux)
        out_lines = []
        for line in demux.stdout:
            out_lines.append(line)
            m = re.search(r"(\d+(?:\.\d+)?)%", line)
            if m and progress_cb:
                progress_cb("demux", float(m.group(1)), 100)
            if stop_event is not None and stop_event.is_set():
                demux.kill()
                raise Cancelled()
        demux.wait()
        if demux.returncode != 0:
            raise RuntimeError(f"tsMuxeR demux failed: {''.join(out_lines)[-2000:]}")

        if progress_cb:
            progress_cb("interleave", 0, 1)
        base_n, dep_n, n = interleave_mvc(base_es, dep_es, combined_es)
        if base_n != dep_n:
            print(f"[mvc-extract] WARNING: base AUs={base_n} != dependent AUs={dep_n}; using the first {n}",
                 file=sys.stderr)
        if progress_cb:
            progress_cb("interleave", 1, 1)
        if stop_event is not None and stop_event.is_set():
            raise Cancelled()

        if progress_cb:
            progress_cb("mux", 0, 1)
        mux = subprocess.Popen([mkvmerge_bin, "-o", video_only, "--default-duration", f"0:{fps_str}", combined_es],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(mux)
        mux_lines = []
        for line in mux.stdout:
            mux_lines.append(line)
            m = re.search(r"Progress:\s*(\d+)%", line)
            if m and progress_cb:
                progress_cb("mux", float(m.group(1)), 100)
            if stop_event is not None and stop_event.is_set():
                mux.kill()
                raise Cancelled()
        mux.wait()
        if mux.returncode != 0:
            raise RuntimeError(f"mkvmerge failed: {''.join(mux_lines)[-2000:]}")

        if include_av:
            _restore_disc_av(ssif_path, video_only, out_mkv, cut_start, cut_end, progress_cb)
        return n
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        if not keep_temp:
            _remove_stale_temp(base_es, dep_es, combined_es, meta_path,
                               *((video_only,) if include_av else ()))


def _find_frim():
    """Local copy of sbs_to_mvc_cli.py's own find_frim() -- duplicated rather than
    imported: sbs_to_mvc_cli.py already imports FROM this module (AUTOCROP_MODES,
    Cancelled, detect_eye_crop, interleave_mvc), so importing back from it at module
    load time would be circular. A tiny, self-contained binary locator, not
    nontrivial logic that could drift out of sync -- see extract_and_reencode_mvc()'s
    own docstring for the (lazy, function-body) import used for the real shared math
    (eye_filter/bd_frame_rate/probe_video), the same pattern mux_lossless_mvc_mkv()
    already established for probe_video() alone."""
    import shutil
    found = shutil.which("FRIMEncode64") or shutil.which("FRIMEncode64.exe")
    if found:
        return found
    root = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    candidate = path.join(root, "frim", "FRIMEncode64.exe")
    return candidate if path.exists(candidate) else None


MVC_MKV_CROPPED_LAYOUT = "mvc_mkv_cropped"


def extract_and_reencode_mvc(ssif_path, avc_track, mvc_track, cut_start, cut_end, work_dir, output_path,
                             bitrate_mbps=20.0, autocrop=None, swap_eyes=False, include_av=True,
                             keep_temp=False, stop_event=None, progress_cb=None):
    """ADR-259: real user request -- go directly from a real 3D Blu-ray disc to a
    fresh, auto-cropped MVC .mkv in ONE re-encode, instead of the two-step
    workaround (extract_and_decode() to a flat, cropped SBS file, then a completely
    separate sbs_to_mvc_cli.convert() pass to turn THAT into MVC again). Removing
    black bars always needs a real re-encode -- a compressed video's own bytes can't
    be cropped without decoding and re-encoding it -- this does that exactly once
    instead of twice, by feeding the decoded, cropped video straight into FRIMEncode
    instead of writing an intermediate flat video file first.

    Same real limitation as every other auto-crop path in this project (see
    sbs_to_mvc_cli.eye_filter()'s own docstring): a real MVC/Blu-ray-spec video is
    locked to exactly 1920x1080 per eye, so a genuinely non-16:9 movie still gets
    padded back out to that size after cropping -- this tidies/tightens bars, it
    does not guarantee a bar-free picture for real widescreen content. Confirmed
    directly with the user before building this.

    Pipeline: tsMuxeR demux -> interleave (streamed) -> edge264-mvc decode -> ffmpeg
    (crop ONLY, raw video out, no actual encode) -> FRIMEncode (the real MVC
    re-encode) -> interleave_mvc() (recombine FRIMEncode's own fresh base+dependent
    output into one real MVC bitstream, same function mux_lossless_mvc_mkv() already
    uses for the lossless case) -> mkvmerge (mux) -> _restore_disc_av() (the disc's
    real audio/subtitles, same engine every other real-disc path here already uses).

    eye_filter()/bd_frame_rate()/probe_video() are lazy-imported from
    sbs_to_mvc_cli.py inside this function body (not at module top) -- the same
    circular-import-avoidance pattern mux_lossless_mvc_mkv() already established for
    probe_video() alone, extended here to the crop math too rather than duplicating it.

    progress_cb(stage, done, total), stage in {"demux", "autocrop", "encode",
    "interleave", "mux", "restore"}; stop_event (threading.Event) cancels cleanly.
    Not verified end to end against a real disc in a real MVC-capable player this
    session -- same real-hardware-verification gap mux_lossless_mvc_mkv() already
    flags for its own (unmodified, still lossless) MVC muxing step, which this reuses."""
    from .sbs_to_mvc_cli import eye_filter, bd_frame_rate, probe_video

    tsmuxer_bin = _find_tsmuxer()
    edge264_bin = _find_edge264_mvc()
    ffmpeg_bin = _get_ffmpeg_bin()
    mkvmerge_bin = _find_mkvmerge()
    frim_bin = _find_frim()
    if tsmuxer_bin is None:
        raise RuntimeError("tsMuxeR not found -- see docs/ai/AI_DECISIONS.md ADR-182")
    if edge264_bin is None:
        raise RuntimeError("edge264-mvc not found -- see docs/ai/AI_DECISIONS.md ADR-182")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found")
    if mkvmerge_bin is None:
        raise RuntimeError("mkvmerge not found -- it ships in this project's own mkvtoolnix/ folder")
    if frim_bin is None:
        raise RuntimeError("FRIMEncode not found -- run `python -m iw3.install_mvc_tools`")
    if path.splitext(output_path)[1].lower() != ".mkv":
        raise ValueError("this direct-to-MVC output must be saved as an .mkv file")
    if not 2 <= bitrate_mbps <= 40:
        raise ValueError("bitrate must be between 2 and 40 Mbps (3D Blu-ray allows about 40 combined)")

    stem = path.splitext(path.basename(ssif_path))[0]
    m2ts = path.join(path.dirname(path.dirname(path.abspath(ssif_path))), stem + ".m2ts")
    if not path.exists(m2ts):
        raise RuntimeError(f"{m2ts} not found -- can't determine the disc's real frame rate/dimensions")
    eye_width, eye_height, rate, _, _ = probe_video(m2ts)
    # the disc's own 2D-compatible clip is exactly one eye's own picture -- edge264's
    # raw decode output is the full side-by-side frame, i.e. both eyes, twice the width.
    sbs_width, sbs_height = eye_width * 2, eye_height
    _, fps_frac = bd_frame_rate(rate)
    # fps_frac is always one of bd_frame_rate()'s own two hardcoded "num/den" strings
    # ("24000/1001" or "24/1") -- mkvmerge's --default-duration wants a decimal "fps"
    # string instead, same conversion mux_lossless_mvc_mkv() already does from its own
    # probed rate.
    _fps_num, _fps_den = fps_frac.split("/")
    mkv_fps_str = f"{int(_fps_num) / int(_fps_den):.6f}fps"

    crop = None
    if autocrop:
        if progress_cb:
            progress_cb("autocrop", 0, 1)
        crop = detect_eye_crop(m2ts, autocrop)
        if crop is None:
            print("[mvc-extract] auto-crop: no black bars found; nothing cropped", file=sys.stderr)
        else:
            print(f"[mvc-extract] auto-crop: each eye cut to x={crop[0]} y={crop[1]} {crop[2]}x{crop[3]}",
                 file=sys.stderr)

    os.makedirs(work_dir, exist_ok=True)
    meta_path = path.join(work_dir, "_mvc_extract.meta")
    base_es_src = path.join(work_dir, f"{stem}.track_{avc_track}.264")
    dep_es_src = path.join(work_dir, f"{stem}.track_{mvc_track}.mvc")
    edge_log = path.join(work_dir, "_edge264_stderr.log")
    frim_base_es = path.join(work_dir, f"{stem}.frim_base.264")
    frim_dep_es = path.join(work_dir, f"{stem}.frim_dep.264")
    combined_es = path.join(work_dir, f"{stem}.combined_mvc.264")
    video_only = (path.splitext(output_path)[0] + ".video_only.mkv") if include_av else output_path
    procs = []
    _remove_stale_temp(base_es_src, dep_es_src, edge_log, frim_base_es, frim_dep_es, combined_es)
    try:
        cut_opts = f" --cut-start={cut_start} --cut-end={cut_end}" if cut_end else f" --cut-start={cut_start}"
        ssif_meta = path.abspath(ssif_path).replace(chr(92), "/")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"MUXOPT --demux{cut_opts}\n")
            f.write(f"V_MPEG4/ISO/AVC, {ssif_meta}, track={avc_track}\n")
            f.write(f"V_MPEG4/ISO/MVC, {ssif_meta}, track={mvc_track}\n")

        if progress_cb:
            progress_cb("demux", 0, 1)
        demux = subprocess.Popen([tsmuxer_bin, meta_path, work_dir],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(demux)
        out_lines = []
        for line in demux.stdout:
            out_lines.append(line)
            m = re.search(r"(\d+(?:\.\d+)?)%", line)
            if m and progress_cb:
                progress_cb("demux", float(m.group(1)), 100)
            if stop_event is not None and stop_event.is_set():
                demux.kill()
                raise Cancelled()
        demux.wait()
        if demux.returncode != 0:
            raise RuntimeError(f"tsMuxeR demux failed: {''.join(out_lines)[-2000:]}")

        base_bounds = _find_au_boundaries(base_es_src, delimiter_types={9})
        dep_bounds = _find_au_boundaries(dep_es_src, delimiter_types={24, 9})
        n = min(len(base_bounds), len(dep_bounds))
        if len(base_bounds) != len(dep_bounds):
            print(f"[mvc-extract] WARNING: base AUs={len(base_bounds)} != dependent AUs={len(dep_bounds)}; "
                  f"using the first {n}", file=sys.stderr)
        print(f"[mvc-extract] base AUs={len(base_bounds)} dependent AUs={len(dep_bounds)} using={n}",
              file=sys.stderr)

        vf = eye_filter("full_sbs", sbs_width, sbs_height, crop)
        ff_cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", "-",
                 "-an", "-sn", "-vf", vf, "-pix_fmt", "yuv420p", "-r", fps_frac,
                 "-f", "rawvideo", "-"]
        target = int(bitrate_mbps * 1000)
        frim_cmd = [frim_bin, "-i", "-", "-o:mvc", frim_base_es, frim_dep_es, "-viewoutput", "-sbs", "2",
                   "-w", "1920", "-h", "1080", "-f", fps_frac, "-profile", "high", "-level", "4.1",
                   "-vbr", str(target), str(int(target * 1.25)), "-sw"]
        if swap_eyes:
            frim_cmd.append("-swaplr")

        if progress_cb:
            progress_cb("encode", 0, 1)
        with open(edge_log, "wb") as edge_err:
            edge264_proc = subprocess.Popen([edge264_bin, "-", "-O", "-k", "-y"],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=edge_err)
            procs.append(edge264_proc)
            ffmpeg_proc = subprocess.Popen(ff_cmd, stdin=edge264_proc.stdout,
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            procs.append(ffmpeg_proc)
            edge264_proc.stdout.close()
            frim_proc = subprocess.Popen(frim_cmd, stdin=ffmpeg_proc.stdout,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            procs.append(frim_proc)
            ffmpeg_proc.stdout.close()

            feed_errors = []
            feeder = threading.Thread(target=_stream_interleaved, daemon=True,
                                      args=(edge264_proc.stdin, base_es_src, base_bounds, dep_es_src, dep_bounds,
                                            n, stop_event, feed_errors))
            tail = []

            def _drain_frim():
                for chunk in iter(lambda: frim_proc.stdout.read(256), b""):
                    tail.append(chunk)

            reader = threading.Thread(target=_drain_frim, daemon=True)
            feeder.start()
            reader.start()
            while frim_proc.poll() is None:
                if stop_event is not None and stop_event.is_set():
                    for p in (frim_proc, ffmpeg_proc, edge264_proc):
                        p.kill()
                    raise Cancelled()
                try:
                    frim_proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            feeder.join()
            reader.join()
            ffmpeg_proc.wait()
            edge264_proc.wait()

        frim_tail = b"".join(tail).decode(errors="replace")
        if frim_proc.returncode != 0:
            raise RuntimeError(f"the MVC encode failed (FRIMEncode exit code {frim_proc.returncode}):\n"
                              f"{frim_tail[-3000:] if frim_tail.strip() else '(FRIMEncode produced no output)'}")
        if edge264_proc.returncode not in (0, 1):
            with open(edge_log, "rb") as f:
                edge_msg = f.read()[-2000:].decode(errors="replace")
            raise RuntimeError(f"decode failed (edge264 exit {edge264_proc.returncode}, "
                              f"feeder errors {feed_errors}):\n{edge_msg}")
        if not (path.exists(frim_base_es) and path.exists(frim_dep_es)):
            raise RuntimeError("FRIMEncode exited cleanly but produced no output streams")

        if progress_cb:
            progress_cb("interleave", 0, 1)
        base_n, dep_n, combined_n = interleave_mvc(frim_base_es, frim_dep_es, combined_es)
        if base_n != dep_n:
            print(f"[mvc-extract] WARNING: FRIM base AUs={base_n} != dependent AUs={dep_n}; using the "
                 f"first {combined_n}", file=sys.stderr)
        if progress_cb:
            progress_cb("interleave", 1, 1)
        if stop_event is not None and stop_event.is_set():
            raise Cancelled()

        if progress_cb:
            progress_cb("mux", 0, 1)
        mux = subprocess.Popen([mkvmerge_bin, "-o", video_only, "--default-duration", f"0:{mkv_fps_str}",
                               combined_es], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(mux)
        mux_lines = []
        for line in mux.stdout:
            mux_lines.append(line)
            m = re.search(r"Progress:\s*(\d+)%", line)
            if m and progress_cb:
                progress_cb("mux", float(m.group(1)), 100)
            if stop_event is not None and stop_event.is_set():
                mux.kill()
                raise Cancelled()
        mux.wait()
        if mux.returncode != 0:
            raise RuntimeError(f"mkvmerge failed: {''.join(mux_lines)[-2000:]}")

        if include_av:
            _restore_disc_av(ssif_path, video_only, output_path, cut_start, cut_end, progress_cb)
        return combined_n
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        if not keep_temp:
            _remove_stale_temp(base_es_src, dep_es_src, edge_log, frim_base_es, frim_dep_es, combined_es,
                               meta_path, *((video_only,) if include_av else ()))


def _powershell(script):
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[-500:])
    return result.stdout.strip()


def mount_iso(iso_path):
    """Mounts an ISO with Windows' built-in support (7-Zip's UDF reader failed on the
    25GB+ file inside a real disc). Returns (drive_root, mounted_by_us) -- an ISO the
    user already had mounted is used as-is and never dismounted afterwards."""
    p = iso_path.replace("'", "''")
    out = _powershell(
        f"$p='{p}'; $was=(Get-DiskImage -ImagePath $p).Attached; "
        f"if(-not $was){{ Mount-DiskImage -ImagePath $p | Out-Null }}; "
        f"$v=Get-DiskImage -ImagePath $p | Get-Volume; Write-Output ([string]$was + '|' + $v.DriveLetter)")
    was, letter = out.splitlines()[-1].split("|")
    if not letter:
        raise RuntimeError("the ISO mounted but Windows gave it no drive letter")
    return letter + ":\\", was.strip().lower() != "true"


def dismount_iso(iso_path):
    p = iso_path.replace("'", "''")
    try:
        _powershell(f"Dismount-DiskImage -ImagePath '{p}' | Out-Null")
    except Exception as e:
        print(f"[mvc-extract] WARNING: could not dismount {iso_path}: {e}", file=sys.stderr)


def find_disc_ssif(root):
    """The main movie's 3D stream: the largest file in BDMV/STREAM/SSIF (extras and
    menus are tiny next to the feature). `root` may be the disc/drive root, its BDMV
    folder, a folder containing BDMV, or (ADR-260) a ripping tool's own output folder
    that wraps the real disc root one level deeper (e.g. <output>/<Disc Title>/BDMV)."""
    root = path.abspath(root)
    tried = []

    def _check(bdmv):
        # ADR-258: report exactly why a candidate didn't match, instead of one generic
        # refusal, so a folder-level mismatch is immediately visible.
        is_bdmv = path.basename(bdmv).upper() == "BDMV"
        if not is_bdmv:
            tried.append(f"{bdmv} (not named BDMV)")
            return None
        ssif_dir = path.join(bdmv, "STREAM", "SSIF")
        if not path.isdir(ssif_dir):
            tried.append(f"{bdmv} (named BDMV, but no STREAM/SSIF folder inside it)")
            return None
        files = [path.join(ssif_dir, f) for f in os.listdir(ssif_dir) if f.lower().endswith(".ssif")]
        if files:
            return max(files, key=path.getsize)
        raise RuntimeError("this disc has no 3D video (its BDMV/STREAM/SSIF folder is empty)")

    for bdmv in (path.join(root, "BDMV"), root, path.dirname(root)):
        result = _check(bdmv)
        if result:
            return result

    # ADR-260: real user report -- a real, existing BDMV/STREAM/SSIF folder (confirmed
    # by the user directly) still hit this refusal when pointed at a ripped folder
    # (Xreveal). Reproduced directly against a real disc's own real structure and volume
    # label: ripping tools commonly wrap the real disc root one level deeper than the
    # folder the user is asked to choose, i.e. <chosen output folder>/<Disc Title>/BDMV/...
    # -- pointing this tool at the OUTER folder (the one the ripping tool's own dialog
    # asked for) hits exactly this refusal, one level above where BDMV actually lives.
    # If none of the 3 direct candidates matched, scan one level of immediate
    # subfolders for a real BDMV/STREAM/SSIF inside any of them.
    try:
        subdirs = sorted(e for e in os.listdir(root) if path.isdir(path.join(root, e)))
    except OSError:
        subdirs = []
    for sub in subdirs:
        result = _check(path.join(root, sub, "BDMV"))
        if result:
            return result

    raise RuntimeError(
        "no Blu-ray 3D content found -- expected a BDMV/STREAM/SSIF folder (is this a 2D-only disc?)\n"
        "Checked these locations based on the path given:\n" + "\n".join(f"  - {t}" for t in tried) +
        "\nIf you know a real BDMV/STREAM/SSIF folder exists, point this tool directly at the folder "
        "that CONTAINS the BDMV folder (the disc's own root), at the BDMV folder itself, or at the "
        "folder a ripping tool wrote its output into (its real disc-root subfolder is searched too).")


def import_disc(source, output_path, work_dir=None, progress_cb=None, cut_end=None, **kwargs):
    """One call for the whole job: `source` is an .iso, a disc/BDMV folder, or a .ssif.
    Mounts the ISO if needed, picks the main movie, finds the AVC/MVC tracks itself,
    checks there is enough free disk space for the temporary streams, runs
    extract_and_decode(), and dismounts what it mounted. Returns the frame count."""
    import shutil
    source = path.abspath(source)
    mounted_here = False
    work_created = False
    stem = path.splitext(path.basename(output_path))[0]
    work_dir = work_dir or path.join(path.dirname(path.abspath(output_path)), f"_mvc_work_{stem}")
    try:
        if source.lower().endswith(".iso"):
            if progress_cb:
                progress_cb("mount", 0, 1)
            root, mounted_here = mount_iso(source)
            ssif = find_disc_ssif(root)
        elif source.lower().endswith(".ssif"):
            ssif = source
        else:
            ssif = find_disc_ssif(source)
        print(f"[mvc-extract] main movie stream: {ssif}", file=sys.stderr)

        tsmuxer_bin = _find_tsmuxer()
        if tsmuxer_bin is None:
            raise RuntimeError("tsMuxeR not found -- run `python -m iw3.install_mvc_tools`")
        avc_track, mvc_track = _find_video_track(ssif, tsmuxer_bin)
        if mvc_track is None or avc_track is None:
            raise RuntimeError("no 3D (MVC) video track found in this disc's main movie")
        print(f"[mvc-extract] tracks: AVC={avc_track} MVC={mvc_track}", file=sys.stderr)

        os.makedirs(path.dirname(path.abspath(output_path)), exist_ok=True)
        if kwargs.get("layout") == ISO_LAYOUT:
            # lossless copy: no temp streams, the .iso itself is the only big file
            free = shutil.disk_usage(path.dirname(path.abspath(output_path))).free
            need = path.getsize(ssif)
            if free < need * 1.05:
                raise RuntimeError(f"not enough free disk space for the ISO: need about "
                                   f"{need / 1e9:.0f} GB, only {free / 1e9:.0f} GB free")
            return mux_bd3d_iso(ssif, output_path, include_av=kwargs.get("restore_av", True),
                                stop_event=kwargs.get("stop_event"), progress_cb=progress_cb,
                                cut_end=cut_end)
        if kwargs.get("layout") == MVC_MKV_LAYOUT:
            # lossless copy: the two demuxed streams (~0.75x the .ssif) plus the
            # combined interleaved stream built from them (the same data again,
            # ~0.75x) both exist on disk at once before the video-only .mkv is
            # written -- unlike the flat-SBS path below, nothing here is streamed
            # straight into a decoder, so this needs real room for both.
            work_created = not path.isdir(work_dir)
            os.makedirs(work_dir, exist_ok=True)
            need = int(path.getsize(ssif) * 1.5)
            free = shutil.disk_usage(work_dir).free
            if free < need * 1.05:
                raise RuntimeError(f"not enough free disk space in {work_dir}: need about "
                                   f"{need / 1e9:.0f} GB for temporary files, only {free / 1e9:.0f} GB free")
            return mux_lossless_mvc_mkv(ssif, avc_track, mvc_track, output_path,
                                        include_av=kwargs.get("restore_av", True), cut_end=cut_end,
                                        work_dir=work_dir, keep_temp=kwargs.get("keep_temp", False),
                                        stop_event=kwargs.get("stop_event"), progress_cb=progress_cb)
        if kwargs.get("layout") == MVC_MKV_CROPPED_LAYOUT:
            # ADR-259: real re-encode, so the two demuxed source streams (~0.75x the
            # .ssif) plus FRIMEncode's own fresh base+dependent output (at the chosen
            # bitrate -- roughly comparable to or smaller than the source at a typical
            # 20-40Mbps combined target) all exist on disk at some point; same
            # conservative multiplier mux_lossless_mvc_mkv() already uses above.
            work_created = not path.isdir(work_dir)
            os.makedirs(work_dir, exist_ok=True)
            need = int(path.getsize(ssif) * 1.5)
            free = shutil.disk_usage(work_dir).free
            if free < need * 1.05:
                raise RuntimeError(f"not enough free disk space in {work_dir}: need about "
                                   f"{need / 1e9:.0f} GB for temporary files, only {free / 1e9:.0f} GB free")
            return extract_and_reencode_mvc(
                ssif, avc_track, mvc_track, "0s", cut_end, work_dir, output_path,
                bitrate_mbps=kwargs.get("bitrate_mbps", 20.0), autocrop=kwargs.get("autocrop"),
                swap_eyes=kwargs.get("swap_eyes", False), include_av=kwargs.get("restore_av", True),
                keep_temp=kwargs.get("keep_temp", False), stop_event=kwargs.get("stop_event"),
                progress_cb=progress_cb)
        work_created = not path.isdir(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        # the two demuxed streams together are roughly 3/4 of the .ssif
        need = int(path.getsize(ssif) * 0.75)
        free = shutil.disk_usage(work_dir).free
        if free < need * 1.05:
            raise RuntimeError(f"not enough free disk space in {work_dir}: need about "
                               f"{need / 1e9:.0f} GB for temporary files, only {free / 1e9:.0f} GB free")
        return extract_and_decode(ssif, avc_track, mvc_track, "0s", cut_end, work_dir, output_path,
                                  progress_cb=progress_cb, **kwargs)
    finally:
        if mounted_here:
            dismount_iso(source)
        if work_created:
            try:
                os.rmdir(work_dir)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--disc", type=str, default=None,
                         help="a 3D Blu-ray .iso, a disc/BDMV folder, or a .ssif file -- the main movie, its tracks "
                              "and temp folder are all found automatically")
    parser.add_argument("--ssif", "-i", type=str, default=None,
                         help="(manual mode) path to the .ssif file (BDMV/STREAM/SSIF/*.ssif)")
    parser.add_argument("--mpls", type=str, default=None,
                         help="(manual mode) a playlist or the .ssif itself, used to auto-detect the AVC/MVC track "
                              "IDs. If omitted, --avc-track/--mvc-track must be given directly.")
    parser.add_argument("--avc-track", type=int, default=None)
    parser.add_argument("--mvc-track", type=int, default=None)
    parser.add_argument("--cut-start", type=str, default="0s")
    parser.add_argument("--cut-end", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None,
                         help="folder for the temporary demuxed streams (~75%% of the movie's size); "
                              "default: a _mvc_work_* folder next to --output")
    parser.add_argument("--output", "-o", type=str, required=True,
                         help="final encoded output path (.mkv) -- a real compressed video, "
                              "never a raw intermediate")
    parser.add_argument("--video-codec", type=str, default="hevc_nvenc", choices=CODECS)
    parser.add_argument("--quality", "--crf", type=int, default=18, dest="quality",
                         help="CRF (x264/x265) or constant-quality level (NVENC); lower = better/larger")
    parser.add_argument("--layout", type=str, default="full_sbs",
                         choices=LAYOUTS + (ISO_LAYOUT, MVC_MKV_LAYOUT, MVC_MKV_CROPPED_LAYOUT),
                         help="bd3d_iso = lossless copy into a 3D Blu-ray .iso (no re-encode; --output must end in "
                              ".iso; codec/quality ignored) | "
                              "mvc_mkv = lossless copy into a plain .mkv holding the real combined MVC video "
                              "stream, no disc structure, no re-encode (--output must end in .mkv; codec/quality "
                              "ignored) -- for a library that plays real MVC files directly (e.g. via MakeMKV/"
                              "CloneBD-style rips), instead of this tool's own SBS/TB layouts | "
                              "mvc_mkv_cropped = ADR-259: same real MVC .mkv output as mvc_mkv, but a genuine "
                              "re-encode (via FRIMEncode, see --bitrate/--swap-eyes) with --autocrop applied -- "
                              "for removing black bars in ONE pass instead of decoding to a flat video and "
                              "re-encoding to MVC again as two separate steps | "
                              "full_sbs 3840x1080 | half_sbs 1920x1080 | full_tb 1920x2160 | half_tb 1920x1080 | "
                              "*_4k = the same four with each eye enlarged to 4K (full_sbs_4k 7680x2160, "
                              "half_sbs_4k 3840x2160, full_tb_4k 3840x4320, half_tb_4k 3840x2160) | "
                              "frame_packed = full_tb plus the Frame Packing SEI flag (libx264 only)")
    parser.add_argument("--autocrop", type=str.upper, default=None, choices=AUTOCROP_MODES,
                         help="remove black bars: BLACK = all sides, BLACK_TB = top/bottom only (FLAT / FLAT_TB do "
                              "the same for flat-colour borders). Both eyes get the same crop. Not used by "
                              "--layout bd3d_iso or --layout mvc_mkv (both copy the disc's video untouched).")
    parser.add_argument("--bitrate", type=float, default=20.0,
                         help="only with --layout mvc_mkv_cropped: target Mbps for the fresh FRIMEncode MVC "
                              "re-encode (2-40; 3D Blu-ray allows about 40 combined) -- same meaning as "
                              "sbs_to_mvc_cli's own --bitrate.")
    parser.add_argument("--swap-eyes", action="store_true",
                         help="only with --layout mvc_mkv_cropped: swap left/right in the fresh MVC re-encode.")
    parser.add_argument("--no-audio-subs", action="store_true",
                         help="skip restoring the disc's audio and subtitle tracks (video only)")
    parser.add_argument("--keep-temp", action="store_true", help="keep the demuxed elementary streams")
    parser.add_argument("--gui-progress", action="store_true",
                         help="print machine-readable 'IW3_MVC_PROGRESS <stage> <done> <total>' lines to stdout")
    args = parser.parse_args()
    if (args.disc is None) == (args.ssif is None):
        parser.error("give exactly one of --disc (automatic) or --ssif (manual)")

    if args.gui_progress:
        def show(stage, done, total):
            print(f"IW3_MVC_PROGRESS {stage} {done} {total}", flush=True)
    else:
        def show(stage, done, total):
            print(f"\r[mvc-extract] {stage}: {done}/{total}      ", end="", file=sys.stderr, flush=True)

    common = dict(video_codec=args.video_codec, quality=args.quality, layout=args.layout,
                  restore_av=not args.no_audio_subs, keep_temp=args.keep_temp, progress_cb=show)
    if args.layout == MVC_MKV_CROPPED_LAYOUT:
        common["bitrate_mbps"] = args.bitrate
        common["swap_eyes"] = args.swap_eyes
    if args.autocrop and args.layout in (ISO_LAYOUT, MVC_MKV_LAYOUT):
        print("[mvc-extract] note: --autocrop is ignored for this lossless layout (nothing is re-encoded)",
              file=sys.stderr)
    elif args.autocrop:
        common["autocrop"] = args.autocrop
    try:
        if args.disc is not None:
            frames = import_disc(args.disc, args.output, work_dir=args.work_dir, cut_end=args.cut_end, **common)
        else:
            if args.work_dir is None:
                parser.error("--work-dir is required with --ssif")
            avc_track, mvc_track = args.avc_track, args.mvc_track
            if avc_track is None or mvc_track is None:
                detect_from = args.mpls or args.ssif
                avc_track, mvc_track = _find_video_track(detect_from, _find_tsmuxer())
                if mvc_track is None:
                    parser.error(f"no MVC track found via {detect_from} -- not real 3D content, "
                                 f"or the wrong file")
                print(f"[mvc-extract] auto-detected tracks: AVC={avc_track} MVC={mvc_track}", file=sys.stderr)
            if args.layout in (ISO_LAYOUT, MVC_MKV_LAYOUT, MVC_MKV_CROPPED_LAYOUT):
                parser.error(f"--layout {args.layout} works with --disc, not manual --ssif mode")
            frames = extract_and_decode(args.ssif, avc_track, mvc_track, args.cut_start, args.cut_end,
                                        args.work_dir, args.output, **common)
    except Cancelled:
        print("\n[mvc-extract] cancelled", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    unit = "tracks" if args.layout == ISO_LAYOUT else "frames"
    print(f"\n[mvc-extract] done: {args.output} ({frames} {unit})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
