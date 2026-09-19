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

from .utils import _find_tsmuxer, _find_edge264_mvc, _get_ffmpeg_bin


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


def layout_filter(layout):
    """ffmpeg -vf chain turning edge264's full-width side-by-side frame (left eye |
    right eye, each at full 1920x1080) into the requested layout. None = no change."""
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
        args = ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
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
                        restore_av=True, keep_temp=False, stop_event=None, progress_cb=None):
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
    os.makedirs(work_dir, exist_ok=True)
    meta_path = path.join(work_dir, "_mvc_extract.meta")
    base_es = path.join(work_dir, f"{path.splitext(path.basename(ssif_path))[0]}.track_{avc_track}.264")
    dep_es = path.join(work_dir, f"{path.splitext(path.basename(ssif_path))[0]}.track_{mvc_track}.mvc")
    edge_log = path.join(work_dir, "_edge264_stderr.log")
    procs = []
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

        vf = layout_filter(layout)
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
            for tmp in (base_es, dep_es, meta_path, edge_log) + ((video_only,) if restore_av else ()):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


ISO_LAYOUT = "bd3d_iso"


def list_tracks(ssif_path, tsmuxer_bin):
    """Every track tsMuxeR sees in the .ssif, as dicts: id, codec (e.g. V_MPEG4/ISO/MVC,
    A_AC3, S_HDMV/PGS) and lang (3-letter code or "")."""
    out = subprocess.run([tsmuxer_bin, str(ssif_path)], capture_output=True, text=True).stdout
    tracks, cur = [], None
    for line in out.splitlines():
        if line.startswith("Track ID:"):
            cur = {"id": int(line.split(":", 1)[1].strip()), "codec": "", "lang": ""}
            tracks.append(cur)
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
    ssif_meta = path.abspath(ssif_path).replace(chr(92), "/")
    cut = f" --cut-start=0s --cut-end={cut_end}" if cut_end else ""
    lines = [f"MUXOPT --blu-ray --auto-chapters=10{cut}"]
    for t in wanted:
        extra = f", lang={t['lang']}" if t["lang"] else ""
        lines.append(f"{t['codec']}, {ssif_meta}, track={t['id']}{extra}")
    os.makedirs(path.dirname(path.abspath(out_iso)), exist_ok=True)
    meta_path = path.splitext(path.abspath(out_iso))[0] + ".mux.meta"
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
        try:
            os.remove(meta_path)
        except OSError:
            pass
        if not ok:
            try:
                os.remove(out_iso)
            except OSError:
                pass


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
    folder, or a folder containing BDMV."""
    root = path.abspath(root)
    candidates = [path.join(root, "BDMV"), root, path.dirname(root)]
    for bdmv in candidates:
        ssif_dir = path.join(bdmv, "STREAM", "SSIF")
        if path.basename(bdmv).upper() == "BDMV" and path.isdir(ssif_dir):
            files = [path.join(ssif_dir, f) for f in os.listdir(ssif_dir) if f.lower().endswith(".ssif")]
            if files:
                return max(files, key=path.getsize)
            raise RuntimeError("this disc has no 3D video (its BDMV/STREAM/SSIF folder is empty)")
    raise RuntimeError("no Blu-ray 3D content found -- expected a BDMV/STREAM/SSIF folder "
                       "(is this a 2D-only disc?)")


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
    parser.add_argument("--layout", type=str, default="full_sbs", choices=LAYOUTS + (ISO_LAYOUT,),
                         help="bd3d_iso = lossless copy into a 3D Blu-ray .iso (no re-encode; --output must end in "
                              ".iso; codec/quality ignored) | "
                              "full_sbs 3840x1080 | half_sbs 1920x1080 | full_tb 1920x2160 | half_tb 1920x1080 | "
                              "*_4k = the same four with each eye enlarged to 4K (full_sbs_4k 7680x2160, "
                              "half_sbs_4k 3840x2160, full_tb_4k 3840x4320, half_tb_4k 3840x2160) | "
                              "frame_packed = full_tb plus the Frame Packing SEI flag (libx264 only)")
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
            if args.layout == ISO_LAYOUT:
                parser.error("--layout bd3d_iso works with --disc, not manual --ssif mode")
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
