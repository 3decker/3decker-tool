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


LAYOUTS = ("full_sbs", "half_sbs", "full_tb", "half_tb", "frame_packed")
CODECS = ("hevc_nvenc", "libx265", "libx264")

# Only libx264 can write the H.264 Frame Packing SEI that lets a 3D TV/player
# auto-detect the layout (ADR-181). x265 defines the SEI type but exposes no way
# to set it; NVENC has no equivalent at all.
_SEI_TYPE = {"half_sbs": 3, "half_tb": 4, "frame_packed": 4}

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
    # A 3D Blu-ray is Rec.709 limited range; Y4M carries no colour tags, so without
    # these a player has to guess.
    args += ["-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
             "-color_trc", "bt709", "-color_range", "tv"]
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
        ffmpeg_args = [ffmpeg_bin, "-y", "-hide_banner", "-i", "-"]
        if vf:
            ffmpeg_args += ["-vf", vf]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssif", "-i", type=str, required=True, help="path to the .ssif file (BDMV/STREAM/SSIF/*.ssif)")
    parser.add_argument("--mpls", type=str, default=None,
                         help="the disc's playlist (BDMV/PLAYLIST/*.mpls) -- used to auto-detect the AVC/MVC track "
                              "IDs. If omitted, --avc-track/--mvc-track must be given directly.")
    parser.add_argument("--avc-track", type=int, default=None)
    parser.add_argument("--mvc-track", type=int, default=None)
    parser.add_argument("--cut-start", type=str, default="0s")
    parser.add_argument("--cut-end", type=str, default=None)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True,
                         help="final encoded output path (e.g. .mkv) -- a real compressed video, "
                              "never a raw intermediate")
    parser.add_argument("--video-codec", type=str, default="hevc_nvenc", choices=CODECS)
    parser.add_argument("--quality", "--crf", type=int, default=18, dest="quality",
                         help="CRF (x264/x265) or constant-quality level (NVENC); lower = better/larger")
    parser.add_argument("--layout", type=str, default="full_sbs", choices=LAYOUTS,
                         help="full_sbs 3840x1080 | half_sbs 1920x1080 | full_tb 1920x2160 | half_tb 1920x1080 | "
                              "frame_packed = full_tb plus the Frame Packing SEI flag (libx264 only)")
    parser.add_argument("--no-audio-subs", action="store_true",
                         help="skip restoring the disc's audio and subtitle tracks (video only)")
    parser.add_argument("--keep-temp", action="store_true", help="keep the demuxed elementary streams")
    args = parser.parse_args()

    avc_track, mvc_track = args.avc_track, args.mvc_track
    if avc_track is None or mvc_track is None:
        if args.mpls is None:
            parser.error("either --mpls (to auto-detect tracks) or both --avc-track and --mvc-track are required")
        tsmuxer_bin = _find_tsmuxer()
        avc_track, mvc_track = _find_video_track(args.mpls, tsmuxer_bin)
        if mvc_track is None:
            parser.error(f"no MVC track found via {args.mpls} -- this title may not be real 3D content, "
                          f"or --mpls points at the wrong playlist")
        print(f"[mvc-extract] auto-detected tracks: AVC={avc_track} MVC={mvc_track}", file=sys.stderr)

    def show(stage, done, total):
        print(f"\r[mvc-extract] {stage}: {done}/{total}      ", end="", file=sys.stderr, flush=True)

    frames = extract_and_decode(args.ssif, avc_track, mvc_track, args.cut_start, args.cut_end, args.work_dir,
                                 args.output, video_codec=args.video_codec, quality=args.quality,
                                 layout=args.layout, restore_av=not args.no_audio_subs,
                                 keep_temp=args.keep_temp, progress_cb=show)
    print(f"\n[mvc-extract] done: {args.output} ({frames} frames)", file=sys.stderr)


if __name__ == "__main__":
    main()
