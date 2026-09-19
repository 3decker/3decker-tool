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
import subprocess
import sys
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


def _split_access_units(data, delimiter_types):
    """Splits a raw NAL-unit elementary stream into access units, each
    starting at one of the given delimiter NAL types. Confirmed directly
    against real extracted streams (not assumed from spec alone): base-view
    access units start with a type-9 (AUD) NAL; dependent-view access units
    start with a type-24 NAL."""
    starts = []
    idx = 0
    while True:
        idx = data.find(b"\x00\x00\x01", idx)
        if idx == -1:
            break
        starts.append(idx)
        idx += 3

    aus = []
    current_au_start = None
    for pos in starts:
        nal_type = data[pos + 3] & 0x1F
        if nal_type in delimiter_types:
            if current_au_start is not None:
                aus.append(data[current_au_start:pos])
            current_au_start = pos
    if current_au_start is not None:
        aus.append(data[current_au_start:len(data)])
    return aus


def interleave_mvc(base_path, dependent_path, out_path):
    """Recombines a separately-demuxed base (AVC) and dependent (MVC) view
    elementary stream into one combined bitstream a real MVC decoder can
    read. Returns (base_au_count, dependent_au_count, written_au_count)."""
    with open(base_path, "rb") as f:
        base_data = f.read()
    with open(dependent_path, "rb") as f:
        dep_data = f.read()

    base_aus = _split_access_units(base_data, delimiter_types={9})
    dep_aus = _split_access_units(dep_data, delimiter_types={24, 9})

    n = min(len(base_aus), len(dep_aus))
    with open(out_path, "wb") as out:
        for i in range(n):
            out.write(base_aus[i])
            out.write(dep_aus[i])

    return len(base_aus), len(dep_aus), n


def extract_and_decode(ssif_path, avc_track, mvc_track, cut_start, cut_end, work_dir, output_path,
                        video_codec="libx264", crf=16):
    """Full pipeline: tsMuxeR demux -> interleave -> edge264-mvc decode -> ffmpeg encode.

    The decoder's raw Y4M output is piped DIRECTLY into ffmpeg's stdin -- never
    written to disk. A real 15-second test produced a 2.2GB raw Y4M file; a full
    ~90-minute movie at that rate would be roughly 800GB, not practical. Piping
    keeps disk usage down to just the (much smaller) demuxed elementary streams
    and the final compressed output."""
    tsmuxer_bin = _find_tsmuxer()
    edge264_bin = _find_edge264_mvc()
    ffmpeg_bin = _get_ffmpeg_bin()
    if tsmuxer_bin is None:
        raise RuntimeError("tsMuxeR not found -- see docs/ai/AI_DECISIONS.md ADR-182")
    if edge264_bin is None:
        raise RuntimeError("edge264-mvc not found -- see docs/ai/AI_DECISIONS.md ADR-182")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found")

    meta_path = path.join(work_dir, "_mvc_extract.meta")
    cut_opts = f" --cut-start={cut_start} --cut-end={cut_end}" if cut_end else f" --cut-start={cut_start}"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"MUXOPT --demux{cut_opts}\n")
        f.write(f"V_MPEG4/ISO/AVC, {path.abspath(ssif_path).replace(chr(92), '/')}, track={avc_track}\n")
        f.write(f"V_MPEG4/ISO/MVC, {path.abspath(ssif_path).replace(chr(92), '/')}, track={mvc_track}\n")

    result = subprocess.run([tsmuxer_bin, meta_path, work_dir], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"tsMuxeR demux failed: {result.stdout}\n{result.stderr}")

    base_es = path.join(work_dir, f"00000.track_{avc_track}.264")
    dep_es = path.join(work_dir, f"00000.track_{mvc_track}.mvc")
    combined = path.join(work_dir, "_combined_mvc.264")
    base_n, dep_n, written_n = interleave_mvc(base_es, dep_es, combined)
    print(f"[mvc-extract] base AUs={base_n} dependent AUs={dep_n} written={written_n}", file=sys.stderr)

    # Pipe edge264-mvc's stdout directly into ffmpeg's stdin -- no raw Y4M ever
    # touches disk. Standard Python subprocess pipe-chaining: pass the first
    # process's stdout pipe object directly as the second process's stdin, then
    # close the parent's duplicate handle so the first process receives a
    # real SIGPIPE/broken-pipe signal if ffmpeg exits early instead of hanging.
    edge264_proc = subprocess.Popen(
        [edge264_bin, combined, "-O", "-k", "-y"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ffmpeg_args = [ffmpeg_bin, "-y", "-i", "-",
                   "-c:v", video_codec, "-crf", str(crf), "-pix_fmt", "yuv420p", output_path]
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_args, stdin=edge264_proc.stdout,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    edge264_proc.stdout.close()

    ffmpeg_out, ffmpeg_err = ffmpeg_proc.communicate()
    edge264_proc.wait()
    edge264_err = edge264_proc.stderr.read()

    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {ffmpeg_err.decode(errors='replace')}")
    if edge264_proc.returncode != 0:
        raise RuntimeError(f"edge264-mvc decode failed: {edge264_err.decode(errors='replace')}")


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
    parser.add_argument("--video-codec", type=str, default="libx264")
    parser.add_argument("--crf", type=int, default=16)
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

    extract_and_decode(args.ssif, avc_track, mvc_track, args.cut_start, args.cut_end, args.work_dir, args.output,
                        video_codec=args.video_codec, crf=args.crf)
    print(f"[mvc-extract] done: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
