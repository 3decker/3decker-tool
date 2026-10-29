"""python -m iw3.av_restore_cli -- standalone tool to restore EVERY audio AND
subtitle track from an original source movie onto an already-converted 3D video,
in one combined pass.

This is the engine behind the main conversion pipeline's automatic "Restore
Audio & Subtitles from Source" post-processing checkbox (see ADR-172) -- it is
deliberately a SEPARATE module from iw3.audio_restore_cli (ADR-169), not a
replacement or extension of it. audio_restore_cli.py stays exactly as it is
(audio-only, its own standalone GUI tool, its own tests) for anyone who
specifically wants audio without subtitles; this module additionally restores
subtitle tracks, which iw3's main conversion never carries over AT ALL (unlike
audio, which at least keeps the source's first track -- see
nunif/utils/video/processor.py's export_audio() and iw3/audio_restore_cli.py's
own docstring for that half of the story). There is no in-pipeline subtitle
handling of any kind to work around here -- confirmed by reading
nunif/utils/video/processor.py directly, no subtitle-related code exists there.

Follows the same create_parser()/run()/main() shape, and the same MKV-only/
--output-is-never---input/temp-then-os.replace() safety conventions,
audio_restore_cli.py and subtitle_mux_cli.py (ADR-032) already established.

Graceful degradation (deliberately NOT a hard requirement that --source have
both track types): a source with audio but no subtitles (the common case for
many discs) still gets its audio restored -- refusing outright just because
subtitles are absent would defeat the whole point of an automatic
post-processing step nobody has to think about. Only refuses outright when
--source has NEITHER audio nor subtitle tracks to restore -- nothing useful this
tool could do in that case.

Trimming (see _trim_all_av): reuses audio_restore_cli.py's exact
-ss/-to-as-absolute-input-position + -avoid_negative_ts make_zero technique,
extended to pull both audio (`-map 0:a?`) and subtitle (`-map 0:s?`) tracks in
ONE ffmpeg pass -- the trailing `?` makes each map optional (ffmpeg skips it
without erroring if that source genuinely has zero streams of that type,
confirmed against real ffmpeg documentation for the optional-stream-specifier
syntax), rather than requiring two separate trim passes or hand-detecting which
map to include. -c copy is tried first; one re-encode fallback (ffmpeg's own
default encoders) if that fails outright, same single-fallback shape
audio_restore_cli.py already uses.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from os import path

# ADR-253: this module runs as its own separate subprocess (both as the main
# pipeline's "Restore Audio & Subtitles" post-step and, potentially, standalone)
# and calls ffmpeg/mkvmerge as its own nested subprocesses -- each one would
# flash its own console window without this, which only the GUI's own process
# (iw3/gui.py) imported before now.
import nunif.gui.subprocess_patch  # noqa

import av

from nunif.utils.video.metadata import parse_time
from .utils import _find_mkvmerge, _get_ffmpeg_bin, _find_ffprobe
# ADR-254: real user request -- restored subtitles should be able to come back
# already positioned for real 3D (ADR-053's "dual-eye" stereo positioning), not
# just flat-copied from source the way they always have been. That positioning
# math (_build_stereo_positioned_subs and its supporting helpers) lives in
# subtitle_mux_cli.py, the one place it's implemented and tested -- imported here
# rather than duplicated, a deliberate, narrow exception to this project's usual
# "each standalone CLI tool imports only from iw3.utils, never from each other"
# convention (see subtitle_mux_cli.py's own docstring), made specifically to
# avoid two copies of nontrivial positioning math drifting out of sync.
from .subtitle_mux_cli import (
    _resolve_format, _probe_video_dimensions, _build_stereo_positioned_subs,
    _default_font_size, _iso639_1_to_2, _HORIZONTAL_SPLIT_FORMATS, _VERTICAL_SPLIT_FORMATS,
    _VALID_FORMATS,
)


def _format_cmd(cmd):
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


# ADR-403 Amendment: real follow-up question -- decker directly asked whether an
# ETA is possible for the pre-mux window (mkvmerge's own brief file analysis,
# before it ever reports a real "Progress: NN%"). It genuinely isn't -- there is
# no real progress SIGNAL during that window (nothing done/total to measure a
# rate from), so any number shown there would be a GUESS wearing a measurement's
# clothing, not an estimate. Rather than show nothing at all, this gives a wide,
# explicitly-labeled guess from file size alone, bounded by two generic
# throughput assumptions -- wide on purpose, since real disk speed varies
# enormously (HDD vs. SSD, antivirus scanning, a network share), and a
# narrow-looking number that turns out very wrong would mislead more than a
# clearly-labeled range helps.
_ROUGH_MUX_FAST_MBPS = 400   # a typical SATA SSD doing sequential read+write
_ROUGH_MUX_SLOW_MBPS = 50    # a busy/slow disk -- an external HDD, a network share, AV scanning


def _format_rough_minutes(seconds):
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{hours}h {rem_minutes}min" if rem_minutes else f"{hours}h"


def _rough_mux_eta_range(num_bytes):
    """A wide, explicitly-labeled GUESS at how long muxing num_bytes will take --
    see the constants/comment above for why this can only ever be a guess, never a
    real measurement, for the pre-percent window before mkvmerge reports real
    progress. Returns a "LOW-HIGH" string, e.g. "1 min-8 min"."""
    fast_sec = num_bytes / (_ROUGH_MUX_FAST_MBPS * 1024 ** 2)
    slow_sec = num_bytes / (_ROUGH_MUX_SLOW_MBPS * 1024 ** 2)
    return f"{_format_rough_minutes(fast_sec)}-{_format_rough_minutes(slow_sec)}"


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.av_restore_cli",
        description=(
            "Restore every audio AND subtitle track from an original source movie onto an "
            "already-converted 3D (SBS/TB) MKV video, replacing whichever single audio track the "
            "conversion kept and adding subtitles the conversion never carried over at all. Video "
            "track in --input is preserved untouched. Never modifies --input or --source -- always "
            "writes a new file at --output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                         help="Path to the already-converted 3D video. Must be a .mkv file -- "
                              "this tool does not convert containers. Read-only, never modified.")
    parser.add_argument("--source", "-s", type=str, required=True,
                         help="Path to the original source movie (any container ffmpeg/mkvmerge "
                              "already read directly). Every audio and subtitle track found in it "
                              "is carried into --output, each with its own language/name metadata "
                              "preserved as-is. Read-only, never modified.")
    parser.add_argument("--output", "-o", type=str, required=True,
                         help="Path to write the new file to. Must be a different path from "
                              "--input -- this tool never overwrites the input.")
    parser.add_argument("--source-start-time", type=str, default=None,
                         help="Start time WITHIN --source to trim its audio/subtitle tracks to "
                              "(HH:MM:SS, MM:SS, or seconds), for when --input is only a short clip "
                              "of the full --source movie. The trimmed tracks are also "
                              "automatically shifted to start at t=0 so they line up with --input's "
                              "first frame. Default: start of --source.")
    parser.add_argument("--source-end-time", type=str, default=None,
                         help="End time within --source to trim its audio/subtitle tracks to. "
                              "Default: end of --source.")
    parser.add_argument("--dual-eye-subtitles", action="store_true",
                         help=("ADR-254: position restored TEXT subtitle tracks (SRT/ASS/SSA/...) for "
                               "real 3D instead of a flat copy -- same idea and same math as "
                               "iw3.subtitle_mux_cli's own --dual-eye-subtitles (ADR-053): each cue is "
                               "duplicated into two positioned copies, one centered in each eye-half, "
                               "so it displays with real depth on a split-eye layout (Half/Full SBS, "
                               "Half/Full TB, Cross-Eyed) instead of a plain centered subtitle landing "
                               "on the seam between the two eyes. No effect on a non-split layout "
                               "(RGB-D, Anaglyph) or on a picture-based (PGS/VobSub) subtitle track --"
                               " both are restored unchanged either way. Off by default."))
    parser.add_argument("--format", type=str, default="auto", choices=_VALID_FORMATS,
                         help="Only with --dual-eye-subtitles: --input's stereo layout. 'auto' "
                              "(default) detects it from --input's own filename (the same tag iw3 "
                              "itself writes); set explicitly if that detection fails.")
    parser.add_argument("--font-size", type=float, default=None,
                         help="Only with --dual-eye-subtitles: exact ASS Fontsize (pixels) for the "
                              "positioned track. Default: auto-scaled to the per-eye video height "
                              "(same default iw3.subtitle_mux_cli uses).")
    return parser


def _count_av_tracks(source_path):
    """Existence + audio/subtitle-track-count check for --source. Returns
    (audio_count, subtitle_count, error_message_or_None) -- error is only set for
    a missing file, a file av can't open, or a source with NEITHER track type
    (nothing this tool could restore either way). A source with one type but not
    the other is NOT an error -- see module docstring's "Graceful degradation"."""
    if not path.exists(source_path):
        return None, None, f"ERROR: --source file does not exist: {source_path}"
    try:
        container = av.open(source_path, mode="r", metadata_errors="ignore")
        try:
            audio_count = len(container.streams.audio)
            subtitle_count = len(container.streams.subtitles)
        finally:
            container.close()
    except Exception as e:
        return None, None, f"ERROR: failed to open --source to inspect its tracks: {e}"
    if audio_count == 0 and subtitle_count == 0:
        return 0, 0, f"ERROR: --source has no audio or subtitle tracks to restore: {source_path}"
    return audio_count, subtitle_count, None


_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_FFMPEG_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_MKV_PROGRESS_RE = re.compile(r"Progress:\s*(\d+)%")


def _emit_progress(stage, current, total):
    """Progress channel on STDOUT ("IW3_AVRESTORE_PROGRESS <stage> <current> <total>"), read by iw3.utils and
    iw3.gui; every human-readable message stays on stderr. Stages: "trim" (seconds) and "mux" (percent)."""
    print(f"IW3_AVRESTORE_PROGRESS {stage} {current:.2f} {total:.2f}", flush=True)


class _Result:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _run_trim_with_progress(cmd, start_sec, end_sec):
    """The trim ffmpeg, read live so its "time=" lines become progress."""
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
    chunks, total, last = [], None, 0.0
    for line in proc.stderr:
        chunks.append(line)
        if total is None:
            m = _FFMPEG_DURATION_RE.search(line)
            if m:
                duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                total = max(0.1, (end_sec if end_sec is not None else duration) - start_sec)
        m = _FFMPEG_TIME_RE.search(line)
        if m and total:
            now = time.monotonic()
            if now - last >= 0.2:
                last = now
                current = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                _emit_progress("trim", min(current, total), total)
    proc.wait()
    if total and proc.returncode == 0:
        _emit_progress("trim", total, total)
    return _Result(proc.returncode, "", "".join(chunks))


def _run_mkvmerge_with_progress(cmd):
    """mkvmerge prints "Progress: NN%" (carriage-return separated) on stdout; turn it into progress lines."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    err_chunks = []
    drain = __import__("threading").Thread(target=lambda: err_chunks.extend(proc.stderr), daemon=True)
    drain.start()
    out_chunks, last = [], -1
    for line in proc.stdout:
        m = _MKV_PROGRESS_RE.search(line)
        if m:
            pct = int(m.group(1))
            if pct != last:
                last = pct
                _emit_progress("mux", pct, 100)
        else:
            out_chunks.append(line)
    proc.wait()
    drain.join(timeout=5)
    if proc.returncode < 2:
        _emit_progress("mux", 100, 100)
    return _Result(proc.returncode, "".join(out_chunks), "".join(err_chunks))


def _trim_all_av(ffmpeg_bin, source_path, start_time, end_time, work_dir, progress=False):
    """Extracts every audio AND subtitle track from source_path over [start_time,
    end_time) into a new temp .mkv in work_dir, in ONE ffmpeg pass, ALSO shifting
    the result to start at t=0 -- see module docstring's "Trimming" section for
    the optional-map (`0:a?`/`0:s?`) reasoning. Tries a fast lossless -c copy
    first, falling back to one re-encode (ffmpeg's own default encoders) only if
    stream copy fails outright.

    Returns (trimmed_path_or_None, error_message_or_None, cmds_tried (list of str)).
    """
    trimmed_path = path.join(work_dir, "trimmed_av.mkv")

    start_sec = parse_time(start_time) if start_time else 0.0
    end_sec = parse_time(end_time) if end_time is not None else None
    if end_sec is not None and end_sec <= start_sec:
        return None, (
            f"ERROR: --source-end-time ({end_time}) must be after --source-start-time "
            f"({start_time})."), []

    def _build_cmd(use_copy):
        cmd = [ffmpeg_bin, "-y", "-ss", str(start_sec)]
        if end_sec is not None:
            cmd += ["-to", str(end_sec)]
        cmd += ["-i", str(source_path), "-vn", "-map", "0:a?", "-map", "0:s?"]
        if use_copy:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        cmd += [trimmed_path]
        return cmd

    cmds_tried = []

    copy_cmd = _build_cmd(use_copy=True)
    cmds_tried.append(_format_cmd(copy_cmd))
    try:
        proc = (_run_trim_with_progress(copy_cmd, start_sec, end_sec) if progress
                else subprocess.run(copy_cmd, capture_output=True, text=True))
    except Exception as e:
        return None, f"ERROR: failed to run ffmpeg for audio/subtitle trim (stream copy): {e}", cmds_tried

    if proc.returncode == 0 and path.exists(trimmed_path) and path.getsize(trimmed_path) > 0:
        return trimmed_path, None, cmds_tried

    if path.exists(trimmed_path):
        try:
            os.remove(trimmed_path)
        except Exception:
            pass

    # ADR-403: real user request -- "give more details of the process". A stream-copy
    # failure here (this source's audio/subtitle codec can't just be sliced/repackaged
    # as-is at this trim point) means the fallback below has to actually RE-ENCODE
    # every audio/subtitle track instead of a near-instant byte copy -- genuinely much
    # slower, and a real, plausible reason a specific movie takes far longer than
    # others at this step. Said plainly, as it's happening, rather than silently.
    print("[av-restore] could not trim --source's audio/subtitle tracks with a fast stream copy -- "
          "falling back to a slower re-encode (this specific source's audio/subtitle codec needs it; "
          "most sources don't hit this path).", file=sys.stderr)

    reencode_cmd = _build_cmd(use_copy=False)
    cmds_tried.append(_format_cmd(reencode_cmd))
    try:
        proc2 = (_run_trim_with_progress(reencode_cmd, start_sec, end_sec) if progress
                 else subprocess.run(reencode_cmd, capture_output=True, text=True))
    except Exception as e:
        return None, f"ERROR: failed to run ffmpeg for audio/subtitle trim (re-encode fallback): {e}", cmds_tried

    if proc2.returncode == 0 and path.exists(trimmed_path) and path.getsize(trimmed_path) > 0:
        return trimmed_path, None, cmds_tried

    stderr_tail = (proc2.stderr or proc.stderr or "").strip().splitlines()[-10:]
    return None, (
        "ERROR: ffmpeg failed to trim --source's audio/subtitle tracks, both as a stream copy and "
        "as a re-encode fallback. Last ffmpeg stderr:\n" + "\n".join(stderr_tail)), cmds_tried


# Text-based subtitle codec names av/ffprobe report -- the only kind
# _build_stereo_positioned_subs can reposition (it operates on real cue text, not
# pixels). Matches iw3.sbs_to_mvc_cli._ffprobe_list_tracks's own S_TEXT set for the
# same codec names, kept as a separate local copy rather than a third cross-import --
# it's a one-line constant, not nontrivial logic that could drift out of sync.
_TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}

# ADR-255: real user question -- without a distinguishing name, a repositioned
# track shows up in a player's subtitle menu as just a bare language ("English"),
# indistinguishable from an ordinary flat track (this project has no general
# language-name lookup anywhere, so this is intentionally NOT exhaustive -- covers
# the common cases; anything else still gets a real, working, just less pretty
# label via the code itself, see _dual_eye_track_name's own fallback).
_COMMON_LANGUAGE_NAMES = {
    "eng": "English", "spa": "Spanish", "fre": "French", "ger": "German",
    "ita": "Italian", "por": "Portuguese", "jpn": "Japanese", "chi": "Chinese",
    "kor": "Korean", "rus": "Russian", "ara": "Arabic", "hin": "Hindi",
    "dut": "Dutch", "swe": "Swedish", "nor": "Norwegian", "dan": "Danish",
    "pol": "Polish", "tur": "Turkish", "fin": "Finnish", "gre": "Greek",
    "vie": "Vietnamese", "ind": "Indonesian", "cze": "Czech", "hun": "Hungarian",
    "ukr": "Ukrainian",
}


def _language_display_name(lang):
    """Readable language name for `lang` (whatever the source track's own language
    metadata says -- 2 or 3 letter, or empty/unset), e.g. "English" for "eng".
    Returns None when nothing usable is known -- callers each pick their own
    fallback text for that case (_dual_eye_track_name's "Subtitle (3D)" is a
    different fallback than ADR-403's own "Unknown language" below, so this
    deliberately doesn't bake either one in)."""
    code = _iso639_1_to_2(lang) if lang else ""
    if not code:
        return None
    return _COMMON_LANGUAGE_NAMES.get(code, code.upper())


def _dual_eye_track_name(lang):
    """ADR-255: readable mkvmerge --track-name for a repositioned subtitle track,
    e.g. "English (3D)" -- so it's distinguishable at a glance from a flat/original
    track in a player's own subtitle menu."""
    name = _language_display_name(lang)
    return f"{name} (3D)" if name else "Subtitle (3D)"


def _describe_tracks(source_path):
    """ADR-403: real user request -- "give more details of the process", starting
    with which language is actually being restored, not just a bare count. Returns
    (audio_desc, subtitle_desc), each a comma-separated "Name (codec)" list (e.g.
    "English (dts), Spanish (ac3)"), or "" when that track type is empty or the
    source couldn't be inspected for some reason (never raises -- this is purely an
    informational message, not a gate; _count_av_tracks already did the real
    existence/gating check before this is ever called)."""
    try:
        container = av.open(source_path, mode="r", metadata_errors="ignore")
        try:
            def _parts(streams):
                out = []
                for s in streams:
                    lang = (s.metadata or {}).get("language", "")
                    name = _language_display_name(lang) or "Unknown language"
                    codec = s.codec_context.name if s.codec_context else "unknown"
                    out.append(f"{name} ({codec})")
                return out
            audio_desc = ", ".join(_parts(container.streams.audio))
            subtitle_desc = ", ".join(_parts(container.streams.subtitles))
        finally:
            container.close()
        return audio_desc, subtitle_desc
    except Exception:
        return "", ""


def _dual_eye_reposition_subs(mux_source_path, work_dir, ffmpeg_bin, resolved_format, width, height, font_size):
    """ADR-254: for every TEXT subtitle stream in mux_source_path, extracts it and
    builds a dual-eye stereo-positioned ASS replacement via subtitle_mux_cli.py's own
    ADR-053 math (never duplicated here). Picture-based subtitle streams (PGS/VobSub)
    are left completely alone -- there is no text to reposition, and this project has
    no way to reposition a bitmap subtitle's own baked-in pixels; they're restored
    unchanged either way, same as before this feature existed.

    Returns (keep_ids, positioned_files, notes):
    - keep_ids: comma-separated mkvmerge --subtitle-tracks value listing which of
      mux_source_path's OWN subtitle tracks to still pull in as-is (every non-text
      one, plus any text one that failed to extract/parse -- fails safe to the
      original flat copy rather than silently dropping a track). None if there were
      no subtitle streams at all (caller keeps today's unrestricted blanket copy).
    - positioned_files: list of (ass_path, lang) tuples to mux in as new, separate
      inputs alongside mux_source_path.
    - notes: human-readable per-track outcome lines, for the caller to print.
    """
    import pysubs2

    container = av.open(mux_source_path, mode="r", metadata_errors="ignore")
    try:
        tracks = []
        for i, s in enumerate(container.streams.subtitles):
            codec_name = s.codec_context.name if s.codec_context else ""
            lang = (s.metadata or {}).get("language", "")
            tracks.append((i, codec_name, lang))
    finally:
        container.close()

    if not tracks:
        return None, [], []

    keep_ids = []
    positioned_files = []
    notes = []
    for i, codec_name, lang in tracks:
        if codec_name not in _TEXT_SUBTITLE_CODECS:
            keep_ids.append(str(i))
            notes.append(f"subtitle track {i + 1} ({codec_name or 'unknown'}): picture-based, "
                         f"restored unchanged -- only text subtitles can be repositioned for 3D")
            continue

        srt_path = path.join(work_dir, f"dualeye_sub_{i}.srt")
        r = subprocess.run([ffmpeg_bin, "-y", "-v", "error", "-i", mux_source_path,
                            "-map", f"0:s:{i}", "-c:s", "srt", srt_path],
                           capture_output=True, text=True)
        if r.returncode != 0 or not path.exists(srt_path):
            keep_ids.append(str(i))
            notes.append(f"subtitle track {i + 1} ({codec_name}): could not extract it for "
                         f"repositioning, restored unchanged instead")
            continue
        try:
            subs = pysubs2.load(srt_path)
        except Exception as e:
            keep_ids.append(str(i))
            notes.append(f"subtitle track {i + 1} ({codec_name}): could not parse it for "
                         f"repositioning ({e}), restored unchanged instead")
            continue

        positioned = _build_stereo_positioned_subs(subs, resolved_format, width, height, font_size=font_size)
        ass_path = path.join(work_dir, f"dualeye_sub_{i}.ass")
        positioned.save(ass_path)
        positioned_files.append((ass_path, lang))
        notes.append(f"subtitle track {i + 1} ({codec_name}, {lang or 'unknown language'}): "
                     f"repositioned for 3D (dual-eye)")

    keep_ids_str = ",".join(sorted(set(keep_ids), key=int)) if keep_ids else ""
    return keep_ids_str, positioned_files, notes


def run(args):
    input_path = str(args.input)
    source_path = str(args.source)
    output_path = str(args.output)

    if not path.exists(input_path):
        print(f"ERROR: --input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if path.splitext(input_path)[1].lower() != ".mkv":
        print(
            f"ERROR: --input must be an .mkv file (got '{input_path}'). This tool only supports "
            f"MKV output/input -- it does not convert containers. Re-run with an .mkv input, or "
            f"remux your existing file to .mkv first (e.g. with mkvmerge) before using this tool.",
            file=sys.stderr)
        return 1

    if path.abspath(output_path) == path.abspath(input_path):
        print("ERROR: --output must be a different path from --input -- this tool never "
              "overwrites the input.", file=sys.stderr)
        return 1

    audio_count, subtitle_count, source_error = _count_av_tracks(source_path)
    if source_error:
        print(source_error, file=sys.stderr)
        return 1

    mkvmerge_bin = _find_mkvmerge()
    if not mkvmerge_bin:
        print("ERROR: mkvmerge (MKVToolNix) was not found -- cannot mux. Expected it bundled "
              "alongside ffmpeg, or on PATH.", file=sys.stderr)
        return 1

    # ADR-403: real user request -- "which language is being processed" -- the most
    # honest answer this tool's architecture can give, since ffmpeg/mkvmerge each
    # process every track in ONE combined pass (there's no real per-track "now
    # doing X" moment to report mid-run), is to say clearly, up front, exactly
    # which tracks are about to be restored.
    audio_desc, subtitle_desc = _describe_tracks(source_path)
    audio_detail = f" [{audio_desc}]" if audio_desc else ""
    subtitle_detail = f" [{subtitle_desc}]" if subtitle_desc else ""
    print(f"[av-restore] --source has {audio_count} audio track(s){audio_detail} and "
          f"{subtitle_count} subtitle track(s){subtitle_detail} -- restoring all of them",
          file=sys.stderr)

    out_dir = path.dirname(path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output = path.splitext(output_path)[0] + ".avrestore_tmp" + path.splitext(output_path)[1]

    dual_eye_subtitles = bool(getattr(args, "dual_eye_subtitles", False))

    work_dir = None
    mux_source_path = source_path
    subtitle_track_select = []  # extra mkvmerge args placed right before mux_source_path
    positioned_files = []       # extra (ass_path, lang) mkvmerge inputs, added after mux_source_path
    try:
        if args.source_start_time or args.source_end_time or dual_eye_subtitles:
            work_dir = tempfile.mkdtemp(prefix="iw3_avrestore_")

        ffmpeg_bin = _get_ffmpeg_bin() if work_dir else None

        if args.source_start_time or args.source_end_time:
            print(f"[av-restore] trimming --source's audio/subtitle tracks to "
                  f"[{args.source_start_time or '0'}, {args.source_end_time or 'end'}) and "
                  f"shifting to start at 0...", file=sys.stderr)
            trimmed_path, trim_error, cmds_tried = _trim_all_av(
                ffmpeg_bin, source_path, args.source_start_time, args.source_end_time, work_dir, progress=True)
            for cmd_str in cmds_tried:
                print(f"[av-restore] running: {cmd_str}", file=sys.stderr)
            if trim_error:
                print(trim_error, file=sys.stderr)
                return 1
            mux_source_path = trimmed_path
            print(f"[av-restore] trimmed audio/subtitles ready: {mux_source_path}", file=sys.stderr)

        # ADR-254: --dual-eye-subtitles opt-in -- reposition every TEXT subtitle track
        # for real 3D instead of restoring it flat. Runs AFTER trimming above (so it
        # operates on the already-trimmed/rebased tracks when both are requested,
        # same "compose correctly together" ordering subtitle_mux_cli.py's own
        # --start-time/--end-time + --dual-eye-subtitles established). No effect when
        # off (the default) or when --source has no subtitle tracks at all -- the
        # final mkvmerge command below is then byte-for-byte what it always was.
        if dual_eye_subtitles and subtitle_count > 0:
            resolved_format, format_error = _resolve_format(getattr(args, "format", "auto"), input_path)
            if format_error:
                print(format_error, file=sys.stderr)
                return 1
            is_split = resolved_format in _HORIZONTAL_SPLIT_FORMATS or resolved_format in _VERTICAL_SPLIT_FORMATS
            if not is_split:
                print(f"[av-restore] --dual-eye-subtitles has no effect for format "
                      f"'{resolved_format}' -- it isn't a two-eye split layout (see ADR-053). "
                      f"Restoring subtitles unchanged.", file=sys.stderr)
            else:
                ffprobe_bin = _find_ffprobe()
                width, height, probe_error = _probe_video_dimensions(input_path, ffprobe_bin)
                if probe_error:
                    print(probe_error, file=sys.stderr)
                    print("ERROR: cannot correctly position stereo subtitles for a split-eye "
                          "layout without --input's real video dimensions -- refusing rather "
                          "than restoring possibly-mispositioned subtitles.", file=sys.stderr)
                    return 1
                font_size = getattr(args, "font_size", None)
                resolved_font_size = font_size if font_size is not None else _default_font_size(
                    resolved_format, height)
                print(f"[av-restore] --dual-eye-subtitles: {resolved_format}, "
                      f"{width}x{height}, font size {resolved_font_size}px "
                      f"({'explicit --font-size' if font_size is not None else 'auto-scaled'})",
                      file=sys.stderr)
                keep_ids, positioned_files, notes = _dual_eye_reposition_subs(
                    mux_source_path, work_dir, ffmpeg_bin or _get_ffmpeg_bin(),
                    resolved_format, width, height, font_size)
                for n in notes:
                    print(f"[av-restore] {n}", file=sys.stderr)
                if keep_ids is not None:
                    subtitle_track_select = ["--subtitle-tracks", keep_ids] if keep_ids else ["--no-subtitles"]

        # --no-audio --no-subtitles on input_path drops its own single audio track
        # (see audio_restore_cli.py's docstring -- it's already just a copy of the
        # source's own first track) and any subtitle tracks it somehow had (none,
        # in practice -- the main pipeline never adds any), keeping video/chapters
        # untouched. --no-video --no-chapters on the source keeps everything else
        # (audio + subtitles together). No --language/--track-name overrides --
        # every track's own metadata from --source is preserved as mkvmerge's
        # default "copy everything from this file" behavior for a file with no
        # options in front of it. subtitle_track_select (ADR-254, empty unless
        # --dual-eye-subtitles actually repositioned something) narrows which of
        # mux_source_path's OWN subtitle tracks still come through unchanged --
        # empty list here is a true no-op, so the default path is byte-for-byte
        # what this command always was.
        cmd = [mkvmerge_bin, "-o", tmp_output, "--no-audio", "--no-subtitles", input_path,
               "--no-video", "--no-chapters"] + subtitle_track_select + [mux_source_path]
        # Positioned ASS tracks (ADR-254), each as its own separate mkvmerge input --
        # same "--language 0:xx precedes a single-track file" convention
        # subtitle_mux_cli.py's own mux command already established.
        for ass_path, lang in positioned_files:
            if lang:
                cmd += ["--language", f"0:{_iso639_1_to_2(lang)}"]
            # ADR-255: a real, readable name ("English (3D)") so this track is
            # distinguishable at a glance from a flat/original one in a player's
            # own subtitle menu -- real user question ("will have something like
            # a 3d name tag on it?"), previously unset/blank either here or in
            # subtitle_mux_cli.py's own --dual-eye-subtitles track.
            cmd += ["--track-name", f"0:{_dual_eye_track_name(lang)}"]
            cmd.append(ass_path)

        # ADR-403: real user request -- "give more details of the process". mkvmerge's
        # own mux is a single combined pass over everything being written (--input's
        # video untouched + the restored audio/subtitles), so its own "Progress: NN%"
        # can sit at 0% for a while before the first tick (file analysis) on a large or
        # many-track source -- this gives a sense of SCALE up front, before that gap,
        # rather than leaving the user looking at a stuck 0% with no explanation.
        try:
            mux_bytes = path.getsize(input_path) + path.getsize(mux_source_path)
            for ass_path, _lang in positioned_files:
                mux_bytes += path.getsize(ass_path)
            rough_eta = _rough_mux_eta_range(mux_bytes)
            print(f"[av-restore] muxing ~{mux_bytes / (1024 ** 3):.1f} GB combined (video + restored "
                  f"audio/subtitles) into the final file -- this step is disk-bound, so it can take a "
                  f"while on large or many-track sources even though nothing is being re-encoded. Very "
                  f"roughly {rough_eta} depending on your disk speed -- this is a GUESS from file size "
                  f"alone, not a real measurement (there's no real progress to measure yet until "
                  f"mkvmerge actually starts writing).",
                  file=sys.stderr)
        except OSError:
            pass  # purely informational -- never block the real mux over a failed size check

        print(f"[av-restore] running: {_format_cmd(cmd)}", file=sys.stderr)

        try:
            proc = _run_mkvmerge_with_progress(cmd)
        except Exception as e:
            print(f"ERROR: failed to run mkvmerge: {e}", file=sys.stderr)
            return 1

        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        # mkvmerge exit codes: 0 = success, 1 = success with warnings, 2+ = failure.
        if proc.returncode >= 2:
            print(f"ERROR: mkvmerge failed (exit code {proc.returncode}).", file=sys.stderr)
            if path.exists(tmp_output):
                try:
                    os.remove(tmp_output)
                except Exception:
                    pass
            return 1

        if not path.exists(tmp_output):
            print("ERROR: mkvmerge reported success but produced no output file.", file=sys.stderr)
            return 1

        os.replace(tmp_output, output_path)
        reposition_note = f" ({len(positioned_files)} repositioned for 3D)" if positioned_files else ""
        print(f"[av-restore] done -- wrote {output_path} with {audio_count} audio track(s) and "
              f"{subtitle_count} subtitle track(s) restored from source{reposition_note}", file=sys.stderr)
        return 0
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _self_test_trim_cmd_construction():
    """Mocked test of _trim_all_av's ffmpeg command construction -- no real
    ffmpeg/network/GPU needed. Confirms both `-map 0:a?` and `-map 0:s?` (optional
    streams, so a source missing one track type doesn't error) are used together
    in one pass."""
    from unittest.mock import patch, MagicMock

    with patch.object(subprocess, "run") as mock_run, \
         patch("os.path.getsize", return_value=1234):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch("os.path.exists", return_value=True):
            trimmed, error, cmds = _trim_all_av(
                "ffmpeg", "movie.mkv", "00:01:15", "00:06:15", "/tmp/work")
        assert error is None, error
        assert trimmed.endswith(".mkv"), trimmed
        assert len(cmds) == 1, cmds
        copy_cmd = mock_run.call_args_list[0][0][0]
        assert "-ss" in copy_cmd and "75.0" in copy_cmd, copy_cmd
        assert "-to" in copy_cmd and "375.0" in copy_cmd, copy_cmd
        assert "0:a?" in copy_cmd, copy_cmd
        assert "0:s?" in copy_cmd, copy_cmd
        assert "-c" in copy_cmd and "copy" in copy_cmd, copy_cmd

    # Stream copy fails -> falls back to re-encode, which succeeds
    with patch.object(subprocess, "run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="copy failed"),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=999), \
             patch("os.remove"):
            trimmed, error, cmds = _trim_all_av(
                "ffmpeg", "movie.mkv", "10", "20", "/tmp/work")
        assert error is None, error
        assert len(cmds) == 2, cmds
        reencode_cmd = mock_run.call_args_list[1][0][0]
        assert "-c" not in reencode_cmd, reencode_cmd

    # Both attempts fail -> clear error
    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="nope")
        with patch("os.path.exists", return_value=False):
            trimmed, error, cmds = _trim_all_av(
                "ffmpeg", "movie.mkv", "10", "20", "/tmp/work")
        assert trimmed is None
        assert error is not None and "ffmpeg failed to trim" in error

    # end_time <= start_time caught before ever invoking ffmpeg
    trimmed, error, cmds = _trim_all_av("ffmpeg", "movie.mkv", "20", "10", "/tmp/work")
    assert trimmed is None and error is not None and cmds == []

    print("_self_test_trim_cmd_construction: PASS")


def _self_test_av_track_counting():
    """Mocked test of _count_av_tracks -- missing file, a source with neither
    track type (error), audio-only (NOT an error -- graceful degradation),
    subtitle-only (also not an error), and both present. No real file/av decode
    needed."""
    from unittest.mock import patch, MagicMock

    audio, sub, error = _count_av_tracks("/does/not/exist.mkv")
    assert audio is None and sub is None and error is not None and "does not exist" in error

    with patch("os.path.exists", return_value=True), patch.object(av, "open") as mock_open:
        mock_container = MagicMock()
        mock_container.streams.audio = []
        mock_container.streams.subtitles = []
        mock_open.return_value = mock_container
        audio, sub, error = _count_av_tracks("nothing.mkv")
        assert audio == 0 and sub == 0 and error is not None and "no audio or subtitle" in error

    with patch("os.path.exists", return_value=True), patch.object(av, "open") as mock_open:
        mock_container = MagicMock()
        mock_container.streams.audio = [MagicMock(), MagicMock()]
        mock_container.streams.subtitles = []
        mock_open.return_value = mock_container
        audio, sub, error = _count_av_tracks("audio_only.mkv")
        assert audio == 2 and sub == 0 and error is None, (audio, sub, error)

    with patch("os.path.exists", return_value=True), patch.object(av, "open") as mock_open:
        mock_container = MagicMock()
        mock_container.streams.audio = []
        mock_container.streams.subtitles = [MagicMock()]
        mock_open.return_value = mock_container
        audio, sub, error = _count_av_tracks("subs_only.mkv")
        assert audio == 0 and sub == 1 and error is None, (audio, sub, error)

    with patch("os.path.exists", return_value=True), patch.object(av, "open") as mock_open:
        mock_container = MagicMock()
        mock_container.streams.audio = [MagicMock()]
        mock_container.streams.subtitles = [MagicMock(), MagicMock(), MagicMock()]
        mock_open.return_value = mock_container
        audio, sub, error = _count_av_tracks("both.mkv")
        assert audio == 1 and sub == 3 and error is None, (audio, sub, error)

    print("_self_test_av_track_counting: PASS")


def _self_test_describe_tracks():
    """ADR-403: real user request -- "which language is being processed". Mocked
    test of _describe_tracks/_language_display_name -- no real file/av decode
    needed. Confirms: known language codes resolve to their real names, an unknown
    code falls back to its own uppercased code (never silently dropped), a track
    with no language metadata at all falls back to "Unknown language" (distinct
    from _dual_eye_track_name's own "Subtitle (3D)" fallback -- same underlying
    helper, two different callers, two different fallback strings), empty track
    lists produce an empty string (not "Unknown language" with nothing after it),
    and a source av.open() can't even inspect returns ("", "") rather than raising."""
    from unittest.mock import patch, MagicMock

    def _stream(lang, codec_name):
        s = MagicMock()
        s.metadata = {"language": lang} if lang else {}
        s.codec_context = MagicMock()
        s.codec_context.name = codec_name
        return s

    with patch("os.path.exists", return_value=True), patch.object(av, "open") as mock_open:
        mock_container = MagicMock()
        mock_container.streams.audio = [_stream("eng", "dts"), _stream("spa", "ac3"),
                                        _stream("", "aac"), _stream("xyz", "flac")]
        mock_container.streams.subtitles = [_stream("eng", "subrip")]
        mock_open.return_value = mock_container
        audio_desc, subtitle_desc = _describe_tracks("movie.mkv")
        assert audio_desc == "English (dts), Spanish (ac3), Unknown language (aac), XYZ (flac)", audio_desc
        assert subtitle_desc == "English (subrip)", subtitle_desc

    # no subtitle tracks at all -> "" (not "Unknown language" with nothing to pair it with)
    with patch("os.path.exists", return_value=True), patch.object(av, "open") as mock_open:
        mock_container = MagicMock()
        mock_container.streams.audio = [_stream("eng", "aac")]
        mock_container.streams.subtitles = []
        mock_open.return_value = mock_container
        audio_desc, subtitle_desc = _describe_tracks("audio_only.mkv")
        assert audio_desc == "English (aac)", audio_desc
        assert subtitle_desc == "", subtitle_desc

    # av.open() itself fails -> ("", ""), never raises (purely informational)
    with patch.object(av, "open", side_effect=RuntimeError("could not decode")):
        audio_desc, subtitle_desc = _describe_tracks("broken.mkv")
        assert (audio_desc, subtitle_desc) == ("", ""), (audio_desc, subtitle_desc)

    # _dual_eye_track_name's own fallback is unaffected by the refactor into
    # _language_display_name -- same real-code/unknown-code/no-code behavior as before.
    assert _dual_eye_track_name("eng") == "English (3D)"
    assert _dual_eye_track_name("xyz") == "XYZ (3D)"
    assert _dual_eye_track_name("") == "Subtitle (3D)"
    assert _dual_eye_track_name(None) == "Subtitle (3D)"

    print("_self_test_describe_tracks: PASS")


def _self_test_rough_mux_eta_range():
    """ADR-403 Amendment: decker's direct follow-up -- "so there is no way to show
    an ETA for that?" -- confirms _rough_mux_eta_range's own math and formatting,
    pure arithmetic, no real file/subprocess needed: the fast-assumption bound is
    always <= the slow-assumption bound (never inverted), a tiny size still rounds
    up to at least "1 min" on both ends (never "0 min", which would misleadingly
    read as "basically instant"), and _format_rough_minutes crosses over to the
    "Xh Ymin" form correctly at and past 60 minutes."""
    GB = 1024 ** 3

    # 400 MB/s fast bound vs. 50 MB/s slow bound -- an 8x spread, so for any real
    # size the slow-side minute count must be >= the fast-side one.
    for size in (1 * GB, 10 * GB, 50 * GB, 100 * GB):
        result = _rough_mux_eta_range(size)
        fast_str, slow_str = result.split("-", 1)
        fast_minutes = int(fast_str.split()[0]) if "min" in fast_str and "h" not in fast_str else None
        slow_minutes = int(slow_str.split()[0]) if "min" in slow_str and "h" not in slow_str else None
        if fast_minutes is not None and slow_minutes is not None:
            assert fast_minutes <= slow_minutes, (size, result)

    # a tiny size never rounds down to "0 min" on either bound (would misleadingly
    # read as "basically instant").
    tiny_result = _rough_mux_eta_range(1024)  # 1 KB
    assert "0 min" not in tiny_result, tiny_result
    assert tiny_result == "1 min-1 min", tiny_result

    # _format_rough_minutes: under 60 -> "N min"; at/over 60 -> "Xh Ymin" / "Xh"
    assert _format_rough_minutes(30) == "1 min", _format_rough_minutes(30)
    assert _format_rough_minutes(59 * 60) == "59 min", _format_rough_minutes(59 * 60)
    assert _format_rough_minutes(60 * 60) == "1h", _format_rough_minutes(60 * 60)
    assert _format_rough_minutes(75 * 60) == "1h 15min", _format_rough_minutes(75 * 60)
    assert _format_rough_minutes(125 * 60) == "2h 5min", _format_rough_minutes(125 * 60)

    print("_self_test_rough_mux_eta_range: PASS")


def _self_test_run_gating():
    """Synthetic/mocked test of run()'s pre-mkvmerge gates (non-mkv input,
    output==input, source with neither track type) -- proves each refuses before
    mkvmerge/ffmpeg is ever invoked, same convention as audio_restore_cli.py's
    own _self_test_run_gating()."""
    import tempfile as _tempfile
    from unittest.mock import patch

    with _tempfile.TemporaryDirectory(prefix="iw3_avrestore_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie_3d.mkv")
        mp4_input = path.join(tmpdir, "movie_3d.mp4")
        source_path = path.join(tmpdir, "source.mkv")
        output_path = path.join(tmpdir, "out.mkv")

        for p in (mkv_input, mp4_input, source_path):
            with open(p, "wb") as f:
                f.write(b"0")

        def _args(**overrides):
            base = dict(input=mkv_input, source=source_path, output=output_path,
                        source_start_time=None, source_end_time=None)
            base.update(overrides)
            return argparse.Namespace(**base)

        with patch(f"{__name__}._find_mkvmerge") as mock_find:
            rc = run(_args(input=mp4_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

            rc = run(_args(output=mkv_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

        with patch(f"{__name__}._count_av_tracks", return_value=(0, 0, "ERROR: no tracks")), \
             patch(f"{__name__}._find_mkvmerge") as mock_find:
            rc = run(_args())
            assert rc == 1, rc
            mock_find.assert_not_called()

        with patch(f"{__name__}._count_av_tracks", return_value=(2, 0, None)), \
             patch(f"{__name__}._find_mkvmerge", return_value=None):
            rc = run(_args())
            assert rc == 1, rc  # reaches the mkvmerge stage, refuses because it's missing

    print("_self_test_run_gating: PASS")


def _self_test_dual_eye_subtitles_real_ffmpeg():
    """ADR-254: real user request -- "Restore Audio & Subtitles" should be able to
    restore a subtitle already positioned for real 3D, not just flat-copied, the
    same way iw3.subtitle_mux_cli's own --dual-eye-subtitles (ADR-053) already can
    for a manually-added track. Not mocked -- builds a real source .mkv (audio +
    one real text subtitle cue) and a real "already-converted" 3D .mkv via the
    bundled ffmpeg (no GPU needed), then calls the real run() end to end, twice:
    once with --dual-eye-subtitles on (confirms the restored subtitle track is a
    real repositioned ASS with two '\\pos(...)' events for the one source cue, one
    per eye) and once off (confirms today's default behavior -- a flat, unpositioned
    copy -- is completely unchanged)."""
    import tempfile as _tempfile
    from .utils import _get_ffmpeg_bin as get_ffmpeg_bin

    ffmpeg_bin = get_ffmpeg_bin()
    assert ffmpeg_bin is not None, "bundled ffmpeg must resolve for this test to be meaningful"

    with _tempfile.TemporaryDirectory(prefix="iw3_avrestore_dualeye_selftest_") as tmpdir:
        srt_path = path.join(tmpdir, "sub.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

        source_path = path.join(tmpdir, "source.mkv")
        r = subprocess.run(
            [ffmpeg_bin, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-i", srt_path, "-map", "0:a", "-map", "1:s", "-c:a", "aac", "-c:s", "srt",
             "-metadata:s:s:0", "language=eng", source_path],
            capture_output=True, text=True)
        assert r.returncode == 0 and path.exists(source_path), r.stderr

        input_path = path.join(tmpdir, "movie_LR.mkv")
        r = subprocess.run(
            [ffmpeg_bin, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:d=1",
             "-c:v", "libx264", input_path], capture_output=True, text=True)
        assert r.returncode == 0 and path.exists(input_path), r.stderr

        def _args(**overrides):
            base = dict(input=input_path, source=source_path,
                        source_start_time=None, source_end_time=None,
                        dual_eye_subtitles=False, format="half_sbs", font_size=None)
            base.update(overrides)
            return argparse.Namespace(**base)

        # (a) off (default): flat copy, no repositioning at all.
        output_flat = path.join(tmpdir, "out_flat.mkv")
        rc = run(_args(output=output_flat))
        assert rc == 0, rc
        assert path.exists(output_flat)
        extracted_flat = path.join(tmpdir, "extracted_flat.srt")
        subprocess.run([ffmpeg_bin, "-y", "-v", "error", "-i", output_flat, "-map", "0:s:0",
                        extracted_flat], capture_output=True, text=True)
        with open(extracted_flat, encoding="utf-8") as f:
            flat_text = f.read()
        assert "\\pos(" not in flat_text, flat_text
        assert "Hello" in flat_text, flat_text

        # (b) on: the restored subtitle must be a real, positioned ASS track --
        # two '\pos(...)' events (one per eye) for the one original cue.
        output_3d = path.join(tmpdir, "out_3d.mkv")
        rc = run(_args(output=output_3d, dual_eye_subtitles=True))
        assert rc == 0, rc
        assert path.exists(output_3d)
        extracted_3d = path.join(tmpdir, "extracted_3d.ass")
        r = subprocess.run([ffmpeg_bin, "-y", "-v", "error", "-i", output_3d, "-map", "0:s:0",
                            extracted_3d], capture_output=True, text=True)
        assert r.returncode == 0 and path.exists(extracted_3d), r.stderr
        with open(extracted_3d, encoding="utf-8") as f:
            positioned_text = f.read()
        assert positioned_text.count("\\pos(") == 2, positioned_text
        assert positioned_text.count("Hello") == 2, positioned_text

        # ADR-255: real user question -- the repositioned track must have a real,
        # readable name ("English (3D)") so it's distinguishable from a flat one in
        # a player's own subtitle menu, not just a bare language. Read back via
        # ffprobe (mkvmerge's own --track-name becomes the container's real title tag).
        ffprobe_bin = _find_ffprobe()
        r = subprocess.run(
            [ffprobe_bin, "-v", "error", "-select_streams", "s:0", "-show_entries",
             "stream_tags=title", "-of", "default=noprint_wrappers=1:nokey=1", output_3d],
            capture_output=True, text=True)
        assert r.stdout.strip() == "English (3D)", (r.stdout, r.stderr)

    print("_self_test_dual_eye_subtitles_real_ffmpeg: PASS")


def _run_self_tests():
    _self_test_trim_cmd_construction()
    _self_test_av_track_counting()
    _self_test_describe_tracks()
    _self_test_rough_mux_eta_range()
    _self_test_run_gating()
    _self_test_dual_eye_subtitles_real_ffmpeg()
    print("All av_restore_cli self-tests PASSED")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
