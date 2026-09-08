"""python -m iw3.audio_mux_cli -- standalone tool to add an audio file (e.g. a
different-language dub) as a NEW track to an already-converted 3D video, preserving
every existing track (video, existing audio, subtitles) untouched.

Follows the exact same create_parser()/run()/main() shape as iw3.subtitle_mux_cli
(see docs/ai/AI_DECISIONS.md ADR-032) and iw3.reinject_hdr_cli (ADR-031) -- read
those two modules' docstrings for the established conventions this one reuses:
MKV output only, --output is never --input, temp-then-os.replace() safety, mkvmerge
"copy every track from a file with no options in front of it" behavior for
preserving the input untouched.

Real-world use case (see docs/ai/AI_DECISIONS.md ADR-042): a user has a short
already-converted 3D test clip (e.g. a 5-minute excerpt, 1:15-6:15 of a 2-hour
movie) and a full-movie audio dub track they want to add to it. The dub audio
covers more content (and a different absolute timing offset) than the clip -- it
must be TRIMMED to the exact [1:15, 6:15) window AND shifted so the trimmed audio
starts at t=0 to line up with the clip's own first frame. --source-start-time/
--source-end-time do exactly that automatically via ffmpeg, rather than requiring
the user to hand-cut the audio first (as had to be done manually for the SRT
case this mirrors -- see subtitle_mux_cli.py's own docstring/ADR-032 for why a
plain soft track, not a stereo-baked one, is also the right shape for audio here:
audio has no per-eye geometry concern at all, so that whole question doesn't even
apply -- this is simply "add one more soft audio track").

Trimming implementation notes (see _trim_audio_source): ffmpeg input-side -ss/-to
(the same absolute-range convention already used by reinject_hdr_cli.py's
_probe_frames_and_duration and the Dolby Vision RPU extraction step documented in
docs/ai/domains/DOLBY_VISION.md -- -ss/-to as INPUT options are absolute positions
within the source, not "-to relative to -ss") performs both the trim AND the
start-at-0 shift in one step: input seeking rebases output timestamps so the first
kept sample lands at (near) PTS 0, and -avoid_negative_ts make_zero guards against
a small negative-timestamp edge case some players mishandle after a stream-copy
trim. -c copy is tried first (fast, lossless, no re-encode) since it is reliable
for arbitrary cut points on every common compressed/PCM audio codec this tool
targets -- ffmpeg simply drops/keeps whole codec frames/packets around the
requested boundary, which for AAC/MP3/AC3/DTS/Opus/FLAC (short, tens-of-ms frames)
means at most one short frame's worth of imprecision, and WAV/PCM cuts on exact
sample boundaries. If stream copy fails outright (uncommon container quirk,
unsupported cut, etc.) this falls back to a re-encode into the same file extension
using ffmpeg's own default encoder for that container, which always works for a
timestamp-based trim at the cost of a generation of lossy re-encoding when the
source was already lossy.

Language convention (see docs/ai/AI_DECISIONS.md ADR-044): --language takes the
same user-facing ISO 639-1 two-letter form (en/es/fr/...) subtitle_mux_cli.py
already standardized on in ADR-042 (landed before this tool was finished -- checked
at implementation time per this tool's own task brief). Rather than writing a
second copy of that ISO 639-1 -> ISO 639-2 table, this module imports
subtitle_mux_cli.py's exact `_ISO_639_1_TO_2` table and `_iso639_1_to_2()` function
directly -- see ADR-044 for why this direction (audio_mux importing from
subtitle_mux, not the reverse) was chosen: subtitle_mux_cli.py's table landed
first and is the more complete of the two (broader language coverage), and
audio_mux_cli.py is the newer module of the two, so it is the natural importer
rather than subtitle_mux_cli.py taking on a dependency on the module that came
after it.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from os import path

from nunif.utils.video.metadata import parse_time
from .utils import _find_mkvmerge, _get_ffmpeg_bin
from .subtitle_mux_cli import _iso639_1_to_2


def _format_cmd(cmd):
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


def _resolve_mkv_language(user_code):
    """Converts a user-facing ISO 639-1 two-letter code to the ISO 639-2 three-letter
    code mkvmerge stores as track metadata, via subtitle_mux_cli.py's shared
    `_iso639_1_to_2()` (see module docstring's "Language convention" section). That
    function already handles an already-3-letter code (passed through unchanged) and
    any code not in its table (passed through lowercased/stripped) -- mkvmerge itself
    has the final say on a genuinely invalid code, rather than this tool guessing/
    silently mangling it.

    Returns (resolved_code, was_recognized_bool) -- was_recognized_bool is kept in
    this function's own return shape (rather than exposing subtitle_mux_cli's table
    directly) so callers here don't need to reach into that module's internals just
    to log whether the code was a known one."""
    code = str(user_code).strip().lower()
    resolved = _iso639_1_to_2(code)
    return resolved, resolved != code


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.audio_mux_cli",
        description=(
            "Add an audio file (e.g. a different-language dub) as a new audio track to an "
            "already-converted 3D (SBS/TB) MKV video, preserving every existing track (video, "
            "audio, subtitles) untouched. Never modifies --input or --audio -- always writes a "
            "new file at --output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                         help="Path to the already-converted 3D video. Must be a .mkv file -- "
                              "this tool does not convert containers. Read-only, never modified.")
    parser.add_argument("--audio", "-a", type=str, required=True,
                         help="Path to the audio file to add as a new track (AAC, AC3, DTS, FLAC, "
                              "MP3, Opus, WAV, or any other format mkvmerge/ffmpeg already read "
                              "directly -- no format conversion is attempted beyond the optional "
                              "trim below). Read-only, never modified.")
    parser.add_argument("--output", "-o", type=str, required=True,
                         help="Path to write the new file to. Must be a different path from "
                              "--input -- this tool never overwrites the input.")
    parser.add_argument("--language", type=str, default="en",
                         help="ISO 639-1 two-letter language code for the new audio track (e.g. "
                              "en, es, fr, ja). Converted to the ISO 639-2 three-letter code "
                              "mkvmerge stores as metadata (e.g. en -> eng) -- purely metadata, "
                              "does not affect muxing or the actual audio content.")
    parser.add_argument("--track-name", type=str, default=None,
                         help="Display name for the new audio track, shown in player track menus "
                              "(e.g. 'Spanish Dub'). Default: the audio file's own name (without "
                              "extension).")
    parser.add_argument("--default", action="store_true",
                         help="Mark the new audio track as the default track a player selects "
                              "automatically. Off by default -- usually you're adding an ALTERNATE "
                              "audio track, not replacing the primary one. Only the new track's own "
                              "default flag is set either way -- any existing track's default flag "
                              "in --input is preserved unchanged (see module docstring / "
                              "docs/ai/AI_DECISIONS.md ADR-042 for the resulting edge case if "
                              "--input already had a default audio track).")
    parser.add_argument("--source-start-time", type=str, default=None,
                         help="Start time WITHIN --audio to trim to (HH:MM:SS, MM:SS, or seconds), "
                              "for when --audio covers more content than --input (e.g. a full-movie "
                              "dub being added to a short test clip). The trimmed audio is also "
                              "automatically shifted to start at t=0 so it lines up with --input's "
                              "first frame -- you do not need to do this by hand. Default: start of "
                              "--audio (the whole file is used as-is if both --source-start-time and "
                              "--source-end-time are omitted).")
    parser.add_argument("--source-end-time", type=str, default=None,
                         help="End time within --audio to trim to. Default: end of --audio.")
    return parser


def _validate_audio(audio_path):
    """Existence check for --audio. Deliberately does NOT parse/decode the file the
    way subtitle_mux_cli.py's _validate_srt() does for SRT (there is no equivalent
    lightweight Python-level audio parser already a project dependency the way
    pysubs2 is for subtitles) -- a malformed/unsupported audio file is instead caught
    by ffmpeg (during an optional trim) or mkvmerge (during the final mux), both of
    which print a clear reason to the log either way."""
    if not path.exists(audio_path):
        return f"ERROR: --audio file does not exist: {audio_path}"
    return None


def _trim_audio_source(ffmpeg_bin, audio_path, start_time, end_time, work_dir):
    """Extracts [start_time, end_time) from audio_path into a new temp file in
    work_dir, ALSO shifting the result to start at t=0 (see module docstring's
    "Trimming implementation notes"). Tries a fast lossless -c copy first, falling
    back to a re-encode (ffmpeg's own default encoder for the output extension) only
    if stream copy fails outright.

    Returns (trimmed_path_or_None, error_message_or_None, cmds_tried (list of str)).
    """
    ext = path.splitext(audio_path)[1] or ".mka"
    trimmed_path = path.join(work_dir, f"trimmed_audio{ext}")

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
        cmd += ["-i", str(audio_path), "-vn", "-map", "0:a:0"]
        if use_copy:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        cmd += [trimmed_path]
        return cmd

    cmds_tried = []

    copy_cmd = _build_cmd(use_copy=True)
    cmds_tried.append(_format_cmd(copy_cmd))
    try:
        proc = subprocess.run(copy_cmd, capture_output=True, text=True)
    except Exception as e:
        return None, f"ERROR: failed to run ffmpeg for audio trim (stream copy): {e}", cmds_tried

    if proc.returncode == 0 and path.exists(trimmed_path) and path.getsize(trimmed_path) > 0:
        return trimmed_path, None, cmds_tried

    # Stream copy failed outright -- fall back to a re-encode into the same extension
    # using ffmpeg's own default encoder for that container. Clean up any partial file
    # the failed attempt may have left behind first.
    if path.exists(trimmed_path):
        try:
            os.remove(trimmed_path)
        except Exception:
            pass

    reencode_cmd = _build_cmd(use_copy=False)
    cmds_tried.append(_format_cmd(reencode_cmd))
    try:
        proc2 = subprocess.run(reencode_cmd, capture_output=True, text=True)
    except Exception as e:
        return None, f"ERROR: failed to run ffmpeg for audio trim (re-encode fallback): {e}", cmds_tried

    if proc2.returncode == 0 and path.exists(trimmed_path) and path.getsize(trimmed_path) > 0:
        return trimmed_path, None, cmds_tried

    stderr_tail = (proc2.stderr or proc.stderr or "").strip().splitlines()[-10:]
    return None, (
        "ERROR: ffmpeg failed to trim --audio, both as a stream copy and as a re-encode "
        "fallback. Last ffmpeg stderr:\n" + "\n".join(stderr_tail)), cmds_tried


def run(args):
    input_path = str(args.input)
    audio_path = str(args.audio)
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

    audio_error = _validate_audio(audio_path)
    if audio_error:
        print(audio_error, file=sys.stderr)
        return 1

    mkvmerge_bin = _find_mkvmerge()
    if not mkvmerge_bin:
        print("ERROR: mkvmerge (MKVToolNix) was not found -- cannot mux. Expected it bundled "
              "alongside ffmpeg, or on PATH.", file=sys.stderr)
        return 1

    resolved_lang, recognized = _resolve_mkv_language(args.language)
    print(f"[audio-mux] resolved language: {args.language} -> {resolved_lang}"
          f"{'' if recognized else ' (not in the built-in ISO 639-1 table, passed through as-is)'}",
          file=sys.stderr)

    track_name = args.track_name or path.splitext(path.basename(audio_path))[0]

    out_dir = path.dirname(path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output = path.splitext(output_path)[0] + ".audiomux_tmp" + path.splitext(output_path)[1]

    work_dir = None
    mux_audio_path = audio_path
    try:
        if args.source_start_time or args.source_end_time:
            ffmpeg_bin = _get_ffmpeg_bin()
            work_dir = tempfile.mkdtemp(prefix="iw3_audiomux_")
            print(f"[audio-mux] trimming --audio to [{args.source_start_time or '0'}, "
                  f"{args.source_end_time or 'end'}) and shifting to start at 0...", file=sys.stderr)
            trimmed_path, trim_error, cmds_tried = _trim_audio_source(
                ffmpeg_bin, audio_path, args.source_start_time, args.source_end_time, work_dir)
            for cmd_str in cmds_tried:
                print(f"[audio-mux] running: {cmd_str}", file=sys.stderr)
            if trim_error:
                print(trim_error, file=sys.stderr)
                return 1
            mux_audio_path = trimmed_path
            print(f"[audio-mux] trimmed audio ready: {mux_audio_path}", file=sys.stderr)

        # mkvmerge: per-file options (--language/--track-name/--default-track-flag) must
        # appear BEFORE the file they apply to. "0:" addresses track 0 OF THAT FILE (the
        # audio file has exactly one audio track we care about). <input_path> is given
        # last with no options preceding it, which is mkvmerge's default "copy every
        # track from this file, unmodified" behavior -- this is what preserves the
        # original video/audio/subtitle tracks untouched, same as already confirmed for
        # subtitle_mux_cli.py's SRT case (see docs/ai/AI_DECISIONS.md ADR-032/042).
        cmd = [mkvmerge_bin, "-o", tmp_output,
               "--language", f"0:{resolved_lang}",
               "--track-name", f"0:{track_name}"]
        if args.default:
            cmd += ["--default-track-flag", "0:yes"]
        cmd += [mux_audio_path, input_path]
        print(f"[audio-mux] running: {_format_cmd(cmd)}", file=sys.stderr)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
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
        print(f"[audio-mux] done -- wrote {output_path}", file=sys.stderr)
        return 0
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _self_test_language_resolution():
    """Synthetic test of the ISO 639-1 -> ISO 639-2 table -- no ffmpeg/mkvmerge/GPU
    needed."""
    cases = {
        "en": "eng", "EN": "eng", " es ": "spa", "fr": "fre", "de": "ger",
        "ja": "jpn", "zh": "chi",
    }
    for code, expected in cases.items():
        resolved, recognized = _resolve_mkv_language(code)
        assert resolved == expected and recognized, (code, resolved, recognized)

    # 3-letter code already -- passed through unchanged, not "recognized" by this table
    resolved, recognized = _resolve_mkv_language("eng")
    assert resolved == "eng" and not recognized, (resolved, recognized)

    # Unknown code -- passed through as-is (lowercased), mkvmerge gets the final say
    resolved, recognized = _resolve_mkv_language("xx")
    assert resolved == "xx" and not recognized, (resolved, recognized)

    print("_self_test_language_resolution: PASS")


def _self_test_trim_cmd_construction():
    """Mocked test of _trim_audio_source's ffmpeg command construction -- no real
    ffmpeg/network/GPU needed. Covers: stream-copy-succeeds (the common case),
    stream-copy-fails-then-reencode-succeeds (the fallback path), and both attempts
    failing (a clear error is returned, not a silent False)."""
    from unittest.mock import patch, MagicMock

    with patch.object(subprocess, "run") as mock_run, \
         patch("os.path.getsize", return_value=1234):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch("os.path.exists", return_value=True):
            trimmed, error, cmds = _trim_audio_source(
                "ffmpeg", "dub.mp3", "00:01:15", "00:06:15", "/tmp/work")
        assert error is None, error
        assert trimmed.endswith(".mp3"), trimmed
        assert len(cmds) == 1, cmds  # only the copy attempt was needed
        copy_cmd = mock_run.call_args_list[0][0][0]
        assert "-ss" in copy_cmd and "75.0" in copy_cmd, copy_cmd
        assert "-to" in copy_cmd and "375.0" in copy_cmd, copy_cmd
        assert "-c" in copy_cmd and "copy" in copy_cmd, copy_cmd
        assert "-avoid_negative_ts" in copy_cmd, copy_cmd

    # Stream copy fails (returncode != 0) -> falls back to re-encode, which succeeds
    with patch.object(subprocess, "run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="copy failed"),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=999), \
             patch("os.remove"):
            trimmed, error, cmds = _trim_audio_source(
                "ffmpeg", "dub.flac", "10", "20", "/tmp/work")
        assert error is None, error
        assert len(cmds) == 2, cmds  # copy attempt, then re-encode attempt
        reencode_cmd = mock_run.call_args_list[1][0][0]
        assert "-c" not in reencode_cmd, reencode_cmd  # no explicit codec -> ffmpeg default

    # Both attempts fail -> clear error, not a crash or silent None
    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="nope")
        with patch("os.path.exists", return_value=False):
            trimmed, error, cmds = _trim_audio_source(
                "ffmpeg", "dub.wav", "10", "20", "/tmp/work")
        assert trimmed is None
        assert error is not None and "ffmpeg failed to trim" in error

    # No trim requested at all is handled entirely in run(), not this function -- but
    # end_time <= start_time must be caught here before ever invoking ffmpeg.
    trimmed, error, cmds = _trim_audio_source("ffmpeg", "dub.wav", "20", "10", "/tmp/work")
    assert trimmed is None and error is not None and cmds == []

    print("_self_test_trim_cmd_construction: PASS")


def _self_test_run_gating():
    """Synthetic/mocked test of run()'s pre-mkvmerge gates (non-mkv input, output==input,
    missing audio file) -- proves each refuses before mkvmerge/ffmpeg is ever invoked,
    same convention as subtitle_mux_cli.py's _self_test_run_gating()."""
    import tempfile as _tempfile
    from unittest.mock import patch

    with _tempfile.TemporaryDirectory(prefix="iw3_audiomux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        mp4_input = path.join(tmpdir, "movie.mp4")
        audio_path = path.join(tmpdir, "dub.mp3")
        output_path = path.join(tmpdir, "out.mkv")

        for p in (mkv_input, mp4_input, audio_path):
            with open(p, "wb") as f:
                f.write(b"0")

        def _args(**overrides):
            base = dict(input=mkv_input, audio=audio_path, output=output_path,
                        language="en", track_name=None, default=False,
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

            missing_audio = path.join(tmpdir, "does_not_exist.mp3")
            rc = run(_args(audio=missing_audio))
            assert rc == 1, rc
            mock_find.assert_not_called()

        with patch(f"{__name__}._find_mkvmerge", return_value=None):
            rc = run(_args())
            assert rc == 1, rc  # reaches the mkvmerge stage, refuses because it's missing

    print("_self_test_run_gating: PASS")


def _run_self_tests():
    _self_test_language_resolution()
    _self_test_trim_cmd_construction()
    _self_test_run_gating()
    print("All audio_mux_cli self-tests PASSED")


def main(argv=None):
    # Special-cased ahead of the real parser (rather than added as a parser argument) so
    # it can run without also satisfying --input/--audio/--output=required -- see
    # docs/ai/TEST_MATRIX.md's "isolated (no-GPU) test pattern" convention.
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
