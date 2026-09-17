"""python -m iw3.audio_restore_cli -- standalone tool to restore EVERY audio track
from an original source movie onto an already-converted 3D video, replacing whatever
single audio track iw3's own conversion kept.

Follows the same create_parser()/run()/main() shape as iw3.audio_mux_cli (ADR-042)
and iw3.subtitle_mux_cli (ADR-032) -- read those two modules' docstrings for the
established conventions reused here: MKV output only, --output is never --input,
temp-then-os.replace() safety, mkvmerge "copy every track from a file with no
options in front of it" behavior for preserving tracks untouched.

Real-world reason this tool exists (see docs/ai/AI_DECISIONS.md ADR-169): iw3's own
video pipeline (see nunif/utils/video/processor.py's export_audio()) always reads
only `input_container.streams.audio[0]` -- the FIRST audio track in the source --
and carries just that one into the converted output. A source with multiple
language tracks silently loses every track after the first. There is no in-pipeline
option to change this (by design -- iw3's job is the video conversion, not audio
track selection), so this is a separate post-processing step: take the converted
video's picture/subtitles as-is, and take EVERY audio track from the original
source instead of the converted file's own single one.

Unlike audio_mux_cli.py (which ADDS one new track and keeps the input's existing
audio track), this tool REPLACES the input's audio track(s) wholesale with the
source's full set -- the input's existing track is just a copy of the source's own
first track to begin with (per the export_audio() behavior above), so keeping it
alongside the source's full set would only produce a redundant duplicate.

No --language/--track-name options exist here (unlike audio_mux_cli.py) -- every
track's language and name metadata is read directly from --source and passed
through unchanged, which is the actual advantage of this tool over manually
re-adding tracks one at a time with Add Audio Track.

Trimming (see _trim_all_audio): reuses the exact --source-start-time/--source-end-time
convention and ffmpeg -ss/-to-as-absolute-input-position + -avoid_negative_ts
make_zero technique audio_mux_cli.py already established (see that module's own
"Trimming implementation notes"), extended to trim ALL of --source's audio tracks
in one ffmpeg pass (`-map 0:a`) instead of one. -c copy is tried first; if stream
copy fails for the whole set, this falls back to one re-encode pass (ffmpeg's own
default audio encoder for the output container) rather than trying each track
individually -- mirrors audio_mux_cli.py's own single-fallback-attempt shape.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from os import path

import av

from nunif.utils.video.metadata import parse_time
from .utils import _find_mkvmerge, _get_ffmpeg_bin


def _format_cmd(cmd):
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.audio_restore_cli",
        description=(
            "Restore every audio track from an original source movie onto an already-converted "
            "3D (SBS/TB) MKV video, replacing whichever single audio track the conversion kept. "
            "Video and subtitle tracks in --input are preserved untouched. Never modifies --input "
            "or --source -- always writes a new file at --output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                         help="Path to the already-converted 3D video. Must be a .mkv file -- "
                              "this tool does not convert containers. Read-only, never modified.")
    parser.add_argument("--source", "-s", type=str, required=True,
                         help="Path to the original source movie (any container ffmpeg/mkvmerge "
                              "already read directly -- e.g. the Blu-ray remux iw3 converted from). "
                              "Every audio track found in it is carried into --output, with each "
                              "track's own language/name metadata preserved as-is. Read-only, "
                              "never modified.")
    parser.add_argument("--output", "-o", type=str, required=True,
                         help="Path to write the new file to. Must be a different path from "
                              "--input -- this tool never overwrites the input.")
    parser.add_argument("--source-start-time", type=str, default=None,
                         help="Start time WITHIN --source to trim its audio tracks to (HH:MM:SS, "
                              "MM:SS, or seconds), for when --input is only a short clip of the "
                              "full --source movie (e.g. iw3 converted 12:16-15:16 of a 2-hour "
                              "film). The trimmed audio is also automatically shifted to start at "
                              "t=0 so it lines up with --input's first frame. Default: start of "
                              "--source (the whole file's audio is used as-is if both "
                              "--source-start-time and --source-end-time are omitted).")
    parser.add_argument("--source-end-time", type=str, default=None,
                         help="End time within --source to trim its audio tracks to. Default: end "
                              "of --source.")
    return parser


def _count_audio_tracks(source_path):
    """Existence + audio-track-count check for --source, same pattern
    nunif/utils/video/processor.py's export_audio() already uses
    (`len(input_container.streams.audio)`) to decide whether a source has any audio
    at all. Returns (count_or_None, error_message_or_None)."""
    if not path.exists(source_path):
        return None, f"ERROR: --source file does not exist: {source_path}"
    try:
        container = av.open(source_path, mode="r", metadata_errors="ignore")
        try:
            count = len(container.streams.audio)
        finally:
            container.close()
    except Exception as e:
        return None, f"ERROR: failed to open --source to inspect its audio tracks: {e}"
    if count == 0:
        return 0, f"ERROR: --source has no audio tracks to restore: {source_path}"
    return count, None


def _trim_all_audio(ffmpeg_bin, source_path, start_time, end_time, work_dir):
    """Extracts every audio track from source_path over [start_time, end_time) into a
    new temp .mkv in work_dir, ALSO shifting the result to start at t=0 (see module
    docstring's "Trimming" section) -- same technique as audio_mux_cli.py's
    _trim_audio_source, extended from one track (`-map 0:a:0`) to all of them
    (`-map 0:a`). Tries a fast lossless -c copy first, falling back to one re-encode
    (ffmpeg's own default encoder) only if stream copy fails outright.

    Returns (trimmed_path_or_None, error_message_or_None, cmds_tried (list of str)).
    """
    trimmed_path = path.join(work_dir, "trimmed_audio.mkv")

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
        cmd += ["-i", str(source_path), "-vn", "-sn", "-map", "0:a"]
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
        "ERROR: ffmpeg failed to trim --source's audio tracks, both as a stream copy and as a "
        "re-encode fallback. Last ffmpeg stderr:\n" + "\n".join(stderr_tail)), cmds_tried


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

    track_count, source_error = _count_audio_tracks(source_path)
    if source_error:
        print(source_error, file=sys.stderr)
        return 1

    mkvmerge_bin = _find_mkvmerge()
    if not mkvmerge_bin:
        print("ERROR: mkvmerge (MKVToolNix) was not found -- cannot mux. Expected it bundled "
              "alongside ffmpeg, or on PATH.", file=sys.stderr)
        return 1

    print(f"[audio-restore] --source has {track_count} audio track(s) -- restoring all of them",
          file=sys.stderr)

    out_dir = path.dirname(path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output = path.splitext(output_path)[0] + ".audiorestore_tmp" + path.splitext(output_path)[1]

    work_dir = None
    mux_source_path = source_path
    try:
        if args.source_start_time or args.source_end_time:
            ffmpeg_bin = _get_ffmpeg_bin()
            work_dir = tempfile.mkdtemp(prefix="iw3_audiorestore_")
            print(f"[audio-restore] trimming --source's audio tracks to "
                  f"[{args.source_start_time or '0'}, {args.source_end_time or 'end'}) and "
                  f"shifting to start at 0...", file=sys.stderr)
            trimmed_path, trim_error, cmds_tried = _trim_all_audio(
                ffmpeg_bin, source_path, args.source_start_time, args.source_end_time, work_dir)
            for cmd_str in cmds_tried:
                print(f"[audio-restore] running: {cmd_str}", file=sys.stderr)
            if trim_error:
                print(trim_error, file=sys.stderr)
                return 1
            mux_source_path = trimmed_path
            print(f"[audio-restore] trimmed audio ready: {mux_source_path}", file=sys.stderr)

        # --no-audio on input_path drops its own single (already-just-a-copy-of-source)
        # audio track, keeping video/subtitles/chapters untouched. --no-video
        # --no-subtitles --no-chapters on the source restricts it to audio tracks only
        # -- for the untrimmed case source_path may still be the original movie file
        # (with its own video/subtitles/chapters we do NOT want); the trimmed
        # intermediate is already audio-only from ffmpeg's -vn -sn, but the flags are
        # harmless/defensive there too. No --language/--track-name overrides -- every
        # track's own metadata from --source is preserved as mkvmerge's default
        # "copy everything from this file" behavior for a file with no options in
        # front of it (same convention as audio_mux_cli.py/subtitle_mux_cli.py).
        cmd = [mkvmerge_bin, "-o", tmp_output, "--no-audio", input_path,
               "--no-video", "--no-subtitles", "--no-chapters", mux_source_path]
        print(f"[audio-restore] running: {_format_cmd(cmd)}", file=sys.stderr)

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
        print(f"[audio-restore] done -- wrote {output_path} with {track_count} audio track(s) "
              f"restored from source", file=sys.stderr)
        return 0
    finally:
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _self_test_trim_cmd_construction():
    """Mocked test of _trim_all_audio's ffmpeg command construction -- no real
    ffmpeg/network/GPU needed. Mirrors audio_mux_cli.py's own
    _self_test_trim_cmd_construction, extended to check `-map 0:a` (all tracks) is
    used instead of `-map 0:a:0` (one track)."""
    from unittest.mock import patch, MagicMock

    with patch.object(subprocess, "run") as mock_run, \
         patch("os.path.getsize", return_value=1234):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch("os.path.exists", return_value=True):
            trimmed, error, cmds = _trim_all_audio(
                "ffmpeg", "movie.mkv", "00:01:15", "00:06:15", "/tmp/work")
        assert error is None, error
        assert trimmed.endswith(".mkv"), trimmed
        assert len(cmds) == 1, cmds
        copy_cmd = mock_run.call_args_list[0][0][0]
        assert "-ss" in copy_cmd and "75.0" in copy_cmd, copy_cmd
        assert "-to" in copy_cmd and "375.0" in copy_cmd, copy_cmd
        assert "-map" in copy_cmd and "0:a" in copy_cmd, copy_cmd
        assert "0:a:0" not in copy_cmd, copy_cmd  # all tracks, not just the first
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
            trimmed, error, cmds = _trim_all_audio(
                "ffmpeg", "movie.mkv", "10", "20", "/tmp/work")
        assert error is None, error
        assert len(cmds) == 2, cmds
        reencode_cmd = mock_run.call_args_list[1][0][0]
        assert "-c" not in reencode_cmd, reencode_cmd

    # Both attempts fail -> clear error
    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="nope")
        with patch("os.path.exists", return_value=False):
            trimmed, error, cmds = _trim_all_audio(
                "ffmpeg", "movie.mkv", "10", "20", "/tmp/work")
        assert trimmed is None
        assert error is not None and "ffmpeg failed to trim" in error

    # end_time <= start_time caught before ever invoking ffmpeg
    trimmed, error, cmds = _trim_all_audio("ffmpeg", "movie.mkv", "20", "10", "/tmp/work")
    assert trimmed is None and error is not None and cmds == []

    print("_self_test_trim_cmd_construction: PASS")


def _self_test_audio_track_counting():
    """Mocked test of _count_audio_tracks -- missing file, zero-audio source, and a
    real count are each distinguished, no real file/av decode needed."""
    from unittest.mock import patch, MagicMock

    count, error = _count_audio_tracks("/does/not/exist.mkv")
    assert count is None and error is not None and "does not exist" in error

    with patch("os.path.exists", return_value=True), patch.object(av, "open") as mock_open:
        mock_container = MagicMock()
        mock_container.streams.audio = []
        mock_open.return_value = mock_container
        count, error = _count_audio_tracks("silent.mkv")
        assert count == 0 and error is not None and "no audio tracks" in error

    with patch("os.path.exists", return_value=True), patch.object(av, "open") as mock_open:
        mock_container = MagicMock()
        mock_container.streams.audio = [MagicMock(), MagicMock(), MagicMock()]
        mock_open.return_value = mock_container
        count, error = _count_audio_tracks("multilang.mkv")
        assert count == 3 and error is None, (count, error)

    print("_self_test_audio_track_counting: PASS")


def _self_test_run_gating():
    """Synthetic/mocked test of run()'s pre-mkvmerge gates (non-mkv input, output==input,
    source with no audio tracks) -- proves each refuses before mkvmerge/ffmpeg is ever
    invoked, same convention as audio_mux_cli.py's _self_test_run_gating()."""
    import tempfile as _tempfile
    from unittest.mock import patch, MagicMock

    with _tempfile.TemporaryDirectory(prefix="iw3_audiorestore_selftest_") as tmpdir:
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

        with patch(f"{__name__}._count_audio_tracks", return_value=(0, "ERROR: no audio")), \
             patch(f"{__name__}._find_mkvmerge") as mock_find:
            rc = run(_args())
            assert rc == 1, rc
            mock_find.assert_not_called()

        with patch(f"{__name__}._count_audio_tracks", return_value=(2, None)), \
             patch(f"{__name__}._find_mkvmerge", return_value=None):
            rc = run(_args())
            assert rc == 1, rc  # reaches the mkvmerge stage, refuses because it's missing

    print("_self_test_run_gating: PASS")


def _run_self_tests():
    _self_test_trim_cmd_construction()
    _self_test_audio_track_counting()
    _self_test_run_gating()
    print("All audio_restore_cli self-tests PASSED")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
