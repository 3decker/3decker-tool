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


def _trim_all_av(ffmpeg_bin, source_path, start_time, end_time, work_dir):
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
        proc = subprocess.run(copy_cmd, capture_output=True, text=True)
    except Exception as e:
        return None, f"ERROR: failed to run ffmpeg for audio/subtitle trim (stream copy): {e}", cmds_tried

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
        return None, f"ERROR: failed to run ffmpeg for audio/subtitle trim (re-encode fallback): {e}", cmds_tried

    if proc2.returncode == 0 and path.exists(trimmed_path) and path.getsize(trimmed_path) > 0:
        return trimmed_path, None, cmds_tried

    stderr_tail = (proc2.stderr or proc.stderr or "").strip().splitlines()[-10:]
    return None, (
        "ERROR: ffmpeg failed to trim --source's audio/subtitle tracks, both as a stream copy and "
        "as a re-encode fallback. Last ffmpeg stderr:\n" + "\n".join(stderr_tail)), cmds_tried


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

    print(f"[av-restore] --source has {audio_count} audio track(s) and {subtitle_count} "
          f"subtitle track(s) -- restoring all of them", file=sys.stderr)

    out_dir = path.dirname(path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output = path.splitext(output_path)[0] + ".avrestore_tmp" + path.splitext(output_path)[1]

    work_dir = None
    mux_source_path = source_path
    try:
        if args.source_start_time or args.source_end_time:
            ffmpeg_bin = _get_ffmpeg_bin()
            work_dir = tempfile.mkdtemp(prefix="iw3_avrestore_")
            print(f"[av-restore] trimming --source's audio/subtitle tracks to "
                  f"[{args.source_start_time or '0'}, {args.source_end_time or 'end'}) and "
                  f"shifting to start at 0...", file=sys.stderr)
            trimmed_path, trim_error, cmds_tried = _trim_all_av(
                ffmpeg_bin, source_path, args.source_start_time, args.source_end_time, work_dir)
            for cmd_str in cmds_tried:
                print(f"[av-restore] running: {cmd_str}", file=sys.stderr)
            if trim_error:
                print(trim_error, file=sys.stderr)
                return 1
            mux_source_path = trimmed_path
            print(f"[av-restore] trimmed audio/subtitles ready: {mux_source_path}", file=sys.stderr)

        # --no-audio --no-subtitles on input_path drops its own single audio track
        # (see audio_restore_cli.py's docstring -- it's already just a copy of the
        # source's own first track) and any subtitle tracks it somehow had (none,
        # in practice -- the main pipeline never adds any), keeping video/chapters
        # untouched. --no-video --no-chapters on the source keeps everything else
        # (audio + subtitles together). No --language/--track-name overrides --
        # every track's own metadata from --source is preserved as mkvmerge's
        # default "copy everything from this file" behavior for a file with no
        # options in front of it.
        cmd = [mkvmerge_bin, "-o", tmp_output, "--no-audio", "--no-subtitles", input_path,
               "--no-video", "--no-chapters", mux_source_path]
        print(f"[av-restore] running: {_format_cmd(cmd)}", file=sys.stderr)

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
        print(f"[av-restore] done -- wrote {output_path} with {audio_count} audio track(s) and "
              f"{subtitle_count} subtitle track(s) restored from source", file=sys.stderr)
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


def _run_self_tests():
    _self_test_trim_cmd_construction()
    _self_test_av_track_counting()
    _self_test_run_gating()
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
