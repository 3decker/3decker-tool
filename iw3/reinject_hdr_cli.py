"""python -m iw3.reinject_hdr_cli -- standalone retroactive Dolby Vision RPU /
HDR10+ reinjection tool.

Real-world use case (see docs/ai/AI_DECISIONS.md ADR-031): a user already ran an
iw3 conversion WITHOUT "Preserve Dolby Vision" turned on, ending up with an SDR
3D output, but still has the ORIGINAL Dolby Vision / HDR10+ source file and wants
to retroactively apply that grading to the already-converted output -- without
re-running the expensive depth/stereo conversion a second time.

This is a genuinely standalone tool, invoked as its OWN subprocess (same
"separate subprocess, never sharing state with the main app" convention as
iw3.rife_cli / waifu2x.cli -- see docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001),
even though unlike those two it needs no GPU at all: every step here is ffmpeg /
dovi_tool / hdr10plus_tool subprocess orchestration reusing the exact same
extraction/injection/remux functions Dual-Pass Depth Blend uses internally
(iw3.depth_blend._extract_hdr_rpu_files, iw3.utils._inject_hdr_rpu,
iw3.utils._remux_injected_hevc) -- see docs/ai/domains/DOLBY_VISION.md.

Design decision (explicitly confirmed, do not "improve" without re-confirming):
the user must explicitly pass --start-time/--end-time to say which segment of
--source matches --converted. There is deliberately NO auto-detection of the
matching segment -- default is "the whole source" when omitted, nothing fancier.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from os import path

from nunif.utils.video.metadata import parse_time
from .utils import (
    _get_ffmpeg_bin, _find_ffprobe, _find_dovi_tool, _find_hdr10plus_tool,
    _inject_hdr_rpu, _remux_injected_hevc,
)
from .depth_blend import _extract_hdr_rpu_files


# Matches iw3.utils._run_rife_interpolation's own output naming
# (f"{base}_rife{ext}", optionally with a model-tier suffix appended directly
# after "_rife", e.g. "_rife_lite") -- kept as a nicer, clearer error message
# when available, NOT the only protection (see module docstring / ADR-031):
# the frame-count mismatch check below is the actual authoritative gate, since
# this filename pattern is a convention, not something structurally guaranteed.
_RIFE_FILENAME_RE = re.compile(r"_rife(_[A-Za-z0-9]+)?$")


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.reinject_hdr_cli",
        description=(
            "Retroactively inject Dolby Vision RPU / HDR10+ metadata extracted from an ORIGINAL "
            "source file into a copy of an ALREADY-CONVERTED iw3 3D output, without re-running "
            "depth/stereo conversion. Never modifies --source or --converted -- always writes a "
            "new file at --output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True,
                         help="Path to the ORIGINAL source file that has real Dolby Vision / HDR10+ "
                              "metadata (the file iw3 originally converted FROM).")
    parser.add_argument("--converted", type=str, required=True,
                         help="Path to the already-converted (currently SDR) 3D output made from "
                              "--source. Read-only -- never modified.")
    parser.add_argument("--output", type=str, required=True,
                         help="Path to write the new HDR-reinjected copy to. Must not be the same "
                              "path as --source or --converted.")
    parser.add_argument("--start-time", type=str, default=None,
                         help="Start time within --source matching the start of --converted "
                              "(HH:MM:SS, MM:SS, or seconds). Default: start of --source (whole "
                              "source used if both --start-time and --end-time are omitted). There "
                              "is deliberately no auto-detection of this -- you must know and supply "
                              "the exact range that was actually converted.")
    parser.add_argument("--end-time", type=str, default=None,
                         help="End time within --source matching the end of --converted. Default: "
                              "end of --source.")
    parser.add_argument("--frame-count-tolerance", type=int, default=0,
                         help="DELIBERATE ESCAPE HATCH -- NOT a normal setting. Number of frames of "
                              "mismatch to allow between --source's decoded frame count (trimmed to "
                              "--start-time/--end-time) and --converted's decoded frame count before "
                              "refusing to inject. Default 0 requires an EXACT match, which is what "
                              "you want almost all of the time. Only raise this above 0 after you "
                              "have already confirmed by other means that the mismatch is a single "
                              "known dropped/duplicated boundary frame -- it does NOT fix a genuinely "
                              "wrong --start-time/--end-time range, it just allows the tool to inject "
                              "anyway despite one. If you find yourself needing more than 1-2, your "
                              "--start-time/--end-time is almost certainly wrong -- fix that instead.")
    return parser


def _format_cmd(cmd):
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


def _probe_frames_and_duration(input_path, ffprobe_bin, start_time=None, end_time=None, timeout=3600):
    """Decoded (not estimated) frame count + best-effort duration for input_path, trimmed
    to [start_time, end_time] when given (input-side -ss/-to, same absolute-range semantics
    as iw3.depth_blend._extract_hdr_rpu_files's own trim args -- see DOLBY_VISION.md).

    Uses ffprobe -count_frames rather than trusting nb_frames, which is often absent or
    only estimated from container metadata (see docs/ai/domains/DOLBY_VISION.md: exact
    frame-range matching over approximate timestamp math is this project's established
    verification method for DV/HDR reinjection).

    Returns (frame_count_or_None, duration_seconds_or_None, cmd_list).
    """
    trim_args = []
    if start_time:
        trim_args += ["-ss", str(parse_time(start_time))]
    if end_time:
        trim_args += ["-to", str(parse_time(end_time))]

    cmd = [ffprobe_bin, "-v", "error", *trim_args,
           "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=nb_read_frames,r_frame_rate:format=duration",
           "-of", "json", str(input_path)]

    frame_count = None
    full_duration = None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        data = json.loads(proc.stdout) if proc.stdout else {}
        streams = data.get("streams", [])
        if streams:
            try:
                frame_count = int(streams[0].get("nb_read_frames"))
            except (TypeError, ValueError):
                frame_count = None
        try:
            full_duration = float(data.get("format", {}).get("duration"))
        except (TypeError, ValueError):
            full_duration = None
    except Exception:
        pass

    # The container's format.duration field reflects the WHOLE file's metadata duration
    # even when -ss/-to are passed as input-side trim options (they change which packets
    # get decoded, not that reported metadata field) -- so for a trimmed probe, prefer the
    # explicit requested range's own arithmetic as the duration we report, purely as a
    # human-readable corroborating signal (frame count above is the actual gate).
    if start_time or end_time:
        start_sec = parse_time(start_time) if start_time else 0.0
        end_sec = parse_time(end_time) if end_time else full_duration
        duration = (end_sec - start_sec) if (end_sec is not None) else full_duration
    else:
        duration = full_duration

    return frame_count, duration, cmd


def _check_rife_guard(converted_path, ffprobe_bin):
    """Returns a list of human-readable reasons this file looks like RIFE'd output, or an
    empty list if neither signal fires. NOTE (see module docstring): the comment tag is
    only reliably present on RIFE's INPUT file, not its own '_rife'-suffixed output, so
    this is a nicer/clearer error message when available -- not the only protection. The
    frame-count mismatch check in run() is the real, authoritative gate either way."""
    reasons = []
    basename_noext = path.splitext(path.basename(str(converted_path)))[0]
    if _RIFE_FILENAME_RE.search(basename_noext):
        reasons.append(
            f"filename '{path.basename(str(converted_path))}' matches the '_rife' suffix pattern "
            f"iw3's RIFE post-processing step uses for its output files")
    try:
        proc = subprocess.run(
            [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_format", str(converted_path)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(proc.stdout) if proc.stdout else {}
        tags = data.get("format", {}).get("tags", {}) or {}
        comment = ""
        for key in ("comment", "Comment", "COMMENT"):
            if key in tags:
                comment = tags[key]
                break
        if "iw3_rife_interpolate=1" in comment:
            reasons.append("container comment metadata contains 'iw3_rife_interpolate=1'")
    except Exception:
        pass
    return reasons


def run(args):
    ffmpeg_bin = _get_ffmpeg_bin()
    ffprobe_bin = _find_ffprobe()

    source = str(args.source)
    converted = str(args.converted)
    output = str(args.output)

    for label, p in (("--source", source), ("--converted", converted)):
        if not path.exists(p):
            print(f"ERROR: {label} file does not exist: {p}", file=sys.stderr)
            return 1

    if path.abspath(output) in (path.abspath(source), path.abspath(converted)):
        print("ERROR: --output must be a different path from both --source and --converted -- "
              "this tool never overwrites either input.", file=sys.stderr)
        return 1

    # --- RIFE guard: fast heuristic check, run before the slow frame-count probe below ---
    rife_reasons = _check_rife_guard(converted, ffprobe_bin)
    if rife_reasons:
        print("ERROR: This file appears to have been processed with RIFE frame interpolation, "
              "which changes frame count -- HDR reinjection cannot work on RIFE'd output.",
              file=sys.stderr)
        for reason in rife_reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 1

    # --- required pre-flight: exact (by default) decoded frame-count match ---
    print("[reinject-hdr] probing source (trimmed) and converted decoded frame counts -- this "
          "decodes the full range so it may take a while on long clips...", file=sys.stderr)
    src_frames, src_duration, src_cmd = _probe_frames_and_duration(
        source, ffprobe_bin, args.start_time, args.end_time)
    conv_frames, conv_duration, conv_cmd = _probe_frames_and_duration(
        converted, ffprobe_bin, None, None)

    print(f"[reinject-hdr] source ffprobe command:    {_format_cmd(src_cmd)}", file=sys.stderr)
    print(f"[reinject-hdr] converted ffprobe command: {_format_cmd(conv_cmd)}", file=sys.stderr)
    print(f"[reinject-hdr] source (trimmed) decoded frame count: {src_frames}  duration: {src_duration}",
          file=sys.stderr)
    print(f"[reinject-hdr] converted decoded frame count:        {conv_frames}  duration: {conv_duration}",
          file=sys.stderr)

    if src_frames is None or conv_frames is None:
        print("ERROR: could not determine a decoded frame count for one or both files via ffprobe "
              "-- refusing to proceed without a reliable frame-count comparison. Re-run the exact "
              "ffprobe command(s) printed above by hand to see why.", file=sys.stderr)
        return 1

    tolerance = max(0, int(args.frame_count_tolerance))
    mismatch = abs(src_frames - conv_frames)
    if mismatch > tolerance:
        print(
            "ERROR: frame count mismatch between --source (trimmed to --start-time/--end-time) and "
            "--converted exceeds the allowed tolerance -- refusing to inject.\n"
            f"  source (trimmed) frames: {src_frames}   duration: {src_duration}\n"
            f"  converted frames:        {conv_frames}   duration: {conv_duration}\n"
            f"  mismatch: {mismatch} frame(s)   tolerance: {tolerance}\n"
            "This almost always means --start-time/--end-time doesn't exactly match the range that "
            "was actually converted into --converted -- double check the exact range and try again. "
            "If you have independently confirmed this specific mismatch is a single known dropped/"
            "duplicated boundary frame (NOT a wrong range), you can override this with "
            "--frame-count-tolerance -- that is a deliberate escape hatch, not a normal setting.",
            file=sys.stderr)
        return 1

    print(f"[reinject-hdr] frame counts match within tolerance ({mismatch} <= {tolerance}). Proceeding.",
          file=sys.stderr)

    # --- copy --converted to a working temp file BEFORE touching anything, so a failure
    # partway through injection can never corrupt --converted or leave a half-written
    # file at --output (see docs/ai/CODING_STANDARDS.md CS-IO-001: temp-then-replace). ---
    out_dir = path.dirname(path.abspath(output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output = path.splitext(output)[0] + ".hdr_inject_tmp" + path.splitext(output)[1]
    print(f"[reinject-hdr] copying --converted to a working copy: {tmp_output}", file=sys.stderr)
    shutil.copyfile(converted, tmp_output)

    work_dir = tempfile.mkdtemp(prefix="iw3_reinject_hdr_")
    hdr_rpu_path = path.join(work_dir, "dv_rpu.bin")
    hdr_h10p_path = path.join(work_dir, "hdr10plus.json")
    try:
        # Minimal shim -- _extract_hdr_rpu_files only ever reads args.start_time /
        # args.end_time (verified by reading its source; it does not touch args.state),
        # but a state attribute is still provided for forward-compatibility should that
        # change.
        shim_args = argparse.Namespace(start_time=args.start_time, end_time=args.end_time, state=None)
        print("[reinject-hdr] extracting Dolby Vision / HDR10+ metadata from --source...",
              file=sys.stderr)
        dv_ok, h10p_ok, failures = _extract_hdr_rpu_files(
            source, work_dir, shim_args, hdr_rpu_path, hdr_h10p_path)
        for failure in failures:
            print(f"[reinject-hdr] WARNING: {failure}", file=sys.stderr)

        have_rpu = path.exists(hdr_rpu_path)
        have_h10p = path.exists(hdr_h10p_path)
        if not have_rpu and not have_h10p:
            print("ERROR: no Dolby Vision RPU or HDR10+ metadata was extracted from --source -- "
                  "nothing to inject. Check that --source genuinely has DV/HDR10+ metadata and that "
                  "--start-time/--end-time are correct.", file=sys.stderr)
            return 1

        dovi_bin = _find_dovi_tool() if have_rpu else None
        hdr10plus_bin = _find_hdr10plus_tool() if have_h10p else None

        print("[reinject-hdr] injecting into a copy of --converted...", file=sys.stderr)
        injected_hevc, fps_str, reason = _inject_hdr_rpu(
            tmp_output,
            hdr_rpu_path if have_rpu else None,
            hdr_h10p_path if have_h10p else None,
            ffmpeg_bin, dovi_bin, hdr10plus_bin, work_dir,
            configured_video_codec=None,  # unknown provenance -- force the safe ffprobe-verified path
        )
        if injected_hevc is None:
            print(f"ERROR: HDR injection failed: {reason}", file=sys.stderr)
            return 1

        print("[reinject-hdr] remuxing injected stream back into the container...", file=sys.stderr)
        _remux_injected_hevc(tmp_output, injected_hevc, fps_str, ffmpeg_bin, work_dir)

        os.replace(tmp_output, output)
        print(f"[reinject-hdr] done -- wrote {output}", file=sys.stderr)
        return 0
    finally:
        if path.exists(tmp_output):
            try:
                os.remove(tmp_output)
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)


def _self_test_probe_frames_and_duration():
    """Synthetic/mocked test (no real ffmpeg/GPU -- see docs/ai/CODING_STANDARDS.md
    CS-TEST-001) of _probe_frames_and_duration's frame-count parsing and its
    trimmed-duration arithmetic (container format.duration must NOT be trusted as the
    trimmed-range duration -- see the comment in that function)."""
    from unittest.mock import patch, MagicMock

    def _stdout(frame_count, duration):
        return json.dumps({
            "streams": [{"nb_read_frames": str(frame_count), "r_frame_rate": "24/1"}],
            "format": {"duration": str(duration)},
        })

    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(stdout=_stdout(240, 10.0), returncode=0)
        frames, duration, cmd = _probe_frames_and_duration("fake.mkv", "ffprobe")
        assert frames == 240, frames
        assert duration == 10.0, duration
        assert "-count_frames" in cmd and "-ss" not in cmd

    with patch.object(subprocess, "run") as mock_run:
        # Container metadata says 100s (the WHOLE file) even though we asked for a 10s
        # trimmed window -- the function must report the requested range (20-10=10),
        # not the untrimmed container duration.
        mock_run.return_value = MagicMock(stdout=_stdout(240, 100.0), returncode=0)
        frames, duration, cmd = _probe_frames_and_duration(
            "fake.mkv", "ffprobe", start_time="10", end_time="20")
        assert frames == 240, frames
        assert duration == 10.0, duration
        assert "-ss" in cmd and "-to" in cmd

    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(stdout="not json", returncode=1)
        frames, duration, cmd = _probe_frames_and_duration("fake.mkv", "ffprobe")
        assert frames is None, frames

    print("_self_test_probe_frames_and_duration: PASS")


def _self_test_rife_guard():
    """Synthetic/mocked test of the RIFE guard's two independent signals (filename
    suffix, container comment tag) -- see module docstring for why neither alone is
    fully authoritative and the frame-count gate in run() backstops both."""
    from unittest.mock import patch, MagicMock

    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"format": {"tags": {"comment": "iw3_depth_model=x"}}}), returncode=0)
        assert _check_rife_guard("movie_3d.mkv", "ffprobe") == []

    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(stdout=json.dumps({"format": {"tags": {}}}), returncode=0)
        reasons = _check_rife_guard("movie_3d_rife.mkv", "ffprobe")
        assert any("_rife" in r for r in reasons), reasons

    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"format": {"tags":
                {"comment": "iw3_rife_interpolate=1 iw3_rife_model=rife_425"}}}),
            returncode=0)
        # Filename has no '_rife' suffix here -- this simulates RIFE's INPUT file, the
        # one that actually carries the comment tag (see module docstring).
        reasons = _check_rife_guard("movie_3d.mkv", "ffprobe")
        assert any("comment" in r for r in reasons), reasons

    print("_self_test_rife_guard: PASS")


def _self_test_run_preflight_gating():
    """Synthetic/mocked test of run()'s hard-gate ORDER: RIFE guard -> frame-count
    comparison -> only then extraction/injection. Uses a real tempdir with placeholder
    source/converted files (their content is never read -- every probing/extraction/
    injection call is mocked) so path.exists() checks behave realistically."""
    import argparse as _argparse
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_reinject_selftest_") as tmpdir:
        source = path.join(tmpdir, "source.mkv")
        converted = path.join(tmpdir, "converted.mkv")
        output = path.join(tmpdir, "output.mkv")
        for p in (source, converted):
            with open(p, "wb") as f:
                f.write(b"0")

        def _args(tolerance=0, out=None):
            return _argparse.Namespace(
                source=source, converted=converted, output=out or output,
                start_time=None, end_time=None, frame_count_tolerance=tolerance)

        # 1) frame-count mismatch beyond tolerance -> refuse BEFORE touching dovi_tool
        with patch(f"{__name__}._check_rife_guard", return_value=[]), \
             patch(f"{__name__}._probe_frames_and_duration") as mock_probe, \
             patch(f"{__name__}._extract_hdr_rpu_files") as mock_extract:
            mock_probe.side_effect = [(241, 10.04, ["ffprobe"]), (240, 10.0, ["ffprobe"])]
            rc = run(_args())
            assert rc == 1, rc
            mock_extract.assert_not_called()

        # 2) RIFE guard fires -> refuse immediately, never even reaches the frame probe
        with patch(f"{__name__}._check_rife_guard", return_value=["container comment metadata contains 'iw3_rife_interpolate=1'"]), \
             patch(f"{__name__}._probe_frames_and_duration") as mock_probe:
            rc = run(_args())
            assert rc == 1, rc
            mock_probe.assert_not_called()

        # 3) output path collides with an input -> refuse
        rc = run(_args(out=converted))
        assert rc == 1, rc

        # 4) within tolerance and extraction produces an RPU file -> gate passes and
        # extraction really is reached with the right source/work_dir args (injection
        # itself is stubbed to decline, just to prove we got past the gate cleanly)
        with patch(f"{__name__}._check_rife_guard", return_value=[]), \
             patch(f"{__name__}._probe_frames_and_duration") as mock_probe, \
             patch(f"{__name__}._extract_hdr_rpu_files") as mock_extract, \
             patch(f"{__name__}._find_dovi_tool", return_value=None), \
             patch(f"{__name__}._find_hdr10plus_tool", return_value=None), \
             patch(f"{__name__}._inject_hdr_rpu", return_value=(None, None, "stubbed decline")):
            mock_probe.side_effect = [(241, 10.04, ["ffprobe"]), (240, 10.0, ["ffprobe"])]

            def _fake_extract(input_path, work_dir, shim_args, rpu_path, h10p_path):
                assert input_path == source
                assert shim_args.start_time is None and shim_args.end_time is None
                with open(rpu_path, "wb") as f:
                    f.write(b"rpu")
                return True, True, []
            mock_extract.side_effect = _fake_extract

            rc = run(_args(tolerance=1))
            mock_extract.assert_called_once()
            assert rc == 1, rc  # injection was stubbed to decline -- proves gate passed, nothing more

        # 5) nothing extracted at all (source genuinely has no DV/HDR10+) -> refuse
        with patch(f"{__name__}._check_rife_guard", return_value=[]), \
             patch(f"{__name__}._probe_frames_and_duration") as mock_probe, \
             patch(f"{__name__}._extract_hdr_rpu_files", return_value=(True, True, [])):
            mock_probe.side_effect = [(240, 10.0, ["ffprobe"]), (240, 10.0, ["ffprobe"])]
            rc = run(_args())
            assert rc == 1, rc

    print("_self_test_run_preflight_gating: PASS")


def _run_self_tests():
    _self_test_probe_frames_and_duration()
    _self_test_rife_guard()
    _self_test_run_preflight_gating()
    print("All reinject_hdr_cli self-tests PASSED")


def main(argv=None):
    # Special-cased ahead of the real parser (rather than added as a parser argument)
    # so it can run without also satisfying --source/--converted/--output=required --
    # see docs/ai/TEST_MATRIX.md's "isolated (no-GPU) test pattern" convention.
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
