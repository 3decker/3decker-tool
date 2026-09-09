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

# Real end-to-end testing against this project's own DV test source (2026-09-08,
# see docs/ai/AI_DECISIONS.md ADR-051/ADR-064 amendments) found that a genuinely
# correct --start-time/--end-time can still make the RPU extraction step (ffmpeg
# `-ss`/`-to` in `-c:v copy` mode, keyframe/timestamp-approximate) produce MORE
# real frames than the RIFE manifest's own recorded source_frame_count -- the
# EXACT number the real conversion pipeline actually rendered for that range
# (the authoritative value, per DOLBY_VISION.md's own "exact frame count over
# approximate timestamp math" invariant). A real run measured a 31-38 frame gap
# (~1.3-1.6s at ~24fps) from this cause alone -- not a wrong time range. This is
# the sanity cap on how much excess is auto-corrected (see _expand_rpu_for_rife
# and the source-side check in _run_with_rife_manifest below) before refusing
# outright as a likely genuinely-wrong --start-time/--end-time instead of
# routine trim slop -- generous (~5s at 24fps) relative to the real observed
# gap, not a tight fit to it.
_SOURCE_TRIM_SLOP_MAX_FRAMES = 120


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
    parser.add_argument("--rife-manifest", type=str, default=None,
                         help="Path to the '<rife_output>.rife_manifest.json' sidecar iw3.rife_cli "
                              "writes next to its own output every time it runs (see "
                              "docs/ai/AI_DECISIONS.md ADR-051). When given, this SWITCHES the tool "
                              "from the normal exact-frame-count gate to a distinct RIFE-aware mode: "
                              "--converted is expected to be RIFE's interpolated output (a LARGER "
                              "frame count than --source's, by design, not a mismatch), and this tool "
                              "expands --source's Dolby Vision RPU to match RIFE's exact output frame "
                              "count -- duplicating each synthetic frame's nearest real neighbor's "
                              "actual RPU entry, never blending across a scene cut -- before "
                              "injecting. Workflow: (1) convert normally WITHOUT --preserve-dowi (RIFE "
                              "and --preserve-dowi remain mutually exclusive within a single "
                              "conversion run), (2) run RIFE on that output "
                              "(python -m iw3.rife_cli), which writes '<output>_rife<ext>' plus "
                              "'<output>_rife<ext>.rife_manifest.json' alongside it, (3) run this tool "
                              "with --converted pointing at the RIFE output and --rife-manifest "
                              "pointing at its manifest. HDR10+ dynamic metadata is NOT expanded in "
                              "this mode (Dolby Vision RPU only, currently) -- a warning is printed "
                              "and HDR10+ is skipped rather than injected mismatched. This does NOT "
                              "loosen the normal (no --rife-manifest) exact-frame-count gate at all -- "
                              "that path is completely unaffected by this flag's existence.")
    return parser


def _format_cmd(cmd):
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


def _read_intervals_for_range(start_time, end_time):
    """Builds an ffprobe `-read_intervals` spec for [start_time, end_time] -- ffprobe's
    own native mechanism for restricting which packets/frames get read, used here INSTEAD
    OF ffmpeg-style `-ss`/`-to` (see the real, confirmed bug this replaces, below). Syntax
    (ffprobe's own, confirmed directly against this project's bundled binary -- not
    ffmpeg's -ss/-to at all): one interval as `[START][%[END|+DURATION]]` -- "START%+DUR"
    for an absolute start plus a duration, "START%" for start-to-EOF, "%END" for
    beginning-to-END. Returns None when neither start_time nor end_time is given (no
    interval restriction needed).

    Real, confirmed bug this fixes (2026-09-08, see docs/ai/AI_DECISIONS.md ADR-051/
    ADR-064 amendments): this project's bundled ffprobe.exe genuinely does not implement
    `-ss`/`-to` at all -- running it directly, several ways, always failed identically
    with "Failed to set value '75.0' for option 'ss': Option not found", and `-ss` is
    absent from `ffprobe -h full`'s own option listing on this exact binary. This is not
    a mistake in how the old code built its command (the flags were placed correctly) --
    the installed ffprobe binary itself lacks the option, while the bundled ffmpeg.exe
    (used elsewhere for the real HEVC/RPU extraction, e.g.
    iw3.depth_blend._extract_hdr_rpu_files) supports `-ss`/`-to` fine, which is why that
    extraction step was never affected by this bug -- only this ffprobe-based probe was.
    `-read_intervals` was confirmed working on the same real bundled ffprobe.exe binary,
    all three forms, against real 4K Dolby Vision source footage, before being adopted
    here."""
    if start_time and end_time:
        start_sec = parse_time(start_time)
        end_sec = parse_time(end_time)
        return f"{start_sec}%+{end_sec - start_sec}"
    elif start_time:
        return f"{parse_time(start_time)}%"
    elif end_time:
        return f"%{parse_time(end_time)}"
    return None


def _probe_frames_and_duration(input_path, ffprobe_bin, start_time=None, end_time=None, timeout=3600):
    """Decoded (not estimated) frame count + best-effort duration for input_path, trimmed
    to [start_time, end_time] when given (via ffprobe's own `-read_intervals` -- see
    _read_intervals_for_range's docstring for why this is used instead of `-ss`/`-to`,
    which this project's bundled ffprobe.exe does not support at all).

    Uses ffprobe -count_frames rather than trusting nb_frames, which is often absent or
    only estimated from container metadata (see docs/ai/domains/DOLBY_VISION.md: exact
    frame-range matching over approximate timestamp math is this project's established
    verification method for DV/HDR reinjection).

    Returns (frame_count_or_None, duration_seconds_or_None, cmd_list).
    """
    trim_args = []
    read_intervals = _read_intervals_for_range(start_time, end_time)
    if read_intervals:
        trim_args = ["-read_intervals", read_intervals]

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
    # even when -read_intervals restricts which packets get decoded (it doesn't change
    # that reported metadata field) -- so for a trimmed probe, prefer the explicit
    # requested range's own arithmetic as the duration we report, purely as a
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
    """Dispatches to the RIFE-manifest-aware path (ADR-051) when --rife-manifest
    is given, or the original strict exact-frame-count path (ADR-031)
    otherwise. The strict path (_run_strict, below -- the exact, unmodified
    body of this function before ADR-051) is completely untouched by this
    dispatch: --rife-manifest defaulting to None/absent means getattr() below
    always falls through to it for every pre-existing caller."""
    if getattr(args, "rife_manifest", None):
        return _run_with_rife_manifest(args)
    return _run_strict(args)


def _build_duplicate_ops_from_manifest(manifest):
    """Walks a RIFE frame manifest's "frames" list (see iw3.rife_cli / ADR-051)
    and returns (duplicate_ops, source_frame_count):

    - duplicate_ops: a list of {"source": int, "offset": int, "length": int}
      dicts in dovi_tool's own `editor --json` config "duplicate" operation
      format, expressed entirely in ORIGINAL (pre-expansion) source-RPU frame
      indices. This is correct regardless of submission order or count --
      verified directly against dovi_tool's real Rust source
      (quietvoid/dovi_tool src/dovi/editor.rs, fetched 2026-09-08): duplicate
      ops are applied via Vec::splice in DESCENDING offset order (dovi_tool
      sorts them itself before applying), so a higher-offset insertion is
      always applied before a lower-offset one and can never shift the
      original-index positions a lower-offset/source op still needs to
      reference. For two ops sharing the SAME offset (the non-2x-multiplier
      case, e.g. 3x, where a pair's two synthetic frames duplicate from
      DIFFERENT real neighbors), the same descending-sort-then-splice
      mechanics preserve the ops' ORIGINAL SUBMISSION ORDER as their final
      visual order -- so this function emits them left-to-right (nearest-first)
      exactly as they should appear. Both of these were independently
      confirmed with a REAL end-to-end dovi_tool run against real Dolby Vision
      RPU data (see ADR-051's verification section), not just reasoned about.
      Consecutive synthetic frames sharing the same nearest-real source are
      merged into a single op (length > 1) purely to keep the JSON small --
      dovi_tool produces an identical result from one-op-per-frame.

    - source_frame_count: the number of REAL frames counted while walking the
      manifest -- returned so the caller can cross-check it against both the
      manifest's own declared "source_frame_count" and the actual extracted
      RPU's real frame count (exact-frame-count matching over approximate
      arithmetic, per docs/ai/domains/DOLBY_VISION.md).

    Raises ValueError if "frames" is missing/empty, or contains a synthetic
    frame before the first real frame or after the last one -- RIFE's own
    pairing logic (iw3.rife_cli) never produces this, but a hand-edited or
    corrupt manifest could."""
    frames = manifest.get("frames")
    if not frames:
        raise ValueError('RIFE manifest has no "frames" list (or it is empty)')

    ops = []
    original_position = 0
    run_source = None
    run_offset = None
    run_length = 0

    def _flush_run():
        nonlocal run_source, run_offset, run_length
        if run_length > 0:
            ops.append({"source": run_source, "offset": run_offset, "length": run_length})
        run_source = None
        run_offset = None
        run_length = 0

    for i, entry in enumerate(frames):
        if entry.get("real"):
            _flush_run()
            original_position += 1
        else:
            if original_position == 0:
                raise ValueError(
                    f"manifest frame {i} is synthetic but no real frame has been seen yet -- "
                    f"a RIFE output can never start with a synthetic frame")
            src = entry.get("nearest_real_index")
            if src is None:
                raise ValueError(f"manifest frame {i} is synthetic but has no nearest_real_index")
            if run_length > 0 and src == run_source and run_offset == original_position:
                run_length += 1
            else:
                _flush_run()
                run_source = src
                run_offset = original_position
                run_length = 1
    if run_length > 0:
        raise ValueError(
            "manifest ends with synthetic frame(s) after the last real frame -- a RIFE output can "
            "never end with a synthetic frame (the final real frame is always flushed alone)")
    return ops, original_position


def _expand_rpu_for_rife(source_rpu_path, ops, expected_source_frame_count, dovi_bin, work_dir):
    """Expands source_rpu_path (an already-extracted RPU, TRIMMED to exactly the
    range that was fed into RIFE -- see docs/ai/domains/DOLBY_VISION.md's
    exact-trim invariant) into a new RPU binary matching RIFE's output frame
    count, using dovi_tool's own `editor --json` "duplicate" operations (see
    _build_duplicate_ops_from_manifest above). Cross-checks the ACTUAL
    extracted RPU's real frame count (via `dovi_tool export -d all`, exact
    frame-count matching, not approximate arithmetic) against
    expected_source_frame_count before trusting `ops`' offsets/sources -- those
    were computed against the MANIFEST's own idea of the source frame count,
    and must genuinely match the real RPU or every offset/source index in
    `ops` would silently point at the wrong frame.

    Real, confirmed correction (2026-09-08, see docs/ai/AI_DECISIONS.md ADR-051's
    amendment): when the extraction finds MORE real frames than
    expected_source_frame_count, this is routine `-ss`/`-to` (`-c:v copy`)
    keyframe/timestamp trim slop, not a genuine mismatch -- real testing against
    this project's own DV test source found the extraction's OWN leading frames
    matched the real conversion pipeline's actual rendered frames byte-for-byte;
    only the excess lived at the END. This is auto-corrected here by trimming the
    RPU's tail down to expected_source_frame_count (via dovi_tool `editor`'s own
    `remove` operation) BEFORE building/applying `ops` -- confirmed, via a real
    dovi_tool run against real extracted DV RPU data, to produce a result
    byte-identical to ground truth. This mirrors `dovi_tool inject-rpu`'s own
    real, already-trusted auto-correction for the exact same situation
    (confirmed directly: "Metadata will be skipped at the end to match video
    length") -- done here explicitly and earlier so `ops`' indices (built
    against exactly expected_source_frame_count frames) stay valid. Bounded by
    _SOURCE_TRIM_SLOP_MAX_FRAMES -- a larger excess is refused as more likely a
    genuinely wrong --start-time/--end-time than routine trim slop. A SHORTAGE
    (fewer real frames than expected) is never auto-corrected -- there is no
    real metadata to invent for missing frames -- and still refuses.

    Returns the expanded RPU binary's path -- or source_rpu_path itself
    (trimmed first, if it had excess frames), verbatim, if `ops` is empty (an
    all-real manifest, nothing to expand)."""
    export_json_path = path.join(work_dir, "_rife_expand_source_export.json")
    subprocess.run(
        [dovi_bin, "export", "-i", str(source_rpu_path), "-d", f"all={export_json_path}"],
        check=True, capture_output=True,
    )
    with open(export_json_path, "r", encoding="utf-8") as f:
        exported_frames = json.load(f)
    if not isinstance(exported_frames, list):
        raise ValueError("dovi_tool export -d all did not produce the expected JSON array of frames")

    if len(exported_frames) != expected_source_frame_count:
        excess = len(exported_frames) - expected_source_frame_count
        if excess < 0:
            raise ValueError(
                f"Extracted source RPU has {len(exported_frames)} real frames, but the RIFE manifest "
                f"expects {expected_source_frame_count} real source frames -- the extraction found "
                f"{-excess} FEWER frames than expected, which can never be safely auto-corrected (there "
                f"is no real metadata to invent for the missing frames). This is very likely a "
                f"--start-time/--end-time mismatch between this RPU extraction and the original RIFE "
                f"input. Refusing to proceed.")
        if excess > _SOURCE_TRIM_SLOP_MAX_FRAMES:
            raise ValueError(
                f"Extracted source RPU has {len(exported_frames)} real frames, {excess} MORE than the "
                f"RIFE manifest's expected {expected_source_frame_count} -- too many to be routine "
                f"-ss/-to trim slop (this only auto-corrects up to {_SOURCE_TRIM_SLOP_MAX_FRAMES} "
                f"excess frames -- see docs/ai/AI_DECISIONS.md). This is very likely a genuinely wrong "
                f"--start-time/--end-time. Refusing to proceed.")
        print(f"[reinject-hdr] extracted source RPU has {len(exported_frames)} real frames, {excess} "
              f"more than the RIFE manifest's expected {expected_source_frame_count} -- trimming the "
              f"extra {excess} frame(s) from the END (routine -ss/-to trim slop, confirmed by real "
              f"testing to land there, not a genuine mismatch -- see docs/ai/AI_DECISIONS.md).",
              file=sys.stderr)
        trim_config = {"remove": [f"{expected_source_frame_count}-{len(exported_frames) - 1}"]}
        trim_config_path = path.join(work_dir, "_rife_expand_trim_config.json")
        with open(trim_config_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(trim_config, f)
        trimmed_rpu_path = path.join(work_dir, "_rife_expand_trimmed_source_rpu.bin")
        subprocess.run(
            [dovi_bin, "editor", "-i", str(source_rpu_path), "-j", trim_config_path,
             "-o", trimmed_rpu_path],
            check=True, capture_output=True,
        )
        source_rpu_path = trimmed_rpu_path

    if not ops:
        return str(source_rpu_path)

    edit_config = {"duplicate": ops}
    edit_config_path = path.join(work_dir, "_rife_expand_edit_config.json")
    # No BOM -- dovi_tool's JSON parser breaks on a UTF-8-with-BOM file (see
    # docs/ai/domains/DOLBY_VISION.md's manual-reinjection-workflow note); a
    # plain "w" text-mode open with encoding="utf-8" never adds one.
    with open(edit_config_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(edit_config, f)

    expanded_rpu_path = path.join(work_dir, "_rife_expanded_rpu.bin")
    subprocess.run(
        [dovi_bin, "editor", "-i", str(source_rpu_path), "-j", edit_config_path, "-o", expanded_rpu_path],
        check=True, capture_output=True,
    )
    return expanded_rpu_path


def _run_with_rife_manifest(args):
    """RIFE-manifest-aware retroactive DV RPU reinjection (ADR-051) -- a
    DISTINCT code path from _run_strict()'s exact-frame-count gate, never a
    relaxation of it (only reached when --rife-manifest is explicitly given).

    Use case: --converted was produced by (1) converting --source normally
    WITHOUT --preserve-dowi (RIFE and --preserve-dowi remain mutually
    exclusive within a single iw3 conversion run -- ADR-029, unchanged), then
    (2) running iw3.rife_cli against that plain output, which always writes a
    '<rife_output>.rife_manifest.json' sidecar alongside its own output
    (ADR-051). That manifest records, for every RIFE OUTPUT frame, whether
    it's real (and its original source-frame index) or synthetic (and its
    nearest real neighbor's source-frame index) -- letting this function build
    a Dolby Vision RPU EXPANDED to match RIFE's actual output frame count,
    duplicating each synthetic frame's nearest real neighbor's actual RPU
    entry (never a blend -- see ADR-051 for why this is safe across a scene
    cut by construction)."""
    ffmpeg_bin = _get_ffmpeg_bin()
    ffprobe_bin = _find_ffprobe()

    source = str(args.source)
    converted = str(args.converted)
    output = str(args.output)
    manifest_path = str(args.rife_manifest)

    for label, p in (("--source", source), ("--converted", converted), ("--rife-manifest", manifest_path)):
        if not path.exists(p):
            print(f"ERROR: {label} file does not exist: {p}", file=sys.stderr)
            return 1

    if path.abspath(output) in (path.abspath(source), path.abspath(converted)):
        print("ERROR: --output must be a different path from both --source and --converted -- "
              "this tool never overwrites either input.", file=sys.stderr)
        return 1

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        print(f"ERROR: could not read/parse --rife-manifest {manifest_path}: {e}", file=sys.stderr)
        return 1

    try:
        ops, source_frame_count = _build_duplicate_ops_from_manifest(manifest)
    except (KeyError, ValueError, TypeError) as e:
        print(f"ERROR: --rife-manifest {manifest_path} is malformed: {e}", file=sys.stderr)
        return 1
    output_frame_count = len(manifest.get("frames", []))

    print("[reinject-hdr] (RIFE-manifest mode) probing --source (trimmed) and --converted decoded "
          "frame counts against the manifest's own recorded counts -- this decodes the full range so "
          "it may take a while on long clips...", file=sys.stderr)
    src_frames, src_duration, src_cmd = _probe_frames_and_duration(
        source, ffprobe_bin, args.start_time, args.end_time)
    conv_frames, conv_duration, conv_cmd = _probe_frames_and_duration(converted, ffprobe_bin, None, None)

    print(f"[reinject-hdr] source ffprobe command:    {_format_cmd(src_cmd)}", file=sys.stderr)
    print(f"[reinject-hdr] converted ffprobe command: {_format_cmd(conv_cmd)}", file=sys.stderr)
    print(f"[reinject-hdr] source (trimmed) decoded frame count: {src_frames}  "
          f"manifest source_frame_count: {source_frame_count}", file=sys.stderr)
    print(f"[reinject-hdr] converted decoded frame count:        {conv_frames}  "
          f"manifest output_frame_count: {output_frame_count}", file=sys.stderr)

    if src_frames is None or conv_frames is None:
        print("ERROR: could not determine a decoded frame count for one or both files via ffprobe "
              "-- refusing to proceed without a reliable frame-count comparison. Re-run the exact "
              "ffprobe command(s) printed above by hand to see why.", file=sys.stderr)
        return 1

    tolerance = max(0, int(getattr(args, "frame_count_tolerance", 0) or 0))
    conv_mismatch = abs(conv_frames - output_frame_count)

    # Source side: real testing (2026-09-08, see docs/ai/AI_DECISIONS.md ADR-051's
    # amendment) found a genuinely correct --start-time/--end-time can still probe
    # MORE frames on --source than the manifest's own source_frame_count -- ffprobe/
    # ffmpeg range-trimming is keyframe/timestamp-approximate, while
    # source_frame_count is the EXACT count the real conversion pipeline actually
    # rendered for that range (the authoritative value -- see DOLBY_VISION.md's own
    # "exact frame count over approximate timestamp math" invariant). An excess
    # here is expected and safe -- _expand_rpu_for_rife (below) trims the extracted
    # RPU's own tail to match, mirroring dovi_tool inject-rpu's own real, already-
    # trusted auto-correction for the identical situation. A SHORTAGE is never safe
    # (missing DV metadata can't be fabricated) and still refuses immediately, same
    # as before this fix. A large excess (beyond _SOURCE_TRIM_SLOP_MAX_FRAMES) also
    # still refuses -- more likely a genuinely wrong --start-time/--end-time than
    # routine trim slop. The converted/RIFE side is NOT given this same allowance:
    # --converted is probed as a whole file with no time-range trimming involved at
    # all, and RIFE's own frame-doubling math is fully deterministic, so any
    # mismatch there still means --converted/--rife-manifest are from different
    # runs, exactly as before.
    src_shortage = max(0, source_frame_count - src_frames)
    src_excess = max(0, src_frames - source_frame_count)
    src_refused = src_shortage > tolerance or src_excess > max(tolerance, _SOURCE_TRIM_SLOP_MAX_FRAMES)

    if src_refused or conv_mismatch > tolerance:
        print(
            "ERROR: frame counts don't match the RIFE manifest's own recorded counts -- refusing to "
            "inject.\n"
            f"  source (trimmed) frames: {src_frames}   manifest source_frame_count: "
            f"{source_frame_count}   shortage: {src_shortage}   excess: {src_excess}   "
            f"(excess auto-corrected up to {_SOURCE_TRIM_SLOP_MAX_FRAMES} frames)\n"
            f"  converted frames:        {conv_frames}   manifest output_frame_count: "
            f"{output_frame_count}   mismatch: {conv_mismatch}\n"
            f"  tolerance: {tolerance}\n"
            "This almost always means --start-time/--end-time doesn't exactly match the range that "
            "was actually fed into RIFE, or --converted/--rife-manifest are from different RIFE runs. "
            "Use --frame-count-tolerance as a deliberate escape hatch only after independently "
            "confirming a specific known boundary-frame discrepancy.",
            file=sys.stderr)
        return 1

    if src_excess > 0:
        print(f"[reinject-hdr] (RIFE-manifest mode) source has {src_excess} more decoded frame(s) than "
              f"the manifest expects -- routine -ss/-to trim slop, not a mismatch; the extraction step "
              f"below trims the extracted RPU to match exactly.", file=sys.stderr)
    print("[reinject-hdr] (RIFE-manifest mode) frame counts match the manifest (within tolerance/"
          "the auto-corrected trim-slop allowance). Proceeding.", file=sys.stderr)

    out_dir = path.dirname(path.abspath(output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output = path.splitext(output)[0] + ".hdr_inject_tmp" + path.splitext(output)[1]
    print(f"[reinject-hdr] copying --converted to a working copy: {tmp_output}", file=sys.stderr)
    shutil.copyfile(converted, tmp_output)

    work_dir = tempfile.mkdtemp(prefix="iw3_reinject_hdr_rife_")
    hdr_rpu_path = path.join(work_dir, "dv_rpu.bin")
    hdr_h10p_path = path.join(work_dir, "hdr10plus.json")
    try:
        shim_args = argparse.Namespace(start_time=args.start_time, end_time=args.end_time, state=None)
        print("[reinject-hdr] extracting Dolby Vision / HDR10+ metadata from --source...",
              file=sys.stderr)
        dv_ok, h10p_ok, failures = _extract_hdr_rpu_files(
            source, work_dir, shim_args, hdr_rpu_path, hdr_h10p_path)
        for failure in failures:
            print(f"[reinject-hdr] WARNING: {failure}", file=sys.stderr)

        have_rpu = path.exists(hdr_rpu_path)
        if path.exists(hdr_h10p_path):
            print("[reinject-hdr] WARNING: --source also has HDR10+ dynamic metadata, but "
                  "RIFE-manifest expansion currently only expands Dolby Vision RPU -- HDR10+ will "
                  "NOT be reinjected in this mode (see ADR-051).", file=sys.stderr)
        if not have_rpu:
            print("ERROR: no Dolby Vision RPU was extracted from --source -- nothing to expand/inject. "
                  "Check that --source genuinely has DV metadata and that --start-time/--end-time are "
                  "correct.", file=sys.stderr)
            return 1

        dovi_bin = _find_dovi_tool()
        if not dovi_bin:
            print("ERROR: dovi_tool not found -- cannot expand or inject the RPU.", file=sys.stderr)
            return 1

        print(f"[reinject-hdr] expanding source RPU ({source_frame_count} real frames) to match "
              f"RIFE's output ({output_frame_count} frames) via {len(ops)} duplicate operation(s)...",
              file=sys.stderr)
        expanded_rpu_path = _expand_rpu_for_rife(
            hdr_rpu_path, ops, source_frame_count, dovi_bin, work_dir)

        print("[reinject-hdr] injecting expanded RPU into a copy of --converted...", file=sys.stderr)
        injected_hevc, fps_str, reason = _inject_hdr_rpu(
            tmp_output, expanded_rpu_path, None,
            ffmpeg_bin, dovi_bin, None, work_dir,
            configured_video_codec=None,
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


def _run_strict(args):
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
        assert "-count_frames" in cmd and "-ss" not in cmd and "-read_intervals" not in cmd

    with patch.object(subprocess, "run") as mock_run:
        # Container metadata says 100s (the WHOLE file) even though we asked for a 10s
        # trimmed window -- the function must report the requested range (20-10=10),
        # not the untrimmed container duration.
        mock_run.return_value = MagicMock(stdout=_stdout(240, 100.0), returncode=0)
        frames, duration, cmd = _probe_frames_and_duration(
            "fake.mkv", "ffprobe", start_time="10", end_time="20")
        assert frames == 240, frames
        assert duration == 10.0, duration
        # Real, confirmed bug fix (see _read_intervals_for_range's docstring): this
        # project's bundled ffprobe.exe does not support -ss/-to at all -- must never
        # reappear in the built command. -read_intervals is ffprobe's own real,
        # confirmed-working native equivalent.
        assert "-ss" not in cmd and "-to" not in cmd
        assert "-read_intervals" in cmd
        assert cmd[cmd.index("-read_intervals") + 1] == "10.0%+10.0", cmd

    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(stdout="not json", returncode=1)
        frames, duration, cmd = _probe_frames_and_duration("fake.mkv", "ffprobe")
        assert frames is None, frames

    print("_self_test_probe_frames_and_duration: PASS")


def _self_test_read_intervals_for_range():
    """Regression test for _read_intervals_for_range -- the real fix for the confirmed
    bug that this project's bundled ffprobe.exe does not support -ss/-to at all (see its
    own docstring). Covers all three real forms confirmed working against the real
    bundled binary during this task: start+end (a bounded range), start-only (to EOF),
    end-only (from the beginning)."""
    assert _read_intervals_for_range(None, None) is None
    assert _read_intervals_for_range("10", "20") == "10.0%+10.0"
    assert _read_intervals_for_range("00:01:15", "00:01:30") == "75.0%+15.0"
    assert _read_intervals_for_range("10", None) == "10.0%"
    assert _read_intervals_for_range(None, "20") == "%20.0"
    print("_self_test_read_intervals_for_range: PASS")


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


def _simulate_dovi_editor_duplicate(data, ops):
    """Test-only pure-Python reimplementation of dovi_tool's real "duplicate"
    algorithm (verified directly against its actual Rust source,
    quietvoid/dovi_tool src/dovi/editor.rs, fetched 2026-09-08: sort ops by
    offset ascending, reverse to descending, then apply each via
    Vec::splice(offset..offset, repeat(data[source], length))) -- used ONLY to
    test _build_duplicate_ops_from_manifest's output against fake per-frame
    markers without shelling out to the real dovi_tool binary for every case.
    The REAL binary's behavior (including this exact tie-break/ordering
    mechanics) was independently confirmed via a real end-to-end run against
    real Dolby Vision RPU data extracted from this project's own DV test
    source -- see docs/ai/AI_DECISIONS.md ADR-051's verification section; this
    reimplementation itself is not what proves correctness, that real run is."""
    data = list(data)
    ordered = sorted(ops, key=lambda o: o["offset"])
    ordered.reverse()
    for op in ordered:
        source_item = data[op["source"]]
        data[op["offset"]:op["offset"]] = [source_item] * op["length"]
    return data


def _self_test_build_duplicate_ops_from_manifest():
    """Synthetic/mocked test (CS-TEST-001 -- no dovi_tool/network/GPU) of
    _build_duplicate_ops_from_manifest for a 2x manifest, a non-2x (3x)
    manifest that forces the same-offset tie-break case, and a simulated
    scene-cut boundary. Applies the returned ops (via
    _simulate_dovi_editor_duplicate, a test-only reimplementation of
    dovi_tool's real algorithm) against fake per-frame RPU markers and
    confirms: every real frame passes through unchanged; every synthetic frame
    duplicates exactly the correct nearest-neighbor marker; and -- the
    scene-cut-safety reasoning ADR-051 documents -- a synthetic frame sitting
    exactly at a scene-cut boundary (its two real neighbors tagged as
    different "scenes", one carrying scene_refresh_flag=1) duplicates from
    WHICHEVER SIDE the manifest's own t<0.5/t>=0.5 rule already picked, never
    a mix of both scenes' data -- duplication is inherently scene-cut-safe by
    construction: it always copies ONE real frame's entire entry verbatim
    (including whatever scene_refresh_flag that real frame happens to carry),
    it never averages/blends two entries together."""
    # 2x case: 5 real frames, one synthetic between each pair, always the
    # LATER real neighbor (t>=0.5 side of the rule, per rife_cli's own
    # _nearest_real_index -- exercises the ">=" branch specifically).
    manifest_2x = {
        "frames": [
            {"real": True, "source_index": 0},
            {"real": False, "nearest_real_index": 1},
            {"real": True, "source_index": 1},
            {"real": False, "nearest_real_index": 1},
            {"real": True, "source_index": 2},
            {"real": False, "nearest_real_index": 3},
            {"real": True, "source_index": 3},
            {"real": False, "nearest_real_index": 3},
            {"real": True, "source_index": 4},
        ]
    }
    ops, count = _build_duplicate_ops_from_manifest(manifest_2x)
    assert count == 5, count
    fake_rpu = [{"id": i} for i in range(5)]
    expanded = _simulate_dovi_editor_duplicate(fake_rpu, ops)
    expected = [
        fake_rpu[0], fake_rpu[1], fake_rpu[1], fake_rpu[1], fake_rpu[2],
        fake_rpu[3], fake_rpu[3], fake_rpu[3], fake_rpu[4],
    ]
    assert expanded == expected, expanded

    # 3x case: two synthetic frames per gap, one nearest each side -- both ops
    # of a given gap share the SAME offset (the real tie-break case).
    manifest_3x = {
        "frames": [
            {"real": True, "source_index": 0},
            {"real": False, "nearest_real_index": 0},
            {"real": False, "nearest_real_index": 1},
            {"real": True, "source_index": 1},
            {"real": False, "nearest_real_index": 1},
            {"real": False, "nearest_real_index": 2},
            {"real": True, "source_index": 2},
        ]
    }
    ops3, count3 = _build_duplicate_ops_from_manifest(manifest_3x)
    assert count3 == 3, count3
    fake_rpu3 = [{"id": i} for i in range(3)]
    expanded3 = _simulate_dovi_editor_duplicate(fake_rpu3, ops3)
    expected3 = [
        fake_rpu3[0], fake_rpu3[0], fake_rpu3[1],
        fake_rpu3[1], fake_rpu3[1], fake_rpu3[2],
        fake_rpu3[2],
    ]
    assert expanded3 == expected3, expanded3

    # Scene-cut boundary: real frame 2 is the LAST frame of scene A, real
    # frame 3 is the scene-cut frame (FIRST frame of scene B,
    # scene_refresh_flag=1). A synthetic frame at t<0.5 (nearest real frame 2,
    # still scene A) must duplicate scene A's data; one at t>=0.5 (nearest
    # real frame 3, already scene B) must duplicate scene B's data -- proving
    # the "nearest by timestep" rule already handles this correctly by
    # construction, with no special-case scene-boundary code needed anywhere
    # in this pipeline.
    manifest_cut = {
        "frames": [
            {"real": True, "source_index": 0},
            {"real": False, "nearest_real_index": 0},
            {"real": True, "source_index": 1},
            {"real": False, "nearest_real_index": 1},
            {"real": True, "source_index": 2},        # last frame of scene A
            {"real": False, "nearest_real_index": 2},  # t<0.5 -> still scene A
            {"real": True, "source_index": 3},         # the real scene-cut frame (scene B starts)
            {"real": False, "nearest_real_index": 3},  # t>=0.5 -> already scene B
            {"real": True, "source_index": 4},
        ]
    }
    ops_cut, count_cut = _build_duplicate_ops_from_manifest(manifest_cut)
    assert count_cut == 5, count_cut
    fake_rpu_cut = [
        {"scene": "A", "scene_refresh_flag": 0, "id": 0},
        {"scene": "A", "scene_refresh_flag": 0, "id": 1},
        {"scene": "A", "scene_refresh_flag": 0, "id": 2},
        {"scene": "B", "scene_refresh_flag": 1, "id": 3},
        {"scene": "B", "scene_refresh_flag": 0, "id": 4},
    ]
    expanded_cut = _simulate_dovi_editor_duplicate(fake_rpu_cut, ops_cut)
    expected_cut = [
        fake_rpu_cut[0], fake_rpu_cut[0],
        fake_rpu_cut[1], fake_rpu_cut[1],
        fake_rpu_cut[2], fake_rpu_cut[2],  # synthetic before the cut stays scene A
        fake_rpu_cut[3], fake_rpu_cut[3],  # synthetic after the cut is already scene B
        fake_rpu_cut[4],
    ]
    assert expanded_cut == expected_cut, expanded_cut
    # No expanded entry is ever a mix of the two scenes -- each is exactly one
    # of the two real dicts (by value equality, which for these unique-"id"
    # fake entries also confirms no fields were merged/blended together).
    for entry in expanded_cut:
        assert entry["scene"] in ("A", "B") and entry in fake_rpu_cut

    # Malformed-manifest guards.
    try:
        _build_duplicate_ops_from_manifest({"frames": []})
        raise AssertionError("expected ValueError for empty frames list")
    except ValueError:
        pass
    try:
        _build_duplicate_ops_from_manifest(
            {"frames": [{"real": False, "nearest_real_index": 0}]})
        raise AssertionError("expected ValueError for a manifest starting with a synthetic frame")
    except ValueError:
        pass
    try:
        _build_duplicate_ops_from_manifest(
            {"frames": [{"real": True, "source_index": 0}, {"real": False, "nearest_real_index": 0}]})
        raise AssertionError("expected ValueError for a manifest ending with a synthetic frame")
    except ValueError:
        pass

    print("_self_test_build_duplicate_ops_from_manifest: PASS")


def _self_test_expand_rpu_for_rife():
    """Synthetic/mocked test (subprocess.run mocked throughout -- no real
    dovi_tool binary needed) of _expand_rpu_for_rife: (a) a frame-count
    mismatch against the real exported RPU is caught BEFORE dovi_tool editor
    is ever invoked; (b) a genuine match proceeds to write a well-formed
    {"duplicate": [...]} edit config (no BOM) and invoke exactly
    `dovi_tool editor -i <source_rpu> -j <edit_config> -o <expanded_rpu>`;
    (c) an empty ops list (an all-real manifest) short-circuits to returning
    source_rpu_path verbatim, with NO editor call at all (export is still
    called, since the frame-count cross-check itself is not optional)."""
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory(prefix="iw3_reinject_rife_selftest_") as work_dir:
        source_rpu = path.join(work_dir, "dv_rpu.bin")
        with open(source_rpu, "wb") as f:
            f.write(b"fake-rpu")

        def _fake_export(cmd, **kwargs):
            export_arg = next(c for c in cmd if str(c).startswith("all="))
            export_json_path = export_arg.split("=", 1)[1]
            with open(export_json_path, "w", encoding="utf-8") as f:
                json.dump([{"id": i} for i in range(5)], f)
            return MagicMock(returncode=0)

        # (a) mismatch: caller expects 6 real frames, real export has 5 -> refuse before editor.
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = _fake_export
            try:
                _expand_rpu_for_rife(source_rpu, [{"source": 0, "offset": 1, "length": 1}],
                                      6, "dovi_tool", work_dir)
                raise AssertionError("expected ValueError for frame-count mismatch")
            except ValueError as e:
                assert "5" in str(e) and "6" in str(e), e
            assert mock_run.call_count == 1
            assert mock_run.call_args_list[0][0][0][1] == "export"

        # (b) genuine match with real ops -> both export and editor invoked, editor args correct.
        editor_calls = []

        def _fake_export_and_editor(cmd, **kwargs):
            if cmd[1] == "export":
                return _fake_export(cmd, **kwargs)
            elif cmd[1] == "editor":
                editor_calls.append(cmd)
                out_path = cmd[cmd.index("-o") + 1]
                with open(out_path, "wb") as f:
                    f.write(b"fake-expanded-rpu")
                return MagicMock(returncode=0)
            raise AssertionError(f"unexpected dovi_tool subcommand: {cmd}")

        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = _fake_export_and_editor
            ops = [{"source": 0, "offset": 1, "length": 1}, {"source": 4, "offset": 5, "length": 2}]
            result = _expand_rpu_for_rife(source_rpu, ops, 5, "dovi_tool", work_dir)
            assert path.exists(result) and result != source_rpu
            assert len(editor_calls) == 1
            assert editor_calls[0][2] == "-i" and editor_calls[0][3] == source_rpu
            edit_config_path = editor_calls[0][editor_calls[0].index("-j") + 1]
            with open(edit_config_path, "r", encoding="utf-8") as f:
                written = json.load(f)
            assert written == {"duplicate": ops}, written
            with open(edit_config_path, "rb") as f:
                assert not f.read(3).startswith(b"\xef\xbb\xbf"), "edit config must not have a BOM"

        # (c) empty ops -> export still runs (cross-check), editor never does; source path returned.
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = _fake_export
            result = _expand_rpu_for_rife(source_rpu, [], 5, "dovi_tool", work_dir)
            assert result == source_rpu
            assert mock_run.call_count == 1

        # (d) real, confirmed fix (2026-09-08, see docs/ai/AI_DECISIONS.md ADR-051's
        # amendment): excess within the trim-slop cap is auto-corrected -- the RPU is
        # trimmed (dovi_tool editor "remove") down to expected_source_frame_count
        # BEFORE ops are applied, mirroring dovi_tool inject-rpu's own real
        # auto-correction for the identical situation. Two editor calls: one to trim
        # 7->5 (remove "5-6"), one to apply ops -- and the SECOND call must run
        # against the TRIMMED rpu, not the original extraction.
        editor_calls.clear()

        def _fake_export_7(cmd, **kwargs):
            export_arg = next(c for c in cmd if str(c).startswith("all="))
            export_json_path = export_arg.split("=", 1)[1]
            with open(export_json_path, "w", encoding="utf-8") as f:
                json.dump([{"id": i} for i in range(7)], f)  # 7 real frames extracted, 2 excess
            return MagicMock(returncode=0)

        def _fake_trim_and_duplicate(cmd, **kwargs):
            if cmd[1] == "export":
                return _fake_export_7(cmd, **kwargs)
            elif cmd[1] == "editor":
                editor_calls.append(cmd)
                out_path = cmd[cmd.index("-o") + 1]
                with open(out_path, "wb") as f:
                    f.write(f"fake-rpu-stage-{len(editor_calls)}".encode())
                return MagicMock(returncode=0)
            raise AssertionError(f"unexpected dovi_tool subcommand: {cmd}")

        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = _fake_trim_and_duplicate
            ops = [{"source": 0, "offset": 1, "length": 1}]
            result = _expand_rpu_for_rife(source_rpu, ops, 5, "dovi_tool", work_dir)
            assert len(editor_calls) == 2, editor_calls
            trim_call, dup_call = editor_calls
            assert trim_call[trim_call.index("-i") + 1] == source_rpu
            trim_config_path = trim_call[trim_call.index("-j") + 1]
            with open(trim_config_path, "r", encoding="utf-8") as f:
                trim_written = json.load(f)
            assert trim_written == {"remove": ["5-6"]}, trim_written
            with open(trim_config_path, "rb") as f:
                assert not f.read(3).startswith(b"\xef\xbb\xbf"), "trim config must not have a BOM"
            # The duplicate-ops call must run against the TRIMMED rpu, not the original
            # extraction -- so ops' indices (built against exactly 5 frames) stay valid.
            assert dup_call[dup_call.index("-i") + 1] != source_rpu
            assert path.exists(result)

        # (e) excess beyond the trim-slop cap -> refuses outright, no editor call at
        # all (more likely a genuinely wrong --start-time/--end-time than routine
        # trim slop -- see _SOURCE_TRIM_SLOP_MAX_FRAMES).
        with patch.object(subprocess, "run") as mock_run:
            def _fake_export_huge(cmd, **kwargs):
                export_arg = next(c for c in cmd if str(c).startswith("all="))
                export_json_path = export_arg.split("=", 1)[1]
                with open(export_json_path, "w", encoding="utf-8") as f:
                    json.dump([{"id": i} for i in range(5 + _SOURCE_TRIM_SLOP_MAX_FRAMES + 1)], f)
                return MagicMock(returncode=0)
            mock_run.side_effect = _fake_export_huge
            try:
                _expand_rpu_for_rife(source_rpu, [], 5, "dovi_tool", work_dir)
                raise AssertionError("expected ValueError for excess beyond the trim-slop cap")
            except ValueError as e:
                assert "too many" in str(e), e
            assert all(c.args[0][1] != "editor" for c in mock_run.call_args_list)

    print("_self_test_expand_rpu_for_rife: PASS")


def _self_test_rife_manifest_source_excess_tolerance():
    """Regression test for the real, confirmed fix (2026-09-08, see
    docs/ai/AI_DECISIONS.md ADR-051's amendment): a genuinely correct
    --start-time/--end-time can still probe MORE source frames than the RIFE
    manifest's own source_frame_count (routine -ss/-to trim slop -- real testing
    against this project's own DV test source found a real 31-38 frame gap from
    this cause alone), and _run_with_rife_manifest's own pre-flight gate must
    proceed past this (not refuse) as long as the excess is within
    _SOURCE_TRIM_SLOP_MAX_FRAMES and the converted side still matches exactly --
    NOT an unconditional relaxation: a genuine shortage, or an excess beyond the
    cap, must still refuse before extraction, exactly as before this fix."""
    import argparse as _argparse
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_reinject_rife_excess_selftest_") as tmpdir:
        source = path.join(tmpdir, "source.mkv")
        converted = path.join(tmpdir, "converted_rife.mkv")
        output = path.join(tmpdir, "output.mkv")
        manifest_path = path.join(tmpdir, "converted_rife.mkv.rife_manifest.json")
        for p in (source, converted):
            with open(p, "wb") as f:
                f.write(b"0")
        # 345 real source frames -> 689 output frames -- matches this project's own
        # real 2026-09-08 end-to-end test numbers exactly (2x RIFE multiplier).
        # Content of "frames" doesn't matter here (_build_duplicate_ops_from_manifest
        # is mocked below) -- only its LENGTH, since output_frame_count is derived
        # from len(manifest["frames"]).
        manifest = {"frames": [{"real": True}] * 689, "source_frame_count": 345}
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        def _args():
            return _argparse.Namespace(
                source=source, converted=converted, output=output,
                start_time=None, end_time=None, frame_count_tolerance=0,
                rife_manifest=manifest_path)

        # (a) source has a small excess (31 frames, matching the real observed gap)
        # and converted matches exactly -> must proceed past the gate (extraction is
        # reached), not refuse.
        with patch(f"{__name__}._probe_frames_and_duration") as mock_probe, \
             patch(f"{__name__}._extract_hdr_rpu_files") as mock_extract, \
             patch(f"{__name__}._build_duplicate_ops_from_manifest", return_value=([], 345)):
            mock_probe.side_effect = [(376, 15.0, ["ffprobe"]), (689, 30.0, ["ffprobe"])]
            mock_extract.return_value = (False, False, [])  # no RPU file -> refused AFTER the gate
            _run_with_rife_manifest(_args())
            mock_extract.assert_called_once()  # proves the gate was passed

        # (b) source excess beyond the cap -> still refuses, before extraction.
        with patch(f"{__name__}._probe_frames_and_duration") as mock_probe, \
             patch(f"{__name__}._extract_hdr_rpu_files") as mock_extract, \
             patch(f"{__name__}._build_duplicate_ops_from_manifest", return_value=([], 345)):
            mock_probe.side_effect = [(345 + _SOURCE_TRIM_SLOP_MAX_FRAMES + 1, 15.0, ["ffprobe"]),
                                       (689, 30.0, ["ffprobe"])]
            rc = _run_with_rife_manifest(_args())
            assert rc == 1, rc
            mock_extract.assert_not_called()

        # (c) source SHORTAGE (fewer frames than expected) -> still refuses
        # immediately regardless of how small -- never auto-corrected (no real
        # metadata to invent for missing frames).
        with patch(f"{__name__}._probe_frames_and_duration") as mock_probe, \
             patch(f"{__name__}._extract_hdr_rpu_files") as mock_extract, \
             patch(f"{__name__}._build_duplicate_ops_from_manifest", return_value=([], 345)):
            mock_probe.side_effect = [(344, 15.0, ["ffprobe"]), (689, 30.0, ["ffprobe"])]
            rc = _run_with_rife_manifest(_args())
            assert rc == 1, rc
            mock_extract.assert_not_called()

    print("_self_test_rife_manifest_source_excess_tolerance: PASS")


def _self_test_run_with_rife_manifest_dispatch_and_gating():
    """Synthetic/mocked test confirming: (1) run() dispatches to
    _run_with_rife_manifest ONLY when args.rife_manifest is truthy -- an args
    object with no such attribute at all (exactly what every pre-existing
    caller/test builds) still reaches the untouched strict path; (2) within
    the RIFE-manifest path, a frame count that doesn't match the manifest's
    OWN recorded source_frame_count/output_frame_count refuses BEFORE
    extraction; (3) a genuine match reaches extraction/expansion/injection
    with the right arguments."""
    import argparse as _argparse
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_reinject_rife_dispatch_selftest_") as tmpdir:
        source = path.join(tmpdir, "source.mkv")
        converted = path.join(tmpdir, "converted_rife.mkv")
        output = path.join(tmpdir, "output.mkv")
        manifest_path = path.join(tmpdir, "converted_rife.mkv.rife_manifest.json")
        for p in (source, converted):
            with open(p, "wb") as f:
                f.write(b"0")
        manifest = {
            "frames": [
                {"real": True, "source_index": 0},
                {"real": False, "nearest_real_index": 1},
                {"real": True, "source_index": 1},
            ],
            "source_frame_count": 2,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        def _args(rife_manifest=None):
            ns = _argparse.Namespace(
                source=source, converted=converted, output=output,
                start_time=None, end_time=None, frame_count_tolerance=0)
            if rife_manifest is not None:
                ns.rife_manifest = rife_manifest
            return ns

        # 1) no rife_manifest attribute at all -> dispatches to the untouched strict path
        # (proven by _check_rife_guard, a strict-path-only function, actually being called).
        with patch(f"{__name__}._check_rife_guard", return_value=["forced refusal"]) as mock_guard:
            rc = run(_args())
            assert rc == 1, rc
            mock_guard.assert_called_once()

        # 2) rife_manifest given, but real frame counts don't match the manifest -> refuse
        # before extraction, WITHOUT ever calling the strict path's RIFE guard.
        with patch(f"{__name__}._check_rife_guard") as mock_guard, \
             patch(f"{__name__}._probe_frames_and_duration") as mock_probe, \
             patch(f"{__name__}._extract_hdr_rpu_files") as mock_extract:
            mock_probe.side_effect = [(9, 9.0, ["ffprobe"]), (9, 9.0, ["ffprobe"])]
            rc = run(_args(rife_manifest=manifest_path))
            assert rc == 1, rc
            mock_guard.assert_not_called()
            mock_extract.assert_not_called()

        # 3) matching frame counts -> reaches extraction/expansion/injection with the right args.
        expand_calls = []

        def _fake_expand(rpu_path, ops, expected_count, dovi_bin, work_dir):
            expand_calls.append((rpu_path, ops, expected_count))
            return rpu_path

        with patch(f"{__name__}._probe_frames_and_duration") as mock_probe, \
             patch(f"{__name__}._extract_hdr_rpu_files") as mock_extract, \
             patch(f"{__name__}._find_dovi_tool", return_value="dovi_tool"), \
             patch(f"{__name__}._expand_rpu_for_rife", side_effect=_fake_expand), \
             patch(f"{__name__}._inject_hdr_rpu", return_value=(None, None, "stubbed decline")):
            mock_probe.side_effect = [(2, 2.0, ["ffprobe"]), (3, 3.0, ["ffprobe"])]

            def _fake_extract(input_path, work_dir, shim_args, rpu_path, h10p_path):
                assert input_path == source
                with open(rpu_path, "wb") as f:
                    f.write(b"rpu")
                return True, True, []
            mock_extract.side_effect = _fake_extract

            rc = run(_args(rife_manifest=manifest_path))
            assert rc == 1, rc  # injection stubbed to decline -- proves the gate passed cleanly
            assert len(expand_calls) == 1, expand_calls
            _, ops_used, expected_count_used = expand_calls[0]
            assert expected_count_used == 2, expected_count_used
            # manifest's synthetic frame has nearest_real_index=1 (the LATER real
            # neighbor) -> source=1, inserted right after original position 1.
            assert ops_used == [{"source": 1, "offset": 1, "length": 1}], ops_used

    print("_self_test_run_with_rife_manifest_dispatch_and_gating: PASS")


def _run_self_tests():
    _self_test_probe_frames_and_duration()
    _self_test_read_intervals_for_range()
    _self_test_rife_guard()
    _self_test_run_preflight_gating()
    _self_test_build_duplicate_ops_from_manifest()
    _self_test_expand_rpu_for_rife()
    _self_test_run_with_rife_manifest_dispatch_and_gating()
    _self_test_rife_manifest_source_excess_tolerance()
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
