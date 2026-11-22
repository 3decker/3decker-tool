"""python -m iw3.depth_curve_calibrate_cli -- builds the Depth Curve Match LUT
(see iw3/depth_curve_match.py) that lets the main conversion pipeline reshape one
depth model's output to resemble another model's distinctive value distribution,
without running that second model on every real job.

This is a genuinely standalone tool, invoked as its own subprocess (same "separate
subprocess, never sharing state with the main app" convention as iw3.rife_cli /
iw3.sharpen_cli -- see docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001), run ONCE
(or occasionally, if decker wants to re-calibrate against different content) --
not part of every normal conversion job.

What it actually does: runs one or more SHORT sample clips (not the whole movie --
see --clip, repeatable) through BOTH models, one model at a time (same
sequential-load-then-release VRAM discipline as depth_blend.py's Dual-Pass Depth
Blend -- the two models are never resident at once, and each model is loaded ONCE
and reused across every clip, not reloaded per clip), using iw3's own existing
--export --export-depth-only machinery (export_main in iw3/utils.py) -- no new
video-reading code. It then samples a modest number of random pixels per exported
frame from each model's depth output (full-frame, not a full pixel dump -- plenty
for a stable percentile curve, see --pixels-per-frame), pools the samples across
every clip, builds a histogram-specification LUT
(iw3.depth_curve_match.build_curve_lut) mapping the SOURCE model's typical values
onto the TARGET model's typical values at the same percentile rank, and saves it
to --output (a small JSON file, default iw3.depth_curve_match.DEFAULT_CURVE_PATH)
for the main pipeline's DepthCurveMatcher to load automatically on every future
job.

Multiple clips (--clip, repeatable) exist because a single curve is a fixed,
averaged shape -- a real, known trade-off noticed on live footage (decker
reported some moments still looking flat/under-reshaped at Strength 1.0): a clip
sampled from only ONE kind of shot produces a curve that fits that shot type well
but under-serves very different shots elsewhere in the movie. Sampling a FEW
separate, varied moments (a wide shot, a close-up, a flatter scene, etc.) and
pooling them into one curve gives a more generally-useful result than one longer
continuous stretch from a single scene.

See docs/ai/AI_DECISIONS.md ADR-467 for the research/reasoning this feature is
built on.
"""
import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime
from os import path

import numpy as np
import cv2

from .depth_curve_match import (build_curve_lut, save_curve, DEFAULT_CURVE_PATH,
                                refine_curve, max_curve_stretch, DEFAULT_MAX_STRETCH)


def _pass_depth_dir(output_dir, input_path):
    """Mirrors export_video's own output layout exactly (iw3/utils.py
    export_video): <output_dir>/<basename stripped>/depth/*.png (u16 PNGs).
    Duplicated here rather than imported, same "each standalone CLI tool reads
    iw3.utils's own layout rather than depending on a private helper" convention
    already used by sharpen_cli.py/stereo_mode_tag_cli.py for filename tags."""
    basename = path.splitext(path.basename(input_path))[0].strip()
    return path.join(output_dir, basename, "depth")


def _run_export_pass(input_path, model_name, pass_dir, start_time, end_time,
                      gpu, resolution, limit_resolution, depth_model=None, extra_argv=None):
    """Runs one depth-only export pass with one model, using iw3's own real
    argument parser + export pipeline (the same `create_parser() ->
    set_state_args() -> export_main()` sequence iw3.cli.main() uses for a normal
    job -- see CLAUDE.md's Architecture section) rather than hand-building a
    partial args object.

    If `depth_model` is given (an already-loaded model returned from a PREVIOUS
    call for this same model_name), reuses it instead of constructing a fresh one
    -- this is what lets a multi-clip calibration load each model exactly ONCE
    and export every clip's range with it, rather than paying a reload per clip.
    Returns the (now loaded) depth_model, for the caller to pass into the next
    clip's call, and to release once (see _release_model) after the LAST clip for
    this model.

    `extra_argv` (ADR-469): extra raw CLI flags appended as-is (e.g.
    ["--ema-normalize", "--ema-decay", "0.75", "--depth-aa"]) -- these are the
    depth-MODEL-level settings (Object Stability, Depth Refine, EMA Normalize,
    Depth Anti-aliasing) that genuinely change what the model outputs, as
    opposed to stereo-render-only settings (Convergence, Pop, Divergence, etc.)
    which never run during a depth-only export regardless. Without this, every
    one of those model-level settings silently defaults to OFF here even when
    decker's real jobs have them on, so the calibrated curve would be measuring
    a different (bare) depth output than his real jobs actually produce."""
    from .utils import create_parser, set_state_args, export_main

    argv = [
        "-i", input_path, "-o", pass_dir,
        "--depth-model", model_name,
        "--export", "--export-depth-only",
        "--gpu", str(gpu),
        "--yes",
    ]
    if start_time:
        argv += ["--start-time", str(start_time)]
    if end_time:
        argv += ["--end-time", str(end_time)]
    if resolution:
        argv += ["--resolution", str(resolution)]
    if limit_resolution:
        argv += ["--limit-resolution"]
    if extra_argv:
        argv += list(extra_argv)

    pass_args = create_parser().parse_args(argv)
    pass_args = set_state_args(pass_args, depth_model=depth_model)
    depth_model = pass_args.state["depth_model"]
    if not depth_model.loaded():
        depth_model.load(gpu=pass_args.gpu, resolution=pass_args.resolution,
                          limit_resolution=pass_args.limit_resolution)
    else:
        # ADR-141-style reuse: already loaded from a previous clip in this same
        # calibration run -- just confirm it's on the right device.
        depth_model.move_to(depth_model.device)
    export_main(pass_args)
    return depth_model


def _release_model(depth_model):
    """Fully releases a model loaded by _run_export_pass -- same `model = None` +
    gc_collect() discipline as depth_blend.py's own sequential pass release,
    called once after ALL clips for one model are done, not per clip."""
    from .utils import gc_collect
    if depth_model is not None:
        depth_model.model = None
        gc_collect()


def _sample_pixels_from_dir(depth_dir, pixels_per_frame=4000, seed=0):
    """Reads every exported u16 depth PNG in depth_dir, draws a random sample of
    `pixels_per_frame` pixels from EACH frame (not a full dump -- depth values are
    heavily spatially correlated, so a modest per-frame sample across many frames
    gives a stable percentile estimate without the memory/runtime cost of every
    pixel of every frame), and returns one flat float32 array normalized to
    [0, 1] (u16 full range is 0-65535, matching the SAME per-model EMA
    normalization base_depth_model.py already exports in). A fixed seed makes
    repeated runs over the same export reproducible."""
    files = sorted(f for f in os.listdir(depth_dir) if f.lower().endswith(".png"))
    if not files:
        raise RuntimeError(f"no exported depth frames found in {depth_dir}")
    rng = np.random.default_rng(seed)
    samples = []
    for fname in files:
        img = cv2.imread(path.join(depth_dir, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        flat = img.reshape(-1).astype(np.float32) / 65535.0
        n = min(pixels_per_frame, flat.shape[0])
        idx = rng.choice(flat.shape[0], size=n, replace=False)
        samples.append(flat[idx])
    if not samples:
        raise RuntimeError(f"found {len(files)} file(s) in {depth_dir} but none could be read")
    return np.concatenate(samples), len(samples)


def calibrate(input_path, source_model, target_model, clips,
              gpu=0, resolution=None, limit_resolution=False,
              pixels_per_frame=4000, num_points=512, output_path=None,
              work_dir=None, log=print, extra_argv=None, max_stretch=DEFAULT_MAX_STRETCH,
              target_reset_look=False):
    """Full calibration run over one or more clips: export both models over
    EVERY clip (each model loaded once, reused across all its clips), sample
    pixels from each clip/side, pool every clip's samples together, build the
    LUT from the pooled data, save it. `clips` is a list of (start_time,
    end_time) tuples. `extra_argv` (ADR-469) is forwarded unchanged to every
    export pass -- see _run_export_pass's own docstring for why this exists.
    Returns the saved file path.
    ADR-488 `target_reset_look`: the TARGET pass is exported with Scene Detection on, so
    its depth range (EMA) and model state restart at every scene cut ("reset look":
    each scene stretched over its own range), while the SOURCE pass stays exactly like a
    real conversion. The curve then reshapes normal depth toward the reset look, to be
    used in real jobs WITHOUT any reset. The target may be the source model itself."""
    if not clips:
        raise ValueError("calibrate() requires at least one clip")
    # ADR-480: a clip may name its own movie file -- (path, start, end) -- so one curve
    # can be pooled from several movies; plain (start, end) clips use input_path.
    clips = [((c[0], c[1], c[2]) if len(c) == 3 else (input_path, c[0], c[1])) for c in clips]
    for clip_path, _, _ in clips:
        if not path.exists(clip_path):
            raise ValueError(f"clip movie file does not exist: {clip_path}")
    output_path = output_path or DEFAULT_CURVE_PATH
    own_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="iw3_depth_curve_calibrate_")
    try:
        pass_dirs = [[None, None] for _ in clips]

        log(f"[depth-curve-calibrate] [1/3] exporting depth with source model "
            f"({source_model}) over {len(clips)} clip(s)...")
        model_obj = None
        for i, (clip_path, start_time, end_time) in enumerate(clips):
            log(f"  clip {i + 1}/{len(clips)}: {path.basename(clip_path)} "
                f"{start_time or 'start'}-{end_time or 'end'}")
            pass_a_dir = path.join(work_dir, f"clip{i}_source")
            model_obj = _run_export_pass(clip_path, source_model, pass_a_dir, start_time, end_time,
                                          gpu, resolution, limit_resolution, depth_model=model_obj,
                                          extra_argv=extra_argv)
            pass_dirs[i][0] = pass_a_dir
        _release_model(model_obj)

        target_argv = list(extra_argv or [])
        if target_reset_look:
            target_argv += ["--scene-detect"]  # ADR-488: depth restarts at every scene cut
        log(f"[depth-curve-calibrate] [2/3] exporting depth with target model "
            f"({target_model}{', reset look' if target_reset_look else ''}) over the same "
            f"{len(clips)} clip(s)...")
        model_obj = None
        for i, (clip_path, start_time, end_time) in enumerate(clips):
            log(f"  clip {i + 1}/{len(clips)}: {path.basename(clip_path)} "
                f"{start_time or 'start'}-{end_time or 'end'}")
            pass_b_dir = path.join(work_dir, f"clip{i}_target")
            model_obj = _run_export_pass(clip_path, target_model, pass_b_dir, start_time, end_time,
                                          gpu, resolution, limit_resolution, depth_model=model_obj,
                                          extra_argv=target_argv)
            pass_dirs[i][1] = pass_b_dir
        _release_model(model_obj)

        log(f"[depth-curve-calibrate] [3/3] sampling pixels and building the curve...")
        source_values_all, target_values_all = [], []
        total_frame_count = 0
        for i, (pass_a_dir, pass_b_dir) in enumerate(pass_dirs):
            clip_path = clips[i][0]
            source_depth_dir = _pass_depth_dir(pass_a_dir, clip_path)
            target_depth_dir = _pass_depth_dir(pass_b_dir, clip_path)
            source_values, source_frame_count = _sample_pixels_from_dir(
                source_depth_dir, pixels_per_frame, seed=i * 2)
            target_values, target_frame_count = _sample_pixels_from_dir(
                target_depth_dir, pixels_per_frame, seed=i * 2 + 1)
            if source_frame_count != target_frame_count:
                _, start_time, end_time = clips[i]
                raise RuntimeError(
                    f"clip {i + 1} ({start_time}-{end_time}): source model exported "
                    f"{source_frame_count} frame(s) but target model exported "
                    f"{target_frame_count} -- the two passes must cover the exact same frame "
                    f"range to build a valid curve. This usually means this clip's start/end "
                    f"time landed on a different frame count for each model (rare), or one "
                    f"pass was interrupted. Try a slightly different time range for this clip.")
            source_values_all.append(source_values)
            target_values_all.append(target_values)
            total_frame_count += source_frame_count

        source_values = np.concatenate(source_values_all)
        target_values = np.concatenate(target_values_all)
        lut_x, lut_y = build_curve_lut(source_values, target_values, num_points=num_points)
        # ADR-474: tidy the far-end tie cluster and cap the curve's local stretch.
        raw_stretch = max_curve_stretch(lut_x, lut_y)
        lut_x, lut_y = refine_curve(lut_x, lut_y, max_stretch)
        log(f"[depth-curve-calibrate] max stretch: raw {raw_stretch:.2f}x -> "
            f"{max_curve_stretch(lut_x, lut_y):.2f}x (limit {max_stretch if max_stretch else 'off'})")

        meta = {
            "source_model": source_model,
            "target_model": target_model,
            "input": path.basename(input_path),
            "clips": [(f"{s}-{e}" if p == input_path else f"{path.basename(p)} | {s}-{e}")
                      for p, s, e in clips],
            "inputs": sorted({path.basename(p) for p, _, _ in clips}),  # ADR-480
            "frame_count": total_frame_count,
            "pixels_per_frame": pixels_per_frame,
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "max_stretch": max_stretch,
            "target_reset_look": bool(target_reset_look),  # ADR-488
        }
        save_curve(output_path, lut_x, lut_y, meta=meta)
        log(f"[depth-curve-calibrate] done -- wrote {output_path} "
            f"({source_model} -> {target_model}{' (reset look)' if target_reset_look else ''}, "
            f"{total_frame_count} frames across "
            f"{len(clips)} clip(s))")
        return output_path
    finally:
        if own_work_dir and path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


def parse_clip_spec(raw):
    """Parses one --clip value ("START-END", e.g. "00:02:00-00:05:30") into a
    (start, end) tuple of strings. Splits on the FIRST "-" only -- HH:MM:SS itself
    never contains one, so this is unambiguous."""
    if "-" not in raw:
        raise ValueError(
            f"--clip value must be START-END (e.g. 00:02:00-00:05:30), got {raw!r}")
    start, end = raw.split("-", 1)
    start, end = start.strip(), end.strip()
    if not start or not end:
        raise ValueError(f"--clip value must have both a start and an end, got {raw!r}")
    return start, end


def parse_clip_line(raw):
    """ADR-480: one sample-clip line, either "START-END" (the main --input movie)
    or "PATH | START-END" (a clip from ANOTHER movie, so one curve can be pooled
    from several movies of a genre). Splits on the LAST "|" -- file names often
    contain "-" (e.g. "DTS-HD.MA 5.1-W32") but never "|" on Windows. Returns
    (path_or_None, start, end)."""
    raw = raw.strip()
    if "|" in raw:
        file_part, time_part = raw.rsplit("|", 1)
        file_part = file_part.strip().strip('"').strip()
        if not file_part:
            raise ValueError(f"clip line has '|' but no file before it: {raw!r}")
        start, end = parse_clip_spec(time_part.strip())
        return file_part, start, end
    start, end = parse_clip_spec(raw)
    return None, start, end


def resolve_clips(args):
    """Returns the list of (start, end) clips to calibrate from: every --clip
    value if any were given (repeatable -- see create_parser), otherwise a
    single clip from --start-time/--end-time (the original, single-range
    behavior, unchanged for anyone not using --clip)."""
    if getattr(args, "clip", None):
        out = []
        for raw in args.clip:
            p, s, e = parse_clip_line(raw)
            out.append((s, e) if p is None else (p, s, e))
        return out
    return [(args.start_time, args.end_time)]


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.depth_curve_calibrate_cli",
        description=(
            "Build the Depth Curve Match LUT: runs one or more short sample clips through "
            "both --source-model and --target-model, and saves a lookup table that lets the "
            "main pipeline reshape --source-model's depth output to resemble "
            "--target-model's distinctive distribution, without running --target-model on "
            "every real job. Run this once (or whenever you want to re-calibrate against "
            "different content); normal conversions just load the saved result."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                         help="Path to a representative source video to calibrate from.")
    parser.add_argument("--source-model", type=str, required=True,
                         help="The depth model you actually use for real jobs (e.g. VDA_L) "
                              "-- the one whose output will be reshaped at conversion time.")
    parser.add_argument("--target-model", type=str, required=True,
                         help="The depth model whose distinctive look you want to borrow "
                              "(e.g. Any_V3_Metric_Large) -- never actually run during real "
                              "jobs after this, only here during calibration.")
    parser.add_argument("--clip", action="append", default=None, metavar="START-END",
                         help="One sample range as START-END (HH:MM:SS-HH:MM:SS), e.g. "
                              "00:02:00-00:05:30. Repeat --clip to sample SEVERAL separate "
                              "moments from the movie instead of one continuous stretch -- "
                              "a curve built from more VARIETY (a wide shot, a close-up, a "
                              "flatter scene, a busy scene) reshapes more of the movie well "
                              "than one that only ever saw one kind of shot. If given at "
                              "least once, --start-time/--end-time below are ignored.")
    parser.add_argument("--start-time", type=str, default="00:01:00",
                         help="Start of the sample clip (HH:MM:SS). Only used when --clip is "
                              "never given. Default skips the first minute (often titles/"
                              "black frames).")
    parser.add_argument("--end-time", type=str, default="00:02:30",
                         help="End of the sample clip (HH:MM:SS). Only used when --clip is "
                              "never given. Keep any one clip short -- 60-120 seconds of "
                              "real, varied content is plenty.")
    parser.add_argument("--gpu", "-g", type=int, default=0,
                         help="GPU device index. -1 forces CPU.")
    parser.add_argument("--resolution", type=int, default=None,
                         help="Depth model inference resolution, same meaning as the main "
                              "pipeline's --resolution. Use the same value you use for real "
                              "jobs so the sampled curve matches real conditions.")
    parser.add_argument("--limit-resolution", action="store_true",
                         help="Same meaning as the main pipeline's --limit-resolution flag.")
    parser.add_argument("--pixels-per-frame", type=int, default=4000,
                         help="Random pixel sample size drawn from each exported frame.")
    parser.add_argument("--max-stretch", type=float, default=DEFAULT_MAX_STRETCH,
                        help="ADR-474: cap how strongly the curve may stretch any part of the "
                             "depth range (1.5 = at most 1.5x). Stops distant backgrounds being "
                             "over-deepened and their small ripples amplified. 0 = no limit.")
    parser.add_argument("--target-reset-look", action="store_true",
                        help=("ADR-488: export the TARGET pass with its depth range restarted at every "
                              "scene cut (each scene stretched over its own range -- the 'reset look'), "
                              "so the curve reshapes normal depth toward that look. Use the curve in "
                              "real conversions as usual (no reset there). Allows the same model as "
                              "source and target (e.g. VDA_L -> VDA_L reset look); with a different "
                              "target (e.g. Any_V3_Metric_Large) the curve carries both looks."))
    parser.add_argument("--num-points", type=int, default=512,
                         help="Resolution of the saved lookup table.")
    parser.add_argument("--output", "-o", type=str, default=None,
                         help=f"Where to save the curve file. Default: {DEFAULT_CURVE_PATH} "
                              f"(the same path the main pipeline auto-loads from).")
    parser.add_argument("--work-dir", type=str, default=None,
                         help="Temp directory for the export passes. Default: a fresh "
                              "system temp directory, deleted automatically when done.")

    group = parser.add_argument_group(
        "depth-model settings",
        "ADR-469: these change what the depth model actually outputs (unlike Convergence/"
        "Divergence/Pop/etc., which only affect the later stereo-render step and never run "
        "during calibration at all). Pass the SAME values your real jobs use here, so the "
        "calibrated curve measures the depth models the way they actually behave for you, "
        "not a bare/default configuration. Same flag names/meanings as the main pipeline.")
    group.add_argument("--depth-refine", action="store_true")
    group.add_argument("--depth-refine-strength", type=float, default=1.0)
    group.add_argument("--temporal-stabilize", action="store_true")
    group.add_argument("--temporal-stabilize-strength", type=float, default=0.7)
    group.add_argument("--temporal-stabilize-max-shift-velocity", type=float, default=None)
    group.add_argument("--temporal-stabilize-flat-region-boost", type=float, default=0.0)
    group.add_argument("--temporal-stabilize-edge-protection", type=float, default=0.0)
    group.add_argument("--temporal-stabilize-fast", action="store_true")
    group.add_argument("--temporal-stabilize-flow-engine", type=str, default="farneback",
                        choices=["farneback", "raft", "raft_fb"])
    group.add_argument("--ema-normalize", action="store_true")
    group.add_argument("--ema-decay", type=float, default=0.75)
    group.add_argument("--ema-buffer", type=int, default=30)
    group.add_argument("--depth-aa", action="store_true")
    return parser


def model_settings_argv(args):
    """Builds the extra argv (see _run_export_pass's docstring) from this tool's own
    depth-model-settings group above -- only emits a flag when it differs from that
    flag's own default, so a caller who passes none of them gets the exact same bare
    argv as before this existed (byte-for-byte backward compatible)."""
    argv = []
    if getattr(args, "depth_refine", False):
        argv += ["--depth-refine"]
        if getattr(args, "depth_refine_strength", 1.0) != 1.0:
            argv += ["--depth-refine-strength", str(args.depth_refine_strength)]
    if getattr(args, "temporal_stabilize", False):
        argv += ["--temporal-stabilize",
                 "--temporal-stabilize-strength", str(getattr(args, "temporal_stabilize_strength", 0.7)),
                 "--temporal-stabilize-flat-region-boost",
                 str(getattr(args, "temporal_stabilize_flat_region_boost", 0.0)),
                 "--temporal-stabilize-edge-protection",
                 str(getattr(args, "temporal_stabilize_edge_protection", 0.0)),
                 "--temporal-stabilize-flow-engine",
                 str(getattr(args, "temporal_stabilize_flow_engine", "farneback") or "farneback")]
        max_shift = getattr(args, "temporal_stabilize_max_shift_velocity", None)
        if max_shift is not None:
            argv += ["--temporal-stabilize-max-shift-velocity", str(max_shift)]
        if getattr(args, "temporal_stabilize_fast", False):
            argv += ["--temporal-stabilize-fast"]
    if getattr(args, "ema_normalize", False):
        argv += ["--ema-normalize",
                 "--ema-decay", str(getattr(args, "ema_decay", 0.75)),
                 "--ema-buffer", str(getattr(args, "ema_buffer", 30))]
    if getattr(args, "depth_aa", False):
        argv += ["--depth-aa"]
    return argv


def run(args):
    if not path.exists(args.input):
        print(f"ERROR: --input file does not exist: {args.input}", file=sys.stderr)
        return 1
    if args.source_model == args.target_model and not getattr(args, "target_reset_look", False):
        print("ERROR: --source-model and --target-model must be different -- there is "
              "nothing to reshape toward if they're the same model (unless --target-reset-look "
              "is on).", file=sys.stderr)
        return 1
    try:
        clips = resolve_clips(args)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    try:
        calibrate(
            args.input, args.source_model, args.target_model, clips,
            gpu=args.gpu, resolution=args.resolution, limit_resolution=args.limit_resolution,
            pixels_per_frame=args.pixels_per_frame, num_points=args.num_points,
            output_path=args.output, work_dir=args.work_dir,
            log=lambda msg: print(msg, file=sys.stderr),
            extra_argv=model_settings_argv(args),
            max_stretch=getattr(args, "max_stretch", DEFAULT_MAX_STRETCH),
            target_reset_look=bool(getattr(args, "target_reset_look", False)),
        )
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def _make_fake_depth_png(file_path, value, size=16):
    """Writes a tiny, uniform-except-noise u16 PNG at `value` (0.0-1.0) for self-tests --
    real cv2.imwrite/imread round trip, no GPU or real model needed."""
    import numpy as _np
    rng = _np.random.default_rng(abs(hash(file_path)) % (2**32))
    base = int(round(value * 65535))
    noise = rng.integers(-500, 500, size=(size, size))
    img = _np.clip(base + noise, 0, 65535).astype(_np.uint16)
    cv2.imwrite(file_path, img)


def _self_test_sample_pixels_from_dir():
    import tempfile as _tempfile
    with _tempfile.TemporaryDirectory(prefix="iw3_curve_calib_selftest_") as tmpdir:
        for i in range(5):
            _make_fake_depth_png(path.join(tmpdir, f"{i:06d}.png"), value=0.5)
        values, frame_count = _sample_pixels_from_dir(tmpdir, pixels_per_frame=50)
        assert frame_count == 5
        assert values.shape[0] == 5 * 50
        assert 0.0 <= values.min() and values.max() <= 1.0
        assert abs(float(values.mean()) - 0.5) < 0.05
    print("_self_test_sample_pixels_from_dir: PASS")


def _self_test_pass_depth_dir_layout():
    assert _pass_depth_dir("/out", "/videos/My Movie.mkv") == path.join("/out", "My Movie", "depth")
    # Windows-style trailing-space basenames must be stripped, same as export_video's own rule.
    assert _pass_depth_dir("/out", "/videos/My Movie .mkv") == path.join("/out", "My Movie", "depth")
    print("_self_test_pass_depth_dir_layout: PASS")


def _self_test_parse_clip_spec_and_resolve_clips():
    assert parse_clip_spec("00:02:00-00:05:30") == ("00:02:00", "00:05:30")
    assert parse_clip_spec(" 00:02:00 - 00:05:30 ") == ("00:02:00", "00:05:30")
    for bad in ("no dash here", "-00:05:30", "00:02:00-"):
        try:
            parse_clip_spec(bad)
            assert False, f"expected parse_clip_spec to reject {bad!r}"
        except ValueError:
            pass

    args_no_clip = argparse.Namespace(clip=None, start_time="00:01:00", end_time="00:02:30")
    assert resolve_clips(args_no_clip) == [("00:01:00", "00:02:30")]

    args_with_clips = argparse.Namespace(
        clip=["00:02:00-00:05:30", "00:20:00-00:21:30"],
        start_time="00:01:00", end_time="00:02:30")
    assert resolve_clips(args_with_clips) == [("00:02:00", "00:05:30"), ("00:20:00", "00:21:30")], \
        "when --clip is given, --start-time/--end-time must be ignored"

    print("_self_test_parse_clip_spec_and_resolve_clips: PASS")


def _self_test_model_settings_argv():
    """Pure, GPU-free argparse test: confirms model_settings_argv() only emits flags
    for settings that are actually ON/non-default (byte-for-byte unchanged bare argv
    for anyone passing none of them), AND -- the real point of this test -- that every
    flag name it emits is spelled exactly right by feeding the built argv into the
    REAL iw3.utils.create_parser() (not a stand-in), which would raise on any typo'd
    or unrecognized flag name."""
    from .utils import create_parser

    base_args = argparse.Namespace(
        depth_refine=False, depth_refine_strength=1.0,
        temporal_stabilize=False, temporal_stabilize_strength=0.7,
        temporal_stabilize_max_shift_velocity=None, temporal_stabilize_flat_region_boost=0.0,
        temporal_stabilize_edge_protection=0.0, temporal_stabilize_fast=False,
        temporal_stabilize_flow_engine="farneback",
        ema_normalize=False, ema_decay=0.75, ema_buffer=30, depth_aa=False)
    assert model_settings_argv(base_args) == [], "all-default settings must produce an empty argv"

    on_args = argparse.Namespace(
        depth_refine=True, depth_refine_strength=0.8,
        temporal_stabilize=True, temporal_stabilize_strength=0.5,
        temporal_stabilize_max_shift_velocity=12.0, temporal_stabilize_flat_region_boost=0.2,
        temporal_stabilize_edge_protection=0.3, temporal_stabilize_fast=True,
        temporal_stabilize_flow_engine="raft_fb",
        ema_normalize=True, ema_decay=0.9, ema_buffer=50, depth_aa=True)
    extra_argv = model_settings_argv(on_args)
    for expected in ("--depth-refine", "--depth-refine-strength", "--temporal-stabilize",
                      "--temporal-stabilize-strength", "--temporal-stabilize-max-shift-velocity",
                      "--temporal-stabilize-flat-region-boost", "--temporal-stabilize-edge-protection",
                      "--temporal-stabilize-fast", "--temporal-stabilize-flow-engine",
                      "--ema-normalize", "--ema-decay", "--ema-buffer", "--depth-aa"):
        assert expected in extra_argv, f"{expected!r} missing from {extra_argv}"

    # The real point: these flag names must be exactly what iw3.utils.create_parser()
    # (the REAL main parser, not a stand-in) actually recognizes.
    base_argv = ["-i", "in.mp4", "-o", "out", "--depth-model", "VDA_L"]
    parsed = create_parser(required_true=False).parse_args(base_argv + extra_argv)
    assert parsed.depth_refine is True
    assert parsed.depth_refine_strength == 0.8
    assert parsed.temporal_stabilize is True
    assert parsed.temporal_stabilize_strength == 0.5
    assert parsed.temporal_stabilize_max_shift_velocity == 12.0
    assert parsed.temporal_stabilize_flow_engine == "raft_fb"
    assert parsed.ema_normalize is True
    assert parsed.ema_decay == 0.9
    assert parsed.ema_buffer == 50
    assert parsed.depth_aa is True

    print("_self_test_model_settings_argv: PASS")


def _fake_export_pass_for_tests(input_path, model_name, pass_dir, start_time, end_time,
                                 gpu, resolution, limit_resolution, depth_model=None, extra_argv=None):
    """Shared fake export used by both calibrate() self-tests below -- writes real,
    correctly-shaped PNG files (so the real sampling/LUT code runs unmocked), without
    any GPU or real model. "source" model samples narrow around 0.5, "target" model
    samples wide around 0.5 -- same scenario depth_curve_match's own self-tests use.
    Returns a tiny stand-in object with a `.model` attribute so _release_model(...)
    (which only ever does `depth_model.model = None`) has something valid to call."""
    import types
    depth_dir = _pass_depth_dir(pass_dir, input_path)
    os.makedirs(depth_dir, exist_ok=True)
    rng = np.random.default_rng(0 if "source" in model_name else 1)
    scale = 0.08 if "source" in model_name else 0.30
    for i in range(6):
        value = float(np.clip(rng.normal(0.5, scale), 0.0, 1.0))
        _make_fake_depth_png(path.join(depth_dir, f"{i:06d}.png"), value)
    return depth_model or types.SimpleNamespace(model="fake", loaded=lambda: True,
                                                 move_to=lambda device: None)


def _self_test_calibrate_end_to_end_with_fake_passes():
    """Mocks ONLY the GPU-dependent export step (_run_export_pass) -- everything else
    (directory layout, pixel sampling, LUT building, saving) is the real code, proven
    against real fake-but-correctly-shaped PNG files on disk, same "mock the heavy step,
    test the real plumbing" pattern as sharpen_cli._self_test_run_gating. Single clip,
    same scenario as before multi-clip support existed."""
    import tempfile as _tempfile
    from unittest.mock import patch

    with _tempfile.TemporaryDirectory(prefix="iw3_curve_calib_selftest_") as tmpdir:
        input_path = path.join(tmpdir, "Clip.mkv")
        with open(input_path, "wb") as f:
            f.write(b"0")
        work_dir = path.join(tmpdir, "work")
        output_path = path.join(tmpdir, "curve.json")

        with patch(f"{__name__}._run_export_pass", side_effect=_fake_export_pass_for_tests):
            result_path = calibrate(
                input_path, "fake_source_model", "fake_target_model",
                [("00:00:00", "00:00:10")],
                pixels_per_frame=200, num_points=64,
                output_path=output_path, work_dir=work_dir, log=lambda msg: None)

        assert result_path == output_path
        assert path.exists(output_path)
        from .depth_curve_match import load_curve
        lut_x, lut_y, meta = load_curve(output_path)
        assert meta["source_model"] == "fake_source_model"
        assert meta["target_model"] == "fake_target_model"
        assert meta["frame_count"] == 6
        assert meta["clips"] == ["00:00:00-00:00:10"]
        x_spread = float(lut_x[-1] - lut_x[0])
        y_spread = float(lut_y[-1] - lut_y[0])
        assert y_spread > x_spread, (
            f"expected the 'target' (wide) side to have more spread than 'source' "
            f"(narrow): x_spread={x_spread:.4f} y_spread={y_spread:.4f}")
        # work_dir is cleaned up by default (own_work_dir branch) -- here work_dir was
        # explicitly passed, so calibrate() must NOT delete it (a caller-supplied
        # work_dir may be one the caller wants to inspect/reuse afterward).
        assert path.exists(work_dir)

    print("_self_test_calibrate_end_to_end_with_fake_passes: PASS")


def _self_test_calibrate_pools_multiple_movies():
    """ADR-480: clips from SEVERAL movie files pooled into one curve. parse_clip_line
    accepts "START-END" and "PATH | START-END" (file names with '-' in them, quoted
    paths); calibrate() exports each clip from ITS OWN file (both passes), pools all
    frames, keeps one model per side across files, and records every input in meta.
    A missing clip file is refused before any export."""
    import tempfile as _tempfile
    from unittest.mock import patch

    assert parse_clip_line("00:01:00-00:02:30") == (None, "00:01:00", "00:02:30")
    p, s0, e0 = parse_clip_line(r'"E:\Movies\A DTS-HD.MA 5.1-W32.mkv" | 00:10:00-00:11:30')
    assert p == r"E:\Movies\A DTS-HD.MA 5.1-W32.mkv" and (s0, e0) == ("00:10:00", "00:11:30"), p
    try:
        parse_clip_line(" | 00:01:00-00:02:00")
        assert False, "empty file before | must be refused"
    except ValueError:
        pass

    calls = []

    def _fake(input_path, model_name, pass_dir, start_time, end_time,
              gpu, resolution, limit_resolution, depth_model=None, extra_argv=None):
        calls.append((path.basename(input_path), model_name, start_time))
        return _fake_export_pass_for_tests(input_path, model_name, pass_dir, start_time, end_time,
                                           gpu, resolution, limit_resolution, depth_model=depth_model)

    with _tempfile.TemporaryDirectory(prefix="iw3_curve_calib_multimovie_") as tmpdir:
        main = path.join(tmpdir, "Live Action.mkv")
        other = path.join(tmpdir, "Cartoon-Movie.mkv")
        for f_ in (main, other):
            with open(f_, "wb") as fh:
                fh.write(b"0")
        clips = [("00:02:00", "00:03:00"), (other, "00:10:00", "00:11:00"), (other, "00:40:00", "00:41:00")]
        out = path.join(tmpdir, "curve.json")
        with patch(f"{__name__}._run_export_pass", side_effect=_fake):
            calibrate(main, "fake_source_model", "fake_target_model", clips,
                      pixels_per_frame=200, num_points=64, output_path=out,
                      work_dir=path.join(tmpdir, "work"), log=lambda msg: None)
        files_source = [c[0] for c in calls if "source" in c[1]]
        files_target = [c[0] for c in calls if "target" in c[1]]
        assert files_source == ["Live Action.mkv", "Cartoon-Movie.mkv", "Cartoon-Movie.mkv"], files_source
        assert files_target == files_source, files_target
        from .depth_curve_match import load_curve
        _, _, meta = load_curve(out)
        assert meta["frame_count"] == 18, meta["frame_count"]
        assert meta["inputs"] == ["Cartoon-Movie.mkv", "Live Action.mkv"], meta["inputs"]
        assert meta["clips"][0] == "00:02:00-00:03:00" and meta["clips"][1].startswith("Cartoon-Movie.mkv | "), meta["clips"]

        try:
            calibrate(main, "fake_source_model", "fake_target_model",
                      [(path.join(tmpdir, "missing.mkv"), "00:00:01", "00:00:02")],
                      output_path=out, work_dir=path.join(tmpdir, "w2"), log=lambda msg: None)
            assert False, "missing clip file must be refused"
        except ValueError:
            pass
    print("_self_test_calibrate_pools_multiple_movies: PASS")


def _self_test_calibrate_pools_multiple_clips():
    """The real reason multi-clip support exists: confirms calibrate() with several
    --clip ranges actually POOLS every clip's samples into the frame_count/meta
    (not just the last one), and that each model is loaded/reused across clips
    rather than re-created per clip (model identity stays the same object across
    calls for the same side)."""
    import tempfile as _tempfile
    from unittest.mock import patch

    seen_model_objs = {"source": [], "target": []}

    def _tracking_fake_export_pass(input_path, model_name, pass_dir, start_time, end_time,
                                    gpu, resolution, limit_resolution, depth_model=None, extra_argv=None):
        side = "source" if "source" in model_name else "target"
        result = _fake_export_pass_for_tests(input_path, model_name, pass_dir, start_time, end_time,
                                              gpu, resolution, limit_resolution, depth_model=depth_model)
        seen_model_objs[side].append(result)
        return result

    with _tempfile.TemporaryDirectory(prefix="iw3_curve_calib_selftest_") as tmpdir:
        input_path = path.join(tmpdir, "Clip.mkv")
        with open(input_path, "wb") as f:
            f.write(b"0")
        work_dir = path.join(tmpdir, "work")
        output_path = path.join(tmpdir, "curve.json")
        clips = [("00:02:00", "00:03:00"), ("00:20:00", "00:21:00"), ("01:00:00", "01:01:00")]

        with patch(f"{__name__}._run_export_pass", side_effect=_tracking_fake_export_pass):
            calibrate(
                input_path, "fake_source_model", "fake_target_model", clips,
                pixels_per_frame=200, num_points=64,
                output_path=output_path, work_dir=work_dir, log=lambda msg: None)

        from .depth_curve_match import load_curve
        _, _, meta = load_curve(output_path)
        # 6 fake frames per clip x 3 clips = 18 total, not just one clip's 6.
        assert meta["frame_count"] == 18, meta["frame_count"]
        assert meta["clips"] == ["00:02:00-00:03:00", "00:20:00-00:21:00", "01:00:00-01:01:00"]

        # Each side's model object is the SAME instance reused across all 3 of
        # that side's calls (first call gets depth_model=None and returns a fresh
        # object; every later call must be handed that exact object back in).
        assert len(seen_model_objs["source"]) == 3
        assert seen_model_objs["source"][0] is seen_model_objs["source"][1] is seen_model_objs["source"][2]
        assert len(seen_model_objs["target"]) == 3
        assert seen_model_objs["target"][0] is seen_model_objs["target"][1] is seen_model_objs["target"][2]
        assert seen_model_objs["source"][0] is not seen_model_objs["target"][0]

    print("_self_test_calibrate_pools_multiple_clips: PASS")


def _self_test_run_gating():
    """Synthetic test of run()'s pre-flight checks -- missing input, same source/target
    model, and a malformed --clip value -- proves each refuses before any export pass
    is attempted."""
    from unittest.mock import patch
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory(prefix="iw3_curve_calib_selftest_") as tmpdir:
        missing = path.join(tmpdir, "does_not_exist.mkv")
        existing = path.join(tmpdir, "clip.mkv")
        with open(existing, "wb") as f:
            f.write(b"0")

        def _args(**overrides):
            base = dict(input=missing, source_model="VDA_L", target_model="Any_V3_Metric_Large",
                        clip=None, start_time="00:00:00", end_time="00:00:10", gpu=0, resolution=None,
                        limit_resolution=False, pixels_per_frame=100, num_points=64,
                        output=None, work_dir=None)
            base.update(overrides)
            return argparse.Namespace(**base)

        with patch(f"{__name__}.calibrate") as mock_calibrate:
            rc = run(_args())
            assert rc == 1, rc
            mock_calibrate.assert_not_called()

            rc = run(_args(input=existing, target_model="VDA_L"))  # same as source_model
            assert rc == 1, rc
            mock_calibrate.assert_not_called()

            # malformed --clip -- refused before any export pass is attempted
            rc = run(_args(input=existing, clip=["totally garbled no dash at all"]))
            assert rc == 1, rc
            mock_calibrate.assert_not_called()
            rc = run(_args(input=existing, clip=["00:02:00-"]))  # dash present but no end
            assert rc == 1, rc
            mock_calibrate.assert_not_called()

            mock_calibrate.return_value = "curve.json"
            rc = run(_args(input=existing))
            assert rc == 0, rc
            mock_calibrate.assert_called_once()
            assert mock_calibrate.call_args[0][3] == [("00:00:00", "00:00:10")]

    print("_self_test_run_gating: PASS")


def _self_test_target_reset_look():
    """ADR-488: only the TARGET pass gets --scene-detect; the source pass keeps the
    real-job settings; meta records it; same model allowed only with the option."""
    import tempfile as _tf
    from . import depth_curve_calibrate_cli as M
    seen = []
    def _fake(input_path, model_name, pass_dir, start_time, end_time, gpu, resolution,
              limit_resolution, depth_model=None, extra_argv=None):
        seen.append((model_name, list(extra_argv or [])))
        d = _pass_depth_dir(pass_dir, input_path); os.makedirs(d, exist_ok=True)
        v = 0.3 if "--scene-detect" in (extra_argv or []) else 0.5
        for k in range(3):
            _make_fake_depth_png(path.join(d, f"{k:06d}.png"), v + 0.1 * k)
        return object()
    orig, orig_rel = M._run_export_pass, M._release_model
    M._run_export_pass, M._release_model = _fake, (lambda m: None)
    try:
        with _tf.TemporaryDirectory() as td:
            vid = path.join(td, "movie.mkv"); open(vid, "wb").close()
            out = path.join(td, "curve.json")
            M.calibrate(vid, "VDA_L", "VDA_L", [("00:00:01", "00:00:02")], output_path=out,
                        work_dir=path.join(td, "w"), log=lambda *_: None,
                        extra_argv=["--ema-normalize", "--ema-decay", "0.99"], target_reset_look=True)
            import json
            meta = json.load(open(out, encoding="utf-8"))["meta"]
            assert meta["target_reset_look"] is True, meta
    finally:
        M._run_export_pass, M._release_model = orig, orig_rel
    assert seen[0] == ("VDA_L", ["--ema-normalize", "--ema-decay", "0.99"]), seen
    assert seen[1] == ("VDA_L", ["--ema-normalize", "--ema-decay", "0.99", "--scene-detect"]), seen
    p = create_parser()
    a = p.parse_args(["-i", "x.mkv", "--source-model", "VDA_L", "--target-model", "VDA_L"])
    assert run(a) == 1, "same model without the option must still be refused (missing file also refuses)"
    a = p.parse_args(["-i", "x.mkv", "--source-model", "VDA_L", "--target-model", "VDA_L", "--target-reset-look"])
    assert a.target_reset_look is True
    print("_self_test_target_reset_look: PASS")


def _run_self_tests():
    _self_test_pass_depth_dir_layout()
    _self_test_sample_pixels_from_dir()
    _self_test_parse_clip_spec_and_resolve_clips()
    _self_test_model_settings_argv()
    _self_test_target_reset_look()
    _self_test_calibrate_end_to_end_with_fake_passes()
    _self_test_calibrate_pools_multiple_clips()
    _self_test_calibrate_pools_multiple_movies()
    _self_test_run_gating()
    print("All depth_curve_calibrate_cli self-tests PASSED")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
