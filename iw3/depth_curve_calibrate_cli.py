"""python -m iw3.depth_curve_calibrate_cli -- builds the Depth Curve Match LUT
(see iw3/depth_curve_match.py) that lets the main conversion pipeline reshape one
depth model's output to resemble another model's distinctive value distribution,
without running that second model on every real job.

This is a genuinely standalone tool, invoked as its own subprocess (same "separate
subprocess, never sharing state with the main app" convention as iw3.rife_cli /
iw3.sharpen_cli -- see docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001), run ONCE
(or occasionally, if decker wants to re-calibrate against different content) --
not part of every normal conversion job.

What it actually does: runs a SHORT sample clip (not the whole movie -- a minute
or two is plenty, see --start-time/--end-time) through BOTH models, one at a time
(same sequential-load-then-release VRAM discipline as depth_blend.py's Dual-Pass
Depth Blend -- the two models are never resident at once), using iw3's own
existing --export --export-depth-only machinery (export_main in iw3/utils.py) --
no new video-reading code. It then samples a modest number of random pixels per
exported frame from each model's depth output (full-frame, not a full pixel dump
-- plenty for a stable percentile curve, see --pixels-per-frame), builds a
histogram-specification LUT (iw3.depth_curve_match.build_curve_lut) mapping the
SOURCE model's typical values onto the TARGET model's typical values at the same
percentile rank, and saves it to --output (a small JSON file, default
iw3.depth_curve_match.DEFAULT_CURVE_PATH) for the main pipeline's
DepthCurveMatcher to load automatically on every future job.

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

from .depth_curve_match import build_curve_lut, save_curve, DEFAULT_CURVE_PATH


def _pass_depth_dir(output_dir, input_path):
    """Mirrors export_video's own output layout exactly (iw3/utils.py
    export_video): <output_dir>/<basename stripped>/depth/*.png (u16 PNGs).
    Duplicated here rather than imported, same "each standalone CLI tool reads
    iw3.utils's own layout rather than depending on a private helper" convention
    already used by sharpen_cli.py/stereo_mode_tag_cli.py for filename tags."""
    basename = path.splitext(path.basename(input_path))[0].strip()
    return path.join(output_dir, basename, "depth")


def _run_export_pass(input_path, model_name, pass_dir, start_time, end_time,
                      gpu, resolution, limit_resolution):
    """Runs one depth-only export pass with one model, using iw3's own real
    argument parser + export pipeline (the same `create_parser() ->
    set_state_args() -> export_main()` sequence iw3.cli.main() uses for a normal
    job -- see CLAUDE.md's Architecture section) rather than hand-building a
    partial args object. Fully releases the model before returning, same
    discipline as depth_blend.py's sequential pass_a/pass_b export."""
    from .utils import create_parser, set_state_args, export_main, gc_collect

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

    pass_args = create_parser().parse_args(argv)
    pass_args = set_state_args(pass_args)
    depth_model = pass_args.state["depth_model"]
    depth_model.load(gpu=pass_args.gpu, resolution=pass_args.resolution,
                      limit_resolution=pass_args.limit_resolution)
    export_main(pass_args)
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


def calibrate(input_path, source_model, target_model, start_time, end_time,
              gpu=0, resolution=None, limit_resolution=False,
              pixels_per_frame=4000, num_points=512, output_path=None,
              work_dir=None, log=print):
    """Full calibration run: export both models over the same short clip, sample
    pixels from each, build the LUT, save it. Returns the saved file path."""
    output_path = output_path or DEFAULT_CURVE_PATH
    own_work_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="iw3_depth_curve_calibrate_")
    try:
        pass_a_dir = path.join(work_dir, "pass_source")
        pass_b_dir = path.join(work_dir, "pass_target")

        log(f"[depth-curve-calibrate] [1/3] exporting depth with source model "
            f"({source_model}) over {start_time or 'start'}-{end_time or 'end'}...")
        _run_export_pass(input_path, source_model, pass_a_dir, start_time, end_time,
                          gpu, resolution, limit_resolution)

        log(f"[depth-curve-calibrate] [2/3] exporting depth with target model "
            f"({target_model}) over the same clip...")
        _run_export_pass(input_path, target_model, pass_b_dir, start_time, end_time,
                          gpu, resolution, limit_resolution)

        source_depth_dir = _pass_depth_dir(pass_a_dir, input_path)
        target_depth_dir = _pass_depth_dir(pass_b_dir, input_path)

        source_values, source_frame_count = _sample_pixels_from_dir(
            source_depth_dir, pixels_per_frame)
        target_values, target_frame_count = _sample_pixels_from_dir(
            target_depth_dir, pixels_per_frame)

        if source_frame_count != target_frame_count:
            raise RuntimeError(
                f"source model exported {source_frame_count} frame(s) but target model "
                f"exported {target_frame_count} -- the two passes must cover the exact "
                f"same frame range to build a valid curve. This usually means the clip's "
                f"start/end time landed on a different frame count for each model (rare), "
                f"or one pass was interrupted. Try a different --start-time/--end-time.")

        log(f"[depth-curve-calibrate] [3/3] building curve from {source_frame_count} "
            f"sampled frame(s) ({len(source_values)} pixels each side)...")
        lut_x, lut_y = build_curve_lut(source_values, target_values, num_points=num_points)

        meta = {
            "source_model": source_model,
            "target_model": target_model,
            "input": path.basename(input_path),
            "start_time": start_time,
            "end_time": end_time,
            "frame_count": source_frame_count,
            "pixels_per_frame": pixels_per_frame,
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_curve(output_path, lut_x, lut_y, meta=meta)
        log(f"[depth-curve-calibrate] done -- wrote {output_path} "
            f"({source_model} -> {target_model}, {source_frame_count} frames)")
        return output_path
    finally:
        if own_work_dir and path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.depth_curve_calibrate_cli",
        description=(
            "Build the Depth Curve Match LUT: runs a short sample clip through both "
            "--source-model and --target-model, and saves a lookup table that lets the "
            "main pipeline reshape --source-model's depth output to resemble "
            "--target-model's distinctive distribution, without running --target-model on "
            "every real job. Run this once (or whenever you want to re-calibrate against "
            "different content); normal conversions just load the saved result."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                         help="Path to a representative source video to calibrate from. "
                              "A short, visually varied clip (some near content, some far "
                              "background) gives a better curve than a static shot.")
    parser.add_argument("--source-model", type=str, required=True,
                         help="The depth model you actually use for real jobs (e.g. VDA_L) "
                              "-- the one whose output will be reshaped at conversion time.")
    parser.add_argument("--target-model", type=str, required=True,
                         help="The depth model whose distinctive look you want to borrow "
                              "(e.g. Any_V3_Metric_Large) -- never actually run during real "
                              "jobs after this, only here during calibration.")
    parser.add_argument("--start-time", type=str, default="00:01:00",
                         help="Start of the sample clip (HH:MM:SS). Default skips the first "
                              "minute (often titles/black frames).")
    parser.add_argument("--end-time", type=str, default="00:02:30",
                         help="End of the sample clip (HH:MM:SS). Keep this short -- 60-120 "
                              "seconds of real, varied content is plenty; this runs twice "
                              "(once per model), not once per whole movie.")
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
    parser.add_argument("--num-points", type=int, default=512,
                         help="Resolution of the saved lookup table.")
    parser.add_argument("--output", "-o", type=str, default=None,
                         help=f"Where to save the curve file. Default: {DEFAULT_CURVE_PATH} "
                              f"(the same path the main pipeline auto-loads from).")
    parser.add_argument("--work-dir", type=str, default=None,
                         help="Temp directory for the two export passes. Default: a fresh "
                              "system temp directory, deleted automatically when done.")
    return parser


def run(args):
    if not path.exists(args.input):
        print(f"ERROR: --input file does not exist: {args.input}", file=sys.stderr)
        return 1
    if args.source_model == args.target_model:
        print("ERROR: --source-model and --target-model must be different -- there is "
              "nothing to reshape toward if they're the same model.", file=sys.stderr)
        return 1
    try:
        calibrate(
            args.input, args.source_model, args.target_model,
            args.start_time, args.end_time,
            gpu=args.gpu, resolution=args.resolution, limit_resolution=args.limit_resolution,
            pixels_per_frame=args.pixels_per_frame, num_points=args.num_points,
            output_path=args.output, work_dir=args.work_dir,
            log=lambda msg: print(msg, file=sys.stderr),
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


def _self_test_calibrate_end_to_end_with_fake_passes():
    """Mocks ONLY the GPU-dependent export step (_run_export_pass) -- everything else
    (directory layout, pixel sampling, LUT building, saving) is the real code, proven
    against real fake-but-correctly-shaped PNG files on disk, same "mock the heavy step,
    test the real plumbing" pattern as sharpen_cli._self_test_run_gating."""
    import tempfile as _tempfile
    from unittest.mock import patch

    with _tempfile.TemporaryDirectory(prefix="iw3_curve_calib_selftest_") as tmpdir:
        input_path = path.join(tmpdir, "Clip.mkv")
        with open(input_path, "wb") as f:
            f.write(b"0")
        work_dir = path.join(tmpdir, "work")
        output_path = path.join(tmpdir, "curve.json")

        def _fake_export_pass(input_path, model_name, pass_dir, start_time, end_time,
                               gpu, resolution, limit_resolution):
            depth_dir = _pass_depth_dir(pass_dir, input_path)
            os.makedirs(depth_dir, exist_ok=True)
            # "source" model (narrow spread) vs "target" model (wide spread) --
            # same scenario depth_curve_match's own self-tests use.
            rng = np.random.default_rng(0 if "source" in model_name else 1)
            scale = 0.08 if "source" in model_name else 0.30
            for i in range(6):
                value = float(np.clip(rng.normal(0.5, scale), 0.0, 1.0))
                _make_fake_depth_png(path.join(depth_dir, f"{i:06d}.png"), value)

        with patch(f"{__name__}._run_export_pass", side_effect=_fake_export_pass):
            result_path = calibrate(
                input_path, "fake_source_model", "fake_target_model",
                "00:00:00", "00:00:10",
                pixels_per_frame=200, num_points=64,
                output_path=output_path, work_dir=work_dir, log=lambda msg: None)

        assert result_path == output_path
        assert path.exists(output_path)
        from .depth_curve_match import load_curve
        lut_x, lut_y, meta = load_curve(output_path)
        assert meta["source_model"] == "fake_source_model"
        assert meta["target_model"] == "fake_target_model"
        assert meta["frame_count"] == 6
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


def _self_test_run_gating():
    """Synthetic test of run()'s pre-flight checks -- missing input, same source/target
    model -- proves both refuse before any export pass is attempted."""
    from unittest.mock import patch
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory(prefix="iw3_curve_calib_selftest_") as tmpdir:
        missing = path.join(tmpdir, "does_not_exist.mkv")
        existing = path.join(tmpdir, "clip.mkv")
        with open(existing, "wb") as f:
            f.write(b"0")

        def _args(**overrides):
            base = dict(input=missing, source_model="VDA_L", target_model="Any_V3_Metric_Large",
                        start_time="00:00:00", end_time="00:00:10", gpu=0, resolution=None,
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

            mock_calibrate.return_value = "curve.json"
            rc = run(_args(input=existing))
            assert rc == 0, rc
            mock_calibrate.assert_called_once()

    print("_self_test_run_gating: PASS")


def _run_self_tests():
    _self_test_pass_depth_dir_layout()
    _self_test_sample_pixels_from_dir()
    _self_test_calibrate_end_to_end_with_fake_passes()
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
