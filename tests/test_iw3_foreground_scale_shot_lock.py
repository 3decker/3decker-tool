"""Regression test for docs/ai/AI_DECISIONS.md ADR-452 ("shot_lock" for
--foreground-scale, a completely separate feature/state from Convergence Plane's
own shot_lock mode).

Synthetic/isolated (CS-TEST-001): no GPU, no real video, no network, no torch
tensors needed for ForegroundScaleEstimator itself (it is a pure, stateless
function of a single float -- see iw3/foreground_scale_estimator.py's module
docstring for why no averaging/settle state is needed here, unlike
ShotLockEstimator). Filename/comment-metadata wiring is exercised the same way
test_iw3_filename_metadata_tags.py exercises every other optional tag: a real
args Namespace via iw3.utils.create_parser(required_true=False).parse_args([]).

Run directly: python tests/test_iw3_foreground_scale_shot_lock.py (from the
nunif/ dir), or import and call main().
"""
import sys
import tempfile
from os import path
from unittest.mock import MagicMock

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from iw3.foreground_scale_estimator import ForegroundScaleEstimator  # noqa: E402
from iw3.utils import (  # noqa: E402
    create_parser, make_output_filename, _build_iw3_comment_metadata, apply_divergence,
    reset_foreground_scale_scene_log, finish_foreground_scale_scene_log,
    reset_convergence_scene_log, finish_convergence_scene_log,
)


def _base_args(**overrides):
    args = create_parser(required_true=False).parse_args([])
    args.metadata = "filename"
    args.video_extension = ".mkv"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _test_classify_2way():
    # No threshold_low/mid_value set -> plain 2-way split at threshold_high.
    est = ForegroundScaleEstimator(wide_value=2.0, closeup_value=-1.0, threshold_high=4.0)
    assert est._classify(5.0) == 2.0   # above high -> wide
    assert est._classify(4.0) == -1.0  # exactly at high (not strictly greater) -> closeup
    assert est._classify(0.5) == -1.0  # well below -> closeup
    assert est._classify(None) == -1.0  # missing divergence -> closeup fallback
    print("_test_classify_2way: PASS")


def _test_classify_3way():
    est = ForegroundScaleEstimator(wide_value=2.0, closeup_value=-1.0, threshold_high=4.0,
                                   mid_value=0.5, threshold_low=2.0)
    assert est._classify(5.0) == 2.0   # above high -> wide
    assert est._classify(3.0) == 0.5   # between low and high -> mid
    assert est._classify(2.0) == -1.0  # at or below low -> closeup
    assert est._classify(1.0) == -1.0  # well below low -> closeup
    print("_test_classify_3way: PASS")


def _test_3way_with_mid_value_none_falls_back_to_closeup():
    # threshold_low set but mid_value left None: band between the two thresholds
    # still falls back to closeup_value (per the spec's explicit fallback rule),
    # not a crash or a silent "mid" of None.
    est = ForegroundScaleEstimator(wide_value=2.0, closeup_value=-1.0, threshold_high=4.0,
                                   mid_value=None, threshold_low=2.0)
    assert est._classify(3.0) == -1.0
    print("_test_3way_with_mid_value_none_falls_back_to_closeup: PASS")


def _test_call_returns_cached_mapper_function():
    est = ForegroundScaleEstimator(wide_value=1.0, closeup_value=-1.0, threshold_high=4.0,
                                   metric_depth=False, mapper_type=None)
    fn_wide_a = est(divergence=10.0)
    fn_wide_b = est(divergence=9.0)  # different divergence, same classified value (1.0)
    assert fn_wide_a is fn_wide_b, "same classified value must reuse the cached mapper fn"
    assert len(est._mapper_fn_for_cache) == 1

    fn_closeup = est(divergence=0.0)
    assert fn_closeup is not fn_wide_a
    assert len(est._mapper_fn_for_cache) == 2

    import torch
    x = torch.tensor([0.0, 0.5, 1.0])
    # sanity: the two mapper functions actually behave differently (not identical
    # no-op mappers) for a non-trivial foreground_scale value.
    assert not torch.equal(fn_wide_a(x), fn_closeup(x))
    print("_test_call_returns_cached_mapper_function: PASS")


def _test_filename_and_comment_tags_independent_of_convergence_lock():
    # constant mode (default): no "fsl" tag, no shot_lock comment fields at all.
    args_constant = _base_args()
    name_constant = make_output_filename("in.mkv", args_constant, video=True)
    assert "fsl" not in name_constant, name_constant
    comment_constant = _build_iw3_comment_metadata(args_constant, video=True)
    assert "iw3_foreground_scale_mode" not in comment_constant

    # shot_lock mode: filename carries the "fsl" tag (never colliding with the
    # existing plain "fs<value>" constant-mode tag), comment carries every field,
    # Mid omitted (left None/empty).
    args_shot_lock = _base_args(
        foreground_scale_mode="shot_lock",
        foreground_scale_lock_wide=1.5,
        foreground_scale_lock_closeup=-0.5,
        foreground_scale_lock_threshold_high=4.0,
    )
    name_shot_lock = make_output_filename("in.mkv", args_shot_lock, video=True)
    assert "_fsllockW150C-50_sigdiv400" in name_shot_lock, name_shot_lock
    assert "loM" not in name_shot_lock

    comment_shot_lock = _build_iw3_comment_metadata(args_shot_lock, video=True)
    assert "iw3_foreground_scale_mode=shot_lock" in comment_shot_lock
    assert "iw3_foreground_scale_lock_wide=1.5" in comment_shot_lock
    assert "iw3_foreground_scale_lock_closeup=-0.5" in comment_shot_lock
    assert "iw3_foreground_scale_lock_threshold_high=4.0" in comment_shot_lock
    assert "iw3_foreground_scale_lock_mid" not in comment_shot_lock
    assert "iw3_foreground_scale_lock_threshold_low" not in comment_shot_lock

    # Mid + Low both set -> both appear, in both filename and comment.
    args_3way = _base_args(
        foreground_scale_mode="shot_lock",
        foreground_scale_lock_wide=1.5,
        foreground_scale_lock_mid=0.25,
        foreground_scale_lock_closeup=-0.5,
        foreground_scale_lock_threshold_high=4.0,
        foreground_scale_lock_threshold_low=2.0,
    )
    name_3way = make_output_filename("in.mkv", args_3way, video=True)
    assert "M25" in name_3way, name_3way
    assert "lo200" in name_3way, name_3way
    comment_3way = _build_iw3_comment_metadata(args_3way, video=True)
    assert "iw3_foreground_scale_lock_mid=0.25" in comment_3way
    assert "iw3_foreground_scale_lock_threshold_low=2.0" in comment_3way

    # Running Convergence Plane's OWN shot_lock at the same time must leave its
    # tags/fields completely independent -- the real point of ADR-452 (separate
    # thresholds/values, by explicit user choice).
    args_both = _base_args(
        convergence_mode="shot_lock",
        convergence_lock_wide=0.75,
        convergence_lock_closeup=0.20,
        foreground_scale_mode="shot_lock",
        foreground_scale_lock_wide=1.5,
        foreground_scale_lock_closeup=-0.5,
        foreground_scale_lock_threshold_high=4.0,
    )
    name_both = make_output_filename("in.mkv", args_both, video=True)
    assert "lockW75" in name_both and "C20" in name_both, name_both  # convergence's own tag
    assert "_fsllockW150" in name_both, name_both  # foreground-scale's own, separate tag
    comment_both = _build_iw3_comment_metadata(args_both, video=True)
    assert "iw3_convergence_lock_wide=0.75" in comment_both
    assert "iw3_foreground_scale_lock_wide=1.5" in comment_both

    print("_test_filename_and_comment_tags_independent_of_convergence_lock: PASS")


def _test_apply_divergence_resolves_mapper_fn_once_per_call():
    # End-to-end, CPU-only (method="grid_sample" needs no GPU/side_model, no
    # inpaint buffering): verifies apply_divergence() resolves mapper_fn exactly
    # ONCE per call via the shared foreground_scale_model lookup, and reuses it --
    # not a separate get_mapper()/fs_model() per call site -- across all three of
    # its branches (convergence_model None; convergence_model set, non-shot_lock;
    # convergence_model set, shot_lock).
    import torch

    H, W = 8, 8
    depth = torch.rand(2, 1, H, W)
    im = torch.rand(2, 3, H, W)
    real_fs = ForegroundScaleEstimator(wide_value=2.0, closeup_value=-2.0, threshold_high=4.0)

    def make_args(convergence_mode):
        args = create_parser(required_true=False).parse_args([])
        args.method = "grid_sample"
        args.divergence = 5.0  # > threshold_high=4.0 -> classifies to wide_value
        args.convergence_mode = convergence_mode
        args.mapper = "none"  # unused once foreground_scale_model is set; harmless otherwise
        fs_model = MagicMock(side_effect=real_fs)
        return args, fs_model

    # Branch 1: args.state["convergence_model"] is None.
    args1, fs1 = make_args("constant")
    args1.state = {"convergence_model": None, "foreground_scale_model": fs1}
    left1, right1 = apply_divergence(depth.clone(), im.clone(), args1, side_model=None)
    assert left1 is not None and right1 is not None
    assert fs1.call_count == 1, fs1.call_count

    # Branch 2: convergence_model set, non-shot_lock mode (mapper_fn also applied
    # to `convergence`, per the existing, unchanged "other modes" comment/invariant).
    def dummy_conv_other(im_, depth_, reset_pts=None):
        return torch.full((2, 1, 1, 1), 0.5)

    args2, fs2 = make_args("depth_percentile")
    args2.state = {"convergence_model": dummy_conv_other, "foreground_scale_model": fs2}
    left2, right2 = apply_divergence(depth.clone(), im.clone(), args2, side_model=None)
    assert left2 is not None and right2 is not None
    assert fs2.call_count == 1, fs2.call_count

    # Branch 3: convergence_model set, shot_lock mode (ADR-423: convergence stays
    # unmapped here -- only depth uses mapper_fn).
    def dummy_conv_shot_lock(im_, depth_, reset_pts=None, log_reset_pts=None, divergence=None):
        return torch.full((2, 1, 1, 1), 0.5)

    args3, fs3 = make_args("shot_lock")
    args3.state = {"convergence_model": dummy_conv_shot_lock, "foreground_scale_model": fs3}
    left3, right3 = apply_divergence(depth.clone(), im.clone(), args3, side_model=None)
    assert left3 is not None and right3 is not None
    assert fs3.call_count == 1, fs3.call_count

    print("_test_apply_divergence_resolves_mapper_fn_once_per_call: PASS")


def _test_apply_divergence_mapper_matches_classified_foreground_scale():
    # Verifies the actual numeric effect: apply_divergence() with
    # foreground_scale_model set must warp depth using the SAME mapper a direct
    # ForegroundScaleEstimator classification + get_mapper/resolve_mapper_name
    # call would pick for that divergence -- not args.mapper (which is "none"/
    # identity here, so any mismatch would be visible as a wrong output).
    import torch
    from iw3.mapper import resolve_mapper_name, get_mapper
    from iw3.backward_warp import apply_divergence_grid_sample

    H, W = 8, 8
    torch.manual_seed(0)
    depth = torch.rand(1, 1, H, W)
    im = torch.rand(1, 3, H, W)

    args = create_parser(required_true=False).parse_args([])
    args.method = "grid_sample"
    args.divergence = 5.0  # > threshold_high=4.0 -> wide_value=2.0
    args.convergence_mode = "constant"
    args.mapper = "none"
    fs_model = ForegroundScaleEstimator(wide_value=2.0, closeup_value=-2.0, threshold_high=4.0)
    args.state = {"convergence_model": None, "foreground_scale_model": fs_model}

    left, right = apply_divergence(depth.clone(), im.clone(), args, side_model=None)

    expected_mapper_name = resolve_mapper_name(mapper=None, foreground_scale=2.0,
                                               metric_depth=False, mapper_type=None)
    expected_mapper_fn = get_mapper(expected_mapper_name)
    expected_depth = expected_mapper_fn(depth.clone())
    expected_left, expected_right = apply_divergence_grid_sample(
        im.clone(), expected_depth, args.divergence, convergence=args.convergence,
        synthetic_view=args.synthetic_view)

    assert torch.allclose(left, expected_left), "wide_value=2.0 classification was not applied"
    assert torch.allclose(right, expected_right), "wide_value=2.0 classification was not applied"

    # Sanity: NOT the identity mapper (args.mapper="none") -- i.e. the feature
    # really did override the mapper, this isn't passing by coincidence.
    identity_left, identity_right = apply_divergence_grid_sample(
        im.clone(), depth.clone(), args.divergence, convergence=args.convergence,
        synthetic_view=args.synthetic_view)
    assert not torch.allclose(left, identity_left), \
        "output must differ from the plain identity mapper for a non-trivial foreground_scale"

    print("_test_apply_divergence_mapper_matches_classified_foreground_scale: PASS")


def _make_fsl_log_args(tmpdir, **overrides):
    args = create_parser(required_true=False).parse_args([])
    args.method = "grid_sample"
    args.mapper = "none"
    args.metadata = "filename"
    args.video_extension = ".mkv"
    for k, v in overrides.items():
        setattr(args, k, v)
    input_path = path.join(tmpdir, "in.mp4")
    output_path = path.join(tmpdir, "out.mp4")
    return args, input_path, output_path


def _run_two_scenes(args, fs_model, divergence_a, divergence_b):
    """Two apply_divergence() calls, each a 2-frame batch with a cut after the
    2nd frame -- the minimum shape needed to produce two separate scene rows."""
    import torch

    args.state = {"convergence_model": None, "foreground_scale_model": fs_model}
    depth = torch.rand(2, 1, 4, 4)
    im = torch.rand(2, 3, 4, 4)

    args.divergence = divergence_a
    left1, right1 = apply_divergence(depth.clone(), im.clone(), args, side_model=None,
                                     reset_pts=[False, True], log_reset_pts=[False, True])
    assert left1 is not None

    args.divergence = divergence_b
    left2, right2 = apply_divergence(depth.clone(), im.clone(), args, side_model=None,
                                     reset_pts=[False, True], log_reset_pts=[False, True])
    assert left2 is not None


def _test_flag_defaults_false():
    args = create_parser(required_true=False).parse_args([])
    assert args.foreground_scale_scene_log is False
    reparsed = create_parser(required_true=False).parse_args(
        ["-i", "a.mp4", "-o", "o", "--foreground-scale-scene-log"])
    assert reparsed.foreground_scale_scene_log is True
    print("_test_flag_defaults_false: PASS")


def _test_scene_log_suffix_and_no_collision_with_convergence_log():
    # Both logs enabled at once (the additive/independence requirement) must
    # produce two SEPARATE files with distinct suffixes, neither clobbering
    # the other.
    with tempfile.TemporaryDirectory() as tmpdir:
        args, input_path, output_path = _make_fsl_log_args(
            tmpdir,
            foreground_scale_mode="constant",
            foreground_scale=0.5,
            foreground_scale_scene_log=True,
            convergence_mode="constant",
            convergence_scene_log=True,
        )
        reset_foreground_scale_scene_log(args, input_path=input_path, output_path=output_path)
        reset_convergence_scene_log(args, input_path=input_path, output_path=output_path)
        _run_two_scenes(args, fs_model=None, divergence_a=5.0, divergence_b=1.0)
        finish_foreground_scale_scene_log(args, input_path, output_path)
        finish_convergence_scene_log(args, input_path, output_path)

        fsl_path = path.join(tmpdir, "out_foreground_scale_log.txt")
        conv_path = path.join(tmpdir, "out_convergence_log.txt")
        assert path.exists(fsl_path), "Foreground Scale scene log must be written"
        # Convergence's own log never writes in "constant" mode (no tensor to log --
        # see convergence_scene_log_active in apply_divergence()); this just confirms
        # the two logs' paths/suffixes stay independent, not that both always write.
        assert not path.exists(conv_path)
        assert fsl_path != conv_path
        print("_test_scene_log_suffix_and_no_collision_with_convergence_log: PASS")


def _test_shot_lock_rows_tagged_wide_and_closeup():
    with tempfile.TemporaryDirectory() as tmpdir:
        args, input_path, output_path = _make_fsl_log_args(
            tmpdir,
            foreground_scale_mode="shot_lock",
            foreground_scale=0.0,
            foreground_scale_scene_log=True,
            foreground_scale_lock_wide=2.0,
            foreground_scale_lock_mid=None,
            foreground_scale_lock_closeup=-2.0,
            foreground_scale_lock_threshold_high=4.0,
            foreground_scale_lock_threshold_low=None,
        )
        fs_model = ForegroundScaleEstimator(wide_value=2.0, closeup_value=-2.0, threshold_high=4.0)
        reset_foreground_scale_scene_log(args, input_path=input_path, output_path=output_path)
        _run_two_scenes(args, fs_model=fs_model, divergence_a=5.0, divergence_b=1.0)
        finish_foreground_scale_scene_log(args, input_path, output_path)

        log_path = path.join(tmpdir, "out_foreground_scale_log.txt")
        assert path.exists(log_path)
        with open(log_path, encoding="utf-8") as f:
            text = f.read()
        assert "Foreground Scale log --" in text
        assert "shot_lock mode" in text
        lines = [ln for ln in text.splitlines() if ln.startswith("Scene ")]
        assert len(lines) == 2, lines
        assert "Foreground Scale: 2.00" in lines[0] and "[Wide]" in lines[0], lines[0]
        assert "Foreground Scale: -2.00" in lines[1] and "[Close-Up]" in lines[1], lines[1]
        assert "Shot classification" in text
        print("_test_shot_lock_rows_tagged_wide_and_closeup: PASS")


def _test_constant_mode_rows_have_no_tag():
    with tempfile.TemporaryDirectory() as tmpdir:
        args, input_path, output_path = _make_fsl_log_args(
            tmpdir,
            foreground_scale_mode="constant",
            foreground_scale=0.75,
            foreground_scale_scene_log=True,
        )
        reset_foreground_scale_scene_log(args, input_path=input_path, output_path=output_path)
        # args.state["foreground_scale_model"] stays None in constant mode (see
        # set_state_args()/iw3_main()) -- divergence value is irrelevant here.
        _run_two_scenes(args, fs_model=None, divergence_a=5.0, divergence_b=1.0)
        finish_foreground_scale_scene_log(args, input_path, output_path)

        log_path = path.join(tmpdir, "out_foreground_scale_log.txt")
        assert path.exists(log_path)
        with open(log_path, encoding="utf-8") as f:
            text = f.read()
        assert "constant mode, value 0.75" in text
        lines = [ln for ln in text.splitlines() if ln.startswith("Scene ")]
        assert len(lines) == 2, lines
        for ln in lines:
            assert "Foreground Scale: 0.75" in ln, ln
            assert "[Wide]" not in ln and "[Mid]" not in ln and "[Close-Up]" not in ln, ln
        assert "Shot classification" not in text
        print("_test_constant_mode_rows_have_no_tag: PASS")


def _test_flag_off_writes_no_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        args, input_path, output_path = _make_fsl_log_args(
            tmpdir,
            foreground_scale_mode="constant",
            foreground_scale=0.5,
            foreground_scale_scene_log=False,
        )
        reset_foreground_scale_scene_log(args, input_path=input_path, output_path=output_path)
        _run_two_scenes(args, fs_model=None, divergence_a=5.0, divergence_b=1.0)
        finish_foreground_scale_scene_log(args, input_path, output_path)

        log_path = path.join(tmpdir, "out_foreground_scale_log.txt")
        assert not path.exists(log_path), "flag off must never write a log file"
        print("_test_flag_off_writes_no_file: PASS")


def main():
    _test_classify_2way()
    _test_classify_3way()
    _test_3way_with_mid_value_none_falls_back_to_closeup()
    _test_call_returns_cached_mapper_function()
    _test_filename_and_comment_tags_independent_of_convergence_lock()
    _test_apply_divergence_resolves_mapper_fn_once_per_call()
    _test_apply_divergence_mapper_matches_classified_foreground_scale()
    _test_flag_defaults_false()
    _test_scene_log_suffix_and_no_collision_with_convergence_log()
    _test_shot_lock_rows_tagged_wide_and_closeup()
    _test_constant_mode_rows_have_no_tag()
    _test_flag_off_writes_no_file()
    print("ALL PASS")


if __name__ == "__main__":
    main()
