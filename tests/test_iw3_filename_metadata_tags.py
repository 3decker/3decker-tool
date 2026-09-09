"""Regression test for docs/ai/AI_DECISIONS.md ADR-050 (tonight's filename/comment
metadata tag audit, ADR-029 through ADR-048 plus the RIFE multiplier/target-fps work
tracked in code as ADR-049).

Synthetic/isolated (CS-TEST-001): builds a real args Namespace via
iw3.utils.create_parser(required_true=False).parse_args([]) -- no GPU, no real video,
no network -- and exercises:

  1. The one real gap ADR-050 found and closed: the stereo-aware 4K/8K waifu2x
     upscale path (_run_waifu2x_upscale_stereo, ADR-040) produced the exact same
     "<name>_w2x<ext>" derivative filename as the plain whole-frame upscale path
     (_run_waifu2x_upscale), even though _build_iw3_comment_metadata's own comment
     claims "the derivative file's own '_w2x' suffix already marks it" -- that claim
     was false until this fix. subprocess.run and path.exists are mocked so this
     never shells out to a real waifu2x process.

  2. RIFE's --rife-multiplier/--rife-target-fps filename tag (_rife3x / _rife60fps)
     and comment metadata fields (iw3_rife_multiplier / iw3_rife_target_fps) --
     found already correctly implemented during the audit (landed concurrently with
     this session's other work, referenced in code as ADR-049), verified here as a
     regression guard rather than re-implemented.

  3. --pause-frees-vram (ADR-038) deliberately produces NO filename/comment tag --
     it is pure runtime GPU-memory behavior during processing and never changes the
     actual output content, so it is correctly excluded from this tagging scheme.

Run directly: python tests/test_iw3_filename_metadata_tags.py (from the nunif/ dir,
matching this project's other tests/ path conventions), or import and call main().
"""
import os
import sys
from os import path
from unittest.mock import patch

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from iw3.utils import (  # noqa: E402
    create_parser, make_output_filename, _build_iw3_comment_metadata,
    _run_waifu2x_upscale, _run_waifu2x_upscale_stereo,
)


def _base_args(**overrides):
    args = create_parser(required_true=False).parse_args([])
    args.metadata = "filename"
    args.video_extension = ".mkv"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _test_waifu2x_stereo_vs_plain_derivative_filename():
    output_path = "movie_Any_S_row_flow_d20_c05_ipd0_crf20.mkv"

    # Plain whole-frame path (waifu2x_upscale_target left at "auto", the default) --
    # unchanged "<name>_w2x<ext>" behavior, must NOT be touched by this fix.
    args_plain = _base_args(waifu2x_upscale=True, waifu2x_upscale_target="auto")
    with patch("iw3.utils._invoke_waifu2x_cli", return_value=True) as mock_invoke:
        result_plain = _run_waifu2x_upscale(output_path, args_plain)
    assert result_plain == "movie_Any_S_row_flow_d20_c05_ipd0_crf20_w2x.mkv", result_plain
    assert mock_invoke.called

    # Stereo-aware 4K path -- must now produce a DIFFERENT, distinguishable filename
    # from the plain path above (this is the actual bug ADR-050 fixes).
    args_4k = _base_args(waifu2x_upscale=True, waifu2x_upscale_target="4k")
    with patch("iw3.utils.subprocess.run") as mock_run, \
         patch("iw3.utils.path.exists", return_value=True):
        result_4k = _run_waifu2x_upscale_stereo(output_path, args_4k)
    assert result_4k == "movie_Any_S_row_flow_d20_c05_ipd0_crf20_w2x4k.mkv", result_4k
    assert mock_run.called

    # Stereo-aware 8K path -- also distinguishable from both the plain path and 4K.
    args_8k = _base_args(waifu2x_upscale=True, waifu2x_upscale_target="8k")
    with patch("iw3.utils.subprocess.run") as mock_run, \
         patch("iw3.utils.path.exists", return_value=True):
        result_8k = _run_waifu2x_upscale_stereo(output_path, args_8k)
    assert result_8k == "movie_Any_S_row_flow_d20_c05_ipd0_crf20_w2x8k.mkv", result_8k

    assert result_plain != result_4k != result_8k != result_plain, \
        "plain/4k/8k waifu2x upscale derivative filenames must all be distinguishable"

    print("_test_waifu2x_stereo_vs_plain_derivative_filename: PASS")


def _test_rife_multiplier_target_fps_tags():
    # Default: rife_interpolate on, no multiplier/target-fps given -> no extra suffix
    # beyond the plain "_rife" tag (matches the original hardcoded-2x behavior).
    args_default = _base_args(rife_interpolate=True)
    name = make_output_filename("in.mkv", args_default, video=True)
    assert "_rife" in name and "_rife3x" not in name and "fps" not in name, name
    comment = _build_iw3_comment_metadata(args_default, video=True)
    assert "iw3_rife_interpolate=1" in comment
    assert "iw3_rife_multiplier=" not in comment
    assert "iw3_rife_target_fps=" not in comment

    # Explicit multiplier=2 (== the default) must also add nothing extra.
    args_m2 = _base_args(rife_interpolate=True, rife_multiplier=2)
    name_m2 = make_output_filename("in.mkv", args_m2, video=True)
    assert name_m2 == name, (name_m2, name)

    # Non-default multiplier=3 -> "_rife3x" in filename, iw3_rife_multiplier=3 in comment.
    args_m3 = _base_args(rife_interpolate=True, rife_multiplier=3)
    name_m3 = make_output_filename("in.mkv", args_m3, video=True)
    assert "_rife3x" in name_m3, name_m3
    comment_m3 = _build_iw3_comment_metadata(args_m3, video=True)
    assert "iw3_rife_multiplier=3" in comment_m3, comment_m3
    assert "iw3_rife_target_fps=" not in comment_m3

    # target-fps=60.0 -> "_rife60fps" in filename, iw3_rife_target_fps=60.0 in comment,
    # and target-fps wins over multiplier when both happen to be set on the Namespace.
    args_fps = _base_args(rife_interpolate=True, rife_target_fps=60.0, rife_multiplier=3)
    name_fps = make_output_filename("in.mkv", args_fps, video=True)
    assert "_rife60fps" in name_fps, name_fps
    assert "3x" not in name_fps, name_fps
    comment_fps = _build_iw3_comment_metadata(args_fps, video=True)
    assert "iw3_rife_target_fps=60.0" in comment_fps, comment_fps
    assert "iw3_rife_multiplier=" not in comment_fps

    print("_test_rife_multiplier_target_fps_tags: PASS")


def _test_pause_frees_vram_excluded_from_tags():
    args_off = _base_args()
    args_on = _base_args(pause_frees_vram=True)
    name_off = make_output_filename("in.mkv", args_off, video=True)
    name_on = make_output_filename("in.mkv", args_on, video=True)
    assert name_off == name_on, \
        "--pause-frees-vram is pure runtime behavior and must never affect the filename"
    comment_off = _build_iw3_comment_metadata(args_off, video=True)
    comment_on = _build_iw3_comment_metadata(args_on, video=True)
    assert comment_off == comment_on, \
        "--pause-frees-vram is pure runtime behavior and must never affect comment metadata"
    assert "pause_frees_vram" not in comment_on

    print("_test_pause_frees_vram_excluded_from_tags: PASS")


def main():
    _test_waifu2x_stereo_vs_plain_derivative_filename()
    _test_rife_multiplier_target_fps_tags()
    _test_pause_frees_vram_excluded_from_tags()
    print("ALL PASS")


if __name__ == "__main__":
    main()
