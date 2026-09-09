"""Regression test for the new post-render Sharpen filter (docs/ai/AI_DECISIONS.md
ADR-061): an unsharp-mask-style detail/sharpness enhancement pass on the FINISHED,
fully-rendered stereo picture -- the one thing missing from iw3's existing sharpness-
adjacent controls (Depth Resolution, Depth Detail Refinement, Dual-Pass Depth Blend,
depth-model choice, waifu2x upscaling), all of which work on the depth map or on
upscaling, never on the delivered RGB picture itself.

Synthetic/isolated (CS-TEST-001): no GPU, no real movie file, no downloaded model.

  1. Unit tests directly against iw3.depth_effects.apply_sharpen/_local_detail_weight
     with synthetic tensors: exact no-op at strength=0.0 (`is` identity check), None
     eyes handled safely, the detail-aware weight is measurably higher at a real
     synthetic edge than in a flat region, a flat region with added per-pixel random
     noise (simulating film grain) is left close to untouched while a real edge is
     visibly sharpened, strength scales the effect, and output is always clipped
     back into the pipeline's [0, 1] tensor range even at strength=1.0 on a hard
     0.0/1.0 edge.

  2. Integration test against the REAL, unmocked apply_divergence()/make_output_
     filename()/_build_iw3_comment_metadata() in iw3/utils.py via create_parser(),
     using the grid_sample method (needs no downloaded model, same choice ADR-021's
     own Edge Repair integration test made) -- confirms the real hook point (after
     Edge Repair, inside apply_divergence()'s common return path), the real
     --sharpen/--sharpen-strength flags, and the real "only tagged when strength is
     non-zero" filename/comment metadata convention all work end-to-end.

Run directly: python tests/test_iw3_sharpen.py (from the nunif/ dir, matching this
project's other tests/ path conventions), or import and call main().
"""
import sys
from os import path

import torch

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

import iw3.depth_effects as DE  # noqa: E402
from iw3.utils import (  # noqa: E402
    create_parser, apply_divergence, make_output_filename, _build_iw3_comment_metadata,
)


# Excludes the outermost few rows, which pick up the blur/Sobel convolutions' own
# (expected) zero-padding boundary response at the image's top/bottom edge --
# unrelated to the synthetic column regions these tests measure.
ROW_CORE = slice(4, -4)


def _base_args(**overrides):
    args = create_parser(required_true=False).parse_args([])
    args.metadata = "filename"
    args.video_extension = ".mkv"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _edge_and_flat_noisy_image(h=32, w=120, seed=0):
    """(1, 3, h, w) synthetic RGB tensor in [0, 1] with three well-separated regions
    (each block boundary is >=10px from the "core" slice used to measure it, safely
    outside the ~2px receptive field of the blur+Sobel weighting, so one region's
    real edge never contaminates another region's "flat" measurement):
      cols [0:30)  a flat 0.5 gray region with small per-pixel random noise added
                   (stand-in for film grain) -- core measurement slice [5:25).
      cols [30:70) perfectly flat/constant, no noise at all -- core slice [40:60).
      cols [70:120) a hard vertical step edge at column 95 (0.2 -> 0.8) -- core
                   slice [93:98), tightly straddling the real transition (a
                   region-wide mean would otherwise be diluted by the many flat
                   columns on either side of the actual 1-pixel-wide transition).
    Returns (img, edge_core_slice, noisy_core_slice, clean_core_slice)."""
    g = torch.Generator().manual_seed(seed)
    img = torch.zeros(1, 3, h, w)
    noise = (torch.rand(1, 3, h, 30, generator=g) - 0.5) * 0.06
    img[:, :, :, 0:30] = 0.5 + noise
    img[:, :, :, 30:70] = 0.5
    img[:, :, :, 70:95] = 0.2
    img[:, :, :, 95:120] = 0.8
    return img.clamp(0.0, 1.0), slice(93, 98), slice(5, 25), slice(40, 60)


def _test_apply_sharpen_noop_at_zero_strength():
    left = torch.rand(1, 3, 16, 16)
    right = torch.rand(1, 3, 16, 16)
    out_left, out_right = DE.apply_sharpen(left, right, strength=0.0)
    assert out_left is left and out_right is right, \
        "strength=0.0 must be an exact no-op (identity), not just numerically equal"

    out_left2, out_right2 = DE.apply_sharpen(left, right, strength=-1.0)
    assert out_left2 is left and out_right2 is right, "negative strength must also be a no-op"

    print("_test_apply_sharpen_noop_at_zero_strength: PASS")


def _test_apply_sharpen_none_eyes_handled():
    assert DE.apply_sharpen(None, None, strength=0.5) == (None, None)
    left = torch.rand(1, 3, 8, 8)
    assert DE.apply_sharpen(left, None, strength=0.5) == (left, None)
    assert DE.apply_sharpen(None, left, strength=0.5) == (None, left)

    print("_test_apply_sharpen_none_eyes_handled: PASS")


def _test_local_detail_weight_edge_vs_flat():
    img, edge_sl, noisy_sl, clean_sl = _edge_and_flat_noisy_image()
    blur_kernel = DE._BLUR3.view(1, 1, 3, 3)
    weight = DE._local_detail_weight(img, blur_kernel, detail_percentile=95.0)
    assert weight.shape == (1, 1, img.shape[2], img.shape[3])

    # Rows within a few pixels of the top/bottom image border pick up their own
    # (expected, harmless) zero-padding boundary response from the blur/Sobel
    # convolutions -- excluded here (ROW_CORE) so this measures each COLUMN
    # region's real content, not an unrelated top/bottom-edge artifact.
    edge_weight = weight[:, :, ROW_CORE, edge_sl].mean().item()
    clean_flat_weight = weight[:, :, ROW_CORE, clean_sl].mean().item()
    noisy_flat_weight = weight[:, :, ROW_CORE, noisy_sl].mean().item()

    assert clean_flat_weight < 1e-4, f"perfectly flat region must get ~zero detail weight, got {clean_flat_weight}"
    assert edge_weight > 0.3, f"a real hard edge must get a substantial detail weight, got {edge_weight}"
    assert edge_weight > noisy_flat_weight * 3, \
        f"edge weight ({edge_weight}) must be clearly higher than grain-noise weight ({noisy_flat_weight})"

    print("_test_local_detail_weight_edge_vs_flat: PASS")


def _test_apply_sharpen_edge_vs_flat_region():
    img, edge_sl, noisy_sl, clean_sl = _edge_and_flat_noisy_image()
    out, _ = DE.apply_sharpen(img.clone(), img.clone(), strength=1.0)
    diff = (out - img).abs()

    edge_change = diff[:, :, ROW_CORE, edge_sl].mean().item()
    clean_flat_change = diff[:, :, ROW_CORE, clean_sl].mean().item()
    noisy_flat_change = diff[:, :, ROW_CORE, noisy_sl].mean().item()

    assert edge_change > 0.02, f"a real edge must be visibly sharpened, got mean change {edge_change}"
    assert clean_flat_change < 1e-5, \
        f"a perfectly flat region must be left essentially untouched, got {clean_flat_change}"
    assert edge_change > noisy_flat_change * 3, \
        (f"sharpening at a real edge ({edge_change}) must clearly exceed sharpening in a "
         f"grain-noisy flat region ({noisy_flat_change}) -- this is the noise-amplification guard")

    print("_test_apply_sharpen_edge_vs_flat_region: PASS")


def _test_apply_sharpen_strength_scales_effect():
    img, edge_sl, _, _ = _edge_and_flat_noisy_image()
    out_low, _ = DE.apply_sharpen(img.clone(), img.clone(), strength=0.25)
    out_high, _ = DE.apply_sharpen(img.clone(), img.clone(), strength=1.0)

    change_low = (out_low - img).abs()[:, :, :, edge_sl].mean().item()
    change_high = (out_high - img).abs()[:, :, :, edge_sl].mean().item()
    assert change_high > change_low, \
        f"higher strength must produce a larger effect at a real edge ({change_high} vs {change_low})"

    print("_test_apply_sharpen_strength_scales_effect: PASS")


def _test_apply_sharpen_clips_to_valid_range():
    img = torch.zeros(1, 3, 16, 16)
    img[:, :, :, 8:] = 1.0  # hard 0.0/1.0 edge, worst case for overshoot
    out, _ = DE.apply_sharpen(img.clone(), img.clone(), strength=1.0)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0, \
        f"sharpened output must stay clipped to [0, 1], got [{out.min().item()}, {out.max().item()}]"

    print("_test_apply_sharpen_clips_to_valid_range: PASS")


def _test_apply_divergence_real_hook_and_order():
    torch.manual_seed(0)
    im, _, _, _ = _edge_and_flat_noisy_image(h=32)
    depth = torch.rand(1, 1, im.shape[2], im.shape[3])

    def _run(**overrides):
        args = _base_args(method="grid_sample", mapper="none", **overrides)
        args.state = {"convergence_model": None}
        left, right = apply_divergence(depth.clone(), im.clone(), args, side_model=None)
        return left, right

    left_off, right_off = _run(sharpen=False)
    left_on, right_on = _run(sharpen=True, sharpen_strength=1.0)

    assert not torch.equal(left_off, left_on), \
        "--sharpen must actually change the rendered output through the real apply_divergence() hook"
    assert not torch.equal(right_off, right_on)

    # strength=0.0 with --sharpen set must behave the same as --sharpen not set at all
    # (the gate + magnitude computation in apply_divergence() must treat both as off).
    left_zero, right_zero = _run(sharpen=True, sharpen_strength=0.0)
    assert torch.equal(left_off, left_zero) and torch.equal(right_off, right_zero)

    print("_test_apply_divergence_real_hook_and_order: PASS")


def _test_sharpen_filename_and_comment_tags():
    args_off = _base_args()
    args_on = _base_args(sharpen=True, sharpen_strength=0.5)
    args_on_default_strength = _base_args(sharpen=True)  # default sharpen_strength=0.5

    name_off = make_output_filename("in.mkv", args_off, video=True)
    name_on = make_output_filename("in.mkv", args_on, video=True)
    assert "_sharp" not in name_off, name_off
    assert "_sharp50" in name_on, name_on

    name_on_default = make_output_filename("in.mkv", args_on_default_strength, video=True)
    assert name_on_default == name_on, "checkbox-on with the default 0.5 strength must match explicit 0.5"

    comment_off = _build_iw3_comment_metadata(args_off, video=True)
    comment_on = _build_iw3_comment_metadata(args_on, video=True)
    assert "iw3_sharpen_strength=" not in comment_off
    assert "iw3_sharpen_strength=0.5" in comment_on, comment_on

    # --sharpen set but strength explicitly 0.0 must tag nothing (only tag when
    # strength is non-zero, matching this project's established convention).
    args_zero = _base_args(sharpen=True, sharpen_strength=0.0)
    name_zero = make_output_filename("in.mkv", args_zero, video=True)
    comment_zero = _build_iw3_comment_metadata(args_zero, video=True)
    assert "_sharp" not in name_zero, name_zero
    assert "iw3_sharpen_strength=" not in comment_zero

    print("_test_sharpen_filename_and_comment_tags: PASS")


def main():
    _test_apply_sharpen_noop_at_zero_strength()
    _test_apply_sharpen_none_eyes_handled()
    _test_local_detail_weight_edge_vs_flat()
    _test_apply_sharpen_edge_vs_flat_region()
    _test_apply_sharpen_strength_scales_effect()
    _test_apply_sharpen_clips_to_valid_range()
    _test_apply_divergence_real_hook_and_order()
    _test_sharpen_filename_and_comment_tags()
    print("ALL PASS")


if __name__ == "__main__":
    main()
