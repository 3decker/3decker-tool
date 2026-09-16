"""Regression test for the Metric3D clamp-outlier fix (docs/ai/AI_DECISIONS.md
ADR-161): Metric3D's decode head hard-clamps its own regressed depth to a fixed,
architectural [min_val, max_val] range (e.g. (0.1, 200) meters for the ViT-RAFT5
family, confirmed by reading RAFTDepthNormalDPTDecoder5.py directly). Pure-black,
featureless content (a movie's own letterbox bars -- present in most real BluRay
sources) makes the model saturate to exactly that bound there, and -- confirmed
live on a real frame from the user's own footage -- the ViT's receptive field
blends that signal into a smooth multi-row DECAY in nearby real-content rows too
(200 -> 98 -> 37 -> ... -> 8 over 7 rows in one real measured case), not a clean
single-value jump. Because this project's depth normalization (depth_scaler.py's
EMAMinMaxScaler) always stretches across the frame's TRUE min/max with zero
outlier protection, that one small contaminated region was enough on its own to
crush 99%+ of the real depth detail for the whole frame into a near-blank result.

Fix: iw3.metric3d_model._desaturate_clamp_outliers(), called from batch_infer's
_run() before the resize step (resizing first would bilinear-blend the exact
outlier value into a wider band, defeating detection). Purely statistical --
does not use min_val/max_val in its check -- finds a robust "normal" range via
kthvalue at a small tail fraction, and only winsorizes the frame's TRUE extreme
in when it sits dramatically farther beyond that normal range than the normal
range's own width. A real frame's true min/max sit close to its own percentile
range by construction, so this is a no-op for ordinary content.

Synthetic/isolated (CS-TEST-001): no GPU, no real movie file, no downloaded model.

Run directly: python tests/test_iw3_metric3d_clamp_outlier_desaturation.py
"""
import sys
from os import path

import torch

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from iw3.metric3d_model import _desaturate_clamp_outliers  # noqa: E402


def _well_behaved_frame(h=64, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    # real-ish canonical-space depth: smaller=nearer, roughly 1.3-8.0 range,
    # matching the real graveyard-frame measurement this fix was built against.
    return (torch.rand(1, 1, h, w, generator=g) * (8.0 - 1.3) + 1.3)


def _test_noop_on_well_behaved_frame():
    frame = _well_behaved_frame()
    before = frame.clone()
    out = _desaturate_clamp_outliers(frame, min_val=0.1, max_val=200.0)
    assert torch.equal(out, before), "must not alter a frame with no dramatic outlier tail"

    print("_test_noop_on_well_behaved_frame: PASS")


def _test_clamps_isolated_high_outlier():
    frame = _well_behaved_frame()
    frame[0, 0, 0, 0] = 200.0  # exact clamp-saturated pixel, real letterbox case
    out = _desaturate_clamp_outliers(frame, min_val=0.1, max_val=200.0)
    assert out[0, 0, 0, 0].item() < 10.0, \
        f"isolated clamp-saturated pixel must be winsorized back toward the normal range, got {out[0, 0, 0, 0].item()}"
    # every other pixel must be provably untouched
    assert torch.equal(out[0, 0, 1:, :], frame[0, 0, 1:, :])

    print("_test_clamps_isolated_high_outlier: PASS")


def _test_clamps_decaying_halo_not_just_exact_value():
    # the real bug this fix targets: a smooth multi-row decay from the clamp
    # bound down to normal, not a single exact value -- an earlier, naive
    # "exact value match" version of this fix missed this and left the frame
    # still badly blown out (confirmed live: min stayed at -111 instead of ~-8
    # in the real equivalent case). None of these decay values are the exact
    # clamp bound, so a detector that only matches min_val/max_val exactly must
    # fail this test; the statistical approach must not.
    #
    # Matches the real proportions measured live (not "the whole row"): the
    # real contamination was confined to a narrow width-strip within each
    # decay row (~2% of row width), not the full row -- confirmed via
    # locate_outlier.py against the real checkpoint (extreme pixels: 654 out
    # of 8.3M, well under 1% of the frame total even though spread over 8
    # rows). A test contaminating the FULL width of every decay row would
    # itself be an unrealistically large fraction and isn't what this
    # function is designed to protect against (see
    # _test_preserves_genuine_large_fraction_saturation for that boundary).
    frame = _well_behaved_frame(h=64, w=200)
    decay = [200.0, 98.0, 37.0, 31.0, 25.0, 21.0, 14.0]
    for row, value in enumerate(decay):
        frame[0, 0, row, 0:4] = value  # narrow strip, ~2% of this row's width
    out = _desaturate_clamp_outliers(frame, min_val=0.1, max_val=200.0)
    for row, orig_value in enumerate(decay):
        assert out[0, 0, row, 0:4].max().item() < orig_value, \
            f"row {row} (orig {orig_value}) was not pulled down toward the normal range"
    # the well-behaved rows below the halo must be untouched
    assert torch.equal(out[0, 0, len(decay):, :], frame[0, 0, len(decay):, :])

    print("_test_clamps_decaying_halo_not_just_exact_value: PASS")


def _test_preserves_genuine_large_fraction_saturation():
    # a real all-sky or all-macro-closeup shot could legitimately saturate a
    # LARGE fraction of the frame at/near the clamp bound -- must never be
    # discarded just because it's an extreme value; only a small, dramatically-
    # isolated MINORITY of pixels should ever trigger winsorization.
    frame = torch.full((1, 1, 32, 32), 200.0)
    frame[0, 0, 28:, :] = torch.rand(4, 32) * (8.0 - 1.3) + 1.3  # small "ground" strip
    before = frame.clone()
    out = _desaturate_clamp_outliers(frame, min_val=0.1, max_val=200.0)
    assert torch.equal(out, before), \
        "a genuinely large fraction of the frame at the extreme must be left untouched"

    print("_test_preserves_genuine_large_fraction_saturation: PASS")


def _test_handles_low_side_outlier_symmetrically():
    # Uses a tight cluster (not _well_behaved_frame's wide uniform spread) --
    # real depth content clusters much more tightly than "uniform across the
    # model's whole theoretical range" (the real Giant2 case measured std=1.8
    # around a mean of -3.5, a narrow cluster relative to the 0.1-200 clamp
    # range), so a genuinely isolated outlier reads as dramatically far from
    # the normal cluster's own width, same as the real case.
    torch.manual_seed(2)
    frame = torch.rand(1, 1, 64, 64) * 0.5 + 3.0  # tight cluster: [3.0, 3.5]
    frame[0, 0, 5, 5] = 0.1  # exact near-clamp-saturated pixel
    out = _desaturate_clamp_outliers(frame, min_val=0.1, max_val=200.0)
    assert out[0, 0, 5, 5].item() > 1.0, \
        f"isolated near-clamp-saturated pixel must be winsorized toward the normal range too, got {out[0, 0, 5, 5].item()}"

    print("_test_handles_low_side_outlier_symmetrically: PASS")


def _test_real_graveyard_case_shape():
    # a scaled-down reconstruction of the real, live-measured graveyard-frame
    # case that motivated this fix: min went from -200.0 (broken) to -8.14
    # (matching the real content boundary) after this exact function ran, in
    # the real end-to-end test against the live Metric3D_ViT_Giant2 checkpoint.
    torch.manual_seed(1)
    frame = torch.rand(1, 1, 100, 200) * (8.0 - 1.3) + 1.3
    decay = [200.0, 98.0, 37.0, 31.0, 25.0, 21.0, 14.0]
    for row, value in enumerate(decay):
        frame[0, 0, row, 0:4] = value  # narrow strip, matching real measured proportions
    out = _desaturate_clamp_outliers(frame, min_val=0.1, max_val=200.0)
    assert out.max().item() < 10.0, f"expected the whole contaminated halo gone, got max={out.max().item()}"
    assert out.min().item() >= 1.0, f"unexpected low-side change, got min={out.min().item()}"

    print("_test_real_graveyard_case_shape: PASS")


def main():
    _test_noop_on_well_behaved_frame()
    _test_clamps_isolated_high_outlier()
    _test_clamps_decaying_halo_not_just_exact_value()
    _test_preserves_genuine_large_fraction_saturation()
    _test_handles_low_side_outlier_symmetrically()
    _test_real_graveyard_case_shape()
    print("ALL PASS")


if __name__ == "__main__":
    main()
