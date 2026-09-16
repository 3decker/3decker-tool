"""Regression test for the mask_mlbw periodic-banding fix (docs/ai/AI_DECISIONS.md
ADR-160): mlbw_l2_inpaint could produce a large, washed-out, vertically-banded
inpaint patch specifically with the MoGe3 depth models. Root-caused via a live,
real-content reproduction (Hocus Pocus 1993 UHD graveyard scene) and direct FFT
analysis of the raw mask_mlbw hole-mask logits: MLBW's mask head processes the
image via pixel_unshuffle(downscaling_factor=(1, 8)) + windowed attention (see
iw3/models/mlbw.py), and over large, very flat/low-variance depth regions its
output becomes dominated by its own learned per-window positional bias, leaking
through as a periodic ~8px-period vertical banding artifact in the predicted hole
logits -- a real architectural quirk of the trained mask_mlbw checkpoint itself
(confirmed present, to a lesser extent, with Any_V3_Mono_01 depth too on the same
real frame), which MoGe3's unusually large, smooth flat regions trigger far more
extensively than other depth models. A tiny blur or added noise on the depth input
did NOT remove it (tested directly against the real checkpoint); a narrow FFT notch
filter at exactly 1/8 cycles/px does, because real hole shapes are low-frequency
broad silhouettes while this artifact sits at one sharp, content-independent
frequency.

Synthetic/isolated (CS-TEST-001): no GPU, no real movie file, no downloaded model.
Tests iw3.dilation.remove_periodic_banding directly, plus a synthetic call through
iw3.backward_warp.postprocess_hole_mask.

Run directly: python tests/test_iw3_mask_mlbw_periodic_banding.py
"""
import sys
from os import path

import math
import torch

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from iw3.dilation import remove_periodic_banding  # noqa: E402
from iw3.backward_warp import postprocess_hole_mask  # noqa: E402


def _periodic_signal(period=8.0, w=256, h=8, amplitude=5.0):
    x = torch.arange(w, dtype=torch.float32)
    row = amplitude * torch.sin(2 * math.pi * x / period)
    return row.view(1, 1, 1, w).repeat(1, 1, h, 1)


def _test_remove_periodic_banding_suppresses_target_period():
    banding = _periodic_signal(period=8.0, w=256, amplitude=5.0)
    filtered = remove_periodic_banding(banding, period=8.0)
    assert filtered.shape == banding.shape

    # power at the target frequency must drop sharply (the whole point of the notch)
    freqs = torch.fft.rfftfreq(256, d=1.0)
    target_bin = int(torch.argmin(torch.abs(freqs - 1.0 / 8.0)))
    before = torch.abs(torch.fft.rfft(banding, dim=-1))[0, 0, 0, target_bin].item()
    after = torch.abs(torch.fft.rfft(filtered, dim=-1))[0, 0, 0, target_bin].item()
    assert after < before * 0.05, f"target-frequency power not suppressed: before={before:.3f} after={after:.3f}"

    print("_test_remove_periodic_banding_suppresses_target_period: PASS")


def _test_remove_periodic_banding_preserves_broad_shape():
    # a real hole is a broad, low-frequency blob -- must survive the notch close
    # to untouched, since the notch only targets the narrow band around 1/8 c/px.
    w = 256
    x = torch.arange(w, dtype=torch.float32)
    blob = 10.0 * torch.exp(-((x - 128) ** 2) / (2 * 40.0 ** 2))
    blob = blob.view(1, 1, 1, w).repeat(1, 1, 8, 1)

    filtered = remove_periodic_banding(blob, period=8.0)
    max_abs_diff = (filtered - blob).abs().max().item()
    assert max_abs_diff < 0.5, f"broad low-frequency hole shape was distorted: max_abs_diff={max_abs_diff:.3f}"

    print("_test_remove_periodic_banding_preserves_broad_shape: PASS")


def _test_remove_periodic_banding_noop_on_tiny_width():
    x = torch.randn(1, 1, 4, 8)
    out = remove_periodic_banding(x, period=8.0)
    assert torch.equal(out, x), "widths below the FFT's usable minimum must pass through unchanged"

    print("_test_remove_periodic_banding_noop_on_tiny_width: PASS")


def _test_remove_periodic_banding_dtype_roundtrip():
    x = torch.randn(1, 1, 8, 64, dtype=torch.float16)
    out = remove_periodic_banding(x, period=8.0)
    assert out.dtype == torch.float16, "must return the input's original dtype (fp16 under autocast)"

    print("_test_remove_periodic_banding_dtype_roundtrip: PASS")


def _test_postprocess_hole_mask_with_banding_still_valid():
    # synthetic hole_mask_logits: a real broad hole blob + injected 8px banding,
    # same shape convention apply_divergence_nn_delta_weight's caller uses (BCHW).
    w, h = 128, 96
    x = torch.arange(w, dtype=torch.float32)
    blob = 6.0 * torch.exp(-((x - 64) ** 2) / (2 * 20.0 ** 2))
    banding = 4.0 * torch.sin(2 * math.pi * x / 8.0)
    logits = (blob + banding).view(1, 1, 1, w).repeat(1, 1, h, 1)

    mask = postprocess_hole_mask(logits, target_size=(h, w), threshold=0.15)
    assert mask.shape == (1, 1, h, w)
    assert mask.dtype == torch.bool

    # the real blob region (near center) should still be flagged as a hole;
    # the notch filter must not have erased genuine, broad hole content.
    assert mask[0, 0, h // 2, w // 2].item() is True

    print("_test_postprocess_hole_mask_with_banding_still_valid: PASS")


def main():
    _test_remove_periodic_banding_suppresses_target_period()
    _test_remove_periodic_banding_preserves_broad_shape()
    _test_remove_periodic_banding_noop_on_tiny_width()
    _test_remove_periodic_banding_dtype_roundtrip()
    _test_postprocess_hole_mask_with_banding_still_valid()
    print("ALL PASS")


if __name__ == "__main__":
    main()
