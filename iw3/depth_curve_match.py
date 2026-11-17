"""Depth Curve Match -- reshapes one depth model's per-frame value DISTRIBUTION to
resemble another model's distinctive shape, without running that second model on
every actual frame.

Real motivation (see docs/ai/AI_DECISIONS.md ADR-467 for the research this is
based on): decker switched his main depth model to VDA_L for its genuinely better
frame-to-frame stability, but found VDA_L's depth noticeably flatter/less "punchy"
than Any_V3_Metric_Large's distinctive big near/far separation. Existing options
didn't solve this:
  - Foreground/Midground/Background Pop (depth_effects.py) only stretches VDA_L's
    OWN existing values within one zone -- it can't manufacture separation that
    isn't there, and decker found it subtle/artifact-prone when pushed hard.
  - Dual-Pass Depth Blend (depth_blend.py) can transplant Any_V3's real values, but
    needs a full second depth-model pass over the ENTIRE video every single job --
    heavy on both time and disk.

Mechanism: "histogram specification" (classic image-processing technique --
https://en.wikipedia.org/wiki/Histogram_matching -- applied here to depth instead
of pixel brightness). A LUT built ONCE (see depth_curve_calibrate_cli.py) from a
short sample clip run through BOTH models records, at every percentile rank,
"what value did the SOURCE model (VDA_L) give here" and "what value did the TARGET
model (Any_V3) give at that same rank." At conversion time, every frame's real
depth value is looked up in that table and remapped toward the TARGET model's
typical value at that same rank -- the same numeric shape transplant Dual-Pass
Depth Blend does, but using a small pre-built table instead of a live second
model, so it costs one cheap interpolation per pixel instead of a second full
inference pass.

ADR-467 also investigated and ruled out the Lang et al. 2010 "Nonlinear Disparity
Mapping for Stereoscopic 3D" paper -- real academic research, no maintained
implementation anywhere, and built for a different pipeline (stereo-view warping)
entirely -- so this file does not attempt to implement it.
"""
import json
import os
from os import path

import numpy as np
import torch


DEFAULT_CURVE_PATH = path.join(path.dirname(__file__), "..", "tmp", "depth_curve_match.json")


def build_curve_lut(source_values, target_values, num_points=512):
    """Builds a monotonic lookup table mapping SOURCE model depth values onto
    TARGET model depth values at the same percentile rank ("histogram
    specification" -- see module docstring). source_values/target_values are flat
    arrays of raw depth samples, already normalized to each model's own 0..1 the
    same way the live pipeline already does.

    Returns (lut_x, lut_y) as float32 numpy arrays of length num_points: lut_x are
    evenly spaced percentile VALUES from the source distribution, lut_y are the
    target distribution's values at those same percentile ranks. Both are strictly
    increasing (see the nudge loop below) so torch.searchsorted/np.interp can walk
    them safely. Apply with DepthCurveMatcher.apply for a live torch tensor."""
    source_values = np.asarray(source_values, dtype=np.float32).ravel()
    target_values = np.asarray(target_values, dtype=np.float32).ravel()
    percentiles = np.linspace(0.0, 100.0, num_points)
    lut_x = np.clip(np.percentile(source_values, percentiles).astype(np.float32), 0.0, 1.0)
    lut_y = np.clip(np.percentile(target_values, percentiles).astype(np.float32), 0.0, 1.0)
    # np.percentile over a sorted set of quantiles is already non-decreasing, but a
    # flat run of identical values (e.g. a frame that's mostly one flat depth, or a
    # degenerate test array) can produce EQUAL neighbors -- nudge those up by a tiny
    # epsilon so lut_x is always strictly increasing, which torch.searchsorted needs
    # and which also guarantees the matched output below is itself monotonic (never
    # flips the real near/far order of two different input values).
    for i in range(1, len(lut_x)):
        if lut_x[i] <= lut_x[i - 1]:
            lut_x[i] = lut_x[i - 1] + 1e-6
    return lut_x, lut_y


def save_curve(file_path, lut_x, lut_y, meta=None):
    """Atomic write (temp file + os.replace), same convention as every other
    on-disk state file in this project (CS-IO-001)."""
    data = {
        "lut_x": np.asarray(lut_x, dtype=np.float32).tolist(),
        "lut_y": np.asarray(lut_y, dtype=np.float32).tolist(),
        "meta": meta or {},
    }
    os.makedirs(path.dirname(file_path) or ".", exist_ok=True)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, file_path)


def load_curve(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lut_x = np.asarray(data["lut_x"], dtype=np.float32)
    lut_y = np.asarray(data["lut_y"], dtype=np.float32)
    return lut_x, lut_y, data.get("meta", {})


class DepthCurveMatcher:
    """Loads a saved curve (see save_curve/load_curve) and applies it to live depth
    tensors during normal conversion. A true no-op (returns depth completely
    unchanged, by identity) whenever no curve file exists at the given path, or
    strength<=0.0 -- both are checked up front so a user who never calibrates a
    curve, or leaves this off, pays zero cost and gets byte-identical behavior."""

    def __init__(self, file_path=None):
        self.file_path = file_path or DEFAULT_CURVE_PATH
        self.enabled = False
        self.meta = {}
        self._lut_x_t = None
        self._lut_y_t = None
        if self.file_path and path.exists(self.file_path):
            try:
                lut_x, lut_y, meta = load_curve(self.file_path)
                self._lut_x_t = torch.from_numpy(lut_x)
                self._lut_y_t = torch.from_numpy(lut_y)
                self.meta = meta
                self.enabled = True
            except Exception:
                # A corrupt/partial curve file must never crash a real conversion --
                # just fall back to behaving as if no curve was calibrated yet.
                self.enabled = False

    def to(self, device):
        if self._lut_x_t is not None:
            self._lut_x_t = self._lut_x_t.to(device)
            self._lut_y_t = self._lut_y_t.to(device)
        return self

    def apply(self, depth, strength=1.0):
        """depth: any-shape float tensor in [0, 1]. strength 0.0=unchanged (exact
        identity), 1.0=fully remapped to the target curve, in between is a linear
        blend between the two. Returns a tensor of the same shape/dtype/device."""
        if not self.enabled or strength <= 0.0:
            return depth
        if self._lut_x_t.device != depth.device:
            self._lut_x_t = self._lut_x_t.to(depth.device)
            self._lut_y_t = self._lut_y_t.to(depth.device)
        flat = depth.reshape(-1).clamp(0.0, 1.0)
        lut_x = self._lut_x_t.to(flat.dtype)
        lut_y = self._lut_y_t.to(flat.dtype)
        idx = torch.searchsorted(lut_x, flat).clamp(1, lut_x.shape[0] - 1)
        x0 = lut_x[idx - 1]
        x1 = lut_x[idx]
        y0 = lut_y[idx - 1]
        y1 = lut_y[idx]
        span = (x1 - x0).clamp_min(1e-8)
        t = ((flat - x0) / span).clamp(0.0, 1.0)
        matched = (y0 + t * (y1 - y0)).reshape(depth.shape)
        if strength >= 1.0:
            return matched
        return depth * (1.0 - strength) + matched * strength


def _self_test_build_curve_lut_widens_spread():
    """A target distribution with more spread than the source must produce a lut_y
    range wider than lut_x's -- the concrete proof this LUT actually carries the
    "bigger separation" information, not just a same-shape copy."""
    rng = np.random.default_rng(0)
    source = rng.normal(0.5, 0.08, 20000)
    target = rng.normal(0.5, 0.30, 20000)
    lut_x, lut_y = build_curve_lut(source, target, num_points=256)
    assert np.all(np.diff(lut_x) > 0), "lut_x must be strictly increasing"
    x_spread = lut_x[-1] - lut_x[0]
    y_spread = lut_y[-1] - lut_y[0]
    assert y_spread > x_spread, (x_spread, y_spread)
    print("_self_test_build_curve_lut_widens_spread: PASS")


def _self_test_save_load_round_trip():
    import tempfile
    lut_x, lut_y = build_curve_lut(
        np.linspace(0.0, 1.0, 1000), np.linspace(0.0, 1.0, 1000) ** 2, num_points=64)
    with tempfile.TemporaryDirectory(prefix="iw3_curve_selftest_") as tmpdir:
        file_path = path.join(tmpdir, "curve.json")
        save_curve(file_path, lut_x, lut_y, meta={"source_model": "VDA_L", "target_model": "Any_V3_Metric_Large"})
        loaded_x, loaded_y, meta = load_curve(file_path)
        assert np.allclose(loaded_x, lut_x, atol=1e-6)
        assert np.allclose(loaded_y, lut_y, atol=1e-6)
        assert meta["source_model"] == "VDA_L"
    print("_self_test_save_load_round_trip: PASS")


def _self_test_matcher_disabled_without_file():
    matcher = DepthCurveMatcher(file_path=path.join("does", "not", "exist.json"))
    assert matcher.enabled is False
    depth = torch.rand(1, 1, 8, 8)
    out = matcher.apply(depth, strength=1.0)
    assert torch.equal(out, depth), "matcher with no curve file must be a pure identity no-op"
    print("_self_test_matcher_disabled_without_file: PASS")


def _self_test_matcher_corrupt_file_disables_safely():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="iw3_curve_selftest_") as tmpdir:
        bad_path = path.join(tmpdir, "corrupt.json")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        matcher = DepthCurveMatcher(file_path=bad_path)
        assert matcher.enabled is False
        depth = torch.rand(1, 1, 8, 8)
        out = matcher.apply(depth, strength=1.0)
        assert torch.equal(out, depth)
    print("_self_test_matcher_corrupt_file_disables_safely: PASS")


def _self_test_matcher_strength_zero_is_identity():
    import tempfile
    lut_x, lut_y = build_curve_lut(
        np.random.default_rng(1).normal(0.5, 0.08, 5000),
        np.random.default_rng(2).normal(0.5, 0.30, 5000))
    with tempfile.TemporaryDirectory(prefix="iw3_curve_selftest_") as tmpdir:
        file_path = path.join(tmpdir, "curve.json")
        save_curve(file_path, lut_x, lut_y)
        matcher = DepthCurveMatcher(file_path=file_path)
        depth = torch.rand(1, 1, 16, 16)
        out = matcher.apply(depth, strength=0.0)
        assert torch.equal(out, depth), "strength=0.0 must be an exact no-op"
    print("_self_test_matcher_strength_zero_is_identity: PASS")


def _self_test_matcher_full_strength_widens_spread():
    """The actual end-to-end promise: a depth tensor with a narrow (VDA_L-like)
    spread, run through a curve built from a narrow source + a wide target, comes
    out with a visibly WIDER spread at strength=1.0 -- real "bigger jumps."""
    import tempfile
    rng = np.random.default_rng(3)
    source_sample = rng.normal(0.5, 0.08, 20000)
    target_sample = rng.normal(0.5, 0.30, 20000)
    lut_x, lut_y = build_curve_lut(source_sample, target_sample, num_points=512)
    with tempfile.TemporaryDirectory(prefix="iw3_curve_selftest_") as tmpdir:
        file_path = path.join(tmpdir, "curve.json")
        save_curve(file_path, lut_x, lut_y)
        matcher = DepthCurveMatcher(file_path=file_path)
        depth = torch.from_numpy(rng.normal(0.5, 0.08, (1, 1, 128, 128)).astype(np.float32)).clamp(0, 1)
        out = matcher.apply(depth, strength=1.0)
        assert out.std().item() > depth.std().item() * 1.5, (
            f"expected a clearly wider spread after full-strength curve match, "
            f"got in={depth.std().item():.4f} out={out.std().item():.4f}")
    print("_self_test_matcher_full_strength_widens_spread: PASS")


def _self_test_matcher_preserves_order():
    """A monotonic LUT must never flip the real near/far ordering of two different
    input values -- the concrete regression guard against a "banded"/reversed-depth
    artifact. At full strength, ascending input values must stay non-decreasing."""
    import tempfile
    lut_x, lut_y = build_curve_lut(
        np.random.default_rng(4).normal(0.5, 0.1, 5000),
        np.random.default_rng(5).beta(0.3, 0.3, 5000))
    with tempfile.TemporaryDirectory(prefix="iw3_curve_selftest_") as tmpdir:
        file_path = path.join(tmpdir, "curve.json")
        save_curve(file_path, lut_x, lut_y)
        matcher = DepthCurveMatcher(file_path=file_path)
        ascending = torch.linspace(0.0, 1.0, 2000)
        out = matcher.apply(ascending, strength=1.0)
        diffs = out[1:] - out[:-1]
        assert torch.all(diffs >= -1e-6), "matched output must stay non-decreasing for ascending input"
    print("_self_test_matcher_preserves_order: PASS")


def _run_self_tests():
    _self_test_build_curve_lut_widens_spread()
    _self_test_save_load_round_trip()
    _self_test_matcher_disabled_without_file()
    _self_test_matcher_corrupt_file_disables_safely()
    _self_test_matcher_strength_zero_is_identity()
    _self_test_matcher_full_strength_widens_spread()
    _self_test_matcher_preserves_order()
    print("All iw3.depth_curve_match self-tests PASSED")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
    else:
        print("Usage: python -m iw3.depth_curve_match --self-test", file=sys.stderr)
        sys.exit(1)
