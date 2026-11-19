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


DEFAULT_MAX_STRETCH = 1.5


def collapse_tied_points(lut_x, lut_y, eps=1e-4):
    """ADR-474: merge runs of LUT points whose SOURCE values are effectively equal
    (within eps) into one point at their mean x/y. These runs come from many sampled
    pixels sharing the exact same source value -- in practice VDA_L's farthest pixels,
    all exactly 0.0 after per-frame normalization (2.7% of a real full-movie
    calibration). build_curve_lut's epsilon nudge kept lut_x strictly increasing but
    left the whole run's TARGET values (0.0 -> 0.04 in that real curve) spread over
    1e-6-wide steps, i.e. a near-vertical cliff: a pixel at exactly 0 stayed 0 while
    one a hair nearer jumped 4% of the depth range. Returns new (lut_x, lut_y)."""
    lut_x = np.asarray(lut_x, dtype=np.float64)
    lut_y = np.asarray(lut_y, dtype=np.float64)
    out_x, out_y = [], []
    i = 0
    n = len(lut_x)
    while i < n:
        j = i + 1
        while j < n and lut_x[j] - lut_x[i] < eps:
            j += 1
        out_x.append(lut_x[i:j].mean())
        out_y.append(lut_y[i:j].mean())
        i = j
    # Keep the curve's own end points, so the remap still spans the same 0..1 range.
    if out_x[0] > lut_x[0]:
        out_x.insert(0, float(lut_x[0]))
        out_y.insert(0, float(lut_y[0]))
    if out_x[-1] < lut_x[-1]:
        out_x.append(float(lut_x[-1]))
        out_y.append(float(lut_y[-1]))
    return np.asarray(out_x, dtype=np.float32), np.asarray(out_y, dtype=np.float32)


def limit_stretch(lut_x, lut_y, max_stretch, max_iterations=200):
    """ADR-474: cap how strongly the curve may stretch any part of the depth range
    (the LUT's local slope, output change per unit of input change) at max_stretch,
    without changing its two end points. Whatever depth range the capped segments
    give up is shared out proportionally across the uncapped segments, so the curve
    still spans the same total range (same overall "punch") and stays monotonic --
    only the distribution changes. max_stretch is the TRUE final maximum (iterated,
    since sharing out the leftover range can push other segments over the cap).

    Real motivation: decker's full-movie Mario calibration stretched the farthest
    ~15% of VDA_L's range 2.2-2.3x (plus a near-vertical cliff at 0), which made
    wide, background-heavy shots look odd -- distant structures pushed very far back,
    with VDA_L's small background ripples amplified into visible patchy layering.
    max_stretch None/<=0 = unchanged. Returns new (lut_x, lut_y)."""
    lut_x = np.asarray(lut_x, dtype=np.float64)
    lut_y = np.asarray(lut_y, dtype=np.float64)
    if max_stretch is None or max_stretch <= 0 or len(lut_x) < 2:
        return lut_x.astype(np.float32), lut_y.astype(np.float32)
    dx = np.diff(lut_x)
    dy = np.maximum(np.diff(lut_y), 0.0)
    span = lut_y[-1] - lut_y[0]
    # The cap can't be lower than the curve's average slope, or the end points
    # couldn't be reached at all -- clamp to that floor instead of failing.
    floor = span / max(dx.sum(), 1e-12)
    cap = max(float(max_stretch), floor)
    capped = np.zeros_like(dx, dtype=bool)
    for _ in range(max_iterations):
        capped |= dy > cap * dx * (1 + 1e-9)
        dy = np.where(capped, cap * dx, dy)
        free = ~capped
        need = span - dy[capped].sum()
        free_total = dy[free].sum()
        if not free.any() or free_total <= 0:
            break
        dy[free] *= need / free_total
        if not np.any(dy[free] > cap * dx[free] * (1 + 1e-9)):
            break
    new_y = np.concatenate([[lut_y[0]], lut_y[0] + np.cumsum(dy)])
    new_y[-1] = lut_y[-1]
    return lut_x.astype(np.float32), np.clip(new_y, 0.0, 1.0).astype(np.float32)


def max_curve_stretch(lut_x, lut_y):
    """Largest local stretch (slope) anywhere in the curve -- for logs/meta."""
    dx = np.diff(np.asarray(lut_x, dtype=np.float64))
    dy = np.diff(np.asarray(lut_y, dtype=np.float64))
    return float(np.max(dy / np.maximum(dx, 1e-12))) if len(dx) else 0.0


def refine_curve(lut_x, lut_y, max_stretch=DEFAULT_MAX_STRETCH):
    """collapse_tied_points + limit_stretch, the two ADR-474 fixes in order."""
    lut_x, lut_y = collapse_tied_points(lut_x, lut_y)
    return limit_stretch(lut_x, lut_y, max_stretch)


def reshape_curve_file(input_path, output_path, max_stretch=DEFAULT_MAX_STRETCH):
    """ADR-474: apply refine_curve() to an ALREADY-SAVED curve file -- no depth
    models, no GPU, takes a second -- so an existing calibration (e.g. a long
    full-movie one) gets the fixes without being redone. Writes a new file; the
    original is left untouched. Returns (max stretch before, max stretch after)."""
    lut_x, lut_y, meta = load_curve(input_path)
    before = max_curve_stretch(lut_x, lut_y)
    new_x, new_y = refine_curve(lut_x, lut_y, max_stretch)
    meta = dict(meta or {})
    meta["max_stretch"] = max_stretch
    meta["reshaped_from"] = path.basename(input_path)
    save_curve(output_path, new_x, new_y, meta=meta)
    return before, max_curve_stretch(new_x, new_y)


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


def _self_test_limit_stretch_caps_and_keeps_range():
    """ADR-474: after limit_stretch the steepest segment is <= the cap (the TRUE
    final cap, not just the first-pass one), both end points are unchanged, the
    curve stays strictly non-decreasing, and the capped range is redistributed
    (shallow segments get steeper) rather than lost."""
    x = np.linspace(0.0, 1.0, 101)
    y = np.sqrt(x)  # very steep near 0, shallow near 1 -- like a real far-end stretch
    before = max_curve_stretch(x, y)
    nx, ny = limit_stretch(x, y, 1.5)
    after = max_curve_stretch(nx, ny)
    assert before > 3.0, before
    assert after <= 1.5 + 1e-4, after
    assert abs(float(ny[0]) - float(y[0])) < 1e-6 and abs(float(ny[-1]) - float(y[-1])) < 1e-6, (ny[0], ny[-1])
    assert np.all(np.diff(ny) >= -1e-7), "must stay monotonic"
    assert float(ny[90] - ny[80]) > float(y[90] - y[80]), "shallow region should gain range"
    lx, ly = limit_stretch(x, y, 0)
    assert np.allclose(ly, y), "max_stretch 0 must be a no-op"
    print("_self_test_limit_stretch_caps_and_keeps_range: PASS")


def _self_test_collapse_removes_far_end_cliff():
    """ADR-474: reproduces the real full-movie curve's far end -- many sampled
    pixels exactly 0.0 in the source (nudged apart by 1e-6) while the target spans
    0.0 -> 0.04 over them -- and confirms refine_curve removes the cliff: a value a
    hair above 0 no longer jumps ~4% of the depth range."""
    rng = np.random.default_rng(1)
    source = np.concatenate([np.zeros(300), rng.uniform(0.0, 1.0, 10000)])
    target = np.concatenate([rng.uniform(0.0, 0.08, 300), rng.uniform(0.0, 1.0, 10000) ** 0.6])
    lx, ly = build_curve_lut(source, target, num_points=512)
    raw_jump = float(np.interp(0.0005, lx, ly) - np.interp(0.0, lx, ly))
    rx, ry = refine_curve(lx, ly, 1.5)
    new_jump = float(np.interp(0.0005, rx, ry) - np.interp(0.0, rx, ry))
    assert raw_jump > 0.01, raw_jump
    assert new_jump < 0.005, new_jump
    assert np.all(np.diff(rx) > 0), "lut_x must stay strictly increasing"
    assert max_curve_stretch(rx, ry) <= 1.5 + 1e-4
    print("_self_test_collapse_removes_far_end_cliff: PASS")


def _self_test_reshape_curve_file_round_trip():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src = path.join(tmp, "in.json")
        dst = path.join(tmp, "out.json")
        x = np.linspace(0.0, 1.0, 64).astype(np.float32)
        save_curve(src, x, np.sqrt(x), meta={"source_model": "VDA_L"})
        before, after = reshape_curve_file(src, dst, 1.5)
        assert before > 1.5 and after <= 1.5 + 1e-4, (before, after)
        _, _, meta = load_curve(dst)
        assert meta["max_stretch"] == 1.5 and meta["reshaped_from"] == "in.json" and meta["source_model"] == "VDA_L"
        assert DepthCurveMatcher(dst).enabled, "reshaped file must load in the live matcher"
        _, oy, _ = load_curve(src)
        assert np.allclose(oy, np.sqrt(x)), "original file must be untouched"
    print("_self_test_reshape_curve_file_round_trip: PASS")


def _run_self_tests():
    _self_test_limit_stretch_caps_and_keeps_range()
    _self_test_collapse_removes_far_end_cliff()
    _self_test_reshape_curve_file_round_trip()
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
    elif "--reshape" in sys.argv[1:]:
        # ADR-474: python -m iw3.depth_curve_match --reshape IN.json -o OUT.json [--max-stretch 1.5]
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--reshape", required=True, help="existing curve file")
        ap.add_argument("--output", "-o", required=True, help="new curve file (must differ)")
        ap.add_argument("--max-stretch", type=float, default=DEFAULT_MAX_STRETCH)
        a = ap.parse_args()
        if path.abspath(a.reshape) == path.abspath(a.output):
            print("ERROR: --output must be a different file (the original is kept).", file=sys.stderr)
            sys.exit(1)
        before, after = reshape_curve_file(a.reshape, a.output, a.max_stretch)
        print(f"max stretch {before:.2f}x -> {after:.2f}x, saved {a.output}")
    else:
        print("Usage: python -m iw3.depth_curve_match --self-test | "
              "--reshape IN.json -o OUT.json [--max-stretch 1.5]", file=sys.stderr)
        sys.exit(1)
