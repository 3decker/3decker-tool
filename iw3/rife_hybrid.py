"""ADR-384: "Hybrid Frame Rate" -- automatic, motion-based scene selection for RIFE
frame interpolation.

Real-world precedent, researched directly before building this (not guessed): Avatar:
The Way of Water uses exactly this idea for its own theatrical 48fps HFR presentation --
the file stays ONE constant frame rate throughout, but calm/dialogue scenes show each
real frame TWICE in a row (an effective lower motion cadence) while actual action scenes
get genuinely new, distinct frames at the full rate. This module decides WHICH scenes
get which treatment; iw3.rife_cli's own _make_frame_callback already has the mechanism
to act on that decision (the same per-pair "clone the nearest real frame instead of
blending" path already used for scene-cut protection and near-duplicate-source
detection -- see that file's own module docs).

Fully automatic (decker's own explicit choice over a manual scene list, 2026-10-19):
scores each scene's own motion energy (mean frame-to-frame pixel difference at a tiny
downscaled resolution -- cheap, not a real optical-flow computation) and excludes the
CALMEST scenes from real interpolation, by runtime, until roughly the configured
fraction of the movie's total duration remains for genuine interpolation. Deliberately
reuses scene boundaries ALREADY computed for some other reason (Scene Detection / Scene
Batch, same as iw3.utils._load_rife_scene_cut_times's own scene-cut-protection feature)
rather than ever triggering a fresh scene-boundary scan itself -- this module's own
motion scan is a SEPARATE, additional cost on top of that, so Hybrid Frame Rate is
gated (in iw3/gui.py) on Scene Detection/Scene Batch already being on, exactly like
Auto EMA by Scene Length's own precedent for the same reason.

ADR-387 "Auto" percentage mode (real decker follow-up, 2026-10-27): the GUI's
"Hybrid: Action %" field also accepts the literal value "Auto" instead of a fixed
number -- see auto_select_interpolate_fraction() below for how that picks a
per-movie percentage instead of a single number everyone shares, and
compute_hybrid_exclude_ranges()'s own print() calls for how the percentage actually
used (fixed or Auto-detected) always gets logged, live in the Output panel and in
the job log (both tee the same stderr stream -- see utils.py's _job_log_scope())."""
import sys
import torch
import nunif.utils.video as VU


# One sample every 0.25s is plenty for a coarse per-scene motion signal -- real
# scenes run several seconds at minimum, and this isn't trying to catch single-frame
# motion, only "is this scene generally calm or generally busy". Keeping this low
# keeps the extra scan pass cheap regardless of the source's own real frame rate.
_MOTION_SCAN_FPS = 4.0
# Tiny downscale, same spirit as nunif.utils.shot_boundary_detection.detect_boundary's
# own 48x27 TransNetV2 input size -- a frame-difference signal doesn't need real
# resolution, and a bigger size would only make this extra pass needlessly slower.
_MOTION_SCAN_SIZE = "64:36"


def _motion_scan_config_callback(metadata):
    return VU.VideoOutputConfig(fps=_MOTION_SCAN_FPS)


def _scan_scene_motion(video_path, start_time=None, end_time=None,
                        stop_event=None, suspend_event=None, tqdm_fn=None,
                        hwaccel=None, disable_software_fallback=False, device="cpu"):
    """Decodes video_path ONCE, at _MOTION_SCAN_SIZE resolution and _MOTION_SCAN_FPS,
    computing a per-sampled-frame motion-energy value (mean absolute difference from
    the immediately PRECEDING sampled frame, in to_tensor()'s normalized [0, 1] float
    units -- same primitive iw3.rife_cli._is_near_duplicate_pair already uses, just
    aggregated rather than threshold-compared). Returns a list of
    (frame_index, motion) tuples in processing order -- frame_index is this scan's
    OWN output-sequence index (0, 1, 2, ...), so frame_index / _MOTION_SCAN_FPS is
    real seconds since start_time. The very first sampled frame has nothing to
    compare against and is recorded with motion=0.0 (never biases a real scene's
    score, since a scene is many samples long in practice)."""
    results = []
    prev = [None]

    def batch_callback(x, pts):
        for i in range(x.shape[0]):
            frame = x[i:i + 1]
            if prev[0] is not None:
                motion = (frame - prev[0]).abs().mean().item()
            else:
                motion = 0.0
            results.append((pts[i], motion))
            prev[0] = frame

    callback_pool = VU.FrameCallbackPool(
        batch_callback, require_pts=True, batch_size=16, device=device, max_workers=0)
    interpolation = "area" if hwaccel == "cuda" else "bilinear"
    VU.hook_frame(
        video_path, callback_pool,
        config_callback=_motion_scan_config_callback,
        title="Hybrid Frame Rate: scanning scene motion",
        vf=f"scale={_MOTION_SCAN_SIZE}:flags={interpolation}",
        start_time=start_time, end_time=end_time,
        stop_event=stop_event, suspend_event=suspend_event,
        tqdm_fn=tqdm_fn, hwaccel=hwaccel,
        disable_software_fallback=disable_software_fallback,
        device=device,
    )
    return results


def _bucket_motion_by_scene(motion_samples, scan_fps, scene_cut_times, total_duration):
    """Groups per-sample motion values into one mean score per scene, where scenes
    are the ranges between consecutive scene_cut_times (seconds, "end of segment"
    convention -- same as iw3.utils._load_rife_scene_cut_times's own return value),
    covering the full [0, total_duration) range. Returns a list of
    (start_sec, end_sec, mean_motion) tuples, one per scene, sorted by start time.
    A scene with no samples at all (vanishingly short, or the scan produced nothing
    for it) gets mean_motion=0.0 -- treated as calm, never as an error."""
    boundaries = sorted(set([0.0] + [float(t) for t in (scene_cut_times or [])] + [float(total_duration)]))
    # Drop a cut time that lands at/after the real end (stale cache against a
    # shorter re-trim, or float rounding) -- never produces a zero/negative-length
    # trailing scene.
    boundaries = [b for b in boundaries if b <= total_duration]
    if not boundaries or boundaries[-1] < total_duration:
        boundaries.append(float(total_duration))

    n_scenes = len(boundaries) - 1
    if n_scenes <= 0:
        return []
    sums = [0.0] * n_scenes
    counts = [0] * n_scenes
    bi = 0
    for frame_index, motion in motion_samples:
        t = frame_index / scan_fps
        while bi < n_scenes - 1 and t >= boundaries[bi + 1]:
            bi += 1
        sums[bi] += motion
        counts[bi] += 1

    scenes = []
    for i in range(n_scenes):
        mean = (sums[i] / counts[i]) if counts[i] else 0.0
        scenes.append((boundaries[i], boundaries[i + 1], mean))
    return scenes


def select_hybrid_exclude_ranges(scenes, interpolate_fraction=0.5):
    """Given per-scene motion scores (see _bucket_motion_by_scene), returns the list
    of (start_sec, end_sec) ranges to EXCLUDE from real interpolation -- the CALMEST
    scenes, by runtime, accumulated (in ascending-motion order) until excluding them
    would remove more than (1 - interpolate_fraction) of the movie's total duration.
    The remaining (busier) scenes keep real RIFE interpolation.

    `interpolate_fraction` is clamped to [0.0, 1.0]: 1.0 excludes nothing (every
    scene interpolated, byte-for-byte the pre-Hybrid-Frame-Rate behavior), 0.0
    excludes everything (every scene duplicated, no real interpolation anywhere --
    a real, if unusual, choice, not refused).

    Fewer than 2 scenes (Scene Detection found no real cuts at all, or the whole
    scan covers a single segment) returns an empty list rather than ever excluding
    the ENTIRE movie from interpolation on a single degenerate "scene" that covers
    everything -- Hybrid Frame Rate genuinely needs more than one scene to make a
    meaningful per-scene choice; with only one, defaulting to full interpolation
    (this feature's own no-op case) is the safe choice, not a silent all-duplicate
    surprise."""
    if len(scenes) < 2:
        return []
    interpolate_fraction = max(0.0, min(1.0, float(interpolate_fraction)))
    total_duration = sum(end - start for start, end, _ in scenes)
    if total_duration <= 0:
        return []
    target_exclude_duration = total_duration * (1.0 - interpolate_fraction)
    if target_exclude_duration <= 0:
        return []

    ordered = sorted(scenes, key=lambda s: s[2])  # ascending motion -- calmest first
    exclude = []
    accumulated = 0.0
    for start, end, _motion in ordered:
        if accumulated >= target_exclude_duration:
            break
        exclude.append((start, end))
        accumulated += (end - start)
    return sorted(exclude)


def auto_select_interpolate_fraction(scenes, min_fraction=0.1, max_fraction=0.9,
                                      default_fraction=0.5):
    """ADR-387 "Auto" mode for the Action % field (real decker follow-up, 2026-10-27):
    instead of a fixed runtime percentage chosen ahead of time, finds THIS movie's own
    natural break between its calmer and busier scenes -- the single largest jump in
    motion score anywhere in the scene list sorted by motion (same per-scene scores
    select_hybrid_exclude_ranges already works from) -- and splits there, so different
    movies get different percentages automatically instead of everyone sharing one
    number.

    Returns (interpolate_fraction, detected) -- detected=False means no usable gap was
    found (fewer than 2 scenes, every scene reads as equally calm/busy, or every
    candidate gap's own exclude-fraction falls outside [min_fraction, max_fraction] of
    the movie's total runtime) and interpolate_fraction is default_fraction instead.
    Auto never silently produces a 0%/100% extreme off a single degenerate scene or
    scan noise -- it falls back to the same safe default a user would otherwise type
    in by hand.

    min_fraction/max_fraction bound how lopsided an AUTO-DETECTED split is allowed to
    be, as a fraction of total runtime (not scene count): without this, one unusually
    long calm (or busy) scene sitting right next to the biggest raw gap could swing
    the whole movie to interpolating almost nothing, or almost everything, off that
    single gap."""
    if len(scenes) < 2:
        return default_fraction, False
    total_duration = sum(end - start for start, end, _ in scenes)
    if total_duration <= 0:
        return default_fraction, False

    ordered = sorted(scenes, key=lambda s: s[2])  # ascending motion -- calmest first
    cumulative = 0.0
    cumulative_durations = []
    for start, end, _motion in ordered:
        cumulative += (end - start)
        cumulative_durations.append(cumulative)

    best_gap = 0.0
    best_index = None
    for i in range(len(ordered) - 1):
        exclude_fraction = cumulative_durations[i] / total_duration
        if exclude_fraction < min_fraction or exclude_fraction > max_fraction:
            continue
        gap = ordered[i + 1][2] - ordered[i][2]
        if gap > best_gap:
            best_gap = gap
            best_index = i

    if best_index is None:
        return default_fraction, False

    exclude_fraction = cumulative_durations[best_index] / total_duration
    return max(0.0, min(1.0, 1.0 - exclude_fraction)), True


def compute_hybrid_exclude_ranges(video_path, scene_cut_times, args):
    """Top-level entry point called from iw3.utils._run_rife_interpolation. Scans
    video_path's own motion (see _scan_scene_motion), buckets it against
    scene_cut_times into per-scene scores, and returns the exclude ranges for
    rife_cli.py's own --hybrid-exclude-ranges-file (see
    _compute_hybrid_exclude_pair_indices there). Returns None (never an empty list
    used ambiguously) on ANY failure -- a bad scan must never block a RIFE run that
    would otherwise have worked fine; the caller falls back to ordinary unconditional
    interpolation, exactly as if Hybrid Frame Rate had been off. Never raises."""
    try:
        total_duration = float(VU.VideoMetadata.from_file(video_path).get_duration())
    except Exception as e:
        print(f"[iw3] Hybrid Frame Rate: could not read {video_path}'s duration "
              f"({e.__class__.__name__}: {e}) -- continuing with unconditional full "
              f"interpolation.", file=sys.stderr)
        return None
    if total_duration <= 0:
        return None

    state = getattr(args, "state", None) or {}
    try:
        samples = _scan_scene_motion(
            video_path,
            stop_event=state.get("stop_event"),
            suspend_event=state.get("suspend_event"),
            tqdm_fn=state.get("tqdm_fn"),
            hwaccel=getattr(args, "hwaccel", None),
            disable_software_fallback=getattr(args, "disable_software_fallback", False),
            device=state.get("device", "cpu"),
        )
    except Exception as e:
        print(f"[iw3] Hybrid Frame Rate: scene-motion scan failed "
              f"({e.__class__.__name__}: {e}) -- continuing with unconditional full "
              f"interpolation.", file=sys.stderr)
        return None

    scenes = _bucket_motion_by_scene(samples, _MOTION_SCAN_FPS, scene_cut_times, total_duration)

    raw_percent = getattr(args, "rife_hybrid_percent", 50)
    auto_requested = isinstance(raw_percent, str) and raw_percent.strip().lower() == "auto"
    if auto_requested:
        interpolate_fraction, auto_detected = auto_select_interpolate_fraction(scenes)
        if auto_detected:
            print(f"[iw3] Hybrid Frame Rate: Auto mode detected {interpolate_fraction * 100:.0f}% "
                  f"as this movie's own action/calm split (largest motion gap between its scenes).",
                  file=sys.stderr)
        else:
            print(f"[iw3] Hybrid Frame Rate: Auto mode found no clear action/calm split for this "
                  f"movie -- using the {interpolate_fraction * 100:.0f}% default instead.",
                  file=sys.stderr)
    else:
        interpolate_fraction = max(0.0, min(1.0, float(raw_percent) / 100.0))
    exclude_ranges = select_hybrid_exclude_ranges(scenes, interpolate_fraction)

    if len(scenes) < 2:
        print("[iw3] Hybrid Frame Rate: fewer than 2 scenes found -- nothing meaningful to "
              "select between, continuing with unconditional full interpolation.", file=sys.stderr)
        return exclude_ranges

    excluded_duration = sum(end - start for start, end in exclude_ranges)
    print(f"[iw3] Hybrid Frame Rate: interpolating {len(scenes) - len(exclude_ranges)}/{len(scenes)} "
          f"scene(s) as action ({(1 - excluded_duration / total_duration) * 100:.0f}% of runtime), "
          f"duplicating the rest as calm.", file=sys.stderr)
    return exclude_ranges


def _test():
    import unittest
    from unittest.mock import patch

    class TestBucketMotionByScene(unittest.TestCase):
        def test_basic_bucketing(self):
            # scan_fps=4 -> frame_index i is at i/4 seconds. Scene cuts at 2.0 and
            # 5.0 -> three scenes: [0, 2), [2, 5), [5, 10).
            samples = [(i, float(i % 3)) for i in range(40)]  # 0..9.75s
            scenes = _bucket_motion_by_scene(samples, scan_fps=4.0,
                                              scene_cut_times=[2.0, 5.0], total_duration=10.0)
            self.assertEqual(len(scenes), 3)
            starts_ends = [(round(s, 3), round(e, 3)) for s, e, _ in scenes]
            self.assertEqual(starts_ends, [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)])
            # every sample landed in exactly one bucket -- no sample dropped/duplicated
            total_samples_back = 0
            bi = 0
            for i, _ in samples:
                t = i / 4.0
                while bi < 2 and t >= scenes[bi][1]:
                    bi += 1
                total_samples_back += 1
            self.assertEqual(total_samples_back, len(samples))

        def test_no_cuts_is_one_scene(self):
            samples = [(i, 0.1) for i in range(20)]
            scenes = _bucket_motion_by_scene(samples, scan_fps=4.0,
                                              scene_cut_times=None, total_duration=5.0)
            self.assertEqual(len(scenes), 1)
            self.assertEqual((scenes[0][0], scenes[0][1]), (0.0, 5.0))

        def test_empty_scene_defaults_to_zero_motion_not_an_error(self):
            # Cuts at 0.10 and 0.15 carve out a scene strictly between two
            # consecutive samples (scan_fps=4 samples land at 0.0, 0.25, 0.5, ...)
            # -- that middle scene gets NO samples at all, and must default to
            # 0.0 rather than raise a ZeroDivisionError or similar.
            samples = [(i, 0.5) for i in range(40)]
            scenes = _bucket_motion_by_scene(samples, scan_fps=4.0,
                                              scene_cut_times=[0.10, 0.15], total_duration=10.0)
            starts_ends = [(round(s, 3), round(e, 3)) for s, e, _ in scenes]
            empty_scene_index = starts_ends.index((0.1, 0.15))
            self.assertEqual(scenes[empty_scene_index][2], 0.0)

        def test_stale_cut_beyond_total_duration_is_dropped(self):
            # A cached cut time at/after the real (possibly re-trimmed) end must
            # never produce a zero/negative-length trailing scene.
            scenes = _bucket_motion_by_scene([], scan_fps=4.0,
                                              scene_cut_times=[3.0, 999.0], total_duration=5.0)
            self.assertEqual([(round(s, 3), round(e, 3)) for s, e, _ in scenes],
                              [(0.0, 3.0), (3.0, 5.0)])

    class TestSelectHybridExcludeRanges(unittest.TestCase):
        def test_excludes_calmest_scenes_by_runtime(self):
            # 4 scenes, 25s each (100s total). Motion: 0.01 (calmest), 0.5, 0.9, 0.05.
            # interpolate_fraction=0.5 -> exclude up to 50s -- calmest first (0.01,
            # then 0.05) = exactly 50s excluded, the two busiest (0.5, 0.9) kept.
            scenes = [(0, 25, 0.01), (25, 50, 0.5), (50, 75, 0.9), (75, 100, 0.05)]
            exclude = select_hybrid_exclude_ranges(scenes, interpolate_fraction=0.5)
            self.assertEqual(exclude, [(0, 25), (75, 100)])

        def test_fraction_1_excludes_nothing(self):
            scenes = [(0, 10, 0.01), (10, 20, 0.5)]
            self.assertEqual(select_hybrid_exclude_ranges(scenes, interpolate_fraction=1.0), [])

        def test_fraction_0_excludes_everything(self):
            scenes = [(0, 10, 0.01), (10, 20, 0.5)]
            exclude = select_hybrid_exclude_ranges(scenes, interpolate_fraction=0.0)
            self.assertEqual(exclude, [(0, 10), (10, 20)])

        def test_out_of_range_fraction_is_clamped(self):
            scenes = [(0, 10, 0.01), (10, 20, 0.5)]
            self.assertEqual(select_hybrid_exclude_ranges(scenes, interpolate_fraction=5.0), [])
            self.assertEqual(select_hybrid_exclude_ranges(scenes, interpolate_fraction=-1.0),
                              [(0, 10), (10, 20)])

        def test_fewer_than_two_scenes_never_excludes_everything(self):
            # Real safety behavior: Scene Detection found no real cuts at all (one
            # giant "scene" covering the whole movie) -- must NOT silently duplicate
            # the entire movie just because its own single scene happens to be calm.
            self.assertEqual(select_hybrid_exclude_ranges([(0, 7200, 0.01)], 0.5), [])
            self.assertEqual(select_hybrid_exclude_ranges([], 0.5), [])

    class TestAutoSelectInterpolateFraction(unittest.TestCase):
        def test_finds_largest_gap(self):
            # 4 equal-length scenes, motion scores 0.01, 0.05, 0.5, 0.9 -- the biggest
            # jump is between 0.05 and 0.5 (index 1->2), which excludes exactly the
            # first two scenes (50 of 100s = 50% exclude fraction, within bounds).
            scenes = [(0, 25, 0.01), (25, 50, 0.05), (50, 75, 0.5), (75, 100, 0.9)]
            fraction, detected = auto_select_interpolate_fraction(scenes)
            self.assertTrue(detected)
            self.assertAlmostEqual(fraction, 0.5)

        def test_fewer_than_two_scenes_falls_back_to_default(self):
            fraction, detected = auto_select_interpolate_fraction([(0, 100, 0.5)])
            self.assertFalse(detected)
            self.assertEqual(fraction, 0.5)
            fraction, detected = auto_select_interpolate_fraction([])
            self.assertFalse(detected)
            self.assertEqual(fraction, 0.5)

        def test_all_equal_motion_falls_back_to_default(self):
            # No real gap anywhere -- every scene reads as equally calm/busy.
            scenes = [(0, 10, 0.3), (10, 20, 0.3), (20, 30, 0.3)]
            fraction, detected = auto_select_interpolate_fraction(scenes)
            self.assertFalse(detected)
            self.assertEqual(fraction, 0.5)

        def test_out_of_bounds_gap_is_skipped_for_one_within_bounds(self):
            # The biggest raw gap (index 0->1, 4.99) would exclude only 5% of runtime
            # -- outside the default [0.1, 0.9] window -- so it's skipped in favor of
            # the only remaining gap (index 1->2, just 0.1) that lands inside bounds,
            # even though it's a far smaller jump in motion.
            scenes = [(0, 5, 0.01), (5, 10, 5.0), (10, 100, 5.1)]
            fraction, detected = auto_select_interpolate_fraction(scenes)
            self.assertTrue(detected)
            self.assertAlmostEqual(fraction, 0.9)  # excludes the first 10% (scenes 0+1)

        def test_no_gap_within_bounds_falls_back_to_default(self):
            # A single huge scene dominates runtime -- every candidate split lands
            # outside [min_fraction, max_fraction], so this must fall back rather
            # than ever picking a 0%/100%-adjacent extreme.
            scenes = [(0, 1, 0.01), (1, 99, 0.5), (99, 100, 0.9)]
            fraction, detected = auto_select_interpolate_fraction(scenes)
            self.assertFalse(detected)
            self.assertEqual(fraction, 0.5)

    class TestComputeHybridExcludeRanges(unittest.TestCase):
        def test_orchestration_and_graceful_failure(self):
            import types

            args = types.SimpleNamespace(state={}, hwaccel=None, disable_software_fallback=False,
                                          rife_hybrid_percent=50)

            # (a) duration read failure -> None, never raises
            with patch.object(VU.VideoMetadata, "from_file", side_effect=OSError("no such file")):
                self.assertIsNone(compute_hybrid_exclude_ranges("missing.mkv", [10.0], args))

            # (b) scan failure -> None, never raises
            fake_metadata = types.SimpleNamespace(get_duration=lambda: 100.0)
            with patch.object(VU.VideoMetadata, "from_file", return_value=fake_metadata), \
                 patch(f"{__name__}._scan_scene_motion", side_effect=RuntimeError("decode failed")):
                self.assertIsNone(compute_hybrid_exclude_ranges("video.mkv", [10.0], args))

            # (c) genuine success -> real exclude ranges come back, scan called with
            # this job's own stop/suspend/tqdm hooks forwarded from args.state.
            captured = {}

            def fake_scan(video_path, **kwargs):
                captured.update(kwargs)
                # two scenes: [0, 50) calm, [50, 100) busy
                return [(i, 0.01) for i in range(0, 200)] + [(i, 0.9) for i in range(200, 400)]

            with patch.object(VU.VideoMetadata, "from_file", return_value=fake_metadata), \
                 patch(f"{__name__}._scan_scene_motion", side_effect=fake_scan):
                result = compute_hybrid_exclude_ranges("video.mkv", [50.0], args)
            self.assertEqual(result, [(0.0, 50.0)])
            self.assertIn("hwaccel", captured)

        def test_auto_mode_detects_split_and_logs_it(self):
            import io
            import types
            from contextlib import redirect_stderr

            args = types.SimpleNamespace(state={}, hwaccel=None, disable_software_fallback=False,
                                          rife_hybrid_percent="Auto")
            fake_metadata = types.SimpleNamespace(get_duration=lambda: 100.0)

            def fake_scan(video_path, **kwargs):
                # two scenes: [0, 50) calm, [50, 100) busy -- same as test (c) above,
                # but driven through Auto mode instead of a fixed percentage.
                return [(i, 0.01) for i in range(0, 200)] + [(i, 0.9) for i in range(200, 400)]

            buf = io.StringIO()
            with patch.object(VU.VideoMetadata, "from_file", return_value=fake_metadata), \
                 patch(f"{__name__}._scan_scene_motion", side_effect=fake_scan), \
                 redirect_stderr(buf):
                result = compute_hybrid_exclude_ranges("video.mkv", [50.0], args)
            self.assertEqual(result, [(0.0, 50.0)])
            # The actual percentage Auto picked must land in the printed log (live
            # Output panel + job log both tee this same stream) -- decker's own
            # explicit ask, not just a correct return value.
            self.assertIn("Auto mode detected 50%", buf.getvalue())

        def test_auto_mode_is_case_insensitive_and_falls_back_cleanly(self):
            import types

            args = types.SimpleNamespace(state={}, hwaccel=None, disable_software_fallback=False,
                                          rife_hybrid_percent="auto")  # lowercase must also work
            fake_metadata = types.SimpleNamespace(get_duration=lambda: 100.0)

            def fake_scan(video_path, **kwargs):
                # one giant calm scene, no real scene cuts at all -- Auto has nothing
                # to detect, must fall back cleanly rather than raising.
                return [(i, 0.01) for i in range(0, 400)]

            with patch.object(VU.VideoMetadata, "from_file", return_value=fake_metadata), \
                 patch(f"{__name__}._scan_scene_motion", side_effect=fake_scan):
                result = compute_hybrid_exclude_ranges("video.mkv", None, args)
            self.assertEqual(result, [])

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (TestBucketMotionByScene, TestSelectHybridExcludeRanges, TestAutoSelectInterpolateFraction,
                 TestComputeHybridExcludeRanges):
        suite.addTests(loader.loadTestsFromTestCase(case))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    _test()
