"""Regression test for docs/ai/AI_DECISIONS.md ADR-059 (scene-boundary cache
staleness/unit-mismatch investigation).

Synthetic/isolated (CS-TEST-001): no GPU, no real video decode, no network. Uses
iw3.scene_boundary_cache's real save_cache/try_load_cache functions against a
temp-dir cache + a dummy placeholder video file (only its path/size/mtime are read
by get_cache_path -- content is never decoded), plus direct calls to the two small
pure functions this investigation added:

  - iw3.utils.should_save_scene_cache(disable_scene_cache, is_preview, stop_event)
    -- closes the gap where SBD.detect_boundary() returns an incomplete/empty
    result the instant a scan is cancelled (stop_event set) partway through, but
    the caller used to save that anyway under cache metadata claiming the FULL
    originally-requested start_time/end_time range. Both iw3.utils.process_video_full
    and iw3.utils.export_video, plus iw3.scene_batch._detect_scenes, now gate their
    save_scene_cache() call through this function, checked BEFORE (not after) their
    stop_event early-return.

  - iw3.scene_batch.resolve_scene_scan_fps(native_fps, max_fps) -- SBD.detect_boundary()
    resamples frame pts to a frame-index counter in units of 1/min(native_fps, max_fps)
    seconds whenever max_fps is not None (see FPSFilter in
    nunif/utils/video/video_filter/fps.py), NOT literal milliseconds. iw3.scene_batch
    used to convert those pts to seconds with a flat "/ 1000.0", which is only
    coincidentally close to correct when the resolved fps happens to be near 1000,
    and silently wrong for any real movie's frame rate (~24/25/30fps). It also never
    passed max_fps to its own SBD.detect_boundary() call, meaning a cache file it
    wrote or read (keyed by args.max_fps, shared with iw3.utils's own --scene-detect
    cache -- see iw3.scene_boundary_cache.get_cache_path) could hold pts in a
    completely different unit than what that same cache key means for a normal iw3
    run using the same max_fps.

Also covers the exact real-world scenario that triggered this investigation: a
short/preview-style scan and a full-range scan of overlapping content, confirming a
full-range request never treats the short scan's cache entry as if it were complete
coverage (task item 3).

Run directly: python tests/test_iw3_scene_cache_regression.py (from the nunif/ dir,
matching this project's other tests/ path conventions), or import and call main().
"""
import os
import sys
import tempfile
import shutil
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from iw3.utils import should_save_scene_cache  # noqa: E402
from iw3.scene_batch import resolve_scene_scan_fps  # noqa: E402
from iw3 import scene_boundary_cache as SceneBoundaryCache  # noqa: E402


class _FakeStopEvent:
    def __init__(self, is_set):
        self._is_set = is_set

    def is_set(self):
        return self._is_set


def _test_should_save_scene_cache_gate():
    # Normal completed scan, cache enabled, not a preview -- must save.
    assert should_save_scene_cache(False, False, None) is True
    assert should_save_scene_cache(False, False, _FakeStopEvent(False)) is True

    # --disable-scene-cache always wins, regardless of everything else.
    assert should_save_scene_cache(True, False, None) is False
    assert should_save_scene_cache(True, False, _FakeStopEvent(False)) is False

    # Preview-quality scan (process_video_full's fps=1.0-clamped --preview) must
    # never overwrite the real cache -- this part already existed before this fix.
    assert should_save_scene_cache(False, True, None) is False

    # THE BUG: a scan cancelled partway through (stop_event set) must never be
    # persisted -- SBD.detect_boundary() returns an incomplete/empty result the
    # moment it notices stop_event, but the caller used to save that regardless,
    # tagged with the full originally-requested start_time/end_time range.
    assert should_save_scene_cache(False, False, _FakeStopEvent(True)) is False

    print("_test_should_save_scene_cache_gate: PASS")


def _test_resolve_scene_scan_fps_matches_fps_filter_semantics():
    # A real movie's native fps is far below both the historical hardcoded "/1000.0"
    # assumption and a generous max_fps cap -- the resolved scan fps must be the
    # native fps itself (FPSFilter's target_fps = min(native_fps, max_fps)), not 1000
    # and not max_fps.
    native_fps = 23.976
    assert abs(resolve_scene_scan_fps(native_fps, 30.0) - native_fps) < 1e-6
    assert abs(resolve_scene_scan_fps(native_fps, 1000.0) - native_fps) < 1e-6

    # max_fps below native fps -- must clamp down to max_fps (matches _fps_config's
    # min(native, max_fps), the exact clamp FPSFilter's target_fps uses).
    assert resolve_scene_scan_fps(native_fps, 15.0) == 15.0

    # max_fps=None -- SBD.detect_boundary() skips FPSFilter entirely and pts stay in
    # the stream's raw/native time_base, so seconds-conversion must use native_fps
    # unchanged, not silently substitute something else.
    assert resolve_scene_scan_fps(native_fps, None) == native_fps

    # Guard against reintroducing the old bug: for any ordinary movie fps, the
    # resolved scan fps must NOT be anywhere near 1000.
    for native in (23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0):
        assert resolve_scene_scan_fps(native, 30.0) != 1000.0
        assert resolve_scene_scan_fps(native, 30.0) < 100.0

    print("_test_resolve_scene_scan_fps_matches_fps_filter_semantics: PASS")


def _test_short_scan_never_satisfies_full_range_request():
    # Mirrors the real observed scenario: a short/preview-style scan (e.g. Quick
    # Preview's clamped clip, or an earlier narrower test range) writes an
    # accurately-scoped but NARROW cache entry; a later request for the full,
    # wider range must never silently accept it as complete coverage.
    tmp_dir = tempfile.mkdtemp(prefix="iw3_scene_cache_regression_")
    try:
        video_path = path.join(tmp_dir, "movie.mkv")
        with open(video_path, "w") as f:
            f.write("dummy")
        cache_dir = path.join(tmp_dir, "cache")
        max_fps = 30

        short_pts = [1909, 1980, 2028, 2063, 2109]
        SceneBoundaryCache.save_cache(
            video_path, short_pts, max_fps,
            start_time="00:01:15", end_time="00:02:00",  # 45s preview-style clip
            cache_dir=cache_dir,
        )

        # A full 5-minute request overlapping the short scan's start must NOT be
        # satisfied by the short scan's cache entry -- this is the exact shape of
        # the real cache file this investigation started from
        # (5da009c3932918b5aa6286fcc29f2c06.json: start_time "00:01:15", end_time
        # "00:06:15").
        full_range = SceneBoundaryCache.try_load_cache(
            video_path, max_fps, start_time="00:01:15", end_time="00:06:15",
            cache_dir=cache_dir,
        )
        assert full_range is None, \
            "a short/preview-scoped cache entry must never satisfy a wider full-range request"

        # Now simulate the real full scan actually completing and saving over the
        # same cache key (same input file + same max_fps) with the full range --
        # the cache is allowed to GROW to cover more, and once it does, both the
        # full range and the original short sub-range must be served from it.
        full_pts = short_pts + [2397, 2896, 3056, 3143, 3341, 3439]
        SceneBoundaryCache.save_cache(
            video_path, full_pts, max_fps,
            start_time="00:01:15", end_time="00:06:15",
            cache_dir=cache_dir,
        )
        result_full = SceneBoundaryCache.try_load_cache(
            video_path, max_fps, start_time="00:01:15", end_time="00:06:15",
            cache_dir=cache_dir,
        )
        assert result_full == set(full_pts), result_full

        result_sub_range = SceneBoundaryCache.try_load_cache(
            video_path, max_fps, start_time="00:01:15", end_time="00:02:00",
            cache_dir=cache_dir,
        )
        assert result_sub_range == set(full_pts), \
            "a cache entry covering the full range must also satisfy a narrower sub-range request"

        print("_test_short_scan_never_satisfies_full_range_request: PASS")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _test_cancelled_scan_leaves_no_cache_file_behind():
    # End-to-end simulation of the exact call pattern used in iw3.utils.process_video_full
    # / export_video / iw3.scene_batch._detect_scenes after this fix: the gate is
    # checked, and only if it passes is save_cache actually called.
    tmp_dir = tempfile.mkdtemp(prefix="iw3_scene_cache_regression_")
    try:
        video_path = path.join(tmp_dir, "movie.mkv")
        with open(video_path, "w") as f:
            f.write("dummy")
        cache_dir = path.join(tmp_dir, "cache")
        max_fps = 30

        # detect_boundary() would return an empty set here (see
        # nunif.utils.shot_boundary_detection.detect_boundary's
        # "if stop_event is not None and stop_event.is_set(): return set()").
        segment_pts = set()
        stop_event = _FakeStopEvent(True)

        if should_save_scene_cache(False, False, stop_event):
            SceneBoundaryCache.save_cache(
                video_path, segment_pts, max_fps,
                start_time="00:01:15", end_time="00:06:15",
                cache_dir=cache_dir,
            )

        cache_path = SceneBoundaryCache.get_cache_path(video_path, max_fps, cache_dir=cache_dir)
        assert not path.exists(cache_path), \
            "a cancelled scan must never leave a cache file claiming the full requested range"

        # Confirm a normal (not cancelled) completed scan with the exact same shape
        # of call DOES persist -- proves the gate isn't just always False.
        stop_event_completed = _FakeStopEvent(False)
        if should_save_scene_cache(False, False, stop_event_completed):
            SceneBoundaryCache.save_cache(
                video_path, {1909, 1980}, max_fps,
                start_time="00:01:15", end_time="00:06:15",
                cache_dir=cache_dir,
            )
        assert path.exists(cache_path)

        print("_test_cancelled_scan_leaves_no_cache_file_behind: PASS")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    _test_should_save_scene_cache_gate()
    _test_resolve_scene_scan_fps_matches_fps_filter_semantics()
    _test_short_scan_never_satisfies_full_range_request()
    _test_cancelled_scan_leaves_no_cache_file_behind()
    print("ALL PASS")


if __name__ == "__main__":
    main()
