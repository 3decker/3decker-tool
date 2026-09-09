"""Regression test for extending "Auto EMA by Scene Length" (--scene-batch-auto-ema,
ADR-057) to a REGULAR (non---scene-batch) --scene-detect conversion, not just
--scene-batch's own separate per-scene-file pipeline. See docs/ai/AI_DECISIONS.md
ADR-060.

Synthetic/isolated (CS-TEST-001): no GPU, no real video decode, no network. Exercises:

  1. iw3.utils.compute_scene_ema_schedule(segment_pts, args, native_fps, range_start,
     range_end) -- the pure function that, given the COMPLETE upfront set of detected
     scene-cut pts (known before frame processing starts, exactly like --scene-batch
     already has for its own pipeline), precomputes each scene's real duration and
     looks up its (ema_buffer, ema_decay) in the same duration table --scene-batch
     uses. Covers: the first scene (measured from range_start, not from 0 -- matters
     when --start-time trims the front of the clip), middle scenes (measured between
     two consecutive cuts), the very last scene (measured out to range_end, i.e. the
     remaining video length, since it has no next boundary), and the zero-cuts case
     (the whole processed range treated as one scene).

  2. iw3.base_depth_model.BaseDepthModel.minmax_normalize's new `ema_updates` param --
     at each reset_ema[i]=True point, an (ema_buffer, ema_decay) update actually
     reconfigures the EMA scaler (verified via get_ema_state()), while None (the
     default, and what a plain --scene-batch-auto-ema-off run always passes) is an
     exact no-op reproducing the untouched prior reset_ema() behavior.

  3. An end-to-end simulation combining 1+2: a short synthetic "movie" with two
     scene cuts (three scenes of different real lengths) is run frame-by-frame
     through compute_scene_ema_schedule + BaseDepthModel.minmax_normalize, and each
     scene is confirmed (via get_ema_state() introspection) to actually get the
     Buffer/Decay matching ITS OWN real measured duration -- not the first scene's,
     not a fixed run-wide value.

  4. The ADR-057 user-editable override file (scene_batch.EMA_OVERRIDES_PATH) is
     reused as-is by compute_scene_ema_schedule's table lookup (via
     iw3.utils._resolve_auto_ema_table) -- an edit made for --scene-batch governs a
     regular conversion's Auto EMA too, without needing a separate table.

  5. Off-by-default exact no-op: iw3.utils._scene_auto_ema_active() (the gate used by
     make_output_filename/_build_iw3_comment_metadata/process_video_full) is False
     whenever --scene-batch-auto-ema is off, whenever --scene-detect is off, and for
     --scene-batch's OWN per-scene file conversions (which must keep tagging their
     real per-scene ema_decay/ema_buffer exactly as before, not the new marker) --
     and the filename/comment tags themselves reflect this correctly in each case.

  6. ADR-057 amendment (table extended 0-15s+ -> 0-20s+, 16 -> 21 buckets): 0-15s is
     unchanged for both models, 15-16s through 19-20s (and the 20s+ ceiling) hold flat
     at each model's old 15s+ value instead of extrapolating growth further, and a
     stale pre-extension (16-entry) override file is safely ignored -- falls back to
     the new hardcoded defaults rather than crashing or misaligning buckets.

  7. ADR-062 -- user-visible reporting for the regular path's Auto EMA schedule:
     _scene_ema_schedule_rows (the internal helper compute_scene_ema_schedule and the
     report/live-log both now share) returns the exact same per-scene values as
     compute_scene_ema_schedule's own (first, boundary) pair, just as full per-scene
     rows; _write_scene_ema_report writes a real CSV with real per-scene data (and
     only the rows a caller actually passes it -- callers are responsible for the
     "off by default" gate, exercised end-to-end in
     nunif/tests/smoke_iw3_auto_ema_report.py); _scene_ema_report_summary computes the
     end-of-run summary line's (scene_count, distinct_count) correctly, including the
     case where two different scenes land in the same Buffer/Decay bucket (distinct
     count must be smaller than scene count, not just equal to it).

Run directly: python tests/test_iw3_scene_auto_ema_regular.py (from the nunif/ dir,
matching this project's other tests/ path conventions), or import and call main().
"""
import sys
import tempfile
import csv
import io
import contextlib
import subprocess
from unittest.mock import patch
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

import torch  # noqa: E402

from iw3.utils import (  # noqa: E402
    create_parser, make_output_filename, _build_iw3_comment_metadata,
    _scene_auto_ema_active, compute_scene_ema_schedule, _scene_ema_schedule_rows,
    _write_scene_ema_report, _scene_ema_report_summary, set_state_args, iw3_main,
    _get_ffmpeg_bin,
)
from iw3.null_depth_model import NullDepthModel  # noqa: E402
import iw3.utils as iw3_utils_mod  # noqa: E402
import iw3.scene_batch as scene_batch_mod  # noqa: E402
import nunif.utils.video as VU  # noqa: E402


def _base_args(**overrides):
    args = create_parser(required_true=False).parse_args([])
    args.metadata = "filename"
    args.video_extension = ".mkv"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _test_compute_scene_ema_schedule():
    # 24fps-equivalent scan (native_fps == max_fps), matching a plain --max-fps 24 run.
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="3DECKER VDA_L")
    native_fps = 24.0

    # Cuts at real times 2.0s, 5.5s, 20.0s -> pts = round(sec * 24) (frame-index units,
    # see resolve_scene_scan_fps/ADR-059) -- range_start=0, range_end=30.0 (whole clip).
    segment_pts = {48, 132, 480}
    first, boundary = compute_scene_ema_schedule(segment_pts, args, native_fps, 0.0, 30.0)

    # scene 0: [0.0, 2.0) -> 2.0s duration -> VDA_L bucket 2-3s -> 16/0.74
    assert first == (16, 0.74), first
    # scene 1 (starts at cut pts=48): [2.0, 5.5) -> 3.5s -> bucket 3-4s -> 20/0.78
    assert boundary[48] == (20, 0.78), boundary[48]
    # scene 2 (starts at cut pts=132): [5.5, 20.0) -> 14.5s -> bucket 14-15s -> 116/0.977
    assert boundary[132] == (116, 0.977), boundary[132]
    # scene 3 (starts at cut pts=480, no next boundary): [20.0, 30.0) -> 10.0s
    # (remaining video length) -> bucket 10-11s -> 76/0.945
    assert boundary[480] == (76, 0.945), boundary[480]

    print("_test_compute_scene_ema_schedule: PASS")


def _test_compute_scene_ema_schedule_trimmed_start():
    # --start-time trims the front of the clip -- the first scene must be measured
    # from range_start (the real point conversion begins), NOT from absolute 0.
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="3DECKER VDA_L")
    native_fps = 24.0
    segment_pts = {180}  # one cut at 7.5s (180 / 24)

    first, boundary = compute_scene_ema_schedule(segment_pts, args, native_fps, 5.0, 10.0)
    # scene 0: [5.0, 7.5) -> 2.5s -> bucket 2-3s -> 16/0.74
    assert first == (16, 0.74), first
    # scene 1 (last, no next boundary): [7.5, 10.0) -> 2.5s -> bucket 2-3s -> 16/0.74
    assert boundary[180] == (16, 0.74), boundary[180]

    print("_test_compute_scene_ema_schedule_trimmed_start: PASS")


def _test_compute_scene_ema_schedule_zero_cuts():
    # No cuts detected at all -- the whole processed range is treated as one scene.
    # (3.9, not 4.0 -- bucket ranges are end-exclusive, so an exact 4.0 duration would
    # land in the NEXT bucket, 4-5s; 3.9 unambiguously exercises 3-4s.)
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="3DECKER Any_V3_Mono_01")
    first, boundary = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 3.9)
    # 3.9s duration -> Any_V3_Mono_01 bucket 3-4s -> 40/0.89
    assert first == (40, 0.89), first
    assert boundary == {}

    print("_test_compute_scene_ema_schedule_zero_cuts: PASS")


def _test_compute_scene_ema_schedule_override_file_respected():
    # The ADR-057 user-editable override file governs a regular conversion's Auto EMA
    # too -- redirect EMA_OVERRIDES_PATH to a throwaway temp file (never touches the
    # real nunif/tmp/iw3_auto_ema_overrides.json), matching that ADR's own test.
    tmp_dir = tempfile.mkdtemp(prefix="iw3_scene_auto_ema_selftest_")
    orig_path = scene_batch_mod.EMA_OVERRIDES_PATH
    scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
    try:
        scene_batch_mod.save_ema_overrides_file({
            "3DECKER VDA_L": [
                {"ema_buffer": buf, "ema_decay": dec}
                for buf, dec in ((999, 0.5),) + tuple(
                    (r["overrides"]["ema_buffer"], r["overrides"]["ema_decay"])
                    for r in scene_batch_mod.EMA_BY_DURATION_VDA_L[1:]
                )
            ]
        })
        args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="3DECKER VDA_L")
        first, boundary = compute_scene_ema_schedule({48}, args, 24.0, 0.0, 10.0)
        # scene 0 falls in the 0-1s bucket (2 frames at 24fps == 0.083s < 1s... use a
        # tighter cut to land exactly in bucket 0): re-derive with a 0.5s-long scene.
        first2, _ = compute_scene_ema_schedule({12}, args, 24.0, 0.0, 10.0)
        assert first2 == (999, 0.5), \
            f"override file was not picked up by a regular conversion's schedule: {first2}"
    finally:
        scene_batch_mod.EMA_OVERRIDES_PATH = orig_path

    print("_test_compute_scene_ema_schedule_override_file_respected: PASS")


def _test_ema_table_20s_flat_ceiling():
    # ADR-057 amendment: the table was extended from 16 to 21 buckets (0-1s through
    # 19-20s, plus a 20s+ ceiling). 0-15s must be byte-for-byte unchanged; 15-16s
    # through 19-20s (and 20s+) deliberately hold FLAT at each model's old "15s+"
    # ceiling value rather than continuing to grow.
    assert len(scene_batch_mod.EMA_BY_DURATION_VDA_L) == 21
    assert len(scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01) == 21

    for rule in scene_batch_mod.EMA_BY_DURATION_VDA_L[15:]:
        assert rule["overrides"] == {"ema_buffer": 120, "ema_decay": 0.98}, rule
    for rule in scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01[15:]:
        assert rule["overrides"] == {"ema_buffer": 240, "ema_decay": 0.99}, rule

    # Bucket boundaries for the new rows: 15-16s ... 19-20s, then 20s+ (open-ended).
    expected_bounds = [(15 + i, 16 + i) for i in range(5)]
    for (lo, hi), rule in zip(expected_bounds, scene_batch_mod.EMA_BY_DURATION_VDA_L[15:20]):
        assert (rule["min_duration"], rule["max_duration"]) == (lo, hi), rule
    assert scene_batch_mod.EMA_BY_DURATION_VDA_L[20] == {
        "min_duration": 20, "overrides": {"ema_buffer": 120, "ema_decay": 0.98}}
    assert scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01[20] == {
        "min_duration": 20, "overrides": {"ema_buffer": 240, "ema_decay": 0.99}}

    # compute_scene_ema_schedule actually returns the flat values for real 15-20s+
    # scene durations, both at a bucket interior point and at the 20s+ open end.
    args_vda = _base_args(max_fps=24.0, scene_batch_auto_ema_model="3DECKER VDA_L")
    first, _ = compute_scene_ema_schedule(set(), args_vda, 24.0, 0.0, 17.5)
    assert first == (120, 0.98), first
    first, _ = compute_scene_ema_schedule(set(), args_vda, 24.0, 0.0, 45.0)
    assert first == (120, 0.98), first

    args_any = _base_args(max_fps=24.0, scene_batch_auto_ema_model="3DECKER Any_V3_Mono_01")
    first, _ = compute_scene_ema_schedule(set(), args_any, 24.0, 0.0, 19.9)
    assert first == (240, 0.99), first
    first, _ = compute_scene_ema_schedule(set(), args_any, 24.0, 0.0, 300.0)
    assert first == (240, 0.99), first

    print("_test_ema_table_20s_flat_ceiling: PASS")


def _test_old_format_override_file_ignored():
    # ADR-057 amendment: _table_from_override's existing length check
    # (len(overrides) != len(base_table)) means a saved override file from BEFORE this
    # table extension (16 entries) must be silently ignored now that the base table has
    # 21 entries -- falling back to the new hardcoded defaults, never crashing or
    # misaligning buckets.
    tmp_dir = tempfile.mkdtemp(prefix="iw3_scene_auto_ema_selftest_")
    orig_path = scene_batch_mod.EMA_OVERRIDES_PATH
    scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
    try:
        old_format_entries = [
            {"ema_buffer": 999, "ema_decay": 0.5}
            for _ in range(16)
        ]
        scene_batch_mod.save_ema_overrides_file({"3DECKER VDA_L": old_format_entries})

        result = scene_batch_mod._table_from_override(
            "3DECKER VDA_L", scene_batch_mod.EMA_BY_DURATION_VDA_L)
        assert result is None, \
            "an old 16-entry override file must be ignored against the new 21-entry table"

        rules = scene_batch_mod._load_scene_settings(
            None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
        assert rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
            "a stale old-format override file must fall back to the new hardcoded defaults"
    finally:
        scene_batch_mod.EMA_OVERRIDES_PATH = orig_path

    print("_test_old_format_override_file_ignored: PASS")


def _test_nagadomi_reference_table():
    # New third table (ADR-057 second amendment): anchored to nagadomi's own real
    # EMAMinMaxScaler presets (iw3/depth_scaler.py) rather than custom-derived. Ramps
    # from IncrementalEMAScaler (buffer=1, decay=0.75) up to WindowEMAScaler (buffer=30,
    # decay=0.9) over the first 4 one-second buckets, then holds flat at nagadomi's own
    # (30, 0.9) pair, unmodified, through 20s+ -- never extrapolated past it.
    table = scene_batch_mod.EMA_BY_DURATION_NAGADOMI_REFERENCE
    assert len(table) == 21
    assert scene_batch_mod.EMA_BY_DURATION_TABLES["Nagadomi_Reference"] is table

    # Spot-check per the task spec: 0-1s, 10-11s, 20s+.
    assert table[0]["min_duration"] == 0 and table[0]["max_duration"] == 1
    assert table[0]["overrides"] == {"ema_buffer": 1, "ema_decay": 0.750}, table[0]

    assert table[10]["min_duration"] == 10 and table[10]["max_duration"] == 11
    assert table[10]["overrides"] == {"ema_buffer": 30, "ema_decay": 0.900}, table[10]

    assert table[20] == {"min_duration": 20, "overrides": {"ema_buffer": 30, "ema_decay": 0.900}}, table[20]

    # Full ramp (0-4s) plus the flat ceiling from 4-5s through 20s+.
    expected = [
        (1, 0.750), (8, 0.786), (16, 0.828), (24, 0.869),
    ] + [(30, 0.900)] * 17
    for rule, (buf, dec) in zip(table, expected):
        assert rule["overrides"] == {"ema_buffer": buf, "ema_decay": dec}, rule

    # Existing tables must be provably untouched by this addition.
    assert len(scene_batch_mod.EMA_BY_DURATION_VDA_L) == 21
    assert scene_batch_mod.EMA_BY_DURATION_VDA_L[0]["overrides"] == {"ema_buffer": 8, "ema_decay": 0.65}
    assert len(scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01) == 21
    assert scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01[0]["overrides"] == {"ema_buffer": 16, "ema_decay": 0.825}

    # compute_scene_ema_schedule actually returns these values for real durations.
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="Nagadomi_Reference")
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 10.5)
    assert first == (30, 0.900), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 0.5)
    assert first == (1, 0.750), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 200.0)
    assert first == (30, 0.900), first

    print("_test_nagadomi_reference_table: PASS")


def _test_gemini_ai_table():
    # New fourth table (ADR-057 third amendment), dropdown choice "GEMINI AI". Unlike
    # the other three tables, its specific per-second curve came from a different AI
    # assistant's suggestion (pasted in by the user) and was NOT verified against
    # nunif's own source/docs -- it's included verbatim, as-given, for real A/B
    # comparison, not as a "corrected" reinterpretation. Buffer grows linearly every
    # bucket out to a full 480-frame (20s) lookahead by the 20s+ bucket.
    table = scene_batch_mod.EMA_BY_DURATION_GEMINI_AI
    assert len(table) == 21
    assert scene_batch_mod.EMA_BY_DURATION_TABLES["GEMINI AI"] is table

    # Spot-check per the task spec: 0-1s, 9-10s, 19-20s, 20s+.
    assert table[0]["min_duration"] == 0 and table[0]["max_duration"] == 1
    assert table[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.750}, table[0]

    assert table[9]["min_duration"] == 9 and table[9]["max_duration"] == 10
    assert table[9]["overrides"] == {"ema_buffer": 240, "ema_decay": 0.943}, table[9]

    assert table[19]["min_duration"] == 19 and table[19]["max_duration"] == 20
    assert table[19]["overrides"] == {"ema_buffer": 480, "ema_decay": 0.975}, table[19]

    assert table[20] == {"min_duration": 20, "overrides": {"ema_buffer": 480, "ema_decay": 0.975}}, table[20]

    # Full 21-row sequence, exactly as given by the user.
    expected = [
        (24, 0.750), (48, 0.800), (72, 0.840), (96, 0.870), (120, 0.890),
        (144, 0.905), (168, 0.918), (192, 0.928), (216, 0.936), (240, 0.943),
        (264, 0.949), (288, 0.954), (312, 0.958), (336, 0.962), (360, 0.965),
        (384, 0.968), (408, 0.970), (432, 0.972), (456, 0.974), (480, 0.975),
        (480, 0.975),
    ]
    for rule, (buf, dec) in zip(table, expected):
        assert rule["overrides"] == {"ema_buffer": buf, "ema_decay": dec}, rule

    # Existing three tables must be provably untouched by this addition.
    assert len(scene_batch_mod.EMA_BY_DURATION_VDA_L) == 21
    assert scene_batch_mod.EMA_BY_DURATION_VDA_L[0]["overrides"] == {"ema_buffer": 8, "ema_decay": 0.65}
    assert len(scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01) == 21
    assert scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01[0]["overrides"] == {"ema_buffer": 16, "ema_decay": 0.825}
    assert len(scene_batch_mod.EMA_BY_DURATION_NAGADOMI_REFERENCE) == 21
    assert scene_batch_mod.EMA_BY_DURATION_NAGADOMI_REFERENCE[0]["overrides"] == {"ema_buffer": 1, "ema_decay": 0.750}

    # compute_scene_ema_schedule actually returns these values for real durations.
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="GEMINI AI")
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 0.5)
    assert first == (24, 0.750), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 9.5)
    assert first == (240, 0.943), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 200.0)
    assert first == (480, 0.975), first

    print("_test_gemini_ai_table: PASS")


def _test_chatgpt_table():
    # New fifth table (ADR-057 fourth amendment), dropdown choice "ChatGPT". Like
    # "GEMINI AI", its specific per-second curve came from a different AI
    # assistant's suggestion (pasted in by the user) and was NOT verified against
    # nunif's own source/docs -- it's included verbatim, as-given, for real A/B
    # comparison, not as a "corrected" reinterpretation. Buffer grows linearly every
    # bucket out to a full 600-frame (25s at 24fps) lookahead by the 20s+ bucket --
    # the largest lookahead of any table in this tool.
    table = scene_batch_mod.EMA_BY_DURATION_CHATGPT
    assert len(table) == 21
    assert scene_batch_mod.EMA_BY_DURATION_TABLES["ChatGPT"] is table

    # Spot-check per the task spec: 0-1s, 9-10s, 19-20s, 20s+.
    assert table[0]["min_duration"] == 0 and table[0]["max_duration"] == 1
    assert table[0]["overrides"] == {"ema_buffer": 30, "ema_decay": 0.75}, table[0]

    assert table[9]["min_duration"] == 9 and table[9]["max_duration"] == 10
    assert table[9]["overrides"] == {"ema_buffer": 300, "ema_decay": 0.88}, table[9]

    assert table[19]["min_duration"] == 19 and table[19]["max_duration"] == 20
    assert table[19]["overrides"] == {"ema_buffer": 600, "ema_decay": 0.99}, table[19]

    assert table[20] == {"min_duration": 20, "overrides": {"ema_buffer": 600, "ema_decay": 0.99}}, table[20]

    # Full 21-row sequence, exactly as given by the user.
    expected = [
        (30, 0.75), (60, 0.77), (90, 0.79), (120, 0.81), (150, 0.83),
        (180, 0.84), (210, 0.85), (240, 0.86), (270, 0.87), (300, 0.88),
        (330, 0.885), (360, 0.89), (390, 0.90), (420, 0.91), (450, 0.92),
        (480, 0.93), (510, 0.94), (540, 0.95), (570, 0.97), (600, 0.99),
        (600, 0.99),
    ]
    for rule, (buf, dec) in zip(table, expected):
        assert rule["overrides"] == {"ema_buffer": buf, "ema_decay": dec}, rule

    # Existing four tables must be provably untouched by this addition.
    assert len(scene_batch_mod.EMA_BY_DURATION_VDA_L) == 21
    assert scene_batch_mod.EMA_BY_DURATION_VDA_L[0]["overrides"] == {"ema_buffer": 8, "ema_decay": 0.65}
    assert len(scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01) == 21
    assert scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01[0]["overrides"] == {"ema_buffer": 16, "ema_decay": 0.825}
    assert len(scene_batch_mod.EMA_BY_DURATION_NAGADOMI_REFERENCE) == 21
    assert scene_batch_mod.EMA_BY_DURATION_NAGADOMI_REFERENCE[0]["overrides"] == {"ema_buffer": 1, "ema_decay": 0.750}
    assert len(scene_batch_mod.EMA_BY_DURATION_GEMINI_AI) == 21
    assert scene_batch_mod.EMA_BY_DURATION_GEMINI_AI[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.750}

    # compute_scene_ema_schedule actually returns these values for real durations.
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="ChatGPT")
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 0.5)
    assert first == (30, 0.75), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 9.5)
    assert first == (300, 0.88), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 200.0)
    assert first == (600, 0.99), first

    print("_test_chatgpt_table: PASS")


def _test_grok_table():
    # New sixth table (ADR-057 fifth amendment), dropdown choice "Grok". Like
    # "GEMINI AI"/"ChatGPT", its specific per-second curve came from a different AI
    # assistant's suggestion (pasted in by the user) and was NOT verified against
    # nunif's own source/docs -- it's included verbatim, as-given, for real A/B
    # comparison, not as a "corrected" reinterpretation. Its own character: decay
    # saturates very early (reaches 0.99 by the 8-9s bucket) and holds flat there,
    # while buffer keeps climbing linearly all the way to 480 (20s) at the top end.
    table = scene_batch_mod.EMA_BY_DURATION_GROK
    assert len(table) == 21
    assert scene_batch_mod.EMA_BY_DURATION_TABLES["Grok"] is table

    # Spot-check per the task spec: 0-1s, 8-9s (decay first hits 0.99), 19-20s, 20s+.
    assert table[0]["min_duration"] == 0 and table[0]["max_duration"] == 1
    assert table[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.90}, table[0]

    assert table[8]["min_duration"] == 8 and table[8]["max_duration"] == 9
    assert table[8]["overrides"] == {"ema_buffer": 216, "ema_decay": 0.99}, table[8]

    assert table[19]["min_duration"] == 19 and table[19]["max_duration"] == 20
    assert table[19]["overrides"] == {"ema_buffer": 480, "ema_decay": 0.99}, table[19]

    assert table[20] == {"min_duration": 20, "overrides": {"ema_buffer": 480, "ema_decay": 0.99}}, table[20]

    # Full 21-row sequence, exactly as given by the user.
    expected = [
        (24, 0.90), (48, 0.95), (72, 0.97), (96, 0.975), (120, 0.98),
        (144, 0.98), (168, 0.986), (192, 0.988), (216, 0.99), (240, 0.99),
        (264, 0.99), (288, 0.99), (312, 0.99), (336, 0.99), (360, 0.99),
        (384, 0.99), (408, 0.99), (432, 0.99), (456, 0.99), (480, 0.99),
        (480, 0.99),
    ]
    for rule, (buf, dec) in zip(table, expected):
        assert rule["overrides"] == {"ema_buffer": buf, "ema_decay": dec}, rule

    # Existing five tables must be provably untouched by this addition.
    assert len(scene_batch_mod.EMA_BY_DURATION_VDA_L) == 21
    assert scene_batch_mod.EMA_BY_DURATION_VDA_L[0]["overrides"] == {"ema_buffer": 8, "ema_decay": 0.65}
    assert len(scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01) == 21
    assert scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01[0]["overrides"] == {"ema_buffer": 16, "ema_decay": 0.825}
    assert len(scene_batch_mod.EMA_BY_DURATION_NAGADOMI_REFERENCE) == 21
    assert scene_batch_mod.EMA_BY_DURATION_NAGADOMI_REFERENCE[0]["overrides"] == {"ema_buffer": 1, "ema_decay": 0.750}
    assert len(scene_batch_mod.EMA_BY_DURATION_GEMINI_AI) == 21
    assert scene_batch_mod.EMA_BY_DURATION_GEMINI_AI[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.750}
    assert len(scene_batch_mod.EMA_BY_DURATION_CHATGPT) == 21
    assert scene_batch_mod.EMA_BY_DURATION_CHATGPT[0]["overrides"] == {"ema_buffer": 30, "ema_decay": 0.75}

    # compute_scene_ema_schedule actually returns these values for real durations.
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="Grok")
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 0.5)
    assert first == (24, 0.90), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 8.5)
    assert first == (216, 0.99), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 200.0)
    assert first == (480, 0.99), first

    print("_test_grok_table: PASS")


def _test_fast_action_table():
    # New seventh table (ADR-057 Amendment 10), dropdown choice "Fast Action". Like
    # "GEMINI AI"/"ChatGPT"/"Grok", its per-second curve came from a ChatGPT
    # conversation the user had (this time proposing genre-specific pacing tables)
    # and was NOT verified against nunif's own source/docs -- included verbatim.
    # Unlike those three, it's explicitly capped at 10 seconds: every bucket from
    # 10-11s through 20s+ holds flat at the 9-10s bucket's value.
    table = scene_batch_mod.EMA_BY_DURATION_FAST_ACTION
    assert len(table) == 21
    assert scene_batch_mod.EMA_BY_DURATION_TABLES["Fast Action"] is table

    # Spot-check per the task spec: 0-1s, 9-10s, and confirm 10-11s through 20s+
    # all hold flat at the 9-10s value.
    assert table[0]["min_duration"] == 0 and table[0]["max_duration"] == 1
    assert table[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.750}, table[0]

    assert table[9]["min_duration"] == 9 and table[9]["max_duration"] == 10
    assert table[9]["overrides"] == {"ema_buffer": 240, "ema_decay": 0.774}, table[9]

    for i in range(10, 20):
        assert table[i]["min_duration"] == i and table[i]["max_duration"] == i + 1
        assert table[i]["overrides"] == {"ema_buffer": 240, "ema_decay": 0.774}, (i, table[i])
    assert table[20] == {"min_duration": 20, "overrides": {"ema_buffer": 240, "ema_decay": 0.774}}, table[20]

    # Full 21-row sequence, exactly as given by the user.
    expected = [
        (24, 0.750), (48, 0.753), (72, 0.755), (96, 0.758), (120, 0.761),
        (144, 0.763), (168, 0.766), (192, 0.768), (216, 0.771), (240, 0.774),
        (240, 0.774), (240, 0.774), (240, 0.774), (240, 0.774), (240, 0.774),
        (240, 0.774), (240, 0.774), (240, 0.774), (240, 0.774), (240, 0.774),
        (240, 0.774),
    ]
    for rule, (buf, dec) in zip(table, expected):
        assert rule["overrides"] == {"ema_buffer": buf, "ema_decay": dec}, rule

    # Existing six tables must be provably untouched by this addition.
    assert len(scene_batch_mod.EMA_BY_DURATION_VDA_L) == 21
    assert scene_batch_mod.EMA_BY_DURATION_VDA_L[0]["overrides"] == {"ema_buffer": 8, "ema_decay": 0.65}
    assert len(scene_batch_mod.EMA_BY_DURATION_GROK) == 21
    assert scene_batch_mod.EMA_BY_DURATION_GROK[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.90}

    # compute_scene_ema_schedule actually returns these values for real durations.
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="Fast Action")
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 0.5)
    assert first == (24, 0.750), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 9.5)
    assert first == (240, 0.774), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 200.0)
    assert first == (240, 0.774), first

    print("_test_fast_action_table: PASS")


def _test_medium_magical_table():
    # New eighth table (ADR-057 Amendment 10), dropdown choice "Medium Magical".
    # Same source/verbatim-inclusion/10s-cap-then-flat design as "Fast Action" above.
    table = scene_batch_mod.EMA_BY_DURATION_MEDIUM_MAGICAL
    assert len(table) == 21
    assert scene_batch_mod.EMA_BY_DURATION_TABLES["Medium Magical"] is table

    assert table[0]["min_duration"] == 0 and table[0]["max_duration"] == 1
    assert table[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.820}, table[0]

    assert table[9]["min_duration"] == 9 and table[9]["max_duration"] == 10
    assert table[9]["overrides"] == {"ema_buffer": 240, "ema_decay": 0.847}, table[9]

    for i in range(10, 20):
        assert table[i]["min_duration"] == i and table[i]["max_duration"] == i + 1
        assert table[i]["overrides"] == {"ema_buffer": 240, "ema_decay": 0.847}, (i, table[i])
    assert table[20] == {"min_duration": 20, "overrides": {"ema_buffer": 240, "ema_decay": 0.847}}, table[20]

    expected = [
        (24, 0.820), (48, 0.823), (72, 0.826), (96, 0.829), (120, 0.832),
        (144, 0.835), (168, 0.838), (192, 0.841), (216, 0.844), (240, 0.847),
        (240, 0.847), (240, 0.847), (240, 0.847), (240, 0.847), (240, 0.847),
        (240, 0.847), (240, 0.847), (240, 0.847), (240, 0.847), (240, 0.847),
        (240, 0.847),
    ]
    for rule, (buf, dec) in zip(table, expected):
        assert rule["overrides"] == {"ema_buffer": buf, "ema_decay": dec}, rule

    assert len(scene_batch_mod.EMA_BY_DURATION_FAST_ACTION) == 21
    assert scene_batch_mod.EMA_BY_DURATION_FAST_ACTION[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.750}

    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="Medium Magical")
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 0.5)
    assert first == (24, 0.820), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 9.5)
    assert first == (240, 0.847), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 200.0)
    assert first == (240, 0.847), first

    print("_test_medium_magical_table: PASS")


def _test_drama_slow_paced_table():
    # New ninth table (ADR-057 Amendment 10), dropdown choice "Drama Slow Paced".
    # Same source/verbatim-inclusion/10s-cap-then-flat design as the two above.
    table = scene_batch_mod.EMA_BY_DURATION_DRAMA_SLOW_PACED
    assert len(table) == 21
    assert scene_batch_mod.EMA_BY_DURATION_TABLES["Drama Slow Paced"] is table

    assert table[0]["min_duration"] == 0 and table[0]["max_duration"] == 1
    assert table[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.900}, table[0]

    assert table[9]["min_duration"] == 9 and table[9]["max_duration"] == 10
    assert table[9]["overrides"] == {"ema_buffer": 240, "ema_decay": 0.924}, table[9]

    for i in range(10, 20):
        assert table[i]["min_duration"] == i and table[i]["max_duration"] == i + 1
        assert table[i]["overrides"] == {"ema_buffer": 240, "ema_decay": 0.924}, (i, table[i])
    assert table[20] == {"min_duration": 20, "overrides": {"ema_buffer": 240, "ema_decay": 0.924}}, table[20]

    expected = [
        (24, 0.900), (48, 0.903), (72, 0.905), (96, 0.908), (120, 0.911),
        (144, 0.913), (168, 0.916), (192, 0.918), (216, 0.921), (240, 0.924),
        (240, 0.924), (240, 0.924), (240, 0.924), (240, 0.924), (240, 0.924),
        (240, 0.924), (240, 0.924), (240, 0.924), (240, 0.924), (240, 0.924),
        (240, 0.924),
    ]
    for rule, (buf, dec) in zip(table, expected):
        assert rule["overrides"] == {"ema_buffer": buf, "ema_decay": dec}, rule

    assert len(scene_batch_mod.EMA_BY_DURATION_MEDIUM_MAGICAL) == 21
    assert scene_batch_mod.EMA_BY_DURATION_MEDIUM_MAGICAL[0]["overrides"] == {"ema_buffer": 24, "ema_decay": 0.820}

    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="Drama Slow Paced")
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 0.5)
    assert first == (24, 0.900), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 9.5)
    assert first == (240, 0.924), first
    first, _ = compute_scene_ema_schedule(set(), args, 24.0, 0.0, 200.0)
    assert first == (240, 0.924), first

    print("_test_drama_slow_paced_table: PASS")


def _test_minmax_normalize_ema_updates():
    depth_model = NullDepthModel("NULL")
    depth_model.enable_ema(decay=0.75, buffer_size=1)
    assert depth_model.get_ema_state() == (0.75, 1)

    depth = torch.rand(1, 1, 4, 4)

    # An update at the reset point actually reconfigures the scaler.
    depth_model.minmax_normalize(depth, reset_ema=[True], ema_updates=[(5, 0.42)])
    assert depth_model.get_ema_state() == (0.42, 5), depth_model.get_ema_state()

    # None (what an auto-ema-off run always passes, and the default) is an exact
    # no-op -- decay/buffer_size stay exactly whatever they were, matching the
    # pre-existing bare reset_ema() behavior.
    depth_model.minmax_normalize(depth, reset_ema=[True], ema_updates=[None])
    assert depth_model.get_ema_state() == (0.42, 5), depth_model.get_ema_state()

    # ema_updates entirely omitted (the real call shape for every existing caller
    # untouched by this feature) must behave identically.
    depth_model.minmax_normalize(depth, reset_ema=[True])
    assert depth_model.get_ema_state() == (0.42, 5), depth_model.get_ema_state()

    print("_test_minmax_normalize_ema_updates: PASS")


def _test_end_to_end_different_scenes_different_settings():
    """Simulates a short synthetic clip with two real detected cuts (three scenes of
    different lengths) run through compute_scene_ema_schedule + the real
    BaseDepthModel reset mechanism, confirming each scene actually gets ITS OWN
    Buffer/Decay matching its real measured duration -- not the first scene's, not a
    single fixed value for the whole run. This is the regular (non---scene-batch)
    single-pass path's real integration seam (bind_single_frame_callback/
    bind_batch_frame_callback both funnel through exactly this mechanism)."""
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="3DECKER VDA_L")
    native_fps = 24.0
    # Scene 0: 0.0-1.5s (bucket 1-2s -> 12/0.70)
    # Scene 1: 1.5-8.0s (bucket 6-7s -> 38/0.86)
    # Scene 2 (last): 8.0-9.5s, range_end=9.5 (bucket 1-2s -> 12/0.70)
    cut_a, cut_b = 36, 192  # 1.5s, 8.0s at 24fps
    segment_pts = {cut_a, cut_b}
    range_end = 9.5

    first, boundary = compute_scene_ema_schedule(segment_pts, args, native_fps, 0.0, range_end)
    assert first == (12, 0.70), first
    assert boundary[cut_a] == (38, 0.86), boundary[cut_a]
    assert boundary[cut_b] == (12, 0.70), boundary[cut_b]

    depth_model = NullDepthModel("NULL")
    depth_model.enable_ema(decay=first[1], buffer_size=first[0], motion_adaptive=False)
    assert depth_model.get_ema_state() == (0.70, 12)

    seen_states = [depth_model.get_ema_state()]
    for pts in (cut_a, cut_b):
        depth = torch.rand(1, 1, 4, 4)
        update = boundary.get(pts)
        depth_model.minmax_normalize(depth, reset_ema=[True], ema_updates=[update])
        seen_states.append(depth_model.get_ema_state())

    assert seen_states == [(0.70, 12), (0.86, 38), (0.70, 12)], seen_states
    # The three scenes did not all get the same settings (the actual bug this whole
    # feature fixes -- a single fixed --ema-decay/--ema-buffer for the entire movie).
    assert len(set(seen_states)) == 2, "scenes 0 and 2 share a bucket by design; scene 1 must differ"

    print("_test_end_to_end_different_scenes_different_settings: PASS")


def _test_scene_auto_ema_active_gate():
    # Off by default: neither --scene-batch-auto-ema nor --scene-detect set.
    args_off = _base_args()
    assert not _scene_auto_ema_active(args_off, args_off.ema_normalize)

    # --scene-batch-auto-ema alone (no --scene-detect) -- inert (iw3_main itself
    # raises ValueError for this combo; this gate must also stay False defensively).
    args_no_detect = _base_args(scene_batch_auto_ema=True, ema_normalize=True)
    assert not _scene_auto_ema_active(args_no_detect, args_no_detect.ema_normalize)

    # The real regular-path combo: on.
    args_regular = _base_args(scene_batch_auto_ema=True, scene_detect=True, ema_normalize=True)
    assert _scene_auto_ema_active(args_regular, args_regular.ema_normalize)

    # --scene-batch's OWN per-scene file conversion (scene_batch=True) must NOT be
    # treated as the new regular-path marker -- it already tags its real per-scene
    # ema_decay/ema_buffer values (mutated by scene_batch._process_scenes) and that
    # existing behavior must be untouched.
    args_scene_batch = _base_args(scene_batch_auto_ema=True, scene_detect=True,
                                  ema_normalize=True, scene_batch=True)
    assert not _scene_auto_ema_active(args_scene_batch, args_scene_batch.ema_normalize)

    print("_test_scene_auto_ema_active_gate: PASS")


def _test_filename_and_comment_tags():
    # Off (plain fixed EMA, matching prior behavior exactly): --scene-batch-auto-ema
    # not set -- still gets the old "_emaXXbYY" tag.
    args_fixed = _base_args(ema_normalize=True, ema_decay=0.75, ema_buffer=30, scene_detect=True)
    name_fixed = make_output_filename("in.mkv", args_fixed, video=True)
    assert "_ema75b30" in name_fixed, name_fixed
    assert "_autoema" not in name_fixed, name_fixed
    comment_fixed = _build_iw3_comment_metadata(args_fixed, video=True)
    assert "iw3_ema_decay=0.75 iw3_ema_buffer=30" in comment_fixed, comment_fixed
    assert "iw3_scene_auto_ema" not in comment_fixed, comment_fixed

    # On, regular path: fixed EMA tag replaced by a truthful "varied per scene" marker
    # instead of a misleading single number that was never uniformly used.
    args_auto = _base_args(ema_normalize=True, ema_decay=0.75, ema_buffer=30, scene_detect=True,
                           scene_batch_auto_ema=True, scene_batch_auto_ema_model="3DECKER Any_V3_Mono_01")
    name_auto = make_output_filename("in.mkv", args_auto, video=True)
    assert "_autoema3DECKER Any_V3_Mono_01" in name_auto, name_auto
    assert "_ema75b30" not in name_auto, name_auto
    comment_auto = _build_iw3_comment_metadata(args_auto, video=True)
    assert "iw3_scene_auto_ema=1 iw3_scene_auto_ema_model=3DECKER Any_V3_Mono_01" in comment_auto, comment_auto
    assert "iw3_ema_decay=" not in comment_auto, comment_auto

    # --scene-batch's own per-file tagging is untouched: even with scene_batch_auto_ema
    # set (as it now always is, from the shared GUI checkbox), a scene_batch=True
    # per-scene conversion still tags its real overridden ema_decay/ema_buffer, exactly
    # as it did before this feature existed (scene_batch._process_scenes mutates these
    # to the real per-scene values before calling process_video/make_output_filename).
    args_scene_batch_scene = _base_args(ema_normalize=True, ema_decay=0.83, ema_buffer=30,
                                        scene_detect=True, scene_batch_auto_ema=True,
                                        scene_batch=True)
    name_sb = make_output_filename("in.mkv", args_scene_batch_scene, video=True)
    assert "_ema83b30" in name_sb, name_sb
    assert "_autoema" not in name_sb, name_sb

    print("_test_filename_and_comment_tags: PASS")


def _test_scene_ema_schedule_rows_matches_compute_schedule():
    # _scene_ema_schedule_rows (ADR-062) is the shared internal helper
    # compute_scene_ema_schedule now wraps -- confirm the row-based view and the
    # (first, boundary) pair describe exactly the same schedule, not two
    # independently-computed (and potentially drifting) things.
    args = _base_args(max_fps=24.0, scene_batch_auto_ema_model="3DECKER VDA_L")
    native_fps = 24.0
    segment_pts = {48, 132, 480}  # same fixture as _test_compute_scene_ema_schedule

    rows = _scene_ema_schedule_rows(segment_pts, args, native_fps, 0.0, 30.0)
    first, boundary = compute_scene_ema_schedule(segment_pts, args, native_fps, 0.0, 30.0)

    assert len(rows) == 4, rows  # scene 0 + 3 boundary scenes
    assert rows[0]["scene_index"] == 0 and "pts" not in rows[0]
    assert rows[0]["settings"] == first == (16, 0.74), rows[0]

    for i, pts in enumerate((48, 132, 480), start=1):
        assert rows[i]["scene_index"] == i, rows[i]
        assert rows[i]["pts"] == pts, rows[i]
        assert rows[i]["settings"] == boundary[pts], (rows[i], boundary[pts])

    # start_sec/end_sec are real, ordered scene boundaries covering the whole range.
    assert rows[0]["start_sec"] == 0.0 and rows[0]["end_sec"] == 2.0, rows[0]
    assert rows[1]["start_sec"] == 2.0 and rows[1]["end_sec"] == 5.5, rows[1]
    assert rows[2]["start_sec"] == 5.5 and rows[2]["end_sec"] == 20.0, rows[2]
    assert rows[3]["start_sec"] == 20.0 and rows[3]["end_sec"] == 30.0, rows[3]

    print("_test_scene_ema_schedule_rows_matches_compute_schedule: PASS")


def _test_write_scene_ema_report():
    # Real per-scene data (three scenes, two distinct Buffer/Decay pairs, matching
    # the task's own worked example) written to a real CSV via _write_scene_ema_report
    # (ADR-062), read back and checked field-by-field.
    tmp_dir = tempfile.mkdtemp(prefix="iw3_auto_ema_report_selftest_")
    report_path = path.join(tmp_dir, "movie_Half-SBS.mp4.auto_ema_report.csv")
    rows = [
        {"scene_index": 0, "start_time_sec": "0.000", "duration_sec": "2.000",
         "ema_buffer": 16, "ema_decay": 0.74},
        {"scene_index": 1, "start_time_sec": "2.000", "duration_sec": "3.500",
         "ema_buffer": 20, "ema_decay": 0.78},
        {"scene_index": 2, "start_time_sec": "5.500", "duration_sec": "14.500",
         "ema_buffer": 16, "ema_decay": 0.74},
    ]

    result_path = _write_scene_ema_report(report_path, rows)
    assert result_path == report_path
    assert path.exists(report_path)
    assert not path.exists(report_path + ".tmp"), "temp file must be renamed away, not left behind"

    with open(report_path, "r", encoding="utf-8", newline="") as f:
        read_back = list(csv.DictReader(f))
    assert len(read_back) == 3, read_back
    assert read_back[0] == {"scene_index": "0", "start_time_sec": "0.000", "duration_sec": "2.000",
                            "ema_buffer": "16", "ema_decay": "0.74"}, read_back[0]
    assert read_back[1] == {"scene_index": "1", "start_time_sec": "2.000", "duration_sec": "3.500",
                            "ema_buffer": "20", "ema_decay": "0.78"}, read_back[1]
    assert read_back[2] == {"scene_index": "2", "start_time_sec": "5.500", "duration_sec": "14.500",
                            "ema_buffer": "16", "ema_decay": "0.74"}, read_back[2]

    print("_test_write_scene_ema_report: PASS")


def _test_scene_ema_report_summary():
    # Three scenes, two distinct (buffer, decay) pairs -- scene 0 and scene 2 share
    # a bucket by design (mirrors _test_end_to_end_different_scenes_different_settings),
    # so distinct_count must be 2, not 3 -- the summary must count unique VALUES used,
    # not just how many scenes ran.
    applied_rows = [
        {"scene_index": 0, "start_sec": 0.0, "end_sec": 1.5, "settings": (12, 0.70)},
        {"scene_index": 1, "start_sec": 1.5, "end_sec": 8.0, "settings": (38, 0.86)},
        {"scene_index": 2, "start_sec": 8.0, "end_sec": 9.5, "settings": (12, 0.70)},
    ]
    scene_count, distinct_count = _scene_ema_report_summary(applied_rows)
    assert scene_count == 3, scene_count
    assert distinct_count == 2, distinct_count

    # No scenes at all -- both counts are 0 (the caller never calls this when
    # applied_rows is empty in production, but the function itself must not error).
    assert _scene_ema_report_summary([]) == (0, 0)

    # Every scene distinct -- distinct_count == scene_count.
    all_distinct = [
        {"scene_index": i, "start_sec": float(i), "end_sec": float(i + 1), "settings": (i, 0.5 + i * 0.01)}
        for i in range(4)
    ]
    assert _scene_ema_report_summary(all_distinct) == (4, 4)

    print("_test_scene_ema_report_summary: PASS")


def _test_end_to_end_real_conversion_report_log_and_summary():
    """Real end-to-end run (ADR-062): a genuine synthetic video, decoded and
    re-encoded for real via the bundled ffmpeg and the REAL iw3_main/
    process_video_full/bind_batch_frame_callback pipeline (NullDepthModel,
    method="NULL" -- no GPU, no downloaded checkpoint, no network, same choice
    this project's own NullDepthModel benchmark path and ADR-021/ADR-061's
    grid_sample integration tests made), confirming all three ADR-062
    deliverables together against a real run's real output: the saved
    "<output>.auto_ema_report.csv" sidecar has the correct real per-scene rows,
    the live "[auto-ema] scene N: ..." lines appear in the log DURING the run,
    and the end-of-run "[auto-ema] Auto EMA by Scene Length: ..." summary line
    appears with the correct scene/distinct-value counts.

    The only thing mocked is the scene-boundary NEURAL NET scan itself
    (SBD.detect_boundary) -- it needs a downloaded model this sandbox doesn't
    have, matching this project's own established precedent (see
    test_iw3_scene_cache_regression.py's module docstring: "no GPU... no
    network"). Everything downstream of that -- real video decode/encode, real
    frame callbacks, the real EMA schedule/report/log/summary wiring -- runs
    for real, and expected values are derived from the SAME real
    _scene_ema_schedule_rows/_scene_ema_report_summary functions applied to
    the real file's own real metadata (not hardcoded bucket numbers, which the
    other tests in this file already cover) -- this test's real job is
    checking the WIRING between process_video_full and those functions."""
    tmp_dir = tempfile.mkdtemp(prefix="iw3_auto_ema_e2e_selftest_")
    input_path = path.join(tmp_dir, "in.mp4")
    output_path = path.join(tmp_dir, "out.mp4")

    ffmpeg_bin = _get_ffmpeg_bin()
    subprocess.run(
        [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
         "-t", "6", "-pix_fmt", "yuv420p", input_path],
        check=True, capture_output=True)

    # Fake scene cuts at pts=24 (1.0s) and pts=96 (4.0s), in scan-fps (24fps)
    # frame-index units (ADR-059) -- three scenes: [0.0,1.0), [1.0,4.0), [4.0, end).
    fake_segment_pts = {24, 96}

    def _fake_detect_boundary(*a, **kw):
        return set(fake_segment_pts)

    args = create_parser(required_true=False).parse_args([])
    args.input = input_path
    args.output = output_path
    args.depth_model = "NULL"
    args.method = "NULL"
    # ema_normalize is only actually active when max_fps >= 15 (process_video_full's
    # own gate) -- must match native_fps here so scan_fps == native_fps (ADR-059).
    args.max_fps = 24.0
    args.batch_size = 4
    args.yes = True
    args.metadata = "none"
    args.scene_detect = True
    args.disable_scene_cache = True
    args.ema_normalize = True
    args.scene_batch_auto_ema = True
    args.scene_batch_auto_ema_model = "3DECKER VDA_L"
    args = set_state_args(args)

    # Real metadata, exactly as process_video_full itself reads it -- used to
    # independently derive the EXPECTED schedule via the real, unmocked helper.
    metadata = VU.VideoMetadata.from_file(input_path)
    native_fps = float(metadata.get_fps())
    duration = metadata.guess_duration(to_int=False)
    expected_rows = _scene_ema_schedule_rows(fake_segment_pts, args, native_fps, 0.0, duration)
    expected_applied = [r for r in expected_rows if r["settings"] is not None]
    assert len(expected_applied) == 3, expected_rows  # sanity: all 3 scenes matched a bucket

    captured = io.StringIO()
    with patch.object(iw3_utils_mod.SBD, "detect_boundary", _fake_detect_boundary):
        with contextlib.redirect_stderr(captured):
            iw3_main(args)
    log_output = captured.getvalue()

    # 1. The saved report file exists, next to the real output, with the correct
    # real per-scene rows.
    report_path = output_path + ".auto_ema_report.csv"
    assert path.exists(output_path), "conversion did not produce its real output file"
    assert path.exists(report_path), "no auto_ema_report.csv written"
    with open(report_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(expected_applied), (rows, expected_applied)
    for row, expected in zip(rows, expected_applied):
        assert int(row["scene_index"]) == expected["scene_index"], (row, expected)
        assert abs(float(row["start_time_sec"]) - expected["start_sec"]) < 0.01, (row, expected)
        assert abs(float(row["duration_sec"]) - (expected["end_sec"] - expected["start_sec"])) < 0.01, \
            (row, expected)
        assert int(row["ema_buffer"]) == expected["settings"][0], (row, expected)
        assert abs(float(row["ema_decay"]) - expected["settings"][1]) < 1e-6, (row, expected)

    # 2. Live per-scene lines appeared in the log DURING the run -- one per scene,
    # matching the real applied Buffer/Decay, in the documented [auto-ema] format.
    for r in expected_applied:
        expected_line = (f"[auto-ema] scene {r['scene_index'] + 1}: "
                         f"{r['start_sec']:.1f}s-{r['end_sec']:.1f}s "
                         f"({r['end_sec'] - r['start_sec']:.1f}s) -> "
                         f"buffer={r['settings'][0]} decay={r['settings'][1]:.3f}")
        assert expected_line in log_output, (expected_line, log_output)

    # 3. The end-of-run summary line appears, with the correct scene/distinct counts.
    scene_count, distinct_count = _scene_ema_report_summary(expected_applied)
    expected_summary = (f"[auto-ema] Auto EMA by Scene Length: {scene_count} scenes, "
                        f"{distinct_count} distinct Buffer/Decay values used "
                        f"(see {report_path})")
    assert expected_summary in log_output, (expected_summary, log_output)

    print("_test_end_to_end_real_conversion_report_log_and_summary: PASS")


def main():
    _test_compute_scene_ema_schedule()
    _test_compute_scene_ema_schedule_trimmed_start()
    _test_compute_scene_ema_schedule_zero_cuts()
    _test_compute_scene_ema_schedule_override_file_respected()
    _test_ema_table_20s_flat_ceiling()
    _test_old_format_override_file_ignored()
    _test_nagadomi_reference_table()
    _test_gemini_ai_table()
    _test_chatgpt_table()
    _test_grok_table()
    _test_fast_action_table()
    _test_medium_magical_table()
    _test_drama_slow_paced_table()
    _test_minmax_normalize_ema_updates()
    _test_end_to_end_different_scenes_different_settings()
    _test_scene_auto_ema_active_gate()
    _test_filename_and_comment_tags()
    _test_scene_ema_schedule_rows_matches_compute_schedule()
    _test_write_scene_ema_report()
    _test_scene_ema_report_summary()
    _test_end_to_end_real_conversion_report_log_and_summary()
    print("ALL PASS")


if __name__ == "__main__":
    main()
