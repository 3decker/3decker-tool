"""Regression test for ADR-283 (Direct-to-MVC single-pass mode): `VideoOutputConfig.raw_frame_sink`
and the branch it adds to `nunif/utils/video/processor.py`'s `_process_video()`.

Background: `iw3/direct_mvc_cli.py` (new orchestrator) pipes the main iw3 conversion's
finished frames straight into a downstream ffmpeg's stdin instead of writing a finished
SBS file to disk first -- see docs/ai/AI_DECISIONS.md ADR-283. `_process_video()` is the
real per-frame encode funnel shared with waifu2x, so this test proves two things about
the new `config.raw_frame_sink` branch it added:

  1. When set, every frame's raw bytes reach the sink, in order, with the byte length a
     real `yuv420p` frame of that size has -- and no output container/temp file is ever
     created on disk (the whole point of "single pass, no intermediate file").
  2. When left `None` (every pre-existing caller, including waifu2x), the original
     container/encode/mux path still runs and produces a real file -- i.e. the new branch
     is a true no-op for anyone who doesn't opt in.

Synthetic/isolated: a tiny real 4-frame clip is built with PyAV (libx264, no GPU needed)
and fed through the real, unmocked `_process_video()` -- only the frame source is
synthetic, matching how this project's own hwaccel/color-transform tests are built.

Real known coverage gap, found the hard way: this test runs on `device="cpu"`, where
frames really do come out as `yuv420p`. It would NOT have caught the real live-test bug
this feature shipped with -- on a CUDA/hwaccel pipeline, `output_reformatter`'s frames
are frequently `nv12` (interleaved UV) instead, and the first version of this feature
declared the CONFIGURED pix_fmt ("yuv420p") instead of the frame's own REAL one,
silently feeding a raw-video reader the wrong byte layout. Confirmed pixel-identical to
the existing two-stage MVC path only after fixing that (`reformatted_frame.format.name`,
not `config.pix_fmt`) in a real GPU run against real footage -- no synthetic CPU-only
test alone would have surfaced this; treat this file as necessary, not sufficient.

Run directly: python tests/test_iw3_direct_mvc_raw_frame_sink.py (from the nunif/ dir),
or import and call main().
"""
import os
import sys
import tempfile
from os import path

import av
import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from nunif.utils.video import processor as VP  # noqa: E402
from nunif.utils.video.output_config import VideoOutputConfig  # noqa: E402

WIDTH, HEIGHT, NUM_FRAMES, FPS = 64, 32, 4, 24


def _make_synthetic_clip(out_path):
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    for i in range(NUM_FRAMES):
        arr = np.full((HEIGHT, WIDTH, 3), i * 30, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def _test_raw_frame_sink_receives_ordered_frames_and_writes_no_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = path.join(tmp_dir, "in.mp4")
        output_path = path.join(tmp_dir, "out.mp4")
        _make_synthetic_clip(input_path)

        calls = []

        def fake_sink(raw_bytes, width, height, pix_fmt, colorspace, color_primaries, color_trc, color_range):
            calls.append((raw_bytes, width, height, pix_fmt, colorspace, color_primaries, color_trc, color_range))

        def config_callback(metadata):
            return VideoOutputConfig(fps=FPS, pix_fmt="yuv420p", raw_frame_sink=fake_sink)

        VP.process_video(
            input_path, output_path,
            frame_callback=lambda frame: frame,
            config_callback=config_callback,
            device="cpu",
        )

        assert len(calls) == NUM_FRAMES, f"expected {NUM_FRAMES} sink calls, got {len(calls)}"
        expected_len = WIDTH * HEIGHT * 3 // 2  # yuv420p: Y + U/4 + V/4
        for raw_bytes, width, height, pix_fmt, colorspace, color_primaries, color_trc, color_range in calls:
            assert width == WIDTH and height == HEIGHT, f"unexpected frame size {width}x{height}"
            assert pix_fmt == "yuv420p", f"unexpected pix_fmt {pix_fmt!r}"
            assert len(raw_bytes) == expected_len, f"expected {expected_len} raw bytes, got {len(raw_bytes)}"
            # Real live-test regression check (ADR-283 follow-up): raw video carries
            # no embedded color metadata on its own -- these must be real, non-null
            # ints (not e.g. all zeros/None), or a downstream decoder has nothing to
            # go on and guesses wrong, producing a visibly wrong-colored file.
            for name, value in (("colorspace", colorspace), ("color_primaries", color_primaries),
                                ("color_trc", color_trc), ("color_range", color_range)):
                assert isinstance(value, int) and value > 0, f"{name} must be a real, non-zero int, got {value!r}"

        # Frames were encoded with strictly increasing mean luma (0, 30, 60, 90) -- the
        # Y plane's mean byte value must come back out in that same order, proving the
        # sink received real, correctly-ordered frame data (not e.g. all-zero buffers).
        means = [np.frombuffer(raw_bytes, dtype=np.uint8)[:WIDTH * HEIGHT].mean() for raw_bytes, *_ in calls]
        assert means == sorted(means), f"frames arrived out of order: {means}"

        assert not path.exists(output_path), "raw_frame_sink mode must never write the output file"
        tmp_output = VP.make_temporary_file_path(output_path)
        assert not path.exists(tmp_output), "raw_frame_sink mode must never create a temp output file either"

    print("_test_raw_frame_sink_receives_ordered_frames_and_writes_no_file: PASS")


def _test_unset_raw_frame_sink_still_writes_a_real_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = path.join(tmp_dir, "in.mp4")
        output_path = path.join(tmp_dir, "out.mp4")
        _make_synthetic_clip(input_path)

        def config_callback(metadata):
            return VideoOutputConfig(fps=FPS, pix_fmt="yuv420p", video_codec="libx264",
                                     options={"preset": "ultrafast", "crf": "28"})

        VP.process_video(
            input_path, output_path,
            frame_callback=lambda frame: frame,
            config_callback=config_callback,
            device="cpu",
        )

        assert path.exists(output_path), "the default (no raw_frame_sink) path must still produce a real file"
        assert path.getsize(output_path) > 0, "the produced file must not be empty"

    print("_test_unset_raw_frame_sink_still_writes_a_real_file: PASS")


def main():
    _test_raw_frame_sink_receives_ordered_frames_and_writes_no_file()
    _test_unset_raw_frame_sink_still_writes_a_real_file()
    print("ALL PASS")


if __name__ == "__main__":
    main()
