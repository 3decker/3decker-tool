"""
ADR-073 verification (no real GPU required): confirms
OutputTransform.from_cuda_tensor() (nunif/utils/video/color_transform.py) falls back
to the CPU-copy dlpack path -- instead of crashing the whole conversion job -- when
PyAV's av.VideoFrame.from_dlpack() raises av.error.OSError on the zero-copy NVENC
handoff (the exact failure a real captured traceback showed happening partway
through a real, long conversion job, independent of --hwaccel/torch.compile), and
that it does NOT keep retrying the failing zero-copy path on every later frame.

Run (from the nunif project root, same convention as the ADR-070/071 tests):
    python tests/test_iw3_nvenc_dlpack_cuda_context_fallback.py
"""
import sys
from unittest import mock

import torch
import av

import nunif.utils.video.color_transform as CT  # noqa: E402
from nunif.utils.video.color_transform import OutputTransform  # noqa: E402


def _make_input_tensor():
    # Small float32 RGB-like tensor in [0, 1], shape (3, H, W) -- matches what
    # transform()/from_cuda_tensor() expect from the real frame_callback pipeline.
    torch.manual_seed(0)
    return torch.rand(3, 8, 8, dtype=torch.float32)


def test_falls_back_and_disables_zero_copy_after_one_failure():
    calls = {"n": 0}
    real_from_dlpack = av.VideoFrame.from_dlpack

    def fake_from_dlpack(planes, format, **kwargs):
        calls["n"] += 1
        if "cuda_context" in kwargs or "primary_ctx" in kwargs:
            # This is the zero-copy GPU branch -- simulate the real, reported failure.
            raise av.error.OSError(129, "Error number -129 occurred")
        # CPU fallback branch: let the real PyAV call run (planes are real CPU tensors).
        return real_from_dlpack(planes, format=format)

    ot = OutputTransform(
        dst_pix_fmt="yuv420p",
        dst_colorspace=CT.Colorspace.ITU709,
        dst_color_primaries=CT.ColorPrimaries.BT709,
        dst_color_trc=CT.ColorTrc.BT709,
        dst_color_range=CT.ColorRange.MPEG,
        cuda_context=object(),  # any non-None sentinel -- only identity/None-ness matters here
    )

    with mock.patch("nunif.utils.video.color_transform.is_nvidia_gpu", return_value=True), \
         mock.patch("av.VideoFrame.from_dlpack", side_effect=fake_from_dlpack):
        x = _make_input_tensor()

        # First call: zero-copy path is attempted, raises av.error.OSError internally,
        # must be caught here -- NOT propagate -- and fall back to a real CPU frame.
        frame1 = ot.from_cuda_tensor(x)
        assert isinstance(frame1, av.VideoFrame), f"expected a real av.VideoFrame, got {type(frame1)}"
        assert ot.cuda_context is None, "cuda_context must be cleared after the failure"
        assert calls["n"] == 2, f"expected 2 from_dlpack calls (failed zero-copy + CPU fallback), got {calls['n']}"

        # Second call: must NOT retry the zero-copy path at all now (cuda_context is
        # None -> is_cuda_dlpack_supported() is False by construction) -- only one more
        # from_dlpack call (the CPU path), not another failing zero-copy attempt.
        x2 = _make_input_tensor()
        frame2 = ot.from_cuda_tensor(x2)
        assert isinstance(frame2, av.VideoFrame)
        assert calls["n"] == 3, f"expected exactly 1 more from_dlpack call on the 2nd frame, got {calls['n'] - 2}"

    print("PASS: test_falls_back_and_disables_zero_copy_after_one_failure")


def test_normal_zero_copy_path_unaffected_when_no_failure():
    """Sanity check the fix didn't change behavior on the success path: no exception,
    zero-copy from_dlpack still called with cuda_context/primary_ctx, cuda_context
    stays set."""
    calls = []
    real_from_dlpack = av.VideoFrame.from_dlpack

    def fake_from_dlpack(planes, format, **kwargs):
        calls.append(kwargs)
        return real_from_dlpack(planes, format=format)  # simulate success either way

    sentinel_ctx = object()
    ot = OutputTransform(
        dst_pix_fmt="yuv420p",
        dst_colorspace=CT.Colorspace.ITU709,
        dst_color_primaries=CT.ColorPrimaries.BT709,
        dst_color_trc=CT.ColorTrc.BT709,
        dst_color_range=CT.ColorRange.MPEG,
        cuda_context=sentinel_ctx,
    )

    with mock.patch("nunif.utils.video.color_transform.is_nvidia_gpu", return_value=True), \
         mock.patch("av.VideoFrame.from_dlpack", side_effect=fake_from_dlpack):
        x = _make_input_tensor()
        frame = ot.from_cuda_tensor(x)
        assert isinstance(frame, av.VideoFrame)
        assert ot.cuda_context is sentinel_ctx, "cuda_context must be untouched on the success path"
        assert len(calls) == 1 and ("cuda_context" in calls[0] or "primary_ctx" in calls[0]), \
            "success path must call from_dlpack exactly once, via the zero-copy branch"

    print("PASS: test_normal_zero_copy_path_unaffected_when_no_failure")


if __name__ == "__main__":
    test_falls_back_and_disables_zero_copy_after_one_failure()
    test_normal_zero_copy_path_unaffected_when_no_failure()
    print("ALL PASS")
