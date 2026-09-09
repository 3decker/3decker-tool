"""Regression test for a real crash: a raw "Error: OSError [Errno 129]" popup was
reported across multiple, seemingly unrelated stages of a real conversion run
(AutoCrop Analysis, Scene Boundary Detection, and the main Depth & Stereo Conversion
loop), including at least one occurrence with the torch.compile checkbox unchecked and
"Software Fallback" (--disable-software-fallback off, i.e. fallback allowed) checked.

Root cause: all three of those stages decode video frames through the same shared
helper, `safe_decode()` (nunif/utils/video/processor.py), called as
`safe_decode(packet, strict=disable_software_fallback)` from every real frame-decode
loop (`process_video`, `hook_frame`, and the export-config decode path). When
`strict=False` (the "Software Fallback" checkbox checked / --disable-software-fallback
NOT passed -- the reported scenario), `safe_decode()` already tolerated three specific
PyAV decode-time error classes (av.error.InvalidDataError, av.error.PermissionError,
av.error.PatchWelcomeError) by logging a warning and skipping the frame instead of
crashing -- but did not tolerate `av.error.OSError`, which is exactly the class PyAV
raises for a raw OS-level failure from the underlying hwaccel/codec call (it subclasses
both FFmpegError and the builtin OSError -- e.g. a broken CUDA/NVDEC decode path on a
specific machine/driver combination surfaces exactly as "OSError [Errno N] <message>").
This is unrelated to torch.compile: none of AutoCrop.from_video_file(),
SBD.detect_boundary(), or the main conversion loop's torch.compile usage share any code
path with frame decoding -- they only share this one decode helper, which explains why
the crash appeared to "move" between stages regardless of the torch.compile/Scene
Detection checkboxes.

Synthetic/isolated (CS-TEST-001): a fake packet object whose `.decode()` is set to
raise `av.error.OSError` -- no GPU, no real video file, no real broken hwaccel driver
needed.

Run directly: python tests/test_iw3_hwaccel_decode_crash.py (from the nunif/ dir), or
import and call main().
"""
import sys
from os import path

import av

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from nunif.utils.video.processor import safe_decode  # noqa: E402


class _FakePacket:
    def __init__(self, exc):
        self._exc = exc

    def decode(self):
        raise self._exc


def _test_safe_decode_catches_av_oserror_when_not_strict():
    packet = _FakePacket(av.error.OSError(129, "Error number -129 occurred"))
    frames = safe_decode(packet, strict=False)
    assert frames == [], f"a real av.error.OSError must be caught and yield no frames, got {frames!r}"
    print("_test_safe_decode_catches_av_oserror_when_not_strict: PASS")


def _test_safe_decode_still_catches_existing_error_types():
    for exc in (av.error.InvalidDataError(1, "corrupted"),
                av.error.PermissionError(13, "drm"),
                av.error.PatchWelcomeError(1, "unimplemented")):
        packet = _FakePacket(exc)
        frames = safe_decode(packet, strict=False)
        assert frames == [], \
            f"a real {exc.__class__.__name__} must still be caught, got {frames!r}"
    print("_test_safe_decode_still_catches_existing_error_types: PASS")


def _test_safe_decode_strict_still_raises():
    # --disable-software-fallback (Software Fallback unchecked): the user explicitly
    # asked for a hard failure instead of silently continuing -- must be unchanged.
    packet = _FakePacket(av.error.OSError(129, "Error number -129 occurred"))
    raised = False
    try:
        safe_decode(packet, strict=True)
    except av.error.OSError:
        raised = True
    assert raised, "strict=True (Software Fallback disabled) must still let a real decode error raise"
    print("_test_safe_decode_strict_still_raises: PASS")


def main():
    _test_safe_decode_catches_av_oserror_when_not_strict()
    _test_safe_decode_still_catches_existing_error_types()
    _test_safe_decode_strict_still_raises()
    print("ALL PASS")


if __name__ == "__main__":
    main()
