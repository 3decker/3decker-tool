"""Regression test for a real crash (ADR-071 follow-up to ADR-070/ADR-068): a raw
"Error: OSError [Errno 129]" popup was reported immediately at "Step 1/5: Scene
Boundary Detection (00:00)" -- i.e. before any packet was ever decoded -- whenever the
torch.compile checkbox was checked, regardless of output video codec (h264_qsv,
h264_nvenc, hevc_nvenc all reproduced it; a torch.compile-off run with otherwise
identical settings did not).

Root cause, confirmed with a REAL, 100%-reproducible traceback on a real RTX 5090 (not
just synthetically): `check_compile_support()` (`nunif/models/utils.py`, called by
`compile_model()` whenever --compile is set -- e.g. `process_video_full`'s side_model
compile, which runs BEFORE the Scene Boundary Detection stage) does a REAL, eager,
synchronous torch.compile build-and-run on the target CUDA device to test whether
compilation works there. Run first in the same process, this reliably leaves the
process's CUDA primary context in a state where a subsequent
`av.open(..., hwaccel=<cuda, primary_ctx=1>)` (create_hwaccel() uses
`options={"primary_ctx": "1"}`, i.e. FFmpeg's NVDEC hwaccel explicitly shares the SAME
primary CUDA context PyTorch uses -- by design, for CUDA tensor-frame interop) fails
immediately with `av.error.OSError: [Errno 129] Error number -129 occurred`, raised
from `av.codec.hwaccel.HWAccel._initialize_hw_context` -- i.e. at container/hwaccel-
device INITIALIZATION time, before any packet exists. This is why it happens at 00:00
elapsed and is unrelated to output codec, and why ADR-070's `safe_decode()` fix (which
only wraps `packet.decode()` inside an already-running demux loop) could never have
caught it -- there is no packet yet at the point this fails. It is also unrelated to
ADR-068's amendment (that one is the REAL model's own DEFERRED first forward() call,
wrapped by `torch._dynamo.config.suppress_errors`) -- this fires from
`check_compile_support()`'s own probe, which necessarily runs its tiny model for real,
eagerly, to determine whether compile is viable at all.

Fix: `open_input_container_with_hwaccel()` (`nunif/utils/video/processor.py`), the new
single choke point all three input-container entry points
(`_process_video`/`hook_frame`/`sample_frames`) now use instead of duplicating
`create_hwaccel()` + `av.open(..., hwaccel=...)` inline. If the container fails to open
with hwaccel requested, and Software Fallback is allowed
(disable_software_fallback=False), it retries once with hwaccel=None (pure software
decode) and returns the corrected effective hwaccel value, matching safe_decode()'s own
"log and continue" precedent for the exact same class of error, one level higher
(container-open time instead of per-packet). Strict mode
(disable_software_fallback=True) and the "no hwaccel was even requested" case are both
re-raised unchanged.

Real, unmocked, GPU reproduction performed (not included here, requires an actual
CUDA device/driver): a real `check_compile_support(torch.device("cuda:0"))` call
followed by a real `av.open(<synthetic h264 clip>, hwaccel=create_hwaccel("cuda", ...))`
reproduced the EXACT reported traceback deterministically (2/2 runs) on a real RTX 5090;
the same sequence through `open_input_container_with_hwaccel()` with
disable_software_fallback=False fell back to software decode and completed
successfully, and disable_software_fallback=True still raised the original error
unchanged.

Synthetic/isolated tests below (no GPU, no real video file, no real hwaccel device
needed): `av.open` and `create_hwaccel` are mocked so the exact real exception class
(`av.error.OSError`, constructed via PyAV's real `FFmpegError.__init__`) is raised on
the first (hwaccel-requested) open call only.

Run directly: python tests/test_iw3_hwaccel_container_open_crash.py (from the nunif/
dir), or import and call main().
"""
import sys
from os import path
from unittest import mock

import av

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from nunif.utils.video import processor as VP  # noqa: E402


class _FakeContainer:
    def __init__(self, hwaccel):
        self.hwaccel = hwaccel


def _fake_device(index=0):
    class _D:
        pass
    d = _D()
    d.index = index
    return d


def _test_falls_back_to_software_when_allowed():
    calls = []

    def fake_open(input_path, mode="r", metadata_errors="ignore", hwaccel=None):
        calls.append(hwaccel)
        if hwaccel is not None:
            raise av.error.OSError(129, "Error number -129 occurred")
        return _FakeContainer(hwaccel)

    with mock.patch.object(VP, "create_hwaccel", return_value=object()), \
         mock.patch.object(VP.av, "open", side_effect=fake_open):
        container, effective_hwaccel = VP.open_input_container_with_hwaccel(
            "dummy.mp4", "cuda", _fake_device(), disable_software_fallback=False
        )
    assert effective_hwaccel is None, f"expected fallback to hwaccel=None, got {effective_hwaccel!r}"
    assert isinstance(container, _FakeContainer)
    assert calls == [mock.ANY, None] and calls[0] is not None, \
        f"expected a hwaccel-requested open then a hwaccel=None retry, got {calls!r}"
    print("_test_falls_back_to_software_when_allowed: PASS")


def _test_strict_mode_still_raises():
    def fake_open(input_path, mode="r", metadata_errors="ignore", hwaccel=None):
        if hwaccel is not None:
            raise av.error.OSError(129, "Error number -129 occurred")
        return _FakeContainer(hwaccel)

    with mock.patch.object(VP, "create_hwaccel", return_value=object()), \
         mock.patch.object(VP.av, "open", side_effect=fake_open):
        raised = False
        try:
            VP.open_input_container_with_hwaccel(
                "dummy.mp4", "cuda", _fake_device(), disable_software_fallback=True
            )
        except av.error.OSError:
            raised = True
    assert raised, "disable_software_fallback=True (Software Fallback unchecked) must still raise"
    print("_test_strict_mode_still_raises: PASS")


def _test_no_hwaccel_requested_reraises_without_retry():
    calls = []

    def fake_open(input_path, mode="r", metadata_errors="ignore", hwaccel=None):
        calls.append(hwaccel)
        raise av.error.OSError(2, "No such file or directory")

    with mock.patch.object(VP, "create_hwaccel", return_value=None), \
         mock.patch.object(VP.av, "open", side_effect=fake_open):
        raised = False
        try:
            VP.open_input_container_with_hwaccel(
                "dummy.mp4", None, _fake_device(), disable_software_fallback=False
            )
        except av.error.OSError:
            raised = True
    assert raised, "an error with no hwaccel involved must still raise (nothing to fall back from)"
    assert calls == [None], f"must not retry an identical hwaccel=None call, got {calls!r}"
    print("_test_no_hwaccel_requested_reraises_without_retry: PASS")


def _test_success_path_returns_original_hwaccel_unchanged():
    def fake_open(input_path, mode="r", metadata_errors="ignore", hwaccel=None):
        return _FakeContainer(hwaccel)

    with mock.patch.object(VP, "create_hwaccel", return_value=object()), \
         mock.patch.object(VP.av, "open", side_effect=fake_open):
        container, effective_hwaccel = VP.open_input_container_with_hwaccel(
            "dummy.mp4", "cuda", _fake_device(), disable_software_fallback=False
        )
    assert effective_hwaccel == "cuda", f"a clean open must keep the original hwaccel, got {effective_hwaccel!r}"
    print("_test_success_path_returns_original_hwaccel_unchanged: PASS")


def main():
    _test_success_path_returns_original_hwaccel_unchanged()
    _test_falls_back_to_software_when_allowed()
    _test_strict_mode_still_raises()
    _test_no_hwaccel_requested_reraises_without_retry()
    print("ALL PASS")


if __name__ == "__main__":
    main()
