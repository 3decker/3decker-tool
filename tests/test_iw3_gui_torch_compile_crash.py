"""Regression test for a real crash: clicking iw3's torch.compile checkbox (`chk_compile`
in iw3/gui.py) with a specific GPU/CPU selected used to throw a raw, uncaught error --
confirmed by a real screenshot, a Windows message box titled "Error: OSError" reading
"[Errno 129] Error number -129 occurred". `update_compile()` (iw3/gui.py) calls
`check_compile_support()` (nunif/models/utils.py), which actually builds and
`torch.compile()`s a tiny real model on the selected device to probe whether compilation
genuinely works there. On this project's real machine, an underlying real compile attempt
can raise an uncaught OSError (a Windows-specific torch.compile/Triton toolchain
compatibility issue -- e.g. a missing MSVC/cl.exe invocation) instead of returning False
cleanly.

Synthetic/isolated (CS-TEST-001): no GPU, no real compile, no real movie file --
`torch.compile` itself is monkeypatched to raise, so this runs identically on any
machine. Covers `nunif.models.utils.check_compile_support` directly (the layer closest
to the real probe); the GUI-level regression (`update_compile()` never letting any
failure here crash the app, checkbox ending up unchecked, and a status-bar message
appearing) is covered separately by `_self_test_compile_probe_crash_handled` in
iw3/gui.py, run via `python -m iw3.gui --self-test` (see tests/smoke_iw3.sh) --
that one needs a real wx.App/MainFrame, which this lightweight script avoids.

Run directly: python tests/test_iw3_gui_torch_compile_crash.py (from the nunif/ dir,
matching this project's other tests/ path conventions), or import and call main().
"""
import sys
from os import path

import torch

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

import nunif.models.utils as MU  # noqa: E402


def _reset_cache():
    MU._COMPILER_SUPPORTED_DEVICES.clear()


def _test_check_compile_support_catches_oserror():
    _reset_cache()
    orig_compile = torch.compile

    def _raise(*a, **kw):
        raise OSError(129, "Error number -129 occurred")

    torch.compile = _raise
    try:
        result = MU.check_compile_support("cpu")
    finally:
        torch.compile = orig_compile

    assert result is False, f"a real OSError from the probe must be caught and return False, got {result!r}"
    print("_test_check_compile_support_catches_oserror: PASS")


def _test_check_compile_support_catches_other_exception_types():
    for exc in (RuntimeError("compile backend failed"), AssertionError("probe assertion failed"),
                ValueError("unexpected probe failure")):
        _reset_cache()
        orig_compile = torch.compile

        def _raise(*a, _exc=exc, **kw):
            raise _exc

        torch.compile = _raise
        try:
            result = MU.check_compile_support("cpu")
        finally:
            torch.compile = orig_compile

        assert result is False, \
            f"a real {exc.__class__.__name__} from the probe must be caught and return False, got {result!r}"

    print("_test_check_compile_support_catches_other_exception_types: PASS")


def _test_check_compile_support_true_on_success():
    # Mocked success path too (not a real compiler-toolchain run) -- this machine may
    # not have MSVC/Triton set up at all (see docs/torch_compile.md), so the "probe
    # succeeds" case must stay just as synthetic/portable as the failure cases above.
    _reset_cache()
    orig_compile = torch.compile
    torch.compile = lambda fn_or_model, *a, **kw: fn_or_model
    try:
        result = MU.check_compile_support("cpu")
    finally:
        torch.compile = orig_compile

    assert result is True, f"a successful probe must return True, got {result!r}"
    print("_test_check_compile_support_true_on_success: PASS")


def _test_check_compile_support_caches_result():
    _reset_cache()
    calls = []
    orig_compile = torch.compile

    def _raise(*a, **kw):
        calls.append(1)
        raise OSError(129, "Error number -129 occurred")

    torch.compile = _raise
    try:
        first = MU.check_compile_support("cpu")
        second = MU.check_compile_support("cpu")
    finally:
        torch.compile = orig_compile

    assert first is False and second is False
    assert len(calls) == 1, "a cached (device already probed) result must not re-run the real probe"
    print("_test_check_compile_support_caches_result: PASS")


def main():
    _test_check_compile_support_catches_oserror()
    _test_check_compile_support_catches_other_exception_types()
    _test_check_compile_support_true_on_success()
    _test_check_compile_support_caches_result()
    print("ALL PASS")


if __name__ == "__main__":
    main()
