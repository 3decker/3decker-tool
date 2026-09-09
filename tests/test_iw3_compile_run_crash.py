"""Regression test for a second, real torch.compile crash -- ADR-068 amendment
(2026-09-08).

ADR-068 fixed the raw "Error: OSError [Errno 129]" popup that happened when clicking
iw3's torch.compile checkbox (`update_compile()` -> `check_compile_support()` in
iw3/gui.py, a lightweight probe that builds-and-runs a tiny throwaway model). This
second bug is different: a real screenshot showed the SAME raw OSError popup during an
actual conversion run (title bar: "Step 1/5: Scene Boundary Detection (00:00)"), with
torch.compile checked and a specific real GPU selected -- not from clicking the
checkbox.

Root cause: `compile_model()`/`compile_function()` (nunif/models/utils.py) are the
real, single choke point every real torch.compile invocation in this codebase goes
through (iw3/utils.py's side_model compile, BaseDepthModel.compile() used by every
depth model via compile_context(), base_inpaint.py's compile_function() call, the sod_v1
convergence estimator, video_depth_anything head/pretrained compiles, waifu2x, etc.) --
but unlike check_compile_support(), they had no exception handling at all. Worse:
`torch.compile(model)` only *wraps* a model -- the real compiler-toolchain invocation
(Triton / cl.exe) is deferred by torch to the model's first real forward() call, which
happens deep inside the real pipeline (not at the compile_model() call site), so even
wrapping that call in try/except would not have caught the failure the screenshot shows.
The fix has to make the *first real execution* of a compiled model fall back to eager
instead of raising -- torch's own `torch._dynamo.config.suppress_errors` flag does
exactly this (confirmed for real, not just in docs: a custom torch.compile backend
forced to raise OSError(129, ...) on this project's real dev machine crashes with
`suppress_errors=False` and falls back to a correct eager result with
`suppress_errors=True`, on both CPU and the real CUDA device).

Synthetic/isolated (CS-TEST-001) where possible:
- Wrap-time failures are tested with `torch.compile` monkeypatched to raise, no GPU
  needed, no real compile.
- The actual deferred/execution-time failure this bug report is about is tested with
  REAL (unmocked) torch.compile/dynamo machinery on the CPU device, using a custom
  `backend=` that deliberately raises OSError(129, ...) -- this reproduces the exact
  exception class torch._dynamo raises in production (`BackendCompilerFailed`) without
  needing a GPU or a real broken MSVC/Triton toolchain.

Run directly: python tests/test_iw3_compile_run_crash.py (from the nunif/ dir), or
import and call main().
"""
import sys
import warnings
from os import path

import torch
import torch.nn as nn

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

import nunif.models.utils as MU  # noqa: E402


def _reset_cache():
    MU._COMPILER_SUPPORTED_DEVICES.clear()
    MU._COMPILE_FALLBACK_WARNED = False


def _force_supported(monkeypatch_compile=None):
    # Make check_compile_support() report True without doing a real compile, so
    # compile_model()/compile_function() proceed to the real torch.compile() call
    # under test.
    _reset_cache()
    MU._COMPILER_SUPPORTED_DEVICES["cpu"] = True


class _TinyModel(nn.Module):
    def forward(self, x):
        return x + 1.0


def _tiny_func(x):
    return x + 1.0


def _test_compile_model_catches_wrap_time_failure():
    _force_supported()
    orig_compile = torch.compile

    def _raise(*a, **kw):
        raise OSError(129, "Error number -129 occurred")

    torch.compile = _raise
    try:
        model = _TinyModel()
        result = MU.compile_model(model, device="cpu")
    finally:
        torch.compile = orig_compile

    assert result is model, "a real wrap-time failure must fall back to the original, uncompiled model"
    print("_test_compile_model_catches_wrap_time_failure: PASS")


def _test_compile_function_catches_wrap_time_failure():
    _force_supported()
    orig_compile = torch.compile

    def _raise(*a, **kw):
        raise RuntimeError("compile backend failed")

    torch.compile = _raise
    try:
        result = MU.compile_function(_tiny_func, device="cpu")
    finally:
        torch.compile = orig_compile

    assert result is _tiny_func, "a real wrap-time failure must fall back to the original, uncompiled function"
    print("_test_compile_function_catches_wrap_time_failure: PASS")


def _test_compile_model_enables_dynamo_suppress_errors():
    # This is the actual fix for the real bug: torch.compile(model) only wraps the
    # model -- the real compiler-toolchain invocation is deferred to the model's first
    # real forward() call, deep inside the real pipeline, not at this call site. Unless
    # torch._dynamo.config.suppress_errors is enabled, that deferred failure crashes the
    # whole conversion exactly like the real screenshot.
    _force_supported()
    orig_compile = torch.compile
    orig_suppress = torch._dynamo.config.suppress_errors
    torch._dynamo.config.suppress_errors = False
    torch.compile = lambda fn_or_model, *a, **kw: fn_or_model
    try:
        MU.compile_model(_TinyModel(), device="cpu")
        assert torch._dynamo.config.suppress_errors is True, \
            "compile_model() must enable torch._dynamo's suppress_errors fallback"
    finally:
        torch.compile = orig_compile
        torch._dynamo.config.suppress_errors = orig_suppress
    print("_test_compile_model_enables_dynamo_suppress_errors: PASS")


def _test_compile_fallback_warns_once():
    _force_supported()
    orig_compile = torch.compile
    orig_suppress = torch._dynamo.config.suppress_errors
    torch.compile = lambda fn_or_model, *a, **kw: fn_or_model
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MU.compile_model(_TinyModel(), device="cpu")
            MU.compile_model(_TinyModel(), device="cpu")
    finally:
        torch.compile = orig_compile
        torch._dynamo.config.suppress_errors = orig_suppress

    compile_warnings = [w for w in caught if "torch.compile is enabled" in str(w.message)]
    assert len(compile_warnings) == 1, \
        f"the compile-fallback notice must be surfaced once per process, not per model, got {len(compile_warnings)}"
    print("_test_compile_fallback_warns_once: PASS")


def _test_real_dynamo_execution_failure_falls_back_to_eager():
    # Real (unmocked) torch.compile/dynamo machinery on CPU, with a custom backend
    # engineered to raise OSError(129, ...) -- the exact real failure from the
    # screenshot -- the first time it is actually invoked (mirrors: the wrap always
    # succeeds, the real compiler-toolchain failure only happens on first real
    # execution, deep inside the pipeline). Confirms the true end-to-end fix: with
    # suppress_errors enabled (as compile_model() now does), a real BackendCompilerFailed
    # is caught by torch itself and the call falls back to eager instead of raising.
    device = torch.device("cpu")

    def raising_backend(gm, example_inputs):
        raise OSError(129, "Error number -129 occurred")

    orig_suppress = torch._dynamo.config.suppress_errors
    try:
        torch._dynamo.reset()
        torch._dynamo.config.suppress_errors = False
        model = _TinyModel().eval().to(device)
        compiled = torch.compile(model, backend=raising_backend)
        raised = False
        try:
            with torch.inference_mode():
                compiled(torch.zeros(4, device=device))
        except Exception:
            raised = True
        assert raised, "sanity check: the forced real OSError must actually crash without suppress_errors"

        torch._dynamo.reset()
        torch._dynamo.config.suppress_errors = True
        model2 = _TinyModel().eval().to(device)
        compiled2 = torch.compile(model2, backend=raising_backend)
        with torch.inference_mode():
            y = compiled2(torch.zeros(4, device=device))
        assert torch.equal(y, torch.ones(4)), "must still produce the correct (eager) result after falling back"
    finally:
        torch._dynamo.config.suppress_errors = orig_suppress
        torch._dynamo.reset()
    print("_test_real_dynamo_execution_failure_falls_back_to_eager: PASS")


def main():
    _test_compile_model_catches_wrap_time_failure()
    _test_compile_function_catches_wrap_time_failure()
    _test_compile_model_enables_dynamo_suppress_errors()
    _test_compile_fallback_warns_once()
    _test_real_dynamo_execution_failure_falls_back_to_eager()
    print("ALL PASS")


if __name__ == "__main__":
    main()
