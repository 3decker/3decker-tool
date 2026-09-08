"""python -m iw3.rife_cli -- standalone RIFE frame-interpolation entry point.

Invoked as a SEPARATE subprocess by iw3.utils._run_rife_interpolation() only
after the main iw3 conversion has fully finished, exactly mirroring how
waifu2x.cli is invoked by _run_waifu2x_upscale() -- see
docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001 and docs/ai/AI_DECISIONS.md
ADR-014/ADR-029. Running as its own process means RIFE's model never shares GPU
memory with iw3's depth/stereo models still resident in the caller's process.

Interpolates the FINAL PACKED stereo frame (both eyes already combined into one
frame, e.g. Half-SBS/Full-SBS) as a single image, not each eye separately --
simpler, and avoids any risk of left/right eye desync since both eyes always
move through interpolation together, in lockstep. NOTE: RIFE will see a seam
between the two packed eyes it was never trained on -- an accepted, known
tradeoff, not a bug (see ADR-029)."""
import argparse
import sys

import torch

import nunif.utils.video as VU
from nunif.device import create_device
from .rife_model import DEFAULT_RIFE_MODEL, RIFE_TIERS, interpolate_frame, load_rife_model


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--rife-model", type=str, default=DEFAULT_RIFE_MODEL,
                        choices=list(RIFE_TIERS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def _make_frame_callback(model, device):
    # Buffers exactly the previous decoded frame; when the next frame arrives,
    # emits [previous, interpolated-middle] so playback order stays correct.
    # The very last frame has no successor to interpolate against and is
    # emitted verbatim on flush (frame=None).
    state = {"pending": None}

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            pending = state.pop("pending", None)
            return pending.squeeze(0) if pending is not None else None
        x = VU.to_tensor(frame, device=device).unsqueeze(0)
        pending = state.get("pending")
        if pending is None:
            state["pending"] = x
            return None
        middle = interpolate_frame(model, pending, x, scale=1.0)
        state["pending"] = x
        return [pending.squeeze(0), middle.squeeze(0)]

    return frame_callback


def run(input_path, output_path, rife_model=DEFAULT_RIFE_MODEL, gpu=0):
    device = create_device(gpu)
    model = load_rife_model(rife_model, device)
    frame_callback = _make_frame_callback(model, device)

    def config_callback(sw_format):
        orig_fps = sw_format.get_fps()
        return VU.VideoOutputConfig(
            fps=None,  # no input resampling -- every real decoded frame is kept
            output_fps=float(orig_fps) * 2,
            options={"preset": "medium", "crf": "16"},
        )

    VU.process_video(
        input_path,
        output_path,
        frame_callback,
        config_callback=config_callback,
        title="RIFE",
        device=device,
    )


def _test_ensure_rife_model_downloads_model_package():
    """Regression test for the real bug found on the first genuine end-to-end
    RIFE run (2026-09-08, see docs/ai/AI_DECISIONS.md ADR-029's amendment note):
    the per-tier Google Drive weight archive only contains train_log/
    (version-specific *.py + flownet.pkl) -- RIFE_HDv3.py's own `from
    model.warplayer import warp` / `from model.loss import *` imports need a
    SEPARATE, sibling `model/` package that is NOT in that archive (it's
    checked into the Practical-RIFE git repo itself, fetched separately by
    rife_model._ensure_rife_model_package()). Without that fix,
    `import train_log.RIFE_HDv3` fails with `ModuleNotFoundError: No module
    named 'model'` even though train_log/ downloaded completely successfully.

    Synthetic/mocked (CS-TEST-001): `_RifeModelDownloader.run`/
    `_RifeModelPackageDownloader.run` are patched to call their own real
    `handle()` against locally-built fixture directories shaped exactly like
    the real train_log/ archive root and the real Practical-RIFE repo zip root
    (verified directly against both on 2026-09-08), rather than hitting the
    network -- everything downstream of that (search-by-content, copytree,
    sys.path wiring, the actual `importlib.import_module` call) is real.
    Covers both tiers (rife_425, rife_425_lite) since the fix is in the shared,
    tier-parameterized ensure_rife_model()/_ensure_rife_model_package(), not
    tier-specific code."""
    import importlib as _importlib
    import os as _os
    import shutil as _shutil
    import sys as _sys
    import tempfile
    import textwrap
    from os import path as _path
    from unittest.mock import patch

    from . import rife_model as rm

    def _clear_train_log_modules():
        # Also drop the sibling "model" package cache -- otherwise a prior
        # successful import (earlier tier in the loop below) leaves
        # sys.modules["model.warplayer"] etc. cached, and the "model/ missing"
        # reproduction step at the end would false-pass from that cache instead
        # of actually re-resolving imports from disk.
        for mod_name in list(_sys.modules):
            if (mod_name in ("train_log", "model")
                    or mod_name.startswith("train_log.")
                    or mod_name.startswith("model.")):
                del _sys.modules[mod_name]

    sandbox = tempfile.mkdtemp(prefix="nunif-rife-selftest-")
    try:
        # Fixture 1: shaped like the real per-tier Google Drive archive root
        # (a single top-level train_log/ containing the version-specific *.py
        # + flownet.pkl -- matches the real downloaded 4.25 directory exactly).
        train_log_archive_root = _path.join(sandbox, "fake_train_log_archive")
        train_log_src = _path.join(train_log_archive_root, "train_log")
        _os.makedirs(train_log_src)
        with open(_path.join(train_log_src, "flownet.pkl"), "wb") as f:
            f.write(b"fake-weights")
        with open(_path.join(train_log_src, "IFNet_HDv3.py"), "w") as f:
            f.write("from model.warplayer import warp\nIFNet = object\n")
        with open(_path.join(train_log_src, "RIFE_HDv3.py"), "w") as f:
            f.write(textwrap.dedent("""\
                from model.warplayer import warp
                from model.loss import *
                from train_log.IFNet_HDv3 import *

                class Model:
                    version = 4.25
                """))

        # Fixture 2: shaped like the real Practical-RIFE GitHub repo zip root
        # (top-level "Practical-RIFE-main/model/" -- verified directly against
        # the live repo archive on 2026-09-08).
        repo_archive_root = _path.join(sandbox, "fake_repo_archive")
        model_src = _path.join(repo_archive_root, "Practical-RIFE-main", "model")
        _os.makedirs(model_src)
        with open(_path.join(model_src, "warplayer.py"), "w") as f:
            f.write("def warp(*a, **k):\n    return a[0]\n")
        with open(_path.join(model_src, "loss.py"), "w") as f:
            f.write("EPE = object\nSOBEL = object\n")

        def fake_train_log_run(self, show_progress=True):
            self.handle(train_log_archive_root)

        def fake_model_pkg_run(self, show_progress=True):
            self.handle(repo_archive_root)

        with patch.object(rm, "RIFE_MODEL_DIR", sandbox):
            for tier in rm.RIFE_TIERS:
                with patch.object(rm._RifeModelDownloader, "run", fake_train_log_run), \
                     patch.object(rm._RifeModelPackageDownloader, "run", fake_model_pkg_run):
                    train_log_dir = rm.ensure_rife_model(tier, show_progress=False)

                assert rm.rife_model_available(tier), tier
                assert rm._model_package_available(tier), tier

                train_log_parent = _path.dirname(train_log_dir)
                added = train_log_parent not in _sys.path
                if added:
                    _sys.path.insert(0, train_log_parent)
                _clear_train_log_modules()
                try:
                    mod = _importlib.import_module("train_log.RIFE_HDv3")
                    assert mod.Model.version == 4.25
                finally:
                    _clear_train_log_modules()
                    if added:
                        _sys.path.remove(train_log_parent)

            # Directly reproduce the exact real bug in isolation: train_log/
            # present but model/ absent -- must raise ModuleNotFoundError
            # naming 'model', confirming this is what the fix actually solves.
            tier = "rife_425"
            _shutil.rmtree(_path.join(rm.get_rife_dir(tier), "model"))
            train_log_dir = _path.join(rm.get_rife_dir(tier), "train_log")
            train_log_parent = _path.dirname(train_log_dir)
            added = train_log_parent not in _sys.path
            if added:
                _sys.path.insert(0, train_log_parent)
            _clear_train_log_modules()
            try:
                try:
                    _importlib.import_module("train_log.RIFE_HDv3")
                    raise AssertionError("expected ModuleNotFoundError reproducing the real bug")
                except ModuleNotFoundError as e:
                    assert "model" in str(e), e
            finally:
                _clear_train_log_modules()
                if added:
                    _sys.path.remove(train_log_parent)
    finally:
        _shutil.rmtree(sandbox, ignore_errors=True)

    print("_test_ensure_rife_model_downloads_model_package: PASS")


def _run_self_tests():
    _test_ensure_rife_model_downloads_model_package()
    print("All iw3.rife_cli self-tests PASSED")


def main(argv=None):
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
        return
    args = create_parser().parse_args(argv)
    run(args.input, args.output, rife_model=args.rife_model, gpu=args.gpu)


if __name__ == "__main__":
    sys.exit(main())
