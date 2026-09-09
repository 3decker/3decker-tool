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
tradeoff, not a bug (see ADR-029).

Frame rate multiplier: defaults to 2x (the original behavior) but supports
3x/4x (--rife-multiplier) or an exact target fps (--rife-target-fps, e.g. 24
source -> 60 target) via RIFE >=3.9's explicit-timestep inference call -- see
docs/ai/AI_DECISIONS.md ADR-049 for the scheduling design
(_resolve_target_fps/_compute_pair_timesteps below).

Frame manifest (ADR-051): every run() call also writes a small sidecar JSON
file, "<output>.rife_manifest.json", recording -- for every OUTPUT frame --
whether it's a real source frame (and its original source-frame index) or a
RIFE-synthetic in-between frame (and its nearest real neighbor's source-frame
index, by timestep: the earlier real frame if its local timestep is < 0.5,
the later one otherwise -- same rule _compute_pair_timesteps' callers already
use). This is always-on, cheap metadata, not an opt-in flag. It exists so a
retroactive Dolby Vision RPU reinjection pass (iw3.reinject_hdr_cli
--rife-manifest) can expand the ORIGINAL source's RPU to exactly match RIFE's
expanded frame count -- duplicating each synthetic frame's nearest real
neighbor's actual RPU entry, never blending -- without needing to re-derive
frame correspondence after the fact. See docs/ai/AI_DECISIONS.md ADR-051."""
import argparse
import json
import os
import sys
from fractions import Fraction

import torch

import nunif.utils.video as VU
from nunif.device import create_device
from nunif.utils.video.metadata import convert_fps_fraction
from .rife_model import DEFAULT_RIFE_MODEL, RIFE_TIERS, interpolate_frame, load_rife_model


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--rife-model", type=str, default=DEFAULT_RIFE_MODEL,
                        choices=list(RIFE_TIERS.keys()))
    parser.add_argument("--rife-multiplier", type=int, default=None, choices=[2, 3, 4],
                        help="frame-rate multiplier -- inserts N-1 evenly-spaced frames per real "
                             "pair. Mutually exclusive with --rife-target-fps. Defaults to 2 "
                             "(the original hardcoded doubling behavior) when neither is given.")
    parser.add_argument("--rife-target-fps", type=float, default=None,
                        help="interpolate to this exact output fps instead of a simple multiplier "
                             "(e.g. 60 from a 24fps source). Must be higher than the source's own "
                             "fps. Mutually exclusive with --rife-multiplier.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--video-codec", "-vc", type=str, default=None,
                         help="output video codec (same flag name/convention as the main iw3 "
                              "conversion pipeline's own --video-codec/-vc). Default: unset, which "
                              "keeps this tool's original, unchanged behavior (libx264/H.264) -- most "
                              "RIFE use cases have nothing to do with Dolby Vision/HDR and don't need "
                              "HEVC's larger file size/slower encode. Set this to an HEVC-family codec "
                              "(e.g. libx265, or hevc_nvenc for GPU encoding) ONLY when this RIFE "
                              "output is headed for Dolby Vision/HDR10+ reinjection afterward "
                              "(python -m iw3.reinject_hdr_cli --rife-manifest) -- that step requires "
                              "HEVC output and otherwise always refuses (see "
                              "docs/ai/domains/DOLBY_VISION.md).")
    return parser


def _resolve_target_fps(orig_fps, rife_multiplier, rife_target_fps):
    """Resolves the requested interpolation mode against the real source fps and
    returns the exact target output fps as a Fraction -- see docs/ai/AI_DECISIONS.md
    ADR-049. `orig_fps` may be a Fraction, int, or float (matches VideoMetadata.get_fps()'s
    real return type). Off-by-default guarantee: when both rife_multiplier and
    rife_target_fps are None (no flags given), this reduces to exactly the original
    hardcoded `orig_fps * 2` behavior."""
    if rife_multiplier is not None and rife_target_fps is not None:
        raise ValueError(
            "--rife-multiplier and --rife-target-fps are mutually exclusive -- specify at most one.")
    orig_fps_frac = orig_fps if isinstance(orig_fps, Fraction) else Fraction(orig_fps)
    if rife_target_fps is not None:
        target_fps = convert_fps_fraction(float(rife_target_fps))
        if target_fps <= orig_fps_frac:
            raise ValueError(
                f"--rife-target-fps ({float(target_fps):.3f}) must be higher than the source's own "
                f"frame rate ({float(orig_fps_frac):.3f}fps) -- interpolating to a lower or equal "
                f"frame rate is not supported.")
        return target_fps
    multiplier = rife_multiplier if rife_multiplier is not None else 2
    return orig_fps_frac * multiplier


def _compute_pair_timesteps(frames_out, pair_index, ratio):
    """Drift-free frame-count scheduling for one real-frame pair, modeled
    directly on nunif/utils/video/video_filter/fps.py's FPSFilter.update()
    (round-half-up against a running output-frame counter) -- this project's
    existing precedent for resampling a frame sequence to an arbitrary target
    fps, referenced for the scheduling approach rather than reused directly
    since FPSFilter's job (nearest-real-frame duplication for decode
    resampling) differs from RIFE's (synthesizing genuinely new frames). See
    docs/ai/AI_DECISIONS.md ADR-049.

    A naive PURE per-pair formula (treating real frame `pair_index` as if it
    sat exactly at the continuous position `pair_index * ratio`, independent of
    every other pair) was tried and rejected: for a non-integer ratio (e.g.
    24->60fps, ratio=2.5) it emits a constant 2 synthetic frames every single
    pair -- averaging 2, not the required 1.5 -- so total output frame count
    diverges unboundedly from the target duration over a long video instead of
    tracking it to within one frame. Carrying the ACTUAL running frame count
    forward (as done here, exactly like FPSFilter's `self.frames_out`) fixes
    this: pairs alternate between 1 and 2 synthetic frames as needed so the
    long-run average matches `ratio - 1` exactly.

    `frames_out`: total real+synthetic frames already emitted, INCLUDING real
    frame `pair_index` itself (the running counter's value right after that
    frame was emitted -- 1 for the very first call, before any pair is
    processed). `ratio` = target_fps / orig_fps (exact Fraction, > 1).

    Returns `(timesteps, new_frames_out)`: `timesteps` is the sorted list of
    Fraction timesteps in the open interval (0, 1) for the synthetic frames
    needed between real frame `pair_index` and `pair_index + 1`; `new_frames_out`
    is the updated running counter (advanced past those synthetics AND the next
    real frame) to pass into the following call.

    For an integer ratio N (the simple multiplier case), `(pair_index+1)*ratio`
    is always an exact integer, so round-half-up never actually corrects
    anything and this reduces exactly to the evenly-spaced N-1 timesteps
    1/N, 2/N, ..., (N-1)/N for every pair."""
    # Target 0-indexed output-sequence slot for real frame `pair_index + 1`
    # (round-half-up, matching FPSFilter's own `int(x + Fraction(1, 2))`
    # convention). Clamped to never move backward relative to what's already
    # been emitted (ratio > 1 always makes this a no-op in practice).
    expected_index = max(int((pair_index + 1) * ratio + Fraction(1, 2)), frames_out)
    timesteps = [Fraction(j) / ratio - pair_index for j in range(frames_out, expected_index)]
    new_frames_out = expected_index + 1  # the real frame occupies expected_index; the next slot follows it
    return timesteps, new_frames_out


def _nearest_real_index(pending_source_index, cur_source_index, t):
    """The nearest-real-neighbor rule a synthetic frame at local timestep `t`
    (in the open interval (0, 1) between real source frames
    `pending_source_index` and `cur_source_index == pending_source_index + 1`)
    duplicates its Dolby Vision RPU entry from -- see ADR-051. `t < 0.5` means
    the synthetic frame sits closer to the EARLIER real frame in time; `t >=
    0.5` means it sits closer to (or exactly at) the LATER one. This is pure
    bookkeeping, independent of _compute_pair_timesteps' drift-correction
    machinery -- it only asks "which real frame is this specific timestep
    closer to", using the same timestep values interpolate_frame() itself
    already consumes."""
    return pending_source_index if t < 0.5 else cur_source_index


def _make_frame_callback(model, device, ratio_state, manifest_frames=None):
    # Buffers exactly the previous decoded frame; when the next frame arrives,
    # computes this pair's interpolation timesteps (see _compute_pair_timesteps)
    # and emits [previous, *interpolated-in-between-frames] so playback order
    # stays correct. The very last frame has no successor to interpolate against
    # and is emitted verbatim on flush (frame=None). `ratio_state["ratio"]` is
    # filled in by run()'s config_callback (which runs before any real frame
    # reaches this callback) once the source's real fps is known.
    #
    # `manifest_frames` (ADR-051), if given, is a plain list this function
    # appends to, in the EXACT order frames are emitted to the output video --
    # one dict per OUTPUT frame: {"real": True, "source_index": i} for a real
    # frame (i = its 0-based index in the original decoded input sequence), or
    # {"real": False, "nearest_real_index": i} for a synthetic frame (i = the
    # real source frame nearest to it by timestep, via _nearest_real_index()
    # above). Purely additive bookkeeping -- omitting manifest_frames (the
    # default) makes this byte-for-byte the same frame_callback as before this
    # feature existed; the returned/emitted video frames themselves are
    # completely unaffected by whether it's provided.
    state = {"pending": None, "pending_source_index": None, "pair_index": 0, "frames_out": 1}
    next_source_index = [0]

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            pending = state.pop("pending", None)
            pending_source_index = state.pop("pending_source_index", None)
            if pending is None:
                return None
            if manifest_frames is not None:
                manifest_frames.append({"real": True, "source_index": pending_source_index})
            return pending.squeeze(0)
        x = VU.to_tensor(frame, device=device).unsqueeze(0)
        cur_source_index = next_source_index[0]
        next_source_index[0] += 1
        pending = state.get("pending")
        if pending is None:
            state["pending"] = x
            state["pending_source_index"] = cur_source_index
            return None
        pending_source_index = state["pending_source_index"]
        timesteps, new_frames_out = _compute_pair_timesteps(
            state["frames_out"], state["pair_index"], ratio_state["ratio"])
        out_frames = [pending.squeeze(0)]
        if manifest_frames is not None:
            manifest_frames.append({"real": True, "source_index": pending_source_index})
        for t in timesteps:
            middle = interpolate_frame(model, pending, x, timestep=float(t), scale=1.0)
            out_frames.append(middle.squeeze(0))
            if manifest_frames is not None:
                nearest = _nearest_real_index(pending_source_index, cur_source_index, t)
                manifest_frames.append({"real": False, "nearest_real_index": nearest})
        state["pending"] = x
        state["pending_source_index"] = cur_source_index
        state["pair_index"] += 1
        state["frames_out"] = new_frames_out
        return out_frames

    return frame_callback


def _resolve_encoder_options(video_codec, gpu):
    """ffmpeg encoder options for RIFE's output stream, keyed by --video-codec (see
    create_parser). Mirrors, in miniature, the SAME per-codec-family option-naming
    convention the main iw3 conversion pipeline uses
    (iw3.utils.make_video_codec_option) -- libx264/libx265 take preset+crf,
    hevc_nvenc/h264_nvenc need constant-QP (rc/qp) instead of crf, hevc_qsv/h264_qsv
    add global_quality -- so real DV/HDR reinjection users encoding with libx265 or
    hevc_nvenc get sane, working encoder settings, not just a codec name with no
    matching options.

    Deliberately NOT imported from iw3.utils directly (rife_cli.py stays a
    lightweight, GPU/model-state-isolated subprocess by design -- see this file's
    own module docstring -- and iw3.utils pulls in the entire depth/stereo model
    factory stack at import time, which would defeat that). This is a small,
    self-contained duplicate of just the option-SHAPE logic actually needed here --
    no HDR/tune/profile-level knobs, since RIFE's own output has never exposed
    those and this option exists solely to unblock HEVC output for Dolby Vision
    reinjection downstream, not to replicate the main pipeline's full
    encoder-tuning surface.

    None/"libx264"/"libx265" (the default family, unchanged from before this
    option existed) keeps the exact original {"preset": "medium", "crf": "16"}
    options byte-for-byte -- backward compatibility for every existing caller that
    doesn't pass --video-codec at all."""
    if video_codec in (None, "libx264", "libx265"):
        return {"preset": "medium", "crf": "16"}
    if video_codec in ("hevc_nvenc", "h264_nvenc"):
        options = {"rc": "constqp", "qp": "16"}
        if gpu is not None and gpu >= 0:
            options["gpu"] = str(gpu)
        return options
    if video_codec in ("hevc_qsv", "h264_qsv"):
        return {"preset": "medium", "global_quality": "16"}
    # Unknown/other codec (e.g. hevc_amf, or a future one this project hasn't
    # explicitly tuned for) -- pass through with no extra options rather than
    # guessing at option names that can't be verified against real hardware here;
    # ffmpeg's own default settings for that encoder apply.
    return {}


def _build_output_config(target_fps, video_codec, gpu):
    """Builds the VideoOutputConfig for run()'s config_callback -- pulled out into
    its own function so --video-codec's effect on the real config object is
    directly unit-testable without needing a GPU/real video decode (see
    _test_rife_video_codec_options)."""
    return VU.VideoOutputConfig(
        fps=None,  # no input resampling -- every real decoded frame is kept
        output_fps=float(target_fps),
        video_codec=video_codec,
        options=_resolve_encoder_options(video_codec, gpu),
    )


def _rife_manifest_path(output_path):
    return f"{output_path}.rife_manifest.json"


def _write_rife_manifest(output_path, input_path, manifest_frames, rife_model,
                          rife_multiplier, rife_target_fps, fps_info):
    """Writes the small sidecar manifest (ADR-051) recording, for every OUTPUT
    frame RIFE just wrote, whether it's a real source frame or a RIFE-synthetic
    in-between frame, and (for synthetic frames) which real source frame is its
    nearest neighbor by timestep. Always written when RIFE runs (no opt-in flag
    -- this is cheap metadata, not an expensive extra step), so a later
    retroactive Dolby Vision RPU reinjection pass (`python -m
    iw3.reinject_hdr_cli --rife-manifest`) can expand the ORIGINAL source's RPU
    to exactly match this RIFE output's frame count without needing to
    re-derive frame correspondence after the fact.

    Written via a temp-file-then-replace so a reader can never observe a
    partially-written manifest (see docs/ai/CODING_STANDARDS.md CS-IO-001)."""
    source_frame_count = sum(1 for f in manifest_frames if f["real"])
    manifest = {
        "version": 1,
        "input": str(input_path),
        "output": str(output_path),
        "rife_model": rife_model,
        "rife_multiplier": rife_multiplier,
        "rife_target_fps": rife_target_fps,
        "orig_fps": fps_info.get("orig_fps"),
        "target_fps": fps_info.get("target_fps"),
        "source_frame_count": source_frame_count,
        "output_frame_count": len(manifest_frames),
        "frames": manifest_frames,
    }
    manifest_path = _rife_manifest_path(output_path)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    os.replace(tmp_path, manifest_path)
    print(f"[iw3.rife_cli] wrote RIFE frame manifest: {manifest_path} "
          f"({source_frame_count} real -> {len(manifest_frames)} output frames)",
          file=sys.stderr)
    return manifest_path


def run(input_path, output_path, rife_model=DEFAULT_RIFE_MODEL, gpu=0,
        rife_multiplier=None, rife_target_fps=None, video_codec=None):
    device = create_device(gpu)
    model = load_rife_model(rife_model, device)
    ratio_state = {"ratio": None}
    manifest_frames = []
    frame_callback = _make_frame_callback(model, device, ratio_state, manifest_frames)
    fps_info = {}

    def config_callback(sw_format):
        orig_fps = sw_format.get_fps()
        target_fps = _resolve_target_fps(orig_fps, rife_multiplier, rife_target_fps)
        orig_fps_frac = orig_fps if isinstance(orig_fps, Fraction) else Fraction(orig_fps)
        ratio_state["ratio"] = target_fps / orig_fps_frac
        fps_info["orig_fps"] = float(orig_fps_frac)
        fps_info["target_fps"] = float(target_fps)
        return _build_output_config(target_fps, video_codec, gpu)

    VU.process_video(
        input_path,
        output_path,
        frame_callback,
        config_callback=config_callback,
        title="RIFE",
        device=device,
    )

    _write_rife_manifest(output_path, input_path, manifest_frames, rife_model,
                          rife_multiplier, rife_target_fps, fps_info)


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


def _simulate_pairs(ratio, n_pairs):
    """Test helper: drives _compute_pair_timesteps sequentially (exactly like
    _make_frame_callback's real state machine does) over `n_pairs` real-frame
    pairs. Returns (list_of_timestep_lists, final_frames_out)."""
    frames_out = 1  # real frame 0
    all_timesteps = []
    for pair_index in range(n_pairs):
        timesteps, frames_out = _compute_pair_timesteps(frames_out, pair_index, ratio)
        all_timesteps.append(timesteps)
    return all_timesteps, frames_out


def _test_rife_multiplier_timesteps():
    """Regression test for docs/ai/AI_DECISIONS.md ADR-049's simple-multiplier
    scheduling math (2x/3x/4x): confirms _resolve_target_fps computes
    target_fps = orig_fps * N exactly, and _compute_pair_timesteps inserts
    exactly N-1 evenly-spaced timesteps (1/N, ..., (N-1)/N) for every real frame
    pair (an integer ratio never triggers the round-half-up drift correction,
    so this is constant across pairs). Also confirms the off-by-default no-op
    guarantee: with no flags given (both None), the resolved target fps is
    byte-for-byte identical to the original hardcoded `orig_fps * 2` behavior
    this replaces."""
    orig_fps = Fraction(24000, 1001)  # 23.976fps, a real value seen in ADR-047's own test file
    for multiplier in (2, 3, 4):
        target_fps = _resolve_target_fps(orig_fps, multiplier, None)
        assert target_fps == orig_fps * multiplier, (multiplier, target_fps)
        ratio = target_fps / orig_fps
        assert ratio == multiplier, (multiplier, ratio)
        all_timesteps, _ = _simulate_pairs(ratio, 5)
        expected = [Fraction(k, multiplier) for k in range(1, multiplier)]
        for pair_index, timesteps in enumerate(all_timesteps):
            assert timesteps == expected, (multiplier, pair_index, timesteps, expected)

    default_target = _resolve_target_fps(orig_fps, None, None)
    assert default_target == orig_fps * 2, default_target
    assert float(default_target) == float(orig_fps) * 2, \
        "off-by-default no-op broken: resolved target fps no longer matches the original hardcoded *2"
    default_ratio = default_target / orig_fps
    all_timesteps, _ = _simulate_pairs(default_ratio, 3)
    assert all(timesteps == [Fraction(1, 2)] for timesteps in all_timesteps), all_timesteps

    print("_test_rife_multiplier_timesteps: PASS")


def _test_rife_target_fps_scheduling():
    """Regression test for ADR-049's exact-target-fps scheduling math on a real
    non-integer-ratio case (24fps -> 60fps, ratio 2.5). Real source frames keep
    their original decoded order/position in the output sequence -- only their
    OWN original timestamps are not required to land on the 60fps grid (they
    generally won't, for a non-integer ratio: e.g. real frame 1 at 1/24s is at
    continuous output-grid position 1/24*60=2.5, not an integer -- expected,
    not a bug). What IS guaranteed, and checked here:
    (1) every SYNTHETIC frame's timestep is solved so its absolute timeline
    position lands exactly on the target_fps grid, verified against two pairs
    worked out by hand;
    (2) synthetic-frame count alternates 2,1,2,1,... (averaging ratio-1=1.5)
    rather than a naive per-pair-only formula's constant (and wrong) 2 every
    pair -- the real bug this drift-free design fixes, caught by this exact
    test while building it: a first, simpler implementation using a pure
    per-pair formula (real frame `i` assumed to sit exactly at continuous
    position `i*ratio`, independent of prior pairs' actual rounding) emitted a
    constant 2 synthetic frames every pair here, diverging unboundedly from the
    target duration over a long video instead of tracking it to within one
    frame;
    (3) cumulative emitted frame count after N pairs always stays within one
    frame of the ideal continuous count N*ratio+1, confirming no long-run
    drift -- the same guarantee FPSFilter's own expected_total_out bookkeeping
    provides in nunif/utils/video/video_filter/fps.py (this project's existing
    frame-retiming precedent for the scheduling approach)."""
    orig_fps = Fraction(24)
    target_fps = _resolve_target_fps(orig_fps, None, 60.0)
    assert target_fps == Fraction(60), target_fps
    ratio = target_fps / orig_fps
    assert ratio == Fraction(5, 2), ratio

    all_timesteps, _ = _simulate_pairs(ratio, 4)
    # Pair 0 (real frames at t=0, 1/24s): output grid points j=1,2 (of 60fps)
    # fall strictly inside -> local timesteps 1/2.5=0.4, 2/2.5=0.8.
    assert all_timesteps[0] == [Fraction(2, 5), Fraction(4, 5)], all_timesteps[0]
    # Pair 1 (real frame 1 landed at j=3, not the non-integer ideal 2.5): only
    # ONE synthetic frame is needed to reach real frame 2's target slot j=5.
    assert all_timesteps[1] == [Fraction(3, 5)], all_timesteps[1]
    # Real frame 2 landed exactly on-grid (j=5=2*ratio, integer) since ratio's
    # denominator is 2 -- the pattern repeats with period 2 from here.
    assert all_timesteps[2] == [Fraction(2, 5), Fraction(4, 5)], all_timesteps[2]
    assert all_timesteps[3] == [Fraction(3, 5)], all_timesteps[3]
    # Confirms the fix over the naive formula: alternates 2,1,2,1 (avg 1.5 =
    # ratio-1), not a constant 2 every pair.
    assert [len(t) for t in all_timesteps] == [2, 1, 2, 1]

    all_timesteps, final_frames_out = _simulate_pairs(ratio, 20)
    for pair_index, timesteps in enumerate(all_timesteps):
        for t in timesteps:
            assert 0 < t < 1, t
            # By construction t solves (pair_index + t) * ratio == j for some
            # integer j -- confirms the formula's own internal consistency.
            j = (pair_index + t) * ratio
            assert j == int(j), (pair_index, t, j)

    # No long-run drift: cumulative frame count after every pair stays within
    # one frame of the ideal continuous count.
    frames_out = 1
    for pair_index in range(50):
        timesteps, frames_out = _compute_pair_timesteps(frames_out, pair_index, ratio)
        ideal = Fraction(pair_index + 1) * ratio + 1
        assert abs(frames_out - ideal) <= 1, (pair_index, frames_out, ideal)

    print("_test_rife_target_fps_scheduling: PASS")


def _test_rife_fps_validation():
    """Regression test for ADR-049's validation rules: --rife-multiplier and
    --rife-target-fps are mutually exclusive, and --rife-target-fps must be
    genuinely higher than the source's own fps (not equal, not lower)."""
    orig_fps = Fraction(24000, 1001)  # ~23.976fps

    try:
        _resolve_target_fps(orig_fps, 3, 60.0)
        raise AssertionError("expected ValueError for --rife-multiplier + --rife-target-fps together")
    except ValueError as e:
        assert "mutually exclusive" in str(e), e

    try:
        _resolve_target_fps(orig_fps, None, 20.0)  # lower than source's ~23.976fps
        raise AssertionError("expected ValueError for --rife-target-fps below source fps")
    except ValueError as e:
        assert "must be higher" in str(e), e

    try:
        _resolve_target_fps(orig_fps, None, float(orig_fps))  # exactly equal
        raise AssertionError("expected ValueError for --rife-target-fps equal to source fps")
    except ValueError as e:
        assert "must be higher" in str(e), e

    # A genuinely higher target must succeed and resolve to the requested fps.
    assert _resolve_target_fps(orig_fps, None, 60.0) == Fraction(60)

    print("_test_rife_fps_validation: PASS")


def _test_rife_manifest_emission():
    """Regression/coverage test for ADR-051's RIFE frame manifest: drives the
    REAL, unmodified _make_frame_callback (interpolate_frame and VU.to_tensor
    mocked out -- CS-TEST-001, no GPU/real RIFE model needed) over a synthetic
    sequence of fake frames for both an integer multiplier (2x) and a non-2x
    multiplier (3x), confirming: every real frame's source_index is its
    correct 0-based position in the ORIGINAL input sequence, in order; every
    synthetic frame's nearest_real_index matches the t<0.5/t>=0.5 rule applied
    to the SAME timesteps _compute_pair_timesteps actually produced (checked
    by independently reconstructing the expected manifest from
    _compute_pair_timesteps + _nearest_real_index -- the same building blocks
    production uses -- and comparing byte-for-byte against what the real,
    unmodified callback produced, so this can't silently drift from the real
    scheduling logic); and manifest_frames' real/synthetic counts match the
    expected N-1-per-gap count for an integer multiplier N."""
    from unittest.mock import patch

    def fake_interpolate_frame(model, img0, img1, timestep=0.5, scale=1.0):
        return img0.clone()

    def _run_fake_rife(source_frame_count, ratio):
        manifest_frames = []
        ratio_state = {"ratio": ratio}
        with patch(f"{__name__}.interpolate_frame", fake_interpolate_frame), \
             patch.object(VU, "to_tensor", lambda frame, device=None: torch.zeros(1, 2, 2)):
            frame_callback = _make_frame_callback(
                model=None, device=torch.device("cpu"), ratio_state=ratio_state,
                manifest_frames=manifest_frames)
            for _ in range(source_frame_count):
                frame_callback(object())  # any non-None sentinel "decoded frame"
            frame_callback(None)  # flush the last pending real frame
        return manifest_frames

    for source_frame_count, multiplier in ((8, 2), (7, 3)):
        ratio = Fraction(multiplier)
        manifest_frames = _run_fake_rife(source_frame_count, ratio)

        real_entries = [f for f in manifest_frames if f["real"]]
        synth_entries = [f for f in manifest_frames if not f["real"]]
        assert [e["source_index"] for e in real_entries] == list(range(source_frame_count)), \
            (multiplier, real_entries)
        assert len(synth_entries) == (source_frame_count - 1) * (multiplier - 1), \
            (multiplier, len(synth_entries))

        expected = []
        frames_out = 1
        for pair_index in range(source_frame_count - 1):
            expected.append({"real": True, "source_index": pair_index})
            timesteps, frames_out = _compute_pair_timesteps(frames_out, pair_index, ratio)
            for t in timesteps:
                nearest = _nearest_real_index(pair_index, pair_index + 1, t)
                expected.append({"real": False, "nearest_real_index": nearest})
        expected.append({"real": True, "source_index": source_frame_count - 1})
        assert manifest_frames == expected, (multiplier, manifest_frames, expected)

    print("_test_rife_manifest_emission: PASS")


def _test_rife_video_codec_options():
    """Regression test for the real, confirmed fix (2026-09-08, see
    docs/ai/AI_DECISIONS.md ADR-051/ADR-064 amendments) for the bug that RIFE could
    never output HEVC at all -- confirmed directly via ffprobe against a real RIFE
    output file (codec_name: h264, not hevc), root-caused to rife_cli.py having no
    --video-codec option and its VideoOutputConfig never setting video_codec, so
    nunif.utils.video.utils.get_default_video_codec always filled in libx264.

    Backward compatibility is the critical property here: None (the default when
    --video-codec is not passed -- every existing caller/script) must produce
    BYTE-IDENTICAL options to what this function returned before this fix existed,
    and _build_output_config's video_codec must stay None in that case too (so
    get_default_video_codec's own existing libx264 fallback is untouched)."""
    # Backward-compat: default (None) and libx264 unchanged from before this option
    # existed.
    assert _resolve_encoder_options(None, 0) == {"preset": "medium", "crf": "16"}
    assert _resolve_encoder_options("libx264", 0) == {"preset": "medium", "crf": "16"}
    assert _resolve_encoder_options("libx265", 0) == {"preset": "medium", "crf": "16"}

    # nvenc needs constant-QP (rc/qp), not crf, plus an explicit gpu index -- matching
    # iw3.utils.make_video_codec_option's own real per-codec option shape.
    assert _resolve_encoder_options("hevc_nvenc", 1) == {"rc": "constqp", "qp": "16", "gpu": "1"}
    assert _resolve_encoder_options("h264_nvenc", 0) == {"rc": "constqp", "qp": "16", "gpu": "0"}
    # CPU sentinel (-1) or no gpu info -> no "gpu" option at all (never a negative
    # device index passed to the encoder).
    assert _resolve_encoder_options("hevc_nvenc", -1) == {"rc": "constqp", "qp": "16"}
    assert _resolve_encoder_options("hevc_nvenc", None) == {"rc": "constqp", "qp": "16"}

    assert _resolve_encoder_options("hevc_qsv", 0) == {"preset": "medium", "global_quality": "16"}
    assert _resolve_encoder_options("h264_qsv", 0) == {"preset": "medium", "global_quality": "16"}

    # Unrecognized/untuned codec -- no guessed options, ffmpeg's own defaults apply.
    assert _resolve_encoder_options("hevc_amf", 0) == {}

    # _build_output_config: video_codec passes straight through to VideoOutputConfig
    # (None when unset -- the existing get_default_video_codec fallback in
    # nunif.utils.video.processor still applies exactly as before), and the fps/
    # options wiring stays correct.
    cfg_default = _build_output_config(Fraction(48), None, 0)
    assert cfg_default.video_codec is None
    assert cfg_default.options == {"preset": "medium", "crf": "16"}
    assert cfg_default.output_fps == 48.0
    assert cfg_default.fps is None

    cfg_hevc = _build_output_config(Fraction(48), "libx265", 0)
    assert cfg_hevc.video_codec == "libx265"
    assert cfg_hevc.options == {"preset": "medium", "crf": "16"}

    cfg_nvenc = _build_output_config(Fraction(48), "hevc_nvenc", 0)
    assert cfg_nvenc.video_codec == "hevc_nvenc"
    assert cfg_nvenc.options == {"rc": "constqp", "qp": "16", "gpu": "0"}

    print("_test_rife_video_codec_options: PASS")


def _run_self_tests():
    _test_ensure_rife_model_downloads_model_package()
    _test_rife_multiplier_timesteps()
    _test_rife_target_fps_scheduling()
    _test_rife_fps_validation()
    _test_rife_manifest_emission()
    _test_rife_video_codec_options()
    print("All iw3.rife_cli self-tests PASSED")


def main(argv=None):
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
        return
    args = create_parser().parse_args(argv)
    run(args.input, args.output, rife_model=args.rife_model, gpu=args.gpu,
        rife_multiplier=args.rife_multiplier, rife_target_fps=args.rife_target_fps,
        video_codec=args.video_codec)


if __name__ == "__main__":
    sys.exit(main())
