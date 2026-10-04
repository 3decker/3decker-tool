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
import time
from fractions import Fraction

import torch

import nunif.utils.video as VU
from nunif.device import create_device
from nunif.utils.video.metadata import convert_fps_fraction
from .rife_model import DEFAULT_RIFE_MODEL, RIFE_TIERS, interpolate_frame, load_rife_model
from .utils import read_source_comment_metadata


class _SubprocessProgressPrinter:
    """tqdm-compatible progress reporter for use ACROSS a process boundary -- exact
    copy of sharpen_cli.py's own _SubprocessProgressPrinter (see that module's own
    docstring for the full "why a separate stdout channel, why throttled at
    ~0.1s" reasoning, ADR-170), with only the line prefix changed
    so a caller reading both tools' output at once (e.g. iw3.gui, in principle)
    could never confuse one tool's progress line for the other's: "IW3_RIFE_PROGRESS
    <done> <total>", to stdout, flushed immediately, never mixed with the
    human-readable status/refusal text this tool has always printed to stderr."""

    _MIN_INTERVAL_SEC = 0.1

    def __init__(self, **kwargs):
        self.total = kwargs.get("total") or 0
        self.done = 0
        self._last_emit = 0.0
        self._emit(force=True)

    def update(self, n=1):
        self.done += n
        self._emit()

    def close(self):
        self._emit(force=True)

    def _emit(self, force=False):
        now = time.monotonic()
        if not force and (now - self._last_emit) < self._MIN_INTERVAL_SEC:
            return
        self._last_emit = now
        print(f"IW3_RIFE_PROGRESS {self.done} {self.total}", flush=True)


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
    parser.add_argument("--scene-cut-times-file", type=str, default=None,
                         help="path to a JSON file containing a list of scene-cut times, in seconds, "
                              "relative to THIS video's own start (--input's frame 0). When given, "
                              "RIFE skips flow-warp blending across any real-frame pair a cut falls "
                              "between, duplicating the nearest real frame instead -- avoids the "
                              "morphing artifact RIFE (and any frame-interpolation model) otherwise "
                              "produces across a hard scene cut, where the two real frames show "
                              "completely unrelated content. Written automatically by the main iw3 "
                              "pipeline (iw3.utils._run_rife_interpolation) from its own existing "
                              "scene-boundary detection cache, when one is already available for the "
                              "source this job converted -- not intended to be set by hand. Omitted "
                              "(the default): unchanged prior behavior, every real-frame pair is "
                              "interpolated unconditionally.")
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


def _compute_cut_pair_indices(scene_cut_times, orig_fps):
    """Translates scene-cut times (seconds since this video's own start -- see
    create_parser's --scene-cut-times-file help, and iw3.utils._load_rife_scene_cut_times
    for how the caller derives them) into the set of real-frame PAIR indices (the
    pair connecting real frame i and real frame i+1, using the exact same 0-based
    `pair_index` _make_frame_callback's own state machine already tracks) that a
    hard scene cut falls inside.

    nunif.utils.shot_boundary_detection.detect_boundary's own pts convention is
    "the end point of the segment, not the starting point" (see that module's
    NOTE) -- i.e. a detected cut's frame index IS the last real frame of the
    OUTGOING scene, so the pair to protect is exactly (that index, that index + 1)
    == pair_index == that same index. `round()` recovers the original integer
    frame index from its float-seconds form (seconds = index / scan_fps) --
    tolerant of ordinary float round-tripping, valid as long as scan_fps and this
    video's own orig_fps are the same effective frame rate (true whenever the
    real conversion's own output fps was capped identically to the scan -- see
    docs/ai/AI_DECISIONS.md ADR-059 / resolve_scene_scan_fps)."""
    orig_fps_f = float(orig_fps)
    return {round(t * orig_fps_f) for t in (scene_cut_times or [])}


# Duplicate-source-frame detection (real feature, 2026-10-03): real research comparing
# this project's RIFE integration against Flowframes surfaced a documented equivalent --
# Flowframes' own "Intelligent frame de-duplication" skips real flow-warp interpolation
# between two consecutive real frames that are visually indistinguishable (a held frame
# in 2D animation, or telecine/pulldown-style duplication), since there is no real motion
# there for the model to interpolate -- any synthetic frame between them is already known,
# with certainty, to look like both of them.
#
# Exact pixel equality (torch.equal) was rejected: real encoded video is essentially never
# byte-identical even for a genuinely duplicated source frame (independent per-frame
# compression noise), so it would catch almost nothing real. A near-equality check is used
# instead, but deliberately kept VERY tight -- unlike the scene-cut protection above (whose
# cut times come from an already-CONFIRMED detector, so a false positive there is impossible
# by construction), a learned pixel-difference threshold here could genuinely misfire, and
# the cost of a false positive (skipping real interpolation that should have happened) is a
# real quality regression, not just a missed optimization -- so this is tuned to
# under-detect (miss some real duplicate pairs, costing only a smaller time saving) rather
# than over-detect.
#
# Thresholds are in normalized [0, 1] float units (VU.to_tensor's own output range --
# 1/255 ~= 0.0039, one 8-bit level):
#   - mean absolute difference < 0.004 (about 1 level): the WHOLE frame, on average, must
#     look essentially unchanged -- genuine motion of any meaningful size pushes this well
#     above a single frame's own compression-noise floor.
#   - max absolute difference < 0.02 (about 5 levels): guards the opposite failure mode --
#     a small, high-contrast moving object could keep the MEAN difference low (diluted by a
#     mostly-static background) while still being real, visible motion; capping the single
#     worst pixel too catches that case and refuses to call the pair a duplicate.
# Both must hold (AND, not OR) -- stacking them is strictly more conservative than either
# alone.
#
# On this project's own primary real use case (live-action movies), ordinary sensor/film
# grain alone generates frame-to-frame noise above this bar even for a genuinely static
# shot, so this is expected to fire on almost nothing for that content -- exactly as
# intended (Flowframes documents the same "stays off for camera footage" behavior, for the
# same underlying reason: real camera noise defeats near-exact equality on its own). It is
# expected to fire mainly on 2D animation's held-frame duplicates and literal
# telecine/pulldown duplicate frames, where the two real frames originate from identical
# source content. Unconditional (no new user-facing toggle, matching ADR-325's own
# precedent): at these thresholds a false positive on real content is not a realistic risk,
# so this needs no opt-in/opt-out of its own.
_DUPLICATE_MEAN_DIFF_THRESHOLD = 0.004
_DUPLICATE_MAX_DIFF_THRESHOLD = 0.02


def _is_near_duplicate_pair(pending, x):
    """True when two consecutive REAL frame tensors (CHW/NCHW float32 in [0, 1], same
    shape -- see VU.to_tensor) are close enough to call visually indistinguishable, per
    the conservative dual mean+max threshold explained in the block comment above."""
    diff = (pending - x).abs()
    return (diff.mean().item() < _DUPLICATE_MEAN_DIFF_THRESHOLD
            and diff.max().item() < _DUPLICATE_MAX_DIFF_THRESHOLD)


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
    #
    # Scene-cut protection (real bug fix, 2026-10-03, see --scene-cut-times-file):
    # `ratio_state["cut_pair_indices"]`, if set, is filled in by run()'s
    # config_callback the same way `ratio_state["ratio"]` already is -- a set of
    # `pair_index` values a real hard scene cut falls inside (see
    # _compute_cut_pair_indices). For those pairs ONLY, every synthetic frame is
    # a duplicate of its nearest real neighbor instead of an actual RIFE
    # flow-warp blend -- RIFE blending two frames from unrelated scenes produces
    # a visible morph/warp across the cut (the same artifact Flowframes' own
    # "Fix Scene Changes" feature exists to avoid); a duplicated frame at a cut
    # looks exactly like an ordinary un-interpolated hard cut always has (an
    # instant change, not a smooth one). Every other (non-cut) pair is completely
    # unaffected -- omitting --scene-cut-times-file (the default) makes this
    # byte-for-byte the same frame_callback as before this fix existed.
    #
    # Duplicate-source-frame detection (real feature, 2026-10-03, see
    # _is_near_duplicate_pair above): independent of the cut-pair mechanism, a pair
    # whose two real frames are near-pixel-identical (per that function's conservative
    # threshold) is ALSO cloned instead of interpolated -- same clone-instead-of-blend
    # action as a cut pair, flagged separately in the manifest (duplicate_source_frame,
    # vs. scene_cut_duplicate) since it's a different real reason. The two compose
    # cleanly: the near-duplicate check only runs for a pair that isn't ALREADY a known
    # cut pair (a confirmed cut and a learned pixel-similarity guess are never both
    # computed/trusted for the same pair -- the confirmed cut wins, and checking
    # similarity on it would be wasted work since the outcome is identical either way).
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
        cut_pair_indices = ratio_state.get("cut_pair_indices")
        is_cut_pair = bool(cut_pair_indices) and state["pair_index"] in cut_pair_indices
        # Only checked when NOT already a confirmed cut pair -- see the block comment
        # above _make_frame_callback: a confirmed cut always wins, and the outcome
        # (clone instead of blend) is identical either way, so checking similarity on
        # a cut pair would just be wasted tensor work.
        is_duplicate_pair = (not is_cut_pair) and _is_near_duplicate_pair(pending, x)
        skip_interpolation = is_cut_pair or is_duplicate_pair
        for t in timesteps:
            nearest = _nearest_real_index(pending_source_index, cur_source_index, t)
            if skip_interpolation:
                source_tensor = pending if nearest == pending_source_index else x
                middle = source_tensor.clone()
            else:
                middle = interpolate_frame(model, pending, x, timestep=float(t), scale=1.0)
            out_frames.append(middle.squeeze(0))
            if manifest_frames is not None:
                entry = {"real": False, "nearest_real_index": nearest}
                if is_cut_pair:
                    entry["scene_cut_duplicate"] = True
                if is_duplicate_pair:
                    entry["duplicate_source_frame"] = True
                manifest_frames.append(entry)
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


_HEVC_FAMILY = ("libx265", "hevc_nvenc", "hevc_qsv", "hevc_amf")


def _output_pix_fmt(high_bit_source, video_codec):
    """10-bit output for a 10-bit source when the codec is HEVC. RIFE used to always write 8-bit, which put
    visible banding into 10-bit / HDR (PQ) movies. H.264 stays 8-bit on purpose: 10-bit H.264 is poorly
    supported by TVs and hardware players, and HDR needs HEVC anyway."""
    return "yuv420p10le" if (high_bit_source and video_codec in _HEVC_FAMILY) else "yuv420p"


def _build_output_config(target_fps, video_codec, gpu, high_bit_source=False, comment=None):
    """Builds the VideoOutputConfig for run()'s config_callback -- pulled out into
    its own function so --video-codec's effect on the real config object is
    directly unit-testable without needing a GPU/real video decode (see
    _test_rife_video_codec_options).

    comment (ADR-269): the source file's own iw3_* settings COMMENT tag, read back
    by iw3.utils.read_source_comment_metadata() before this config is built, and
    carried into the new output here -- RIFE previously never set any metadata at
    all, so every RIFE-processed file silently lost its settings tag. None (no tag
    on the source, or a non-iw3 source) keeps prior behavior exactly."""
    return VU.VideoOutputConfig(
        pix_fmt=_output_pix_fmt(high_bit_source, video_codec),
        fps=None,  # no input resampling -- every real decoded frame is kept
        output_fps=float(target_fps),
        video_codec=video_codec,
        options=_resolve_encoder_options(video_codec, gpu),
        metadata={"comment": comment} if comment else {},
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


def _load_scene_cut_times_file(scene_cut_times_file):
    """Reads --scene-cut-times-file (see create_parser), tolerating any failure
    (missing/unreadable/corrupt file) by returning None -- a bad or missing file
    here must never crash a RIFE run; it should just fall back to the original,
    unprotected behavior for every pair, exactly as if the flag had been omitted."""
    if not scene_cut_times_file:
        return None
    try:
        with open(scene_cut_times_file, mode="r", encoding="utf-8") as f:
            times = json.load(f)
        times = [float(t) for t in times]
        print(f"[iw3.rife_cli] loaded {len(times)} scene-cut time(s) for cut-aware interpolation "
              f"from {scene_cut_times_file}", file=sys.stderr)
        return times
    except Exception as e:
        print(f"[iw3.rife_cli] could not read --scene-cut-times-file '{scene_cut_times_file}' "
              f"({e.__class__.__name__}: {e}) -- continuing without scene-cut protection.",
              file=sys.stderr)
        return None


def run(input_path, output_path, rife_model=DEFAULT_RIFE_MODEL, gpu=0,
        rife_multiplier=None, rife_target_fps=None, video_codec=None,
        scene_cut_times_file=None):
    device = create_device(gpu)
    model = load_rife_model(rife_model, device)
    ratio_state = {"ratio": None, "cut_pair_indices": None}
    scene_cut_times = _load_scene_cut_times_file(scene_cut_times_file)
    manifest_frames = []
    frame_callback = _make_frame_callback(model, device, ratio_state, manifest_frames)
    fps_info = {}
    # ADR-269: read once, up front -- input_path never changes mid-run, and this way
    # a source-read failure (logged inside the helper as None) can't happen more than
    # once per job.
    source_comment = read_source_comment_metadata(input_path)

    def config_callback(sw_format):
        orig_fps = sw_format.get_fps()
        target_fps = _resolve_target_fps(orig_fps, rife_multiplier, rife_target_fps)
        orig_fps_frac = orig_fps if isinstance(orig_fps, Fraction) else Fraction(orig_fps)
        ratio_state["ratio"] = target_fps / orig_fps_frac
        ratio_state["cut_pair_indices"] = _compute_cut_pair_indices(scene_cut_times, orig_fps_frac)
        fps_info["orig_fps"] = float(orig_fps_frac)
        fps_info["target_fps"] = float(target_fps)
        return _build_output_config(target_fps, video_codec, gpu,
                                    high_bit_source=bool(getattr(sw_format, "use_16bit", False)),
                                    comment=source_comment)

    VU.process_video(
        input_path,
        output_path,
        frame_callback,
        config_callback=config_callback,
        title="RIFE",
        device=device,
        tqdm_fn=_SubprocessProgressPrinter,
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


def _test_rife_cpu_device_not_overridden_by_cuda_availability():
    """Regression test for a real, confirmed-live bug (2026-09-11, see
    docs/ai/AI_DECISIONS.md ADR-119): the official, downloaded RIFE_HDv3.py
    hardcodes its OWN module-level `device = torch.device("cuda" if
    torch.cuda.is_available() else "cpu")`, and Model.__init__/Model.device()
    move the flownet onto THAT global -- not onto whatever device
    rife_model.load_rife_model()'s own caller actually requested. On a
    CUDA-equipped machine, torch.cuda.is_available() is True regardless of a
    caller asking for CPU (--gpu -1), so a "CPU-only" run silently tried to use
    CUDA anyway -- confirmed live as a real hang when the GPU was already heavily
    loaded by something else (a --gpu -1 run stalled identically to a --gpu 0 run,
    even though it should never have touched the GPU at all).

    Synthetic (CS-TEST-001): builds a fake `train_log.RIFE_HDv3` module shaped
    exactly like the real one for this specific bug -- a module-level `device`
    global that ALWAYS resolves to "cuda" (simulating torch.cuda.is_available()
    being True), and a `Model` class whose device()/load_model() read that same
    module global the same way the real file does. No real GPU, network, or model
    weights needed -- this isolates the device-propagation bug from everything
    else load_rife_model() does."""
    import types
    from unittest.mock import patch
    from . import rife_model as rm

    fake_module = types.ModuleType("train_log.RIFE_HDv3")
    fake_module.device = torch.device("cuda")  # simulates torch.cuda.is_available() == True

    class FakeModel:
        def __init__(self):
            self.placed_device = None
            self.device()  # real Model.__init__ calls self.device() unconditionally

        def device(self):
            # Real RIFE_HDv3.Model.device(): self.flownet.to(device) -- reads the
            # MODULE global, not a device this class was ever given directly.
            self.placed_device = fake_module.device

        def load_model(self, path, rank=0):
            pass  # no real weights needed for this test

        def eval(self):
            pass

    fake_module.Model = FakeModel

    # load_rife_model() actively clears any existing "train_log"/"train_log.*"
    # sys.modules entries and does a real importlib.import_module() call -- so
    # this test mocks import_module itself (returning the fake module for
    # "train_log.RIFE_HDv3", real behavior for anything else) rather than
    # pre-seeding sys.modules, which load_rife_model() would just delete.
    real_import_module = rm.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "train_log.RIFE_HDv3":
            return fake_module
        return real_import_module(name, *args, **kwargs)

    with patch.object(rm, "ensure_rife_model", lambda tier, show_progress=True: "/fake/train_log"), \
         patch.object(rm.importlib, "import_module", fake_import_module):
        model = rm.load_rife_model("rife_425", torch.device("cpu"))
        assert model.placed_device == torch.device("cpu"), (
            f"load_rife_model(device=cpu) placed the model on {model.placed_device} instead -- "
            f"the module-level device global was not overridden before Model() construction")

    print("_test_rife_cpu_device_not_overridden_by_cuda_availability: PASS")


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
        # Each fake "decoded frame" gets DISTINCT content (a running counter), not a
        # constant zero tensor -- pre-existing test-fixture gap uncovered while adding
        # the real 2026-10-03 duplicate-source-frame feature (_is_near_duplicate_pair):
        # every pair here is otherwise pixel-identical by construction, so it would now
        # ALSO get flagged duplicate_source_frame in every entry, which is not what this
        # test is checking (that has its own dedicated test,
        # _test_rife_duplicate_frame_detection) and broke this one's exact-manifest
        # equality assertion below. No production code touched for this.
        counter = [0.0]

        def fake_to_tensor(frame, device=None):
            counter[0] += 1.0
            return torch.full((1, 2, 2), counter[0])

        with patch(f"{__name__}.interpolate_frame", fake_interpolate_frame), \
             patch.object(VU, "to_tensor", fake_to_tensor):
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


def _test_rife_scene_cut_protection():
    """Regression test for the real 2026-10-03 fix: RIFE had no scene-cut
    awareness at all, so it blended across hard cuts, producing a visible
    morph/warp between two unrelated images (the same artifact Flowframes'
    "Fix Scene Changes" feature exists to avoid -- see rife_cli.py's and
    iw3.utils._load_rife_scene_cut_times's own module-level docs for the full
    design).

    Part 1: _compute_cut_pair_indices correctly recovers the exact integer
    frame-pair index from a cut time expressed in seconds (the form
    iw3.utils._load_rife_scene_cut_times actually produces).

    Part 2: drives the REAL, unmodified _make_frame_callback (CS-TEST-001:
    interpolate_frame mocked to return an obviously-distinguishable SENTINEL
    value no real duplicate could ever coincidentally equal; VU.to_tensor mocked
    to turn each fake "decoded frame" -- a plain int standing in for its own
    source index -- into a tensor filled with that same number, so a duplicated
    frame's exact VALUE proves which real frame it came from) over a synthetic
    8-real-frame, 2x sequence with a cut set between real frames 3 and 4
    (pair_index 3): confirms every synthetic frame OUTSIDE that pair is still a
    genuine RIFE interpolation (the sentinel) exactly as before this fix, the
    synthetic frame INSIDE that pair is instead an exact duplicate of its
    nearest real neighbor's own content (never the sentinel), that duplicate's
    manifest entry is flagged scene_cut_duplicate=True, and every other
    synthetic entry has no such flag -- i.e. this is a surgical, pair-scoped
    change with no effect on normal (non-cut) interpolation."""
    from unittest.mock import patch

    # Part 1: time -> pair_index translation.
    assert _compute_cut_pair_indices([0.3], Fraction(10)) == {3}
    assert _compute_cut_pair_indices([0.0, 1.0, 2.5], Fraction(10)) == {0, 10, 25}
    assert _compute_cut_pair_indices(None, Fraction(10)) == set()
    assert _compute_cut_pair_indices([], Fraction(10)) == set()

    # Part 2: end-to-end frame_callback behavior.
    SENTINEL = 999.0

    def fake_interpolate_frame(model, img0, img1, timestep=0.5, scale=1.0):
        return torch.full_like(img0, SENTINEL)

    def fake_to_tensor(frame, device=None):
        return torch.full((1, 2, 2), float(frame))

    source_frame_count = 8
    ratio = Fraction(2)  # integer 2x -> exactly one synthetic frame per pair, at t=0.5
    cut_pair_indices = {3}  # the pair connecting real frame 3 and real frame 4
    manifest_frames = []
    ratio_state = {"ratio": ratio, "cut_pair_indices": cut_pair_indices}

    outputs = []
    with patch(f"{__name__}.interpolate_frame", fake_interpolate_frame), \
         patch.object(VU, "to_tensor", fake_to_tensor):
        frame_callback = _make_frame_callback(
            model=None, device=torch.device("cpu"), ratio_state=ratio_state,
            manifest_frames=manifest_frames)
        for i in range(source_frame_count):
            result = frame_callback(i)
            if result is not None:
                outputs.extend(result)
        last = frame_callback(None)
        if last is not None:
            outputs.append(last)

    assert len(outputs) == len(manifest_frames), (len(outputs), len(manifest_frames))

    synth_seen = 0
    cut_synth_seen = 0
    for out_tensor, entry in zip(outputs, manifest_frames):
        if entry["real"]:
            continue
        synth_seen += 1
        value = out_tensor.flatten()[0].item()
        if entry.get("scene_cut_duplicate"):
            cut_synth_seen += 1
            # Must be an exact duplicate of its nearest real neighbor's own
            # content (4.0, per the t=0.5/nearest=cur_source_index rule below),
            # never the interpolation sentinel.
            assert value == entry["nearest_real_index"], (entry, value)
            assert value != SENTINEL, (entry, value)
        else:
            # Every normal (non-cut) synthetic frame is still a genuine RIFE
            # interpolation, completely unaffected by this feature.
            assert value == SENTINEL, (entry, value)

    # 7 pairs total (8 real frames), exactly 1 of them is the cut pair.
    assert synth_seen == 7, synth_seen
    assert cut_synth_seen == 1, cut_synth_seen

    cut_entry = next(e for e in manifest_frames if e.get("scene_cut_duplicate"))
    # t=0.5 is NOT < 0.5, so _nearest_real_index resolves to the LATER real
    # frame (cur_source_index = 4) for pair_index 3 (real frames 3 and 4).
    assert cut_entry["nearest_real_index"] == 4, cut_entry

    print("_test_rife_scene_cut_protection: PASS")


def _test_rife_duplicate_frame_detection():
    """Regression test for the real 2026-10-03 duplicate-source-frame feature (see
    _is_near_duplicate_pair's own block comment for the full conservative-threshold
    design, modeled on Flowframes' documented "Intelligent frame de-duplication").

    Part 1: _is_near_duplicate_pair's threshold behavior in isolation, including
    values just inside and just outside both the mean and max bounds -- confirms
    this is a genuine AND of both conditions, not either alone:
      - exact zero difference -> duplicate (the easy case).
      - just under both thresholds -> duplicate.
      - mean difference just OVER its threshold -> NOT a duplicate (strict '<',
        confirms the boundary is exclusive, not inclusive).
      - a single outlier pixel whose OWN difference is clearly over the max
        threshold, diluted to a tiny mean by many unchanged pixels (the "small
        moving object" failure mode the max check exists to catch) -> NOT a
        duplicate, even though the mean alone would have passed.

    Part 2: drives the REAL, unmodified _make_frame_callback (same CS-TEST-001
    sentinel-value technique as _test_rife_scene_cut_protection) over a synthetic
    8-real-frame, 2x sequence whose content is [0, 1, 2, 3, 3, 4, 5, 5] -- i.e. real
    frames 3-4 and 6-7 carry IDENTICAL content (a held/duplicated source frame),
    every other consecutive pair genuinely differs. Pair_index 3 (connecting real
    frames 3 and 4) is ALSO put in cut_pair_indices, to confirm composition with the
    scene-cut mechanism: a pair that is both is cloned (as it always would be
    either way) and flagged ONLY scene_cut_duplicate, never ALSO
    duplicate_source_frame (the near-duplicate check is short-circuited for a
    confirmed cut pair -- see _make_frame_callback's own comment). Pair_index 6
    (connecting real frames 6 and 7) is a near-duplicate with NO cut involved at
    all, confirming the mechanism also works standalone, flagged
    duplicate_source_frame with no scene_cut_duplicate. Every other pair (0, 1, 2,
    4, 5) genuinely differs and must still produce a real interpolation (the
    sentinel), completely unaffected by this feature."""
    from unittest.mock import patch

    # Part 1: threshold behavior in isolation. Margins are kept well clear of the
    # exact threshold float value on purpose (not hit bit-for-bit) -- float32
    # rounding of an arbitrary decimal threshold can land a fraction of a ULP on
    # either side of the python-float literal, which would make a bit-exact
    # boundary check flaky for reasons that have nothing to do with the real
    # behavior being verified (strict '<', not a specific rounding direction).
    zeros = torch.zeros(1, 1, 2, 2)
    assert _is_near_duplicate_pair(zeros, zeros.clone())

    just_under = torch.full((1, 1, 2, 2), _DUPLICATE_MEAN_DIFF_THRESHOLD - 0.0015)
    assert _is_near_duplicate_pair(zeros, just_under)

    just_over_mean_threshold = torch.full((1, 1, 2, 2), _DUPLICATE_MEAN_DIFF_THRESHOLD + 0.0015)
    assert not _is_near_duplicate_pair(zeros, just_over_mean_threshold)

    # 100 pixels: 99 unchanged, 1 outlier clearly above the max threshold -- mean
    # stays tiny (diluted), but the single worst pixel must still refuse the
    # "duplicate" call (the small-moving-object case the max check exists for).
    outlier = torch.zeros(1, 1, 10, 10)
    outlier_x = outlier.clone()
    outlier_x[0, 0, 0, 0] = _DUPLICATE_MAX_DIFF_THRESHOLD + 0.005
    outlier_mean = outlier_x.abs().mean().item()
    assert outlier_mean < _DUPLICATE_MEAN_DIFF_THRESHOLD, outlier_mean  # confirms mean alone would have passed
    assert not _is_near_duplicate_pair(outlier, outlier_x)

    # Part 2: end-to-end frame_callback behavior + composition with a cut pair.
    SENTINEL = 999.0

    def fake_interpolate_frame(model, img0, img1, timestep=0.5, scale=1.0):
        return torch.full_like(img0, SENTINEL)

    def fake_to_tensor(frame, device=None):
        return torch.full((1, 2, 2), float(frame))

    content = [0, 1, 2, 3, 3, 4, 5, 5]
    cut_pair_indices = {3}  # pair 3 (real frames 3-4) is ALSO a confirmed scene cut
    manifest_frames = []
    ratio_state = {"ratio": Fraction(2), "cut_pair_indices": cut_pair_indices}

    outputs = []
    with patch(f"{__name__}.interpolate_frame", fake_interpolate_frame), \
         patch.object(VU, "to_tensor", fake_to_tensor):
        frame_callback = _make_frame_callback(
            model=None, device=torch.device("cpu"), ratio_state=ratio_state,
            manifest_frames=manifest_frames)
        for value in content:
            result = frame_callback(value)
            if result is not None:
                outputs.extend(result)
        last = frame_callback(None)
        if last is not None:
            outputs.append(last)

    assert len(outputs) == len(manifest_frames), (len(outputs), len(manifest_frames))

    # Each "real" manifest entry (source_index i) is logged immediately BEFORE the
    # synthetic entries for pair i (pending frame i is the earlier half of that
    # pair) -- so source_index doubles exactly as the pair index its following
    # synthetic entries belong to.
    synth_by_pair = {}
    current_pair = None
    for out_tensor, entry in zip(outputs, manifest_frames):
        value = out_tensor.flatten()[0].item()
        if entry["real"]:
            current_pair = entry["source_index"]
            continue
        synth_by_pair[current_pair] = (value, entry)

    # Pairs 0, 1, 2, 4, 5: genuinely different content -> real interpolation, no flags.
    for p in (0, 1, 2, 4, 5):
        value, entry = synth_by_pair[p]
        assert value == SENTINEL, (p, value)
        assert "scene_cut_duplicate" not in entry, (p, entry)
        assert "duplicate_source_frame" not in entry, (p, entry)

    # Pair 3: cut AND near-duplicate -- cloned (content[4] == 3, never the sentinel),
    # flagged scene_cut_duplicate ONLY (the near-duplicate check never even ran).
    value, entry = synth_by_pair[3]
    assert value == content[4], (value, entry)
    assert value != SENTINEL, (value, entry)
    assert entry.get("scene_cut_duplicate") is True, entry
    assert "duplicate_source_frame" not in entry, entry

    # Pair 6: near-duplicate only, no cut involved -- cloned (content[7] == 5, never
    # the sentinel), flagged duplicate_source_frame ONLY.
    value, entry = synth_by_pair[6]
    assert value == content[7], (value, entry)
    assert value != SENTINEL, (value, entry)
    assert entry.get("duplicate_source_frame") is True, entry
    assert "scene_cut_duplicate" not in entry, entry

    print("_test_rife_duplicate_frame_detection: PASS")


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


def _test_rife_output_pix_fmt():
    """RIFE used to always write 8-bit, putting banding into 10-bit/HDR movies. 10-bit source + HEVC codec ->
    yuv420p10le; H.264/default codec or 8-bit source -> yuv420p (unchanged); old callers (no flag) unchanged."""
    cfg = lambda codec, high: _build_output_config(Fraction(48), codec, 0, high_bit_source=high).pix_fmt  # noqa
    for codec in ("libx265", "hevc_nvenc", "hevc_qsv", "hevc_amf"):
        assert cfg(codec, True) == "yuv420p10le", codec
        assert cfg(codec, False) == "yuv420p", codec
    for codec in (None, "libx264", "h264_nvenc"):
        assert cfg(codec, True) == "yuv420p", codec
    assert _build_output_config(Fraction(48), "hevc_nvenc", 0).pix_fmt == "yuv420p"
    print("_test_rife_output_pix_fmt: PASS")


def _test_rife_carries_forward_source_comment_metadata():
    """ADR-269: real, confirmed bug -- a file that went through the main conversion
    (which embeds an iw3_* settings COMMENT tag) then RIFE lost that tag entirely,
    because _build_output_config() never set any metadata at all. Confirmed directly
    via ffprobe on a real file (..._TB_rife_alldub.mkv had NO iw3_* tag, while the
    earlier ..._TB.mkv stage of the same job did).

    comment=None (no tag on the source, e.g. a plain non-iw3 input) must keep prior
    behavior byte-identical -- empty metadata dict, same as before this fix."""
    cfg_with_comment = _build_output_config(Fraction(48), None, 0, comment="iw3_depth_model=Any_V3_Metric_Large")
    assert cfg_with_comment.metadata == {"comment": "iw3_depth_model=Any_V3_Metric_Large"}, cfg_with_comment.metadata

    cfg_no_comment = _build_output_config(Fraction(48), None, 0, comment=None)
    assert cfg_no_comment.metadata == {}, cfg_no_comment.metadata

    # Backward compat: the pre-ADR-269 call shape (no comment kwarg at all) is unchanged.
    cfg_legacy = _build_output_config(Fraction(48), None, 0)
    assert cfg_legacy.metadata == {}, cfg_legacy.metadata

    print("_test_rife_carries_forward_source_comment_metadata: PASS")


def _run_self_tests():
    _test_ensure_rife_model_downloads_model_package()
    _test_rife_cpu_device_not_overridden_by_cuda_availability()
    _test_rife_multiplier_timesteps()
    _test_rife_target_fps_scheduling()
    _test_rife_fps_validation()
    _test_rife_manifest_emission()
    _test_rife_scene_cut_protection()
    _test_rife_duplicate_frame_detection()
    _test_rife_video_codec_options()
    _test_rife_output_pix_fmt()
    _test_rife_carries_forward_source_comment_metadata()
    print("All iw3.rife_cli self-tests PASSED")


def main(argv=None):
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
        return
    args = create_parser().parse_args(argv)
    run(args.input, args.output, rife_model=args.rife_model, gpu=args.gpu,
        rife_multiplier=args.rife_multiplier, rife_target_fps=args.rife_target_fps,
        video_codec=args.video_codec, scene_cut_times_file=args.scene_cut_times_file)


if __name__ == "__main__":
    sys.exit(main())
