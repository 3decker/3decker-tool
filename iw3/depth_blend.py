import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime
from os import path
from concurrent.futures import ThreadPoolExecutor as PoolExecutor, as_completed
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from . import export_config
from nunif.utils.autocrop import AutoCrop
from nunif.utils.video.metadata import parse_time
import nunif.utils.video as VU


# Sibling of nunif/tmp (not inside it) so this survives a "delete nunif/tmp for a
# factory reset" -- this is a running history the user asked to keep as a future
# reference, not disposable GUI cache.
_STEP_LOG_PATH = path.join(path.dirname(__file__), "..", "logs", "iw3_step_timing.log")


def _log_step(basename, step_name, event, extra=""):
    """Append one line to a persistent, human-readable timing log: real wall-clock
    timestamps for each Dual-Pass Depth Blend stage, so the user has an actual record
    to compare future runs against instead of having to reconstruct it after the fact
    from GPU-monitor logs."""
    os.makedirs(path.dirname(_STEP_LOG_PATH), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {basename} | {step_name} | {event}"
    if extra:
        line += f" | {extra}"
    with open(_STEP_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


_BLEND_USAGE_LOG_PATH = path.join(path.dirname(__file__), "..", "logs", "iw3_depth_blend_usage_log.txt")


def _log_blend_usage(basename, secondary_model, frame_weights, active_threshold=0.05, strength=None):
    """Groups per-frame secondary-model blend weight (see blend_depth_frame's
    return_weight_stats) into contiguous frame ranges where it was meaningfully
    active, and appends a summary to a persistent log. Answers a real question with
    no other way to check after the fact: which parts of the video actually leaned on
    the secondary model, and how much -- the blend itself leaves no visible trace of
    where it engaged once it's done. "Active" means at least `active_threshold`
    (default 5%) of the frame's area had a real (>0.1) blend weight; frame numbers
    match the exported PNG filenames directly, so they translate to a video position
    via the source's fps.

    Only covers frames actually processed in the current run -- on a resume where
    some frames were already blended in an earlier run, those don't get re-measured,
    so a resumed run's summary reflects only what it itself did, not the whole clip.

    `strength` (the depth_blend_strength setting actually used for this run) is
    recorded in the header line so this log is self-contained -- otherwise reading
    "avg blend weight=0.19" means nothing without separately checking
    blend_fingerprint.json to know what the strength CAP was for that run."""
    if not frame_weights:
        return

    def frame_num(fname):
        return int(path.splitext(fname)[0])

    names = sorted(frame_weights.keys(), key=frame_num)
    ranges = []
    run_start = None
    run_weights = []
    run_actives = []
    prev_num = None
    for fname in names:
        num = frame_num(fname)
        mean_w, active_f = frame_weights[fname]
        if active_f > active_threshold:
            if run_start is None:
                run_start = num
            run_weights.append(mean_w)
            run_actives.append(active_f)
        elif run_start is not None:
            ranges.append((run_start, prev_num, run_weights, run_actives))
            run_start, run_weights, run_actives = None, [], []
        prev_num = num
    if run_start is not None:
        ranges.append((run_start, prev_num, run_weights, run_actives))

    os.makedirs(path.dirname(_BLEND_USAGE_LOG_PATH), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    strength_desc = f" | strength={strength:.2f}" if strength is not None else ""
    with open(_BLEND_USAGE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"=== {ts} | {basename} | secondary model: {secondary_model}{strength_desc} "
                f"(active = >{active_threshold:.0%} of frame area blended) ===\n")
        if not ranges:
            f.write(f"  never crossed the {active_threshold:.0%} activity threshold "
                    f"on any frame in this run\n")
        for start, end, weights, actives in ranges:
            avg_w = sum(weights) / len(weights)
            avg_a = sum(actives) / len(actives)
            f.write(
                f"  frames {start:08d}-{end:08d} ({end - start + 1} frames): "
                f"avg blend weight={avg_w:.2f}, avg area affected={avg_a:.0%}\n"
            )


def _format_duration(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _build_active_steps(args, input_path):
    """The real, ordered list of steps this specific job will actually run, given its
    settings and the source's actual content -- not a fixed guess. Computed once up
    front (detecting real HDR/DV presence costs one quick ffprobe call) so progress
    can honestly show "Step 3/6" instead of a count that silently excludes optional
    steps like Auto Crop or Dolby Vision, or wrongly includes an HDR step that will
    turn out to be a no-op because the source has no HDR/DV metadata at all."""
    from .utils import _detect_hdr_types, _find_ffprobe

    steps = []
    if getattr(args, "autocrop", None) is not None:
        steps.append("Auto Crop Analysis")

    hdr_active = False
    if getattr(args, "preserve_dowi", False):
        hdr_types = _detect_hdr_types(input_path, _find_ffprobe())
        hdr_active = bool(hdr_types.get("dv") or hdr_types.get("hdr10plus"))
        if hdr_active:
            steps.append("HDR/DV Extraction")

    steps.append("Audio Extraction")
    steps.append("Primary Depth Export")
    steps.append("Secondary Depth Export")
    if getattr(args, "depth_blend_align", False):
        steps.append("Depth Scale Alignment")
    steps.append("Blending")
    steps.append("Final Render")

    if hdr_active:
        steps.append("HDR/DV Injection")
        steps.append("RPU Remux")

    return steps, hdr_active


def _step_label(steps, name):
    idx = steps.index(name) + 1
    return f"Step {idx}/{len(steps)} - {name}"


def _job_fingerprint(args, input_path):
    """Everything that would make the previously-exported Primary/Secondary depth
    passes (pass_a/pass_b -- by far the most expensive steps) invalid or mismatched
    if changed between runs. Used to tell a genuine resume of the SAME job apart from
    a different job that happens to reuse the same Output path (in which case the
    leftover pass_a/pass_b/blend data belongs to someone else's settings entirely and
    must never be trusted).

    Deliberately does NOT include depth_blend_strength/region/region_percent -- those
    only affect the Blending step, not the two depth exports, so they're tracked
    separately in _blend_fingerprint() and changing only those reuses pass_a/pass_b
    instead of forcing a full re-export of both depth models from scratch."""
    return {
        "input": path.abspath(input_path),
        "start_time": getattr(args, "start_time", None),
        "end_time": getattr(args, "end_time", None),
        "depth_model": args.depth_model,
        "resolution": getattr(args, "resolution", None),
        "limit_resolution": bool(getattr(args, "limit_resolution", False)),
        "max_fps": getattr(args, "max_fps", None),
        "scene_detect": bool(getattr(args, "scene_detect", False)),
        "depth_blend_model": getattr(args, "depth_blend_model", None) or "VDA_L",
        "autocrop": getattr(args, "autocrop", None),
        "preserve_dowi": bool(getattr(args, "preserve_dowi", False)),
    }


def _blend_fingerprint(args):
    """Just the settings that affect the Blending step (and everything after it:
    Final Render, HDR Injection, RPU Remux). Tracked separately from
    _job_fingerprint so that changing depth_blend_strength/region/region_percent --
    to retune how the two already-computed depth passes get mixed -- reuses the
    existing pass_a/pass_b exports instead of redoing them."""
    return {
        "depth_blend_strength": float(getattr(args, "depth_blend_strength", 1.0) or 1.0),
        "depth_blend_region": getattr(args, "depth_blend_region", None) or "detail",
        "depth_blend_region_percent": float(getattr(args, "depth_blend_region_percent", 25.0) or 25.0),
        "depth_blend_feather_blur": int(getattr(args, "depth_blend_feather_blur", 0) or 0),
        "depth_blend_bilateral": bool(getattr(args, "depth_blend_bilateral", False)),
        "depth_blend_bilateral_d": int(getattr(args, "depth_blend_bilateral_d", 12) or 12),
        "depth_blend_bilateral_sigma_color": float(getattr(args, "depth_blend_bilateral_sigma_color", 75.0) or 75.0),
        "depth_blend_bilateral_sigma_space": float(getattr(args, "depth_blend_bilateral_sigma_space", 75.0) or 75.0),
        "depth_blend_clahe": bool(getattr(args, "depth_blend_clahe", False)),
        "depth_blend_clahe_clip": float(getattr(args, "depth_blend_clahe_clip", 2.0) or 2.0),
        "depth_blend_clahe_tile": int(getattr(args, "depth_blend_clahe_tile", 8) or 8),
        "depth_blend_align": bool(getattr(args, "depth_blend_align", False)),
        "depth_blend_align_decay": float(getattr(args, "depth_blend_align_decay", 0.9) or 0.9),
        "depth_blend_edge_suppression": float(getattr(args, "depth_blend_edge_suppression", 0.5) or 0.5),
        "depth_blend_edge_hard_cutoff": bool(getattr(args, "depth_blend_edge_hard_cutoff", False)),
    }


def _prepare_work_dir(args, input_path, work_dir, pass_a_dir, pass_b_dir):
    """Decide whether an existing work_dir is safe to resume from (same job, same
    settings) or must be wiped and started fresh (first run ever, or a different job
    reusing the same Output path). Returns True if resuming (depth passes reusable).

    A second, narrower check runs even when resuming: if only the blend-mixing
    settings (strength/region/region_percent) changed, the expensive pass_a/pass_b
    depth exports are kept, but any previously-blended frames and the blend.complete
    marker are cleared so the Blending step (and Final Render/HDR steps after it)
    redo with the new settings instead of silently reusing blended output made with
    the OLD strength/region under the same filenames."""
    fingerprint_path = path.join(work_dir, "job_fingerprint.json")
    blend_fingerprint_path = path.join(work_dir, "blend_fingerprint.json")
    current_fp = _job_fingerprint(args, input_path)
    current_blend_fp = _blend_fingerprint(args)
    resuming = False
    if path.exists(fingerprint_path):
        try:
            with open(fingerprint_path, encoding="utf-8") as f:
                saved_fp = json.load(f)
        except Exception:
            saved_fp = None
        if saved_fp == current_fp:
            resuming = True
        else:
            print("[depth-blend] settings or input changed since the last run at this output "
                  "path -- the leftover working files don't match, starting fresh instead of "
                  "resuming stale data.", file=sys.stderr)

    if not resuming and path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)

    os.makedirs(pass_a_dir, exist_ok=True)
    os.makedirs(pass_b_dir, exist_ok=True)
    with open(fingerprint_path, "w", encoding="utf-8") as f:
        json.dump(current_fp, f)

    if resuming:
        saved_blend_fp = None
        if path.exists(blend_fingerprint_path):
            try:
                with open(blend_fingerprint_path, encoding="utf-8") as f:
                    saved_blend_fp = json.load(f)
            except Exception:
                saved_blend_fp = None
        if saved_blend_fp != current_blend_fp:
            print("[depth-blend] blend strength/region settings changed since the last run -- "
                  "reusing the existing depth exports, but re-blending and re-rendering with "
                  "the new settings.", file=sys.stderr)
            blended_dir = path.join(work_dir, "blended_depth")
            if path.isdir(blended_dir):
                shutil.rmtree(blended_dir, ignore_errors=True)
            blend_complete = path.join(work_dir, "blend.complete")
            if path.exists(blend_complete):
                os.remove(blend_complete)

    with open(blend_fingerprint_path, "w", encoding="utf-8") as f:
        json.dump(current_blend_fp, f)

    return resuming


def _detail_mask(rgb_uint8):
    """0-1 map of how much fine visual detail is at each pixel (foliage, hair, texture --
    the kind of content a single-frame depth model tends to get wrong). Convention-
    independent: it only looks at the RGB image, never at which way a depth model's
    near/far scale points, so it can't be backwards the way a depth-based signal could.

    Real footage tested confirmed this needs a denoise pass BEFORE edge detection: on a
    dark/grainy shot, sensor/compression noise lit up the raw Laplacian almost uniformly
    across the whole frame, drowning out the real structure (a face, tree bark, branches)
    it was supposed to isolate -- which meant the blend was mixing the two depth models
    together close to randomly, everywhere, not just in genuinely detailed regions. A
    proper denoiser (fastNlMeansDenoising) fixed it but costs ~400ms/frame, roughly
    doubling this pass's total runtime on a real movie -- two passes of a bilateral
    filter gets a comparably clean result for ~15ms/frame, confirmed side-by-side on the
    same problem frame."""
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap = cv2.GaussianBlur(lap, (0, 0), sigmaX=4)
    lo, hi = float(lap.min()), float(lap.max())
    if hi - lo < 1e-6:
        return np.zeros_like(lap, dtype=np.float32)
    return ((lap - lo) / (hi - lo)).astype(np.float32)


def _depth_edge_mask(depth_a_u16, suppression_strength=0.5, hard_cutoff=False, hard_cutoff_threshold=0.5):
    """0-1 map of how strong/confident a depth DISCONTINUITY the primary model already
    has at each pixel -- a real silhouette boundary it has already resolved on its own.

    Real footage testing found "detail" mode's RGB-based mask backfires specifically at
    clean object silhouettes: RGB contrast is naturally high right at an edge too, so
    the detail mask lights up there just as strongly as it does on genuine missed
    detail (foliage, hair) -- blending in the secondary model exactly there mixes in
    ITS slightly different pixel-level edge position, since no two models draw a
    silhouette at the exact same pixel. The visible result is a doubled/ghosted edge,
    not an improvement. This mask lets the blend tell the two cases apart: suppress the
    blend where the primary model's OWN depth already has a strong, well-defined edge
    (nothing to gain, real risk of a doubled edge), keep it where the primary model's
    depth is smooth/flat but RGB shows real texture the depth missed (the actual
    foliage/hair case the detail mode was built for).

    `suppression_strength` (0.0-1.0, default 0.5) linearly interpolates between the
    two real, previously-measured data points on record for this setting: 0.0 = the
    original 2/99th-percentile values (looser -- user reported real, visible ~2px
    silhouette ghosting/soft-edge artifacts on real Dual-Pass Blend footage, e.g.
    "the border will look slightly out of focus"), 1.0 = the previously-tuned 4/97th
    values (measurably fixed that same ghosting on a synthetic silhouette-disagreement
    test case, but was reverted once at the user's earlier request for an
    undocumented reason -- see ADR-015 in docs/ai/AI_DECISIONS.md). 0.5 (the default,
    sigmaX=3/98th-percentile) is a deliberate middle-ground compromise chosen WITHOUT
    knowing what 4/97th's downside was. Exposed as a real GUI/CLI setting (not just a
    hardcoded constant) specifically so it can be tuned per-user/per-footage instead
    of guessed at again. Pure flat-depth foliage (no edge at all) is unaffected by
    this parameter at any value.

    `hard_cutoff` (default False): the band this produces is normally a smooth 0-1
    RAMP -- suppression fades in gradually as you approach a real edge, rather than
    switching on abruptly. Real footage testing found this smooth fade still lets a
    thin sliver of the secondary model's disagreeing edge position leak through in
    the partially-suppressed transition zone, on some objects. `hard_cutoff` turns
    the same ramp into a binary step instead: everywhere the ramp would read at or
    above `hard_cutoff_threshold` becomes fully suppressed (1.0), everywhere below
    becomes fully open (0.0) -- same underlying band WIDTH (still governed by
    `suppression_strength`), just a sharp edge to it instead of a gradual one. This
    should shrink (not necessarily eliminate) silhouette misalignment between two
    independently-trained depth models -- see ADR-026: no established technique
    exists to fully eliminate this, since it's a genuine shape/contour disagreement
    between two models with no shared ground truth to register against, not a
    value-blend problem this mask can fully solve."""
    suppression_strength = float(min(max(suppression_strength, 0.0), 1.0))
    sigma = 2.0 + 2.0 * suppression_strength
    percentile = 99.0 - 2.0 * suppression_strength

    d = depth_a_u16.astype(np.float32) / 65535.0
    grad_x = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Widen with a blur so the whole edge transition is suppressed, not just the exact
    # 1px gradient peak -- a doubled edge shows up as a visible band a few pixels wide,
    # not a single pixel line.
    grad_mag = cv2.GaussianBlur(grad_mag, (0, 0), sigmaX=sigma)
    hi = float(np.percentile(grad_mag, percentile))
    if hi < 1e-6:
        return np.zeros_like(grad_mag, dtype=np.float32)
    mask = np.clip(grad_mag / hi, 0.0, 1.0).astype(np.float32)
    if hard_cutoff:
        mask = (mask >= float(hard_cutoff_threshold)).astype(np.float32)
    return mask


def _region_mask(depth_a_u16, region, percent):
    """0-1 mask selecting a percentile slice of the depth map itself, instead of
    looking at the RGB image at all. Uses the SAME near/far convention already
    established everywhere else in iw3 (apply_foreground_pop, apply_foreground_
    divergence, etc.): a HIGHER depth value is NEARER the camera, lower is farther.
    That convention is confirmed directly from iw3's own existing code, not guessed --
    the earlier detail-based mask deliberately avoided depth-based logic specifically
    because this convention hadn't been nailed down yet at the time.

    'foreground' with percent=25 means: blend in the secondary model over the NEAREST
    25% of the scene by depth. 'background' is the mirror image -- the FARTHEST
    percent%. Ramped smoothly over a band around the cutoff rather than a hard
    threshold, to avoid the same abrupt-seam problem the plain detail mask had at
    ordinary silhouette edges."""
    percent = float(np.clip(percent, 0.0, 100.0))
    d = depth_a_u16.astype(np.float32)
    band = max(1.0, 0xffff * 0.05)  # ~5% of the full depth range, smooth transition
    if region == "foreground":
        cutoff = np.percentile(d, 100.0 - percent)
        mask = (d - cutoff) / band + 0.5
    elif region == "background":
        cutoff = np.percentile(d, percent)
        mask = (cutoff - d) / band + 0.5
    else:
        raise ValueError(f"unknown region '{region}' (expected 'foreground' or 'background')")
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _compute_depth_alignment(depth_a_u16, depth_b_u16, lo_percentile=5.0, hi_percentile=95.0):
    """Robust linear (scale, offset) mapping depth_b's numeric range onto depth_a's,
    for one frame. Both passes are already independently normalized to 0-65535 by
    their own model's own EMA min/max scaler (see base_depth_model.py) -- that only
    guarantees they land in the same NUMERIC range, not that a given value means the
    same real-world distance in both models (different depth curves, different
    metric/relative conventions). Blending two frames whose scales don't actually
    line up can show up as a value discontinuity right at a blend region's edge.

    Uses percentiles rather than min/max so a few outlier/noise pixels at the extreme
    ends don't dominate the fit -- deliberately whole-frame rather than restricted to
    wherever blending will actually apply, since that region depends on this same
    depth data (the "detail" mask/region mask), and a smaller sample would be noisier
    for comparable benefit. Returns (scale, offset) such that
    depth_b*scale+offset lines up with depth_a's distribution for this frame; a
    near-flat frame (no usable spread) returns the identity transform (1.0, 0.0)."""
    a = depth_a_u16.astype(np.float32)
    b = depth_b_u16.astype(np.float32)
    a_lo, a_hi = np.percentile(a, [lo_percentile, hi_percentile])
    b_lo, b_hi = np.percentile(b, [lo_percentile, hi_percentile])
    b_range = b_hi - b_lo
    if b_range < 1e-3:
        return 1.0, 0.0
    scale = float((a_hi - a_lo) / b_range)
    offset = float(a_lo - b_lo * scale)
    return scale, offset


def blend_depth_frame(depth_a_u16, depth_b_u16, rgb_uint8, strength=1.0,
                       region="detail", region_percent=25.0, feather_blur=0,
                       bilateral=False, bilateral_d=12, bilateral_sigma_color=75.0,
                       bilateral_sigma_space=75.0, clahe=False, clahe_clip=2.0,
                       clahe_tile=8, edge_suppression=0.5, edge_hard_cutoff=False,
                       return_weight_stats=False):
    """Combine two already-normalized (0-0xffff) depth frames for the same image.
    depth_a is trusted by default (the crisper/primary model); depth_b is blended in
    more strongly according to `region`:
      - "detail" (default): wherever the RGB frame shows dense fine detail -- the
        failure case described for foliage/close-up subjects.
      - "foreground" / "background": a percentile slice of the depth map itself (see
        _region_mask), e.g. region="foreground", region_percent=25 blends in the
        secondary model over the nearest 25% of the scene by depth, regardless of
        how much visual detail is there.

    Optional post-processing, all OFF by default (0/False) so existing jobs get
    byte-identical output unless a user explicitly opts in:
      - feather_blur: Gaussian-blurs the blend weight mask itself (kernel size in
        pixels, auto-rounded to odd) so the blended/unblended transition is soft
        rather than a hard edge. Applies to any region mode, independent of the
        "detail" mode's own separate depth-edge suppression above.
      - bilateral / bilateral_d / bilateral_sigma_color / bilateral_sigma_space: an
        edge-preserving smoothing pass over the FINAL blended depth, to clean up
        noise introduced by combining two models without softening real depth
        boundaries. bilateral_sigma_color is expressed in familiar 0-255-ish terms
        (matching common bilateral-filter presets) and scaled internally to this
        function's actual 0-65535 16-bit depth range -- entering 75 here behaves
        like entering 75 for an 8-bit image, not a near-no-op.
      - clahe / clahe_clip / clahe_tile: local contrast enhancement over the final
        blended depth. Deliberately OFF by default -- ADR-001 in
        docs/ai/AI_DECISIONS.md documents a prior benchmark on this same kind of
        depth data where CLAHE amplified whatever local variation it found, noise
        included. Provided as an explicit, opt-in experiment, not a recommended
        default. Run AFTER the bilateral pass (if both are enabled) so contrast
        enhancement doesn't amplify noise that step just removed.

    edge_suppression (0.0-1.0, default 0.5): only affects region="detail" mode --
    see _depth_edge_mask for what this actually tunes. Higher = more conservative
    (protects a wider band around real silhouettes from blending, less risk of a
    doubled/ghosted edge, but suppresses slightly more real detail blending right
    near an edge too). Lower = looser (more detail blending everywhere including
    close to edges, more risk of a soft/doubled edge on a real silhouette).

    edge_hard_cutoff (default False): only affects region="detail" mode. Changes
    edge_suppression's band from a smooth fade to a hard on/off step -- see
    _depth_edge_mask's own docstring for the full reasoning. A cheap thing to try
    when edge_suppression alone doesn't fully clear up silhouette misalignment,
    though it can only shrink, not eliminate, a genuine shape disagreement between
    two independently-trained depth models.

    Returns a uint16 array, same shape as depth_a_u16. With return_weight_stats=True,
    returns (blended, mean_weight, active_fraction) instead -- mean_weight is the
    average blend weight across the whole frame, active_fraction is the fraction of
    pixels where the secondary model visibly contributed (weight > 0.1). Together
    these are what let a caller answer "which parts of the video actually leaned on
    the secondary model" after the fact, instead of it being invisible once blending
    is done."""
    if depth_b_u16.shape != depth_a_u16.shape:
        depth_b_u16 = cv2.resize(depth_b_u16, (depth_a_u16.shape[1], depth_a_u16.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
    if rgb_uint8.shape[:2] != depth_a_u16.shape:
        rgb_uint8 = cv2.resize(rgb_uint8, (depth_a_u16.shape[1], depth_a_u16.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    if region == "detail":
        detail = _detail_mask(rgb_uint8)
        depth_edge = _depth_edge_mask(depth_a_u16, suppression_strength=edge_suppression,
                                       hard_cutoff=edge_hard_cutoff)
        # Suppress wherever the primary model already has a confident depth edge of
        # its own -- see _depth_edge_mask. Multiplying (not subtracting) means a
        # region with BOTH real RGB detail AND a strong depth edge (a leaf right at a
        # silhouette boundary, say) still gets scaled down smoothly rather than an
        # all-or-nothing cutoff.
        weight_b = np.clip(detail * (1.0 - depth_edge) * float(strength), 0.0, 1.0)
    else:
        weight_b = np.clip(_region_mask(depth_a_u16, region, region_percent) * float(strength), 0.0, 1.0)

    if feather_blur and int(feather_blur) > 0:
        k = int(feather_blur) | 1  # GaussianBlur requires an odd kernel size
        weight_b = np.clip(cv2.GaussianBlur(weight_b, (k, k), 0), 0.0, 1.0)

    a = depth_a_u16.astype(np.float32)
    b = depth_b_u16.astype(np.float32)
    blended = (1.0 - weight_b) * a + weight_b * b
    result = np.clip(blended, 0, 0xffff).astype(np.uint16)

    if bilateral:
        # bilateralFilter doesn't accept 16-bit input -- run it in float32, and scale
        # the user-facing 0-255-ish sigmaColor up to this data's actual 0-65535 range
        # (65535/255 = 257x) so the number behaves the way an 8-bit-filter user expects
        # instead of silently doing almost nothing against full-range 16-bit depth.
        result_f = result.astype(np.float32)
        result_f = cv2.bilateralFilter(
            result_f, d=int(bilateral_d),
            sigmaColor=float(bilateral_sigma_color) * (65535.0 / 255.0),
            sigmaSpace=float(bilateral_sigma_space))
        result = np.clip(result_f, 0, 0xffff).astype(np.uint16)

    if clahe:
        clahe_op = cv2.createCLAHE(clipLimit=float(clahe_clip),
                                    tileGridSize=(int(clahe_tile), int(clahe_tile)))
        result = clahe_op.apply(result)

    if return_weight_stats:
        mean_weight = float(weight_b.mean())
        active_fraction = float((weight_b > 0.1).mean())
        return result, mean_weight, active_fraction
    return result


def _load_u16(file_path):
    # cv2's PNG decoder measured ~1.8x faster than PIL's for these files -- confirmed
    # by direct benchmark against real depth-blend output, since file I/O (not the
    # actual blend math) is the real bottleneck of this whole step.
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


def _load_rgb(file_path):
    bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_depth_blend(args, depth_model, side_model):
    """Sequential depth blend pipeline: fully exports depth+rgb with the PRIMARY model,
    unloads it, fully exports depth with the SECONDARY model, unloads that too, blends
    the two depth sequences frame-by-frame (favoring the secondary model specifically in
    high visual-detail regions), then renders the final 3D output from the blend using
    iw3's existing "resume from exported depth" path. Auto Crop analysis and Dolby
    Vision/HDR10+ extraction+injection run as their own steps around this core sequence
    when enabled -- see _build_active_steps for the real, ordered step list for any
    given run (it varies: those two are optional and HDR is skipped entirely if the
    source has none).

    Deliberately never has both depth models resident in GPU memory at the same time --
    each pass fully finishes and releases its model before the next one loads. This costs
    real time (a full pass over the clip per active step) and real disk space (a full
    RGB+depth frame dump, twice), but it never risks running out of VRAM by doubling up,
    which a simultaneous approach would. Each step is independently resumable -- see
    _prepare_work_dir and the per-step ".complete" markers -- so a crash, close, or
    reboot only costs whatever step was interrupted."""
    from .utils import export_main, iw3_main, is_yaml, gc_collect
    from .depth_model_factory import create_depth_model
    from nunif.utils.ui import is_video

    input_path = path.abspath(args.input)
    if not is_video(input_path) or is_yaml(input_path):
        raise ValueError("--depth-blend requires a single video file as input")

    output_path = args.output
    work_dir = output_path + ".depth_blend_work"
    pass_a_dir = path.join(work_dir, "pass_a")
    pass_b_dir = path.join(work_dir, "pass_b")
    resuming = _prepare_work_dir(args, input_path, work_dir, pass_a_dir, pass_b_dir)
    steps, hdr_active = _build_active_steps(args, input_path)

    stop_event = args.state.get("stop_event") if getattr(args, "state", None) else None

    def cancelled():
        return stop_event is not None and stop_event.is_set()

    basename = path.splitext(path.basename(input_path))[0].strip()

    _log_step(basename, "Job", "RESUMED" if resuming else "START", f"steps={steps}")
    job_start = time.monotonic()
    try:
        result = _run_depth_blend_passes(
            args, depth_model, input_path, output_path, work_dir,
            pass_a_dir, pass_b_dir, basename, cancelled,
            export_main, iw3_main, create_depth_model, gc_collect,
            steps, hdr_active,
        )
        if result is None:
            _log_step(basename, "Job", "CANCELLED", _format_duration(time.monotonic() - job_start))
        else:
            _log_step(basename, "Job", "END", _format_duration(time.monotonic() - job_start))
        return result
    except Exception as e:
        _log_step(basename, "Job", "ERROR", str(e))
        raise
    finally:
        # However this exits -- cancel, an exception, or a clean finish -- never leave a
        # stale "Step N/M" label sitting in the shared args.state for whatever the next,
        # unrelated job in this same GUI process runs next.
        if getattr(args, "state", None) is not None:
            args.state["progress_step_label"] = None


def _extract_hdr_rpu_files(input_path, work_dir, args, hdr_rpu_path, hdr_h10p_path):
    """Runs the ffmpeg-trim + dovi_tool/hdr10plus_tool extraction that produces
    hdr_rpu_path / hdr_h10p_path. Shared by the normal "HDR/DV Extraction" step and
    the self-healing re-extraction right before injection (see _run_depth_blend_passes)
    -- a real job was observed where dv_rpu.bin existed right after extraction but was
    gone ~40 minutes later at injection time with no code path of ours responsible
    (root cause unconfirmed, suspected external interference e.g. antivirus); rather
    than keep chasing that, injection re-runs this and heals itself.

    Returns (dv_ok, h10p_ok, failures): dv_ok/h10p_ok are True when that HDR type
    either isn't present in the source at all, or was successfully (re-)extracted;
    failures is a list of human-readable messages for anything that didn't work.
    """
    from .utils import _get_ffmpeg_bin, _find_ffprobe, _detect_hdr_types, _find_dovi_tool, _find_hdr10plus_tool
    import subprocess
    ffmpeg_bin = _get_ffmpeg_bin()
    hdr_types = _detect_hdr_types(input_path, _find_ffprobe())
    tmp_hevc = path.join(work_dir, "_hdr_src.hevc")
    trim_args = []
    if getattr(args, "start_time", None):
        trim_args += ["-ss", str(parse_time(args.start_time))]
    if getattr(args, "end_time", None):
        trim_args += ["-to", str(parse_time(args.end_time))]
    failures = []
    dv_ok = not hdr_types.get("dv")
    h10p_ok = not hdr_types.get("hdr10plus")
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", *trim_args, "-i", str(input_path),
             "-c:v", "copy", "-an", "-f", "hevc", tmp_hevc],
            check=True, capture_output=True,
        )
        if hdr_types.get("dv"):
            dovi_bin = _find_dovi_tool()
            if not dovi_bin:
                failures.append("dovi_tool not found")
            else:
                try:
                    subprocess.run(
                        [dovi_bin, "extract-rpu", "-i", tmp_hevc, "-o", hdr_rpu_path],
                        check=True, capture_output=True,
                    )
                    dv_ok = path.exists(hdr_rpu_path)
                    if not dv_ok:
                        failures.append("dovi_tool exited 0 but produced no rpu file")
                except subprocess.CalledProcessError as e:
                    msg = e.stderr.decode(errors="replace").strip()
                    print(f"[depth-blend] DV RPU extraction failed: {msg}", file=sys.stderr)
                    failures.append(f"DV RPU extraction failed: {msg[:200]}")
        if hdr_types.get("hdr10plus"):
            h10p_bin = _find_hdr10plus_tool()
            if not h10p_bin:
                failures.append("hdr10plus_tool not found")
            else:
                try:
                    subprocess.run(
                        [h10p_bin, "extract", "-i", tmp_hevc, "-o", hdr_h10p_path],
                        check=True, capture_output=True,
                    )
                    h10p_ok = path.exists(hdr_h10p_path)
                    if not h10p_ok:
                        failures.append("hdr10plus_tool exited 0 but produced no json file")
                except subprocess.CalledProcessError as e:
                    msg = e.stderr.decode(errors="replace").strip()
                    print(f"[depth-blend] HDR10+ extraction failed: {msg}", file=sys.stderr)
                    failures.append(f"HDR10+ extraction failed: {msg[:200]}")
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace").strip()
        print(f"[depth-blend] source HEVC extraction failed: {msg}", file=sys.stderr)
        failures.append(f"source HEVC extraction failed: {msg[:200]}")
    finally:
        if path.exists(tmp_hevc):
            try:
                os.remove(tmp_hevc)
            except Exception:
                pass
    return dv_ok, h10p_ok, failures


def _run_depth_blend_passes(args, depth_model, input_path, output_path, work_dir,
                             pass_a_dir, pass_b_dir, basename, cancelled,
                             export_main, iw3_main, create_depth_model, gc_collect,
                             steps, hdr_active):
    # --- optional: auto crop analysis, shared identically across both export passes ---
    precomputed_crop_filter = ""
    if "Auto Crop Analysis" in steps:
        label = _step_label(steps, "Auto Crop Analysis")
        step_name = "Auto Crop Analysis"
        print(f"[depth-blend] {label}: analyzing for black bars / borders...", file=sys.stderr)
        args.state["progress_step_label"] = label
        _log_step(basename, step_name, "START")
        step_start = time.monotonic()
        crop = AutoCrop.from_video_file(
            input_path,
            mode=args.autocrop,
            uncrop_enabled=False,
            vf=args.vf,
            hwaccel=args.hwaccel,
            disable_software_fallback=args.disable_software_fallback,
            device=args.state["device"],
            batch_size=args.batch_size,
            stop_event=args.state["stop_event"],
            suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
            tqdm_title=f"{basename}: {label}",
        ).get_crop()
        if cancelled():
            _log_step(basename, step_name, "CANCELLED", _format_duration(time.monotonic() - step_start))
            return None
        if crop is not None:
            precomputed_crop_filter = f"crop=x={crop[0]}:y={crop[1]}:w={crop[2]}:h={crop[3]}"
            crop_info = f"crop applied: {crop}"
        else:
            crop_info = "no crop needed"
        duration = _format_duration(time.monotonic() - step_start)
        _log_step(basename, step_name, "END", f"{duration} | {crop_info}")

    # --- optional: HDR/DV extraction from the ORIGINAL source, trimmed to match the
    # exact range being converted -- done once here (not per-pass) since it's a
    # property of the source clip, not of either depth model. ---
    hdr_rpu_path = path.join(work_dir, "dv_rpu.bin")
    hdr_h10p_path = path.join(work_dir, "hdr10plus.json")
    if "HDR/DV Extraction" in steps:
        label = _step_label(steps, "HDR/DV Extraction")
        step_name = "HDR/DV Extraction"
        hdr_extract_complete = path.join(work_dir, "hdr_extract.complete")
        if path.exists(hdr_extract_complete):
            print(f"[depth-blend] {label}: already extracted from a previous run, skipping.",
                  file=sys.stderr)
            _log_step(basename, step_name, "SKIPPED (already complete)")
        else:
            print(f"[depth-blend] {label}: extracting Dolby Vision / HDR10+ metadata from the "
                  f"source...", file=sys.stderr)
            args.state["progress_step_label"] = label
            _log_step(basename, step_name, "START")
            step_start = time.monotonic()
            # Collected instead of only printed to stderr -- a GUI-launched (pythonw)
            # process's stderr is not visibly surfaced to the user, so a failure here
            # was previously invisible: the step still logged "END" and marked itself
            # permanently complete even when nothing was actually extracted.
            dv_ok, h10p_ok, failures = _extract_hdr_rpu_files(
                input_path, work_dir, args, hdr_rpu_path, hdr_h10p_path)
            duration = _format_duration(time.monotonic() - step_start)
            if dv_ok and h10p_ok:
                # Only mark permanently complete when everything the source actually
                # has was genuinely produced -- otherwise a resume must retry this
                # step instead of skipping it forever having produced nothing.
                with open(hdr_extract_complete, "w", encoding="utf-8") as f:
                    f.write("1")
                _log_step(basename, step_name, "END", duration)
            else:
                _log_step(basename, step_name, "FAILED", f"{duration} | " + "; ".join(failures))

    # --- audio extraction: pulled out as its own visible step, even though it's
    # small/fast, since it's a real, distinct operation and not just bookkeeping.
    # Writes to the exact path Primary Depth Export's own export_video() call expects
    # its audio at, so that call sees it already exists and skips re-extracting (see
    # the existence check added to export_video for this). ---
    label = _step_label(steps, "Audio Extraction")
    step_name = "Audio Extraction"
    audio_export_dir = path.join(pass_a_dir, basename)
    audio_target_path = path.join(audio_export_dir, export_config.AUDIO_FILE)
    if path.exists(audio_target_path):
        print(f"[depth-blend] {label}: already extracted from a previous run, skipping.",
              file=sys.stderr)
        _log_step(basename, step_name, "SKIPPED (already complete)")
    else:
        print(f"[depth-blend] {label}: extracting audio from the source...", file=sys.stderr)
        args.state["progress_step_label"] = label
        _log_step(basename, step_name, "START")
        step_start = time.monotonic()
        os.makedirs(audio_export_dir, exist_ok=True)
        has_audio = VU.export_audio(
            input_path, audio_target_path,
            start_time=args.start_time, end_time=args.end_time,
            title=f"{basename} Audio",
            stop_event=args.state["stop_event"], suspend_event=args.state["suspend_event"],
            tqdm_fn=args.state["tqdm_fn"],
        )
        duration = _format_duration(time.monotonic() - step_start)
        _log_step(basename, step_name, "END",
                   f"{duration} | {'audio extracted' if has_audio else 'source has no audio track'}")

    # --- pass: full export (rgb + depth) with the primary model ---
    pass_a_complete = path.join(pass_a_dir, ".complete")
    label = _step_label(steps, "Primary Depth Export")
    step_name = f"Primary Depth ({args.depth_model})"
    if path.exists(pass_a_complete):
        print(f"[depth-blend] {label}: already complete from a previous run, "
              "skipping (and not even loading the model).", file=sys.stderr)
        _log_step(basename, step_name, "SKIPPED (already complete)")
    else:
        print(f"[depth-blend] {label}: exporting primary depth ({args.depth_model})...", file=sys.stderr)
        args.state["progress_step_label"] = label
        _log_step(basename, step_name, "START")
        step_start = time.monotonic()
        args_a = copy.copy(args)
        args_a.export = True
        args_a.export_depth_only = False
        args_a.export_disparity = False
        args_a.output = pass_a_dir
        args_a.yes = True
        if precomputed_crop_filter:
            args_a.vf = f"{args.vf},{precomputed_crop_filter}" if args.vf else precomputed_crop_filter
            # Already decided above (once, shared with pass 2) -- prevent export_video
            # from redundantly re-running its own AutoCrop analysis on top of this.
            args_a.autocrop = None
        # Picks up mid-export from wherever a previous crash/close left off, using iw3's
        # existing per-frame export-resume support, instead of always redoing this whole
        # (often the single most expensive) pass from frame zero.
        args_a.resume = True
        args_a.compile = False

        if not depth_model.loaded():
            depth_model.load(gpu=args.gpu, resolution=args.resolution, limit_resolution=args.limit_resolution)
        args.state["depth_model"] = depth_model
        export_main(args_a)
        if cancelled():
            _log_step(basename, step_name, "CANCELLED", _format_duration(time.monotonic() - step_start))
            return None
        with open(pass_a_complete, "w", encoding="utf-8") as f:
            f.write("1")
        _log_step(basename, step_name, "END", _format_duration(time.monotonic() - step_start))

        # fully release the primary model before the secondary one loads -- this is the
        # whole point of doing this sequentially instead of holding both at once
        depth_model.model = None
        gc_collect()

    # --- pass: depth-only export with the secondary model (reuses pass 1's rgb) ---
    secondary_name = getattr(args, "depth_blend_model", None) or "VDA_L"
    pass_b_complete = path.join(pass_b_dir, ".complete")
    label = _step_label(steps, "Secondary Depth Export")
    step_name = f"Secondary Depth ({secondary_name})"
    if path.exists(pass_b_complete):
        print(f"[depth-blend] {label}: already complete from a previous run, "
              "skipping (and not even loading the model).", file=sys.stderr)
        _log_step(basename, step_name, "SKIPPED (already complete)")
    else:
        print(f"[depth-blend] {label}: exporting secondary depth ({secondary_name})...", file=sys.stderr)
        args.state["progress_step_label"] = label
        _log_step(basename, step_name, "START")
        step_start = time.monotonic()
        secondary_model = create_depth_model(secondary_name)
        secondary_model.load(gpu=args.gpu, resolution=args.resolution, limit_resolution=args.limit_resolution)

        args_b = copy.copy(args)
        args_b.export = True
        args_b.export_depth_only = True
        args_b.export_disparity = False
        args_b.output = pass_b_dir
        args_b.yes = True
        if precomputed_crop_filter:
            args_b.vf = f"{args.vf},{precomputed_crop_filter}" if args.vf else precomputed_crop_filter
            args_b.autocrop = None
        args_b.resume = True  # see pass 1's comment -- same per-frame export-resume support
        args_b.compile = False
        args_b.depth_model = secondary_name
        args.state["depth_model"] = secondary_model
        export_main(args_b)
        if cancelled():
            _log_step(basename, step_name, "CANCELLED", _format_duration(time.monotonic() - step_start))
            return None
        with open(pass_b_complete, "w", encoding="utf-8") as f:
            f.write("1")
        _log_step(basename, step_name, "END", _format_duration(time.monotonic() - step_start))

        secondary_model.model = None
        gc_collect()

    # --- pass: blend the two depth sequences ---
    pass_a_base = path.join(pass_a_dir, basename)
    pass_b_base = path.join(pass_b_dir, basename)
    rgb_dir = path.join(pass_a_base, "rgb")
    depth_a_dir = path.join(pass_a_base, "depth")
    depth_b_dir = path.join(pass_b_base, "depth")
    config_path = path.join(pass_a_base, "iw3_export.yml")
    # Blended output lives in its OWN folder rather than overwriting depth_a_dir in
    # place. That makes resume trivial and safe: depth_a_dir/depth_b_dir (the raw
    # per-model exports from passes 1-2) are never touched or destroyed by this step,
    # so re-running any subset of frames here (including all of them, redundantly) is
    # always safe -- there's no way to accidentally blend an already-blended frame a
    # second time, which overwriting in place could risk on a crash at exactly the
    # wrong moment (all frames done, .complete marker not yet written).
    blended_dir = path.join(work_dir, "blended_depth")
    os.makedirs(blended_dir, exist_ok=True)
    blend_complete = path.join(work_dir, "blend.complete")

    depth_a_files = sorted(f for f in os.listdir(depth_a_dir) if f.lower().endswith(".png"))
    depth_b_files = sorted(f for f in os.listdir(depth_b_dir) if f.lower().endswith(".png"))
    if len(depth_a_files) != len(depth_b_files):
        raise RuntimeError(
            f"[depth-blend] frame count mismatch between the two passes "
            f"({len(depth_a_files)} vs {len(depth_b_files)}) -- cannot blend. "
            f"This shouldn't happen since both passes decode the same source the same way."
        )

    label = _step_label(steps, "Blending")
    step_name = "Blending Depth"
    if path.exists(blend_complete):
        print(f"[depth-blend] {label}: already complete from a previous run, "
              "skipping.", file=sys.stderr)
        _log_step(basename, step_name, "SKIPPED (already complete)")
    else:
        args.state["progress_step_label"] = label
        strength = float(getattr(args, "depth_blend_strength", 1.0) or 1.0)
        region = getattr(args, "depth_blend_region", None) or "detail"
        region_percent = float(getattr(args, "depth_blend_region_percent", 25.0) or 25.0)
        feather_blur = int(getattr(args, "depth_blend_feather_blur", 0) or 0)
        bilateral = bool(getattr(args, "depth_blend_bilateral", False))
        bilateral_d = int(getattr(args, "depth_blend_bilateral_d", 12) or 12)
        bilateral_sigma_color = float(getattr(args, "depth_blend_bilateral_sigma_color", 75.0) or 75.0)
        bilateral_sigma_space = float(getattr(args, "depth_blend_bilateral_sigma_space", 75.0) or 75.0)
        clahe = bool(getattr(args, "depth_blend_clahe", False))
        clahe_clip = float(getattr(args, "depth_blend_clahe_clip", 2.0) or 2.0)
        clahe_tile = int(getattr(args, "depth_blend_clahe_tile", 8) or 8)
        align = bool(getattr(args, "depth_blend_align", False))
        edge_suppression = float(getattr(args, "depth_blend_edge_suppression", 0.5) or 0.5)
        edge_hard_cutoff = bool(getattr(args, "depth_blend_edge_hard_cutoff", False))

        align_params = None
        if align:
            # Its own visible step (see _build_active_steps), not folded silently into
            # Blending's own label/progress bar -- this pre-pass can take real, visible
            # time on a long clip (one extra sequential load of every frame in both
            # depth folders) and the user asked to see it accounted for separately
            # rather than have it look like part of "Blending" was just slow.
            #
            # Deliberately NOT part of the parallel blend loop below: each frame's raw
            # alignment (see _compute_depth_alignment) is EMA-smoothed across the frame
            # sequence to keep the alignment itself from flickering frame to frame
            # (same anti-flicker convention used for depth min/max normalization
            # elsewhere in this project -- see docs/ai/domains/EMA_FLICKER.md). EMA
            # smoothing needs a fixed frame order, which the blend loop intentionally
            # does NOT have (parallelized across threads for throughput) -- so this has
            # to run first and separately. Runs over ALL frames (not just `remaining`)
            # so a resumed blend keeps the same EMA sequence a from-scratch run would
            # have had, not a discontinuity at the resume point.
            align_label = _step_label(steps, "Depth Scale Alignment")
            align_step_name = "Depth Scale Alignment"
            print(f"[depth-blend] {align_label}: computing depth scale alignment "
                  f"({len(depth_a_files)} frames)...", file=sys.stderr)
            args.state["progress_step_label"] = align_label
            _log_step(basename, align_step_name, "START")
            align_step_start = time.monotonic()
            align_decay = float(getattr(args, "depth_blend_align_decay", 0.9) or 0.9)
            ema_scale, ema_offset = None, None
            align_params = {}
            align_tqdm_fn = args.state["tqdm_fn"] or tqdm
            align_pbar = align_tqdm_fn(total=len(depth_a_files), desc=align_label, ncols=80)
            try:
                for fname in depth_a_files:
                    if cancelled():
                        break
                    depth_a_align = _load_u16(path.join(depth_a_dir, fname))
                    depth_b_align = _load_u16(path.join(depth_b_dir, fname))
                    scale, offset = _compute_depth_alignment(depth_a_align, depth_b_align)
                    if ema_scale is None:
                        ema_scale, ema_offset = scale, offset
                    else:
                        ema_scale = align_decay * ema_scale + (1.0 - align_decay) * scale
                        ema_offset = align_decay * ema_offset + (1.0 - align_decay) * offset
                    align_params[fname] = (ema_scale, ema_offset)
                    align_pbar.update(1)
            finally:
                align_pbar.close()
            if cancelled():
                _log_step(basename, align_step_name, "CANCELLED",
                           _format_duration(time.monotonic() - align_step_start))
                return None
            _log_step(basename, align_step_name, "END",
                       _format_duration(time.monotonic() - align_step_start))
            args.state["progress_step_label"] = label

        step_start = time.monotonic()
        remaining = [f for f in depth_a_files if not path.exists(path.join(blended_dir, f))]

        # Plain tqdm(iterable) never reaches the GUI -- the GUI's own progress bar is
        # driven by args.state["tqdm_fn"] (TQDMGUI), a different, non-iterable class
        # that only supports total=/update()/close(), not "for x in tqdm_fn(items)".
        # Without this, the blend step ran with a real, correct console-only progress
        # bar that the GUI window had no way to see at all -- from the GUI's
        # perspective this whole step looked like a frozen bar showing whatever pass 2
        # last reported, even while blending was genuinely progressing underneath it.
        _log_step(basename, step_name, "START" if not remaining or len(remaining) == len(depth_a_files)
                   else "RESUMED", f"{len(depth_a_files) - len(remaining)}/{len(depth_a_files)} frames already done")
        tqdm_fn = args.state["tqdm_fn"] or tqdm
        pbar = tqdm_fn(total=len(depth_a_files), desc=label, ncols=80)
        pbar.update(len(depth_a_files) - len(remaining))

        # Populated by blend_one below -- each frame writes its OWN key, so this is
        # safe across threads without a lock. Used afterward to report which stretches
        # of the video actually leaned on the secondary model and by how much, since
        # the blend itself leaves no visible trace of where it engaged once it's done.
        frame_weights = {}

        def blend_one(fname):
            if cancelled():
                return
            depth_a = _load_u16(path.join(depth_a_dir, fname))
            depth_b = _load_u16(path.join(depth_b_dir, fname))
            rgb = _load_rgb(path.join(rgb_dir, fname))
            if align_params is not None:
                a_scale, a_offset = align_params[fname]
                depth_b = np.clip(depth_b.astype(np.float32) * a_scale + a_offset, 0, 0xffff).astype(np.uint16)
            blended, mean_weight, active_fraction = blend_depth_frame(
                depth_a, depth_b, rgb, strength=strength,
                region=region, region_percent=region_percent, feather_blur=feather_blur,
                bilateral=bilateral, bilateral_d=bilateral_d,
                bilateral_sigma_color=bilateral_sigma_color, bilateral_sigma_space=bilateral_sigma_space,
                clahe=clahe, clahe_clip=clahe_clip, clahe_tile=clahe_tile,
                edge_suppression=edge_suppression, edge_hard_cutoff=edge_hard_cutoff,
                return_weight_stats=True)
            frame_weights[fname] = (mean_weight, active_fraction)
            # Write to a temp name and atomically rename into place: if the process
            # dies mid-write, a resume must never mistake a half-written file for a
            # finished one (os.replace is atomic on the same volume on Windows too).
            # compress_level=0: these are temporary working files read back exactly
            # once by the final render step, not the deliverable -- confirmed by
            # direct benchmark that this cuts save time by ~75% (33ms -> 8ms/frame)
            # at the cost of larger intermediate files, a trade already accepted for
            # this feature.
            tmp_path = path.join(blended_dir, fname + ".tmp")
            Image.fromarray(blended).save(tmp_path, format="PNG", compress_level=0)
            os.replace(tmp_path, path.join(blended_dir, fname))

        # Each frame's blend is fully independent of every other frame -- unlike
        # Object Stability's optical flow, there's no frame-to-frame state here at all
        # -- so this parallelizes across CPU cores cleanly. Previously ran as a plain
        # single-threaded loop, which only ever kept 1-2 cores busy (whatever OpenCV's
        # own internal multithreading grabbed for a couple of its filter calls) no
        # matter how many cores the machine had. Reuses the same Worker Threads
        # setting (and the same max(., 8) floor) the rest of iw3 already uses for its
        # own per-frame CPU work, instead of a separate/new setting.
        # cv2.setNumThreads(1) stops OpenCV's own internal threading from fighting
        # with these Python threads over the same cores -- confirmed via testing that
        # letting both layers thread at once causes oversubscription/contention
        # instead of clean additive scaling.
        max_workers = max(args.max_workers, 8)
        prev_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        was_cancelled = False
        try:
            with PoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(blend_one, fname) for fname in remaining]
                for f in as_completed(futures):
                    f.result()
                    pbar.update(1)
                    if cancelled():
                        was_cancelled = True
                        break
        finally:
            pbar.close()
            cv2.setNumThreads(prev_cv2_threads)
        if was_cancelled or cancelled():
            _log_step(basename, step_name, "CANCELLED", _format_duration(time.monotonic() - step_start))
            return None

        _log_blend_usage(basename, secondary_name, frame_weights, strength=strength)

        # Point the exported YAML at the blended depth instead of pass_a's raw depth
        # (resolve_path supports absolute paths, so this doesn't need to live inside
        # pass_a_base at all) -- redone every time blending actually runs, harmless
        # if repeated.
        exported = export_config.ExportConfig.load(config_path)
        exported.depth_dir = path.abspath(blended_dir)
        exported.save(config_path)

        with open(blend_complete, "w", encoding="utf-8") as f:
            f.write("1")
        _log_step(basename, step_name, "END", _format_duration(time.monotonic() - step_start))

    label = _step_label(steps, "Final Render")
    step_name = "Final Render"
    print(f"[depth-blend] {label}: rendering final 3D output from the blended depth...", file=sys.stderr)
    args.state["progress_step_label"] = label
    _log_step(basename, step_name, "START")
    step_start = time.monotonic()
    # When --output is a directory (the common case), process_config_video (what
    # iw3_main dispatches to below) resolves the REAL final filename internally via
    # make_output_filename() and never hands it back to this caller -- everything
    # after this point in this function (HDR/DV Injection, the waifu2x auto-upscale
    # hook) was still using the ORIGINAL directory path, not the actual file that
    # got written. That surfaced as ffmpeg being asked to open "-i E:\3d Movies"
    # (the bare directory) once the codec-trust fix stopped short-circuiting before
    # ever reaching that line. Resolve and use the same real filename here,
    # replicating process_config_video's own resolution logic exactly so both stay
    # in sync, and pass that resolved file straight through as args_final.output too
    # (harmless -- process_config_video's own is_output_dir check simply takes its
    # "already a file" branch when given one).
    from nunif.utils.ui import is_output_dir
    from .utils import make_output_filename
    if is_output_dir(output_path):
        exported_for_name = export_config.ExportConfig.load(config_path)
        resolved_basename = exported_for_name.basename or path.basename(path.dirname(config_path))
        output_path = path.join(output_path, make_output_filename(resolved_basename, args, video=True))

    args_final = copy.copy(args)
    args_final.input = config_path
    args_final.output = output_path
    args_final.export = False
    # NOTE: deliberately left True (not set to False here) -- iw3_main's dispatch now
    # tells this apart from a fresh "start a new Depth Blend job" request by checking
    # is_yaml(args.input) instead, specifically so the filename and embedded-metadata
    # tagging (which both key off this same flag) still fire for the final render.
    args_final.depth_blend = True
    args_final.yes = True
    # This render path has no internal checkpointing of its own -- unlike passes 1-2,
    # a crash mid-render always means redoing this whole pass, there's no partial
    # resume within it. This flag only backstops the OUTER case: if this exact final
    # file already exists in full from a previous successful run of this same job
    # (e.g. it crashed/closed AFTER finishing the render but before the job as a
    # whole wrapped up), skip re-rendering entirely instead of redoing 30-40+ minutes
    # of work for nothing. Forced on here regardless of the user's own general Resume
    # setting, since this is an internal safety behavior, not that per-job option.
    args_final.resume = True
    iw3_main(args_final)
    _log_step(basename, step_name, "END", _format_duration(time.monotonic() - step_start))

    # --- optional: inject the DV/HDR10+ metadata extracted earlier into the finished
    # output -- process_config_video (what Final Render just used) has no HDR handling
    # of its own, unlike a normal conversion, so this has to happen as its own step
    # here instead. Split into two: injecting into the raw HEVC stream, then remuxing
    # that stream back into the container -- genuinely separate operations, not one
    # atomic action, so each gets its own progress step. ---
    if hdr_active:
        from .utils import (_get_ffmpeg_bin, _find_dovi_tool, _find_hdr10plus_tool,
                             _inject_hdr_rpu, _remux_injected_hevc)
        label = _step_label(steps, "HDR/DV Injection")
        step_name = "HDR/DV Injection"
        print(f"[depth-blend] {label}: injecting Dolby Vision / HDR10+ metadata into the "
              f"finished output's video stream...", file=sys.stderr)
        args.state["progress_step_label"] = label
        _log_step(basename, step_name, "START")
        step_start = time.monotonic()
        rpu = hdr_rpu_path if path.exists(hdr_rpu_path) else None
        h10p = hdr_h10p_path if path.exists(hdr_h10p_path) else None
        hdr_extract_complete = path.join(work_dir, "hdr_extract.complete")
        if not (rpu or h10p) and path.exists(hdr_extract_complete):
            # Self-healing: the extraction step reported success earlier (the
            # .complete marker is only written when the file(s) it promised were
            # confirmed to exist), yet neither file is here now. Observed on a real,
            # single uninterrupted job -- cause unconfirmed (ruled out: our own
            # os.remove calls, the GUI's Quick Preview feature, a fast-cycling
            # external cleanup). Rather than accept "nothing to inject" for a file
            # that really did have DV/HDR10+, re-run the extraction fresh right here.
            print(f"[depth-blend] {label}: RPU/HDR10+ file(s) produced by the earlier "
                  f"extraction step are missing -- re-extracting now before giving up...",
                  file=sys.stderr)
            _extract_hdr_rpu_files(input_path, work_dir, args, hdr_rpu_path, hdr_h10p_path)
            rpu = hdr_rpu_path if path.exists(hdr_rpu_path) else None
            h10p = hdr_h10p_path if path.exists(hdr_h10p_path) else None
        injected_hevc, fps_str = None, None
        if rpu or h10p:
            try:
                injected_hevc, fps_str, inject_fail_reason = _inject_hdr_rpu(
                    output_path, rpu, h10p,
                    _get_ffmpeg_bin(), _find_dovi_tool(), _find_hdr10plus_tool(),
                    work_dir,
                    configured_video_codec=getattr(args, "video_codec", None),
                )
                if injected_hevc is None:
                    # RPU/HDR10+ file(s) genuinely existed here, but _inject_hdr_rpu
                    # itself declined -- distinguish this clearly from "extraction
                    # produced nothing" above, since the two look identical to the
                    # next step (Remux) but have very different causes. inject_fail_
                    # reason is the ACTUAL cause from inside _inject_hdr_rpu (e.g. its
                    # HEVC-codec verification failing even after retries) -- put
                    # directly in this visible log instead of only stderr, which is
                    # invisible in the GUI's windowless process and previously left
                    # this as an unexplained bare "0m00s".
                    _log_step(basename, step_name, "END",
                               f"{_format_duration(time.monotonic() - step_start)} | "
                               f"RPU/HDR10+ file(s) were present but injection declined: "
                               f"{inject_fail_reason or '(no reason reported)'}")
                else:
                    _log_step(basename, step_name, "END", _format_duration(time.monotonic() - step_start))
            except Exception as e:
                print(f"[depth-blend] HDR metadata injection failed: {e}", file=sys.stderr)
                _log_step(basename, step_name, "ERROR", str(e))
        else:
            duration = _format_duration(time.monotonic() - step_start)
            _log_step(basename, step_name, "END",
                       f"{duration} | nothing to inject -- extraction produced no usable RPU/HDR10+ file")

        label = _step_label(steps, "RPU Remux")
        step_name = "RPU Remux"
        remux_succeeded = False
        if injected_hevc is not None:
            print(f"[depth-blend] {label}: remuxing the injected video stream back "
                  f"into the container...", file=sys.stderr)
            args.state["progress_step_label"] = label
            _log_step(basename, step_name, "START")
            step_start = time.monotonic()
            try:
                _remux_injected_hevc(output_path, injected_hevc, fps_str, _get_ffmpeg_bin(), work_dir)
                _log_step(basename, step_name, "END", _format_duration(time.monotonic() - step_start))
                remux_succeeded = True
            except Exception as e:
                print(f"[depth-blend] RPU remux failed: {e}", file=sys.stderr)
                _log_step(basename, step_name, "ERROR", str(e))
        else:
            _log_step(basename, step_name, "SKIPPED (nothing was injected)")

        # Only delete the source RPU/HDR10+ files once they've actually been used --
        # i.e. injection AND remux both genuinely succeeded. Previously deleted
        # unconditionally here regardless of outcome, which meant a failed/declined
        # injection (see the "codec verification" saga in docs/ai/AI_KNOWN_ISSUES.md
        # KI-009) threw away the very file a manual recovery would need, forcing a
        # full re-extraction from the source movie instead of just reusing what was
        # already sitting right here. Left in place on any failure so a resumed job,
        # or a manual injection attempt, can reuse them directly.
        if remux_succeeded:
            for f in filter(None, [rpu, h10p]):
                try:
                    os.remove(f)
                except Exception:
                    pass
        else:
            surviving = [f for f in (rpu, h10p) if f and path.exists(f)]
            if surviving:
                print(f"[depth-blend] keeping {', '.join(surviving)} in the work folder "
                      f"since injection/remux did not complete -- reused automatically on "
                      f"a resumed run, or usable for a manual recovery.", file=sys.stderr)

    if not cancelled():
        from .utils import _run_waifu2x_upscale
        _run_waifu2x_upscale(output_path, args)

    print(f"[depth-blend] done. Working files (full rgb/depth dumps from both passes) are still in:\n"
          f"  {work_dir}\n"
          f"Safe to delete once you've confirmed the output looks right.", file=sys.stderr)
    return args
