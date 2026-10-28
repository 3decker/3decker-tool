"""ADR-390: `sod_track` Convergence Plane mode.

Real request: sod_v1's existing `subject_lock` parameter only decides, via per-frame
IOU-matching of FRESH SOD redetections, whether to keep "holding" the same region --
it re-runs the SOD saliency model on every single frame regardless, and only compares
masks after the fact. It never actually FOLLOWS the subject's appearance across frames.
`sod_track` is genuinely different: it runs the same SOD model ONCE per shot to get an
initial saliency mask -> bounding box, then hands that box to a real OpenCV appearance
tracker (cv2 TrackerCSRT) that follows it frame-to-frame by actually looking at the
tracked region's own appearance (color/texture), not by re-detecting saliency at all.

Reinitialization (re-running SOD, discarding the tracker's own box): on every real scene
cut (reset_pts), AND periodically every `_REINIT_INTERVAL` frames (90, chosen as roughly
3-4 seconds at typical 24-30fps -- long enough that SOD only runs a small fraction of
frames instead of every one, short enough to bound how far CSRT's own appearance model
can silently drift within one shot before being corrected back to the real saliency
subject), AND immediately whenever CSRT itself reports tracking failure (`ok=False`,
e.g. the subject left the frame or became unrecognizable) rather than waiting for the
next scheduled interval.

Depth position is computed with the same quantile-position histogram math as
depth_percentile/sod_v1 (ConvergenceEstimator._quantile_position()), applied to the
depth values INSIDE the currently tracked box, then smoothed with the same shared
SceneHoldTracker the other auto modes use.
"""
import sys
import torch
import torch.nn.functional as F

from .convergence_tracker import SceneHoldTracker
from .convergence_estimator import ConvergenceEstimator, _load_sod_model
from nunif.device import create_device, autocast

_CV2_IMPORT_ERROR = None
try:
    import cv2
    _CV2_AVAILABLE = True
except Exception as e:
    _CV2_AVAILABLE = False
    cv2 = None
    _CV2_IMPORT_ERROR = e


def _create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise ImportError(
        "This OpenCV (cv2) build has no TrackerCSRT_create -- it needs the contrib "
        "tracking module (opencv-contrib-python).\n"
        "Run in nunif-prompt.bat: pip install opencv-contrib-python==4.10.0.84"
    )


class SodTrackEstimator():
    """
    Places the 3D screen plane at a quantile position within the depth of a subject
    region that is TRACKED continuously across frames (real cv2 appearance tracking),
    initialized once per shot from the SOD_v1 saliency model. See module docstring for
    how this differs from sod_v1's `subject_lock`.
    """

    _REINIT_INTERVAL = 90  # frames -- see module docstring
    # ADR-395: CSRT's own per-frame appearance tracking runs on a frame downscaled to
    # at most this many pixels on its longest side -- real reported perf fix, see
    # _track_scale_factor()'s own docstring for why.
    _TRACK_MAX_DIM = 480
    # ADR-399: minimum box size (in the DOWNSCALED frame's own pixels) handed to
    # cv2's CSRT tracker -- see _clamp_track_box()'s own docstring for why this
    # exists.
    _TRACK_MIN_BOX_DIM = 20

    def __init__(self, convergence, device_id, enable_ema=False, decay=0.9, compile=False,
                 scene_hold=False, cut_smooth=0):
        if not _CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for sod_track convergence mode.\n"
                # ADR-396: opencv-contrib-python, not plain opencv-python -- sod_track
                # needs cv2.TrackerCSRT_create, which only exists in the contrib build.
                f"Run in nunif-prompt.bat: pip install opencv-contrib-python==4.10.0.84"
                + (f"\n({_CV2_IMPORT_ERROR.__class__.__name__}: {_CV2_IMPORT_ERROR})"
                   if _CV2_IMPORT_ERROR is not None else "")
            )
        self.model = _load_sod_model(device_id, compile)
        self.convergence = convergence
        self.device = create_device(device_id)
        self.enable_ema = enable_ema
        self.decay = decay
        self.scene_hold = scene_hold
        self.cut_smooth = cut_smooth
        self.convergence_ema = None
        self.tracker = SceneHoldTracker(decay, cut_smooth)
        self._track_state = None  # per-batch-slot list: {"tracker", "box", "frames_since_reinit"}

    def reset(self, enable_ema=None, decay=None, scene_hold=None, cut_smooth=None, **_ignored):
        if enable_ema is not None:
            self.enable_ema = enable_ema
        if decay is not None:
            self.decay = decay
        if scene_hold is not None:
            self.scene_hold = scene_hold
        if cut_smooth is not None:
            self.cut_smooth = cut_smooth
        self.tracker.set_decay(self.decay)
        self.tracker.set_cut_smooth(self.cut_smooth)
        self.tracker.reset()
        self.convergence_ema = None
        self._track_state = None

    @staticmethod
    def _bbox_from_mask(mask, H, W):
        """mask: 2D bool tensor (H, W). Returns (x, y, w, h), int, at least 1x1.
        Falls back to the full frame when the mask has no True pixel at all."""
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            return (0, 0, W, H)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return (x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1))

    def _detect_box(self, rgb_single, depth_single, H, W):
        """Runs the real SOD model once on a single (1, C, H, W) frame and returns a
        bounding box in the ORIGINAL frame's own pixel coordinates (SOD's own saliency
        map resolution is upsampled to match, so this box lines up directly with both
        the cv2 tracker, which runs on the full-resolution frame, and `depth_single`)."""
        rgb_b = rgb_single.unsqueeze(0).to(self.device)
        depth_b = depth_single.unsqueeze(0).to(self.device)
        with torch.inference_mode(), autocast(self.device):
            saliency_map, _ = self.model.infer(rgb_b, depth_b)
        sal_full = F.interpolate(saliency_map.float(), size=(H, W), mode="bilinear",
                                 align_corners=False)[0, 0]
        return self._bbox_from_mask(sal_full > 0.5, H, W)

    @staticmethod
    def _track_scale_factor(H, W):
        """ADR-395 real perf fix: CSRT's own per-frame CPU cost (and the GPU->CPU
        transfer feeding it) scales with the frame's own resolution -- running it on
        a full 4K frame every single frame (the real reported cause of a 4K job
        dropping from ~5fps to ~2.1fps) is both unnecessary and far more expensive
        than sod_v1's own entirely-GPU-resident hot path. CSRT only needs enough
        detail to track the subject's own appearance, not native resolution -- same
        spirit as SOD's own 192x192 input and the Hybrid Frame Rate motion scan's
        64x36 downscale, just a size actually suited to appearance tracking rather
        than a coarse signal. Never upscales (a sub-480p source tracks at its own
        native size, unchanged)."""
        longest = max(H, W)
        return min(1.0, SodTrackEstimator._TRACK_MAX_DIM / longest) if longest > 0 else 1.0

    @staticmethod
    def _frame_to_bgr(rgb_single, scale=1.0):
        """Downscales ON GPU (cheap) BEFORE the CPU transfer, so both the PCIe
        transfer and cv2's own per-frame work shrink with `scale` -- not just the
        cv2.cvtColor/resize cost, the data volume moved off the GPU too."""
        if scale < 1.0:
            new_h = max(1, round(rgb_single.shape[-2] * scale))
            new_w = max(1, round(rgb_single.shape[-1] * scale))
            rgb_single = F.interpolate(rgb_single.unsqueeze(0), size=(new_h, new_w),
                                       mode="bilinear", align_corners=False)[0]
        frame = (rgb_single.permute(1, 2, 0).cpu().float() * 255).clamp(0, 255).byte().numpy()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _clamp_track_box(x, y, w, h, frame_w, frame_h):
        """ADR-399: real reported crash -- cv2's CSRT tracker.init() asserts inside
        its own internal cv::resize (!ssize.empty()) when handed a bounding box that
        is too small relative to the frame (its internal padded/template resize math
        rounds a dimension down to 0) or that falls even slightly outside the frame
        bounds (an off-by-one from independently rounding x/w and y/h after the
        ADR-395 downscale). Confirmed live on a real 4K job: a small/thin subject
        box, shrunk further by ADR-395's downscale, tripped this. Clamps the box
        CSRT actually tracks to a safe minimum size and strictly inside the frame --
        this only affects what CSRT itself follows between reinits, not the
        full-resolution `box` this same reinit already uses for its own depth
        sampling (computed before this clamp, from the un-clamped SOD detection)."""
        w = min(frame_w, max(SodTrackEstimator._TRACK_MIN_BOX_DIM, w))
        h = min(frame_h, max(SodTrackEstimator._TRACK_MIN_BOX_DIM, h))
        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))
        return (int(x), int(y), int(w), int(h))

    def _reinit(self, rgb_single, depth_single, H, W):
        box = self._detect_box(rgb_single, depth_single, H, W)  # full-res -- SOD only runs here
        scale = self._track_scale_factor(H, W)
        x, y, w, h = box
        scaled_box = (round(x * scale), round(y * scale),
                     max(1, round(w * scale)), max(1, round(h * scale)))
        frame = self._frame_to_bgr(rgb_single, scale)
        frame_h, frame_w = frame.shape[0], frame.shape[1]
        scaled_box = self._clamp_track_box(*scaled_box, frame_w, frame_h)
        tracker = _create_csrt_tracker()
        try:
            tracker.init(frame, scaled_box)
        except cv2.error as e:
            # ADR-399: belt-and-braces -- even after clamping, don't let an
            # unforeseen CSRT edge case crash a multi-hour video job. Fall back to
            # the raw SOD detection with no appearance tracking for this shot;
            # the next scheduled reinit (or scene cut) tries CSRT again fresh.
            print(f"[iw3] sod_track: CSRT tracker.init() failed ({e}) -- "
                  f"falling back to untracked detection for this shot.", file=sys.stderr)
            tracker = None
        return {"tracker": tracker, "box": box, "scale": scale, "frames_since_reinit": 0}

    def __call__(self, rgb, depth, reset_pts=None):
        B = depth.shape[0]
        reset_pts = reset_pts if reset_pts is not None else [False] * B
        if self._track_state is None or len(self._track_state) != B:
            self._track_state = [None] * B

        H, W = depth.shape[-2], depth.shape[-1]
        results = []
        for i in range(B):
            state = self._track_state[i]
            if state is None:
                state = self._reinit(rgb[i], depth[i], H, W)
            elif state["frames_since_reinit"] >= self._REINIT_INTERVAL:
                state = self._reinit(rgb[i], depth[i], H, W)
            elif state["tracker"] is None:
                # ADR-399: tracker.init() failed for this shot -- hold the raw SOD
                # detection's box unchanged until the next scheduled reinit/cut.
                state["frames_since_reinit"] += 1
            else:
                scale = state["scale"]
                ok, box = state["tracker"].update(self._frame_to_bgr(rgb[i], scale))
                if ok:
                    # box is in the DOWNSCALED frame's own coordinates -- convert back
                    # to full resolution before storing/using it against `depth`.
                    sx, sy, sw, sh = box
                    x = max(0, min(W - 1, int(round(sx / scale))))
                    y = max(0, min(H - 1, int(round(sy / scale))))
                    w = max(1, min(W - x, int(round(sw / scale))))
                    h = max(1, min(H - y, int(round(sh / scale))))
                    state["box"] = (x, y, w, h)
                    state["frames_since_reinit"] += 1
                else:
                    # tracker lost the subject -- reinitialize immediately rather than
                    # waiting for the next scheduled interval.
                    state = self._reinit(rgb[i], depth[i], H, W)

            x, y, w, h = state["box"]
            d = depth[i, :, y:y + h, x:x + w].flatten()
            p = ConvergenceEstimator._quantile_position(
                d, self.convergence, depth.device, depth.dtype).reshape(1, 1, 1).clamp(0, 1)

            if self.enable_ema:
                if self.scene_hold:
                    out = self.tracker.update(p.item())
                    p = torch.full_like(p, out)
                    if reset_pts[i]:
                        self.tracker.mark_cut()
                else:
                    if self.convergence_ema is None:
                        self.convergence_ema = p.clone()
                    else:
                        self.convergence_ema = self.decay * self.convergence_ema + (1. - self.decay) * p
                    p = self.convergence_ema.clone()
                    if reset_pts[i]:
                        self.convergence_ema = None

            results.append(p)
            self._track_state[i] = None if reset_pts[i] else state

        return torch.stack(results, dim=0)
