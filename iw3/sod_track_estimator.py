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

    def __init__(self, convergence, device_id, enable_ema=False, decay=0.9, compile=False,
                 scene_hold=False, cut_smooth=0):
        if not _CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for sod_track convergence mode.\n"
                f"Run in nunif-prompt.bat: pip install opencv-python==4.10.0.84"
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
    def _frame_to_bgr(rgb_single):
        frame = (rgb_single.permute(1, 2, 0).cpu().float() * 255).clamp(0, 255).byte().numpy()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _reinit(self, rgb_single, depth_single, H, W):
        box = self._detect_box(rgb_single, depth_single, H, W)
        tracker = _create_csrt_tracker()
        tracker.init(self._frame_to_bgr(rgb_single), box)
        return {"tracker": tracker, "box": box, "frames_since_reinit": 0}

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
            else:
                ok, box = state["tracker"].update(self._frame_to_bgr(rgb[i]))
                if ok:
                    x, y, w, h = box
                    x = max(0, min(W - 1, int(round(x))))
                    y = max(0, min(H - 1, int(round(y))))
                    w = max(1, min(W - x, int(round(w))))
                    h = max(1, min(H - y, int(round(h))))
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
