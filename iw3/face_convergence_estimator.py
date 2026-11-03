import torch
import torch.nn.functional as F

from .convergence_tracker import SceneHoldTracker

_CV2_IMPORT_ERROR = None
_CV2_CASCADE_MISSING = False
try:
    import cv2
    _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(_cascade_path)
    _CV2_AVAILABLE = not _face_cascade.empty()
    if not _CV2_AVAILABLE:
        # Real user report: opencv-python 5.0.0.93 (an unpinned pip install's default at
        # the time) imports fine but ships without its Haar cascade data files at all, so
        # this load silently fails -- CascadeClassifier.empty() is the only signal. Telling
        # the user to "pip install opencv-contrib-python" again (the generic message
        # below) would just reinstall the same broken version, so this case gets its own
        # real diagnosis.
        _CV2_CASCADE_MISSING = True
except Exception as e:
    _CV2_AVAILABLE = False
    _face_cascade = None
    _CV2_IMPORT_ERROR = e


class FaceConvergenceEstimator():
    """
    Places the 3D screen plane at the depth of detected faces.
    Uses OpenCV Haar cascade — no model downloads required.
    Falls back to center-weighted depth when no faces are found in a frame.
    """

    # ADR-412: real reported perf gap -- face_detect ran at 3.67fps vs clip_shot's
    # 5.27fps on the same 4K job. Root cause: detectMultiScale() ran on the FULL
    # native-resolution frame every single frame (unlike clip_shot, which is
    # GPU-resident and always shrinks to 224x224 first) -- Haar cascade cost scales
    # steeply with image area from its own multi-scale sliding-window search. Same
    # fix/reasoning as SodTrackEstimator's own ADR-395 (_TRACK_MAX_DIM): faces don't
    # need native 4K detail to find, just enough pixels to resolve one. Never upscales
    # (a sub-480p source detects at its own native size, unchanged).
    _DETECT_MAX_DIM = 480

    def __init__(self, enable_ema=False, decay=0.9, scene_hold=False, cut_smooth=0):
        if not _CV2_AVAILABLE:
            if _CV2_CASCADE_MISSING:
                raise ImportError(
                    "OpenCV (cv2) is installed, but this version is missing its face "
                    "detection data file (a known packaging issue in some opencv-python "
                    "releases, including 5.0.0.93).\n"
                    # ADR-396: opencv-contrib-python, not plain opencv-python -- this
                    # project's own baseline pin switched so sod_track's own
                    # cv2.TrackerCSRT_create (contrib-only) also works; contrib is a
                    # strict superset, so this still fixes the missing cascade data too.
                    "Run in nunif-prompt.bat: pip install opencv-contrib-python==4.10.0.84"
                )
            raise ImportError(
                "OpenCV (cv2) is required for face_detect convergence mode.\n"
                f"Run in nunif-prompt.bat: pip install opencv-contrib-python==4.10.0.84"
                + (f"\n({_CV2_IMPORT_ERROR.__class__.__name__}: {_CV2_IMPORT_ERROR})"
                   if _CV2_IMPORT_ERROR is not None else "")
            )
        self.enable_ema = enable_ema
        self.decay = decay
        # ADR-232: see ConvergenceEstimator's own comment -- scene_hold=False (default)
        # keeps the original plain-EMA behavior; True opts into ADR-231's tracker.
        self.scene_hold = scene_hold
        self.cut_smooth = cut_smooth
        self.convergence_ema = None
        self.tracker = SceneHoldTracker(decay, cut_smooth)

    def to(self, device):
        """No neural net here (face_detect uses a CPU-only cv2 Haar cascade), but the
        running convergence EMA is still a real tensor on whatever device the depth
        values it was computed from were on. ADR-407: previously
        _release_pause_vram() (iw3/utils.py) skipped this estimator entirely since it
        has no .model attribute, so this one small tensor was never moved."""
        if torch.is_tensor(self.convergence_ema):
            self.convergence_ema = self.convergence_ema.to(device)
        return self

    def reset(self, enable_ema=None, decay=None, scene_hold=None, cut_smooth=None):
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

    @staticmethod
    def _detect_scale_factor(H, W):
        """ADR-412: mirrors SodTrackEstimator._track_scale_factor (ADR-395) exactly --
        see this class's own _DETECT_MAX_DIM docstring for why."""
        longest = max(H, W)
        return min(1.0, FaceConvergenceEstimator._DETECT_MAX_DIM / longest) if longest > 0 else 1.0

    @staticmethod
    def _to_gray_np(rgb_single, scale=1.0):
        """ADR-412: downscales ON GPU (cheap) before the CPU transfer -- same ADR-395
        pattern as SodTrackEstimator._frame_to_bgr -- so both the PCIe transfer and
        cv2's own per-frame work shrink with `scale`, not just the detection cost."""
        if scale < 1.0:
            new_h = max(1, round(rgb_single.shape[-2] * scale))
            new_w = max(1, round(rgb_single.shape[-1] * scale))
            rgb_single = F.interpolate(rgb_single.unsqueeze(0), size=(new_h, new_w),
                                       mode="bilinear", align_corners=False)[0]
        frame = (rgb_single.permute(1, 2, 0).cpu().float() * 255).clamp(0, 255).byte().numpy()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _center_weighted_depth(depth):
        """Fallback: Gaussian-weighted depth average biased toward the center of the frame."""
        H, W = depth.shape[-2], depth.shape[-1]
        y = torch.linspace(-1.0, 1.0, H, device=depth.device)
        x = torch.linspace(-1.0, 1.0, W, device=depth.device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        weights = torch.exp(-(gx ** 2 + gy ** 2) * 2.0)
        d = depth.view(H, W)
        return (d * weights).sum() / weights.sum()

    def _detect_face_depth(self, gray_np, depth_single, scale=1.0):
        """Return median depth inside detected face bounding boxes, or None if no faces.
        `gray_np` may be downscaled by `scale` (ADR-412) -- detected boxes, and minSize,
        are scaled to/from depth_single's own full resolution so detection sensitivity
        to small/distant faces stays proportionally the same as it was at native
        resolution, not silently raised by the downscale."""
        H, W = depth_single.shape[-2], depth_single.shape[-1]
        # ADR-412: scale minSize down with the frame so a face that would have cleared
        # the old full-resolution 30px floor still clears it here, proportionally --
        # floored at 20px since a Haar cascade needs a few real pixels per feature to
        # work at all, a face that small would be unreliable to detect regardless.
        min_size = max(20, round(30 * scale))
        faces = _face_cascade.detectMultiScale(gray_np, scaleFactor=1.1, minNeighbors=5,
                                               minSize=(min_size, min_size))

        if len(faces) == 0:
            return None

        mask = torch.zeros(H, W, device=depth_single.device, dtype=torch.bool)
        for (x, y, w, h) in faces:
            if scale < 1.0:
                x, y, w, h = x / scale, y / scale, w / scale, h / scale
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(W, int(round(x + w))), min(H, int(round(y + h)))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

        face_depths = depth_single.view(H, W)[mask]
        if face_depths.numel() == 0:
            return None

        return face_depths.median()

    def __call__(self, rgb, depth, reset_pts=None):
        B = depth.shape[0]
        results = []

        for i in range(B):
            # ADR-412: downscale before the CPU transfer + Haar cascade (see
            # _DETECT_MAX_DIM's own docstring) -- was previously always full resolution.
            H, W = depth.shape[-2], depth.shape[-1]
            scale = self._detect_scale_factor(H, W)
            gray = self._to_gray_np(rgb[i], scale)

            face_d = self._detect_face_depth(gray, depth[i], scale)

            if face_d is not None:
                p = face_d.to(depth.device).reshape(1, 1, 1).clamp(0, 1)
            else:
                p = self._center_weighted_depth(depth[i]).reshape(1, 1, 1).clamp(0, 1)

            if self.enable_ema:
                if self.scene_hold:
                    out = self.tracker.update(p.item())
                    results.append(torch.full_like(p, out))
                    if reset_pts is not None and reset_pts[i]:
                        self.tracker.mark_cut()
                else:
                    if self.convergence_ema is None:
                        self.convergence_ema = p.clone()
                    else:
                        self.convergence_ema = self.decay * self.convergence_ema + (1.0 - self.decay) * p
                    results.append(self.convergence_ema.clone())
                    if reset_pts is not None and reset_pts[i]:
                        self.convergence_ema = None
            else:
                results.append(p.clone())

        return torch.stack(results, dim=0)  # shape: (B, 1, 1, 1)
