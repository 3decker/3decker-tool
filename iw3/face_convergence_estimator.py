import torch

try:
    import cv2
    _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(_cascade_path)
    _CV2_AVAILABLE = not _face_cascade.empty()
except Exception:
    _CV2_AVAILABLE = False
    _face_cascade = None


class FaceConvergenceEstimator():
    """
    Places the 3D screen plane at the depth of detected faces.
    Uses OpenCV Haar cascade — no model downloads required.
    Falls back to center-weighted depth when no faces are found in a frame.
    """

    def __init__(self, enable_ema=False, decay=0.9):
        if not _CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for face_detect convergence mode.\n"
                "Run in nunif-prompt.bat: pip install opencv-python"
            )
        self.enable_ema = enable_ema
        self.decay = decay
        self.convergence_ema = None

    def reset(self, enable_ema=None, decay=None):
        if enable_ema is not None:
            self.enable_ema = enable_ema
        if decay is not None:
            self.decay = decay
        self.convergence_ema = None

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

    def _detect_face_depth(self, gray_np, depth_single):
        """Return median depth inside detected face bounding boxes, or None if no faces."""
        H, W = depth_single.shape[-2], depth_single.shape[-1]
        faces = _face_cascade.detectMultiScale(gray_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None

        mask = torch.zeros(H, W, device=depth_single.device, dtype=torch.bool)
        for (x, y, w, h) in faces:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
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
            # Convert [0,1] float CHW tensor → uint8 grayscale numpy for Haar cascade
            frame_rgb = (rgb[i].permute(1, 2, 0).cpu().float() * 255).clamp(0, 255).byte().numpy()
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            face_d = self._detect_face_depth(gray, depth[i])

            if face_d is not None:
                p = face_d.to(depth.device).reshape(1, 1, 1).clamp(0, 1)
            else:
                p = self._center_weighted_depth(depth[i]).reshape(1, 1, 1).clamp(0, 1)

            if self.enable_ema:
                if self.convergence_ema is None:
                    self.convergence_ema = p.clone()
                else:
                    self.convergence_ema = self.decay * self.convergence_ema + (1.0 - self.decay) * p
                results.append(self.convergence_ema.clone())
                if reset_pts is not None and reset_pts[i]:
                    self.reset()
            else:
                results.append(p.clone())

        return torch.stack(results, dim=0)  # shape: (B, 1, 1, 1)
