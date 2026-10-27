"""ADR-388: `depth_percentile` Convergence Plane mode.

Real request: sod_v1/face_detect both need an AI model (SOD saliency, or a face
cascade) to decide WHERE in the frame to look before picking a depth. Decker asked
for a zero-new-model option that uses the depth map iw3 already computes every
single frame, with no extra inference cost at all.

Mechanism: reuses ConvergenceEstimator._quantile_position() (the same histogram-based
"quantile position within the depth range" math sod_v1 already uses) applied to the
FULL frame's own depth histogram instead of a saliency-masked region -- so the existing
Convergence Bias value keeps its established meaning (0 = near edge of the frame's own
depth range/more pop-out, 1 = far edge/more recede), just measured over the whole
picture instead of one AI-picked subject. Smoothed with the same shared SceneHoldTracker
(convergence_tracker.py) sod_v1/face_detect already use, respecting Convergence
Smoothing / Hold Steady Per Scene / Cut Transition Smoothing identically. Subject Lock
does not apply here -- there is no per-frame subject pick to flicker between; see
gui.py's update_convergence_subject_lock().
"""
import torch

from .convergence_tracker import SceneHoldTracker
from .convergence_estimator import ConvergenceEstimator


class DepthPercentileEstimator():
    """
    Places the 3D screen plane at a quantile position within the FULL frame's own
    depth histogram -- no AI model, no extra inference cost. See module docstring.
    """

    def __init__(self, convergence, enable_ema=False, decay=0.9, scene_hold=False, cut_smooth=0):
        self.convergence = convergence
        self.enable_ema = enable_ema
        self.decay = decay
        self.scene_hold = scene_hold
        self.cut_smooth = cut_smooth
        self.convergence_ema = None
        self.tracker = SceneHoldTracker(decay, cut_smooth)
        # No AI model -- kept as None/model-less so the shared pause/resume VRAM-release
        # code in utils.py (getattr(convergence_model, "model", None)) is a harmless no-op.
        self.model = None
        self.device = None

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

    @staticmethod
    def depth_position_from_full_frame(depth, pos):
        """Same math as ConvergenceEstimator.depth_position_from_ratio(), but over the
        WHOLE frame's depth values instead of a saliency-masked region."""
        B = depth.shape[0]
        result = []
        for i in range(B):
            d = depth[i].flatten()
            q_pos = ConvergenceEstimator._quantile_position(d, pos, depth.device, depth.dtype)
            result.append(q_pos)
        return torch.stack(result, dim=0).reshape(B, 1, 1, 1).clamp(0, 1)

    def __call__(self, rgb, depth, reset_pts=None):
        reset_pts = reset_pts if reset_pts is not None else [False] * depth.shape[0]
        z_pos = self.depth_position_from_full_frame(depth, self.convergence)

        if self.enable_ema:
            results = []
            for i in range(z_pos.shape[0]):
                p = z_pos[i]
                if self.scene_hold:
                    out = self.tracker.update(p.item())
                    results.append(torch.full_like(p, out))
                    if reset_pts[i]:
                        self.tracker.mark_cut()
                else:
                    if self.convergence_ema is None:
                        self.convergence_ema = p.clone()
                    else:
                        self.convergence_ema = self.decay * self.convergence_ema + (1. - self.decay) * p
                    results.append(self.convergence_ema.clone())
                    if reset_pts[i]:
                        self.convergence_ema = None
            z_pos = torch.stack(results, dim=0)

        return z_pos
