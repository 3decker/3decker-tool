import torch
from nunif.utils.ui import TorchHubDir
from nunif.models import load_model
from nunif.device import create_device, autocast
from .hub_dir import HUB_MODEL_DIR
from .convergence_tracker import SceneHoldTracker


SOD_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_sod_v1_20260125.pth"

# Real confirmed finding: thresholding the raw saliency map at a hard 0.5 cutoff every
# frame, with zero memory of what won last frame, lets two similarly-salient regions in
# noisy real footage flip which one counts as "the subject" from pure per-frame noise --
# a sustained, irregular flicker that SceneHoldTracker's own deadband/decay (one stage
# later, smoothing the resulting scalar) cannot fully absorb, since it only has
# cut-transition logic, not ongoing within-shot mask-noise filtering. subject_lock (see
# __init__ below) fixes this one stage earlier, at mask-selection time, by requiring a
# challenger region to keep winning for several consecutive frames before it actually
# replaces the held region. Two masks are treated as "the same subject, naturally
# drifting" (not a real switch) once they overlap at least this much.
_SUBJECT_MATCH_IOU = 0.5


class ConvergenceEstimator():
    def __init__(self, convergence, device_id, enable_ema=False, decay=0.9, compile=False, scene_hold=False,
                 cut_smooth=0, subject_lock=0):
        with TorchHubDir(HUB_MODEL_DIR):
            self.model, _ = load_model(SOD_URL, device_ids=[device_id], weights_only=True)
            self.model = self.model.eval().fuse()
            # SOD input is resized to 192x192, recompilation does not occur due to image size differences.
            # However, recompilation does occur when the batch size changes.
            self.model = self.model.compile(mode=compile)
            self.convergence = convergence

        self.device = create_device(device_id)
        self.enable_ema = enable_ema
        self.decay = decay
        # ADR-232: scene_hold=False (default) keeps the original plain-EMA behavior
        # (this project's own long-standing default, unchanged since before ADR-231) --
        # scene_hold=True opts into ADR-231's settle-then-hold-with-deadband tracker.
        # Both pieces of state are kept regardless of which is active so toggling
        # scene_hold mid-run (GUI checkbox change between runs) never carries stale
        # state from the other mode into the next call.
        self.scene_hold = scene_hold
        self.cut_smooth = cut_smooth
        # subject_lock=0 (default) keeps the original per-frame, zero-memory mask
        # threshold byte-identical. A nonzero value requires a new candidate mask to
        # keep winning for that many consecutive frames before it replaces the
        # currently held one -- see depth_position_from_ratio_with_subject_lock().
        self.subject_lock = max(0, int(subject_lock or 0))
        self._lock_state = None
        self.convergence_ema = None
        self.tracker = SceneHoldTracker(decay, cut_smooth)

    def reset(self, enable_ema=None, decay=None, scene_hold=None, cut_smooth=None, subject_lock=None):
        if enable_ema is not None:
            self.enable_ema = enable_ema
        if decay is not None:
            self.decay = decay
        if scene_hold is not None:
            self.scene_hold = scene_hold
        if cut_smooth is not None:
            self.cut_smooth = cut_smooth
        if subject_lock is not None:
            self.subject_lock = max(0, int(subject_lock or 0))
        self.tracker.set_decay(self.decay)
        self.tracker.set_cut_smooth(self.cut_smooth)
        self.tracker.reset()
        self.convergence_ema = None
        self._lock_state = None

    @staticmethod
    def _quantile_position(d, pos, device, dtype):
        if d.numel() == 0:
            # Depth values are normalized to the [0, 1] range,
            # with 0.5 corresponding to the mid depth.
            return torch.tensor(0.5, device=device, dtype=dtype)

        q01 = d.quantile(0.1)
        q09 = d.quantile(0.9)
        q_range = (q09 - q01)
        if q_range < 1e-6:
            return q01
        # Convert pos from [0, 1] into an internal range of [-1, 2],
        # effectively 3x the usable depth range around the central region.
        center = (q01 + q09) / 2
        expanded_range = q_range * 3.0
        return center + (pos - 0.5) * expanded_range

    @staticmethod
    def depth_position_from_ratio(saliency_map, depth, pos):
        B = depth.shape[0]
        result = []
        for i in range(B):
            d = depth[i].flatten()
            mask = saliency_map[i].flatten() > 0.5
            q_pos = ConvergenceEstimator._quantile_position(
                d[mask], pos, saliency_map.device, saliency_map.dtype)
            result.append(q_pos)
        return torch.stack(result, dim=0).reshape(B, 1, 1, 1).clamp(0, 1)

    @staticmethod
    def _mask_iou(a, b):
        if a.shape != b.shape:
            return 0.0
        union = (a | b).sum().item()
        if union == 0:
            # both empty -- nothing changed, treat as the same "subject"
            return 1.0
        return (a & b).sum().item() / union

    def depth_position_from_ratio_with_subject_lock(self, saliency_map, depth, pos, reset_pts):
        """subject_lock > 0 variant of depth_position_from_ratio(): keeps a persistent
        "held" mask per batch slot across calls and only switches it to a new candidate
        mask once that candidate (or a consistent successor) has won `self.subject_lock`
        consecutive frames in a row. Resets on a real scene cut (reset_pts[i])."""
        B = depth.shape[0]
        if self._lock_state is None or len(self._lock_state) != B:
            self._lock_state = [None] * B

        result = []
        for i in range(B):
            mask = saliency_map[i].flatten() > 0.5
            state = self._lock_state[i]
            if state is None:
                state = {"held": mask, "challenger": None, "count": 0}
            elif self._mask_iou(mask, state["held"]) >= _SUBJECT_MATCH_IOU:
                # The same subject is still winning -- let the held region track its own
                # natural drift (slight boundary changes), same as it always could.
                state["held"] = mask
                state["challenger"] = None
                state["count"] = 0
            else:
                # A different region won this frame -- it only replaces the held region
                # once it (or a matching successor) keeps winning for `subject_lock`
                # consecutive frames.
                if state["challenger"] is not None and self._mask_iou(mask, state["challenger"]) >= _SUBJECT_MATCH_IOU:
                    state["count"] += 1
                else:
                    state["challenger"] = mask
                    state["count"] = 1
                if state["count"] >= self.subject_lock:
                    state["held"] = state["challenger"]
                    state["challenger"] = None
                    state["count"] = 0

            d = depth[i].flatten()[state["held"]]
            q_pos = self._quantile_position(d, pos, saliency_map.device, saliency_map.dtype)
            result.append(q_pos)
            self._lock_state[i] = None if reset_pts[i] else state
        return torch.stack(result, dim=0).reshape(B, 1, 1, 1).clamp(0, 1)

    def __call__(self, rgb, depth, reset_pts=None):
        rgb = rgb.to(self.device)
        depth = depth.to(self.device)
        reset_pts = reset_pts if reset_pts is not None else [False] * depth.shape[0]

        with torch.inference_mode(), autocast(self.device):
            saliency_map, depth_scaled = self.model.infer(rgb, depth)
            if getattr(self, "subject_lock", 0) > 0:
                z_pos = self.depth_position_from_ratio_with_subject_lock(
                    saliency_map, depth_scaled, self.convergence, reset_pts)
            else:
                z_pos = self.depth_position_from_ratio(saliency_map, depth_scaled, self.convergence)

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
