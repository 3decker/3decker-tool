import torch
import numpy as np
import cv2
import torch.nn.functional as F


_UNSET = object()


class TemporalStabilizer:
    """Approximates what a real video-aware depth model (VDA_L) gets for free from its
    own architecture -- per-pixel stability over time -- for a single-frame model like
    Any_V3_Mono_01 that has zero memory between frames.

    EMAMinMaxScaler (above) only smooths the overall near/far RANGE over time, a single
    scalar; it can't touch a specific object's depth flickering frame to frame while the
    overall range stays put. This works on the actual per-pixel depth values instead.

    The naive version of this idea -- blend each pixel with its own value from the
    previous frame -- ghosts/smears the instant anything moves, since "this pixel" and
    "the same real-world point" stop being the same place. This computes real optical
    flow on the RGB image first (never on the depth map itself, which is too noisy to
    track motion reliably) and warps the previous frame's depth to where its content
    actually moved to before blending, so a moving object's OWN depth history follows
    it around instead of leaking into whatever now occupies its old screen position.
    Confidence in the warp is reduced wherever the flow is large (fast/unreliable
    motion), tapering blending back toward the fresh per-frame result there."""

    def __init__(self, enabled=False, strength=0.7,
                 max_shift_velocity=None, flat_region_boost=0.0, edge_protection=0.0):
        self.enabled = enabled
        self.strength = strength
        self.prev_gray = None
        self.prev_depth = None
        # All three below default to their exact prior (no-op) behavior -- adding
        # these must never change existing jobs' output unless explicitly opted in.
        # max_shift_velocity=None: no cap on how much a pixel's stabilized depth
        #   can move frame-to-frame (same units as the depth tensor itself, e.g.
        #   0-1 if already minmax-normalized). Set to a real number to hard-limit
        #   the per-frame delta regardless of what the flow-confidence blend below
        #   would otherwise allow -- catches occasional large jumps the optical-
        #   flow-based blend alone doesn't fully suppress.
        # flat_region_boost=0.0: extra smoothing strength (added on top of the
        #   normal motion-based local_alpha, clamped to 1.0) applied where the
        #   CURRENT frame's own depth is locally flat/uniform -- a featureless
        #   surface (a wall, sky) has no real depth detail to preserve, so any
        #   frame-to-frame variation there is much more likely to be noise than
        #   signal, and can be smoothed harder than a real depth-edge-carrying
        #   region without losing anything real.
        # edge_protection=0.0: smoothing strength SUBTRACTED (clamped >= 0) where
        #   the current frame has a strong depth edge -- keeps temporal smoothing
        #   from dragging/smearing a real silhouette boundary across frames, same
        #   underlying Sobel-gradient-edge concept as _depth_edge_mask in
        #   depth_blend.py (independently reimplemented here rather than shared,
        #   since the two modules are otherwise unrelated).
        self.max_shift_velocity = max_shift_velocity
        self.flat_region_boost = float(flat_region_boost)
        self.edge_protection = float(edge_protection)

    def reset(self, enabled=None, strength=None,
              max_shift_velocity=_UNSET, flat_region_boost=None, edge_protection=None):
        # Clears frame-to-frame memory (used at every scene cut, alongside the EMA
        # scaler's own reset) without necessarily changing the enabled/strength config --
        # those only change when explicitly passed.
        # max_shift_velocity uses a distinct _UNSET sentinel (not None) because None
        # is ALSO its own legitimate "off" value -- using None as the "don't touch"
        # marker here (like the other params) would make it impossible to ever
        # explicitly turn it back off through reset().
        if enabled is not None:
            self.enabled = bool(enabled)
        if strength is not None:
            self.strength = float(strength)
        if max_shift_velocity is not _UNSET:
            self.max_shift_velocity = max_shift_velocity
        if flat_region_boost is not None:
            self.flat_region_boost = float(flat_region_boost)
        if edge_protection is not None:
            self.edge_protection = float(edge_protection)
        self.prev_gray = None
        self.prev_depth = None

    def stabilize(self, rgb_chw, depth_chw):
        if not self.enabled or rgb_chw is None:
            return depth_chw

        depth_np = depth_chw.squeeze(0).detach().float().cpu().numpy()
        h, w = depth_np.shape

        rgb_small = F.interpolate(rgb_chw.detach().float().unsqueeze(0), size=(h, w),
                                   mode="bilinear", align_corners=False).squeeze(0)
        rgb_np = (rgb_small.clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            # first frame of the scene -- nothing to stabilize against yet
            self.prev_gray = gray
            self.prev_depth = depth_np
            return depth_chw

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                             0.5, 3, 15, 3, 5, 1.2, 0)
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        warped_prev_depth = cv2.remap(self.prev_depth, map_x, map_y,
                                       interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)

        # Large flow = fast/unreliable motion -- taper trust in the warp back toward
        # the fresh per-frame depth rather than assume the warp is still accurate.
        flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        distrust = np.clip(flow_mag / 15.0, 0.0, 1.0)
        local_alpha = self.strength * (1.0 - distrust)

        if self.flat_region_boost > 0.0 or self.edge_protection > 0.0:
            # Both read the CURRENT frame's own depth structure: a Sobel gradient
            # magnitude is a direct, cheap proxy for "how much real depth detail is
            # here" -- near zero in a flat/featureless region (a wall, sky), large
            # at a genuine depth edge/silhouette. Normalized against this frame's
            # own 95th-percentile gradient rather than a fixed constant, since raw
            # gradient scale varies by depth model/scene (same normalization
            # approach as _depth_edge_mask in depth_blend.py, independently
            # reimplemented here for this otherwise-unrelated module).
            grad_x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            ref = float(np.percentile(grad_mag, 95.0))
            edge_strength = np.clip(grad_mag / ref, 0.0, 1.0) if ref > 1e-6 else np.zeros_like(grad_mag)
            flatness = 1.0 - edge_strength

            if self.flat_region_boost > 0.0:
                local_alpha = np.clip(local_alpha + self.flat_region_boost * flatness, 0.0, 1.0)
            if self.edge_protection > 0.0:
                local_alpha = np.clip(local_alpha - self.edge_protection * edge_strength, 0.0, 1.0)

        stabilized = (local_alpha * warped_prev_depth + (1.0 - local_alpha) * depth_np).astype(np.float32)

        if self.max_shift_velocity is not None:
            # Hard cap on how much the OUTPUT can move from the previous OUTPUT
            # frame to frame, regardless of what the flow-confidence blend above
            # produced -- catches occasional large jumps that blend alone doesn't
            # fully suppress (e.g. a brief bad optical-flow estimate).
            delta = np.clip(stabilized - self.prev_depth, -self.max_shift_velocity, self.max_shift_velocity)
            stabilized = (self.prev_depth + delta).astype(np.float32)

        self.prev_gray = gray
        self.prev_depth = stabilized
        return torch.from_numpy(stabilized).unsqueeze(0).to(depth_chw.device, dtype=depth_chw.dtype)


def minmax_normalize(frame, min_value, max_value):
    if torch.is_tensor(min_value):
        min_value = min_value.to(frame.device)
        max_value = max_value.to(frame.device)

    scale = (max_value - min_value)
    if scale > 0:
        frame = (frame - min_value) / scale
        frame = frame.clamp(0.0, 1.0)
    else:
        # all zero
        frame = frame.clamp(0.0, 1.0)

    return frame


def max_normalize(frame, min_value, max_value):
    if torch.is_tensor(max_value):
        max_value = max_value.to(frame.device)
    if max_value > 0:
        frame = frame / max_value
        frame = frame.clamp(0.0, 1.0)
    else:
        # all zero
        frame = frame.clamp(0.0, 1.0)

    return frame


class MinMaxBuffer():
    def __init__(self, size, dtype, device):
        assert size > 0
        self.count = 0
        self.size = size * 2
        self.data = torch.zeros(self.size, dtype=dtype).to(device)

    def _add(self, value):
        index = self.count % self.size
        self.data[index] = value
        self.count += 1

    def _fill(self, min_value, max_value):
        self.data[0::2] = min_value
        self.data[1::2] = max_value

    def add(self, min_value, max_value):
        if self.count == 0:
            self._fill(min_value, max_value)
            self.count = 2
        else:
            self._add(min_value)
            self._add(max_value)

    def is_filled(self):
        return self.count >= self.size

    def get_minmax(self):
        return self.data.amin(), self.data.amax()


class EMAMinMaxScaler():
    #   SimpleMinMaxScaler: decay=0, buffer_size=1
    # IncrementalEMAScaler: decay=0.75, buffer_size=1
    #      WindowEMAScaler: decay=0.9, buffer_size=30
    def __init__(self, decay=0, buffer_size=1, mode="minmax", motion_adaptive=False, motion_spread=0.06):
        assert mode in {"minmax", "max"}
        self.normalize = {"minmax": minmax_normalize, "max": max_normalize}[mode]
        self.frame_queue = []
        assert buffer_size > 0
        self.reset(decay=decay, buffer_size=buffer_size, motion_adaptive=motion_adaptive, motion_spread=motion_spread)

    def reset(self, decay=None, buffer_size=None, motion_adaptive=None, motion_spread=None, **kwargs):
        # assert len(self.frame_queue) == 0  # need flush

        if decay is not None:
            self.decay = float(decay)
        if buffer_size is not None:
            self.buffer_size = int(buffer_size)
        if motion_adaptive is not None:
            self.motion_adaptive = bool(motion_adaptive)
        elif not hasattr(self, "motion_adaptive"):
            self.motion_adaptive = False
        if motion_spread is not None:
            self.motion_spread = float(motion_spread)
        elif not hasattr(self, "motion_spread"):
            self.motion_spread = 0.06
        self.min_value = None
        self.max_value = None
        self.frame_queue = []
        self.minmax_buffer = None
        # last raw (heavily subsampled) frame, used only to measure how much is
        # changing frame-to-frame for --ema-motion-adaptive. Reset at every scene
        # boundary along with everything else, so motion never bleeds across cuts.
        self._prev_motion_frame = None
        self._reset_activity()

    def _reset_activity(self):
        # Bookkeeping for "when/how much did motion-adaptive actually kick in" --
        # answers a real question users have no other way to check (the effective
        # decay is computed and used per-frame but never surfaced anywhere). Cheap
        # enough to always track when motion_adaptive is on: just accumulating
        # already-computed floats, not extra GPU/CPU work.
        self._activity_frames = 0
        self._activity_eased_frames = 0
        self._activity_motion_sum = 0.0
        self._activity_min_effective_decay = None

    def pop_activity(self):
        """Returns a summary of motion-adaptive behavior since the last pop/reset, or
        None if motion_adaptive is off or no frames were processed. Popping clears the
        counters, so the caller can call this at natural checkpoints (a scene boundary,
        or end of export) and get exactly what happened in that stretch."""
        if not self.motion_adaptive or self._activity_frames == 0:
            self._reset_activity()
            return None
        summary = {
            "frames": self._activity_frames,
            "eased_frames": self._activity_eased_frames,
            "avg_motion_score": self._activity_motion_sum / self._activity_frames,
            "min_effective_decay": self._activity_min_effective_decay,
            "base_decay": self.decay,
        }
        self._reset_activity()
        return summary

    def _motion_score(self, frame):
        """Cheap 0-1 estimate of how much this frame differs from the last one, scaled
        against the scene's own current depth range so it works the same regardless of
        which depth model's units are in play. Uses the RAW (pre-smoothing) frame, not
        the depth model's RGB input, so no extra plumbing is needed elsewhere -- and
        real motion (large, spatially-coherent change) reads very differently from
        ordinary per-frame estimation noise (small, incoherent change), so this stays a
        reasonable motion proxy even though it's derived from the noisy signal itself."""
        small = frame[..., ::4, ::4] if frame.ndim >= 2 else frame
        prev = self._prev_motion_frame
        self._prev_motion_frame = small.detach()
        if prev is None or prev.shape != small.shape or self.min_value is None:
            return 0.0
        diff = (small - prev).abs().mean()
        value_range = self.max_value - self.min_value
        if torch.is_tensor(value_range):
            value_range = value_range.clamp_min(1e-6)
        elif value_range <= 0:
            value_range = 1e-6
        return float((diff / value_range).clamp(0.0, 1.0))

    def _effective_decay(self, motion_score):
        if not self.motion_adaptive or motion_score is None:
            return self.decay
        # More motion right now -> ease decay DOWN so the smoothing reacts faster and
        # doesn't smear/lag behind real movement. Calm frames stay at the exact decay
        # you set -- this only ever makes things MORE responsive than your base
        # setting, never smoother, so it's a safe layer on top of it.
        return max(0.0, self.decay - motion_score * self.motion_spread)

    def get_minmax(self):
        assert self.minmax_buffer is not None and self.minmax_buffer.is_filled()
        return self.minmax_buffer.get_minmax()

    def __call__(self, frame, return_minmax=False):
        return self.update(frame, return_minmax=return_minmax)

    def update(self, frame, return_minmax=False):
        if self.minmax_buffer is None:
            self.minmax_buffer = MinMaxBuffer(self.buffer_size, dtype=frame.dtype, device=frame.device)
        motion_score = self._motion_score(frame) if self.motion_adaptive else None
        self.frame_queue.append(frame)
        self.minmax_buffer.add(frame.amin(), frame.amax())
        if not self.minmax_buffer.is_filled():
            # queued
            if return_minmax:
                return None, None, None
            else:
                return None

        min_value, max_value = self.get_minmax()
        if self.min_value is None:
            self.min_value = min_value
            self.max_value = max_value
        else:
            decay = self._effective_decay(motion_score)
            if self.motion_adaptive and motion_score is not None:
                self._activity_frames += 1
                self._activity_motion_sum += motion_score
                if decay < self.decay - 1e-9:
                    self._activity_eased_frames += 1
                if self._activity_min_effective_decay is None or decay < self._activity_min_effective_decay:
                    self._activity_min_effective_decay = decay
            self.min_value = decay * self.min_value + (1. - decay) * min_value
            self.max_value = decay * self.max_value + (1. - decay) * max_value

        frame = self.frame_queue.pop(0)
        frame = self.normalize(frame, self.min_value, self.max_value)

        if return_minmax:
            return (frame, self.min_value, self.max_value)
        else:
            return frame

    def flush(self, return_minmax=False):
        if not self.frame_queue:
            self.reset()
            return []

        if self.min_value is None:
            min_value, max_value = self.minmax_buffer.get_minmax()
        else:
            min_value, max_value = self.min_value, self.max_value

        if return_minmax:
            frames = [(self.normalize(frame, min_value, max_value),
                       min_value, max_value)
                      for frame in self.frame_queue]
            self.reset()
            return frames
        else:
            frames = [self.normalize(frame, min_value, max_value)
                      for frame in self.frame_queue]
            self.reset()
            return frames


def _test():
    import matplotlib.pyplot as plt
    x = [float(i) for i in range(100)]
    zeros = [0 for i in range(100)]

    x = torch.tensor(zeros + x + list(reversed(x)) + zeros, dtype=torch.float32)
    x = torch.stack([x, x + 10]).permute(1, 0).contiguous()

    scaler = EMAMinMaxScaler(decay=0.9, buffer_size=22)
    min_values = []
    max_values = []
    for frame in x:
        frame, min_value, max_value = scaler.update(frame, return_minmax=True)
        if min_value is not None:
            min_values.append(min_value)
            max_values.append(max_value)
    for frame, min_value, max_value in scaler.flush(return_minmax=True):
        min_values.append(min_value)
        max_values.append(max_value)

    min_values = torch.tensor(min_values)
    max_values = torch.tensor(max_values)

    x = torch.stack([
        x.permute(1, 0)[0],
        x.permute(1, 0)[1],
        min_values,
        max_values,
    ]).permute(1, 0)
    plt.plot(x)
    plt.show()


if __name__ == "__main__":
    _test()
