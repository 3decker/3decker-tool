"""ADR-420: `shot_lock` Convergence Plane mode.

Real user request, after reviewing a real sod_v1-vs-clip_shot comparison on their
own footage: "I like how wide and medium-wide shots look using 0.75, and close-ups
around 0.20 -- is there a way to lock the convergence plane to use those 2 values
depending on the shot?" Every existing AI Convergence mode (sod_v1/face_detect/
sod_track/depth_percentile/clip_shot) computes a fresh value from each scene's own
depth histogram (clip_shot only ever NUDGES that by at most 0.25) -- there was no
mode that just answers "is this wide or close-up?" and uses one of two values you
set directly, bypassing the per-scene depth math entirely.

Reuses ClipShotEstimator's own CLIP visual model/anchor embeddings (the same
"how wide/close is this shot" 0..1 score already built for clip_shot) by
composition, not inheritance -- this mode never touches
ConvergenceEstimator._quantile_position() the way clip_shot/depth_percentile do,
so inheriting from ClipShotEstimator would bring along machinery this mode
doesn't use. Only `_closeness()` is called; `ClipShotEstimator.__call__()`
(the depth_percentile bias it's built for) is never invoked on this helper
instance.

Three switch styles between the two locked values (ADR-420, real user choice from
the three options -- all three built so the user can compare them on real footage,
not just one guessed as "best"). The "shot type changes" trigger for snap/snap_ramp
is the CLIP classification itself flipping, checked fresh every frame -- NOT
`reset_pts` (see ADR-421): this mode needs no real scene boundary at all, unlike
every other auto convergence mode.
  snap       instant jump the moment the shot type changes -- simplest, can look
             like a visible pop at a hard cut.
  snap_ramp  same instant classification, but eases into the new value over
             --convergence-cut-smooth frames instead of popping in one frame --
             reuses the exact SceneHoldTracker ramp mechanism sod_v1/clip_shot
             already use for their own cut transitions. NOTE: if
             --convergence-cut-smooth is left at its default of 0, this is
             identical to plain `snap` -- the ramp only exists once that value is
             set above 0.
  blend      no hard classification at all -- continuously interpolates between
             the two values by the raw closeness score (0 = wide_value, 1 =
             closeup_value), smoothed by --convergence-smoothing's plain EMA.
             Never jarring, but a shot that's not clearly wide or close-up won't
             land on exactly either target value.
"""
import torch

from .clip_shot_estimator import ClipShotEstimator
from .convergence_tracker import SceneHoldTracker

SWITCH_MODES = ("snap", "snap_ramp", "blend")
DEFAULT_SWITCH = "snap_ramp"
DEFAULT_WIDE_VALUE = 0.75
DEFAULT_CLOSEUP_VALUE = 0.20


class ShotLockEstimator():
    def __init__(self, wide_value, closeup_value, device_id, switch=DEFAULT_SWITCH,
                 decay=0.9, cut_smooth=0):
        self.wide_value = float(wide_value)
        self.closeup_value = float(closeup_value)
        self.switch = switch if switch in SWITCH_MODES else DEFAULT_SWITCH
        self.decay = decay
        self.cut_smooth = int(cut_smooth or 0)
        # Composition, not inheritance -- see module docstring. convergence=0.5 and
        # the ema/scene_hold/cut_smooth args passed here are irrelevant: only
        # `_closeness()` is ever called on this instance, never `__call__()`.
        self._clip = ClipShotEstimator(convergence=0.5, device_id=device_id,
                                       enable_ema=False, decay=0.9,
                                       scene_hold=False, cut_smooth=0)
        # Only meaningful for snap/snap_ramp -- `blend` uses its own plain EMA
        # below instead, since it has no cut/classification concept at all.
        self.tracker = SceneHoldTracker(decay, self.cut_smooth if self.switch == "snap_ramp" else 0)
        self.blend_ema = None
        self._last_raw = None  # ADR-421: last frame's classified value, see __call__

    def to(self, device):
        """Mirrors ClipShotEstimator.to()/ADR-407 -- moves the composed CLIP model
        (and its fixed anchor embeddings) along with every pause/resume, same as
        every other convergence estimator that holds GPU state between frames."""
        self._clip.to(device)
        return self

    def reset(self, decay=None, cut_smooth=None, **_ignored):
        if decay is not None:
            self.decay = decay
        if cut_smooth is not None:
            self.cut_smooth = int(cut_smooth or 0)
        self.tracker.set_decay(self.decay)
        self.tracker.set_cut_smooth(self.cut_smooth if self.switch == "snap_ramp" else 0)
        self.tracker.reset()
        self.blend_ema = None
        self._last_raw = None

    def __call__(self, rgb, depth, reset_pts=None):
        B = depth.shape[0]
        reset_pts = reset_pts if reset_pts is not None else [False] * B
        closeness = self._clip._closeness(rgb, depth)  # 0 = wide .. 1 = close-up

        results = []
        for i in range(B):
            c = closeness[i].item()
            if self.switch == "blend":
                raw = self.wide_value + c * (self.closeup_value - self.wide_value)
                if self.blend_ema is None:
                    self.blend_ema = raw
                else:
                    self.blend_ema = self.decay * self.blend_ema + (1.0 - self.decay) * raw
                value = self.blend_ema
                if reset_pts[i]:
                    self.blend_ema = None
            else:
                raw = self.closeup_value if c >= 0.5 else self.wide_value
                # ADR-421: real bug -- decker runs with Scene Detection permanently off
                # for stereo generation, so `reset_pts` below is always all-False (by
                # design, same as every other auto convergence mode -- see ADR-402) and
                # SceneHoldTracker's snap-at-a-cut mechanism was NEVER firing after the
                # very first frame. snap/snap_ramp degraded into the same slow
                # settle-then-drift behavior sod_v1/clip_shot fall back to WITHOUT Scene
                # Detection -- which defeats the entire point of this mode (guaranteed to
                # land on exactly wide_value or closeup_value). Unlike those modes,
                # shot_lock doesn't need a real scene boundary at all: the wide/close-up
                # CLASSIFICATION changing IS the only signal this mode cares about, and
                # it's already computed fresh every frame -- so treat a classification
                # flip as its own cut, independent of Scene Detection/reset_pts.
                if self._last_raw is not None and raw != self._last_raw:
                    self.tracker.mark_cut()
                self._last_raw = raw
                value = self.tracker.update(raw)
                if reset_pts[i]:
                    self.tracker.mark_cut()
            results.append(value)

        v = torch.tensor(results, device=depth.device, dtype=depth.dtype).clamp(0, 1)
        return v.reshape(B, 1, 1, 1)
