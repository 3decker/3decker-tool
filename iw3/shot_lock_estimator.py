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
not just one guessed as "best").

Where snap/snap_ramp get their "shot changed" signal (ADR-421 -> ADR-422): ADR-421
first tried triggering a switch off the CLIP classification itself flipping, frame
to frame, since decker runs with Scene Detection permanently off and `reset_pts` is
therefore always all-False (see that ADR's own note). That worked for the original
bug (never snapping at all), but traded it for a worse one on real footage: CLIP's
per-frame score can wobble near the 0.5 line WITHIN one unbroken shot, and every
wobble was treated as a fresh cut -- "it jumps around" mid-shot, the opposite of
"locked." ADR-422: this mode now takes `log_reset_pts`, the SAME real,
independently-scanned scene boundaries Convergence's own scene log and Auto 3D
Strength's scene log already use (ADR-402/417) and which exist regardless of
Scene Detection -- so a real cut (not per-frame AI noise) is what triggers a
switch. Within a real scene, the first `SceneHoldTracker.settle` frames average
the CONTINUOUS closeness score (not the discrete wide/close pick) to decide which
side of 0.5 the shot really lands on, then that decision is LOCKED for the rest
of the scene -- immune to any further per-frame wobble until the next real cut.
`log_reset_pts=None` (a caller that never threads it through, e.g. process_image)
falls back to ADR-421's classification-flip trigger unchanged.
  snap       instant jump the moment a real cut happens -- simplest, can look
             like a visible pop at a hard cut.
  snap_ramp  same instant classification, but eases into the new value over
             --convergence-cut-smooth frames instead of popping in one frame --
             reuses the exact SceneHoldTracker ramp mechanism sod_v1/clip_shot
             already use for their own cut transitions. NOTE: if
             --convergence-cut-smooth is left at its default of 0, this is
             identical to plain `snap` -- the ramp only exists once that value is
             set above 0.
  blend      no hard classification at all -- continuously interpolates between
             the three values by the raw closeness score (wide_value at
             closeness=0, mid_value at closeness=0.5, closeup_value at
             closeness=1), smoothed by --convergence-smoothing's plain EMA.
             Never jarring, but a shot that's not clearly wide, medium, or
             close-up won't land on exactly any target value.

ADR-429: real follow-up question -- "is the AI able to detect mid shots too?"
The CLIP closeness score was always a continuous 0(wide)..1(close-up) spectrum,
just read with a single cut point at 0.5 for the original 2-way pick. Rather than
teaching CLIP a brand new third "medium shot" visual pattern (unlike close-up's
"one big centered subject" or wide's "busy/distant scene", "medium" has no single
fixed look -- a synthetic calibration image for it would be a much shakier anchor
than the other two), this just reads the SAME existing spectrum in three equal
zones instead of two: below 1/3 is wide, above 2/3 is close-up, and the middle
third -- genuinely neither -- is the new mid_value. See _classify()/_blend_value().

ADR-434: real follow-up request -- "what about if I don't want to use a Mid Shot
value, can I make it empty instead?" `mid_value=None` now means "disabled", not
"use the default" (the GUI's text box being literally empty is how it asks for
this; a real default is still filled in for any CLI/preset caller that never
passes the field at all, same as before -- only an EXPLICIT None, which only the
empty-box case produces, reaches here). With it disabled, _classify()/_blend_value()
fall straight back to ADR-420's original 2-way split at the 0.5 midpoint -- no
middle third, no middle target value, exactly the pre-ADR-429 behavior.
"""
import torch

from .clip_shot_estimator import ClipShotEstimator
from .convergence_tracker import SceneHoldTracker

SWITCH_MODES = ("snap", "snap_ramp", "blend")
DEFAULT_SWITCH = "snap_ramp"
DEFAULT_WIDE_VALUE = 0.75
DEFAULT_MID_VALUE = 0.475
DEFAULT_CLOSEUP_VALUE = 0.20
# ADR-429: the two cut points splitting the continuous 0(wide)..1(close-up)
# closeness score into three equal zones -- see module docstring.
MID_LOW = 1.0 / 3.0
MID_HIGH = 2.0 / 3.0


class ShotLockEstimator():
    def __init__(self, wide_value, closeup_value, device_id, switch=DEFAULT_SWITCH,
                 decay=0.9, cut_smooth=0, mid_value=None):
        self.wide_value = float(wide_value)
        self.closeup_value = float(closeup_value)
        # ADR-434: None now means "disabled" (the GUI's Mid Shot Value box left
        # empty) -- a 2-way split at 0.5, no middle target at all. A caller that
        # wants the real default and never passes this field at all gets
        # DEFAULT_MID_VALUE from its own default= (create_parser()/the model
        # factory), never reaching this None branch.
        self.mid_value = None if mid_value is None else float(mid_value)
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
        self._last_raw = None  # ADR-421 fallback path: last frame's classified value
        # ADR-422: per-real-scene settle-then-lock state (only used when the caller
        # supplies log_reset_pts). _pending_scene_reset mirrors SceneHoldTracker's own
        # cut_pending -- True means the FIRST frame seen so far starts a fresh scene
        # (also true for the very first frame of the whole movie).
        self._pending_scene_reset = True
        self._scene_running_c = None
        self._scene_n = 0
        self._scene_locked_value = None

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
        self._pending_scene_reset = True
        self._scene_running_c = None
        self._scene_n = 0
        self._scene_locked_value = None

    def _classify(self, c):
        """ADR-429: 3-way pick along the same close/wide spectrum the original
        2-way pick already used (see module docstring) -- below MID_LOW is wide,
        above MID_HIGH is close-up, the middle third is mid_value. ADR-434: with
        mid_value disabled (None), falls back to the original 2-way split at 0.5."""
        if self.mid_value is None:
            return self.wide_value if c < 0.5 else self.closeup_value
        if c < MID_LOW:
            return self.wide_value
        if c > MID_HIGH:
            return self.closeup_value
        return self.mid_value

    def _blend_value(self, c):
        """ADR-429: continuous 3-point version of the same spectrum, for `blend`
        switch mode only -- wide_value at c=0, mid_value at c=0.5, closeup_value
        at c=1, straight-line interpolated between each adjacent pair. ADR-434:
        with mid_value disabled (None), straight-line interpolated directly from
        wide_value (c=0) to closeup_value (c=1), no middle anchor point."""
        if self.mid_value is None:
            return self.wide_value + c * (self.closeup_value - self.wide_value)
        if c <= 0.5:
            t = c / 0.5
            return self.wide_value + t * (self.mid_value - self.wide_value)
        t = (c - 0.5) / 0.5
        return self.mid_value + t * (self.closeup_value - self.mid_value)

    def __call__(self, rgb, depth, reset_pts=None, log_reset_pts=None):
        B = depth.shape[0]
        reset_pts = reset_pts if reset_pts is not None else [False] * B
        # ADR-422: log_reset_pts (when supplied) is the real, independently-scanned
        # scene-boundary signal -- see module docstring. `has_real_cuts` must be
        # checked BEFORE defaulting the list below, so "never supplied at all" (use
        # the ADR-421 classification-flip fallback) stays distinguishable from
        # "supplied, but no cut on this particular frame."
        has_real_cuts = log_reset_pts is not None
        log_reset_pts = log_reset_pts if has_real_cuts else [False] * B
        closeness = self._clip._closeness(rgb, depth)  # 0 = wide .. 1 = close-up

        results = []
        for i in range(B):
            c = closeness[i].item()
            if self.switch == "blend":
                raw = self._blend_value(c)
                if self.blend_ema is None:
                    self.blend_ema = raw
                else:
                    self.blend_ema = self.decay * self.blend_ema + (1.0 - self.decay) * raw
                value = self.blend_ema
                if reset_pts[i]:
                    self.blend_ema = None
            elif has_real_cuts:
                # ADR-422: settle-then-lock within a REAL scene (see module docstring).
                # While still deciding this scene's value, the PREVIOUS scene's already-
                # locked value is held completely frozen -- the tracker is deliberately
                # NOT fed the still-wobbling raw classification during this window (an
                # earlier version of this fix did, and it corrupted the tracker's own
                # running average into a blended, non-discrete number). Only once the
                # decision is final does exactly one transition (snap or ramp) happen,
                # via a single explicit mark_cut().
                if self._pending_scene_reset:
                    self._scene_running_c = None
                    self._scene_n = 0
                    self._scene_locked_value = None
                    self._pending_scene_reset = False
                if self._scene_locked_value is not None:
                    value = self.tracker.update(self._scene_locked_value)  # idempotent hold
                else:
                    # averaging the continuous score over `settle` frames, then (ADR-429)
                    # reading it against the 3-zone split in _classify() to decide once.
                    self._scene_running_c = (c if self._scene_running_c is None
                                             else self._scene_running_c
                                             + (c - self._scene_running_c) / (self._scene_n + 1))
                    self._scene_n += 1
                    if self._scene_n >= self.tracker.settle:
                        self._scene_locked_value = self._classify(self._scene_running_c)
                        self.tracker.mark_cut()  # the one clean transition into the final decision
                        value = self.tracker.update(self._scene_locked_value)
                    elif self.tracker.out is not None:
                        value = self.tracker.out  # hold the previous scene's value unchanged
                    else:
                        # very first scene of the whole movie, still deciding -- no
                        # previous value exists to hold, so use this frame's own
                        # single-frame best guess (same limitation every other auto
                        # mode already has at the very start of a video).
                        value = self._classify(c)
                if log_reset_pts[i]:
                    # iw3 convention: this is the LAST frame of its scene -- the reset
                    # takes effect starting next frame (mirrors SceneHoldTracker's own
                    # cut_pending and every other reset_pts/log_reset_pts use in iw3).
                    self._pending_scene_reset = True
                if reset_pts[i]:
                    self.tracker.mark_cut()
            else:
                # ADR-421 fallback: no real scene-boundary signal supplied at all for
                # this call -- snap off the classification itself changing instead.
                raw = self._classify(c)
                if self._last_raw is not None and raw != self._last_raw:
                    self.tracker.mark_cut()
                self._last_raw = raw
                value = self.tracker.update(raw)
                if reset_pts[i]:
                    self.tracker.mark_cut()
            results.append(value)

        v = torch.tensor(results, device=depth.device, dtype=depth.dtype).clamp(0, 1)
        return v.reshape(B, 1, 1, 1)
