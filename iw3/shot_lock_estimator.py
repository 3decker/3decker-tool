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

Switch styles between the two locked values (ADR-420, real user choice from the
options below -- each built so the user can compare them on real footage, not
just one guessed as "best"). ADR-435 added a 4th (`ease`), see that ADR entry.

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
  ease       ADR-435 -- same settle-then-lock DECISION as snap/snap_ramp (every
             scene still lands on exactly one of the locked values, never an
             in-between number), but transitions between scenes with a
             continuously-decaying chase (--convergence-smoothing's own decay,
             see _ease_toward()) instead of cut_smooth's fixed-length, constant-
             speed ramp. Real follow-up request: snap_ramp's ramp still read as
             a visible, mechanical "snap" even eased -- constant speed with a
             hard start and a hard stop. A decaying chase starts fast and tapers
             off the closer it gets, with no fixed end point, which is the same
             shape depth_percentile/clip_shot's own EMA smoothing already uses
             and reads as invisible by comparison. --convergence-cut-smooth has
             no effect in this mode (there's no fixed-length ramp to size).

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

ADR-435: real follow-up request -- "I want only those 2 [locked] values used
for the entire scene... what I don't want is that snappy visible change on
each scene -- just want it invisible like using other Convergence Plane
[modes], but using shot_lock." `blend` was already smooth, but it isn't a
per-scene DECISION at all -- it continuously re-interpolates every frame from
the raw closeness score, so a shot that's not decisively wide or close-up can
rest on an in-between number, not one of the 2 (or 3) locked values. The new
`ease` switch keeps the exact same locked, per-scene decision snap/snap_ramp
already make -- only HOW it moves between two different locked decisions
changes. See _ease_toward() and the module-level switch-style table above.

ADR-436: real follow-up question, after ADR-435 shipped -- "scenes 1-2 decide
0.60, scenes 3-5 decide 0.20, scene 6 decides 0.65, scene 7 decides 0.20 --
how do we make the SWITCH between them instant but invisible?" The real
bottleneck isn't the switch style at all (snap/snap_ramp/ease already cover
instant vs. gradual) -- it's that the DECISION itself doesn't land until
`self.tracker.settle` frames after the cut (the first `settle` frames of a
scene are spent averaging `closeness` before _classify() ever runs, see
`__call__` below). That settle count was always derived from
--convergence-smoothing's shared decay, so turning that smoothing UP for a
nicer `ease` glide also pushes the decision LATER -- by the time it fires,
the viewer has already settled into the new scene, so even an invisible glide
reads as a separate, delayed event rather than being masked by the cut.
--convergence-lock-decide-speed decouples the two: an independent decay-shaped
value (same 0..0.95 scale/meaning as --convergence-smoothing) that controls
ONLY how many frames the decision waits, via its own `_decide_settle` (see
_recompute_decide_settle() below) instead of `self.tracker.settle`.
--convergence-smoothing keeps controlling everything else unchanged (ease's
glide speed, snap_ramp's drift/deadband). Left at its default (None), the
decision keeps inheriting --convergence-smoothing exactly like before this
setting existed -- zero behavior change for anyone who doesn't set it.

ADR-437: real follow-up request -- "can we make convergence related to what
3D Strength is being used? e.g. any scene with 3D Strength 3.9+ uses
Convergence 0.60, and 3.8 and under uses 0.10." Direct continuation of the
earlier-confirmed real pattern (high Auto 3D Strength on a scene wants HIGH
Convergence for comfort, low Auto 3D Strength wants LOW Convergence so the
close-up can pop) -- shot_lock already has the exact "classify a scene,
lock one of two values in, transition however you like" machinery this
needs; the only thing that changes is what gets classified. `signal="clip"`
(default, unchanged) keeps reading CLIP's wide-vs-close-up visual judgment
as before. `signal="divergence"` instead reads the REAL `args.divergence`
value in effect at this exact call (apply_divergence() in utils.py) --
when Auto 3D Strength (nt_auto3d) is active, that is its real, already-
computed, non-stale per-scene strength value (nt_auto3d patches
apply_divergence() itself and sets this per sub-batch right before calling
through, so there's no ordering/staleness issue); with Auto 3D Strength off
it's just the single static --divergence number for the whole movie, so
there is nothing real to classify against (documented limitation, not a
bug). `strength_threshold` (--convergence-lock-strength-threshold) is the
single cutoff line: above it locks `wide_value` (relabeled "High Strength
Value" in the GUI for this signal), at or below locks `closeup_value`
("Low Strength Value"). Implemented as a pre-classification step that turns
the real divergence value straight into a binary 0/1 "closeness" (see
`__call__`) -- _classify()/_blend_value() need no changes at all, since a
value that's always exactly 0 or 1 already falls outside ADR-429's 1/3..2/3
middle third on its own; Mid Shot Value is simply disabled (greyed out) in
the GUI for this signal rather than specially coded around, since it could
never trigger anyway. The composed CLIP model (self._clip) is only built
when signal="clip" -- no point loading/downloading it for a user who only
ever wants the divergence signal.

ADR-438: real follow-up request, same session as ADR-437 -- "what about if
I want... anything over 3.9 uses 0.60, anything between 3.7 and 3.89 uses
0.40, and anything lower than 3.69 uses 0.20?" A real 3-way version of
ADR-437's divergence signal, mirroring ADR-429's own CLIP 3-way split
(wide/mid/close-up) but with user-set cutoff NUMBERS instead of CLIP's
fixed 1/3..2/3 zones. `strength_threshold_low` (--convergence-lock-
strength-threshold-low, None/empty by default) is the second cutoff --
ADR-437's existing `strength_threshold` is the HIGH one. With it unset
(default), behavior is byte-for-byte unchanged from ADR-437 (a plain 2-way
split: above `strength_threshold` -> wide_value, at or below ->
closeup_value). Set it, and a 3-way split turns on: above
`strength_threshold` -> wide_value; at or below `strength_threshold_low`
-> closeup_value; the band between the two -> `mid_value` (Mid Shot Value,
no longer disabled for this signal -- it simply has no effect while this
box is empty, same as the 2-way case, so there's no need to grey it out
specially). New `_closeness_for_divergence()` does this by producing the
SAME binary 0/1 "closeness" as ADR-437 for the 2-way case, or 0.5 (which
already falls inside ADR-429's own middle third) for the new band --
_classify()/_blend_value() need no further changes at all, the same way
ADR-437 needed none.

ADR-439: real follow-up, same session as ADR-438 -- "could you add a
threshold for the mid as well?" Clarified via direct back-and-forth: ADR-438's
`strength_threshold_low` only works as a cutoff LINE (the boundary where Mid
stops and Low begins) -- Mid itself never got its own reference number the
way High and Low each have. Confirmed the real ask was a THIRD way to
classify: give each of the (up to) 3 values its own single reference number
on the Auto 3D Strength scale, and pick whichever one the real per-scene
value is numerically CLOSEST to, instead of drawing cutoff lines at all.
`threshold_mode` (--convergence-lock-threshold-mode, "range" default,
unchanged ADR-437/438 behavior) switches to `"nearest"`: `strength_threshold`
(High reference), the new `strength_threshold_mid` (Mid reference, None/
empty by default), and `strength_threshold_low` (Low reference, reused from
ADR-438 but now a point, not a boundary, in this mode) are each candidates;
`_closeness_for_divergence()` picks whichever candidate is nearest to the
real divergence value and returns that candidate's own pseudo-closeness
(0.0/0.5/1.0, same values ADR-437/438 already used) -- still no changes
needed to _classify()/_blend_value(). A candidate whose reference is left
empty (Mid, or even Low) is simply never a candidate -- with only the High
reference set at all (the ADR-437 base case), there is nothing to compare
distances against, so this falls back to that same single-cutoff rule
instead of being undefined.
"""
import torch

from .clip_shot_estimator import ClipShotEstimator
from .convergence_tracker import SceneHoldTracker, settle_frames_for_decay

SWITCH_MODES = ("snap", "snap_ramp", "blend", "ease")
DEFAULT_SWITCH = "snap_ramp"
DEFAULT_WIDE_VALUE = 0.75
DEFAULT_MID_VALUE = 0.475
DEFAULT_CLOSEUP_VALUE = 0.20
# ADR-429: the two cut points splitting the continuous 0(wide)..1(close-up)
# closeness score into three equal zones -- see module docstring.
MID_LOW = 1.0 / 3.0
MID_HIGH = 2.0 / 3.0
# ADR-437: what decides wide-vs-close-up in the first place -- see module
# docstring. "clip" (default, unchanged) reads the CLIP visual judgment;
# "divergence" reads the real per-scene Auto 3D Strength value instead.
SIGNAL_MODES = ("clip", "divergence")
DEFAULT_SIGNAL = "clip"
DEFAULT_STRENGTH_THRESHOLD = 4.0
# ADR-439: "range" draws cutoff lines (ADR-437/438, unchanged default);
# "nearest" instead gives each value its own single reference number and
# picks whichever one the real divergence value is closest to.
THRESHOLD_MODES = ("range", "nearest")
DEFAULT_THRESHOLD_MODE = "range"


class ShotLockEstimator():
    def __init__(self, wide_value, closeup_value, device_id, switch=DEFAULT_SWITCH,
                 decay=0.9, cut_smooth=0, mid_value=None, decide_speed=None,
                 signal=DEFAULT_SIGNAL, strength_threshold=DEFAULT_STRENGTH_THRESHOLD,
                 strength_threshold_low=None, threshold_mode=DEFAULT_THRESHOLD_MODE,
                 strength_threshold_mid=None):
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
        # ADR-436: None (default) means "inherit decay" -- see module docstring
        # and _recompute_decide_settle().
        self.decide_speed = None if decide_speed is None else float(decide_speed)
        # ADR-437: what gets classified -- see module docstring.
        self.signal = signal if signal in SIGNAL_MODES else DEFAULT_SIGNAL
        self.strength_threshold = float(strength_threshold)
        # ADR-438: None (default) means a plain 2-way split, unchanged from
        # ADR-437 -- see module docstring.
        self.strength_threshold_low = None if strength_threshold_low is None else float(strength_threshold_low)
        # ADR-439: "range" (default) keeps ADR-437/438 behavior unchanged;
        # "nearest" reinterprets the thresholds above as reference POINTS --
        # see module docstring and _closeness_for_divergence().
        self.threshold_mode = threshold_mode if threshold_mode in THRESHOLD_MODES else DEFAULT_THRESHOLD_MODE
        self.strength_threshold_mid = None if strength_threshold_mid is None else float(strength_threshold_mid)
        # Composition, not inheritance -- see module docstring. convergence=0.5 and
        # the ema/scene_hold/cut_smooth args passed here are irrelevant: only
        # `_closeness()` is ever called on this instance, never `__call__()`.
        # ADR-437: only built for signal="clip" -- a user who only ever wants the
        # divergence signal shouldn't pay for loading/downloading a CLIP model
        # that would never actually be used.
        self._clip = (ClipShotEstimator(convergence=0.5, device_id=device_id,
                                        enable_ema=False, decay=0.9,
                                        scene_hold=False, cut_smooth=0)
                      if self.signal == "clip" else None)
        # Only meaningful for snap/snap_ramp -- `blend` uses its own plain EMA
        # below instead, since it has no cut/classification concept at all.
        self.tracker = SceneHoldTracker(decay, self.cut_smooth if self.switch == "snap_ramp" else 0)
        self.blend_ema = None
        self.ease_value = None  # ADR-435: `ease` switch only -- see _ease_toward()
        self._last_raw = None  # ADR-421 fallback path: last frame's classified value
        # ADR-422: per-real-scene settle-then-lock state (only used when the caller
        # supplies log_reset_pts). _pending_scene_reset mirrors SceneHoldTracker's own
        # cut_pending -- True means the FIRST frame seen so far starts a fresh scene
        # (also true for the very first frame of the whole movie).
        self._pending_scene_reset = True
        self._scene_running_c = None
        self._scene_n = 0
        self._scene_locked_value = None
        self._recompute_decide_settle()

    def to(self, device):
        """Mirrors ClipShotEstimator.to()/ADR-407 -- moves the composed CLIP model
        (and its fixed anchor embeddings) along with every pause/resume, same as
        every other convergence estimator that holds GPU state between frames.
        ADR-437: no-op when signal="divergence" -- there is no composed CLIP
        model to move in that case (see __init__)."""
        if self._clip is not None:
            self._clip.to(device)
        return self

    def _recompute_decide_settle(self):
        """ADR-436: how many frames __call__'s settle-then-lock decision window
        (see module docstring) waits before classifying a new scene. Normally the
        SAME count as self.tracker.settle (both derived from --convergence-
        smoothing's decay) -- decide_speed, when set, overrides it independently
        instead, so the decision can run on its own schedule."""
        effective = self.decide_speed if self.decide_speed is not None else self.decay
        self._decide_settle = settle_frames_for_decay(effective)

    def reset(self, decay=None, cut_smooth=None, decide_speed=None, **_ignored):
        if decay is not None:
            self.decay = decay
        if cut_smooth is not None:
            self.cut_smooth = int(cut_smooth or 0)
        if decide_speed is not None:
            self.decide_speed = decide_speed
        self.tracker.set_decay(self.decay)
        self.tracker.set_cut_smooth(self.cut_smooth if self.switch == "snap_ramp" else 0)
        self.tracker.reset()
        self._recompute_decide_settle()
        self.blend_ema = None
        self.ease_value = None
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

    def _ease_toward(self, target):
        """ADR-435: `ease` switch only -- continuously-decaying chase toward the
        currently locked `target`, reusing --convergence-smoothing's own decay
        (same shape as `blend`'s own EMA above, just chasing the discrete LOCKED
        per-scene value instead of a continuously re-interpolated one). No fixed
        end point and no hard stop, unlike cut_smooth's linear ramp -- moves fast
        while far from target and tapers off the closer it gets, which is what
        reads as smooth/invisible rather than mechanical."""
        if self.ease_value is None:
            self.ease_value = target
        else:
            self.ease_value += (target - self.ease_value) * (1.0 - self.decay)
        return self.ease_value

    def _advance_locked(self, locked_value):
        """ADR-435: the one step snap/snap_ramp/ease actually differ on -- how the
        output value moves toward a just-decided/already-locked `locked_value`.
        snap/snap_ramp use SceneHoldTracker.update() (instant jump, or a fixed-
        length linear ramp); `ease` uses the continuously-decaying chase instead
        (see _ease_toward())."""
        if self.switch == "ease":
            return self._ease_toward(locked_value)
        return self.tracker.update(locked_value)

    def _closeness_for_divergence(self, divergence):
        """ADR-437/438/439: reduces the real divergence value to the same 0..1
        "closeness" _classify()/_blend_value() already expect -- see module
        docstring.

        threshold_mode="range" (default, ADR-437/438 unchanged): above
        strength_threshold -> 0.0 (wide_value, "High Strength Value"). With
        strength_threshold_low unset (ADR-437, a plain 2-way split): anything
        else -> 1.0 (closeup_value, "Low Strength Value"). With it set
        (ADR-438, 3-way): at or below it -> 1.0; the band between the two
        thresholds -> 0.5, which already falls inside ADR-429's own middle
        third and picks mid_value, same as CLIP's own middle zone.

        threshold_mode="nearest" (ADR-439): strength_threshold/
        strength_threshold_mid/strength_threshold_low are each a candidate
        REFERENCE POINT (mid/low only when actually set) instead of a cutoff
        line -- picks whichever candidate's reference is numerically closest
        to the real divergence value and returns THAT candidate's own pseudo-
        closeness. With only the High reference available (nothing else set),
        falls back to the same single-cutoff rule "range" mode uses for that
        same base case, since there's nothing to compare distances against."""
        if divergence is None:
            return 1.0
        divergence = float(divergence)
        if self.threshold_mode == "nearest":
            candidates = [(self.strength_threshold, 0.0)]
            if self.strength_threshold_mid is not None:
                candidates.append((self.strength_threshold_mid, 0.5))
            if self.strength_threshold_low is not None:
                candidates.append((self.strength_threshold_low, 1.0))
            if len(candidates) == 1:
                return 0.0 if divergence > self.strength_threshold else 1.0
            return min(candidates, key=lambda pair: abs(divergence - pair[0]))[1]
        if divergence > self.strength_threshold:
            return 0.0
        if self.strength_threshold_low is not None and divergence <= self.strength_threshold_low:
            return 1.0
        if self.strength_threshold_low is not None:
            return 0.5
        return 1.0

    def __call__(self, rgb, depth, reset_pts=None, log_reset_pts=None, divergence=None):
        B = depth.shape[0]
        reset_pts = reset_pts if reset_pts is not None else [False] * B
        # ADR-422: log_reset_pts (when supplied) is the real, independently-scanned
        # scene-boundary signal -- see module docstring. `has_real_cuts` must be
        # checked BEFORE defaulting the list below, so "never supplied at all" (use
        # the ADR-421 classification-flip fallback) stays distinguishable from
        # "supplied, but no cut on this particular frame."
        has_real_cuts = log_reset_pts is not None
        log_reset_pts = log_reset_pts if has_real_cuts else [False] * B
        if self.signal == "divergence":
            # ADR-437/438: the real Auto 3D Strength value in effect for this exact
            # call (or the single static --divergence number when that add-on is
            # off -- see module docstring), reduced via _closeness_for_divergence()
            # -- no CLIP call at all for this signal.
            c_val = self._closeness_for_divergence(divergence)
            closeness = torch.full((B,), c_val, device=depth.device, dtype=depth.dtype)
        else:
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
                # ADR-435: `ease` makes exactly the same locked decision as snap/
                # snap_ramp below -- only the FINAL step (how the output value moves
                # toward that decision) differs, so it shares this whole decision
                # block rather than duplicating it. See _advance_locked().
                if self._scene_locked_value is not None:
                    value = self._advance_locked(self._scene_locked_value)  # idempotent hold
                else:
                    # averaging the continuous score over `settle` frames, then (ADR-429)
                    # reading it against the 3-zone split in _classify() to decide once.
                    self._scene_running_c = (c if self._scene_running_c is None
                                             else self._scene_running_c
                                             + (c - self._scene_running_c) / (self._scene_n + 1))
                    self._scene_n += 1
                    if self._scene_n >= self._decide_settle:
                        self._scene_locked_value = self._classify(self._scene_running_c)
                        if self.switch != "ease":
                            self.tracker.mark_cut()  # the one clean transition into the decision
                        value = self._advance_locked(self._scene_locked_value)
                    else:
                        held = self.ease_value if self.switch == "ease" else self.tracker.out
                        if held is not None:
                            value = held  # hold the previous scene's value unchanged
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
                if reset_pts[i] and self.switch != "ease":
                    self.tracker.mark_cut()
            else:
                # ADR-421 fallback: no real scene-boundary signal supplied at all for
                # this call -- snap off the classification itself changing instead.
                raw = self._classify(c)
                if self.switch == "ease":
                    # No discrete snap/ramp state to reset here -- a classification
                    # flip is just a new target the continuous chase moves toward.
                    value = self._ease_toward(raw)
                else:
                    if self._last_raw is not None and raw != self._last_raw:
                        self.tracker.mark_cut()
                    self._last_raw = raw
                    value = self.tracker.update(raw)
                    if reset_pts[i]:
                        self.tracker.mark_cut()
            results.append(value)

        v = torch.tensor(results, device=depth.device, dtype=depth.dtype).clamp(0, 1)
        return v.reshape(B, 1, 1, 1)
