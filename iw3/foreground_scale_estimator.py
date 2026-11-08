"""ADR-452: `shot_lock` for `--foreground-scale`, independent of Convergence Plane's own shot_lock.

Real user request: replicate Convergence Plane's `shot_lock` mode (ADR-420+) --
per-shot Close-up/Mid/Wide classification driven by the real per-scene Auto 3D
Strength divergence value (ADR-437/438) -- for a completely different setting,
`--foreground-scale` (the depth-to-disparity mapping curve, see `.mapper`'s
`resolve_mapper_name()`/`get_mapper()`). The user explicitly wants to run both
features at once, each with its OWN High/Low threshold boxes and its own Wide/
Mid/Close-up values -- this module's state is never shared with
`ShotLockEstimator`/Convergence Plane's shot_lock.

Deliberately NOT a reuse of `ShotLockEstimator`: that class carries a whole
settle-then-lock state machine (N-frame averaging before deciding, a pending-
scene-reset flag, snap/ramp/blend/ease smoothing) because it ALSO supports a
noisy CLIP-similarity classification mode (`signal="clip"`) -- a shot's visual
framing only reveals itself gradually, frame to frame, so the real decision has
to be smoothed/settled before it's trusted. This module only ever needs
`signal="divergence"` + `threshold_mode="range"`. For that combination, the
divergence value fed in on any given call is not a noisy per-frame estimate --
it is already the REAL, non-stale, exact-for-this-call Auto 3D Strength value:
`nt_auto3d/nt_autostrength/autodiv.py`'s `apply_divergence()` wrapper splits
every batch at each divergence-value change point (`while j < n and
divs[j] == divs[i]`) and sets `a.divergence = value` fresh on a copied `args`
object before ever delegating to iw3 core's own `apply_divergence()` -- so
there is no staleness window and nothing to average or settle. Classifying it
is therefore a pure, stateless function of the one number in effect for this
exact call: no averaging buffer, no pending-scene-reset flag, no switch-style
smoothing tracker. If a future caller ever needs the noisy `clip` signal for
this setting too, that is a real reason to revisit this scope -- not a reason
to build it preemptively here.

ADR-455: real follow-up request -- "can Foreground Scale Shot Lock glide
smoothly into a new value too, the way Convergence Plane's own shot_lock
`ease` switch already does?" The classification itself stays exactly what
the paragraph above describes (a pure, stateless function of one number, no
settle/averaging needed) -- `_classify()`/`classify_value()` are untouched.
What changes is only the FINAL step: `switch` (default "snap", 100% backward
compatible -- see `__call__()`) adds an opt-in "ease" mode that mirrors
`ShotLockEstimator._ease_toward()` exactly (same formula, same
`--convergence-smoothing`-style `decay` convention: higher = slower/
smoother, lower = snappier) -- the mapper is resolved from a continuously-
decaying chase toward the classified target instead of snapping straight to
it. Deliberately NOT importing Convergence's `snap_ramp`/`blend` switch
styles too: those exist to solve CLIP visual-judgment noise settling before
a decision is trusted (see ShotLockEstimator's own module docstring), a
problem this module never has in the first place (its divergence value is
already exact/non-stale for every call) -- there is nothing for them to fix
here. With `switch="ease"`, `_mapper_fn_for_cache` is deliberately left
empty/unused: the eased value is a continuously-varying float, so caching by
value would only ever grow the dict by one new, never-reused entry per call.
"""
from .mapper import resolve_mapper_name, get_mapper

SWITCH_MODES = ("snap", "ease")
DEFAULT_SWITCH = "snap"
DEFAULT_DECAY = 0.9


class ForegroundScaleEstimator():
    def __init__(self, wide_value, closeup_value, threshold_high, mid_value=None,
                 threshold_low=None, metric_depth=False, mapper_type=None,
                 switch=DEFAULT_SWITCH, decay=DEFAULT_DECAY):
        self.wide_value = float(wide_value)
        self.closeup_value = float(closeup_value)
        self.threshold_high = float(threshold_high)
        # None means "disabled" -- a plain 2-way wide/close-up split, same
        # convention as ShotLockEstimator's own mid_value/strength_threshold_low.
        self.mid_value = None if mid_value is None else float(mid_value)
        self.threshold_low = None if threshold_low is None else float(threshold_low)
        self.metric_depth = metric_depth
        self.mapper_type = mapper_type
        self._mapper_fn_for_cache = {}
        # ADR-455: "snap" (default) is 100% unchanged, cached-mapper behavior --
        # see __call__(). "ease" glides toward the classified target instead of
        # snapping to it, same decay convention as ShotLockEstimator's own ease
        # switch (see _ease_toward()).
        self.switch = switch if switch in SWITCH_MODES else DEFAULT_SWITCH
        self.decay = float(decay)
        self.ease_value = None

    def _classify(self, divergence):
        """Same "range" rule as `ShotLockEstimator._closeness_for_divergence()` +
        `_classify()` combined (see those for the full rationale) -- no closeness
        indirection needed here since this module never supports the `clip`/
        `nearest` modes those two methods exist to also cover."""
        if divergence is None:
            return self.closeup_value
        divergence = float(divergence)
        if divergence > self.threshold_high:
            return self.wide_value
        if self.threshold_low is not None and divergence <= self.threshold_low:
            return self.closeup_value
        if self.threshold_low is not None:
            return self.mid_value if self.mid_value is not None else self.closeup_value
        return self.closeup_value

    def classify_value(self, divergence):
        """ADR-453: public wrapper for `_classify()` -- the Foreground Scale scene
        log (--foreground-scale-scene-log, see iw3/utils.py's
        `_ForegroundScaleSceneLogState`) needs the classified scalar value itself
        (which of wide_value/mid_value/closeup_value this call locked in), not
        just the mapper_fn `__call__()` maps it to. `_classify()` itself is left
        untouched -- this is a one-line pass-through, not a rename."""
        return self._classify(divergence)

    def _mapper_fn_for(self, value):
        fn = self._mapper_fn_for_cache.get(value)
        if fn is None:
            name = resolve_mapper_name(mapper=None, foreground_scale=value,
                                       metric_depth=self.metric_depth,
                                       mapper_type=self.mapper_type)
            fn = get_mapper(name)
            self._mapper_fn_for_cache[value] = fn
        return fn

    def _ease_toward(self, target):
        """ADR-455: `switch="ease"` only -- exact mirror of
        ShotLockEstimator._ease_toward() (see that method's own docstring for the
        full rationale): a continuously-decaying chase toward the just-classified
        `target`, reusing the same `decay` convention (higher = slower/smoother)."""
        if self.ease_value is None:
            self.ease_value = target
        else:
            self.ease_value += (target - self.ease_value) * (1.0 - self.decay)
        return self.ease_value

    def __call__(self, divergence=None):
        target = self._classify(divergence)
        if self.switch == "ease":
            # ADR-455: resolved fresh every call, never through
            # _mapper_fn_for_cache -- see module docstring.
            value = self._ease_toward(target)
            name = resolve_mapper_name(mapper=None, foreground_scale=value,
                                       metric_depth=self.metric_depth,
                                       mapper_type=self.mapper_type)
            return get_mapper(name)
        return self._mapper_fn_for(target)
