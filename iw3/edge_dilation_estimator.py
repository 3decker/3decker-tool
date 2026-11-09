"""ADR-458: `shot_lock` for `--edge-dilation` ("Edge Fix" in the GUI), independent
of Convergence Plane's (ADR-420+) and Foreground Scale's (ADR-452) own shot_lock
features.

Real user request, directly motivated by Edge Fix's own existing GUI tooltip
(gui.py's cbo_edge_dilation/cbo_edge_dilation_y controls), which already says
Divergence ~2.0-2.25 pairs with Edge Fix 2/1, ~3.0-3.5 pairs with 3/2 -- i.e. Edge
Fix is supposed to track Divergence, and Divergence already auto-varies per shot
via Auto 3D Strength (nt_auto3d). Running one fixed Edge Fix pair for the whole
movie was therefore a real, already-documented mismatch, not a cosmetic ask. This
closes that gap the same way Convergence Plane/Foreground Scale already closed it
for their own settings: classify the real per-scene divergence value into one of
Wide/Mid/Close-up, each with its own fixed Edge Fix [x, y] pair.

Deliberately SNAP ONLY -- no `switch`/"ease" parameter at all, unlike
ForegroundScaleEstimator (ADR-455). Edge Fix's value is an integer dilation LOOP
COUNT (see dilation.py's dilate_edge()/edge_dilation_parse()), not a continuous
float like Convergence Plane or Foreground Scale -- there is no fractional
"1.4 loops" to glide through on the way from, say, 1/1 to 3/2. This is an
accepted, understood tradeoff with the user, not something to redesign around.

Deliberately `signal="divergence"` ONLY -- no CLIP support, and therefore no
settle-then-lock averaging state either. ForegroundScaleEstimator's own docstring
explains why that combination needs no averaging buffer/pending-scene-reset flag:
when the only signal is the real per-call Auto 3D Strength divergence value
(never a noisy frame-to-frame CLIP judgement that has to settle before it's
trusted), classifying it is a pure, stateless function of the one number in
effect for this call. The same reasoning applies here unchanged -- see
`_classify()`, which mirrors `ForegroundScaleEstimator._classify()` exactly.

ONE REAL DIFFERENCE FROM CONVERGENCE/FOREGROUND SCALE'S OWN DIVERGENCE SIGNAL --
read this before changing any call site: Convergence Plane's and Foreground
Scale's shot_lock both run from INSIDE `apply_divergence()` in iw3/utils.py, so
`args.divergence` is already the real, non-stale value Auto 3D Strength's own
`apply_divergence()` wrapper (nt_auto3d/nt_autostrength/autodiv.py) set on a
copied args object for THIS exact call, before ever delegating through to iw3
core. Edge Fix's dilation instead runs INSIDE `depth_model.infer()`
(dilation.py's `dilate_edge()`), which iw3/utils.py always calls BEFORE
`apply_divergence()` for that same frame/batch -- there is no real, non-stale
divergence value available yet at that point, and iw3/utils.py deliberately has
no hard dependency on nt_auto3d (see the comment near the top of that file) to
reach into its internals and get one early.

The fix: `apply_divergence()` unconditionally persists the real divergence value
it was just called with into a generic, add-on-agnostic `args.state` key,
`"_last_applied_divergence"`, before doing anything else. Every live
`depth_model.infer()`/`infer_with_normalize()`/`flush_with_normalize()` call site
then resolves Edge Fix's value from THAT (one call behind) instead of the
in-flight `args.divergence`. In practice this means Edge Fix's decision for a
given scene is exactly one frame/batch behind Convergence Plane's/Foreground
Scale's decision for the SAME scene cut -- i.e. it only matters for the single
frame/batch right at a cut, which is already visually disrupted by the cut
itself. This is a deliberate, documented tradeoff, not a bug -- see
`_resolve_edge_dilation()` in iw3/utils.py for the actual read side.
"""
from .dilation import edge_dilation_parse

DEFAULT_STRENGTH_THRESHOLD = 4.0


class EdgeDilationEstimator():
    def __init__(self, wide_value, closeup_value, threshold_high, mid_value=None,
                 threshold_low=None):
        # Each of wide/mid/closeup is an --edge-dilation-style pair ([x, y], a
        # single int, or already-None for mid) -- normalized once here via
        # dilation.py's own edge_dilation_parse() instead of reinventing that
        # normalization.
        self.wide_value = edge_dilation_parse(wide_value)
        self.closeup_value = edge_dilation_parse(closeup_value)
        self.threshold_high = float(threshold_high)
        # None means "disabled" -- a plain 2-way wide/close-up split, same
        # convention as ForegroundScaleEstimator's/ShotLockEstimator's own
        # mid_value/strength_threshold_low.
        self.mid_value = None if mid_value is None else edge_dilation_parse(mid_value)
        self.threshold_low = None if threshold_low is None else float(threshold_low)

    def _classify(self, divergence):
        """Same "range" rule as `ForegroundScaleEstimator._classify()` (see that
        method and this module's own docstring for the full rationale) -- no
        closeness indirection needed, since this module never supports the
        `clip`/`nearest` modes ShotLockEstimator exists to also cover."""
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

    def __call__(self, divergence=None):
        return self._classify(divergence)
