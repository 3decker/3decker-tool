"""ADR-231: shared "hold steady within a scene, jump only at cuts" time-behavior for
the auto Convergence Plane modes (sod_v1, face_detect).

Real user request: sod_v1/face_detect already re-picked the screen plane every frame,
smoothed with a plain EMA decay -- but a plain EMA still chases whatever the raw
per-frame signal is doing, so it could visibly drift even within one unbroken shot
(camera pans slightly, someone shifts position) instead of holding the value a real
stereographer would pick for the whole shot. This mirrors the "hybrid" behavior
Auto 3D Strength (nt_auto3d) already uses for --divergence: settle on the scene's own
value over the first few frames after a cut, then hold it -- only gliding slowly to a
new value if the underlying signal has moved by more than a deadband, so small,
momentary wobbles never reach the picture at all.

Not tied to any particular signal (SOD closeness, face position, or anything else) --
takes one raw scalar in per frame, returns one stabilized scalar out. Both
ConvergenceEstimator and FaceConvergenceEstimator use the same instance-per-model
pattern, one SceneHoldTracker each.
"""


class SceneHoldTracker():
    def __init__(self, decay=0.9):
        self.set_decay(decay)
        self.reset()

    def set_decay(self, decay):
        # Reuses the existing "Convergence Smoothing" 0..0.95 decay value (higher =
        # smoother/slower to react, matching its own established tooltip) to derive how
        # long to settle after a cut, how fast to drift once settled, and how far the
        # signal must move before the held value follows at all.
        decay = max(0.0, min(0.95, float(decay)))
        self.decay = decay
        self.settle = max(1, round(4 + decay * 20))      # 0 -> 4 frames, 0.95 -> ~23 frames
        self.drift = 0.05 - decay * 0.035                 # 0 -> 0.05/frame, 0.95 -> ~0.0167/frame
        self.deadband = 0.015 + decay * 0.06              # 0 -> 0.015, 0.95 -> ~0.072 (depth units, 0..1)

    def reset(self):
        self.s = None            # running/settled estimate of the current scene's value
        self.n = 0                # frames seen since the last cut
        self.out = None          # the value actually being used this frame
        self.moving = False
        self.cut_pending = True  # the next frame starts a new scene

    def mark_cut(self):
        """Call after processing the LAST frame of a scene (matches how reset_pts is
        already used elsewhere in this codebase) -- the cut takes effect on the NEXT
        frame's update(), not this one."""
        self.cut_pending = True

    def update(self, raw):
        raw = float(raw)
        cut = self.cut_pending
        self.cut_pending = False

        if self.s is None or cut:
            self.s, self.n = raw, 1
        else:
            self.n += 1
            if self.n <= self.settle:
                self.s += (raw - self.s) / self.n          # running mean while settling
            else:
                self.s += (raw - self.s) * self.drift        # slow drift once settled

        if self.out is None or cut:
            self.out, self.moving = self.s, False            # a cut snaps straight to the new scene
        else:
            gap = self.s - self.out
            if not self.moving and abs(gap) >= self.deadband:
                self.moving = True
            if self.moving:
                self.out += gap * 0.1                         # glide, ~10 frames
                if abs(self.s - self.out) < 0.005:
                    self.out, self.moving = self.s, False

        return self.out
