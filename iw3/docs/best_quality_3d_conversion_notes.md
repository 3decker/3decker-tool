# iw3 Depth Models & Best-Quality 3D Conversion Notes

Personal research notes compiled from a conversation about getting the best possible
3D conversion quality out of iw3 — depth model tradeoffs, all quality-relevant settings,
real-world testing findings, and how to get as close as possible to a professional
(hand-done) Blu-ray 3D conversion like Titanic 3D or Avatar.

Not official project documentation — see `docs/gui_settings_reference.md` for the
maintained settings reference this builds on.

Hardware on record: RTX 5090, 32GB VRAM — render speed is not a constraint, only quality.

---

## 1. Depth Model Families — Pros & Cons

A "depth model" is the AI that looks at each frame and figures out what's near vs far.
That map is what drives the whole 3D effect.

| Family | Examples | Best for | Notes |
|---|---|---|---|
| **VDA_\*** (Video Depth Anything) | `VDA_S/B/L`, `VDA_Metric_*`, `VDA_Stream_*` | **Video** | Built specifically to keep depth stable/flicker-free across frames. `_Stream_` = newer low-latency streaming variants. `_Metric_` = trained on real-world-distance data. |
| **Any_V2_\*** | `Any_V2_S/B/L`, `_N_*` (indoor), `_K_*` (outdoor) | Single images | Depth Anything V2 — sharp on stills, can flicker on video (no frame-to-frame memory). |
| **Any_V3_Mono\*** | `Any_V3_Mono`, `Any_V3_Mono_01` | Single images | Newest generation, generally the sharpest for stills. No flicker-resistant video variant exists yet. |
| **Distill_Any_\*** | `Distill_Any_S/B/L` | Single images | Distillation-based, separate project, similar role to Depth Anything V2. **Code-verified 2026-08-14: Depth Anti-aliasing does NOT support this family** (`depth_anything_model.py`'s `AA_SUPPORTED_MODELS` only lists `Any_V2_S/B/L`, excludes `Distill_Any_*`) — same "silently inert" category as ZoeDepth/DepthPro. Also has zero built-in frame-to-frame memory (same `decay=0, buffer_size=1` default as `Any_V3_Mono*`), so same video-instability risk applies if used on a whole movie. No direct A/B against `Any_V2_L`/`Any_V3_Mono` has been done in this project yet — least-tested of the single-image families here. |
| **ZoeD_\*** / **DepthPro** | `ZoeD_N/K/NK`, `ZoeD_Any_N/K`, `DepthPro`, `DepthPro_S` | Older / specialized | Earlier generation, generally superseded, but `ZoeD_Any_N` is specifically called out by the project author as looking best for general 3D scenes. DepthPro is image-only, fixed resolution (1536² or 1024²), distance clipped at 40m. |

**Rule of thumb:** for video, use a `VDA_*` model (flicker resistance matters more than
raw per-frame sharpness). For a single photo, `Any_V3_Mono` is typically the sharpest.

Project author's own stated preference (from README): `ZoeD_Any_N`, `Any_B`, or `VDA_Metric`.

### VDA_L vs VDA_Metric_L — which is better?

Same "Large" backbone, so **identical speed and VRAM cost** — pure quality question.

Important technical finding from the code: even though `VDA_Metric_L` is trained to
predict real-world distances, iw3 has a hardcoded setting (`force_disparity = True`)
that always converts its output back into a plain near/far (disparity) map before use.
So in practice you do **not** get the "consistent absolute scale across scene cuts"
benefit you'd expect from a true metric model — both `VDA_L` and `VDA_Metric_L` end up
feeding the same kind of map into the 3D conversion. The only real difference left is
which training data produced better-looking results.

**Recommendation: `VDA_Metric_L`.** No speed/memory cost difference, and the project
author's documentation expresses a preference for the Metric-trained checkpoints.

### `Any_V3_Mono`/`Any_V3_Mono_01` — "flat cardboard" on extreme close-ups (2026-08-14)

Reported symptom: something very close to the camera (fills a large part of the
frame) pops out correctly as a shape, but shows **no internal depth detail** — reads
as a flat cutout rather than a rounded object.

Code-verified cause (`depth_anything_v3_model.py`, `_forward()`): before the raw
depth is used, it's converted with `depth = 1.0 / (depth + shift)`, `shift = 0.2`
— a hardcoded constant (there's even a code comment: `# TODO: This value should
ideally be adjustable via the foreground scale option, but currently it is not
possible`). This formula saturates for very-close objects: past a point, further
real-world distance differences all round to nearly the same output value, so any
internal depth variation within a close, frame-filling object (face contours,
finger curl, folds) is destroyed *before* Divergence/Foreground Scale/Foreground
Pop ever see the depth map. **None of those downstream settings can fix this** —
the information is already gone by the time they run. Foreground Pop specifically
would likely make it *worse*, since it pushes whatever's already closest further
forward with no internal shape to differentiate.

**Mitigation (model swap for the specific shot, not a setting):** `VDA_L`/
`VDA_Metric_L` use the same style of formula but with a smaller floor (`eps = 0.1`
vs. `0.2`, confirmed in `video_depth_anything_model.py`), so the same saturation
happens later — an object has to be noticeably closer before losing internal
detail. Tradeoff: this reintroduces the open "warped fine-detail" risk (see the
crown/fingers investigation below) on shots with thin/spiky detail instead.
Practical approach: switch to `VDA_L`/`VDA_Metric_L` specifically for shots with
something large and close to the lens (face/hand close-ups); keep
`Any_V3_Mono_01` for normal mid/far framing.

### `Any_V3_Mono` vs `Any_V3_Mono_01` — same checkpoint, different scaling only

Code-verified (`iw3/depth_anything_v3_model.py`, 2026-08-14 check): these two are
**not different trained models** — both load the exact same `da3mono-large.safetensors`
weights file, run through identical inference code. The *only* difference is the
depth-scaler mode applied after inference: `Any_V3_Mono` uses Max=1-only scaling
(`mode="max"`), `Any_V3_Mono_01` uses full Max=1-*and*-Min=0 scaling (`mode="minmax"`)
— i.e. `_01` forces the depth range to fill the full 0–1 span every frame,
`Any_V3_Mono` only pins the far end. **Both use `decay=0, buffer_size=1`** for that
scaler — confirming neither variant has any built-in frame-to-frame smoothing of its
own, consistent with the "depth breathing" finding in the crown-artifact
investigation below (Section 3): whichever one is used, temporal stability has to
come entirely from EMA Decay/Buffer, not from the model itself.
Best way to confirm for your own content: use the GUI's **Compare Presets** feature to
render the same short/complex clip with both before committing to a full movie.

**Important:** the depth model choice does **not** control how much 3D separation/pop
you perceive — that's entirely controlled by **Divergence**, since both models' output
gets normalized (EMA min/max) and then scaled by Divergence before it becomes the 3D
effect. The model choice affects *accuracy* of what's near/far, not the *intensity*.

**Real-world finding — why `VDA_L` can subjectively feel *more* immersive anyway:**
this is not necessarily placebo. Relative-depth models (`VDA_L`) aren't trained to be
physically accurate — they're rewarded for producing a depth map that uses good
contrast to make near/far ordering clear, with no penalty for exaggerating differences.
Metric-trained models (`VDA_Metric_L`) are anchored to real physical distance, so in
scenes with genuinely small real depth variation (faces, interiors, dialogue shots)
they can look comparatively flatter, while a relative model may exaggerate that same
variation into more dramatic-looking "roundness" — more visual pop, less strict
accuracy. This mirrors a similar tradeoff the project's own README notes elsewhere
(metric models = more accurate segmentation but flatter-looking foreground). If you
consistently prefer `VDA_L` blind (test with unlabeled output files to rule out
expectation bias), that's a legitimate choice, not a wrong one.

### VDA_Metric_L vs VDA_Metric_Stream_L

Same checkpoint weights either way — `VDA_Stream_*` reuses the exact same trained
weights as `VDA_*`, just run through a different inference architecture:

- **Regular (`VDA_Metric_L`)**: processes video in overlapping *windows*, aligning
  each chunk against neighboring frames (can effectively look both backward and
  forward within that window). Better raw temporal consistency, needs more memory.
- **Streaming (`VDA_Stream_Metric_L`)**: processes strictly frame-by-frame, causally,
  never looking ahead — built for real-time/live use cases (e.g. iw3-desktop capturing
  a live screen), trading consistency for constant low memory and low latency.

**For converting a movie file you already have in full (not live), use the regular
`VDA_Metric_L`.** The streaming variant's benefit (constant memory, low latency) buys
nothing for an offline batch job, and with 32GB VRAM there's no memory pressure to
justify the consistency trade-off. Only use `_Stream_` for real-time features or if you
hit an out-of-memory error on the regular version.

---

## 2. EMA Decay Rate & EMA Buffer (Flicker Reduction)

These control how much the **depth map itself** is smoothed frame-to-frame, to stop
the 3D "breathing"/flickering that AI depth models are prone to. This is a completely
separate system from **Convergence Smoothing** (see below) — don't confuse the two.

| Setting | What it does | Higher value | Lower value |
|---|---|---|---|
| **EMA Decay Rate** | How much the depth map is smoothed frame to frame | Smoother but slower to react | Reacts faster, may flicker more |
| **EMA Buffer** (Lookahead Buffer Size) | How many frames are looked at together to judge the near/far depth range | More stable but slower to adapt *within* a shot | Faster to adapt, less stable |

**Genre-based recommendations** (pair these with Scene Boundary Detection so smoothing
resets cleanly at real cuts instead of bleeding across unrelated scenes):

| | Slow/contemplative (e.g. *Interstellar*) | Moderate/comedic (e.g. *Joe's Apartment*) | Fast-cut action (e.g. *Mortal Kombat II*) |
|---|---|---|---|
| EMA Decay Rate | `0.95` | `0.85` | `0.75` |
| EMA Buffer | `60` | `30` (default) | `15` |
| Convergence Smoothing | `0.75` | `0.6` | `0.5` |

**Overall best general-purpose default (if not tuning per-film): `0.85` / `30`.**
This is the tool's actual shipped default — a reasonable starting guess before you have
real per-film data (see below for how to get real data instead of guessing).

### Two separate EMA systems — don't confuse them

| Setting | What it smooths | Depends on Convergence Mode? |
|---|---|---|
| **EMA Decay Rate / EMA Buffer** (Flicker Reduction) | The **depth map itself** — near/far range stability, upstream of convergence | **No** — works identically in `constant` or `sod_v1` mode |
| **Convergence Smoothing** | The **auto-convergence point** (only exists in `sod_v1`/`face_detect`) | **Yes** — becomes a complete no-op in `constant` mode |

### What Convergence Smoothing actually does

Only matters with the auto convergence modes (`sod_v1` / `face_detect`). There's a
"screen depth" point — whatever sits at that depth appears to be exactly at the
screen surface; closer things pop out, farther things sink back. In auto mode the AI
re-picks that point based on the detected subject every frame, and without smoothing
its small frame-to-frame wobbles (a head turn, a new face appearing) would make the
whole image visibly jerk in and out. Smoothing blends the point gradually instead of
snapping to it. Trade-off: higher = smoother but laggier reaction to real changes;
lower = reacts fast but can jitter in busy scenes.

### Getting real per-film data instead of guessing genre

iw3's **Scene Boundary Detection** (`--scene-detect-only`, with `--depth-model NULL
--method NULL` to skip loading any real model) can be run standalone on a source file
to produce an actual list of every detected cut, cached as JSON at
`nunif/tmp/iw3_scene_cache/<hash>.json` (a `"pts"` list of frame numbers, plus the
`max_fps` used). From that, real Average Shot Length (mean and median) can be
calculated directly — far better than a genre guess.

**Measured results (both ~96–98 min UHD BluRay remuxes, verified against the actual
file duration via ffprobe):**

| Film | Runtime | Shots | Mean shot length | Median shot length | Longest shot |
|---|---|---|---|---|---|
| Hocus Pocus (1993) | 96.13 min | 1,416 | 4.07 sec | 2.38 sec | 170.55 sec |
| Super Mario Galaxy Movie (2026) | 98.25 min | 1,416 | 4.16 sec | 2.67 sec | 277.90 sec |

(Correction, 2026-08-13: Mario Galaxy's longest shot was originally recorded as
`248.0 sec` — re-verified against exact `ffprobe` runtime (5894.889 sec) and the
regenerated scene cache, the correct figure is **277.90 sec**. The earlier number
undercounted because it (and the shot-count/mean figures above) previously excluded
the very first and very last shot of the film — Scene Boundary Detection's cache only
stores the cut *points*, not the video's start/end, so the head shot (video start →
first cut) and tail shot (last cut → video end) have to be added back in manually
using the real ffprobe duration. Both films' tables above now include them.)

**Key lesson:** genre assumptions can be wrong — Mario Galaxy was guessed as
"fast-cut action" pacing beforehand, but its real measured cut rate turned out to be
essentially the same as (if not slightly slower than) the comedy Hocus Pocus. Trust
measured data over genre guesses when it's available.

**Data-driven buffer sizing logic:** since Scene Boundary Detection resets the buffer
at every real cut, the buffer should be sized off the **median** (not mean) shot
length — target roughly 30–35% of the median so it reliably completes within the
majority of shots instead of being oversized and only useful in rare long outlier
shots. This gave:

| Film | Data-driven EMA Decay | Data-driven EMA Buffer |
|---|---|---|
| Hocus Pocus | `0.80` | `20` |
| Super Mario Galaxy Movie | `0.80` | `22` |

### Shot-length terciles (short / medium / long) and matching EMA settings

Both films' shots were split into three equal-sized groups (terciles) by length, to
see whether the single whole-movie EMA pair above is really representative, and to
have a starting point if a specific unusually-short-cut or unusually-long/lingering
scene ever needs its own re-render via the export/config workflow. Methodology: same
TransNetV2-based Scene Boundary Detection cache, with the head shot (video start →
first detected cut) and tail shot (last detected cut → video end) added back in using
the real `ffprobe` runtime, converted to seconds at each film's native `23.976 fps`
(`24000/1001`).

| Film | Short median | Medium median | Long median |
|---|---|---|---|
| Hocus Pocus | 1.21 sec (range 0.04–1.71s, n=472) | 2.38 sec (range 1.71–3.59s, n=472) | 6.17 sec (range 3.59–170.55s, n=472) |
| Super Mario Galaxy Movie | 1.46 sec (range 0.04–2.04s, n=472) | 2.67 sec (range 2.04–3.75s, n=472) | 6.05 sec (range 3.75–277.90s, n=472) |

Each film's Medium tercile median matches its documented whole-movie median exactly,
as expected (it's the literal middle third). The two films track each other closely
across all three buckets, reinforcing the "measured pacing is nearly identical
despite different genres" finding above.

Applying the same 30–35%-of-median buffer sizing rule to each bucket's median gives:

| Bucket | EMA Buffer | EMA Decay |
|---|---|---|
| Short (~1.2–1.5s median) | ~9–12 | ~0.70–0.73 |
| Medium (~2.4–2.7s median) | `20`–`22` (already the documented per-film values) | `0.80` |
| Long (~6.0–6.2s median) | ~44–52 | ~0.90–0.92 |

**Important caveat — this is not a per-shot auto-switching feature.** EMA Decay/
Buffer is set once for the entire render; Scene Boundary Detection only *resets* the
running buffer at each cut, it doesn't change the Decay/Buffer numbers themselves
mid-movie. The Medium row (`20`/`0.80` or `22`/`0.80`) is what to actually use for a
whole-movie render, since it's centered on the film's true median. The Short/Long
rows are only relevant if re-rendering an individual scene later (e.g. via the
export/config workflow already used for the crown-artifact testing) whose pacing is
clearly unrepresentative of the movie as a whole.

### Fixing smearing/bleeding with a much larger buffer

Real-world finding: occasional smearing/bleeding-looking artifacts in some scenes
(not often) were fixed by raising **EMA Buffer to `120`**. Mechanism: a small buffer
computes the near/far normalization range from just a handful of frames, so one brief
outlier frame (something momentarily very close to camera, a flash, fast object
entering/exiting frame) can yank the whole depth-map scale around. That per-frame
scale "breathing" changes how much 3D separation gets applied moment to moment, which
reads visually as smearing/swimming/bleeding even though no pixel is literally
smeared — it's depth-*scale* instability, not a warp/pixel artifact. A much larger
buffer (`120`) dilutes any single outlier frame's influence across many more frames,
keeping the scale far more stable.

**Paired EMA Decay for Buffer 120: `0.98`** (continuing the same interpolation
pattern: `15`→`0.75`, `30`→`0.85`, `60`→`0.95`, extrapolated further). Pair a large
buffer with a high decay deliberately — both should reinforce the same "resist
momentary spikes" goal. Caveat: at `0.98` the depth scale becomes slow to react to
*genuine* large intentional depth changes too (e.g. a push/pull reveal shot) — if that
specific kind of shot starts looking laggy, drop to `0.97` or `0.95` for it specifically.

**Extrapolated buffer/decay pairs beyond the documented range** (untested by the
project itself, follow the same trend):

| EMA Buffer | Suggested EMA Decay |
|---|---|
| `100` | `~0.97` |
| `120` | `~0.98` |

Practical note: a buffer this large (120 frames ≈ 4–5 sec) only fully engages in a
film's longer continuous shots — most shots (median ~2.4–2.7 sec for both films
tested) will hit a cut and reset before the buffer fully fills, which is fine; it's
still providing whatever stability is possible within each shot's actual length.

**Real-world finding (2026-09-12, Hocus Pocus testing): a high Decay does NOT
substitute for an adequately large Buffer.** Even at Decay `0.99` (above the
documented `0.98` ceiling above), a too-small Buffer still produces bad flickering
— confirmed via direct A/B testing on a difficult scene (heavy foliage over a
cemetery gate, prone to depth-scale instability). Buffer `650` (well beyond the
`120` extrapolation above) was needed alongside Decay `0.99` to substantially calm
the flickering on that specific scene; Decay `0.99` alone, paired with a small
buffer, was not enough by itself. Confirms Decay and Buffer are NOT
interchangeable levers for the same problem: Decay controls how much any single
new frame can move the scale, but Buffer controls how many frames the
normalization range is computed from in the first place — a small buffer stays
noisy/reactive regardless of how high Decay is set, because there simply isn't
enough history to average over. For scenes that keep flickering despite a high
Decay, raise Buffer significantly before assuming Decay needs to go even higher.

### How to visibly tell if EMA Decay/Buffer is too low or too high

EMA Decay/Buffer sets the depth map's overall near/far *scale* for the current
moment. When it's wrong, the *amount* of 3D pop/depth applied to the whole frame
shifts — not any single object, the whole image's depth intensity.

**Too low (reacts too fast, unstable) — what to look for:**
- The whole image's sense of depth subtly "breathing"/pulsing, like someone is
  gently nudging a depth-intensity dial up and down, even when nothing in the shot
  is actually moving closer or farther.
- **Best place to catch it:** a static dialogue shot (someone sitting still,
  talking). Real depth here should be perfectly constant frame to frame — any
  visible pulsing means it's reacting to the AI's own frame-to-frame noise instead
  of real change.
- **The "outlier yank" symptom** (the specific cause of the smearing/bleeding found
  earlier): watch the moment right after something briefly very close to camera (a
  hand reaching out, a flash, an object passing near the lens). If the *entire*
  frame's depth intensity visibly lurches or snaps right at that moment — not just
  the object itself, the whole background too — that's a single outlier frame
  yanking the normalization scale around.

**Too high (reacts too slow, laggy) — what to look for:**
- The 3D effect feels like it's a beat behind what's actually happening on screen —
  looks "off"/flatter for a moment after a real change, then settles into the
  correct depth.
- **Best place to catch it:** a push-in/pull-back shot, a zoom, or the first second
  or two right after a hard cut into a very differently-composed shot. If depth
  looks noticeably flat for a beat before catching up, the EMA is too heavily
  smoothed to react quickly (the risk explicitly flagged with `0.98` decay).

**Reliable testing method:** pick two short test clips — one mostly-static dialogue
scene (to catch "too low") and one with a push/pull or hard cut into a very
different shot (to catch "too high") — render each with the current setting and one
setting up/down from it, and use **Compare Presets** so the versions play back to
back. Subtle pulsing or lag is much easier to notice watching clips consecutively
than trying to judge a single pass from memory.

### Exactly how often the depth scale actually updates (verified against `depth_scaler.py`)

A natural assumption is that "Buffer `22`" means the normalization recalculates once
every 22 frames — **that's not how it works.** Read the actual `EMAMinMaxScaler`
implementation to confirm the real mechanism, which updates **every single frame**
(so ~24 times/second on a 23.976fps film), via two layered steps:

1. **The window (buffer size, e.g. `22`):** a rolling window of the most recent N
   frames' min/max values, recomputed fresh every frame as new frames enter and old
   ones age out of a ring buffer.
2. **The smoothing (decay, e.g. `0.80`):** the freshly-computed window value is
   blended with the *previous* frame's already-smoothed value at roughly an 80%
   old/20% new weighting (for decay `0.80`) — every single frame, nudging gradually
   toward the new value rather than snapping to it.

**What this means in real seconds, at Buffer `22`/Decay `0.80` on a ~24fps film:**
- A brand-new extreme value (something suddenly very close/far) enters the rolling
  window immediately, but lingers in that window's memory for up to `22` frames
  (**~0.9 sec**) before aging out — this is the mechanism that dilutes a one-off
  outlier spike instead of letting it dominate.
- The 80/20 decay blending on top of that means it takes roughly **~10 frames
  (~0.4 sec)** for the smoothed value to catch up most of the way to a genuinely new,
  sustained value.
- Combined: a real, sustained change in the scene's depth composition (a cut, a
  reveal) takes **roughly 1 second or a bit more** to fully settle into the new,
  correct range.
- Inside one continuous, unchanging shot (static dialogue, a locked-off wide shot),
  the smoothed value stays essentially flat — there is no periodic "jump" waiting to
  happen; movement only occurs if something in the shot actually gets meaningfully
  nearer or farther.

**For comparison, at Buffer `120`/Decay `0.98`** (the smearing-fix setting), that same
"settle to a real change" time stretches to several seconds instead of ~1 — much
better at ignoring brief outlier spikes, but noticeably slower to catch up to genuine
depth changes within a long, evolving shot. This is the same too-high lag trade-off
described above, now with concrete timing behind it.

---

## 3. Full Settings Catalog (quality-relevant, from the codebase)

### Depth capture

| Setting | Range/choices | Default | Highest-quality choice |
|---|---|---|---|
| Depth Model | see table above | `ZoeD_Any_N` | `VDA_Metric_L` for video (temporal stability + size); `Any_V3_Mono` for stills |
| Resolution (`--resolution`) | model-specific default: 392/384/512/518; DepthPro fixed 1536/1024 | model default | See caveat below — higher isn't unconditionally better |
| Depth Anti-aliasing (`--depth-aa`) | on/off | off | On — free cleanup, only works on `VDA_*`, `Any_V2_*`, `Any_V3_Mono*` (silently inert on ZoeDepth/DepthPro) |
| TTA (`--tta`) | on/off | off | On — runs depth twice (normal + flipped) and averages; ~2x cost, normally skipped for video but worth it if speed doesn't matter |

**Resolution caveat — real-world finding:** pushing resolution well above default (e.g.
768–1024) does look cleaner/sharper, but was found to cause **quicker eye fatigue**.
Mechanism: higher resolution lets the depth model resolve independent depth values for
fine texture (individual hair strands, grass blades, fabric weave) that a lower
resolution would blur into one smooth value. Each of those gets shifted by a slightly
different amount per eye, so a hair mass moves as many independently-wiggling strands
instead of one coherent block — the eyes have to work to fuse that, causing fatigue
over a long runtime even though the depth map is technically more accurate. This
mirrors why professional conversion artists deliberately do **not** roto individual
hair strands — they use one smooth gradient for the whole mass and let the real 2D
texture carry fine detail; pixel-level depth accuracy doesn't add perceived quality
past a point, it adds noise the eyes must reconcile. **Practical recommendation: test
in the 512–640 range** rather than pushing to 768–1024, and re-test specifically on
busy-texture scenes (hair, foliage, fabric close-ups) — the right value is
footage-dependent and can only really be confirmed by your own eyes on your own clips.

### Turning depth into stereo (the warp/inpaint step)

| Method | Mechanism | Occlusion handling | Divergence range |
|---|---|---|---|
| `row_flow_v3` (default) | ML backward-warp | Stretch/interpolate, no true inpainting | 0.0–5.0 |
| `row_flow_v3_sym` | Symmetric, 2x faster | Same, no inpainting | 0.0–5.0 |
| `mlbw_l2` / `mlbw_l4` | Multi-layer backward warp, more capacity | No inpainting | 0.0–10.0 |
| **`mlbw_l2_inpaint`** | mlbw warp + real learned inpainting network | **True inpainting** of hidden/disoccluded areas | 0.0–5.0 |
| **`forward_inpaint`** | Physically-correct forward warp + inpainting | **True inpainting** | 0.0–5.0 |
| `monobw` / `monobw_inpaint` | Simple single-layer warp (+ optional inpaint) | Baseline | — |
| `grid_sample` / `backward` | Naive, no model | No hole handling — lots of ghosting | — |

**Best quality: `mlbw_l2_inpaint` or `forward_inpaint`** — the only methods that
actually paint in the hidden background behind objects instead of stretching/smearing
nearby pixels into the gap (the single most visible "cheap conversion" tell).

**Inpainting model** (`--inpaint-model`): `Video_Large_Aether` is the largest bundled
option (default is `light_inpaint_v1`, optimized for speed). **Correction, 2026-08-17:
"largest" did not mean "best" for this project in practice.** Real A/B testing on
Mario Galaxy found `Video_Large_Aether` made the fine-detail warp issue (crown/fingers,
below) *worse*, not better, while `light_inpaint_v1` was the actual winner. Bigger
inpainting capacity apparently isn't a free quality upgrade here — treat model size as
a hypothesis to test per-project, not an assumed ranking. `light_inpaint_v1` is this
project's confirmed choice going forward.

**Keep these unset for full quality:** Stereo Width, Inpaint Max Width, Max Output
Width/Height — every one of these is a hidden downscale-for-speed shortcut.

### Depth "feel" (divergence / convergence)

| Setting | Range | Notes |
|---|---|---|
| Divergence (3D Strength) | 0–5 (or 0–10 for mlbw_l2/l4 non-inpaint) | Higher = more dramatic but more artifacts. **Professional conversions are actually restrained** — recommend ~2.0–2.5, not maxed |
| Convergence Plane (mode) | `constant` / `sod_v1` / `face_detect` | **Real-world finding: `constant` is more comfortable than `sod_v1` over a full movie** (see explanation below) — despite `sod_v1` being the theoretically closer automated approximation of a stereographer's convergence pulls |
| Convergence Plane (value) | 0–1 | `0` = convergence at the *farthest* point in the scene → almost everything pops toward viewer. `1` = convergence at the *nearest* point → almost everything recedes, nothing ever pops. `0.5` = balanced 50/50 split. **`0.65`–`0.7` = premium/restrained sweet spot** (see Section 5) |
| Foreground Pop | 0.0–1.0 | Keep modest to off (`0`–`0.25`) — professional depth budget favors background recession over aggressive pop |
| **Background Pop** (custom addition, 2026-08-16) | 0.0–1.0 | **Not part of upstream nunif** — added to this fork specifically for this project. Mirror image of Foreground Pop: only touches the farthest 15% of pixels (vs. Foreground Pop's nearest 15%) and pushes them further away, leaving the foreground untouched. See subsection below |
| **Foreground Scale** | -3.0–3.0 | **Different from Foreground Pop** — see "Foreground Scale — verified mechanism" subsection below for the corrected (code-verified) explanation of what positive vs. negative actually does. It is a **redistribution** between foreground and background separation, not a global exaggerate/compress. Project's own docs recommend `0` (off) for video — it's a corrective tool for flat-looking stills, not a general quality dial. Keep at `0` for a natural, restrained look, or see below for deliberately pushing it off-default |
| Preserve Screen Border | on/off | On — automated cousin of the professional "floating window" technique that prevents objects popping out while being clipped by the frame edge |
| Edge Dilation | X/Y, default 2/1 | Comfort-vs-accuracy dial, not a pure quality slider. `0` = max depth accuracy but more edge tearing; `3–4` = cleaner/more comfortable but less accurate. **Real-world finding:** if testing `1/1` (or even `0/0`) shows no visible smearing/tearing on your footage, it genuinely is better (sharper, more accurate edges "for free") — but this is scene-dependent (busy/high-contrast edges are worst-case), so verify across several representative clips, not just one, before committing for a full movie |

#### Background Pop — custom addition, not upstream nunif (2026-08-16)

Built at the user's request as a direct mirror of `apply_foreground_pop()`
(`iw3/depth_effects.py`). Foreground Pop only had a "push the closest stuff
forward" direction; there was no equivalent for "push the farthest stuff back."

**Mechanism (identical to Foreground Pop, just flipped):**
```python
def apply_background_pop(depth, strength, threshold_percentile=0.15):
    threshold = d.flatten().quantile(threshold_percentile)
    below = (threshold - d).clamp(min=0)   # 0 for foreground pixels
    d_suppressed = d - below * strength * 2.0
```
Only the farthest 15% of pixels (below the 15th percentile) get pushed further
away, proportional to how far below the threshold they already are; the near
85% of the scene is untouched — same surgical, single-cluster behavior as
Foreground Pop, just aimed at the opposite end.

**How it differs from negative Foreground Scale:** Foreground Scale reshapes
the *entire* depth curve (a redistribution — background gains separation only
by taking it from the foreground). Background Pop only touches the extreme
back cluster and leaves everything else — including the rest of the
background — alone. Use Background Pop for a targeted "make the deep
background recede further" push without affecting midground/foreground at
all; use negative Foreground Scale when you want a full-scene redistribution
instead.

**Wired up:** `--background-pop` CLI flag (mirrors `--foreground-pop` exactly,
0.0–1.0, default `0.0`), GUI "Background Pop" combo box (Stereo group, right
below Foreground Pop), and output filename tagging (`_bp<value>`, only appears
when non-zero, so existing filenames for renders not using it are unaffected).
**Untested in real footage as of creation** — same starting-point caution as
any new lever: verify on a real clip before trusting it in a whole-movie
sheet.

**Coverage percentage now adjustable (2026-08-18).** The `threshold_percentile`
(originally hardcoded at `0.15` = farthest 15%) is now exposed as
`--background-pop-coverage` (CLI, `0.0`–`1.0`) and "Background Pop Coverage %"
(GUI, right below Background Pop, shown as a whole percentage). Default stays
`0.15`/`15` — existing renders/filenames are unaffected unless this is changed.
Verified directly: raising it to `0.25` measurably affects more pixels (confirmed
~25% vs ~15% of the frame in a test tensor), and omitting the parameter entirely
still reproduces the original hardcoded-15% behavior exactly.

**Trade-off to know before raising it:** the transition line itself never has a
literal value jump (push is mathematically zero right at the cutoff on both
sides) — but there's still a **kink in the depth gradient** there: pixels just
past the line get pushed, pixels just before it don't, so a smoothly receding
surface (a hallway floor, an open landscape) can show a subtle bend in its rate
of recession at that exact point. At the default `15%`, that line usually sits
deep in the periphery (sky, distant background clutter) where a subtle kink is
easy to miss. Raising coverage moves the line further into the frame, into
territory more likely to be a continuous, visually important surface — raising
the odds the kink is actually noticeable. Bigger coverage = affects more of the
scene, but also raises transition-line risk; it isn't simply "stronger with no
downside."

#### Background Divergence — custom addition, not upstream nunif (2026-08-18)

Built at the user's request as a more literal answer to "can the farthest 15% use a
different Divergence value than the rest of the scene?" — Background Pop (above)
approximates this with an additive push, but isn't a true per-region Divergence
override. This is.

**Mechanism** (`apply_background_divergence()`, `iw3/depth_effects.py`): since final
pixel shift is proportional to `(depth - convergence) × divergence`, rescaling a
pixel's distance from the convergence plane by `background_divergence /
base_divergence` *before* the shared Divergence multiplier runs produces the exact
same result as if that pixel alone had used `background_divergence`:
```python
def apply_background_divergence(depth, convergence, base_divergence, background_divergence,
                                 threshold_percentile=0.15):
    ratio = background_divergence / base_divergence
    threshold = d.flatten().quantile(threshold_percentile)
    mask = (d < threshold)              # farthest 15%, same cutoff as Background Pop
    rescaled = convergence + (d - convergence) * ratio
    d_out = d * (1 - mask) + rescaled * mask
```
Verified by direct test: pixels above the threshold come out bit-for-bit identical to
the input (zero cost to foreground/midground); pixels below the threshold match the
predicted `convergence + (orig - convergence) × ratio` formula exactly; passing the
same value for both `base_divergence` and `background_divergence` is a confirmed
no-op.

**How this differs from Background Pop:** Background Pop pushes proportional to how
far a pixel already sits *below the threshold* (an additive offset, strength 0–1 in
its own units). Background Divergence instead rescales each pixel's distance from the
*convergence plane* by a true ratio of two Divergence values, in the same units as
`--divergence` itself. They can be used independently or together (different
mechanisms, no code conflict) — using both stacks their effects, similar to combining
Foreground Pop and Background Pop, so treat that combination as untested/unverified
if you try it.

**Same seam caveat as Background Pop applies:** it's a hard percentile cutoff, not a
smooth blend, so pushing `background_divergence` far from the base Divergence value
risks the same "layered/disconnected" look discussed for Background Pop — most
relevant the farther apart the two Divergence values are.

**Wired up:** `--background-divergence` CLI flag (same units/range as `--divergence`,
unset by default = disabled), GUI "Background Divergence" combo box (Stereo group,
right below Background Pop, blank = disabled), and output filename tagging
(`_bd<value>`, inserted right after the main `d<value>` divergence tag, only appears
when set). **Untested in real footage as of creation.**

#### Foreground Divergence — custom addition, not upstream nunif (2026-08-18)

Mirror image of Background Divergence, built the same session at the user's request —
same mechanism (`apply_foreground_divergence()`, `iw3/depth_effects.py`), just
`threshold_percentile=0.85` and `mask = d > threshold` instead of `<`, so it targets
the *nearest* 15% of pixels instead of the farthest. Verified with the same test
approach: pixels below the threshold come out bit-for-bit identical to the input,
pixels above match the predicted `convergence + (orig - convergence) × ratio` formula
exactly, and matching `foreground_divergence` to the base Divergence is a confirmed
no-op.

**Wired up:** `--foreground-divergence` CLI flag (same units/range as `--divergence`,
unset by default = disabled), GUI "Foreground Divergence" combo box (Stereo group,
right below Foreground Pop, blank = disabled), and output filename tagging
(`_fd<value>`, inserted right after the main `d<value>` divergence tag, before
`_bd<value>` if both are set). Same hard-cutoff seam caveat as the other three custom
depth-effect additions applies. **Untested in real footage as of creation.**

**The full custom-addition family, for reference:** this project's fork now has four
depth-effect levers beyond upstream nunif, all in `iw3/depth_effects.py`: Foreground
Pop (additive push, nearest 15%), Background Pop (additive push, farthest 15%),
Foreground Divergence (true Divergence-ratio rescale, nearest 15%), Background
Divergence (true Divergence-ratio rescale, farthest 15%). None of these exist in
upstream nunif — remember that if ever comparing against the public project's docs
or updating from upstream.

#### Foreground Scale — verified mechanism (corrected)

An earlier version of these notes had this backward — corrected after tracing the
actual code (`iw3/mapper.py`) rather than trusting a paraphrase. Foreground Scale
selects a different reshaping curve applied to the depth/disparity map before the
stereo warp:

- **Positive values** (`mul_1`/`mul_2`/`mul_3`): computed the curve directly — an
  input of `0.5` (mid-depth) maps to an output of only `~0.03`, while `0.75` maps to
  `~0.30` and `0.9` maps to `~0.70`. Almost the entire output range gets devoted to
  stretching apart the **near half** of the scene (more separation/roundness among
  **foreground** objects), while the **far half is squeezed flat**. **Positive =
  sharpens the foreground, flattens the background.**
- **Negative values** (`inv_mul_1`/`inv_mul_2`/`inv_mul_3`): the inverse shape — an
  input of `0.5` maps to `~0.75`. Most of the output range now stretches apart the
  **far half** of the scene (more separation deep into the **background**), while the
  **near half gets compressed**. **Negative = sharpens the background, flattens the
  foreground.**

**This is a redistribution/trade-off dial, not a "both ends get better" dial** —
whichever end you push toward gains separation, the other end loses it. It cannot
give you a punchier foreground *and* a deeper background simultaneously; pushing it
one way costs the other.

**Practical recommendations:**
- **Want a more immersive/deep-feeling background:** use **negative** values —
  `-0.5` moderate, `-1.0` stronger.
- **Want a punchier, more sculpted foreground:** use **positive** values — `+0.5`
  moderate, `+1.5` stronger (this was the correct direction for the earlier "Punchy
  Pop Out" recommendation).
- **Want some of both, or unsure:** keep it modest (`±0.25`–`0.5`) rather than
  pushing to either extreme, since extremes make one end of the scene noticeably flat.
- **Want more separation on *both* ends at once without trading one for the other:**
  don't use this setting for that — raise **Divergence** instead, since it scales
  both ends up together rather than redistributing between them.

#### Edge Dilation — verified relationship with Divergence (code-checked)

Traced the actual pipeline order in `iw3/utils.py` / `iw3/dilation.py`: Edge Dilation
(`dilate_edge`) is applied to the depth map **before** Divergence is applied
(`depths = -dilate_edge(-depths, args.edge_dilation)` runs, then that result is passed
into `apply_divergence`). `dilate_edge` itself just runs a fixed-size blur/smoothing
kernel a fixed number of times (`n_iter` = the Edge Dilation value) on the depth map —
it has no awareness of what Divergence will be applied afterward, and does not scale
itself based on it.

**Practical consequence:** Edge Dilation smooths a *fixed* amount of roughness out of
the depth map, in depth-map terms. Divergence then multiplies whatever depth
differences remain into actual left/right pixel shift. So the same leftover edge
roughness becomes a small (maybe invisible) pixel-shift error at low Divergence, but
gets stretched into a much bigger, more visible one at high Divergence. **Edge
Dilation does not auto-compensate for Divergence — raise it manually as Divergence
goes up to keep the same level of edge protection.**

This confirms a pattern already implicit in the settings sheets in Section 6 without
being stated outright: Pop-In tier (Divergence ~2.0–2.25) pairs with Edge Dilation
`2 1`; Pop-Out tier (Divergence ~3.0–3.5) pairs with `3 2`. Treat that pairing as a
real rule going forward, not a coincidence — when raising Divergence for a stronger
look, raise Edge Dilation roughly in step with it.

#### Edge Dilation also interacts with Depth Resolution (code-checked, 2026-08-15)

Traced the actual call order: `dilate_edge()` runs **inside** the depth model's own
inference function (`depth_anything_model.py` `batch_infer()`), on the depth map
*while it's still sized at whatever Depth Resolution/`--resolution` produced* (e.g.
`518`, `728`) — **before** that depth map gets upscaled to match the source image's
full resolution (that upscale happens later, in `utils.py`,
`F.interpolate(depth, size=(im.shape...))`).

Edge Dilation's loop runs a fixed-pixel-size smoothing kernel (~3px) a set number of
times — it has no awareness of the depth map's current resolution. So the *same*
Edge Dilation value (e.g. `3 2`) covers a **larger proportion of the frame** when
Depth Resolution is lower (e.g. `518`) than when it's higher (e.g. `718`), because
that fixed few-pixel brush is a bigger slice of a smaller map. Once both get
upscaled to full output resolution, that proportional difference persists as a
visibly stronger/wider smoothing effect at lower Depth Resolution, and a
weaker/narrower one at higher Depth Resolution, for the identical Edge Dilation
number.

**Practical consequence:** an Edge Dilation value tuned at one Depth Resolution is
not guaranteed to look the same after changing Depth Resolution — re-verify Edge
Dilation any time Depth Resolution changes, the same way it already needs
re-verifying any time Divergence changes. Rough scale: `718` vs `518` is about a
1.39x resolution jump, so the same Edge Dilation setting covers roughly 1.39x less
of the frame proportionally at `718` — a real, visible difference, not a rounding
nuance.

#### Why `constant` beat `sod_v1` in practice

Every time the convergence point moves — even gradually, even smoothed — the eyes
have to physically readjust their vergence angle to keep fusing the image comfortably.
`sod_v1` continuously re-evaluates "what's the subject" frame to frame, so even
*within* a single unbroken shot it can subtly drift (someone shifts position, camera
pans, a second face enters frame). Smoothing softens how abruptly it moves but doesn't
stop it moving — the eyes are still doing small constant re-adjustment work
throughout the film. `constant` never moves once set, so there's zero ongoing
vergence-tracking demand. This also matches real stereographer practice: convergence
is held steady *within* a shot and stepped to a new value only at cuts (a deliberate
"convergence pull"), not continuously reacting to subject motion mid-shot — `sod_v1`,
even smoothed, is closer to "constantly reacting" than that.

**When `sod_v1` genuinely wins anyway:** a push/pull "reveal" shot where the camera
moves from a tight close-up to a wide reveal *within one continuous take* — the
depth-of-interest changes dramatically inside a single shot, which a fixed `constant`
value structurally cannot follow (it's wrong for one end of the move or the other).
`sod_v1` can track the subject as the shot evolves. These shots are relatively rare in
most films though, so `constant` wins on average by avoiding unnecessary drift on the
much more common static/dialogue shots. For the closest possible result, the movie
could be split into segments and only the rare reveal shots processed with `sod_v1`,
rest on `constant` — real extra manual work, but the closest automated approximation
of what a human stereographer actually does shot-by-shot.

### Motion / temporal stability

| Setting | Recommendation |
|---|---|
| Flicker Reduction (`--ema-normalize`) | On |
| Scene Boundary Detection (`--scene-detect`) | On — resets smoothing cleanly at real cuts |
| EMA Decay / Buffer | See Section 2 — use real measured per-film data where possible |
| Max FPS | Set above source's actual frame rate (e.g. 60) so no frames get silently dropped to 30fps |

### Color & output

| Setting | Recommendation | Why |
|---|---|---|
| Pixel Format | `yuv444p` (or `rgb24`/`gbrp16le` for max/lossless) **— SDR only, see HDR caveat below** | Default `yuv420p` throws away 3/4 of color detail, causing red-channel degradation and left/right-eye ghosting/crosstalk (**bleeding**) specific to 3D |
| Bit Depth Upgrade | `12` with `libx265` (CPU); `10` if using NVENC/QSV (hardware-capped there anyway) | Reduces banding |

**HDR/Dolby Vision caveat (2026-08-17):** the `yuv444p` recommendation above assumes
SDR content. For a source with HDR/Dolby Vision being preserved (`--preserve-dowi`,
"Preserve Dolby Vision" in GUI), **`yuv420p10le` is the correct, required choice, not
a mistake** — HDR/Dolby Vision metadata pipelines and hardware encoders (`hevc_nvenc`
included) expect 4:2:0 10-bit; `yuv444p` isn't a compatible/supported combination for
that workflow. The bleeding risk this table warns about is a real trade-off for HDR
titles, but it's not avoidable by switching pixel format without dropping HDR/Dolby
Vision preservation entirely — for any Mario-Galaxy-style HDR source, use
`yuv420p10le`.
| Video Codec | `libx265`, CRF 15–18, slower/veryslow preset | Best achievable quality per bit; slow CPU encode |
| Denoise | On only for grainy/old film sources, off for clean modern digital | Grain can confuse the depth model |
| Low VRAM | Off | Pure speed/memory shortcut, no quality benefit to leaving on if you have headroom |

---

## 4. How Professional (Manual) Blu-ray 3D Conversion Actually Works

Research summary — for comparison, to understand what the AI pipeline is approximating.

- **Rotoscoping into depth layers** is the core labor: every surface gets isolated with
  a mask and given its own depth. A single close-up face can need ~7+ separate roto
  shapes (nose, cheeks, ears, lips...) each at a slightly different depth so it reads
  as round, not a flat cutout.
- **Volumetric effects** (smoke, rain, fire) can't be roto'd — need separate handling.
- **In-Three's "Dimensionalization"** uses a hybrid: a human-authored **"depth script"**
  (shot-by-shot depth intent) combined with automated cues. Fully-automatic conversion
  with no human depth script is explicitly called out as producing worse results.
- **Cardboard cutout effect** comes from two causes: (1) non-linear depth compression
  with distance flattening the background into layered "cards," and (2) insufficient
  internal roundness per object — a whole character on one flat depth plane reads as
  flat even if separated from the background. Fix: per-object interaxial/convergence
  and manually sculpted depth *gradients* within each shape, not just flat layers.
- **Window violations**: an object with negative parallax (popping toward viewer) that
  also gets clipped by the frame edge is one of the most uncomfortable stereo errors.
  Professional fix: the **floating window** — an independently-shifted black mask on
  each eye's frame edge that moves the perceived screen boundary forward in space.
- **Depth budget rules of thumb** (from stereography literature):
  - Max divergence ~1.0° (0.5° per eye) for comfort
  - Max parallax (either direction) ≤ ~2.9–3% of screen width
  - Positive parallax should never exceed the viewer's actual interocular distance
  - **~1/3 of depth budget forward of the screen, ~2/3 behind** — favoring "a window
    into a space" over "stuff poking at you" (this is why restrained conversions read
    as premium, and maxed-out pop-out settings read as gimmicky)
  - Interaxial starting point: camera-to-nearest-subject distance ÷ 30 ("1/30 rule"),
    then hand-tuned per shot
  - Typical pixel shift per shot in real productions: ~50–60px average, up to ~80px
- **Occlusion/disocclusion**: cheap conversions stretch/smear nearby pixels into the
  gap left behind a foreground object. Professional work either manually paints in the
  hidden background, or builds rough 3D proxy geometry so it's reconstructed rather
  than guessed.
- **Convergence pulls**: convergence is adjusted per shot, sometimes within a shot
  (literally like a focus pull). James Cameron reportedly went through *Titanic 3D*
  shot-by-shot with a jog wheel by hand.
- **Temporal stability** is largely a non-issue for manual work (artists animate depth
  as a continuous timeline) but a real, distinct problem for frame-independent AI
  depth models — hence why flicker reduction / temporal-consistency settings matter so
  much more for this tool than they would for a human conversion team.
- **Fine detail (hair/fabric/foliage)**: manual conversion doesn't actually get
  pixel-perfect depth on every hair strand — it uses a coarse-but-correct depth volume
  for the mass, and relies on the original 2D texture for fine detail. AI models lose
  detail because most depth networks compress the image internally and the
  upsampling back to full resolution smooths out fine/soft edges — worst on exactly
  hair, fabric, and long-distance objects. (This is also the direct explanation for
  the resolution/eye-fatigue finding in Section 3.)

---

## 5. "Premium Pop-In" Profile — Recreating the Titanic / Avatar / Mario Bros Movie Feel

The reputation those films have for premium, comfortable, non-fatiguing 3D comes
specifically from *restraint*, not intensity. Every choice below traces back to the
professional research in Section 4.

| Setting | Value | Why |
|---|---|---|
| Divergence (3D Strength) | `2.0` | Restrained on purpose — maxing this is what makes cheap conversions feel gimmicky |
| Convergence Mode | `constant` | Confirmed more comfortable in real testing than `sod_v1` (see Section 3) |
| Convergence Value | `0.65` | Implements the 1/3-forward/2/3-behind depth budget rule — "a window into a deep space," not "stuff reaching at you" |
| Foreground Pop | `0.0` (off) | Avatar/Titanic rarely do the "poke the audience" gimmick |
| Foreground Scale | `0.0` (off) | Keep the depth curve natural/unexaggerated |
| Preserve Screen Border | On | Automated floating-window equivalent |
| Edge Dilation | `2`/`1` default (test lower if clean — see Section 3) | At this restrained divergence there's little edge tearing to hide to begin with |

**Adapting to a specific film:** keep the depth-feel values above unchanged (the
restraint itself is what reads as premium, regardless of genre), but swap the
**pacing** settings (EMA Decay/Buffer) to match the *specific film's own* measured
cut rhythm (Section 2) rather than copying Titanic/Avatar's own slower, sweeping-epic
pacing onto a faster-cut film — pacing mismatch feels laggy/wrong even when the depth
philosophy is correct.

---

## 6. Concrete Settings Sheets — Hocus Pocus & Super Mario Galaxy Movie

### Shared "maximum quality" base (both films)

| Setting | Value |
|---|---|
| Depth Model | `VDA_Metric_L` |
| Resolution | Test 512–640 (see eye-fatigue caveat, Section 3) |
| TTA | On |
| Depth Anti-aliasing | On |
| Method | `mlbw_l2_inpaint` or `forward_inpaint` |
| Inpaint Model | `Video_Large_Aether` |
| Inpaint Overlap Frames | 3/3 |
| Convergence Mode | `constant` |
| Foreground Scale | `0.0` |
| Preserve Screen Border | On |
| Flicker Reduction | On |
| Scene Boundary Detection | On |
| Max FPS | Above source FPS |
| Pixel Format | `yuv444p` |
| Bit Depth Upgrade | 12 (`libx265`) or 10 (NVENC/QSV) |
| Codec / CRF / Preset | `libx265`, CRF 15–18, slower |
| Low VRAM | Off |

### Hocus Pocus (1993)

Measured: 96.13 min, 1,416 shots, mean 4.07s / median 2.38s.

| Setting | Value |
|---|---|
| EMA Decay / Buffer | `0.80` / `20` (data-driven) — or `0.98` / `120` if smearing/bleeding appears on your specific copy (see Section 2) |
| Denoise | On — 1993 film print, likely real grain |

| Setting | Pop-In (Deep & Comfortable) | Pop-Out (Theatrical) |
|---|---|---|
| Divergence | `2.0` | `3.0` |
| Convergence Value | `0.65` | `0.3` |
| Foreground Pop | `0.0` | `0.35` |
| Edge Dilation | `2`/`1` | `3`/`2` |

### Super Mario Galaxy Movie (2026)

Measured: 98.25 min, 1,413 shots, mean 4.17s / median 2.67s. (Real data corrected an
earlier genre-based guess of "fast-cut action" pacing — actual measured pacing is
essentially the same as Hocus Pocus, slightly slower on median.)

| Setting | Value |
|---|---|
| EMA Decay / Buffer | `0.80` / `22` (data-driven) — or `0.98` / `120` if smearing/bleeding appears |
| Denoise | Off — clean modern CGI, no grain |

| Setting | Pop-In (Deep & Immersive) | Pop-Out (Theatrical) |
|---|---|---|
| Divergence | `2.25` | `3.5` |
| Convergence Value | `0.65` | `0.25` |
| Foreground Pop | `0.0` | `0.5` |
| Edge Dilation | `2`/`1` | `3`/`2` |

#### Variant: `Any_V3_Mono_01` + `mlbw_l2_inpaint` (2026-08-14, user's active choice)

**Caveat first:** this deviates from the doc's own Section 1 rule of thumb
("video → `VDA_*`") — `Any_V3_Mono_01` is a stills-oriented model with **zero
built-in frame-to-frame memory** (code-verified: `decay=0, buffer_size=1` baked into
its scaler). It was adopted specifically because it removed the "warped crown"
artifact that `VDA_L`/`VDA_Metric_L` both showed, at the cost of introducing "depth
breathing." `VDA_Metric_L` remains the safer default for a whole-movie render if
depth breathing ends up being the bigger problem across 98 minutes vs. the
crown-warp being a narrow, scene-specific issue — worth an honest side-by-side
before committing the whole movie to this model.

**The one setting that matters most for this specific combo — force `--ema-normalize` on explicitly.**
Code-verified: iw3 only *warns* you to enable `--ema-normalize` when the depth model
is specifically `VideoDepthAnything` (`utils.py` line ~3362) — that check does not
fire for `Any_V3_Mono_01`, so nothing nudges you toward it, even though this model
needs it *more* than VDA does (VDA has some native temporal design; this model has
none). Critically, `--ema-normalize`'s Decay/Buffer values **do override** this
model's own zero-memory default (`enable_ema()` calls `scaler.reset(...)`,
confirmed in `base_depth_model.py`) — so turning it on is a real, working mitigation
for the depth-breathing problem, not a placebo. This was very likely **off** during
the earlier crown-artifact test that found depth breathing — worth confirming that
test again with it explicitly on before concluding breathing is unavoidable.

| Setting | Value | Why |
|---|---|---|
| Depth Model | `Any_V3_Mono_01` | User's choice — removes the crown-warp `VDA_*` showed |
| **`--ema-normalize`** | **On (do not skip)** | Not auto-recommended by the tool for this model, but functionally necessary here — see above |
| EMA Decay / Buffer | `0.80` / `22` | Same data-driven whole-movie value as the `VDA_Metric_L` sheet above — no reason to differ, and it's the one actually compensating for this model's lack of native memory |
| Method | `mlbw_l2_inpaint` | User's choice — true inpainting, not pixel-stretching |
| Inpaint Model | `Video_Large_Aether` | Matches the project's existing "maximum quality" base (Section 6 top) |
| Inpaint Overlap Frames | `3`/`3` | Matches existing max-quality base |
| Depth Resolution | `512` (not higher) | This model is already "sharpest for stills" per Section 1 — pushing resolution further adds the Section 3 fine-detail/eye-fatigue risk *and* more per-frame noise for a model with no memory to smooth it. Lower end of the safe 512–640 range is the more conservative, artifact-averse choice here specifically because of the memory gap, not a general rule |
| TTA | **On** | Normally "worth it if speed doesn't matter" — for this model specifically it's more valuable than usual, since it's one of the few remaining levers that reduces per-frame noise before it has a chance to read as flicker/breathing across frames. Cost is irrelevant on the RTX 5090 |
| Depth Anti-aliasing | On | Confirmed supported (`Any_V3_Mono_01` is in `AA_SUPPORTED_MODELS`) |
| Divergence | `2.25` (Pop-In tier) | Deliberately the restrained tier, not `3.5` Pop-Out — the "less artifacts" goal and a memory-less depth model point the same direction: don't stack a stronger warp on top of a source of instability |
| Convergence Mode | `constant` | Confirmed more comfortable over a full movie (Section 3) |
| Convergence Value | `0.65` | Matches Pop-In tier above |
| Foreground Pop | `0.0` | Matches Pop-In tier — avoid stacking another artifact-amplifying lever |
| Edge Dilation | `2`/`1` | Paired to Divergence `2.25` per the verified pairing rule (Section 3) |
| Preserve Screen Border | On | Standard |
| Scene Boundary Detection | On | Resets EMA at real cuts — helps regardless of depth model |
| Denoise | Off | Clean CGI, no grain |
| Pixel Format | `yuv444p` | Avoids the color-fringing/bleeding failure mode (Section 7) |

**If depth breathing is still visible with `--ema-normalize` on:** raise EMA Buffer
toward `120` / Decay toward `0.98` (the project's documented smearing/bleeding fix)
even though this isn't literally smearing — the underlying mechanism (per-frame scale
instability being under-smoothed) is the same failure family, just with a
memory-less depth model as the trigger instead of a rare outlier frame.

**Method used to get the real measured data:** run iw3 with
`--scene-detect --scene-detect-only --depth-model NULL --method NULL` (skips loading
any real depth model — fast, cuts-only pass), then read the resulting
`nunif/tmp/iw3_scene_cache/<hash>.json` cache file's `"pts"` list and the file's real
fps (via `ffprobe`) to compute mean/median shot length directly. Far more reliable
than a genre-based guess.

### Real-world run log (actual command used, in progress as of Aug 2026)

Actual full CLI invocation used for a real render attempt, logged here for reference
along with its measured throughput and the open issue being investigated at the time:

```
python -m iw3 -i D:\Downloads\The.Super.Mario.Galaxy.Movie.2026.mkv -o "E:\3d Movies\Super mario galaxy" --compile --method mlbw_l2_inpaint --preserve-screen-border --convergence 0.65 --max-fps 1000.0 --crf 15 --tune uhq --depth-model VDA_L --foreground-pop 0.15 --pix-fmt yuv420p10le --max-output-width 3840 --max-output-height 2160 --resolution 718 --limit-resolution --ema-normalize --ema-buffer 90 --scene-detect --autocrop BLACK --edge-dilation 1 1 --inpaint-model light_inpaint_v1 --inpaint-overlap-frames 3 3 --depth-aa --max-workers 2 --video-format mkv --video-codec hevc_nvenc --hwaccel cuda --metadata filename --preserve-dowi --auto-resume --upgrade-pix-fmt 12 --yes
```

**Measured throughput: 4.10 fps** (on RTX 5090, 32GB VRAM) — useful real reference
point for estimating full-movie render time at this settings tier (~98 min runtime
at ~24fps source ≈ 141k frames → roughly 9.5–10 hours at this throughput).

**Notes on how this diverges from the "best quality" recommendation above** (kept
here as a record, not necessarily errors — some may be deliberate in-progress tests):

- `--resolution 718` rounds up internally to `728` (nearest multiple of 14 for this
  model family) — meaningfully **above** the recommended `512–640` safe range. Given
  this run coincided with investigating a fine-detail "warped crown" artifact, this
  resolution may be *contributing* to that issue rather than helping it, per the
  fine-detail/eye-fatigue mechanism in Section 3 — worth testing back down in the
  512–640 range specifically for that scene.
- `--ema-buffer 90` has no accompanying `--ema-decay`, so it falls back to the CLI
  default (`0.75`) — a **mismatched pairing** per the interpolation logic in Section 2
  (a buffer this large should be paired with a much higher decay, roughly `~0.965`,
  to actually get the "resist outlier spikes" benefit a large buffer is for;
  `0.75` decay largely undermines that benefit even with a big buffer).
- **No `--tta`** in this run, despite `VDA_L` + TTA being confirmed fully compatible
  and being one of the live hypotheses for reducing the crown-warping issue — worth
  a follow-up test run with `--tta` added back in for direct comparison.
- `--depth-model VDA_L` — deliberate choice (preferred for general immersion), but
  `VDA_Metric_L` is the other standing hypothesis for the crown-warping issue
  specifically (see below) and hasn't been tested on this scene yet.
- `--pix-fmt yuv420p10le` instead of `yuv444p` — still the documented cause of
  potential red-channel/ghosting degradation in 3D output (see Section 3/7);
  `--upgrade-pix-fmt 12` is also a no-op here since `hevc_nvenc` hard-caps at 10-bit
  regardless.
- `--stereo-width` and `--foreground-scale` are both absent (i.e. unset/default) in
  this run — correct for full quality on stereo-width; foreground-scale defaulting
  to `0` (neutral) rather than the earlier-explored `-0.5` (more background
  immersion) — likely fine, just noting the value in use for this specific render.

### GUI baseline snapshot — Mario Galaxy (2026-08-17, user-designated "best baseline")

Transcribed directly from a GUI screenshot the user asked to be recorded as the
reference baseline for Mario-Galaxy-style movies going forward. This **updates** the
"Real-world run log" above (same project, later date) — several values have moved
since that log, some discrepancies noted there are now fixed, others persist.

| Group | Setting | Value |
|---|---|---|
| Stereo Generation | 3D Strength (Divergence) | `2.0` |
| | Convergence Plane | `constant`, `0.55` |
| | Synthetic View | `both` |
| | Method | `mlbw_l2_inpaint` |
| | Inpainting Model | `light_inpaint_v1` |
| | Inpaint Overlap Frames | `3` / `3` |
| | Stereo Processing Width | Default (blank) |
| | Depth Model | `VDA_L` |
| | Depth Resolution | `718`, Limit to source: on |
| | Foreground Scale | `0` |
| | Edge Fix | `3` / `2` |
| | Depth Anti-aliasing | On |
| | Foreground Pop | `0.10` |
| | **Background Pop** | `0.25` |
| | Flicker Reduction (EMA) | On, Decay `0.99` / Buffer `220` |
| | Scene Boundary Detection | On (+ cache) |
| | Preserve Screen Border | On |
| | Stereo Format | Half SBS |
| Video Encoding | Video Codec | `hevc_nvenc` |
| | Pixel Format | `yuv420p10le` |
| | CRF | `15`, Preset `medium`, Tune `uhq` |
| Video Filter | Output Size Limit | `3840x2160` (correct pairing for Half SBS — see the Section 10 gotcha) |
| | Preserve Dolby Vision | On |
| | Bit Depth Upgrade | `12` |
| | Denoise | Off |
| Processor | Depth Batch Size / Worker Threads | `2` / `2` |
| | TTA | **Off** |
| | FP16 | On |
| | torch.compile | On |

**Update, 2026-08-17 (same day): user confirmed three of the items below via real
testing.** Corrected in place rather than left as stale flags:

- **Pixel Format `yuv420p10le` — confirmed CORRECT, not a flag.** This source uses
  HDR/Dolby Vision preservation (`Preserve Dolby Vision` is on in this same
  screenshot), and `yuv420p10le` is the required, compatible format for that
  pipeline — `yuv444p` is only the right call for SDR content. See the HDR caveat
  added to Section 3's Color & Output table. `--upgrade-pix-fmt 12` is still a no-op
  regardless (hevc_nvenc hard-caps at 10-bit), but that's a harmless no-op, not a
  quality cost.
- **Inpainting Model `light_inpaint_v1` — confirmed CORRECT, not a flag.** Real A/B
  testing found `Video_Large_Aether` made results *worse*, not better — this doc's
  prior "bigger model = better quality" assumption was wrong for this project.
  `light_inpaint_v1` is the confirmed, standing choice now — see the correction in
  Section 3 and the crown-investigation update above.
- **TTA off — confirmed CORRECT, not a flag.** Real testing found no visible
  improvement and a real speed cost. This closes out what had been the top
  candidate for the crown/fine-detail investigation — see the ruled-out entry above.

**Still open (not yet re-tested/confirmed one way or the other):**

- **Depth Model `VDA_L`, not `VDA_Metric_L`.** No cost difference between them (same
  architecture/size); `VDA_Metric_L` remains the standing recommendation both from the
  project author's stated preference and as an untested candidate fix for the same
  crown-warp investigation.
- **Convergence Value `0.55`, not the documented `0.65`–`0.7` sweet spot.** Pulls the
  convergence plane closer to center (roughly 50/50 pop/recede) rather than the
  restrained "mostly recession" premium feel documented in Section 5. May be
  deliberate — flagging the deviation, not asserting it's wrong.
- **EMA Decay/Buffer `0.99`/`220`** is well past the documented/tested range (which
  tops out at `120`/`0.98`) — see the dedicated discussion above. Real risk of
  sluggish response to genuine depth changes (push/pull shots), and the buffer is
  larger than nearly every shot in the film, so most of its size goes unused before
  a cut resets it. Untested at this project's current stage.
- **Edge Fix `3`/`2`** — this one is actually **well-justified**, not a regression:
  Depth Resolution `718` (see the interaction note above) and the newly-added
  Foreground Pop `0.10` / Background Pop `0.25` both independently push toward a
  higher Edge Fix than the base `2/1` pairing would suggest for Divergence `2.0`
  alone. `3/2` is a reasonable, if untested, response to both factors at once.

**What's already correct here:** Output Size Limit `3840x2160` paired with Half SBS
(matches the Section 10 finding exactly), Preserve Screen Border on, Scene Boundary
Detection on, `--stereo-width` left at Default, Foreground Scale at neutral `0`.

### Open investigation: "warped fine-detail" artifact (crown, fingers, ...)

Observed on a Princess Peach close-up (SBS screenshot): the crown's fine, spiky
points showed inconsistent/warped shape between the left/right eye views. Diagnosed
as likely a genuine stereo-warp distortion (not blur/softness) caused by the depth
model assigning inconsistent depth values across small, closely-packed geometry —
structurally the same category of problem as the hair-strand/fine-detail issue in
Section 3, here showing up as visible shape distortion rather than felt eye fatigue.

**Second occurrence (2026-08-14):** same symptom reported on fingers (also `VDA_L`)
— thin, closely-spaced geometry against a background, same as the crown's spikes.
Confirms this isn't crown-specific; the real trigger is any thin/closely-packed
fine detail, and the investigation below (ruled-out causes, untested candidates)
applies directly rather than needing to be re-run from scratch for each body part/
object it shows up on.

**Ruled out — depth model choice.** Tested `VDA_Metric_L` in place of `VDA_L`, all
else equal: crown warping was identical. Since these two share the same "Large"
architecture and only differ in training data (relative vs. metric-constrained), and
the result didn't change, the "training-objective-driven tendency" theory from the
earlier live hypothesis is no longer credible. Model choice within the VDA family is
not the cause.

**Ruled out — resolution.** Tested `384`, `448`, `512`, and `648` on the same shot:
crown warping was present at all four values. The earlier fine-detail/eye-fatigue
theory from Section 3 (resolution letting the model over-resolve tiny independent
depth values) does not explain this artifact — it's not a resolution-driven effect.

**Separately tried — `Any_V3_Mono_01`:** the crown warping *did* disappear with this
model, but it introduced a new problem instead — visible "depth breathing" (objects'
depth gently pulsing/swelling frame to frame). Root cause: `Any_V3_Mono_01` is a
stills-oriented model with no frame-to-frame memory, unlike the `VDA_*` family which
is purpose-built to stay temporally consistent across video frames. This is a
structural trade-off (sharper per-frame, unstable across frames), not a fixable
setting — see the Depth Model Guide above. Confirms the crown issue is specific to
the `VDA_*` family somehow, but swapping away from `VDA_*` entirely just trades it for
a worse, whole-video problem.

**New leading hypothesis — the stereo warp/inpaint step, not the depth map.** Since
neither depth model choice nor resolution (both upstream, depth-map-side settings)
change the outcome, the cause more likely sits downstream, in how the depth map gets
turned into two eye views. Two settings from the actual logged run stand out:

1. **Edge Dilation was `1 1`** — below the default `2 1`. Section 7's troubleshooting
   table explicitly names low Edge Dilation as the known cause of "stretched/torn
   look at busy, high-contrast object edges," which describes a crown's spiky points
   against a background well.
2. **Inpaint Model was `light_inpaint_v1`** (the fast/lightweight default), not
   `Video_Large_Aether` (the higher-quality option used in the "best quality" sheets
   elsewhere in this doc). The crown's spikes create a lot of hidden-area-behind-object
   fill work for the inpainting network on each eye; a weaker inpainting model filling
   the two eyes inconsistently would visually present as warping.

**TTA's role, honestly assessed:** TTA cancels *directional* (left/right) bias by
averaging a normal and flipped pass. Originally reasoned to not fully resolve this
issue on the assumption the root cause was training-objective-driven — that
assumption is now void (model choice ruled out above), so TTA is back to being a
genuinely open, untested variable rather than a low-priority one. Still no verified
percentage improvement for it — treat any such number as unfounded if it comes up.

**Ruled out — Edge Dilation `3 2` + Inpaint Model `Video_Large_Aether`.** Tested
together (per the plan above): no improvement to the crown warp — and per the
follow-up finding below, `Video_Large_Aether` specifically made it *worse*. This is a
significant result — both the "low edge dilation causes tearing at busy edges"
troubleshooting rule (Section 7) and the "weak inpaint model fills eyes
inconsistently" theory predicted this combination would help, and it didn't. The
downstream warp/inpaint step is now a weaker hypothesis than it looked; the true
cause may sit somewhere neither purely upstream (depth map) nor purely downstream
(warp/inpaint) in the simple sense already tested.

**Ruled out — Inpaint Model `Video_Large_Aether` generally (2026-08-17, confirmed via
real A/B testing, not just the crown shot).** Contrary to the "bigger model = better"
assumption this doc previously made, `Video_Large_Aether` made results look *worse*
across testing, not just neutral on the crown issue specifically. `light_inpaint_v1`
is the confirmed winner and is now this project's standing choice — see the
correction in Section 3's inpainting-model note.

**Ruled out — TTA (2026-08-17, confirmed via real testing).** User-tested directly:
no visible improvement, and it measurably slows down conversion. This closes out what
had been the most-flagged remaining hypothesis — the "unstable per-pixel depth noise"
theory did not pan out in practice. Do not spend further time on `--tta` for this
issue.

**Still untested — candidates remaining:**
1. **`--depth-aa` on vs. off** — this flag runs a small learned neural network
   (`iw3/models/depth_aa.py`, a window-attention model, 8x8 pixel windows) over the
   depth map to anti-alias/smooth it, as a separate step from Edge Dilation. It has
   never been toggled off in any test so far (all tests, including the crown ones,
   ran with it on). A window-attention model processing a tight repeating pattern
   like a crown's spikes is a plausible place for small inconsistent artifacts to
   originate — worth testing with `--depth-aa` removed, all else equal. **Now the
   top remaining candidate**, since TTA is ruled out.
2. **Stereo method itself (`--method`)** — all testing so far has used
   `mlbw_l2_inpaint`. Switching to `forward_inpaint` (a physically-correct forward
   warp, mechanically different from mlbw's learned backward warp — see Section 4
   table) would test whether the warp *mechanism*, not just the inpaint model
   plugged into it, is where the inconsistency originates.

**Next actual test to run:** same crown shot/scene, same depth model/resolution as
before, test `--depth-aa` off next (cheapest remaining candidate). If that doesn't
help, test `forward_inpaint` as the method. Change one variable at a time.

---

## 7. Troubleshooting: Smearing & Bleeding

Two visually similar but mechanically different problems, with different fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| **Smearing** — stretched/torn look at busy, high-contrast object edges | Edge Dilation set too low (e.g. `1/1` or `0/0`) for that specific scene's complexity | Raise Edge Dilation back up (`2/1` or higher) for the affected scenes |
| **Bleeding** — color fringing/doubling, ghost-like | Pixel Format still on default `yuv420p` instead of `yuv444p` | Switch Pixel Format to `yuv444p` (or `rgb24`/lossless) |
| **Smearing/bleeding-*like* swimming, inconsistent separation, occasional & scene-dependent** | EMA Buffer too small — a brief outlier frame (something momentarily very close, a flash, fast object entering/exiting frame) yanks the whole depth-scale normalization around, changing how much 3D separation gets applied moment to moment | Raise EMA Buffer significantly (tested: `120`, paired with Decay `~0.98`) to dilute any single outlier frame's influence — see Section 2 |

If unsure which is happening: check whether the affected scenes have busy/high-contrast
edges (points to Edge Dilation) or look more like color fringing (points to Pixel
Format) or happen unpredictably regardless of edge complexity (points to EMA Buffer).

---

## 8. Bottom Line — Best-Quality Combination for Full Movies

| Category | Setting |
|---|---|
| Depth Model | `VDA_Metric_L` |
| Resolution | 512–640 (not higher — see eye-fatigue caveat, Section 3) |
| Depth Anti-aliasing | On |
| TTA | On |
| Method | `mlbw_l2_inpaint` or `forward_inpaint` |
| Inpaint Model | `Video_Large_Aether` |
| Inpaint Overlap Frames | 3/3 or higher |
| Stereo Width / Inpaint Max Width / Max Output Size | All unset (full resolution) |
| Divergence | ~2.0–2.5 (not maxed — restrained reads as premium) |
| Convergence Mode | `constant` (confirmed more comfortable than `sod_v1` in real testing) |
| Convergence Value | `0.65`–`0.7` |
| Foreground Pop | `0.0`–`0.25` (modest to off) |
| Foreground Scale | `0.0` (off) |
| Preserve Screen Border | On |
| Edge Dilation | Default `2/1`, test `1/1` per-scene if no artifacts appear |
| Flicker Reduction | On |
| Scene Boundary Detection | On |
| EMA Decay/Buffer | Use real measured per-film data (Section 2) over genre guesses; raise toward `120`/`0.98` if smearing/bleeding appears |
| Max FPS | Above source FPS |
| Pixel Format | `yuv444p` |
| Bit Depth Upgrade | 12 (libx265) or 10 (NVENC/QSV) |
| Codec / CRF / Preset | `libx265`, CRF 15–18, slower |
| Denoise | Only for grainy/old sources |
| Low VRAM | Off |

No AI tool fully replicates hand-rotoscoped, hand-graded professional conversion —
but this combination targets the same failure points professionals specifically solve
for: true inpainting instead of pixel-stretching, a restrained (not maxed) depth
budget, stable convergence instead of distracting drift, and full-color output to
avoid ghosting.

**Practical note:** this combination maxes nearly every quality setting, so a full
movie will take a long time to render. Use **Quick Preview** and **Compare Presets**
on a short, visually complex clip first to confirm the look before committing to a
full run.

---

## 9. Newer AI Models Online (research snapshot, as of Aug 2026)

Not integrated into iw3 — would require a developer to add support. Recorded here as
a watch-list, since this research space is moving fast and should be re-checked
periodically.

**Sanity check:** even paid competitor tools (e.g. Owl3D) currently use the same
Video Depth Anything model iw3 already uses — VDA_L is not behind the curve.

### Depth models that beat VDA_L on benchmarks

| Model | Org | Released | Open weights? | Notes |
|---|---|---|---|---|
| Pixel-Perfect Video Depth | Academic | Jan 2026 | Yes | Beats VDA-L on long-video temporal-stability benchmarks (Bonn δ1 0.979 vs 0.959). Pixel-space diffusion, transformer-only. |
| GemDepth-VDA | Academic (arXiv 2605.10525) | May 2026 | Yes (per rankings) | Explicitly built as a refinement *on top of* VDA/Depth Anything V2, adding a "Geometry-Embedding Module" for camera-motion consistency — an upgrade path, not a from-scratch replacement. |
| Depth Anything 3 | ByteDance-Seed (VDA's own successor line) | Nov 2025 (Streaming variant Dec 2025) | Mostly Apache 2.0; largest "Giant" variants are CC BY-NC 4.0 (non-commercial) | DA3-Streaming targets very long video in under 12GB VRAM. Benchmarked mainly on single-image accuracy rather than video-temporal specifically. |
| DVD v1.1 ("Video Diffusion Models are Overqualified Depth Estimators") | EnVision Research | 2026 | Yes | Repurposes the Wan2.1 video-diffusion model as a depth estimator, single deterministic pass. Likely needs meaningfully more VRAM than VDA's lightweight ViT design (not confirmed numerically). |
| FFN ("Frame Forgetting Network") | Academic | June 2026 | Unclear/unconfirmed | Tops long-video benchmarks, purpose-built for hours-long video without flicker via test-time adaptation. Very new/bleeding-edge — worth revisiting later. |

Community benchmark tracker used for comparison (not an official leaderboard, treat
scores as directional): [AIVFI/Video-Depth-Estimation-Rankings](https://github.com/AIVFI/Video-Depth-Estimation-Rankings-and-Stereo-Video-Conversion-Rankings).

**"AnyDepth" — watch-list only, not integrated (checked 2026-08-14):** an open
discussion on the nunif GitHub ([Discussion #631](https://github.com/nagadomi/nunif/discussions/631),
started Feb 25, 2026) proposes a replacement decoder ("Simple Depth Transformer")
for the standard DPT decoder DA3-family models use — claimed 85%+ smaller decoder,
smoother gradients, and a "Spatial Detail Enhancer" specifically aimed at reducing
edge blur/halo artifacts (directly relevant territory to the crown-artifact
investigation, if it panned out). **Status: still just a discussion/proposal, not
code.** The project maintainer (nagadomi) directly cast doubt on whether the claimed
DA3 results even used the real DA3 pretrained weights vs. a retrained variant — so
applicability to `da3mono-large` (the checkpoint behind `Any_V3_Mono`/`Any_V3_Mono_01`)
is explicitly unconfirmed even by him. Nothing to act on now; worth re-checking
periodically since it targets exactly the kind of artifact this project has been
chasing.

### Stereo/warp+inpaint replacements (bigger shift happening here)

A wave of 2025-2026 research proposes replacing iw3's "warp the image, then inpaint
the holes" approach with AI that generates the second-eye view more holistically.

| Model | Org | Released | Open weights? | Notes |
|---|---|---|---|---|
| **M2SVid** | **Google Research** (3DV 2026) | Weights released Mar 2026 | Yes — `google-research/m2svid` | Strongest candidate found: #1 on the Stereo4D benchmark, preferred 2.6x more often than runner-up in blind viewer tests, and 6x faster than typical diffusion approaches of this kind. |
| StereoPilot | Kuaishou/Kling team | Dec 2025 | Yes — check license (corporate origin) | Explicitly built to replace the "Depth-Warp-Inpaint pipeline" (iw3's current approach) with one feed-forward pass, no explicit depth map needed. |
| StereoWorld | Academic (CVPR 2026) | Dec 2025 | Yes | ~Half the visible error (LPIPS) of the earlier well-known StereoCrafter baseline. |
| StereoCrafter / StereoCrafter-Zero | TencentARC | 2024/2025 | Yes | The original influential diffusion-based approach; now the baseline the newer entries above are measured against and beat. |

**Tradeoff:** these diffusion-based methods can invent plausible hidden content
(rather than stretching/smearing existing pixels), which generally looks more
natural — but nearly all are built on large video-diffusion backbones (Stable Video
Diffusion, Wan2.1), meaning much higher VRAM use and slower processing than iw3's
current small custom-trained `mlbw_l2_inpaint`/`forward_inpaint` networks.

**If iw3 ever adds a newer option, M2SVid is the one most worth watching** — public
code and weights, clear benchmark win, relatively fast for this category of model.

### Reality check on actually adopting one of these (checked M2SVid specifically)

Even with a top-tier GPU (32GB VRAM is not the blocker), swapping in an outside
research model is a real multi-day software project, not an afternoon experiment:

- **CUDA/PyTorch version mismatch risk:** M2SVid's documented setup specifies Python
  3.10.6, PyTorch 2.0.1, CUDA 11.8 — nearly 3 years old, predating RTX 5090 (Blackwell
  architecture, needs CUDA 12.8+) entirely. Would require deviating from the
  documented setup and installing newer versions, adding real risk.
- **Linux-only tooling:** built around conda + bash scripts, no native Windows
  support — would need to run inside **WSL2**, an extra layer of setup.
- **Tested only on datacenter GPUs** (A100/H100), never validated against consumer
  cards.
- **Chained dependencies:** needs a second project (DepthCrafter) plus separate
  checkpoints (Hi3D) on top of its own 8.5GB of weights — three separate
  setups chained together.
- **Trained at 512×512** — real movie resolution needs an extra tiling step.

**Conclusion:** genuinely possible given the hardware, but realistically a standalone,
isolated experiment (never touching the working iw3 install) to prove it even runs
before considering the much bigger separate project of integrating it into iw3 itself.

---

## 10. Deep Code Audit — Undocumented Existing Settings (2026-08-14)

A full pass through `iw3/utils.py`'s argument parser, `mapper.py`,
`convergence_estimator.py`, and `face_convergence_estimator.py`, cross-checked
against everything already written above, to surface real settings that exist in
the code but were never covered in this doc. (Web research on external
state-of-the-art techniques the same day mostly *confirmed* what Section 4/5/9
already say — professional "depth budget" comfort theory and the M2SVid/StereoPilot
diffusion-inpainting watch-list — so nothing new from that side; the real find was
in the code itself.)

### `--mapper` / `--mapper-type` — the depth→disparity curve underneath Foreground Scale

This is the biggest gap. **Foreground Scale (`-3..3`) is just a friendly integer
index into a curated curve** — `--mapper` is the actual underlying curve-selection
mechanism, and it supports finer/different control than the -3..3 integer scale
exposes:

- **Curve families** (`iw3/mapper.py`, `resolve_mapper_function`):
  - `mul_1/2/3` / `inv_mul_1/2/3` — parametrized softplus curves for **relative
    depth** models. `mul_*` pushes foreground out (mild→strong); `inv_mul_*` is the
    inverse (compresses foreground, expands background instead).
  - `shift_045/06/08/14/20/30` — reprojects relative depth into a pseudo-metric
    distance space, effectively shifting the "near clip" distance. `045` = aggressive
    near-field expansion, `30` = gentle. A genuinely different remapping mechanism
    from the softplus curves, not just a stronger/weaker version of the same thing.
  - `div_1/2/4/6/10/25` — for **metric depth** models (e.g. `VDA_Metric_L`), a
    hyperbolic remap parametrized by an assumed virtual-camera-baseline constant.
    `div_6` is what this family resolves to by default for metric models.
- **Chaining/blending syntax, CLI-only, not in the GUI:** `--mapper name1:name2`
  chains two curves sequentially; `--mapper nameA+nameB=weight` blends between two
  curves by a 0.0–1.0 weight (e.g. `--mapper mul_2+mul_3=0.5`). This is real,
  functioning fine-grained control beyond what the 7-step Foreground Scale dropdown
  can reach — worth experimenting with if the discrete Foreground Scale steps ever
  feel too coarse for a specific shot.

### `--warp-steps` — manual override for high-divergence multi-step warping

**Only relevant to `row_flow_v3`/`row_flow_v2`, not `mlbw_l2_inpaint`/`forward_inpaint`
— i.e. not relevant to this project's current method choice unless that changes.**
Those row_flow models were trained only up to a max divergence (5.0 for v3, 2.5 for
v2); past that, iw3 auto-splits the warp into multiple smaller steps
(`calc_auto_warp_steps`) rather than doing one out-of-distribution single-shot warp.
`--warp-steps` lets you manually force *more* steps than the auto-heuristic picks —
free extra smoothness at high divergence on hardware where speed isn't a constraint,
if `row_flow_v3` is ever used instead of the inpainting methods this project prefers.

### `--hdr-to-sdr` — proper tone-mapping pass before depth estimation (real gap for this project's HDR/Dolby Vision workflow)

Distinct from `--preserve-dowi` (which is already covered above and handles keeping
Dolby Vision RPU metadata *in the output*). `--hdr-to-sdr` runs an actual PQ/HDR10/
HDR10+/HLG **tone-map down to normal SDR brightness before conversion**, while
keeping 10-bit precision (less banding than a naive 8-bit SDR conversion). It's a
no-op automatically if the source is already SDR.

**Why this matters specifically here:** depth models are trained on SDR imagery.
Feeding a PQ-curve HDR source straight into the depth estimator without this flag
can skew depth estimation in very bright/dark regions, since raw pixel values don't
map to the luminance the model expects. Given this project has directly worked with
Dolby Vision source files (dovi_tool, HDR10+ tooling, DV Profile 5 test files), this
is a concretely testable lever, not a hypothetical one — worth a controlled test:
same HDR scene, same everything else, `--hdr-to-sdr` on vs. off, check whether depth
estimation quality/stability changes on bright highlights or crushed shadows.

### Convergence Mode internals — what `sod_v1` and `face_detect` actually do under the hood

Both docs above recommend `constant` over `sod_v1` for whole-movie comfort (Section
3) — this is *why*, mechanically, plus a new caveat for `face_detect`:

- **`sod_v1`** (`convergence_estimator.py`): uses a small salient-object-detection
  model to get a saliency mask, then places the convergence point using the salient
  region's own 10th/90th-percentile depth spread — your `--convergence` 0-1 value
  gets mapped into a **3x-expanded range** around that object's depth, not a literal
  0-1 across the whole scene. If no salient pixels are found in a frame, it silently
  falls back to flat `0.5`.
- **`face_detect`** (`face_convergence_estimator.py`): **not a neural face model —
  a plain OpenCV Haar cascade** (frontal-face-only, sensitive to angle/lighting).
  When no face is detected in a frame, it silently falls back to a Gaussian
  center-weighted depth average (assumes the subject is near frame-center) rather
  than any face-aware logic.
  **New risk worth knowing:** this means a shot with a profile face, a small/distant
  face, or momentary occlusion can silently flip between "locked onto the actual
  face" and "assumed center-of-frame" from one frame to the next — a real,
  previously-undocumented source of convergence jumps specifically for `face_detect`
  mode (the "Action" preset's convergence mode). Worth watching for on any render
  using `face_detect` with faces that aren't consistently front-on and well-lit.

### Gotcha: "Output Size Limit" can silently turn Full SBS into Half SBS (2026-08-14)

Symptom that triggered this: a 3840×2160 (4K) source produced a **3840×2160** final
frame in *both* Full SBS and Half SBS — toggling Half SBS made no visible difference.

Root cause, traced through `postprocess_image()` in `utils.py`:

- **Full SBS** builds each eye at full source width, then concatenates them —
  for a 3840-wide source that's a raw **7680×2160** canvas (double width).
- **Half SBS** deliberately resizes each eye to *half* width *before*
  concatenating, landing on **3840×2160** (matches source width) — this is
  intentional, the standard delivery-friendly packing.
- **After packing**, both paths pass through one more resize gated by the GUI's
  **"Output Size Limit"** dropdown (`--max-output-width` / `--max-output-height`).
  If that's set to `3840x2160` (one of the presets, and a natural pick for a 4K
  target), it force-clamps *any* wider frame down to width 3840.
- **"Keep Aspect Ratio" defaults to OFF.** With it off, the clamp doesn't
  proportionally shrink the frame — it just crushes the width down to fit while
  leaving height alone. For Full SBS's 7680-wide canvas, crushing width to 3840
  while keeping height at 2160 is *mathematically the same squeeze* Half SBS
  performs on purpose. Result: Full SBS gets silently converted into something
  pixel-dimension-identical to Half SBS, so the toggle appears to do nothing.

**Fix:** the Output Size Limit and the SBS packing mode have to agree on intent.
- Want true Full SBS (full per-eye detail, double-width frame)? Set Output Size
  Limit to `7680x2160`, or blank (blank = no cap, keep native size).
- Want Half SBS at a 4K delivery frame? `3840x2160` is correct as-is — no change
  needed, and Half SBS should still be checked so the squeeze happens cleanly
  per-eye via a proper bicubic resize, rather than accidentally depending on the
  size-limit's cruder post-pack crush to get there.

Given this project has no VRAM/speed constraint, Full SBS (max per-eye detail)
is the better match for the "match professional Blu-ray 3D" goal — so Output
Size Limit should be `7680x2160` or blank, not `3840x2160`.
