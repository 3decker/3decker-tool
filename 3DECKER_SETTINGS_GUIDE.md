# 3DECKER — iw3 Settings Guide

## About this guide

This is a complete, plain-language reference for **every setting in the iw3 GUI**
(`iw3-gui.bat`) — organized into the exact same sections you see on screen, so you
can find a setting here by looking at the same tab/group heading it appears under
in the app. Every explanation below is pulled directly from that setting's own
tooltip in the app (hover over any control to see the short version live) —
expanded into full sentences here, not reworded from memory or guessed.

For each setting you'll generally find:
- **What it does** — plain-language explanation
- **Pros** / **Cons** — the real trade-offs, where the tooltip spells them out
- **Recommended** — the app's own suggested starting point
- **Values** — the real dropdown choices or numeric range, where relevant

If a setting's tooltip is short and doesn't break out into pros/cons/recommended
separately, this guide keeps it just as short rather than padding it out with
invented detail.

This is a *reference*, not a tutorial — for the basic step-by-step workflow, see
`3DECKER_README.md`.

---

## Table of Contents

1. [Top Toolbar](#top-toolbar)
2. [File & Batch Options](#file--batch-options)
3. [Stereo Generation](#stereo-generation-tab)
4. [Dual-Pass Depth Blend](#dual-pass-depth-blend-tab)
5. [Video Filter](#video-filter-tab)
6. [Video Decoding](#video-decoding-tab)
7. [Video Encoding](#video-encoding-tab)
8. [Processor](#processor-tab)
9. [Post-Processing](#post-processing)
10. [Standalone Tools](#standalone-tools-tab)
11. [Bottom Bar](#bottom-bar)

---

## Top Toolbar

The strip of controls above the main settings area. These stay visible no matter
which tab you're on.

#### Preset (Load / Save / Delete)

Your own saved settings combinations, separate from the three built-in Quick
Presets below. Pick a name in the dropdown and click **Load** to apply it,
**Save** to store your current settings under that name, or **Delete** to remove
it. (This control doesn't have its own detailed tooltip in the app — its
buttons are self-explanatory.)

#### Movie (Quick Preset)

**What it does:** Applies a full combination of settings tuned for subtle,
comfortable 3D suited to movies — 3D Strength 2.0, Convergence Plane 0.5 in
automatic (sod_v1) mode, Foreground Pop off.
**What it's for:** a restrained look favoring comfort over impact, closer to how
a professional stereographer would grade a typical dialogue-driven film.

#### Action (Quick Preset)

**What it does:** Applies a combination tuned for strong pop effects in
action/VFX-heavy scenes — 3D Strength 3.0, Convergence Plane 0.5 in automatic
(face_detect) mode, Foreground Pop 0.5.
**Trade-off:** a more aggressive, attention-grabbing look, at some cost to
comfort over long viewing sessions.

#### 3DECKER Preferred (Quick Preset)

**What it does:** Applies this project's own confirmed-best settings combination
across depth, divergence, convergence, refinement, stability, and flicker
smoothing at once, drawn from a real tuning session. Specifically: Depth Model
Any_V3_Mono_01, Divergence 2.5, Convergence 0.5, Depth Detail Refinement on
(strength 1.0), Object Stability on (strength 0.3, Flat-Area Boost 0, Edge
Protection 0, Max Shift off), Scene Detection on, and Auto EMA by Scene Length
on using the Nagadomi_Reference table.
**Side effect:** because it turns on Auto EMA by Scene Length, it also greys out
Flicker Reduction's Decay Rate/Buffer/Genre Preset fields, the same as checking
"Auto EMA by Scene Length" yourself would.

#### Compare Presets...

**What it does:** Renders the same short test clip using two or more of your
saved presets, then joins the results back-to-back into a single comparison
video — so you can directly see the difference between settings combinations
before committing to a full-length conversion.

#### Copy Command

Copies the equivalent command-line invocation for your current settings to the
clipboard. (No detailed tooltip in the app.)

#### Language

**What it does:** Switches the app's own interface language. (Choices are
populated from this project's bundled translations.)

#### Layout

**What it does:** Chooses how the 100+ conversion settings are organized on
screen.
**Values:**
- **Tabbed** (default) — groups settings into 7 category tabs (Stereo
  Generation, Dual-Pass Depth Blend, Video Filter, Video Decoding, Video
  Encoding, Processor, Standalone Tools) so you only see one category at a
  time.
- **Single Page** — shows all 7 category groups at once on one scrollable
  page, so nothing is hidden behind a tab click.
**Cons:** Single Page needs more scrolling/screen space to see everything at
once; Tabbed hides other categories until you click their tab.
**Recommended:** Tabbed for a smaller, less cluttered window; Single Page if
you'd rather see every setting at once and don't mind scrolling.
**Note:** switches instantly — no restart needed.

#### Zoom

**What it does:** Scales the whole app's text and control size up or down, for
readability on a high-DPI display or just personal preference. Separate from
Windows' own system display scaling, which this app already handles on its
own.
**Values:** 80% (smallest) to 200% (largest); 100% is the original/default
size.
**Cons:** at the largest sizes, some tabs may need more scrolling to see every
setting; Single Page layout scrolls to fit automatically, Tabbed may need a
taller window.
**Recommended:** leave at 100% unless text is hard to read; 125%-150% is a
reasonable middle ground on a high-resolution display.
**Note:** applies immediately — no restart needed.

#### Check for Updates

**What it does:** Checks whether the original upstream nunif project
(github.com/nagadomi/nunif) has new commits that aren't in this fork yet, and
shows you what they are.
**Pros:** lets you see what's changed upstream without any risk to your setup
or this session's own customizations (RIFE, Z-Splat, HDR reinjection,
subtitle muxing, StereoMode tagging, etc.).
**Cons:** read-only — runs `git fetch` plus a comparison only. It never runs
pull/merge/reset, so nothing is ever applied automatically; this button
cannot update anything by itself, and upstream commits could conflict with
this fork's own customizations if applied later.
**Recommended:** safe to click any time — it only reads and reports, never
changes anything.

#### Run Update

**What it does:** Actually runs the real `update.bat` script from inside the
app — the same script you'd otherwise have to find and double-click outside
the app — updating Python packages, downloaded models, and the source code
together in one operation.
**Why it's separate from Check for Updates:** that button only checks and
reports what's different upstream and changes nothing on disk; this button
actually applies an update to real files.
**Cons:** a real, somewhat time-consuming operation (package downloads, model
downloads, a source pull) with no undo — a confirmation dialog appears before
anything runs, and, as Check for Updates already warns, an upstream source
update could in principle conflict with this fork's own customizations.
Disabled while a conversion (or other background job) is running, so
packages/source can't change out from under a job that's using them.
**Recommended:** use it when you actually want to apply an update you already
know about (e.g. from Check for Updates) — not as a routine/automatic click.

---

## File & Batch Options

The row of checkboxes just under the input/output file pickers, always visible.

#### Resume

**What it does:** Skips a file entirely if the output it would create already
exists.
**Pros:** lets you stop a big batch job partway (or have it crash/get
interrupted) and restart later without wasting time redoing files you already
finished.
**Cons:** only checks whether a file with the expected NAME exists, not
whether it's actually complete or correct — a partial/corrupted leftover file
gets treated as "done" too.
**Recommended:** on for batch folders and long jobs. Turn off only if you
specifically want to force-redo everything.

#### Process all subfolders

**What it does:** When the input is a folder, also reaches into its
subfolders instead of only converting files sitting directly inside it.
**Cons:** if unrelated files (extras, samples, trailers) live in subfolders
you didn't mean to include, they'll get converted too — there's no
per-folder include/exclude list.
**Recommended:** on if you organize your movies into subfolders; off if your
input folder mixes in stuff you don't want touched.

#### Skip Error

**What it does:** If a file causes an error during batch processing, notes it
and moves on to the next file instead of stopping the whole batch. Also skips
files that errored on a previous run, so they aren't retried every time.
**Cons:** a genuinely broken/corrupt file just gets silently skipped rather
than fixed — check the log if a file seems to be missing from your results.
**Recommended:** on for large batches, so one bad file doesn't halt an
overnight job. Off if you'd rather the job stop immediately so you notice a
problem right away.

#### EXIF Transpose

**What it does:** Some cameras/phones save a photo already correctly oriented
for viewing but store the actual pixel data sideways/upside-down, with a
hidden tag telling viewers how to rotate it for display. This applies that
rotation before conversion so the 3D effect is built for the image the way
you actually see it, not the raw sideways file.
**Recommended:** on (default) for virtually all real photos. Only turn off if
you've confirmed your specific images have no EXIF tag or it's already
wrong, since that's an unusual case.

#### Add metadata to filename

**What it does:** Encodes your current settings (depth model, 3D strength,
convergence, edge options, etc.) into the output filename as short
abbreviated tags, and also writes them into the video's own comment
metadata.
**Pros:** lets you tell which settings made which file just by looking at the
filename later, compare two versions of the same clip, and lets Resume
correctly match an in-progress job back up after an interruption.
**Cons:** filenames get noticeably longer and less readable at a glance (a
string of abbreviations rather than just the movie's name).
**Recommended:** on, unless you strongly prefer short/clean filenames and are
keeping track of your own settings some other way.

#### Image Format

**What it does:** The file format used when converting still images (has no
effect on video jobs, which always use your Video Format setting instead).
**Values:**
- **png** — lossless, larger files, no quality loss ever — best if you'll
  edit/reprocess the result later.
- **jpeg** — lossy compression, much smaller files, a small amount of quality
  loss (usually invisible at normal viewing sizes) — best for
  sharing/storage space.
- **webp** — modern format, similar quality to jpeg at a smaller file size,
  but less universally supported by older software/devices.
**Recommended:** png if disk space isn't a concern or you might reprocess the
image later; jpeg for everyday viewing/sharing where file size matters.

---

## Stereo Generation Tab

The core 3D-conversion controls: how strong the effect is, which AI does the
work, and how the two eye views get built.

#### 3D Strength (Divergence)

**What it does:** The master strength of the whole 3D effect — how far things
shift between the left/right eye.
**Pros/Cons:** higher = more dramatic depth, but more edge artifacts. Lower =
subtler, cleaner.
**Recommended:** 2.0-3.0 for most movies.
**Values:** editable, common choices 5.0 / 4.0 / 3.0 / 2.5 / 2.0 / 1.0.

#### Convergence Plane (mode dropdown)

**What it does:** How the "screen depth" (the point that looks like it's
exactly at the screen surface, with everything else popping toward or
receding from it) is chosen.
**Values:**
- **constant** — you set one fixed position with the value box, and it never
  moves for the whole video.
- **sod_v1** — an AI model automatically re-picks a focus point every frame,
  based on the most visually important subject.
- **face_detect** — same idea, but automatically centers on detected faces
  specifically, ignoring the value box.
**Cons of sod_v1/face_detect:** every time the convergence point moves — even
smoothed — your eyes have to physically readjust their focus angle to keep
the image comfortable to view. Because sod_v1 re-evaluates every frame, it
can drift even WITHIN a single unbroken shot (someone shifts position, the
camera pans slightly), which is closer to "constantly reacting" than how a
real stereographer works — they hold convergence steady within a shot and
only step it to a new value at cuts. Real testing on this project found
constant produced steadier, more comfortable results on most typical film
content for exactly this reason.
**When sod_v1/face_detect genuinely help:** a push/pull "reveal" shot where
the camera moves from a tight close-up to a wide shot within one continuous
take — a fixed constant value structurally can't be right for both ends of
that move, but sod_v1 can track it.
**Recommended:** constant for most content — steadier and closer to real
stereographer practice. Reserve sod_v1/face_detect for content dominated by
continuous push/pull reveal shots.

#### Convergence Plane (value box)

**What it does:** Where the "screen depth" sits, from 0 (everything pops out
toward you) to 1 (everything sits behind the screen). Only used directly when
mode is "constant" — for sod_v1 it acts as a relative offset within the
detected subject's depth range.
**Recommended:** 0.5 as a balanced starting point.
**Values:** 0.0 to 1.0.

#### Convergence Smoothing

**What it does:** Only affects sod_v1 / Face Detect convergence modes.
Controls how quickly the automatic convergence point reacts to scene
changes.
**Pros/Cons:** higher = smoother but slower to react. Lower = more
aggressive/dynamic, reacts faster but may jitter more. 0 = no smoothing at
all.
**Values:** 0.95 / 0.9 / 0.75 / 0.5 / 0.25 / 0.

#### Your Own Size (IPD Offset)

**What it does:** Also called IPD Offset (interpupillary distance — the gap
between your own eyes). Widens or narrows the simulated eye spacing used to
build the 3D effect, separate from the main 3D Strength slider.
**Values:** 0 = average adult spacing (the default assumption). Positive
values simulate wider-set eyes, which slightly increases the 3D pop for
people with a wider-than-average IPD. Range is -10 to 20.
**Cons:** this is a minor personal-comfort tweak, not a real substitute for
3D Strength — pushing it far from 0 can make the depth feel physically
"wrong" even if stronger.
**Recommended:** leave at 0 unless you specifically know your own IPD differs
a lot from average and have noticed a comfort difference.

#### Synthetic View

**What it does:** Decides which eye view(s) the AI actually generates from
your original flat image.
**Values:**
- **both** — generates BOTH left and right eyes fresh from the center image —
  most natural, since neither eye is just the untouched original.
- **right / left** — keeps the original image exactly as one eye, and only
  generates the other — faster (half the warping work), but can look
  slightly less balanced since one eye is "real" and the other is
  synthesized.
**Recommended:** both, for the best-looking result. right/left mainly useful
if you're very tight on processing time.

#### Method

**What it does:** Which AI model/technique actually builds the second eye
view from your depth map. One of the biggest single quality/speed decisions
in the whole app.
**Values:**
- **row_flow_v3** — fast, AI-based, a solid all-rounder — the default.
- **row_flow_v3_sym** — same family, but shifts BOTH eyes symmetrically
  outward from the original center image instead of keeping one eye as the
  unwarped original — can look more balanced across both eyes.
- **row_flow_v2** — an older generation of the row_flow model, kept for
  compatibility/comparison; row_flow_v3 generally looks better at the same
  speed.
- **mlbw_l2 / mlbw_l4** — a more advanced, multi-layer AI warp. l4 is the
  deeper/more capable version of l2 — handles stronger 3D strength and
  complex scenes with fewer artifacts, at extra GPU time/memory cost.
- **mlbw_l2s** — a smaller/lighter version of mlbw_l2 — faster and lower
  VRAM, at some quality cost versus the full mlbw_l2. Useful on lower-VRAM
  GPUs.
- **mlbw_l2_inpaint / forward_inpaint / monobw_inpaint** — any of the above
  families, plus an AI inpainting pass that fills in the hidden area behind
  objects instead of stretching/smearing it — real extra time cost, but
  noticeably cleaner edges around foreground objects.
- **forward_fill** — a simple, non-AI method — just warps pixels forward and
  fills gaps with a basic algorithm, no learned model involved. Fastest, but
  the roughest edges.
- **forward_splat_fill** — same non-AI forward-warp family as forward_fill,
  but where two pixels land on the same spot, it smoothly blends them
  (nearer one weighted more) instead of the nearer one fully overwriting the
  other — softer, less jagged edges than forward_fill. No learned model
  involved, but NOT free of GPU cost: it holds several full-frame accumulator
  tensors in memory per batch, so its VRAM use scales up directly with Depth
  Batch Size (unlike the inpaint methods, which don't) and can run even a
  high-VRAM GPU out of memory on demanding footage at Depth Batch Size 2 or
  higher. Confirmed real fix if you hit an out-of-memory error here: drop
  Depth Batch Size to 1 and turn on Low VRAM (see below).
- **monobw** — a simpler, lighter backward-warp method than the mlbw family —
  a faster middle ground when mlbw is too slow but forward_fill's quality
  isn't good enough.
**Recommended:** mlbw_l2_inpaint or forward_inpaint for the best quality on a
real GPU; row_flow_v3_sym or mlbw_l2s if you need more speed or have limited
VRAM.

#### Splat Blend Temperature

**What it does:** Only matters for Method forward_splat_fill — how sharply
that method's depth-weighted blend decides which of two colliding pixels
wins.
**Values:** higher = sharper cutoff, closer to forward_fill's old
hard-overwrite behavior (whichever pixel is nearer the camera wins almost
completely); lower = smoother, more even blending between the two competing
pixels.
**Cons:** brand new, and not yet tuned against real footage — unlike most
other numeric settings in this app, there isn't yet a body of real-world
testing behind these numbers.
**Recommended:** start at the default (50.0) and only adjust it if
forward_splat_fill's results look wrong to you at that default.

#### Inpainting Model

**What it does:** Only matters for the `*_inpaint` Method variants
(forward_inpaint, mlbw_l2_inpaint, monobw_inpaint) — picks which AI model
fills in the hidden area behind objects that the 3D shift reveals.
`light_inpaint_v1` is the only model included out of the box, and is what
this stays on for almost everyone. Extra models only appear here if you've
manually added entries to `iw3/inpaint_models.yml` (an advanced/optional
customization, not needed for normal use).
**Recommended:** leave on light_inpaint_v1 unless you've specifically
installed an alternative model and know why you want it.

#### Inpaint Overlap Frames (Pre / Post)

**What it does:** Only matters for the video inpainting methods. How many
extra frames before (Pre) or after (Post) each processed chunk get fed to the
inpainting model purely for context, so the filled-in background doesn't
flicker or change texture at chunk boundaries.
**Values:** 0 = no overlap (fastest, but a visible seam/flicker can appear
where chunks join). 3 (default) = a small buffer that's usually enough to
hide the seam.
**Cons:** higher values cost real extra processing time per chunk boundary,
with diminishing returns past a few frames.
**Recommended:** default (3) for both; raise only if you actually see a
flicker at chunk boundaries.

#### Inpaint Mask Dilation (Inner / Outer)

**What it does:** Only matters for `*_inpaint` methods.
- **Inner** — grows the "needs filling in" mask slightly INTO the
  foreground object's own edge, trimming a thin sliver off the object so the
  inpainted background blends in more smoothly instead of stopping at a hard
  line.
  **Cons:** too high a value visibly shrinks/erodes foreground objects at
  their edges.
  **Recommended:** 0 (default); raise to 1-2 only if you see a thin,
  obviously wrong-colored halo clinging to the outline of foreground
  objects.
- **Outer** — grows the "needs filling in" mask outward INTO the background,
  giving the inpainting model a wider area to work with around each object.
  **Cons:** too high a value makes the model regenerate more background than
  it needs to, which costs a little quality/consistency versus the small
  sliver of real original background right at the object edge.
  **Recommended:** 0 (default); raise to 1-2 only if you still see leftover
  smearing/stretching right behind foreground objects after trying Inner
  dilation.
**Values:** 0 / 1 / 2 for both.

#### Inpaint Max Width

**What it does:** Caps the width the inpainting model processes at,
downscaling internally if the source is wider than this, to save VRAM/time
on large/4K video.
**Cons:** lowering it can make the inpainted (filled-in) areas slightly
softer/less detailed than the rest of the frame, since that step ran at a
lower resolution.
**Recommended:** leave blank (no limit, best quality) unless you're running
out of GPU memory or need to speed up a very high-resolution job — then try
1920 first.

#### Stereo Processing Width

**What it does:** Only used by the row_flow_v3/row_flow_v3_sym/row_flow_v2
methods. Resizes the image to this width specifically for the eye-generation
step — separate from Depth Model's own internal resolution and separate from
your final output resolution.
**Cons:** lowering it trades away fine detail in the warp itself for speed;
the final output is still your source resolution, but the 3D shift
calculation behind it was done coarser.
**Recommended:** Default (uses the source width, best quality). Try 1920 or
1280 only if you need more speed and can accept a small quality tradeoff.
**Values:** Default / 1920 / 1280 / 640.

#### Depth Model

**What it does:** Which AI model looks at your image/video and estimates
what's near vs far. The foundation everything else builds on — probably the
single most important choice in the whole app.
**Values (families):**
- **VDA_\* (Video Depth Anything)** — built specifically for video — has real
  memory across frames, so depth stays steady/flicker-free without needing
  extra smoothing settings. Best choice for movies/video by default.
- **Any_V2_\* / Any_V3_\* / Distill_Any_\*** — single-image models — often
  sharper/more detailed on a single photo, but have NO memory between
  frames, so used on video they can flicker unless you also turn on EMA
  smoothing and/or Object Stability.
- **`*_Metric` variants** — estimate real-world distances (meters) instead of
  a relative near/far scale — a specialized option, not needed for normal
  stereo conversion.
- **Size suffix (_S/_B/_L, small/base/large)** — bigger = noticeably better
  quality, but slower and more VRAM — roughly proportional to size, not
  free.
**Recommended:** a VDA_* model for video (steadiest results with the least
fiddling); an Any_V3_* model for single images or when you want maximum
per-frame detail on video and are willing to tune EMA/Object Stability
yourself.

#### Depth Resolution

**What it does:** How much detail the depth model works with internally (its
short-side resolution in pixels). "Default" uses ~392.
**Cons:** higher = finer depth detail but more VRAM/time — roughly squares
the cost as you increase it.
**Recommended:** Default for most content; try 448-512 if you have VRAM to
spare and want finer depth detail.
**Values:** Default / 512 (editable).

#### Limit to source

**What it does:** Safety cap only: if your typed Depth Resolution is HIGHER
than the source video's own resolution, this brings it back down to match
the source instead of wasting time asking for detail that doesn't exist. It
never raises a lower value up.
**Recommended:** on.

#### Foreground Scale

**What it does:** Reshapes the depth curve for the WHOLE image (-3 to 3) —
but as a redistribution, not a simple push. It's a trade-off dial, not a
"both ends get better" dial: whichever end you push toward gains
separation/detail, the OTHER end gets flattened. It cannot give a punchier
foreground and a deeper background at the same time.
**Values:** 0 = the depth model's own natural curve, unchanged. Positive
values sharpen/spread out the FOREGROUND (more roundness/separation among
near objects) while flattening the background. Negative values do the
reverse — sharpen/spread out the BACKGROUND (more sense of depth into the
distance) while flattening the foreground.
**Cons:** pushed to an extreme in either direction, the opposite end of the
scene will look noticeably flat/compressed — this is inherent to how the
curve works, not a bug.
**Recommended:** 0 for natural, unexaggerated depth. Try a modest negative
value (-0.5 to -1.0) if you want a more immersive/deep-feeling background; a
modest positive value (+0.5 to +1.5) for a more sculpted, punchy foreground.
If you want MORE separation on both ends at once instead of trading one for
the other, raise 3D Strength (Divergence) instead — it scales both ends up
together rather than redistributing between them.
**Values:** -3 to 3.

#### Edge Fix (X / Y edge dilation)

**What it does:** Smooths the depth map right at object silhouettes BEFORE
the 3D shift is applied, reducing the halo/distortion artifacts that show up
around foreground edges.
**Important interaction #1:** this runs BEFORE 3D Strength (Divergence) sees
the depth map, and does a FIXED amount of smoothing regardless of Divergence
— it does not scale itself up automatically. The same leftover roughness
becomes a small, maybe-invisible error at low 3D Strength, but a much
bigger, more visible one at high 3D Strength. Raise this when you raise 3D
Strength to keep the same level of edge protection (roughly: Divergence
~2.0-2.25 pairs with 2/1, Divergence ~3.0-3.5 pairs with 3/2).
**Important interaction #2:** this also runs BEFORE Depth Resolution's
upscale, using a fixed-pixel-size brush on the smaller, not-yet-upscaled
depth map. That means the same number here covers a LARGER proportion of the
frame at a lower Depth Resolution than at a higher one — re-check this any
time you change Depth Resolution, not just when you change 3D Strength.
**Cons:** too high smooths away real fine depth detail near edges, not just
artifacts.
**X value:** horizontal edge smoothing strength (higher = more iterations =
smoother/wider protection, but more real detail lost near edges).
**Recommended:** 2 as a starting point, more if you raise 3D Strength or
lower Depth Resolution.
**Y value:** additional vertical edge smoothing strength, on top of X (left
blank = same as X, fully symmetric). Vertical seams usually matter less for
a left/right eye shift, so this is typically kept lower than X.
**Recommended:** 1 as a starting point (paired with X=2).
**Values:** X: 0-4. Y: blank, 0, 1, or 2.

#### Depth Anti-aliasing

**What it does:** Smooths small jagged/staircase artifacts in the depth map
using a dedicated AI model, without changing the actual depth values much.
Only available for certain depth models (grayed out otherwise).
**Recommended:** on, when available — minor cost, generally cleaner result.

#### Depth Detail Refinement

**What it does:** Cleans up noise WITHIN each depth frame using
edge-preserving smoothing (won't blur across real edges the way a plain blur
would). Different from Flicker Reduction, which smooths ACROSS frames over
time — this works on a single frame at a time. Cheap: no extra passes, no
extra models.
**Side note some users notice:** cleaner depth boundaries here can make the
finished 3D effect FEEL a bit stronger/more solid even though the actual
depth range doesn't change — noisy or fuzzy depth edges read as less
convincing 3D than clean ones at the same strength.
**Recommended:** on, safe to leave on for most content. Use the Strength box
to its right to control how much.

#### Depth Detail Refinement — Strength

**What it does:** How strong Depth Detail Refinement's cleanup is. 1.0 = the
original fixed strength this feature always used.
**Values:** higher = more smoothing reach (cleaner depth boundaries, but
risks softening genuinely fine depth detail if pushed too far); lower =
gentler, closer to doing nothing.
**Recommended:** 1.0 as a safe starting point; try 1.25-1.5 if you want a bit
more of the "cleaner/more solid 3D" effect this setting gives.
**Values:** 1.5 / 1.25 / 1.0 / 0.75 / 0.5 / 0.25.

#### Object Stability (experimental)

**What it does:** Gives a single-frame depth model (like Any_V3_Mono_01) some
of the same per-pixel steadiness over time that a video-native model (VDA_*)
has built in. Tracks real motion (optical flow) and blends each frame's
depth with the PREVIOUS frame's depth warped to where that content actually
moved to, reducing a specific object's depth flickering that Flicker
Reduction's overall-range smoothing can't touch.
**Note:** only works on the single-frame processing path — already active
for --low-vram, --debug-depth, VDA streaming models, and inpaint methods
(e.g. mlbw_l2_inpaint); has no effect otherwise.

#### Object Stability — Strength

**What it does:** How strongly to trust the motion-warped previous frame vs
the fresh per-frame depth (0-1). Automatically tapers down during
fast/unreliable motion regardless of this setting.
**Values:** 0.9 / 0.7 / 0.5 / 0.3.

#### Object Stability — Max Shift

**What it does:** Hard cap on how much depth is allowed to change for the
same pixel between two consecutive output frames (0-1 scale, same units as
depth value). Stops a single-frame spike from ever "popping," no matter how
strong the raw model's disagreement is.
**Recommended:** leave blank to disable (no cap, original behavior).
**Values:** blank, 0.01, 0.02, 0.05.

#### Object Stability — Flat-Area Boost

**What it does:** Extra smoothing specifically in areas the CURRENT frame's
own depth is flat (sky, walls, floors) — exactly the areas where flicker is
most visible and least likely to be real motion. 0 = no extra smoothing
(original behavior).
**Values:** 0.0 / 0.3 / 0.5 / 0.7.

#### Object Stability — Edge Protection

**What it does:** Reduces smoothing where the CURRENT frame has a strong,
real depth edge (an object's silhouette), so Object Stability's flicker
reduction doesn't smear or lag behind a moving object's outline. 0 = no
reduction (original behavior).
**Values:** 0.0 / 0.3 / 0.5 / 0.7.

#### Foreground Pop

**What it does:** Pushes ONLY the nearest pixels further toward the audience
(0=off, 1=strong), leaving the rest of the scene (midground/background)
completely untouched. Different from Foreground Scale, which redistributes
the WHOLE depth curve and always trades foreground against background.
**Cons:** pushed too high, the very closest objects can pop hard enough to
feel uncomfortable or clash with the frame edges (pair with Preserve Screen
Border to reduce that risk).
**Recommended:** 0 (off) for a restrained, professional look; 0.25-0.5 for
deliberate "poke at the audience" moments; reserve 0.75-1.0 for a genuinely
gimmicky effect.
**Values:** 0.0 to 1.0.

#### Foreground Divergence

**What it does:** Uses a SEPARATE 3D Strength (Divergence) value for only the
nearest 15% of pixels, independent of the main 3D Strength slider — lets you
dial the foreground up or down without touching the rest of the scene.
**Values:** leave blank to disable (nearest pixels just use the same 3D
Strength as everything else, the default). Set higher than your main 3D
Strength for an extra-punchy foreground; lower for a gentler one.
**Recommended:** leave blank unless you've specifically noticed the
foreground needs its own strength independent of the rest of the scene.

#### Background Pop

**What it does:** Pushes ONLY the farthest pixels further away from the
audience (0=off, 1=strong), leaving the foreground/midground completely
untouched. Mirror image of Foreground Pop, aimed at the opposite end of the
scene.
**Pros:** gives a deeper-feeling background without needing to redistribute
the whole depth curve (unlike negative Foreground Scale, which achieves a
similar deep-background feel but by taking separation away from the
foreground at the same time).
**Recommended:** 0.15-0.25 for a modestly more immersive background on most
content; go higher for content with a genuinely deep, expansive background
you want to emphasize (landscapes, wide establishing shots).
**Values:** 0.0 to 1.0.

#### Background Pop Coverage %

**What it does:** How much of the scene Background Pop treats as
"background" (default 15% = farthest 15% of pixels by depth).
**Cons:** larger values affect more of the scene, but push the transition
line further into the midground, where a sudden change in how much a
subject recedes is more likely to be noticeable as an odd "step" in the
depth.
**Recommended:** 15% (default) for a subtle, hard-to-notice effect; raise to
20-30% only if you want the effect to reach further into the scene and don't
mind it becoming more visible.
**Values:** 15 / 20 / 25 / 30 / 40.

#### Background Divergence

**What it does:** Uses a SEPARATE 3D Strength (Divergence) value for only the
farthest 15% of pixels, independent of the main 3D Strength slider.
**Values:** leave blank to disable (farthest pixels just use the same 3D
Strength as everything else, the default). Set lower than your main 3D
Strength to keep a busy/detailed background calmer and less prone to edge
artifacts; set higher for a deeper-feeling background.
**Recommended:** leave blank unless you've specifically noticed the
background needs its own strength independent of the rest of the scene.

#### Edge Repair

**What it does:** Final cleanup pass on the finished 3D image (applies no
matter which Stereo Method made it). Gently smooths a thin hairline right
around real depth edges to reduce fringing/ghosting residue left over from
the 3D warp. Cannot affect flat areas or anywhere without a depth edge.
**Values:** 0.0=off (default), 1.0=strongest.

#### Sharpen (Stereo Generation)

**What it does:** Enhances fine detail/sharpness in the FINISHED,
fully-rendered 3D picture itself (after Edge Repair, if that's also on) —
different from every other sharpness-adjacent setting in this app, which all
work on the DEPTH MAP instead (Depth Resolution, Depth Detail Refinement,
Dual-Pass Depth Blend). This is the only control that sharpens the actual
delivered image.
**How it avoids amplifying film grain:** classic unsharp-mask sharpening
(boost = original - blurred) applied flatly makes old scanned film grain
look like ugly speckling, because grain IS high-frequency detail to a naive
sharpener. This one measures how much real local detail/edge structure is
actually at each spot first (the same technique already used elsewhere in
this app to protect real depth edges from over-smoothing) and only applies
the boost where that's genuinely high — real edges and texture get
sharpened, flat or grain-only areas are left close to untouched.
**Pros:** makes fine detail (hair, texture, text) pop more in the final
image, safer on grainy/noisy source than a plain sharpen filter would be.
**Cons:** any sharpening pass can still exaggerate real compression artifacts
or genuinely fine noise that happens to look edge-like; if the image starts
looking harsh/over-crisp, lower the Strength box to its right.
**Recommended:** off (default) unless the finished 3D output looks a little
soft to you; start at the default 0.5 Strength and raise only if you still
want more.

#### Sharpen — Strength (Stereo Generation)

**What it does:** How strong the Sharpen effect is.
**Values:** higher = more pronounced detail boost at real edges/texture, but
also more risk of an over-crisp/harsh look or exaggerating real compression
artifacts.
**Recommended:** 0.5 (default) as a safe starting point.
**Values:** 0.0 to 1.0 (0.25 / 0.5 / 0.75 / 1.0 preset choices).

#### Flicker Reduction

**What it does (video only):** Smooths the depth map's overall near/far
SCALE over time, so the whole image's depth intensity doesn't subtly
pulse/breathe from frame to frame due to the AI model's own frame-to-frame
noise. Separate from Convergence Smoothing (which only exists in auto
convergence modes and smooths a different thing, the focus point) — this one
works identically no matter which Convergence Mode you're using. Uses the
Decay Rate and Buffer settings together.
**Recommended:** on for essentially all video, paired with Scene Boundary
Detection so the smoothing resets cleanly at real cuts instead of blending
across them.

#### Flicker Reduction — Decay Rate

**What it does:** How much weight past frames keep vs. the newest frame when
tracking the depth-map scale. Updates every single frame, not periodically.
**Values:** higher (0.9-0.99) = more resistant to momentary spikes (a hand
reaching toward camera, a flash) but slower to follow a genuine intentional
depth change (a push/pull shot). Lower (0.5-0.75) = reacts faster to real
change, but more exposed to the AI's own per-frame noise. 0 = no smoothing
at all.
**How to tell it's too low:** on a static, unmoving dialogue shot, the whole
image's depth intensity subtly pulses even though nothing is really moving
closer/farther — that's it reacting to noise instead of real change. Also
watch right after something briefly very close to camera passes through: if
the ENTIRE frame's depth (not just that object) visibly lurches, one outlier
frame yanked the whole scale.
**How to tell it's too high:** right after a hard cut or during a push/pull
zoom, depth looks noticeably flat for a beat before catching up to the real
new depth.
**Recommended:** pair with Buffer rather than tuning alone (e.g. Decay 0.98
pairs with Buffer 120 for scenes with real smearing/bleeding artifacts).
**Note:** greyed out when "Auto EMA by Scene Length" is checked, since that
picks its own per-scene Decay/Buffer instead — this value is still kept and
still used as the fallback before the first detected scene boundary.
**Values:** 0.99 / 0.95 / 0.9 / 0.75 / 0.5 / 0.

#### Flicker Reduction — Buffer

**What it does:** How many recent frames contribute to judging the current
near/far depth range, together with Decay Rate.
**Cons of too small:** a single outlier frame (something briefly very close
to the lens) can yank the whole normalization range around by itself, which
reads as sudden smearing/swimming/bleeding in the 3D effect even though no
pixel was literally smeared — it's the depth SCALE lurching, not a warp
artifact.
**Cons of too large:** slower to adapt to a genuine intentional depth change
within one continuous shot (a push/pull reveal).
**Values (paired with Decay Rate, tested/extrapolated from real measured
footage):** 30/0.9 is a reasonable middle default. If you see
smearing/bleeding artifacts, try 120 paired with Decay 0.98 — this dilutes
any single outlier frame's influence across many more frames. Scene Boundary
Detection resets this buffer at every real cut regardless of size, so even a
large buffer only fully engages during a film's longer continuous shots.
**Recommended:** measure your actual footage's typical shot length if
possible (Scene Boundary Detection can log real cut points) and size the
buffer around 30-35% of the MEDIAN shot length, rather than guessing from
genre — measured testing on this project found genre assumptions about
pacing were sometimes simply wrong.
**Note:** greyed out when "Auto EMA by Scene Length" is checked.
**Values:** 150 / 60 / 30 / 1.

#### Auto EMA by Scene Length

**What it does:** Automatically picks EMA Decay/Buffer per scene based on
that scene's own length, using a default table (one step per whole second,
0-20s+) — a short scene gets a smaller Buffer so the smoothing actually
finishes settling before the scene ends, instead of a Buffer sized for one
long continuous shot.
**Works two ways:**
- **With Automated Scene Batch:** tuned for its independent-scene processing.
  Applied before Scene Settings File, so anything that file sets explicitly
  (EMA included) still wins for scenes it covers.
- **Without Automated Scene Batch:** requires Scene Boundary Detection to be
  turned on instead (there are no scene boundaries to key off of otherwise —
  Start will refuse to run and explain this if it's missing). Re-picks EMA
  Decay/Buffer at every detected cut within the same continuous video,
  overriding the fixed Flicker Reduction/Buffer for each scene as it starts —
  the output filename gets an "_autoema<model>" tag instead of a fixed EMA
  number, since no single number was used throughout.
**Note:** while checked, Flicker Reduction's own Decay Rate/Buffer fields are
greyed out (values kept, not cleared — they come right back the moment you
uncheck this).

#### Auto EMA — Model/Table selector

**What it does:** Which Auto EMA by Scene Length table to use, matched to the
Depth Model. Nine tables are available:
- **3DECKER VDA_L** — for a real video depth model with its own
  frame-to-frame memory; needs only light smoothing on top. Buffer/Decay
  climb from 8/0.65 at 0-1s up to a flat 120/0.98 ceiling from 15-16s
  onward.
- **3DECKER Any_V3_Mono_01** — for a stills-only model with no
  frame-to-frame memory of its own (prone to visible "depth breathing"
  without help); uses roughly double VDA_L's Buffer at every scene length
  with a correspondingly higher Decay to compensate, flattening at 240/0.99
  from 15-16s onward.
- **Nagadomi_Reference** — a more conservative option not tied to a specific
  Depth Model, anchored to this tool's original author's own documented
  presets (`iw3/depth_scaler.py`): settles at Buffer 30/Decay 0.9 (nagadomi's
  own "strong" preset) for every scene 4 seconds or longer, never pushed
  higher, since Buffer is a real future-frame lookahead cost, not an
  abstract dial. **Recommended if you're seeing lag or smearing with the
  3DECKER tables on long scenes** and would rather stay within nagadomi's
  own tested range.
- **GEMINI AI / ChatGPT / Grok** — three tables suggested by different AI
  assistants for real A/B testing, at the user's own request. None of these
  were verified against nunif's own source code or documentation the way
  Nagadomi_Reference was — included as-is for comparison, not because
  they're confirmed correct. They request much larger lookahead Buffers than
  the other tables (GEMINI AI and Grok grow to a 480-frame/20-second
  ceiling by 20s+; ChatGPT grows to a 600-frame/25-second ceiling, the
  largest of any table here) — a long scene under these tables needs the
  tool to look nearly all the way through the scene before computing a depth
  range for even its first frame, meaning real added VRAM use and startup
  delay, especially with a heavier Depth Model.
- **Fast Action / Medium Magical / Drama Slow Paced** — three more
  ChatGPT-suggested, unverified tables for genre-paced A/B testing. Unlike
  GEMINI AI/ChatGPT/Grok, these are capped at 10 seconds: Buffer/Decay ramp
  up through the 9-10s bucket, then hold flat through 20s+, at a 240-frame
  ceiling. Buffer is identical bucket-for-bucket across all three — only
  Decay differentiates them, from Fast Action's quick-to-react minimal
  smoothing up to Drama Slow Paced's maximum stability. (Separate mechanism
  from the Genre Preset quick-fill below — that writes one fixed pair for a
  whole movie; these three vary automatically by each scene's own measured
  length.)
**Default selection:** Nagadomi_Reference.

#### Auto EMA — Edit Values...

**What it does:** Opens an editor for the Buffer/Decay numbers Auto EMA by
Scene Length uses, for whichever model is currently selected in the
dropdown.
**What you can change:** Buffer and Decay for each of the 21 scene-length
rows (0-1s through 19-20s, plus 20s+). The scene-length ranges themselves are
fixed.
**Pros:** changes are saved and reused automatically on every future run, no
need to re-enter them each time.
**Cons:** it's easy to type in a value that looks reasonable but actually
smooths too little (visible flicker) or too much (ghosting/lag) for a given
scene length — change one row at a time and re-check the result before
trusting a whole edited table.
**Recommended:** leave untouched unless you've already noticed under- or
over-smoothing at a specific range of scene lengths on your own material.
Click Reset to Default inside the editor at any time to undo your changes
for that model.

#### Genre Preset

**What it does:** A quick-fill shortcut for Flicker Reduction's Decay
Rate/Buffer fields, based on a whole movie's general content/pacing — NOT the
same thing as "Auto EMA by Scene Length," which instead varies Decay/Buffer
automatically PER SCENE based on each scene's own measured length. Pick one
approach for a given job: type Decay/Buffer by hand, use this to quick-fill
them once, or turn on Auto EMA by Scene Length to bypass fixed values
entirely.
**How it works:** selecting a preset immediately writes that preset's
Decay/Buffer numbers into the fields above, once — it is not a live link, so
hand-editing either field afterward is completely safe.
**Why only Decay differs between presets:** Decay is a pure per-frame math
weighting with no extra cost, so it can safely differentiate the presets.
Buffer is a literal future-frame lookahead window (real VRAM/startup-latency
cost, not a "quality" dial), so it stays conservative and close across all
three presets instead of growing with intensity.
**Values:** Fast Action = Decay 0.75/Buffer 30 (max responsiveness, minimal
lookahead). Medium / Magical = Decay 0.85/Buffer 72 (balanced — fantasy/VFX-
heavy content with mixed pacing). Drama / Slow-Paced = Decay 0.94/Buffer 120
(maximum stability, still within nagadomi's own documented reference
ceiling).
**Recommended:** a reasonable starting point, not a guarantee — a movie's
pace can vary within itself, so spot-check the result and hand-adjust if a
specific stretch doesn't match the chosen preset's pace.
**Note:** greyed out, same as Decay Rate/Buffer, whenever "Auto EMA by Scene
Length" is checked.

#### Motion-Adaptive Smoothing

**What it does:** Automatically eases Flicker Reduction's Decay Rate DOWN
during fast motion, instead of using one fixed smoothing strength for the
whole clip.
**Pros:** reduces the "laggy/flat for a beat" effect a high Decay Rate can
cause during fast action, while keeping the full smoothing benefit during
calm scenes.
**Safety note:** this can only ever REDUCE smoothing below your Decay Rate
setting during fast motion, never increase it above what you set — it's a
safe add-on, not a replacement for choosing a sensible Decay Rate.
**Recommended:** on, if your content mixes calm and fast-motion scenes (most
movies do).

#### Scene Boundary Detection

**What it does:** Detects real scene/shot cuts using a trained AI model (not
just a simple brightness-difference trick, so it's less likely to
false-trigger on flashes or fast pans) and resets Flicker Reduction's
smoothing exactly at those points, instead of letting it bleed depth-scale
information across two completely unrelated shots.
**Pros:** also lets Resume align its chunk boundaries to real cuts, and lets
Object Stability/EMA buffers size themselves against real, measured shot
lengths instead of guessing.
**Cons:** a small amount of extra processing time to run the cut-detection
pass (mitigated by the cache setting below on repeat runs of the same file).
**Recommended:** on for essentially all movie/TV content, especially when
Flicker Reduction or Resume is also on.

#### Use scene boundary cache

**What it does:** Saves detected scene cuts to disk (keyed to the source
file's path, size, and modification time), so re-running the same video
later — after a crash, or while testing different 3D settings — doesn't need
to re-scan for cuts from scratch every time.
**Recommended:** on. It auto-invalidates if the source file actually changes
(different size or modified date), so there's no real downside for normal
use.

#### Preserve Screen Border

**What it does:** Tapers the 3D shift down to zero right at the left/right
frame edges, instead of applying full strength all the way to the border.
**How it helps:** this is an automated version of the "floating window"
technique real stereographers use — without it, an object with strong 3D pop
that gets cut off by the frame edge creates a jarring conflict (the object
appears to be in front of the screen, but the screen's own edge is cutting
it off, which shouldn't be possible if it's really floating in front).
Tapering the edges to zero avoids that specific conflict.
**Cons:** very slightly reduces 3D strength right at the extreme edges of
the frame — imperceptible in the middle of the image, only affects the
outermost border.
**Recommended:** on for most content, especially anything with strong 3D
Strength or objects that move near frame edges. Off only if you specifically
want maximum strength everywhere and don't mind occasional edge conflicts.

#### Stereo Format

**What it does:** The final output layout.
**Values:**
- **Full/Half SBS (side-by-side)** and **Full/Half TB (top-bottom)** — the
  standard formats most 3D TVs and players expect. Full keeps both eyes at
  full resolution (bigger file); Half squeezes both into the original frame
  size (smaller file, common for streaming/playback compatibility).
- **VR90** — for VR headsets.
- **Cross Eyed** — for viewing without any equipment.
- **RGB-D / Export / Export disparity** — save the depth data itself instead
  of a finished 3D image.
- **Anaglyph** — the red/cyan glasses look.
- **Debug Depth** — internal/diagnostic depth visualization.
**Recommended:** Half SBS for most TVs and 3D players, unless you know you
need a different format.

#### Anaglyph Method

**What it does:** Only used when Stereo Format is Anaglyph. The
color-filtering recipe used for red/cyan glasses viewing.
**Values:** dubois / dubois2 / color / gray / half-color / wimmer / wimmer2.
dubois/dubois2 give the most natural, least color-distorted result for most
red/cyan glasses. gray avoids color distortion entirely but loses color in
the image.
**Recommended:** dubois2.

#### Tag MKV as 3D (StereoMode)

**What it does:** Writes the standard Matroska "StereoMode" property into the
finished .mkv's video track, describing the 3D layout (side-by-side/top-
bottom) and which eye comes first.
**Pros:** 3D-aware players and TVs (VLC, Kodi, some smart TVs) read this and
automatically switch into the correct 3D display mode — without it, the
viewer has to tell their player "this is 3D, side-by-side, left eye first"
manually every time.
**Only applies to:** .mkv output, and only Full/Half SBS, Full/Half TB, Cross
Eyed, and VR90 — these are real two-eye pairs Matroska's StereoMode can
describe. RGB-D, Half RGB-D, and Anaglyph are skipped automatically (a note
is printed, not a silent no-op).
**Cons:** none for a normal 3D TV/player — this is a pure metadata edit, no
re-encoding. A player that ignores StereoMode entirely just displays the
file exactly as it would have anyway.
**Note for VR90:** this only tells a player the eye order, not that the
video is a 180° spherical projection — VR headset apps still rely on the
existing "_180x180_LR" filename tag to recognize that, so turning this on
for VR90 output is harmless but not a substitute for that filename
convention.
**Recommended:** on if your output goes to a 3D TV, a 3D-aware player like
VLC/Kodi, or you just don't want to manually configure 3D mode every time.
Off (default) if you're not sure your player supports it, or your workflow
already re-muxes/renames the file afterward in a way that could lose this
tag anyway.

#### Depth Only

**What it does:** Only used with Stereo Format "Export"/"Export disparity."
Saves just the raw depth images, skipping the RGB frames a normal export
also saves.
**Cons:** data exported this way is missing what's needed to later re-import
and reconstruct a full stereo render from it — it's for inspecting/reusing
the depth data itself, not for resuming a conversion later.

#### Resize to fit

**What it does:** Only used with Stereo Format "Export"/"Export disparity."
Resizes the exported depth images up to match the RGB frames' full
resolution, instead of staying at the depth model's own (usually smaller)
native resolution.
**Cons:** the exported files become significantly larger, and the process
noticeably slower, since every depth frame is upscaled and re-saved.
**Recommended:** off unless whatever you're using the exported depth images
for specifically needs them pixel-matched to the RGB frames.

---

## Dual-Pass Depth Blend Tab

#### Dual-Pass Depth Blend

**What it does:** Brings in a SECOND depth model and blends it into Depth
Model's result, favoring the second model specifically where the image has
dense fine detail (foliage, hair, close-up texture) that a single-frame
model often gets wrong. Runs as 3 full passes over the clip (export depth A,
export depth B, blend+render), fully releasing each model before the next
loads so both are never in GPU memory at once (this sequential loading is a
deliberate invariant of the feature).
**Cons:** costs roughly 3x the time and real extra disk space (a full frame
dump, kept in a `<output>.depth_blend_work` folder you can delete
afterward). Requires a single video file input; not compatible with
Automated Scene Batch.

#### Dual-Pass Depth Blend — Model

**What it does:** The SECOND depth model for Dual-Pass Depth Blend.
**Recommended:** pick one with strengths your main Depth Model lacks — e.g. a
VDA_* model for its steadier handling of foliage/close-up detail.

#### Dual-Pass Depth Blend — Strength

**What it does:** How strongly to favor the SECOND model within the selected
region (0-1). 1.0 = fully trust the second model there; lower values blend
it in more gently, leaning back toward the main Depth Model's own result.
**Cons:** pushed to 1.0 in the "detail" region, you're fully trusting a model
that may not agree pixel-for-pixel with the primary model at real edges —
see Edge Suppression, which exists specifically to manage that risk.
**Recommended:** 1.0 to start; back off toward 0.5-0.75 if you notice any
softness/ghosting at silhouettes even with Edge Suppression on.
**Values:** 1.0 / 0.75 / 0.5 / 0.25.

#### Dual-Pass Depth Blend — Region

**What it does:** WHERE the second model gets blended in.
**Values:**
- **detail** — wherever the image itself shows dense fine visual detail
  (foliage, hair, close-up texture) — the original use case this feature was
  built for. Automatically avoids blending right at the primary model's own
  confident edges (see Edge Suppression).
- **foreground / background** — a straight percentage slice of the scene BY
  DEPTH (nearest or farthest X%, set below), regardless of how much visual
  detail is actually there.
**Recommended:** detail for the original foliage/hair use case;
foreground/background only if you specifically want the second model's
characteristics applied to a whole depth range (e.g. a steadier video-native
model just for a busy background) rather than detail-seeking.

#### Dual-Pass Depth Blend — Region Percent

**What it does:** For "foreground"/"background" region only (ignored for
"detail"): what percent of the scene, by depth, to blend the second model
into — e.g. 25 = nearest (or farthest) 25% of the scene.
**Cons:** a larger percentage moves the blend/no-blend transition line
further into the midground, where a difference between the two models
becomes more likely to be noticeable.
**Recommended:** 25 as a starting point; keep it modest unless you have a
specific reason to cover more of the scene.
**Values:** 10 / 25 / 40 / 50.

#### Feather Blur

**What it does:** Softens the blend transition ITSELF (blur kernel size in
pixels; 0 = off, a sharp on/off transition) — works with any Region mode,
smoothing the seam between blended/unblended areas so it doesn't read as a
visible hard line.
**Cons:** a large value can blur the transition zone wide enough to become
visible itself as a soft band, rather than fixing a hard edge.
**Recommended:** 0 (off) unless you specifically notice a hard seam where
blending starts/stops; start small (5-15) and increase only if needed.
**Values:** 0 / 5 / 15 / 25 / 35.

#### Bilateral Denoise

**What it does:** Runs an edge-preserving smoothing pass over the FINAL
blended depth (after both models are already combined) to clean up noise
introduced by mixing two models, without softening real depth boundaries the
way a plain blur would.
**Cons:** a real extra processing pass — some added time cost per frame.
**Recommended:** off by default; turn on if you notice speckle/noise texture
in the blended result that a plain Depth Detail Refinement pass doesn't
fully clean up.

Its three sub-settings:
- **Diameter (d):** how large an area around each pixel is considered.
  Higher = smooths a wider neighborhood but costs more processing time.
  **Recommended:** 12. Values: 9 / 12 / 15.
- **Sigma Color:** the filter's depth-value sensitivity, expressed in
  familiar 0-255-ish terms. Higher = willing to smooth across BIGGER depth
  differences (risks blurring real edges); lower = only smooths very similar
  depth values together (safer for real edges, cleans up less noise).
  **Recommended:** 75. Values: 50 / 75 / 100.
- **Sigma Space:** the filter's spatial reach, in pixels. Higher = smooths
  across a physically wider area of the frame; lower = stays more localized.
  **Recommended:** 75. Values: 50 / 75 / 100.

#### CLAHE Contrast (experimental)

**What it does:** Local contrast enhancement over the final blended depth.
**OFF BY DEFAULT and NOT RECOMMENDED:** an earlier benchmark on this
project's own depth data found CLAHE amplifies whatever local variation it
finds, noise included (see `docs/ai/AI_DECISIONS.md` ADR-001, the decision
that originally rejected this as a default-on refinement stage — this
control exists as an explicit opt-in experiment only, never re-enabled by
default).
**Sub-settings (only matter if this is turned on):**
- **Clip Limit** — how aggressively local contrast gets boosted. Higher =
  stronger local contrast, but also amplifies more noise. Values: 1.0 / 2.0
  / 4.0.
- **Tile Grid Size** — how finely the frame is divided up for LOCAL contrast
  adjustment. More tiles = more localized (small-area) contrast changes;
  fewer tiles = smoother, more global adjustment. Values: 4 / 8 / 16.

#### Depth Scale Alignment

**What it does:** Both depth passes are independently normalized to the same
0-1 numeric range, but that doesn't guarantee a given value means the same
real-world distance in both models. This rescales the second model's depth
to match the first model's actual distribution (robust 5th-95th percentile
matching, EMA-smoothed across frames to avoid flicker) before blending,
reducing potential depth-value mismatches at blend region edges.
**Note:** off by default; adds a one-time sequential pre-pass over all
frames before blending starts. See Alignment Decay for the smoothing
strength.

#### Alignment Decay

**What it does:** How much the frame-to-frame alignment fit (Depth Scale
Alignment) is smoothed across frames, instead of being recalculated fresh
and independently for every single frame.
**Values:** higher (0.9-0.95) = steadier alignment, resistant to a single
noisy/outlier frame throwing the fit off, but slower to follow a genuine
change (e.g. a big lighting/scene change shifting how the two models relate
to each other). Lower (0.5-0.75) = reacts faster per frame, more exposed to
noise. 0 = no smoothing at all, each frame's alignment computed completely
independently.
**Recommended:** 0.9 (default, matches this feature's original tuning) as a
safe starting point; lower it only if you specifically notice the alignment
lagging behind a real, sudden change between the two models.
**Values:** 0.95 / 0.9 / 0.75 / 0.5 / 0.

#### Edge Suppression

**What it does:** Only affects the "detail" blend region. Wherever the
primary depth model already has a clean, confident edge (a person's
silhouette, say), the two models rarely agree on the exact pixel — blending
them there risks a soft, slightly doubled-looking edge. This protects a band
around those edges from blending.
**Values:** higher = wider protected band, safer against that soft/doubled-
edge look but suppresses a touch more real detail blending near edges too.
Lower = narrower protection, more detail blending everywhere but more risk
of soft edges.
**Recommended:** default 0.5 is a middle ground between two previously-
tested extremes (see `docs/ai/AI_DECISIONS.md` ADR-022, which made this a
real adjustable setting).
**Values:** 0.0 / 0.25 / 0.5 / 0.75 / 1.0.

#### Hard Edge Cutoff

**What it does:** Only matters together with Edge Suppression ("detail"
region only). Edge Suppression normally fades in smoothly as you approach a
real silhouette — this switches that smooth fade into a hard on/off step
instead (same protected band width, sharper edge to it). A cheap thing to
try if Edge Suppression alone still leaves a soft/misaligned-looking border
on some objects.
**Cons:** cannot fully eliminate a genuine shape disagreement between two
independently-trained depth models — there's no established technique to
fully correct that (verified via research, not assumed), only shrink its
visible impact further than the smooth fade does alone.
**Recommended:** off by default; try on only after Edge Suppression is
already near 1.0 and still isn't enough.

---

## Video Filter Tab

#### Start Time / End Time

**What it does:** Only process the video from/to this timestamp, instead of
the whole file. Useful for testing settings on a specific scene, or trimming
unwanted intro/outro footage.

#### Deinterlace

**What it does:** For old interlaced video sources (combed/striped look on
motion). "yadif" converts it to normal progressive video before 3D
conversion.
**Recommended:** leave blank for modern, already-progressive sources (most
streaming/BluRay video).
**Values:** blank / yadif.

#### -vf (src)

**What it does (advanced):** A raw ffmpeg video filter string applied to the
source before 3D conversion (e.g. cropping, scaling).
**Recommended:** leave blank unless you specifically need this — most common
needs (crop, rotate, pad, resize) already have their own simpler controls.

#### Rotate

**What it does:** Rotate the source video before 3D conversion, e.g. for
footage shot sideways on a phone.
**Values:** blank / Left 90 (counterclockwise) / Right 90 (clockwise).

#### AutoCrop

**What it does:** Automatically detects and removes black bars or plain/flat
borders before 3D conversion (so they don't waste depth detail or get
distorted). `_TB` variants only crop top/bottom bars; the non-TB variants
crop on all sides. Use the "Test" button to preview the detected crop before
running a full conversion.
**Values:** blank / BLACK_TB / BLACK / FLAT_TB / FLAT.

#### Padding

**What it does:** Adds solid padding/border area around the frame before the
3D shift is applied, giving objects near the edges more room to shift into
without hitting the frame boundary.
**Mode values:** tb = top and bottom only. lr = left and right only. top =
top only. 16:9 = pads to a 16:9 aspect ratio specifically.
**Amount:** how much padding to add, as a ratio of frame size (only applies
if Padding Mode is set to something other than blank).
**Cons:** larger amount values waste more of the frame on empty border
instead of picture content.
**Recommended:** leave Mode blank unless you're seeing objects clipped by
the frame edge even with Preserve Screen Border on; if you do use it, start
the amount small (0.01-0.05).

#### Output Size Limit

**What it does:** Caps the FINAL PACKED FRAME's resolution (the whole SBS/TB
image, after both eyes are already combined — not a per-eye limit), to keep
file size/playback requirements manageable on a large source. Leave blank to
keep the source's native size, uncapped.
**Important gotcha (verified by tracing the actual code):** Full SBS builds
a canvas DOUBLE your source's width (both eyes at full width, side by side)
— e.g. a 3840-wide source becomes 7680 wide. If you set this limit to
3840x2160 (a natural-looking pick for "4K") with Keep Aspect Ratio off, it
crushes that 7680-wide Full SBS frame down to 3840 width while leaving
height alone — which is the EXACT SAME pixel dimensions Half SBS already
produces on purpose. Result: Full SBS and Half SBS can end up looking
pixel-for-pixel identical, and toggling between them appears to do nothing.
**Recommended:** for true Full SBS (full per-eye detail), set this to double
your target width/height (e.g. 7680x2160 for a 4K target) or leave it
blank. For Half SBS at a 4K delivery frame, 3840x2160 is correct as-is.
Whichever Stereo Format you picked, make sure this limit's width actually
matches what that format is supposed to produce.
**Values:** blank / 7680x2160 / 3840x2160 / 3840x1608 / 3840x1080 /
1920x1080 / 1280x720 / 640x360 / 1920x3200 / 1080x1920 / 720x1280 / 360x640.

#### Keep Aspect Ratio

**What it does:** When Output Size Limit is set, proportionally shrinks to
fit inside it (preserving the source's width/height ratio) instead of
crushing width and height independently to exactly match the limit's exact
dimensions.
**Cons of leaving off (the default):** as documented in Output Size Limit's
own tooltip, an independent width/height crush can accidentally make Full
SBS collapse down to the same pixel dimensions as Half SBS if the limit
isn't sized correctly for your chosen Stereo Format.
**Recommended:** turn on if you're not carefully matching Output Size Limit
to double/single width for your Stereo Format choice, so an undersized limit
shrinks proportionally instead of distorting the image.

#### Preserve Dolby Vision

**What it does:** Detects Dolby Vision RPU and/or HDR10+ dynamic metadata on
the source and re-attaches it to the finished 3D video, so the HDR grading
survives conversion instead of being silently dropped.
**Requires:** HEVC output (Video Codec set to hevc_nvenc, hevc_qsv, hevc_amf,
or libx265 — this metadata format doesn't exist for other codecs) and
MKVToolNix installed for MKV output.
**Cons:** a small amount of extra time at the end of each job for the
injection/remux step.
**Recommended:** on for any Dolby Vision or HDR10+ source you want to keep
looking correct on an HDR display after conversion.

#### Convert HDR to SDR

**What it does:** Properly tone-maps a PQ/HDR10/HDR10+ or HLG source down to
normal SDR brightness before conversion, while keeping 10-bit color
precision (reduces banding vs. plain 8-bit SDR). Adds one extra encoding
pass. Automatically does nothing if the source isn't actually PQ/HLG HDR.
Turns off Preserve Dolby Vision, since the two are contradictory (there's no
HDR grade left to preserve once tone-mapped to SDR).
**Recommended:** on for HDR sources being watched on a non-HDR display (most
projectors and many 3D setups) — see this project's guidance on HDR/Dolby
Vision for projectors and VR headsets.

#### Auto Resume

**What it does:** Lets a long conversion pick back up exactly where it left
off if it's interrupted (crash, power loss, or clicking Cancel), instead of
starting over from the beginning. Processes the whole video in one
continuous pass — it does NOT split it into fixed-size pieces on a schedule.
A new piece is only ever created at the exact point an interruption actually
happened, so a normal, uninterrupted run has zero seams, identical to not
using this option at all. If interrupted, you get exactly one seam at that
point, not many.
**Recommended:** on for any long/overnight conversion.

#### Denoise

**What it does:** Applies temporal (across-frame) denoising to the SOURCE
before depth estimation even sees it. Reduces film grain/sensor noise that
can otherwise confuse the depth model into reading noise as fake
texture/detail.
**Pros:** particularly useful on older, grainier film sources where visible
grain can translate into a noisier, less stable depth map.
**Cons:** real extra time up front (a full separate ffmpeg pass over the
whole video before conversion starts), and denoising always risks softening
some genuine fine detail along with the grain.
**Recommended:** on for visibly grainy/old sources; off for clean, modern
digital sources where there's no real grain to remove.

#### Preview Mode

**What it does:** Quickly renders just the first 60 seconds at 1fps and low
resolution instead of a full conversion.
**Pros:** lets you check whether your settings (3D Strength, Convergence,
Depth Model, etc.) look right BEFORE committing to a full-length conversion
that might take hours.
**Recommended:** on whenever you're testing new settings on a file for the
first time; off for your actual final conversion run.

#### Automated Scene Batch

**What it does:** A fully automated whole-movie pipeline: removes letterbox
bars once, detects scene cuts once, splits into per-scene clips with exact
keyframe-accurate boundaries, converts each scene independently (a true
fresh start per scene instead of one shared running state), then joins the
results back into one seamless video and reattaches the original audio. If
Preserve Dolby Vision is also on, the RPU is extracted once from the source
and reinjected once at the end — never per-clip.
**Requirement:** input must be a single video file, not a folder.

#### Scene Batch Crop

**What it does (optional, for Automated Scene Batch):** Explicit crop as
`WxH` or `WxH:X:Y` (X/Y default to centered).
**Recommended:** leave blank to auto-detect letterbox bars once from the
whole movie.

#### Scene Settings File

**What it does (optional, for Automated Scene Batch):** A JSON file giving
per-scene setting overrides (e.g. a different Divergence for different
stretches of the movie).
**Recommended:** leave blank to use the same settings for every scene.

#### VR Optimized Merge

**What it does:** For Automated Scene Batch's final join step.
**Off (default):** scenes are joined with a fast, lossless stream copy —
quick, but each scene clip keeps its own independent timestamps/keyframe
structure, which can read as uneven on a VR headset even though normal
playback looks fine.
**On:** re-encodes the joined timeline as one continuous stream with clean
regenerated timestamps and AAC 48kHz audio, intended for smoother VR/headset
playback. Slower than the default.
**Recommended:** only turn this on if you're actually delivering to a VR
headset workflow.

#### VR Merge FPS

**What it does:** Target constant frame rate for VR Optimized Merge.
"Source FPS" keeps the already-converted frame rate as-is (still gets clean
regenerated timestamps, just no specific forced rate) — pick a fixed VR
headset refresh rate instead if your delivery target needs one exactly.
**Values:** Source FPS / 60 / 72 / 80 / 90 / 120.

#### Scene Batch Variant

**What it does (optional, for Automated Scene Batch):** Reuses the shared,
already-done work from a prior run of the SAME movie (Dolby Vision RPU, the
crop+keyframe pass, the split scene clips) but writes its own converted
scenes and final output under this name, so trying different settings here
never touches or overwrites an earlier attempt's progress.
**Recommended:** leave blank for a normal run. Output becomes
`<name>_<variant>.<ext>`.

#### Bit Depth Upgrade

**What it does:** Gives the internal depth/3D math more color precision to
work with, reducing banding (visible color "steps" in skies/shadows) — it
doesn't add detail that wasn't in the source.
**Recommended:** 10 for most sources, a real, visible improvement over 8-bit
with low cost. 12 adds little you can actually see on a normal screen, for
extra file size and slower encoding — usually not worth it. No effect if
your Pixel Format is already higher bit-depth than what you pick here.
**Values:** blank / 10 / 12.

---

## Video Decoding Tab

#### HWAccel

**What it does:** Use your GPU to decode (read) the source video instead of
the CPU — much faster, and frees up CPU time for other work, with no quality
difference.
**Recommended:** select your GPU's option (e.g. cuda/nvdec for NVIDIA) if
available. Leave blank to always use CPU decoding.

#### Software Fallback

**What it does:** Use software (CPU) decoder if hardware acceleration fails
or is unsupported for a particular file, instead of stopping with an error.
**Recommended:** on. (This is the default/on behavior in the current code —
if you're relying on an older description that said this was off by default
or opt-in, that's outdated; it's checked/on out of the box.)

---

## Video Encoding Tab

#### Max FPS

**What it does:** Caps the output frame rate. If the source is already at or
below this, nothing changes.
**Pros:** lowering it reduces processing time roughly proportionally (half
the frames = about half the time), at the cost of less smooth motion.
**Recommended:** leave high (e.g. 1000) to just keep the source's own frame
rate, unless you specifically want to reduce it for speed.
**Values:** 1000 / 60 / 59.94 / 30 / 29.97 / 24 / 23.976 / 15 / 1 / 0.25
(editable).

#### Video Format

**What it does:** The output container/file type.
**Values:**
- **mp4** — the most universally compatible.
- **mkv** — supports more advanced features (like Dolby Vision/HDR10+
  metadata and lossless codecs).
- **avi** — mainly for the lossless utvideo codec.
**Recommended:** mkv if you need HDR/Dolby Vision or lossless output, mp4
otherwise.

#### Video Codec

**What it does:** The compression format used to encode the output video.
**Values:**
- **libx264/libx265** — CPU-based, h264/HEVC, give the best
  quality-per-file-size and work everywhere.
- **h264_nvenc/hevc_nvenc** — use your NVIDIA GPU's hardware encoder — much
  faster, slightly lower quality-per-file-size.
- **utvideo/ffv1** — lossless (huge files, no quality loss at all) for
  archival/further editing.
**Recommended:** libx265 for best quality, hevc_nvenc if encoding speed
matters and you have an NVIDIA GPU.

#### Pixel Format

**What it does:** The color sampling/precision the encoder stores
internally.
**Values:**
- **yuv420p** (8-bit) — the most compatible default.
- **yuv420p10le** — adds the precision-boosting benefit described under Bit
  Depth Upgrade (where available, that setting can apply on top of this).
- **yuv444p/rgb24/gbrp\*** — avoid color-detail loss around sharp red/edge
  areas, at a larger file size.
**Recommended:** yuv420p10le for most uses.

#### Colorspace

**What it does:** How color values are tagged/interpreted (affects color
accuracy on playback, not detail).
**Recommended:** "auto" — matches the source's own colorspace and is correct
for virtually all sources. Only change this if you know your source's
colorspace is mistagged.
**Values:** auto / unspecified / bt709 / bt709-pc / bt709-tv / bt601 /
bt601-pc / bt601-tv.

#### CRF

**What it does:** Constant Rate Factor — the main quality/file-size dial for
most codecs.
**Values:** lower number = higher quality and bigger file; higher number =
smaller file and lower quality.
**Recommended:** 15-18 for near-lossless/archival quality, 20-23 for a good
everyday balance.
**Values:** 16-27 (editable).

#### Bitrate

**What it does:** Only used by libopenh264, which doesn't support CRF. Sets
a fixed data rate for the video (M = megabits/second) — higher = better
quality and bigger file.
**Recommended:** 8M-16M for 1080p, higher for 4K.
**Values:** 160M / 50M / 16M / 12M / 8M / 4M.

#### Level

**What it does:** A technical compatibility limit (resolution/bitrate
ceiling) some older TVs/players check before they'll play a file.
**Recommended:** "auto" — only set this manually if a specific playback
device of yours refuses to play the output.

#### Preset (Video Encoding)

**What it does:** How hard the encoder works to compress efficiently.
**Values:** slower presets (slow/slower/veryslow) give a smaller file at the
same quality (CRF), but take longer to encode. Faster presets
(fast/veryfast/ultrafast) encode quicker but produce a larger file for the
same quality.
**Recommended:** medium-slow for a good balance; veryfast if encoding time
matters more than file size.
**Default:** medium.

#### Tune

**What it does (for libx264/libx265):** Nudges the encoder's choices for a
specific kind of content — e.g. "grain" preserves film grain/noise better,
"animation" suits flat-color cartoon content.
**Recommended:** leave blank unless your source clearly matches one of these
categories.
**What it does (for hevc_nvenc/h264_nvenc):** chooses the GPU encoder's
quality/speed tradeoff. "hq" = high quality (good default). "uhq" (HEVC
only, needs a recent NVIDIA GPU) = an even higher quality mode than hq,
using extra quality heuristics, for a modest speed cost — worth using if
your GPU supports it and you want the best NVENC result. "lossless" = no
quality loss at all, but a very large file. "ll"/"ull" are for low-latency
live-streaming, not useful for a normal conversion.

#### Tune — fastdecode

**What it does:** Makes the resulting file easier/cheaper to decode during
playback (helpful on weaker playback devices), at a small cost to
compression efficiency (slightly bigger file).

#### Tune — zerolatency

**What it does:** For live-streaming/real-time use cases — not useful for a
normal movie file conversion like this.
**Recommended:** leave off.

---

## Processor Tab

#### Device

**What it does:** Which GPU (or CPU) does the AI processing.
**Values:** your detected GPU(s) by name, "All CUDA Device" (splits work
across every GPU you have for faster batch processing), and CPU.
**Cons:** CPU works without a GPU but is dramatically slower — only use it
if you have no compatible graphics card.

#### Depth Batch Size

**What it does (video only):** How many frames are sent to the depth model
at once.
**Pros/Cons:** higher = faster overall but uses more VRAM.
**Recommended:** lower it if you run out of memory; raise it if you have
VRAM to spare and want faster processing.
**Values:** 1 through 64 (default selection 2).

#### Worker Threads

**What it does (video only):** How many CPU threads handle the
stereo/inpainting step and frame I/O in parallel.
**Pros/Cons:** higher can speed things up on a multi-core CPU, but uses more
RAM/VRAM.
**Recommended:** lower it if you run into memory issues or system slowdowns
while converting.
**Values:** 0 through 16 (default selection 0, meaning automatic).

#### Low VRAM

**What it does:** Trades speed for lower memory use, by processing in a way
that needs less VRAM at once. Confirmed real fix for Method forward_splat_fill's
out-of-memory crashes on demanding footage, even when Depth Batch Size is
already set to 1 — turn this on if that method OOMs for you.
**Recommended:** only turn this on if you're actually running out of memory
— it will make things slower.

#### TTA

**What it does:** Use flip augmentation to improve depth quality (slow).
Runs the depth model on both the normal and mirrored image/video and blends
the result, often a little cleaner/more accurate, at roughly double the
processing time. Works for both image depth models and video (VDA_*)
models.
**Recommended:** off for long videos (Flicker Reduction already covers
similar ground there, for free) — worth trying for a single hero image, or a
short clip where the extra quality matters more than the time cost.

#### FP16

**What it does:** Runs the AI math at lower numeric precision, which is
significantly faster and uses less VRAM on modern GPUs, with no visible
quality cost in virtually all cases.
**Recommended:** on (default).

#### Stream

**What it does (experimental):** Use per-thread CUDA Stream — lets multiple
worker threads share the GPU more aggressively.
**Cons:** may speed things up, may do nothing, may occasionally cause
instability.
**Recommended:** try it and turn it back off if you see crashes.

#### torch.compile

**What it does:** Enables model compiling: optimizes the AI model before
running, trading a slower startup (the first run after changing settings has
to compile) for faster processing afterward.
**Recommended:** worth it for long videos; not worth it for a single quick
image or short clip. Requires extra setup (see this project's
`torch_compile` docs) to actually take effect.

#### Free GPU memory while paused

**What it does:** When you click Suspend, also move every loaded model
(depth model, stereo/side model, and the SOD_v1 auto-convergence model, if
used) off the GPU and actually release that VRAM back to Windows, instead of
just pausing while everything stays loaded. Resume moves them all back
before continuing.
**Why it helps:** normally, pausing does NOT free any VRAM — every model
just sits loaded and idle the whole time you're paused, so nothing else can
use that memory. This lets you actually hand the GPU to something else
(another program, a second conversion) while paused.
**Pros:** real VRAM freed while paused; Resume still produces a correct,
uninterrupted output.
**Cons:** Resume is no longer instant — reloading the models back onto the
GPU takes a few seconds (longer if torch.compile is on, since it may
recompile). No effect with multi-GPU ("All CUDA Device") — those models are
already spread across every GPU and are left as-is.
**Values:** off (default) = models stay resident, Resume is instant. On =
frees VRAM while paused, Resume takes a few seconds.
**Recommended:** off, unless you specifically need the GPU free for
something else during a long pause.

---

## Post-Processing

Lives on the Processor tab, below the Processor group.

#### Upscale with waifu2x after conversion

**What it does (single video and Dual-Pass Depth Blend jobs only):** once
this job's finished output is fully written, automatically runs it through
waifu2x (a separate, dedicated AI upscaler bundled with this app) as one
extra step, so you don't need to run waifu2x by hand afterward.
**How it's safe:** saved to a separate `_w2x` file — the original conversion
output is always left untouched, even if the upscale step itself fails.
**Cons:** real extra processing time after the main conversion already
finished, roughly proportional to the upscale factor chosen.
**Recommended:** on if you specifically want a higher-resolution final
delivery file; off if your source resolution is already your target.

#### waifu2x Method

**What it does:** Which waifu2x mode to run.
**Values:** "noise_scale" upscales AND reduces compression artifacts/noise
at the same time; plain "scale" only upscales, leaving existing noise alone.
2x/4x is the resulting size multiplier (width and height each multiplied by
this).
**Recommended:** noise_scale2x for most delivery-quality video (mild, safe
noise cleanup plus a reasonable size bump); use 4x variants only if you
specifically need a much larger frame.

#### waifu2x Noise Level

**What it does:** Noise reduction strength (0=off, 3=strongest). Ignored for
plain "scale" methods.
**Cons:** pushed too high on a source that's actually clean (not
noisy/grainy), can start softening real fine detail along with noise that
isn't really there.
**Recommended:** 1 as a safe default; raise toward 2-3 only for a visibly
noisy/compressed source.

#### waifu2x Style

**What it does:** waifu2x model style.
**Values:** "photo" is the better default for real movie footage; "art" is
tuned for illustration/anime source material.

#### waifu2x Target

**What it does:** Only takes effect on a packed 3D stereo video output
(Half/Full SBS, Half/Full TB, Cross-Eyed, VR90).
**Values:**
- **auto** (default) — leaves the plain whole-frame upscale completely
  unchanged (fixed 2x/4x from Method).
- **4k/8k** — instead switch to a stereo-aware upscale: the video is split
  into its two independent eye images first, each eye is upscaled separately
  (so the AI upscaler never blurs/blends pixels across the seam between the
  two eyes the way upscaling the packed frame whole can), a frame-to-frame
  smoothing pass reduces flicker that becomes more visible at very high
  output resolutions, then the eyes are recombined. The exact per-eye
  enlargement needed is computed from your source's real resolution and
  split direction, so the final packed video actually lands on the
  requested width (3840 for 4k, 7680 for 8k).
**Cons:** several extra full passes over the video beyond the plain
whole-frame path, so real processing time is meaningfully longer.
**Recommended:** auto unless you specifically need a 4K/8K delivery file
from a 3D stereo source and want the cleaner per-eye result.

#### Interpolate frames with RIFE after conversion

**What it does (single video and Dual-Pass Depth Blend jobs only):** once
this job's finished output is fully written, runs it through RIFE (a
separate AI frame-interpolation model) as one extra step, generating new
in-between frames for smoother-looking motion. How many/where is controlled
by the Rate mode dropdown below (2x by default).
**How it's safe:** saved to a separate `_rife` file — the original
conversion output is always left untouched, even if the interpolation step
itself fails.
**Cons:** real extra processing time after the main conversion already
finished; RIFE interpolates the FINAL PACKED stereo frame (both eyes already
combined) as one image, so it will see the seam between the two packed eyes
— it wasn't trained on that, though in practice it moves both eyes together
so this doesn't cause left/right desync.
**Cannot be combined with Preserve Dolby Vision:** there's no way to assign
correct DV/HDR10+ metadata to RIFE's synthetic in-between frames.
**Recommended:** on if your source is naturally low frame rate (e.g. 24fps
film) and you want smoother motion for VR viewing; off if you're already
happy with the source's frame rate or you need Dolby Vision preserved.

#### RIFE Model

**What it does:** Which RIFE model quality tier to use.
**Values:**
- **rife_425** — the recommended full model — better motion accuracy,
  especially on complex/fast motion, at a higher compute cost.
- **rife_425_lite** — a lower-compute-cost variant of the same generation,
  trades a little accuracy for speed.
**Note:** weights are downloaded automatically the first time you use a
given tier (not bundled with the app).
**Recommended:** rife_425 unless interpolation time is a real bottleneck for
you.

#### RIFE Rate

**What it does:** How many new frames RIFE inserts, and where.
**Values:**
- **2x/3x/4x** — inserts 1/2/3 evenly-spaced new frames between every pair
  of real frames, multiplying the frame rate by that exact amount (e.g.
  24fps source -> 48/72/96fps).
- **Custom FPS...** — interpolate to an exact frame rate you choose instead
  (e.g. 24fps source -> 60fps target), even when it isn't a clean multiple
  of the source.
**Cons:** 3x/4x do roughly 2x/3x as many interpolation passes as the 2x
default, so processing time increases proportionally. For a non-integer
Custom FPS target (like 24->60), the new frames aren't perfectly evenly
spaced in time, which can show as very slightly uneven motion smoothness on
some frames.
**Recommended:** 2x for most uses; Custom FPS if you need to match a
specific display or editing timeline's exact frame rate.

#### RIFE Custom Target FPS

**What it does:** The exact output frame rate to interpolate to, used only
when Rate is set to "Custom FPS...".
**Values:** any number higher than your source video's own frame rate (e.g.
60 for a 24fps source). A target at or below the source's frame rate is
rejected.
**Recommended:** 60 for standard smooth-motion displays, or match your
target display/editing timeline's exact refresh rate.

---

## Standalone Tools Tab

Each of these works on a video you've **already converted** — none of them
re-run depth/stereo conversion or touch this app's own GPU/model state. Each
runs as its own background process, and each writes to a brand-new output
file (never overwriting your input) except where noted.

### Retroactive HDR/DV Reinjection

Adds real Dolby Vision/HDR10+ metadata back onto a 3D file you already
converted without "Preserve Dolby Vision" turned on, using your original
source as reference.

#### Original Source File (DV/HDR)

**What it does:** The ORIGINAL video file that still has real Dolby Vision /
HDR10+ metadata — the same file iw3 converted FROM.
**Cons:** a re-encode of your original file may no longer carry the real
DV/HDR10+ metadata at all — point this at your actual master/source file.
**Recommended:** the exact same file (or an identical remux of it) you
originally fed into iw3 for this conversion.

#### Already-Converted 3D File

**What it does:** The iw3 3D output you already made from the source above —
currently SDR because Preserve Dolby Vision wasn't enabled for that
conversion. Read-only.
**Cons:** cannot be output that was already run through RIFE frame
interpolation — RIFE changes the frame count, so it can never line back up
with the source's original timing (this tool detects and refuses that
case).
**Recommended:** the direct, unmodified iw3 output file — not a re-encode or
upscale of it.

#### Output File (HDR Reinjection)

**What it does:** Where to write the new HDR-reinjected copy. Auto-filled
with `<converted file name>_hdr_reinjected<ext>` — change it if you want it
saved elsewhere. Never overwrites the source or converted file.

#### RIFE Manifest (optional)

**What it does:** Only needed when the "Already-Converted 3D File" was ALSO
run through the RIFE Frame Interpolation (Standalone Tool) below. RIFE
changes the frame count, so this tool cannot normally line it back up with
the original source — this manifest (a small
`<rife output>.rife_manifest.json` file the RIFE panel writes next to its
own output) is what lets it expand the Dolby Vision/HDR10+ metadata to match
instead of refusing.
**How it's safe:** leave blank for a normal (non-RIFE) conversion.
**Auto-fill:** picking a converted file with a matching
`.rife_manifest.json` sitting next to it fills this in automatically.
**Recommended:** leave blank unless your converted file came out of the RIFE
panel.

#### Start / End (HDR Reinjection)

**What it does:** Which point in the ORIGINAL SOURCE file the converted
file's first/last frame actually corresponds to — only needed when the
converted file covers just part of the source.
**Cons:** there is no auto-detection — you must know and enter the exact
range that was actually converted. If it's wrong, the tool refuses (an exact
decoded frame-count check) rather than silently producing misaligned HDR
metadata.
**Recommended:** the exact Start/End Time you used for the original iw3
conversion, if any.

#### Run (HDR Reinjection)

**What it does:** Runs the reinjection as a separate background process
(`python -m iw3.reinject_hdr_cli`).
**How it's safe:** before touching anything, it first checks that the
source (trimmed to Start/End Time) and the converted file decode to the
EXACT same number of frames — if they don't match, it refuses and prints
both frame counts to the log instead of producing a mismatched result.
**Cons:** that frame-count check decodes the full clip, so it can take a
while on a long video.
**Recommended:** always check the log box afterward to confirm it actually
succeeded rather than refused.

### Search Subtitles (OpenSubtitles)

Searches and downloads a matching subtitle from OpenSubtitles, then hands it
off to Add Subtitle Track below.

#### Original Source File (optional)

**What it does:** The ORIGINAL, pre-conversion source video — optional. When
given and at least 128KB, its OpenSubtitles moviehash (file size plus a
checksum of only the first and last 64KB) is computed for exact-match
results.
**Why this is a different file from Add Subtitle Track's field:** an
already-converted iw3 SBS/TB output is structurally different and will
essentially never hash-match anything — moviehash only works against the
real original movie file.
**Cons:** files under 128KB can't be hashed this way — falls back to
Title/IMDb ID search automatically.
**Recommended:** point this at the original file you converted from, if you
still have it; otherwise leave blank and use Title/IMDb ID.

#### Title

**What it does:** Movie/show title to search by — optional fallback text
search, used when Original Source File isn't given or didn't match.
**Cons:** text search can return multiple/ambiguous candidates for common
titles — review the results list before downloading.
**Recommended:** the exact title, optionally with year (e.g. "Interstellar
2014") for a more precise match.

#### IMDb ID

**What it does:** Search by IMDb ID (e.g. tt0111161 or 111161) instead of a
text title — optional, more precise than Title when you have it.
**Recommended:** leave blank unless you already know the exact IMDb ID.

#### Language (Search Subtitles)

**What it does:** The subtitle language to search for, as an ISO 639-1
two-letter code (e.g. en, ja, fr, de). Add Subtitle Track's Language field
uses this same two-letter format, so the same code typed in both is always
correct.
**Recommended:** match the language you want the subtitle text to actually
be in; default 'en' if unsure.

#### Search

**What it does:** Searches OpenSubtitles' API for candidates matching
whatever criteria above are filled in — runs in the background.
**How it's safe:** search never spends download quota — only Download
Selected does.
**Cons:** needs a configured OpenSubtitles API key
(`nunif/tmp/opensubtitles_config.json`) — if missing, the log shows how to
register one.
**Recommended:** fill in at least one of Original Source File / Title / IMDb
ID first, then review results before downloading.

#### Results list

**What it does:** Candidate subtitles from the last search, sorted with
plain (non-AI/machine-translated) results first. Columns: Release, Language,
Rating, Downloads, Uploader, Flag.
**Values:** a red "[MT]" Flag marks a machine-translated or AI-translated
result — most users want to avoid these.
**Recommended:** prefer a high-Downloads, high-Rating, unflagged result.

#### Download Selected

**What it does:** Downloads the selected result's .srt file — disabled until
a result is selected.
**Cons:** spends exactly one of your OpenSubtitles daily download-quota
credits, unlike Search which is free.
**How it helps next:** once downloaded, if Add Subtitle Track's Subtitle
File field is still empty, it's auto-filled with the new .srt.
**Recommended:** check the log afterward for the saved file path and your
remaining download quota.

### Add Subtitle Track

Mixes an SRT file into an already-converted 3D .mkv as a normal, selectable
subtitle track.

#### Converted 3D Video (.mkv) [Add Subtitle Track]

**What it does:** The already-converted 3D video to add a subtitle track to.
Must be .mkv — this tool doesn't convert containers.
**Recommended:** the direct iw3 output file, with its normal SBS/TB filename
tag intact (e.g. `..._LR.mkv`) so Format can auto-detect.

#### Subtitle File (.srt)

**What it does:** The SRT file to add as a new track. Validated with
pysubs2 before muxing, so a malformed SRT is caught here with a clear error.
**Cons:** one SRT per run — no multi-language batch support.
**Recommended:** a plain, ordinary SRT — no special stereo formatting needed
(iw3-player already renders subtitles in 3D itself, per-eye, at playback
time).

#### Output File [Add Subtitle Track]

Auto-filled with `<converted file name>_subbed.mkv` — never overwrites the
input video.

#### Format [Add Subtitle Track]

**What it does:** The stereo/output layout of the converted video —
'auto' (default) detects this from its filename using the same tags iw3
itself writes.
**Cons:** if the filename doesn't carry one of those tags (e.g. renamed),
detection is inconclusive and the tool refuses rather than guessing. By
itself this value doesn't change how the subtitle is added; it only matters
together with "Position for external players (dual-eye)" below.
**Recommended:** leave on 'auto' unless the log reports it couldn't detect
the format.

#### Language [Add Subtitle Track]

**What it does:** The language stored as metadata on the new subtitle track,
as an ISO 639-1 two-letter code — the same format as Search Subtitles'
Language field. Converted internally to the three-letter code mkvmerge
needs.
**Cons:** purely metadata — does not translate or verify the actual subtitle
content's language.
**Recommended:** match the SRT file's actual language; default 'en' if
unsure.

#### Track Name [Add Subtitle Track]

**What it does:** An optional display name for the new subtitle track (e.g.
"English (Forced)"). Leave blank to default to the SRT file's own name.

#### Position for external players (dual-eye)

**What it does:** Fixes a real problem when playing the output in an
EXTERNAL player (VLC, MPC-HC, a TV's built-in player, etc.) — not iw3-player.
On a split-eye Format, a plain subtitle is centered against the WHOLE frame
by a normal player, which lands it right on the seam between the two eyes
and tears every line of text in half. Turning this on converts the subtitle
into a track with two copies per line, one positioned inside each eye-half.
**Why it defaults ON:** most output from this project is watched in an
external player, not iw3-player, so this checkbox defaults to the setting
that's correct there. The tradeoff: iw3-player already does its own correct
per-eye rendering of a PLAIN subtitle track, so if this stays checked,
iw3-player will show each line twice, wrongly positioned.
**Cons:** no effect for Format rgbd, half_rgbd, or anaglyph — those aren't a
two-eye split. Also requires probing the video's real width/height via
ffprobe before muxing.
**Recommended:** leave ON for external players (VLC, MPC-HC, TVs, etc.) —
the common case. Uncheck ONLY if you'll specifically watch the result in
iw3-player.

#### Font Size [Add Subtitle Track]

**What it does:** Fixes a real bug — subtitles added with "Position for
external players (dual-eye)" checked used to render TINY, because the
underlying library's own fixed default text size (20px) was never designed
for this project's typical output resolutions (e.g. a 3840x2076 double-wide
4K SBS frame, where 20px is under 1% of the frame's height). Left blank (the
default, recommended), text size now automatically scales with the video's
real resolution.
**Pros:** automatic sizing already accounts for Top/Bottom formats needing
smaller text than Side-by-Side formats at the same resolution, since each
eye only gets half the vertical space in a Top/Bottom video.
**Cons:** only affects the dual-eye track — no effect if that checkbox is
unchecked, and no effect for rgbd/half_rgbd/anaglyph. Does not affect
iw3-player's own subtitle text (that has its own separate size control in
the player itself).
**Values:** any positive number (pixels); typical readable sizes run roughly
40-60px at 1080p and 80-120px at 4K.
**Recommended:** leave blank so the size always matches the video's real
resolution automatically.

#### Start Time / End Time [Add Subtitle Track]

**What it does:** Solves this problem — you converted only a TRIMMED CLIP of
a full movie with iw3's own Start Time/End Time, but Subtitle File has
timestamps for the FULL movie. Without this, a line originally at 1:16 in
the full movie would still say 1:16 in the new track, even in a 5-minute
clip that has already ended. Checking this and entering the SAME Start Time
you used for the original iw3 conversion trims Subtitle File to that point
onward and REBASES it so that point becomes 0:00.
**Cons:** there is no auto-detection — you must know and enter the exact
point that matches the clip's first frame. A cue straddling this point is
clipped rather than dropped. For End Time without Start Time checked, Start
Time is treated as 00:00:00.
**Recommended:** leave unchecked when Subtitle File already matches the
converted video's own timeline. Check it and match iw3's own Start/End Time
exactly when Subtitle File is for the full source instead.

#### Run [Add Subtitle Track]

**What it does:** Runs the mux as a separate background process (`python -m
iw3.subtitle_mux_cli`).
**How it's safe:** every existing track (video, audio, existing subtitles)
is copied into the output completely unchanged — only the new subtitle
track is added.
**Cons:** if Format can't be auto-detected, this refuses immediately rather
than guessing SBS vs TB.
**Recommended:** check the log afterward to confirm it actually succeeded.

### Add Audio Track

Adds an extra audio track (e.g. a different-language dub) to an already-
converted 3D .mkv.

#### Converted 3D Video (.mkv) [Add Audio Track]

**What it does:** The already-converted 3D video to add an audio track to.
Must be .mkv.
**Recommended:** the direct iw3 output file.

#### Audio File (dub)

**What it does:** The audio file to add as a new track. Common formats
mkvmerge/ffmpeg already read directly work here: AAC, AC3, DTS, FLAC, MP3,
Opus, WAV, and more.
**Cons:** read-only. If it covers more content than the video (e.g. a
full-movie dub for a short clip), use Source Start/End Time below to trim it
automatically rather than pre-cutting it by hand.
**Recommended:** match the audio's actual length/content to the video as
closely as you can.

#### Output File [Add Audio Track]

Auto-filled with `<converted file name>_dubbed.mkv` — never overwrites the
input video or audio file.

#### Language [Add Audio Track]

**What it does:** The language of the new audio track, as an ISO 639-1
two-letter code — converted internally to the three-letter code mkvmerge
stores as track metadata.
**Cons:** purely metadata — does not translate or verify the actual audio
content's language.
**Recommended:** match the audio file's actual language; default 'en' if
unsure.

#### Track Name [Add Audio Track]

An optional display name for the new audio track (e.g. "Spanish Dub").
Leave blank to default to the audio file's own name.

#### Set as default track

**What it does:** Marks the new audio track as the one a player selects
automatically, instead of just adding it as a selectable alternate.
**Cons:** any existing audio track's own default flag is left exactly as it
already was — if it was ALSO marked default, some players may then see two
default audio tracks and pick whichever one they encounter first.
**Recommended:** off (default) — usually you're adding an alternate-language
track, not replacing the primary audio.

#### Source Start / Source End

**What it does:** Trims Audio File to start/end at this point, for when the
audio source covers more content than the video (e.g. a full-movie dub
being added to a short test clip). The trimmed audio is also automatically
shifted to start at t=0.
**Cons:** there is no auto-detection — you must know and enter the exact
range within the audio file that matches the video.
**Recommended:** leave unchecked to use the whole audio file as-is, unless
you know the exact matching range.

#### Run [Add Audio Track]

**What it does:** Runs the mux as a separate background process (`python -m
iw3.audio_mux_cli`).
**How it's safe:** every existing track (video, existing audio, subtitles)
is copied into the output completely unchanged.
**Cons:** if Source Start/End Time is set, trimming re-runs ffmpeg first,
which can take a little longer.
**Recommended:** check the log afterward to confirm it actually succeeded.

### Retroactively Tag MKV as 3D (Stereo Mode Tag)

Writes standard 3D metadata into a finished file so 3D-aware players/TVs can
automatically detect and display it in 3D.

#### Converted 3D Video (.mkv) [Stereo Mode Tag]

**What it does:** The already-converted 3D video to tag. Must be .mkv.
**Cons:** this file IS edited in place (unlike the other standalone tools,
which always write a separate new file) — mkvpropedit only rewrites
container metadata, never re-encodes or re-muxes the actual video/audio.
Use Backup below if you still want a safety copy.
**Recommended:** the direct iw3 output file, with its normal SBS/TB filename
tag intact so Format can auto-detect.

#### Format [Stereo Mode Tag]

**What it does:** The stereo/output layout of the video. 'auto' (default)
detects this from its filename.
**Cons:** rgbd/half_rgbd/anaglyph are listed here so detection can name
them, but tagging always refuses for those three (not a two-eye stereo
pair, or already correct without tagging).
**Recommended:** leave on 'auto' unless the log reports it couldn't detect
the format.

#### Backup before editing

**What it does:** Copies the input file to `<name>.mkv.bak` before tagging,
as an extra safety net.
**Cons:** uses extra disk space equal to the whole input file, and takes
time to copy on a large file.
**Recommended:** off (default) is fine for most people — mkvpropedit's edit
only touches metadata, never the actual video/audio. Turn on if you'd rather
have a copy just in case.

#### Run [Stereo Mode Tag]

**What it does:** Runs the tagging as a separate background process
(`python -m iw3.stereo_mode_tag_cli`).
**Cons:** edits the input file in place — see that field's own note.
**Recommended:** check the log afterward to confirm it actually succeeded
rather than refused.

### Sharpen (Standalone Tool)

Applies the same edge-aware sharpening filter the main pipeline offers,
directly to a finished video — no need to re-run the whole conversion.

#### Converted 3D Video (.mkv) [Sharpen]

**What it does:** The already-converted 3D video to sharpen. Must be .mkv —
mkvmerge is what guarantees every other track (audio, subtitles, chapters,
attachments) is copied through byte-for-byte unchanged around the freshly
re-encoded video, and that's only available for Matroska.
**Recommended:** the direct iw3 output file, with its normal SBS/TB/RGBD/
Anaglyph filename tag intact so Format can auto-detect.

#### Output File [Sharpen]

Auto-filled with `<converted file name>_sharpened.mkv` — never overwrites
the input video.

#### Format [Sharpen]

**What it does:** The stereo/output layout of the converted video — needed
so this tool knows how to split it before sharpening (a genuine two-eye
layout is split into its eye halves and each is sharpened independently,
never across the seam; RGBD/Half RGBD only sharpens the RGB half, since the
other half is a depth map; Anaglyph has no seam and is sharpened as one
whole frame). 'auto' (default) detects this from the filename.
**Cons:** if the filename doesn't carry a recognized tag, auto detection is
inconclusive and the tool refuses rather than guessing.
**Recommended:** leave on 'auto' unless the log reports it couldn't detect
the format.

#### Strength [Sharpen, Standalone]

**What it does:** How strong the Sharpen effect is — the exact same
edge-aware unsharp-mask filter, range, and default (0.5) as the in-pipeline
Sharpen control on the Stereo Generation tab, just applied here to an
already-converted file instead of during conversion.
**Values:** higher = more pronounced detail boost at real edges/texture, but
also more risk of an over-crisp/harsh look or exaggerating real compression
artifacts. 0.0 is an exact no-op (the video track is still fully re-encoded,
but every pixel is left unchanged) — useful only for testing.
**Recommended:** 0.5 (default) as a safe starting point.
**Values:** 0.0 to 1.0 (0.25 / 0.5 / 0.75 / 1.0 preset choices).

#### Run [Sharpen]

**What it does:** Runs the sharpen pass as a separate background process
(`python -m iw3.sharpen_cli`).
**How it's safe:** every existing track (audio, subtitles, chapters,
attachments) is copied into the output completely unchanged — only the
video track is re-encoded, and only its pixels are touched.
**Cons:** if Format can't be auto-detected, this refuses immediately.
Re-encoding the video track takes time proportional to the video's length.
**Recommended:** check the log afterward to confirm it actually succeeded.

### RIFE Frame Interpolation (Standalone Tool)

The same RIFE tool as the in-pipeline Post-Processing option, run
retroactively against a video you've already converted.

#### Converted 3D Video [RIFE Standalone]

**What it does:** An already-converted 3D video to smooth the motion of.
RIFE is a separate AI model that creates new, genuinely synthesized
in-between frames (not simple frame duplication/blending) so motion looks
smoother at a higher frame rate.
**Cons:** RIFE interpolates the FINAL PACKED stereo frame (both eyes already
combined) as one image, so it will see the seam between the two packed eyes
— it wasn't trained on that, though in practice both eyes move together so
this doesn't cause left/right desync (same accepted tradeoff as the
in-pipeline RIFE step).
**Recommended:** the direct iw3 output file you already made.

#### Output File [RIFE Standalone]

**What it does:** Auto-filled with `<converted file name>_rife<ext>`. Never
overwrites the input video. Also always writes a small
`<output>.rife_manifest.json` file next to it, recording which output
frames are real and which are RIFE-synthetic — needed by the Retroactive
HDR/DV Reinjection tool if this file also has Dolby Vision to fix
afterward.

#### RIFE Model [Standalone]

Same two tiers and tradeoff as the in-pipeline RIFE step: rife_425 (better
motion accuracy, higher compute cost) vs rife_425_lite (lower cost, less
accurate). Weights download automatically on first use.
**Recommended:** rife_425 unless interpolation time is a real bottleneck.

#### Rate [RIFE Standalone]

Same behavior as the in-pipeline Rate dropdown: 2x/3x/4x insert evenly-
spaced frames; Custom FPS targets an exact frame rate (must be higher than
the source's own rate, or the tool refuses).
**Recommended:** 2x for most uses; Custom FPS to match a specific target
rate.

#### GPU [RIFE Standalone]

**What it does:** Which GPU (or CPU) runs this tool's own RIFE model —
reuses the Device selector's convention from the Processor tab, but without
"All CUDA Device": this tool runs as one single background process and
`iw3.rife_cli`'s own `--gpu` option only ever targets one device.
**Cons:** CPU works without a GPU but is dramatically slower.
**Recommended:** your main GPU (the first entry) unless you're deliberately
running this alongside another GPU job and want to keep them on separate
devices.

#### Output Codec [RIFE Standalone]

**What it does:** Which video format this tool encodes its OUTPUT with.
**Why you would change it:** RIFE's own output has always defaulted to
H.264, unrelated to Dolby Vision/HDR entirely — leave this on the default
for that. But if the video you're interpolating has Dolby Vision or HDR10+
and you plan to fix that metadata afterward with the Retroactive HDR/DV
Reinjection tool (its "RIFE Manifest" field is built for exactly this), that
tool can ONLY inject into an HEVC (H.265) file — RIFE's H.264 default output
can NEVER accept that metadata, no matter what. Pick an HEVC option here
FIRST if that's your plan.
**Values:**
- **H.264 (default)** — RIFE's normal, unrelated-to-HDR output.
- **H.265/HEVC — libx265 (CPU)** — software encode, works on any machine,
  slower and produces a larger file than the H.264 default at the same
  quality setting.
- **H.265/HEVC — hevc_nvenc (GPU)** — hardware encode on the GPU selected
  above, much faster than libx265, requires an NVIDIA GPU with NVENC
  support.
**Cons:** HEVC output is somewhat less universally compatible with older/
non-4K playback devices than H.264, and both HEVC options produce a larger
or slower-to-produce file than the H.264 default.
**Recommended:** leave on the default (H.264) unless you specifically plan
to run the Retroactive HDR/DV Reinjection tool afterward — then pick libx265
(works everywhere) or hevc_nvenc (faster, if your GPU supports it).

#### Run [RIFE Standalone]

**What it does:** Runs RIFE interpolation as a separate background process
(`python -m iw3.rife_cli`).
**How it's safe:** writes to a new output file only; a
`<output>.rife_manifest.json` sidecar is always written alongside it.
**Important — Dolby Vision/HDR:** RIFE itself does NOT touch DV/HDR10+
metadata at all. If the video you're interpolating has Dolby Vision, first
set Output Codec above to an HEVC option, then use the Retroactive HDR/DV
Reinjection tool, pointing its "Converted" field at this Run's output and
its "RIFE Manifest" field at the `.rife_manifest.json` sidecar, against your
ORIGINAL Dolby Vision source. Skipping either step means the output plays
back without correct Dolby Vision metadata.
**Recommended:** check the log afterward to confirm it actually succeeded,
and note the printed manifest file path if you'll need it for Dolby Vision.

---

## Bottom Bar

#### Quick Preview

**What it does:** Processes a short 45-second clip (or a single frame for
images) with the current settings to quickly check the result.

#### Start / Suspend / Cancel

Start begins the conversion with your current settings. Suspend pauses a
running job (see "Free GPU memory while paused" on the Processor tab for
what happens to VRAM while paused). Cancel stops the job. These three
buttons don't carry their own detailed tooltips in the app beyond their
plain button labels.

---

## A note on accuracy

Every explanation above was copied from the real, current `SetToolTip(...)`
text in `iw3/gui.py` (and the shared `VideoEncodingBox`/`VideoDecodingBox`
components in `nunif/gui/`) as of this write-up — not from memory, and not
from an older cached understanding of what a setting used to do. Where a
tooltip referenced a specific default value, table, or behavior (e.g. Auto
EMA's default table selection, Sharpen's strength range, HWAccel's Software
Fallback default), that value was cross-checked directly against the
current code rather than assumed. If a future update to the app changes a
tooltip, this guide will drift out of date exactly the way any static
document does — when in doubt, the live tooltip in the running app is
always the most current source.
