# 3DECKER — iw3

## Welcome

**3DECKER** is this project's own name for its customized build of **iw3**, a
free tool that converts ordinary flat ("2D") video and images into stereoscopic
3D — the kind of 3D you'd watch with a 3D TV, VR headset, or 3D glasses. It uses
AI to estimate how far away everything in a picture is, then uses that
information to generate a second, slightly shifted view for your other eye.

This distribution also bundles a few other tools alongside the main 3D converter:

- **waifu2x** — an AI image upscaler/quality-improver, for making images (or
  video frames) sharper and higher-resolution.
- **iw3-player** — a media player built specifically for watching your
  converted 3D videos, with its own subtitle and playback handling.
- **iw3-desktop** — converts your live desktop or game screen into 3D in real
  time, rather than converting a saved video file.
- **stlizer** — converts a 2D image into a 3D-printable model file.

Everything runs from a self-contained copy of Python and its supporting tools
(ffmpeg for video, MKVToolNix for the video container format, and others) that
comes bundled with this project — you don't need to install anything separately
to get started. Just use the `.bat` launcher files (like `iw3-gui.bat`) in the
project folder.

This guide covers the main iw3 app (the 3D converter) in enough detail to get
you started confidently. For the full, detailed explanation of any individual
setting, **hover your mouse over it in the app** — every control has its own
tooltip explaining what it does, why you'd want it, and what's recommended.

---

## The Basic Workflow: Picking a Video and Converting It

At its heart, using iw3 is three steps:

1. **Pick your input.** Point the app at a video file (or a folder of images/videos
   for batch conversion). You'll also choose where the finished 3D file should go.
2. **Choose your settings.** The app is organized into tabs — things like which
   AI depth model to use, how strong the 3D effect should be, how the two eye
   views should be packed into the output file, and a large number of optional
   quality/stability features (covered below). If you don't want to dig into
   every setting yourself, the **Quick Preset** buttons ("Movie," "Action," or
   "3DECKER Preferred") apply a full, tested combination of settings in one
   click — a great starting point.
3. **Click Start.** The app converts the video in the background, showing
   real progress (which stage it's on, how fast it's going, and roughly how
   long is left). You can Pause/Suspend or Cancel a running job at any time.

That's the shape of it. The dozens of individual settings you'll see beyond that
are all about *how* the conversion happens — depth quality, smoothing, edge
handling, output format — and every one of them has its own detailed tooltip
right there in the app, so you never have to guess what a field does.

A couple of orientation notes worth knowing up front:
- **Presets** (top toolbar) let you save your own favorite settings combination
  and reload it later, in addition to the built-in Quick Presets.
- **Compare Presets** renders a short test clip with two or more presets and
  joins the results into one video, so you can directly compare different
  settings before committing to a full-length conversion.
- The app can take a while on a long movie, especially with the higher-quality
  options turned on — that's normal for AI-based video processing on a single
  computer, not a sign something is wrong.

---

## The Standalone Tools Tab

Beyond the main conversion, iw3 includes a set of standalone tools for working
with a video *after* it's already been converted — so if you just want to tweak
one thing, you don't have to sit through the entire (often lengthy) 3D
conversion process again. You'll find these under the **Standalone Tools** tab.

- **Retroactive HDR/DV Reinjection** — already converted a Dolby Vision or HDR
  movie without turning on "Preserve Dolby Vision"? This tool adds that
  metadata back onto your finished file afterward, using your original source
  file as the reference. See the Dolby Vision section below for the current
  limitations.
- **Search Subtitles (OpenSubtitles)** — searches and downloads a matching
  subtitle file for your movie directly from OpenSubtitles, then hands it
  straight off to the Add Subtitle Track tool below.
- **Add Subtitle Track** — mixes a subtitle file into your converted video as a
  normal, selectable subtitle track. By default it positions subtitles so they
  display correctly on external 3D players and TVs (not just this project's
  own player), since that's how most people actually watch their output.
- **Add Audio Track** — adds an extra audio track, like a different-language
  dub, to an already-converted video.
- **Retroactively Tag MKV as 3D (Stereo Mode Tag)** — writes standard 3D
  metadata into a finished file so 3D-aware players and TVs (VLC, Kodi, many
  smart TVs) can automatically detect and display it in 3D, instead of you
  having to manually switch the player into 3D mode.
- **Sharpen** — already converted a video and it looks a little soft? This
  applies the same subtle, edge-aware sharpening filter the main pipeline
  offers, directly to your finished video — no need to re-run the whole
  conversion just for a sharpness tweak.
- **RIFE Frame Interpolation** — smooths out motion by generating genuinely new
  in-between frames, raising the frame rate (for example, 24fps up to 48fps).
  Like Sharpen, this works on a video you've already converted, so you can add
  smoother motion after the fact without redoing the conversion.
- **Run Update** (top toolbar, not this tab) — actually applies the latest
  update (new packages, AI models, and source code together) from inside the
  app, with a confirmation prompt first. A separate, read-only **Check for
  Updates** button lets you see if an update is available without changing
  anything.

Every tool above works on a *copy* — none of them ever modify your original
files. Each has its own log box (with a Clear button) showing exactly what
happened, so you can always confirm a step actually succeeded.

---

## Keeping the 3D Effect Steady: Flicker, Wobble, and Smoothing

Here's a problem worth understanding before you dive into these settings: AI
depth models don't always agree with themselves from one frame to the next.
Even on a perfectly still shot, the estimated depth can subtly shift frame to
frame — which shows up as a flickering, wobbling, or "breathing" 3D effect,
especially with faster, less video-aware depth models. This project built a
whole toolkit to fight that, and it's genuinely one of the more important
areas to get comfortable with.

You don't need to understand the math behind any of this to use it well —
think of these as different "how steady should the depth be" dials, applied at
different levels:

- **Flicker Reduction** smooths the overall near/far depth *range* over time —
  think of it as smoothing out sudden jumps in how "deep" the whole scene feels,
  frame to frame. You can set this by hand, or use a **Genre Preset** (Fast
  Action / Medium-Magical / Drama-Slow-Paced) as a quick starting point.
- **Object Stability** works differently — it tracks individual objects as they
  move and smooths *their own* depth over time, so a specific object's depth
  doesn't flicker even while everything else stays responsive.
- **Auto EMA by Scene Length** is the more "hands-off" option: rather than
  picking one fixed smoothing strength for an entire movie, it automatically
  adjusts smoothing strength based on *each individual scene's own length* — a
  short scene gets lighter smoothing so it doesn't lag, a long continuous shot
  gets more. There are several built-in tables to choose from (tuned for
  different depth models and different pacing styles), and you can view or
  edit the exact numbers each one uses if you want to fine-tune it for your
  own footage.
- The **"3DECKER Preferred"** button (top toolbar) is the simplest way to try
  all of this at once — it's a one-click preset applying this project's own
  tested combination of depth, stability, and smoothing settings together.
- **Scene Boundary Detection** makes all of the above smarter: it detects real
  cuts in the video so smoothing resets cleanly at each new scene instead of
  blending two completely different shots together.

**If you're not sure where to start:** try the "3DECKER Preferred" quick
preset first, watch the result, and only dig into the individual Flicker
Reduction / Object Stability / Auto EMA fields by hand if you notice a specific
problem (visible flicker, or the opposite — visible lag/smearing) you want to
correct. Every field's own tooltip explains exactly what raising or lowering it
does and why.

There's also a fully automated **Scene Batch** mode, which splits a whole movie
into individual scenes, converts each one completely independently (so one
scene's settings or problems can never bleed into the next), and stitches
everything back together with the original audio — a good option if you want
the most hands-off, scene-aware conversion possible.

---

## Dolby Vision and HDR — What Works Today

If your source video has Dolby Vision or HDR10+ (common on many 4K movies), iw3
can preserve that during conversion — turn on "Preserve Dolby Vision" and the
app will extract that metadata before converting and re-inject it into the
finished 3D file afterward.

**A real, current limitation worth knowing:** Dolby Vision/HDR10+ preservation
and RIFE frame interpolation cannot both happen within the *same* conversion
run — RIFE creates brand-new synthetic frames that have no HDR metadata of
their own to carry. The good news is there's now a working two-step
alternative: convert your video with Dolby Vision preserved, run RIFE
afterward (with its output format set to an HEVC/H.265 option, which the
reinjection step requires), and then run the Retroactive HDR/DV Reinjection
tool one more time, pointing it at the small manifest file RIFE writes. This
full path has been tested end-to-end on real Dolby Vision movie footage. One
remaining gap: this specific RIFE workflow currently carries Dolby Vision
metadata through correctly, but not HDR10+ metadata.

If you didn't turn on Dolby Vision preservation for a conversion you already
ran, you don't have to redo it — the standalone **Retroactive HDR/DV
Reinjection** tool (Standalone Tools tab) can add it back afterward, as long as
you still have your original source file.

---

## Where to Go for More Detail

- **In-app tooltips** are the authoritative, up-to-date reference for every
  individual setting — what it does, when to change it, and what's
  recommended. If you're ever unsure what a checkbox or slider does, hover
  over it before searching anywhere else.
- **`3DECKER_CHANGELOG.md`** (in this same folder) lists everything that's been
  built or fixed on top of the original iw3 tool, organized by topic, in the
  same plain-language style as this guide.

Enjoy converting your movies into 3D.
