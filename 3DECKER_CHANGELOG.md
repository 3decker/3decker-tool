# 3DECKER — iw3

## What's New

This is the changelog for **3DECKER**, this project's own name for its heavily
customized build of iw3 (the 2D-to-3D video converter). Everything below was built
or fixed on top of the original, free iw3/nunif tool. It's written in plain
language, grouped by what each thing actually does for you — not as a technical log.

For what every individual setting does, hover over it in the app — every control has
its own detailed tooltip. This file is the bigger picture: what's changed and why
it matters.

---

## Latest Update — September 14, 2026

**4 new Depth-Anything-3 depth models** (Small, Base, Large-1.1, and Metric-Large)
join the depth model list, matching what VisionDepth3D already offered — plus 8
more existing depth models (Video Depth Anything Base/Large, its Streaming
counterparts, and Distill Any Depth Base/Large) that now download themselves
automatically the first time you pick them. Previously several of these couldn't
even be selected on a fresh install because there was no way for them to download
in the first place.

**A real GPU memory problem was tracked down and fixed.** If you used Flicker
Reduction with a high Buffer setting, a large pool of GPU memory could build up
during a conversion and never come back down afterward — only closing the whole
app released it. This is now fixed and releases itself automatically the moment
a job finishes, confirmed on a real clip: usage went from over 20GB stuck-forever
down to under 100MB right after the fix.

**Every setting's title now shows a tooltip on hover, not just its input box.**
100% of this project's settings are documented now, both label and control.

**Several confusing messages were fixed**, including updates that reported a
false "Error" even after actually succeeding, and a leftover "Press any key to
continue" message after updates that did nothing no matter what you pressed.
torch.compile also now explains itself instead of silently refusing to turn on
when your Device dropdown is set to "All CUDA Device."

**A new "What's New" button shows this changelog right inside the app**, next
to the update-check buttons — no more needing to leave the app or hunt for
this file on disk to see what's changed.

**Object Stability got a new "Speed" option**, and the window can now be
resized much smaller. Object Stability's motion tracking runs on your CPU
and can noticeably slow a conversion down — a new Accurate/Fast choice lets
you trade a little motion-tracking precision for real speed. Separately, the
toolbar and the Start/Suspend/Cancel row now wrap onto extra rows instead of
getting cut off, so the window can be dragged down to about 300px wide
instead of stopping at over 640px.

---

## Update — September 10, 2026

**A friendlier main screen.** The Stereo Generation tab (the one you use on every
conversion) now has real sliders you can drag for the settings you adjust most — 3D
Strength, Convergence, Edge Suppression, and more — instead of only typing numbers into
a box. The long lists of less-common settings are now tucked behind small, clickable,
color-coded sections (like "Pop & Divergence" or "Stability & Flicker") that stay
collapsed until you actually need them, so the tab isn't a wall of controls anymore.
Dual-Pass Depth Blend, Video Filter's batch-processing options, and every Standalone
Tool got the same treatment — each one is now a compact card you expand only when
you're using it.

**Updating the app is clearer now.** There used to be one "Check for Updates" button
that was ambiguous about which project's updates it was actually looking at. That's
been split into two genuinely separate, unmistakable buttons: **"Check for 3DECKER
Updates"** (this project's own new features, UI changes, and fixes) and **"Check for
Nagadomi Updates"** (the original open-source nunif project this tool is built on).
Each one shows you exactly what's new before anything installs, with its own
**"Install Update Now"** button right on that same screen — no more guessing which
button does what, and no separate step needed to actually apply what you just saw.

**The app also quietly lets you know when an update is available**, right when you
open it — no popup at all if you're already current, so this never gets in your way.

**Naming and branding cleanup.** The window title now simply reads "3DECKER" instead
of a long technical string. The app launcher was renamed from `iw3-gui.bat` to the
correct `3decker-gui.bat`, and the app now has its own dedicated icon.

---

## 3D Conversion Quality

**Dual-Pass Depth Blend — combining two AI "depth" models into one better result.**
A single AI depth model is fast, but it can miss fine detail like hair or foliage,
or simply have its own particular weaknesses. Dual-Pass Depth Blend runs *two*
different depth models on your video and blends their results together frame by
frame, leaning on the second model specifically where it helps most (like sharp
edges) without letting it override the first model everywhere. The two models are
always loaded one at a time, never both at once, so this doesn't require double the
graphics card memory. Every step of this process saves its own progress, so if it's
interrupted (a crash, a reboot, closing the app) it picks back up instead of
starting the whole thing over.

**Subpixel Z-Splat — smoother edges where objects overlap.** When the 3D
conversion shifts pixels sideways to create the 3D effect, two different source
pixels can occasionally land on the exact same spot. The old behavior picked one
winner and threw the other away, which could create a slightly hard, jagged edge.
This new option (available as a stereo rendering method) blends the two pixels
together based on which one is genuinely closer to the camera, producing a
softer, more natural edge instead of a hard cutoff.

**Depth Detail Refinement — cleaning up noisy depth without smearing it.** Depth
models sometimes produce small speckles of noise within a single frame. This
feature smooths that out using a technique (a "bilateral filter") that respects
real edges instead of blurring across them. We also tried adding a contrast-boosting
step (CLAHE) on top of this — real testing showed it actually made noise worse, not
better, so it was dropped entirely rather than shipped as a half-working option.

**Edge Suppression and Edge Repair — reducing double edges and fringing.** When
Dual-Pass Depth Blend combines two models, they sometimes disagree slightly on
exactly where an edge is, which can create a faint doubled or ghosted outline.
Edge Suppression turns down the second model's influence right at those edges so
they don't fight each other. Edge Repair is a separate, final cleanup pass that
works on *any* conversion method (not just Dual-Pass Blend) — it looks at the
finished 3D picture, finds real depth edges, and gently smooths a thin band around
them to reduce fringing. Both are adjustable, off by default until you turn them on.

**Depth Scale Alignment — making sure both models agree on "how far away."**
Different depth models don't always use numbers the same way — one model's "far"
might not line up with another's. This optional pre-blend step lines up the two
models' depth ranges so blending them doesn't create a visible seam.

**Object Stability — reducing depth "wobble" on individual objects.** Beyond the
overall smoothing described below, Object Stability tracks how objects actually
move (using real motion tracking on the picture, not the depth map) and carries
each object's own depth history along with it as it moves. This has extra dials —
a hard cap on how much depth can jump frame to frame, extra smoothing in flat
areas where flicker is most visible, and reduced smoothing right at real depth
edges so fast motion doesn't lag or smear. A new **Speed** option (Accurate/Fast)
lets you trade a bit of motion-tracking precision for real conversion speed —
this motion tracking runs on your CPU rather than the graphics card, so it can
meaningfully slow a conversion down; Fast eases that cost for anyone who wants
Object Stability without the full time penalty.

---

## Depth Models

**4 new Depth-Anything-3 options.** Small, Base, Large-1.1, and Metric-Large join
this project's existing depth model list, matching what VisionDepth3D already
offered. A crash that happened when picking Large-1.1 on a real conversion was
found and fixed shortly after adding it — it's now working correctly, alongside
the also-new Metric-Large.

**8 more depth models now download automatically when you pick them**, the same
way the Depth-Anything-3 models above already did — the Base and Large sizes of
Video Depth Anything (regular and Streaming), plus Distill Any Depth Base/Large.
Previously these couldn't even be selected on a fresh install, since the file
they needed had no way to download itself; now picking one just works, and the
file downloads the first time you use it. (Two similarly-named, larger models
stay manual-download-only, since their license specifically requires you to
download them yourself rather than this app doing it for you.)

**A new "Resolution Preset" quick-fill dropdown** sits next to Depth Resolution,
letting you pick from the same 21 resolution values VisionDepth3D offers instead
of needing to know the right number to type in yourself.

---

## Keeping the 3D Effect Steady (Flicker Reduction & Scene Awareness)

See the README for a plain-language introduction to this whole area. In short: 3D
depth can flicker or "breathe" from frame to frame, especially with faster, less
memory-aware depth models. This project added a full smoothing toolkit, tuned and
re-tuned many times based on real footage:

- **Flicker Reduction** with an adjustable Decay Rate and Buffer Size, plus a
  **Genre Preset** quick-fill (Fast Action / Medium-Magical / Drama-Slow-Paced) so
  you don't have to guess starting numbers.
- **Motion-Adaptive Smoothing**, which automatically eases up on smoothing during
  fast motion (never the reverse), so action scenes don't look laggy.
- **Scene Boundary Detection**, so smoothing resets cleanly at every real cut
  instead of blending two unrelated shots together.
- **Auto EMA by Scene Length** — instead of one fixed smoothing setting for an
  entire movie, this automatically picks a smoothing strength based on *each
  individual scene's own length*, using one of nine selectable built-in tables
  (including two tables purpose-built for this project's most-used depth models,
  a conservative option anchored to the original iw3 author's own tested values,
  and several experimental options for side-by-side comparison). Every table is
  fully editable and remembers your changes.
- A one-click **"3DECKER Preferred"** button in the toolbar applies this project's
  own confirmed-best combination of settings across depth, stability, and
  smoothing at once, for anyone who wants a strong starting point without tuning
  each field by hand.
- The **Automated Scene Batch** pipeline (a separate, fully hands-off mode) splits
  a whole movie into individual scenes, converts each one independently with a
  genuinely fresh start (so problems in one scene can't leak into the next), and
  joins everything back together with the original audio re-attached.

---

## Frame Smoothing (RIFE)

**What RIFE does:** RIFE is a separate AI model that creates brand-new,
genuinely synthesized in-between frames (not simple duplication or crossfading),
so motion looks smoother at a higher frame rate — for example, taking a 24fps
movie up to 48fps.

- Available both as an **optional step in the main conversion** and as a
  **standalone tool** you can run afterward on a video you already converted, so
  you don't have to redo the whole conversion just to add smoother motion.
- Choose a simple **2x / 3x / 4x** frame-rate multiplier, or an **exact target
  frame rate** of your choosing (for example, matching a specific display's
  refresh rate).
- Two quality tiers are available — a full-accuracy model and a faster, lighter
  one — with weights downloaded automatically the first time you use them.
- A known, accepted trade-off: RIFE processes the finished, combined stereo
  picture (both eyes already packed together), so it never causes left/right eye
  desync, but it also can't "see" each eye separately.

---

## Dolby Vision & HDR

- The main conversion pipeline can **preserve Dolby Vision and HDR10+ metadata**
  end to end — extracting it from your original source before conversion and
  re-injecting it into the finished 3D output afterward.
- A **standalone Retroactive HDR/DV Reinjection tool** lets you add Dolby
  Vision/HDR10+ metadata onto a video you already converted without Dolby Vision
  turned on, without having to redo the conversion.
- **RIFE and Dolby Vision now work together.** For a while, RIFE-interpolated
  video couldn't carry Dolby Vision at all (RIFE's synthetic in-between frames
  have no metadata of their own). This is now solved with a two-step retroactive
  workflow: RIFE writes a small file recording which frames are real and which
  are synthetic, and the reinjection tool uses it to correctly duplicate the
  right metadata onto the new frames. This was tested end-to-end on real Dolby
  Vision movie footage and confirmed correct, frame by frame. Note: HDR10+
  metadata specifically is not carried through this RIFE path today, only Dolby
  Vision; and RIFE and Dolby Vision preservation still can't both happen within
  the *same* single conversion pass — the retroactive two-step workflow is
  required.
- Several real reliability issues with Dolby Vision injection (files that were
  correctly encoded but briefly "locked" by Windows right after being written,
  causing an occasional false failure) were tracked down and fixed.

---

## Subtitles

- A standalone **Add Subtitle Track** tool mixes a subtitle file into an
  already-converted 3D video as a normal, selectable subtitle track — no need to
  re-run the conversion.
- A standalone **Search Subtitles** tool searches and downloads matching
  subtitles from OpenSubtitles directly, by your original source file or by
  title, and hands the result straight to the Add Subtitle Track tool.
- Subtitles are positioned to display correctly on **external 3D players and
  TVs** by default (duplicating each subtitle line into both eye halves of the
  picture), since most output from this tool is watched outside this project's
  own player.
- Subtitle text size now automatically scales to your video's actual resolution
  — it used to render at a fixed small size that became nearly unreadable on
  high-resolution output.
- Optional trimming lets you line up subtitles sourced from a full movie with a
  shorter, trimmed clip.

---

## Audio

- A standalone **Add Audio Track** tool adds an extra audio track (for example, a
  different-language dub) to an already-converted video, with optional trimming
  if the audio source covers more content than the video.

---

## Sharpening

- A new **Sharpen** filter adds a subtle, edge-aware detail boost to the
  finished 3D picture. It's designed to avoid amplifying film grain or noise —
  it specifically targets real edges and texture, not random speckle.
- Available both as an **in-pipeline option** during conversion and as a
  **standalone tool** for videos you've already converted.

---

## Upscaling

- An optional **automatic waifu2x upscale** can run right after conversion,
  specifically designed for 3D video: it splits the video into its two eye
  views, upscales each one independently (so detail doesn't blur across the
  seam between eyes), smooths the result over time to avoid flicker at very
  high resolutions, and recombines everything at an exact target resolution —
  useful for pushing a conversion up to 4K or 8K.

---

## GUI Improvements

- The app was rebranded **"3DECKER,"** with its own accent color theme, and all
  110+ settings were reorganized from one giant wall of controls into clearly
  labeled tabs (Stereo Generation, Dual-Pass Depth Blend, Video Filter, Video
  Decoding, Video Encoding, Processor, and Standalone Tools).
- A **Layout** option lets you switch between the tabbed view and one long
  scrolling single page, live, without restarting the app.
- A **UI Zoom** control scales the whole app's text and controls larger or
  smaller, independent of your Windows display scaling.
- **Quick Preset** buttons ("Movie," "Action," and "3DECKER Preferred") apply a
  full, tested combination of settings in one click.
- A **Compare Presets** tool renders the same short clip with two or more of
  your saved presets and joins the results back-to-back into one video, so you
  can directly compare the look of different settings.
- The progress bar now shows real step-by-step status (which stage is running,
  elapsed time, speed, and estimated time remaining) instead of one
  undifferentiated bar, and the window can no longer get stuck positioned
  off-screen.
- **The toolbar and the Start/Suspend/Cancel row now wrap onto extra rows
  instead of getting cut off** when the window is narrower than their usual
  width, and the window itself can be dragged much narrower than before —
  down to about 300px wide instead of needing over 640px, so it fits into a
  small corner of your screen without hiding any button.
- Two separate update buttons — **"Check for 3DECKER Updates"** and **"Check for
  Nagadomi Updates"** — let you see what's new from either project without changing
  anything, each with its own **"Install Update Now"** button right on the results
  screen to actually apply it (packages, downloaded models, and source code
  together) — with a confirmation prompt first, and an automatic safety backup of
  any of your own unsaved customizations before it runs. The app also checks
  quietly in the background when you open it and lets you know if a 3DECKER update
  is available, with no popup at all if you're already up to date.
- Every standalone tool's output log now has its own **Clear** button.
- A standalone **Retroactively Tag MKV as 3D** tool writes the correct 3D
  metadata into an already-converted file, so 3D-aware players and TVs (VLC,
  Kodi, many smart TVs) can automatically detect and display it correctly
  instead of you having to select 3D mode by hand.
- **Every setting's title now shows a tooltip on hover, not just its input
  box.** Previously you had to hover exactly over the dropdown, checkbox, or
  slider itself to see an explanation; now hovering the label text next to it
  works too. All 87 setting labels in this app now have a tooltip.
- A new **"What's New"** button, next to the update-check buttons, shows this
  changelog right inside the app — read-only, no download or network check
  involved, just a quick way to see what's changed without leaving the app.

---

## Bug Fixes

- **Auto Resume now warns you when it can't find a matching checkpoint**,
  instead of silently starting your conversion over from scratch. If a
  conversion was interrupted and you change a setting (like Depth
  Anti-aliasing or Object Stability) before resuming, the app previously had
  no way to tell you it couldn't find your earlier progress — it just quietly
  began again from the beginning. Now it tells you exactly what it found and
  what changed, so you can put the setting back and actually resume instead
  of losing real, sometimes hours of, work without knowing it.
- **Fixed GPU memory not releasing after a conversion finishes.** Two
  separate memory problems were tracked down and fixed: dedicated GPU memory
  ("VRAM") that stayed pinned at the same usage until you closed the whole
  app, and a much larger, separate pool of "shared" GPU memory that grew
  directly with your Flicker Reduction Buffer setting and never came back
  down — confirmed on a real 30-second clip, this dropped from over 20GB
  stuck-forever down to under 100MB right after the fix. Both now release
  automatically when a job finishes, with nothing you need to turn on.
- **Fixed torch.compile silently failing to check on** when the Device
  dropdown was set to "All CUDA Device" — it now explains why instead of
  just not responding.
- **Fixed a false "Error" report during updates.** "Install Update Now"
  could report failure and stop partway through even when the update had
  actually succeeded, so packages and models would never get refreshed as a
  result. It now correctly recognizes success in that situation.
- **Fixed a confusing "Press any key to continue" message** left over at the
  end of a successful update that did nothing no matter what you pressed —
  the update had already finished; only "Close" ever worked. That message is
  gone now.
- **Fixed the 3 optional Aether inpainting models never appearing** for any
  install that was originally set up before they were added — running an
  update now registers them automatically instead of requiring a fresh
  install.
- **Fixed a crash when using torch.compile** (a performance-optimization
  option) on some systems — it now fails gracefully with a clear on-screen
  explanation and the conversion continues (just without that speed boost)
  instead of crashing the whole app or the whole conversion.
- **Fixed a real freeze/hang bug** where certain combinations of settings
  (a specific rendering method, multiple worker threads, and the automatic
  scene-length smoothing feature together) could cause a conversion to stall
  forever at 0% with no error message.
- **Fixed subtitle text rendering too small** to comfortably read at high
  video resolutions.
- Fixed several scene-detection caching bugs that could cause incorrect
  behavior on a resumed or re-run job.

---

## Ideas We Tried and Didn't Keep

Not everything that got built stayed. In the interest of being transparent:

- A contrast-boosting step (CLAHE) for depth refinement was built and tested,
  then dropped — it measurably increased noise instead of improving detail.
- Reverse-engineering a competing 3D conversion tool's internals was
  considered and explicitly declined; where a similar *idea* existed in a
  competitor, this project built its own independent version rather than
  copying anything.
- A couple of small visual experiments (an animated "still working" progress
  bar, an accent-colored progress fill) were tried and reverted after testing
  showed they didn't render correctly on this app's actual display setup.
- A Single Page layout experiment (columns automatically rearranging as you
  resize the window narrower, instead of staying a fixed 4-column grid) was
  built, tested at several window sizes, and looked correct in that testing —
  but real day-to-day use showed it could end up hiding part of the settings.
  Reverted back to the reliable fixed 4-column layout rather than leave a
  half-fixed version in place.
