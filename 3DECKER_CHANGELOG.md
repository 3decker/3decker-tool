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

## Latest Update — September 20, 2026

**Improved: every step after the conversion now shows real progress in the main bar.** Before,
steps like "Restoring Audio & Subtitles", the Dolby Vision steps and the waifu2x upscale showed only
"running 01:05" with a full bar. Now each one fills the bar and shows the numbers that fit it:
frames done, speed (FPS) and time left where frames exist (RIFE, the upscaler, reading the movie for
Dolby Vision); GB done, MB/s and time left for file work (copying, pulling out the video and Dolby
Vision data, attaching it, packing the file); and percent plus time left for restoring audio and
subtitles. The same live numbers appear in the standalone RIFE tool's Dolby Vision step. Steps that
finish in a second still jump straight to done.

**Improved: the RIFE step of a normal conversion now shows real progress.** During "Interpolate
frames with RIFE after conversion" the bar used to show only how long the step had been running.
It now shows frames done, speed (FPS), elapsed time and the estimated time left, like the other
steps. Pressing Cancel during RIFE now also stops it and removes the unfinished file.

**Fixed: "SBS to 3D Blu-ray MVC" now includes text subtitles.** Most MKV files carry text
subtitles (SRT/ASS), which a Blu-ray can't hold directly, so they were silently left out and the
disc image played with audio but no subtitles. The tool now turns them into real Blu-ray subtitles
(with a suitable font for Chinese, Japanese, Korean, Thai, Hindi and Arabic), up to the disc's limit
of 32 tracks, and the log lists every track it converted or skipped. Subtitles on the disc sit at
screen depth (they don't float in 3D).

**New: make objects come out MORE toward the audience.** The old "Max Pop-Out Limit" could
only reduce how far things come out of the screen. It is now "Pop-Out Limit / Boost" and works in
both directions: below 1.0 it limits pop-out as before, 1.0 is off, and above 1.0 (1.25, 1.5, 2.0 or
anything up to 2.0) it makes things in front of the screen come out further (1.5 = 50% more). Things
behind the screen are not changed. Very strong values are tiring to watch and stretch the picture
edges harder, so try 1.25 first on a short clip. (If Convergence is 0, nothing is behind the screen,
so this then acts like raising 3D Strength.)

**Fixed: RIFE (smoother motion) no longer causes colour banding on HDR/10-bit movies.**
RIFE used to save its result in 8-bit colour even when your movie was 10-bit, which shows up
as visible stripes in skies, dark scenes and fades. It now keeps 10-bit when the movie is 10-bit
and the output is H.265 (the normal choice for HDR). H.264 output stays 8-bit on purpose for
player compatibility. Files smoothed before this update still have the banding and need to be
made again.

**Fixed: "Restore Audio & Subtitles" together with RIFE.** The restored-tracks file used to be built
from the un-smoothed video, so you got one file with smooth motion but no extra tracks and another
with all the tracks but no smoothing. It now adds the tracks to the smoothed file
(`..._rife_alldub.mkv`). The Dual-Pass Depth Blend mode also used to ignore "Restore Audio &
Subtitles" completely; it now honours it too.

## Update — September 19, 2026

**Fixed: RIFE (smoother motion) and Preserve Dolby Vision now work together.** Before,
the two options refused to run in one job because RIFE creates new in-between frames that
have no Dolby Vision data, so a smoothed file always lost its Dolby Vision. Now you can tick
both: the app converts, smooths with RIFE (forced to H.265, which Dolby Vision needs), and then
automatically puts the original Dolby Vision data back, giving each new in-between frame a copy
of its nearest real frame's data. Checked on a real 4K Dolby Vision movie. If putting it back
ever fails, you still get the smoothed video (just without Dolby Vision) and the log says why.
Only Dolby Vision is carried over this way, not HDR10+. This also works when the Dual-Pass
Depth Blend option is on.

**New: Dolby Vision + Cancel in the standalone RIFE tool (Tools tab).** Under "Original DV
Source" you can pick your original Dolby Vision movie; after smoothing, the tool
automatically puts the Dolby Vision back (it switches the output to H.265 for you). Leave it
empty for the old behaviour. If your 3D video already has Dolby Vision (you converted with
"Preserve Dolby Vision"), you don't need the original movie at all: pick the 3D video itself
there, or leave it empty and answer Yes when the tool asks whether to keep it. There is also a new Cancel button that stops the job; if you cancel
during smoothing, the unfinished file is deleted.

**Fixed: Live 3D no longer breaks when you resize the captured window.** Resizing the window
you're streaming (or changing screen resolution) used to crash the stream, cut off the picture,
or stretch it. The stream now keeps its size, fits the new picture inside it with black bars
instead of distorting it, and resets its depth smoothing so the 3D effect doesn't wobble.

**New: 3D Blu-ray Import (Tools tab).** Got a real 3D Blu-ray disc (a disc image
or a ripped disc folder)? Pick it, pick where to save, and 3DECKER turns it into a
normal 3D video file you can play in VLC, MPC-HC or on a 3D TV. It finds the main
movie by itself, reads both eyes' pictures from the disc, and keeps the disc's
audio tracks and subtitles. You choose the 3D arrangement — Full or Half
Side-by-Side, Full or Half Top-Bottom, or Frame Packed (which many TVs detect
automatically) — and the video format (GPU-accelerated H.265 by default, or
CPU H.265 / H.264). A progress bar shows how far along it is and a Cancel button
stops it cleanly. Needs about 25 GB of free space for temporary files while it
works (deleted afterward). It cannot open copy-protected discs; rip those to an
ISO or folder first with a separate tool. New installs and updates fetch the two
small helper programs it needs automatically.

**Also new: a "Lossless 3D Blu-ray ISO" choice in that same tool.** Instead of making
a video file, it copies the disc's own 3D video into a new 3D Blu-ray ISO with no
re-encoding — zero quality loss, all audio and subtitle tracks included. Play it
on a 3D Blu-ray player or in PowerDVD, or keep it as a perfect backup. (VLC and
MPC-HC can't play this kind, so use the other layouts for those.)

**New: "SBS to 3D Blu-ray MVC" (Tools tab).** Turns one of your own 3D videos
(side-by-side or top-bottom, full or half) into a real 3D Blu-ray disc image
you can play in true full-resolution 3D on a 3D Blu-ray player or PowerDVD, or
burn to a BD-R. It splits the two eyes, encodes them as a proper 3D Blu-ray
pair, and keeps your audio and picture subtitles. 3D Blu-ray only allows
1920x1080 at 23.976 or 24 frames per second, so other frame rates are refused
instead of quietly changing your movie's speed. The encoding runs on the CPU
(no GPU option exists for this format on current hardware) at roughly
real-time speed or faster.

**New: "Upscale with waifu2x" (Tools tab).** The upscaler that could only run
automatically after a conversion is now also a tool you can run on any video,
any time. Pick "Whole frame" for a normal enlargement, or "Stereo-aware 4K / 8K"
for side-by-side or top-bottom 3D videos — that one splits the two eyes,
upscales each separately (so the AI never blends the seam between them), smooths
flicker, and puts them back together at exactly 4K or 8K width. Includes a
progress bar and a Cancel button. Upscaling is slow — try a short clip first.

**Fixed: Scene Batch was quietly limiting the GPU video quality to about 18 Mbps.** Scene Batch's three GPU (hevc_nvenc) steps - the prepared master made from your source and the two steps that join the finished scenes - ignored their quality number below about 18, so on 4K movies the master and the joined result were held to roughly 18 Mbps no matter what. They now really use their quality settings (about 67 Mbps for the master and 51 Mbps for the joined video on a 4K test clip), so expect larger temporary files and better quality. The main conversion, RIFE, Sharpen and the waifu2x upscale were checked and were never affected.

**Fixed: in 3D Blu-ray Import, the Quality setting did nothing below 18 when using the GPU (hevc_nvenc) video format.** Quality 8, 14 and 18 all produced the same file, because the GPU encoder quietly limited itself to about 19 Mbps. Now the Quality number really controls the file size and picture quality (for example 14 gives roughly 65% more data than 18 on the test disc). The default of 18 now uses somewhat more data than before on demanding scenes.

**Fixed: on a first launch, the bottom row of the window (progress bar and Start button) could sit below the bottom of the screen.** On a fresh install the window opens a little way down from the top of the screen. If the screen was shorter than the window (a common 1080p monitor, a high Zoom level, or Single Page layout), the app shrank the window but forgot to move it up, so the Start row hung off the bottom edge. The window now moves up so everything stays visible. Existing installs with a saved window position didn't see this.

**Auto-crop for 3D Blu-ray Import and SBS to 3D Blu-ray MVC.** Both tools have a new Auto-crop choice (off by default) that finds black bars around the picture and removes them from both eyes by the same amount, using the same detector as the main conversion. In 3D Blu-ray Import it makes a widescreen film's video contain only the picture (for example Full Side-by-Side becomes 3840x804 instead of 3840x1080), and the 4K layouts keep the picture's true shape. A 3D Blu-ray frame is always 1920x1080, so in SBS to 3D Blu-ray MVC it can't make the frame smaller - it only tidies uneven edges and off-centre pictures. It isn't used for the Lossless 3D Blu-ray ISO, which copies the disc untouched.

**4K layouts in 3D Blu-ray Import and SBS to 3D Blu-ray MVC.** The 3D Layout list in 3D Blu-ray Import now also offers Full Side-by-Side 4K, Half Side-by-Side 4K, Full Top-Bottom 4K and Half Top-Bottom 4K (each eye enlarged to 4K - a plain enlargement, so for real AI upscaling run the result through the Upscale tool afterward). SBS to 3D Blu-ray MVC has the matching 4K choices in its Input Layout list, picked automatically from your video's size; a 3D Blu-ray itself still holds 1080p per eye. Full Side-by-Side 4K is 7680x2160, which many players can't handle.

**Tool titles on the Standalone Tools tab are now all blue.** Some (Add Audio Track, Restore All Audio Tracks, Sharpen, RIFE and the newest tools) were still black, which was hard to read on the dark theme.

**Safety pop-ups on the buttons that are easy to hit by mistake.** Cancel,
Suspend, Quick Preview and 3DECKER Preferred now ask "are you sure?" first (and
Clear All already did). "No" is the pre-selected answer, so an accidental Enter
key does nothing. Resume (un-pausing) doesn't ask.

**Fixed: 3D Blu-ray Import changed the colors slightly.** The video's color
label was being applied in a way that also shifted the colors a little. Now
it only labels them — the picture is a bit-for-bit match with the disc.

---

## Update — September 18, 2026

**Dark mode is now a real, working option — a new "Theme" dropdown next to
Layout in the toolbar (System / Light / Dark).** This app already had the
code to follow Windows' own Light/Dark setting, but a later visual pass
always forced its own light color scheme on top, regardless of what Windows
was set to — so Windows' dark mode setting was silently having zero effect
on this app for a while. Now you can explicitly force Light or Dark
regardless of Windows, or leave it on System to follow Windows like before.
Switches instantly, no restart needed.

**Fixed: switching to a tab with fewer settings (like Video Encoding) left
the window way oversized, wasting a lot of empty space.** Real feedback from
someone trying this app out: "other pages are extremely empty with only 1/3
of screen space being used." The window now resizes itself to fit whichever
tab you're actually looking at, instead of staying sized for the busiest tab
(Stereo Generation) no matter which one is open. Note this means the window
will now visibly grow/shrink as you click between tabs — that's the point,
not a bug, but it is a real behavior change from before. (A first version of
this accidentally stopped you from manually shrinking the window past a
certain point afterward — caught and fixed the same day; you should be able
to drag it as small as before again, regardless of which tab is open.)

**New setting: "Max Pop-Out Limit," a safety cap on how far anything can pop
out toward you.** This is different from Convergence Plane — Convergence
only sets *where* the screen depth sits, it doesn't limit how far a close
object can end up popping out past it. Max Pop-Out Limit is a hard ceiling on
the actual pop-out amount, independent of Convergence and 3D Strength, in the
Stereo Generation tab right below Convergence Smoothing. Defaults to 1.0
(off — nothing changes unless you lower it). If a specific shot's close-up
pop-out feels uncomfortable, try 0.7 or 0.4 before going all the way to 0.0
(which removes all pop-out).

**Fixed: Auto Resume caused audio to drift further out of sync with the video
every time a job was stopped and resumed.** Real user report — noticed the
delay growing worse each time a conversion was paused/resumed. Two separate
bugs were causing this, both now fixed:
1. Progress was tracked using small time measurements that each carried a
   tiny rounding error, which added up across every resume. Now tracked by
   exact frame count instead, so nothing can drift.
2. The step that stitches your resumed segments back into one file was
   silently failing in some cases (specifically, when a job was resumed from
   a segment recovered after a crash or force-stop) and falling back to a
   less reliable method that could drop frames. Fixed directly.
Verified on a real test clip across 3 stop/resume cycles: the final video's
frame count now matches exactly what was recorded during conversion, with
audio and video staying within milliseconds of each other throughout.

**Half SBS and Half TB videos now signal "this is 3D" directly to real 3D
TVs and players**, when using the H.264 video codec option. Real 3D
Blu-rays and TVs look for a specific marker in the file to automatically
switch into 3D display mode — previously this app only added a marker
software players could read (the file still played fine everywhere, it
just meant some TVs needed you to manually select the 3D mode yourself).
Half SBS already had the real marker; Half TB is now fixed to match. Only
works with the H.264 codec option — HEVC and NVENC don't support this
marker at all, so this doesn't apply if you're using those.

## Update — September 17, 2026

**New checkbox: "Restore Audio & Subtitles from Source after conversion,"
right on the main conversion screen.** The main conversion has always kept
only the FIRST audio track from your source (silently dropping other
languages) and every subtitle track (dropped entirely, not just extras). Turn
this checkbox on and it fixes both automatically once your conversion
finishes — no separate tool, no re-picking files, no re-typing your Start/End
Time trim range. It already knows your source and your finished output, since
it's the same job. Saved as a new `_alldub` file next to your converted
video — the original conversion output is never touched, even if this step
runs into trouble. A source with audio but no subtitles (or vice versa) isn't
treated as an error — whatever it has gets restored. The standalone "Restore
All Audio Tracks" tool (Tools tab) is still there too, for audio-only
restoration on a file you've already converted separately.

**Fixed: Restore Audio & Subtitles (and RIFE Frame Interpolation, and the
waifu2x upscale step) could silently do nothing when your Output was set to
a folder instead of an exact filename** — the normal way most people use
this app. All three of those post-conversion steps were looking at the
wrong file path in that case, so they'd quietly fail instead of running.
Fixed for all three. If you turned on "Restore Audio & Subtitles" and didn't
get the extra tracks you expected, this was why — it should now work
correctly.

**Fixed a real depth-mapping bug in "Any_V3_Metric_Large_Native"** (the new
depth model added earlier this update): the sky was being treated as the
*closest* thing in the picture instead of the farthest, which could make
distant background elements pop toward the viewer and overlap strangely with
closer objects — especially noticeable on sky-heavy or space-themed footage.
Confirmed and fixed by directly comparing a real frame's depth map before and
after. `Any_V3_Metric_Large` (the original, non-"Native" option) was never
affected by this.

**New depth model option: "Any_V3_Metric_Large_Native."** A real gap was found in
how the existing `Any_V3_Metric_Large` model works: it's genuinely a "metric"
depth model (predicts real-world distances), but internally it was being
processed the same way as every non-metric model — never specially handling its
absolute-scale output. This new option is a duplicate that fixes that specifically
— same underlying model/download, but with its real distance values handled
correctly instead of run through a transform meant for a different kind of
model. **Your existing `Any_V3_Metric_Large` option is completely untouched** —
same behavior as always, nothing changes for it. This is purely a new, additional
choice in the Depth Model dropdown for anyone who wants to try the
properly-corrected version alongside the original.

**New standalone tool: "Restore All Audio Tracks."** iw3's own conversion has
always kept only the FIRST audio track from your source movie — if your source
has multiple languages, the rest silently didn't make it into the converted file.
That's not something the main conversion changes (it's just not its job), but you
can now fix it afterward: point this new tool at your converted 3D video and your
original source movie, and it replaces the converted file's one audio track with
every track from the source, each keeping its own language and name automatically
— no need to pick languages or re-type track names by hand. If your converted
video is only a short clip of a longer source movie, optional Source Start/End
Time fields trim the source's audio to match, the same way Add Audio Track's own
trimming already works. Find it on the Tools tab, right below Add Audio Track.

**RIFE Frame Interpolation (Standalone Tool) now shows real progress too — frames,
percentage, FPS, elapsed time, and ETA — instead of a frozen "Running..." message
for however long the interpolation takes.** Same fix Sharpen and Retroactive
HDR/DV Reinjection already got, now extended to RIFE. Nothing else about how RIFE
works changed — this is purely visibility into a job that was already running,
now with a live bar and live numbers instead of silence.

## Update — September 16, 2026

**The Retroactive HDR/DV Reinjection tool now handles a real, tricky case: your
Dolby Vision source and the file you actually converted are different releases
of the same movie.** A UHD Dolby Vision disc and, say, a 1080p REMUX of the
same film often don't start (or end) at exactly the same point — different
leader/logo lengths, different end-credit lengths. A new checkbox, "Allow
Converted To Run Longer (different release)," lets the tool accept a converted
file that runs a bit longer than the source once trimmed, instead of refusing
outright. It relies on dovi_tool's own real, tested behavior for the gap: the
last few seconds just repeat the closest real Dolby Vision metadata instead of
being left ungraded. Only allows the converted file to run longer, never the
reverse — if your source has extra content that never made it into the
converted file, that's still treated as a likely wrong Start/End Time and
refused, same as before.

**That same tool's pre-flight check is also a lot faster now.** It compares
your source and converted files frame-by-frame before doing anything, which
means fully decoding both — that step was using a slower method than it
needed to. Switched to a faster one (same exact frame-count check, just
quicker to run) — confirmed roughly 6.5x faster in real testing. GPU
decoding was tested too but deliberately NOT made the default: real timing
showed it wasn't reliably faster, and was actually slower whenever your GPU
was already busy with another job.

**And a real safety fix for that same tool, found the same day: it will now refuse
if your source and converted files don't actually match in dynamic range (one is
HDR, the other isn't).** A real Dolby Vision RPU only makes sense grafted onto the
exact picture it was created for — if your converted file was made from a plain
SDR release while your source is a genuinely HDR-graded disc, injecting the HDR
disc's Dolby Vision metadata onto it doesn't just risk a small quality hit, it
produces a badly overbright, wrong-looking result on any Dolby Vision-aware
screen. The tool now checks this up front (in seconds, no waiting) and refuses
with a clear explanation instead of quietly producing a broken file. If you hit
this, the correct fix is converting the movie 3D directly from the real HDR
source with "Preserve Dolby Vision" turned on, not grafting metadata after the
fact.

**The standalone Sharpen tool now shows real progress — a bar, a frame count, and
a live percentage — instead of just a frozen "Applying Sharpen..." message.**
Previously there was no way to tell how far along a Sharpen job was or how much
longer it would take; now it updates live as the job runs, and settles cleanly
at 100% when it finishes.

**And a real bug found while building that: Sharpen always produced an H.264
video, with no way to change that — which silently broke Dolby Vision/HDR10+
reinjection afterward, no matter what order you ran the two tools in.** A new
"Output Codec" option (same choices as the RIFE Frame Interpolation tool
already offers: H.264 default, or H.265/HEVC via CPU or GPU) lets you pick
HEVC output when you plan to reinject DV/HDR metadata into a sharpened file
afterward. Leave it on the default if you don't need that — nothing changes
for anyone who doesn't touch this new control.

**The Retroactive HDR/DV Reinjection tool now shows real progress too, the same
day it was asked about.** Its pre-flight check decodes your full source and
converted files before doing anything (to verify they actually match) — that
step now shows a live bar, elapsed/total time, and percentage for each file
being checked, instead of a frozen "Running..." message for however many
minutes that takes on a long movie.

**And a new "Frame Count Tolerance" field on that same tool.** Your Start/End
Time can be exactly right and the tool can still see a source/converted frame
count off by a couple of frames — that's just ffmpeg's trim seeking being
timestamp-approximate, not a real problem, but until now it required manually
running a command-line flag to get past it. This new field (defaults to 5)
lets small, routine gaps like that through automatically. It's not a fix for
an actually wrong time range or an incomplete conversion — those still refuse
as before, exactly as they should.

**And the Start/End Time fields on that same tool now accept fractional
seconds (e.g. 00:03:13.73), not just whole seconds.** Real clip durations are
almost never an exact whole number of seconds — previously you had to round,
which could itself cause a small, entirely avoidable frame-count mismatch.
Typing the exact value now avoids that altogether.

## Update — September 15, 2026

**A new depth model family: Metric3D v2.** Five new choices in the Depth Model
dropdown (Metric3D ConvNeXt Tiny/Large and Metric3D ViT Small/Large/Giant2), a
model VisionDepth3D also offers that iw3 didn't have before. Downloads itself
automatically the first time you pick it, same as the other newer models. Its
working resolution is fixed by the model itself, so the Depth Resolution field
greys out when one of these is selected — that's expected, not a bug.

**And an even bigger one: MoGe-3, Microsoft's newest depth model (2 new choices:
MoGe3 ViT-L and ViT-G).** In real side-by-side testing on your own footage, this
one captured noticeably finer detail than anything else in the app — individual
fingers, thin wire/branch structures that other models blur together. It's a bit
heavier (more GPU memory, slightly slower) than most other options, so it's
offered as a choice, not a replacement for the defaults.

Two other candidates (LBM Depth and FlashDepth) were investigated and NOT added:
LBM Depth's output came out flat/low-detail in real testing, and FlashDepth has a
hard technical blocker (it needs an older, incompatible version of a core library
this app can't safely install) — so neither was worth shipping.

**A new "Clear All" button** sits next to Load/Save/Delete in the toolbar. It resets
every setting back to the app's own defaults and empties the Input/Output boxes,
after asking you to confirm — handy before taking a screenshot to share, or just to
start fresh without closing the app.

**A new experimental Method option: `mlbw_l2_cycle`.** Found by auditing the
project's own official model host for anything real and unused — turned out to be
an alternate-trained version of the existing mlbw_l2 warp method that was never
exposed anywhere. Real side-by-side testing on real footage found it genuinely
produces different results, but no consistent "better" either way — so it's offered
as a choice to try for yourself, not a new recommended default. Only works at 3D
Strength 4 or below (it'll tell you clearly if you go higher).

**Two more Depth-Anything-3 sizes: Giant and Nested-Giant-Large** — bigger than
anything else in the Any_V3 family (1.15B/1.40B parameters). These stay hidden in
the dropdown unless you've already downloaded the checkpoint yourself, because
(unlike every other model here) they're licensed for personal/non-commercial use
only, not for a shared distribution. Loading one for the first time needs one
extra manual step (`pip install evo`) — the app will still auto-download the
checkpoint itself once that's in place.

**Fixed a real bug: large washed-out/striped patches when using the MoGe3 depth
models with the `mlbw_l2_inpaint` Method.** Tracked down to a real quirk in the
small helper network that method uses to predict where to fill in gaps — it could
get confused by unusually smooth/flat areas of depth (which MoGe3's cleaner depth
maps produce a lot more of than other models) and paint a wide banded patch
instead of a real gap. Reproduced the bug on a real scene from actual footage,
confirmed the exact mechanism, and shipped a fix that cut the affected area
roughly in half and eliminated the worst, whole-background version of it in
testing. It's a big improvement, not a full guarantee yet — if you still see any
banding with this combination, switching either the depth model or the Method
remains a reliable workaround.

**Fixed a real bug: the Metric3D depth models could go nearly blank on movies with
black letterbox bars.** A single thin strip of pure-black bar was enough to
confuse the model into predicting one wild "impossible distance" value there —
and because the app always stretches its depth display across the true lowest
and highest values in the frame, that one tiny sliver alone could crush 99%+ of
the real depth detail for the whole frame into a washed-out result. Since most
real movies have those bars, this could have been quietly hurting conversions
with any Metric3D model. Found and fixed the same day it was noticed, confirmed
on real footage: the affected frame went from nearly blank back to full, sharp
depth detail, with no change to normal (non-letterboxed) frames.

---

## Update — September 14, 2026

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
