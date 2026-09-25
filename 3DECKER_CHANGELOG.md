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

## Latest Update — September 24, 2026

**Fixed: the progress bar for "MVC .mkv, Auto-crop applied" sat frozen the entire time, showing only "Importing 3D Blu-ray" with no real progress.** Reported directly while actually using the new one-pass feature. Tracked down to a real gap: this step reports its progress by watching the actual encoder's own live frame-by-frame output, but that watching code was accidentally never connected for this specific new feature — so the progress bar and status text had nothing to update from for the entire encode, which is the slowest part of the whole job. It's now correctly wired up the same proven way an older, related tool in this program already does it — a real, moving progress bar with a live frame count and elapsed time, not just a static message.

**New: "MVC Encoding Speed" option for "MVC .mkv, Auto-crop applied," so you can trade some quality for real speed.** Asked directly: "is it using gpu instead cpu... can we make it quicker?" Honest answer: no GPU is used here at all, by design — the only free 3D encoder available has a hardware mode that's broken on current graphics, and it was never able to use an NVIDIA GPU regardless, so this step has always run entirely on your CPU. What genuinely does help: a real quality/speed tradeoff setting that encoder itself offers, which this program simply never exposed before. There's now a "MVC Encoding Speed" dropdown next to Bitrate — Quality (slower, a bit better detail), Balanced (the encoder's own default), Fast, or Fastest (noticeably quicker, softer detail at the same Bitrate).

**Fixed: "Lossless 3D Blu-ray ISO" no longer produces broken audio on a disc with a TrueHD/Atmos soundtrack.** Reported directly, with a real test file: audio would either play back way too fast, sounding like chipmunks, or be completely silent if you switched to the other audio option. Tracked down to the real cause by actually decoding the produced file's audio, not just checking file info: the tool that builds the disc structure has a real, confirmed bug of its own where it corrupts TrueHD/Atmos audio (a common lossless format on 3D Blu-rays) while rebuilding a new disc from scratch — most of the actual sound data was getting silently dropped, which is exactly what produces sped-up, garbled playback. The video was never affected — this was audio-only. The fix uses that disc-building tool's own documented workaround for this exact situation: instead of trying to keep the full TrueHD/Atmos track (which it can't currently build correctly), it now automatically falls back to that track's other layer, a standard 384kbps 5.1 surround track — real, working audio instead of broken "lossless" audio. Worth knowing honestly: on a disc like this, you no longer get a byte-for-byte untouched copy of the TrueHD/Atmos track specifically — everything else about the disc (video, subtitles, and any non-TrueHD audio) stays exactly as before.

**Fixed: "3D Blu-ray Import" now finds your disc even when pointed at the folder a ripping tool asks you to choose, not just the disc folder itself.** Follow-up on an earlier report that "no Blu-ray 3D content found" even though the right folder genuinely existed. Tracked down using a real disc: most ripping tools (like Xreveal) don't put the disc's real files directly in the folder you pick — they create one more folder inside it, named after the movie, and put the real disc structure in there. This program was only looking at the folder you picked, and the folder one level inside or outside it — not one level further in — so it never found discs ripped that common way. It now also checks inside every folder found directly inside wherever you point it, so this works correctly no matter which of these folder levels you choose. Confirmed by reproducing the exact reported error message using a real disc's own real structure, then confirming it's gone after the fix.

**New: "3D Blu-ray Import" can now go straight from a disc to a cropped, real MVC .mkv file in one pass — no separate re-encode needed to remove black bars.** Asked directly: "can we do from iso import directly to mvc mkv so only one encode happens and including the auto-crop bars?" Until now, getting a black-bar-cropped MVC file meant two full passes — import losslessly first, then run the result back through the separate "SBS to 3D Blu-ray MVC" tool just to apply the crop. There's now a single new option, "MVC .mkv, Auto-crop applied (re-encode, one pass)," that does the whole thing in one real encoding pass: demux from the disc, apply the exact same crop detection the standalone tool already uses (same "top and bottom" / "all sides" choices), encode once to real MVC, then restore audio and subtitles — all in one go. A new Bitrate field (2–40, default 20) controls the quality of that one real encode; for the closest thing to lossless here, use the format's own maximum of 40. One honest correction made along the way: a real 3D Blu-ray/MVC frame is locked to a fixed size by spec, so on a genuinely widescreen movie, Auto-crop tightens the black bars but can't always guarantee they're completely gone — a flat (non-MVC) output doesn't have that limit. Like the other MVC features added recently, this hasn't been confirmed yet against a real disc and a real 3D-capable player — the existing two-pass path remains available if you hit any trouble with the new one-pass option.

**Improved: two "3D Blu-ray Import" error/output messages are now much more specific, from a real user's detailed test report.** Two real issues came up testing a genuine 3D Blu-ray disc: "Lossless 3D Blu-ray ISO" produced a file with video but no audio track (even though the disc clearly has one), and importing from a ripped folder (rather than a mounted disc image) refused with "no Blu-ray 3D content found" even though the right folder genuinely existed. Neither is fully solved yet — both need the actual failing disc/folder to pin down for certain, which isn't available to test against directly. What's fixed now: both situations report real, specific detail instead of a dead end — the Lossless ISO tool prints every audio/video/subtitle track it actually detected (so a missing track is now visible immediately instead of a silent surprise), and the "no 3D content found" refusal now lists exactly which folder locations it checked and why each one didn't match, making it much easier to tell whether the wrong folder was picked versus something is genuinely off. Honest note: if you hit either of these, the message you now see is exactly what's needed to get this properly fixed next.

**New: "Convert to 3D Blu-ray MVC after conversion" can now remove black bars, so a 3D player with no crop/zoom of its own still fills the whole screen.** Asked directly, after finding a 3D-capable player (SyLC 3D Player) that has no way to crop out black bars during playback. That has to be fixed when the file is built, not by the player — the standalone "SBS to 3D Blu-ray MVC" tool already had an "Auto-crop" option for exactly this; the newer one-click "Convert to MVC" checkbox never got it. It now has the same "Auto-crop (MVC)" option, right under the HDR/DV checkbox — pick "all sides" or "top and bottom only" and the bars are cut out before the 3D file is built, so any player shows a full picture with nothing left to crop. Off by default, matching the original tool. (Worth knowing: this MVC step was never a byte-for-byte lossless copy to begin with — it's always a real re-encode — so if you want the closest thing to lossless quality here, raise the Bitrate field to its maximum of 40, which is the format's own real ceiling.)

**New: "Write a Log File for This Job" — an optional text log saved right next to your output, so you can always check what happened with a specific conversion later.** Suggested directly: right now, once a job finishes, there's no way to look back and see exactly what happened during that run once the on-screen output has moved on or the program's been closed and reopened. Turn this on (Post-Processing section) and every job saves a plain text file next to its output — same name, with "_log" added — containing every stage, warning, and note from the whole run, including any post-processing steps like Upscale, RIFE, Restore Audio & Subtitles, or Convert to MVC. Off by default, and — since it works the same way every other setting here does — it's automatically remembered in your saved presets, no extra step needed for that part.

**Improved: the "Position subtitles in 3D (dual-eye)" restored subtitle track now has a real, readable name in your player's subtitle menu.** Asked directly: will it have a "3D" name tag on it? It didn't — checking turned up that it would've just shown up as a bare language name ("English"), impossible to tell apart from a regular subtitle track just by looking at the list. It's now named things like "English (3D)" so you can tell at a glance which one is the depth-positioned version.

**New: "Restore Audio & Subtitles from Source" can now bring subtitles back already positioned for real 3D, not just a flat copy.** Asked directly: "is there a way to restore them but converted in 3d already?" A flat subtitle looks fine on a regular screen, but on a real 3D display it lands exactly on the seam between your two eyes, tearing every line of text in half — this is the same problem the "Add Subtitle Track" tool's own "dual-eye" option already solved for a subtitle you add by hand; it just wasn't available for subtitles restored automatically after conversion. There's now a matching checkbox, "Position subtitles in 3D (dual-eye)," right under "Restore Audio & Subtitles from Source after conversion" — turn it on and every text subtitle track (most .srt/.ass sources) comes back split into two properly-positioned copies, one per eye, instead of flat. Off by default, and it only affects split-eye layouts (Half/Full SBS, Half/Full TB, Cross-Eyed) — a picture-based subtitle (PGS, common on some Blu-ray rips) or a non-split output (RGB-D, Anaglyph) is restored exactly as before either way.

**Fixed: a black command-window kept flashing open and closed throughout the "Convert to MVC" step — and the same fix has now been applied to every other Standalone Tool too.** Reported directly: "there was flashing the cmd window back and forth like opening and closing... the work got done, but that was distracting." This step runs many small background tasks in a row (pulling out each audio/subtitle track, preparing the video, the actual 3D encode, then combining everything) — each one was popping open its own brief window because that particular step runs as its own separate background program, which had never been told to stay hidden the way the rest of this program already is. Once fixed there, a check turned up the exact same gap in every other Standalone Tool that runs its own background programs (Retroactive HDR/DV Reinjection, Restore Audio & Subtitles, Restore All Audio Tracks, Sharpen, Add Subtitle Track, Add Audio Track, Upscale with waifu2x) — all fixed the same way now, so none of them should ever flash a window either. (The MVC step still takes a while regardless — that part is a real limit of the free 3D-video encoder involved, not something this fix changes.)

**New: "Convert to 3D Blu-ray MVC after conversion" can now automatically convert HDR/Dolby Vision sources to SDR first, so it no longer refuses your real movies.** A real 3D Blu-ray/MVC file physically can't carry HDR or Dolby Vision at all — the same limitation a real disc has — so running this on an HDR source (like most UHD Blu-ray rips) would correctly refuse with an error rather than produce a broken file. The separate standalone "SBS to 3D Blu-ray MVC" tool already had a "Convert HDR/DV to SDR" option for exactly this; the one-click main-conversion version never did. There's now a matching checkbox right under the MVC settings — turn it on and an HDR source gets automatically tone-mapped to SDR before the MVC step runs, instead of refusing. Off by default, since it only makes sense for an HDR source and the HDR grade is genuinely gone from that MVC file afterward (your regular converted output is never touched either way).

**Fixed: "Convert to MVC" (the .mkv output option) was silently dropping every text subtitle track — confirmed by a real user report with a screenshot.** After the previous fix made the tool's notes actually visible (see below), a user immediately hit real, useful proof of it working: a popup showing "subtitle track skipped: could not extract it" for both of their subtitle tracks. That pointed straight at the real cause — the video-editing tool this program relies on for the .mkv output didn't recognize the small temporary file type used to hold an extracted subtitle (while the equivalent one for audio worked fine, which is exactly why audio came through but subtitles didn't). Fixed by being explicit about the file type instead of letting it guess. Confirmed directly against a real test file before and after the fix.

**Changed: every Standalone Tool (Sharpen, RIFE, SBS to 3D Blu-ray MVC, Restore Audio, and the rest) now shares ONE output box at the bottom of the Standalone Tools tab, instead of each having its own.** Suggested by a user: since you can only run one of these tools at a time anyway, there's no need for a dozen separate output boxes cluttering up the tab. Whichever tool you run, its results now show up in the same place at the bottom — same "Clear" behavior as before, just one button instead of many.

**Fixed: the little output/log box inside each Standalone Tool (Sharpen, RIFE, SBS to 3D Blu-ray MVC, and the rest) could show leftover text from a previous session after restarting the program.** Suggested by a user: starting a new run already clears that box automatically (that part was already working), but reopening the app could still show whatever was left over from before you closed it, since those boxes get remembered the same way every other setting in the app does. There was already a manual "Clear All" button for this, built after an earlier report of a leftover file path showing up unexpectedly — this makes it automatic, so a fresh launch of the program always starts with a clean slate in every tool's output box, no manual clearing needed. Your remembered file paths and other real settings are untouched — only the run-output text itself resets.

**Fixed: "Convert to MVC" could finish with no error shown at all, even when a subtitle track quietly didn't make it into the final file.** Reported by a user: the conversion completed, audio came through fine, subtitles didn't, and there was no way to find out why — "no errors at all... there is no little window with messages in it." The step doing the real MVC work was actually keeping track of exactly what happened to every audio/subtitle track the whole time (converted, included as-is, or skipped and why) — that information just had nowhere to go once the job finished successfully, so it was being thrown away instead of shown. It now pops up in a message box right when the job finishes, listing the real reason for anything worth mentioning about your audio/subtitle tracks — so if a subtitle format genuinely can't be included, you'll see exactly why instead of just noticing it's missing afterward.

## Update — September 23, 2026

**Improved: "Convert to 3D Blu-ray MVC" now works with Half Side-by-Side and both Top-Bottom layouts too, not just Full Side-by-Side.** Follow-up on the option from earlier tonight — it originally always switched your Stereo Format to Full Side-by-Side the moment you turned MVC conversion on, since that gives the best possible quality. Turns out that was more restrictive than it needed to be: Half SBS, Full Top-Bottom, and Half Top-Bottom all genuinely work fine for MVC conversion too — you'll just get a lower-resolution result from the Half options, the same tradeoff as using them anywhere else in this program. It now only steps in and switches your Stereo Format automatically if you have something MVC genuinely can't use at all (like VR90 or Anaglyph) selected — your SBS/Top-Bottom choice is always respected now. (Separately: real 3D Blu-ray discs don't use "Frame Packing" at all — that's a different, simpler 3D delivery method mainly used for things like older 3D game consoles, not how an actual disc stores its video — so there's nothing to add there; it wouldn't apply either way.)

**New: "Convert to 3D Blu-ray MVC after conversion" — the main 2D-to-3D conversion can now finish the whole job on its own, real MVC file included, no separate step afterward.** Following directly from the new MVC .mkv option above, a user asked the obvious next question: can this become part of the regular conversion itself, instead of something you run by hand afterward? It now can — a new checkbox right next to "Restore Audio & Subtitles" does exactly that: once your main conversion (and audio/subtitle restore, if that's on too) finishes, it automatically runs the finished result through the same real MVC encoding, producing its own separate `_MVC.iso` or `_MVC.mkv` file (your choice) right alongside the normal output — nothing about your regular converted file changes. Since real MVC needs a full-resolution 3D frame to work from, checking this box automatically switches Stereo Format to Full Side-by-Side for you if something else was selected, so you can't accidentally end up with a lower-quality MVC file. As with both MVC features from earlier tonight: genuinely new, not yet confirmed end-to-end on a real conversion — the standalone "SBS to 3D Blu-ray MVC" tool remains available separately if you'd rather run this step by hand and check the result first.

**New: "SBS to 3D Blu-ray MVC" can now write a real MVC file directly, no disc image needed — the real "2D movie in, real 3D file out" path.** Requested by a user who put it plainly: the goal is taking an ordinary 2D movie and ending up with one real MKV-MVC file, with nothing else needed in between. Earlier tonight this program already learned to do that starting from an actual 3D Blu-ray disc (the new "Lossless MVC .mkv" option) — this closes the other half: now the tool that turns this program's own AI-converted 3D video into real MVC can also skip the disc-image step entirely and write straight to a plain `.mkv`. Just type an output filename ending in `.mkv` instead of `.iso` and it writes the real 3D video directly — no separate disc-authoring step, and (as a bonus) none of the audio/subtitle format restrictions a real Blu-ray disc has, since a plain file doesn't need them. As with the other new MVC option from earlier tonight: this is brand new and hasn't been confirmed yet in a real MVC-capable player — the existing ISO option remains the proven choice if you run into playback trouble.

**Improved: when "SBS to 3D Blu-ray MVC" fails at the actual 3D-encoding step, the error message now points at the real culprit instead of a confusing side effect.** That step works by piping video from one program (ffmpeg) directly into another (the disc-authoring encoder) — so if the encoder itself crashes, the first program just sees its output suddenly has nowhere to go and reports a generic "broken pipe," which is what used to get shown as the error — accurate, but unhelpful, since it points at the program that was actually working fine right up until the other one died. The message now always reports the encoder's own exit code too, which is a genuine clue toward the real cause even when the encoder itself printed nothing before crashing. This doesn't yet explain *why* one specific user's encoder crashed on their own clip — this is a diagnostics improvement, not a confirmed fix for that particular crash — but it means the next time anyone hits this, there's finally something concrete to go on instead of a dead end.

**Fixed: "SBS to 3D Blu-ray MVC" could fail completely, with no output at all, if the source's audio was in the TrueHD format.** Reported by a user on their own separate machine. The disc-authoring tool this program relies on for the final step turned out to flatly refuse a TrueHD audio track when it was handed on its own, separate from the rest of the file, even though that same tool has no trouble with TrueHD audio still sitting inside a normal video file — which meant the whole conversion failed at the very last step with nothing to show for it. TrueHD audio is now automatically converted to AC-3 (the same reliable format already used for any other audio type this program can't include as-is) instead, so this can no longer cause a hard failure — the tradeoff is that a TrueHD track specifically is no longer copied byte-for-byte untouched, matching how a couple of other rare audio formats already worked.

**Fixed: cancelling (or a crash/restart during) a 3D Blu-ray Import job could make a later retry with the same output name fail with a confusing "tsMuxeR demux failed" error.** When a job is cancelled, force-killed, or interrupted by a crash or restart, the large temporary files it had already extracted from the disc used to get left behind instead of being cleaned up — because the cleanup code that normally deletes them never gets a chance to run when a job is stopped that abruptly. If you then ran the tool again with the same output file name, it would reuse that same leftover, half-finished temp folder, which could make the next attempt run out of disk space or trip over a file that was still briefly locked from the previous run — a failure that had nothing to do with whatever settings you'd changed since, even though it could look that way. It now clears out that folder itself before starting a new attempt, and does a more thorough job cleaning up afterward too, so a cancelled run can never quietly poison a later one.

**New: "Lossless MVC .mkv" option for 3D Blu-ray Import — the real 3D video straight from your disc into a plain .mkv file, with audio, in one step, with zero quality loss.** Requested by a user who already knew exactly what they wanted: they were using this program's existing Lossless 3D Blu-ray ISO option, then separately re-ripping that ISO with another program (MakeMKV or CloneBD) just to get a real "MVC" file for their movie library — and because the ISO was missing its audio track (see the audio fix below), they then had to manually stitch the audio back on as a third step. This new option does the entire job in one go: point it at your disc or ISO, pick "Lossless MVC .mkv," and it writes out a single file with the real 3D video (byte-for-byte from the disc, no re-encoding, no quality lost) plus every audio and subtitle track, ready for a library or player built around real MVC files. One honest note: this is brand new and hasn't been tested yet against a real disc in a real 3D-capable player, since none was available while building it — the Lossless ISO option remains the proven, older choice if you run into any playback trouble with the new one.

**Fixed: expanding a tool's settings on the Standalone Tools tab (or any other collapsible section) could suddenly snap your horizontal scroll position back to the left, on a window too small to show everything at once.** Reported by a user who noticed it specifically on the Standalone Tools tab, which has the most collapsible tool sections of any tab. If your window isn't wide enough to show the full tab at once, you can scroll sideways to reach controls further along — but clicking any section open or closed used to instantly snap that scroll position back to the start, even though the section you just opened had nothing to do with where you'd scrolled to. You'd have to scroll all the way back over to find your place again. It now remembers exactly where you were and puts you right back there after the section opens or closes.

**Fixed: "SBS to 3D Blu-ray MVC" could silently leave a source's audio track out of the finished disc entirely, even though the source genuinely had one.** Reported by a different user. The tool figures out which audio/subtitle tracks a source file has by asking one of the two bundled disc tools (tsMuxeR) to identify them — which works well for its main job (reading an actual Blu-ray disc), but is narrower than it needs to be for an ordinary movie file: some everyday audio formats weren't recognized in that context at all, so the tool saw "no audio track," not "an audio track I don't support yet," and just left it out with no warning. It now asks ffmpeg (the same all-format engine used everywhere else in this program) to find the tracks instead, which reliably recognizes anything ffmpeg itself can play — closing the actual reported gap, and also making the audio/subtitle extraction step itself consistently line up with what was detected, instead of occasionally risking a mismatch between the two. Not yet confirmed against a real disc from the reporting user (no matching source file was available to test with directly) — confirmed instead through the program's own automated test suite; flagging that distinction here in case anyone still sees an audio gap after this update.

**Fixed: automatic RIFE frame smoothing (and the waifu2x upscale step) during a regular conversion could run dramatically slower than the same step run on its own, and use far more graphics card memory than it should.** Found from a real report: a movie's RIFE step was on track to take about 7 hours instead of its usual ~3, and Task Manager showed two separate copies of the program running on the graphics card at once. The cause: when RIFE or the waifu2x upscale step runs automatically as part of a regular conversion, the program was supposed to hand the graphics card fully over to that step once the main 3D conversion finished — but it was still holding onto its own AI models in graphics memory the whole time, so both the finished conversion and the new step were competing for space on the card at once. Once memory runs out, Windows falls back to a much slower method behind the scenes, which is exactly what caused the slowdown. It now properly lets go of that memory right before RIFE or waifu2x starts (and picks it back up afterward, for the next file if you're converting a batch) — nothing to turn on, this happens automatically. Note: this doesn't speed up a conversion that was already running when this fix was installed — it applies starting with your next one.

**Fixed: cancelling RIFE partway through made the "put Dolby Vision back" step silently disappear, with no explanation.** If you cancel RIFE (or it fails), there's no finished RIFE video for the Dolby Vision step to attach its picture data to — so it correctly skips that step and moves on to Restore Audio & Subtitles instead — but it never told you why, so it looked like the step just vanished. It now logs a plain message explaining that Dolby Vision re-attach was skipped because RIFE didn't finish.

**New: Auto 3D Strength's own settings (mode, min/max range, stability, debug overlay) are now saved into the output filename and file metadata, same as every other setting.** Previously only the plain "typical" 3D Strength number got recorded — if Auto 3D Strength was on, the file gave no indication its actual strength was varying scene-by-scene between your chosen min and max, so a renamed file lost that information. Now the filename and the file's embedded settings both show the full picture whenever Auto 3D Strength is on, and show nothing extra when it's off.

**New: Foreground Pop, Midground Pop, and Background Pop can each be collapsed/hidden independently.** These three used to always show all nine of their rows together inside the "Depth Pop" section, whether or not you were actually using all three. Each one is now its own small expandable/collapsible section (same click-to-expand style as Inpainting Settings), so if you only use one or two of them, you can collapse the ones you don't need out of the way instead of scrolling past all nine rows every time.

**New: "Hold Steady Per Scene" option for the automatic Convergence Plane modes (sod_v1 / Face Detect).** These modes already re-pick the screen position at every scene cut, but between cuts they can still visibly creep — someone shifting slightly, or a small camera pan — since they re-evaluate every single frame. Turning on this new checkbox makes it settle on one value shortly after each cut and hold it firmly, the way a real stereographer does, only easing to a new value mid-shot if the framing genuinely, persistently changes (like a push/pull reveal shot) — small wobbles never move it at all once it's on. Off by default, so nobody's existing setup or saved preset changes on its own — "Convergence Smoothing" still controls how quick vs. steady it is either way, just tuning a slightly different thing depending on whether this new checkbox is on.

**New: "Show Convergence on Video (debug)" for the automatic Convergence Plane modes.** Same idea as Auto 3D Strength's own "Show strength on video (debug)" — writes the actual screen-depth number being used into the corner of every frame (the opposite corner from Auto 3D Strength's own, so both can be on together without overlapping), so you can get real, readable numbers to compare instead of guessing from how a shot looks. Handy for confirming Hold Steady Per Scene is actually holding one value per shot, or comparing sod_v1 against face_detect on the same footage. Off by default — it's for testing, not for a real watch.

**New: "Convergence Bias" — Low/Medium/High labeled shortcuts for the Convergence value.** If you'd rather think in words than decimals, this new dropdown next to the Convergence value box fills in a sensible number for you: Low biases toward more pop-out, Medium is balanced, High biases toward more recede/depth. It's a one-time fill, not a live link — picking one just writes the number in, and you can still hand-edit the value box afterward same as always.

**Fixed: a real crash right at the start of conversion — "Error: ArgumentError, Invalid argument returned 22."** Found from a live user report. Video encoders need the picture's width and height to both be even numbers, and there was one specific situation this program didn't account for: if the finished frame's natural size already happened to fit within your "Output Size Limit" setting (so no resizing was needed), an odd width or height (which can come from AutoCrop, the IPD Offset setting, or just an unusual source video resolution) was never rounded to even before being handed to the encoder — which then refused to start recording at all, crashing immediately on the very first frame. It's now always rounded to an even number no matter what, so this can't happen regardless of why the odd number showed up.

**Fixed: the SAME crash above could also happen for a completely different reason — using "12-bit" output color depth together with an NVIDIA graphics card encoder (hevc_nvenc/h264_nvenc) crashed every single time, no exceptions.** Tracked down by actually reproducing the crash directly against a real movie file: the 12-bit picture format this program was sending to the graphics card's encoder isn't one it actually understands — the encoder needs it packaged slightly differently for 12-bit specifically (the same way it already needed 10-bit packaged differently, which already worked correctly). It's now converted to the format the graphics card encoder actually accepts, confirmed by successfully encoding a real frame from a real 4K Dolby Vision movie that was crashing 100% of the time before this fix.

**Fixed: a real bug in "Show Convergence on Video (debug)" (new earlier today) that could show the wrong number on the wrong frame, or worse.** Found live by the user, who isolated it by testing with the checkbox on vs. off. The inpainting-based conversion methods hold onto a few frames internally before handing them back, so the number this debug overlay was writing onto a given frame could belong to a different, earlier frame instead of the one actually on screen. It now tracks each frame's own real number all the way through, so the label always matches the actual frame, the same careful tracking Auto 3D Strength's own debug overlay already uses for this exact situation.

**Changed: "Convergence Bias" Low/High widened to 0.0/1.0 (were 0.25/0.75), for more range.**

**Fixed: the real, final cause of the "Invalid argument returned 22" crash — this program could build itself a file name that's too long for Windows to accept.** This is the same crash as the two "Fixed" entries just above — both of those were real bugs and are still fixed, but neither one turned out to be the actual cause for every case. This program automatically builds your output file's name from your input file name plus a short tag for every setting you have turned on, so the file itself always tells you what made it. With enough settings active at once (Auto 3D Strength, the debug overlays, Depth Pop, EMA smoothing, and others all add their own tag), that name can grow long enough to break a hard limit Windows itself puts on file names — and when that happens, the recording step fails immediately with a confusing, unrelated-looking error message that gave no hint the real problem was just the name being too long. Found by directly watching what Windows itself was doing, frame by frame, during a real crash. The file name now has a safe length limit: if your settings would make it too long, the settings portion is shortened and a short code is added instead, so it can never happen again — your original file name and the 3D format tag at the end (like "_LR") are never touched, only the settings tags in the middle.

## Update — September 22, 2026

**Faster: the main conversion screen's "Convert HDR to SDR" option (Video Filter tab) now uses your graphics card too.** This is the older, original version of the same feature the newer HDR-to-SDR improvements were built for — it had fallen behind: it was still entirely CPU-only for both reading and writing the video, while the newer versions of this same feature already used the graphics card. It now uses the graphics card for both when available, the same as the other two, with the same automatic, safe fallback to the old CPU method if anything about your specific video doesn't cooperate.

**Clarified: "SBS to 3D Blu-ray MVC" now tells you plainly which of its steps use your graphics card and which don't.** The actual 3D disc-encoding step ("Encoding 3D") has always run on the CPU only — the free program this tool relies on for that specific step doesn't have a working graphics-card mode on current hardware, and there's no other free option. That's always been true and isn't something this update changes, but now that the Fix Frame Rate and Convert HDR to SDR steps in the same tool *do* use your graphics card, it's easy to wonder why this one doesn't. The live progress now says "Encoding 3D (CPU/software)" right on screen during that step, and the Run button's tooltip explains why.

**Faster: converting HDR to SDR (in "SBS to 3D Blu-ray MVC" and the standalone "Convert HDR/DV to SDR" tool) now uses your graphics card to read the video too, not just to write it.** Found live while a user watched their own real 4K movie converting: the step already used the GPU to write the finished video, but was still using only the CPU to read the original — visible as unexpectedly high CPU usage during that step. It now uses the GPU for both when available, which should noticeably speed this step up and free your CPU for other things. If anything about your specific video makes that not work, it automatically and silently falls back to the old, always-reliable method — this can only make things faster, never break a conversion that used to work.

**Fixed: a real crash ("AttributeError: 'NoneType' object has no attribute 'name'") partway through conversion, on some video files.** A deep, low-level step that reads a video's technical details assumed one particular piece of information would always be available immediately — for most videos it is, but a third party hit a real file where it wasn't, and conversion crashed outright with a raw Python error, no explanation, while looking for scene cuts. It now recovers that information a different way when the first way isn't available, and refuses cleanly with a plain message in the rare case neither works, instead of crashing.

**New: "SBS to 3D Blu-ray MVC" can now convert HDR/Dolby Vision video to SDR for you, and there's a separate standalone "Convert HDR/DV to SDR" tool too.** 3D Blu-ray genuinely cannot carry HDR or Dolby Vision at all — there's no combination of the classic 3D Blu-ray format with HDR10/HDR10+, and Dolby Vision's own newer 3D-capable format is a completely different, modern one that today only the Apple Vision Pro headset can play — no 3D Blu-ray player, PowerDVD, or TV supports it. A new "Convert HDR to SDR automatically" checkbox (off by default) tone-maps your video down to standard range for you instead of just refusing, using the same technique this project's main pipeline already uses. Since this is a real, one-way change to the picture (you lose the HDR grade), it's off unless you turn it on. There's also a brand new standalone "Convert HDR/DV to SDR" tool (Standalone Tools tab) for converting any HDR video any time, not just for 3D Blu-ray — it offers a 10-bit output choice (keeps more of the original range), since that tool isn't limited by the 3D Blu-ray format the way the automatic option above is.

**Fixed: a real risk of visible color banding when converting HDR video to SDR.** Found and fixed the same day the HDR-to-SDR conversion above was built: the step that reduces the picture's color precision down to a normal range wasn't smoothing that transition, which can show up as visible "steps" instead of a smooth gradient in skies, dark scenes, and subtle lighting. Fixed in both the new HDR-to-SDR conversion and the main conversion pipeline's own existing "Convert HDR to SDR" option, which had the same gap.

**Fixed: "SBS to 3D Blu-ray MVC" could say "see the log box for the exact reason" with nothing actually in the log box.** If one of the programs this tool relies on (ffmpeg, FRIM, or tsMuxeR) crashed badly enough, nothing ever made it into the log, so that message just pointed at an empty box. It now recognizes when that's happened and tells you plainly that one of those programs crashed (naming the specific kind of crash when it's a common one), instead of a dead end.

**Fixed: opening any of the collapsible sections in the app (Standalone Tools especially) could permanently stop you from making the window narrower again.** A real user hit this right after trying the new Fix Frame Rate checkbox: expanding just one section (like SBS to 3D Blu-ray MVC's own settings) was enough to lock the window's smallest allowed size at whatever that section's full height needed, even after collapsing it again. This is the same class of bug already fixed once for switching between tabs — that fix just never got applied to expanding a section within a tab. Now fixed everywhere it could happen (Stereo Generation, Dual-Pass Depth Blend, Video Filter, and every Standalone Tools section).

**Improved: "Fix frame rate automatically" (SBS to 3D Blu-ray MVC) now shows real progress and runs faster.** The same user reported it sitting on one unchanging status for 20+ minutes with no indication of whether it was still working. It now shows a real percentage, elapsed time, processing speed, and estimated time left while it re-times your movie, the same as every other step in this tool. It also now uses your graphics card to do this step when available (instead of only the CPU), which should noticeably cut down that wait on most systems.

**New: the "SBS to 3D Blu-ray MVC" tool can now fix an incompatible frame rate for you.** 3D Blu-ray only allows 23.976 or 24 frames per second, so a video at any other rate (25fps and 30fps are common) used to just get refused. There's now a "Fix frame rate automatically" checkbox (off by default) that genuinely re-times the whole movie — picture, sound (pitch kept correct, not sped-up-chipmunk voices) and subtitles together — to whichever of 23.976/24 is closer, the same trick used for classic PAL/NTSC conversions. It's off by default and stays that way unless you turn it on, since it is a real, if usually small, change to your movie's speed and length (about 4% for a 25fps source) — this tool still never changes your movie's speed without you asking first.

**Fixed: updating could break 3DECKER on older graphics cards (GTX 900/700-series, GTX 10-series) with "CUDA error: no kernel image is available for execution on the device."** A recent update added support for the newest RTX 50-series cards, but the update process itself didn't know to keep using the older, compatible AI engine build for everyone else — so a working install on an older card could break the next time it updated. Updating now automatically detects your specific graphics card and picks the right build for it, the same way a brand-new install already did. If this already happened to you, just update again — it'll fix itself.

**Fixed: a real "OSError [Errno 129]" crash at AutoCrop Analysis, Quick Preview, AutoCrop Test, or Compare Presets — reliable, not occasional.** A background setup step was running in the wrong order relative to another one, which could leave the graphics card connection unable to do hardware-accelerated video decoding right after clicking Start (or those other three buttons) — every single time, on some setups. Unlike an earlier, related fix, this one had nothing to do with Software Fallback or the Compile setting — both were ruled out directly before the real cause was found. Confirmed fixed against a real conversion that had been crashing 100% of the time.

**New: "Pop Feather %" softens Foreground/Midground/Background Pop's edge.** These three settings (in the Depth Pop section) each push a chosen slice of the picture -- nearest bit, farthest bit, or everything in between -- and leave the rest completely alone. Right at the edge of that slice, a real user found a visible line: the push reaches its full amount, and the very next bit of picture just outside gets none at all. Pop Feather % (0 = off, the old behavior, unchanged) softens that edge into a gradual blend instead of a hard cutoff, the same idea already used to soften Protect Faces' own detection boxes. One setting covers all three Pop tools at once. Try 10 if you spot a line at one of their edges.

**New: a "Show Advanced Settings" checkbox on Stereo Generation, off by default.** That tab used to show 14+ settings at once with no grouping, mixing everyday controls (Depth Model, 3D Strength, Convergence Plane, Method) in with fine-tuning most conversions never touch (Auto 3D Strength, Resolution Preset, Convergence Smoothing, Pop-Out Boost's Max Negative Parallax, Protect Faces, IPD Offset, Synthetic View, Splat Blend Temperature). Now, by default, you see just the everyday controls; tick "Show Advanced Settings" at the top to bring back everything else. Nothing is lost when a setting is hidden — any value you already set for it keeps working exactly the same when you click Start, and ticking the box back on shows it again unchanged.

**Improved: Depth Model and Depth Resolution moved to the top of Stereo Generation, always visible.** They used to live inside the collapsed "Inpainting & Depth Source" section, even though Depth Model -- which AI reads your picture's depth -- matters for every conversion, not just the inpainting methods that section's other settings are actually about. They're now the first thing you see, and that section is renamed "Inpainting Settings" now that it only holds what's genuinely inpainting-specific (plus Stereo Processing Width, used by a couple of other methods). Pop-Out Limit / Boost's tooltip also now says plainly how it differs from the Foreground/Background/Midground Pop sliders further down, since they can sound like the same thing.

**New: Protect Faces.** A new "Protect Faces" setting, right under Pop-Out Limit / Boost, reduces the facial warping (the nose looking pulled toward the audience, the eyes distorted) that a strong 3D Strength or Pop-Out Boost can cause on a close-up face. A face's own real depth relief -- nose tip to eye socket -- is only a few centimetres, but once the face fills the frame that tiny relief gets stretched by whatever the rest of the picture is stretched by, and reads as exaggerated the same way everything else does. This detects faces (the same detector the Convergence Plane "face_detect" mode already uses) and gently flattens each detected face's own depth toward the middle of its own range before Pop-Out Boost sees it. Off by default (0.0); try 0.3-0.5 first. Only the detected face is touched -- everything else in the frame is untouched -- and a face the detector misses (in profile, partly hidden, poorly lit) is simply left as it was.

**New: Auto 3D Strength.** A new control right under 3D Strength picks the 3D Strength per scene from how close the shot looks, instead of one fixed value for the whole movie -- wide shots and landscapes get less, close-ups get more, the way a real stereo camera behaves. It uses an AI model (CLIP) to judge the shot, since iw3's own depth can't reliably tell a landscape from a face filling the frame. Pick a Mode (hybrid, cuts or smooth), an Auto Range (min/max), and how settled it should be (Auto Stability); 3D Strength becomes what an ordinary shot gets. Off by default. A debug option burns the strength used into the corner of the picture for testing. Built by Rowan (iw3-auto3d, MIT licence); its own automatic window-injection didn't fit 3DECKER's layout, so the controls here are 3DECKER's own, built to match.

**New: upscaling a Dolby Vision or HDR video now keeps it Dolby Vision / HDR.** Before, the waifu2x upscale wrote an ordinary 8-bit picture, which threw away the Dolby Vision data and turned HDR colours washed-out. Now an HDR or Dolby Vision video is upscaled as 10-bit HEVC with the right HDR colour tags, and the original Dolby Vision data is put back afterwards (for both the standalone Upscale tool and the after-conversion option, in every mode). A Dolby Vision file must be saved as .mkv.

**New: Full SBS 4K and Full Top-Bottom 4K for the stereo-aware upscale.** Pick "Stereo-aware Full SBS 4K" to make every eye a real 3840-wide picture (3840x2160 for a normal movie) packed side by side (7680x2160), or "Full Top-Bottom 4K" for 3840x4320. It works from any Half or Full SBS / TB file, so a 1080p Half SBS movie becomes a true Full 4K one. (The old "4K" choice only makes the whole frame 3840 wide, so a file that is already 3840 wide does not get any bigger.) The same two choices are in the after-conversion "Target" box.

**Fixed: the stereo-aware upscale gave the wrong playback speed for any movie that is not exactly 24 fps.** The smoothing step wrote its result without a frame rate, so everything came out at 24 fps: a 47.95 fps (RIFE) movie played at half speed and a 23.976 fps one slightly fast. It now keeps the movie's own frame rate.

**Fixed: Dolby Vision could not be put back on an upscaled file whose name contains "_rife".** A safety check meant for RIFE output refused it, although an upscale keeps the same frames. That name check is now skipped for the upscale (the exact frame-count check still runs).

**Fixed: in the Upscale tool, the Flicker Smoothing box (and 3D Layout) could stay greyed out after choosing a stereo-aware mode.** The boxes were only refreshed while the Mode list was still changing, which on Windows can still see the old choice, so they kept the previous mode's state. They now refresh again once the list closes. (The job itself always used the value shown in the box.)

**Much faster: the flicker-smoothing step of the stereo-aware upscale.** After each eye is enlarged, a smoothing pass calms flicker between frames. It analysed motion on the full enlarged frame on one CPU core, about 2 seconds per frame, so on a movie it took hours per eye while the graphics card sat idle. It now shrinks the frame to its final size first, uses lighter motion settings, analyses the motion at half size, and works on both eyes at the same time: about 6 to 7 times faster in tests (0.5 to about 3 frames per second per eye, both eyes together). There is a new "Flicker Smoothing" setting in the Upscale tool (Fast / Accurate / Off): Off skips the smoothing completely and makes that step about 70 times faster (minutes instead of hours). The enlargement itself (the AI model) is unchanged and is still the slowest part.

**New: the after-conversion steps now chain into ONE file.** Before, ticking Upscale with waifu2x, RIFE and Restore Audio & Subtitles gave separate files, because each step started again from the plain converted movie. Now they run in order on each other's result: upscale first (it is the slowest step, and RIFE doubles the frame count), then RIFE, then Dolby Vision is put back, then all audio and subtitles are restored. The names stack, for example "..._w2x_rife_alldub.mkv", and that last file is the one to keep. A step that is off or fails is skipped and the chain carries on. This works in the normal conversion and in Dual-Pass Depth Blend. I also corrected the RIFE tooltip, which still said RIFE could not be used with Preserve Dolby Vision.

**Faster: the waifu2x upscale now uses the graphics card's video encoder instead of the CPU.** While waifu2x enlarged each frame, the CPU (about half of all cores) was busy compressing the result with software H.264 and the GPU had to wait for it. Every pass (split, each eye, smoothing, join, and the whole-frame mode) now encodes with the GPU's HEVC encoder when there is one, so the CPU stays free and the GPU keeps working. Files are now 8-bit HEVC instead of H.264 (H.264 on the GPU stops at 4096 pixels, and an upscaled frame is wider than that). If your computer has no NVIDIA video encoder it works as before.

**Fixed: the standalone Upscale tool now shows real progress in every mode.** The stereo-aware modes run four steps (split, upscale each eye, smooth, join) and the biggest one showed nothing. Every step now shows its frames done, speed (fps), time elapsed and time left, and says which step and which eye it is on. The after-conversion stereo upscale shows progress in the main bar too.

**Improved: the Pop-Out Boost now goes up to 3.0 (it stopped at 2.0).** Useful when you use a higher Convergence (which leaves less room in front of the screen) and want the same pop-out: for example Convergence 0.6 needs a boost of 2.25 to match Convergence 0.4 with 1.5. The dropdown now also offers 2.25, 2.5 and 3.0, and the slider covers the whole range. Very large boosts stretch the picture edges harder, so keep Edge Fix at 3 and 2 or higher and check for halos on close objects.

**Fixed: the 3D tag is now kept on the final file when you use RIFE or Restore Audio & Subtitles.** The file name said `_smtag` (3D tag on), but the tag was only set on the first file. RIFE, the Dolby Vision re-attach and the audio/subtitle restore each make a new file, so the final one was not marked as 3D and VLC or a TV could not switch to 3D by itself. It is now set again on every file those steps produce.

**Improved: your file name now always says which inpainting model was used.** With any inpainting method, the name now includes `_im` plus the model (for example `_imlight_inpaint_v1` or `_imRowan_High-Depth_Inpaint_e594-Medium`). Before, the default model was left out. The model is also stored inside the file's information (the `iw3_inpaint_model` note). The Pop-Out Boost / Limit is now named too: `_pob125` means a boost of 1.25, `_pol70` means a limit of 0.7 (nothing is added when it is off), and it is stored inside the file as `iw3_max_negative_parallax`. Names for methods that don't use inpainting are unchanged. Note: because the default now appears in the name, the resume feature will not recognise a half-finished file made under the old naming.

**Fixed: an update no longer fails because of one dropped internet connection.** Updates and first-time setup download a few helper packages from GitHub. If the connection was reset for a moment, the whole update used to stop with a long error. It now waits 10 seconds and tries again, up to 3 times, and says so on screen, before giving up.

**New: Rowan's High-Depth inpainting model, bundled.** A community-made inpainting model (by Rowan) built for
strong 3D. It fills the gaps that open up behind popped-out objects more cleanly, and it comes with three
helpers: better detection of the smeared band at depth edges, a mode that runs the model small but keeps your
full-resolution picture, and a screen-edge fix so "Preserve Screen Border" works with forward_inpaint. Pick
"Rowan_High-Depth_Inpaint_e594-Medium" in "Inpainting Model" (Method: forward_inpaint, Inpaint Max Width 1280 or
1920). Best at divergence 8 and up. It downloads by itself the first time you use it (about 83 MB). On a first short
test it was about as fast as the default and used a little less video memory; the picture differs mainly along
depth edges. Works from the command line too.

## Update — September 20, 2026

**Fixed: a confusing "HTTP Error 404" when an optional inpainting model can't be downloaded.**
The three optional "Aether" inpainting models are downloaded from the project's GitHub page the
first time you use one. If a model can't be downloaded, you now get a clear message that names the
model, says what went wrong and tells you what to do (pick "light_inpaint_v1", the recommended one,
or check your internet connection) instead of a raw error code.

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
