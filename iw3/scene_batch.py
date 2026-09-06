"""
Automated whole-movie scene-batch pipeline.

Runs the same process that was validated manually:
  1. Pull the Dolby Vision RPU from the pristine source ONCE, before anything
     else happens, and hold it aside. It is never touched per-clip.
  2. Detect scene cuts ONCE, directly on the untouched source movie (shot
     detection works on a heavily downscaled internal copy either way, so
     it doesn't need cropping done first -- this lets step 3 below do the
     crop and the keyframe-forcing in one pass instead of two).
  3. In a SINGLE re-encode pass: remove letterbox bars (if any) and force
     real keyframes into the movie at every exact cut point found in step
     2, so the split in step 4 is frame-accurate (no keyframe-snapping
     drift). This used to be two separate full-movie re-encodes; doing
     both at once roughly halves the time this stage takes.
  4. Split into individual per-scene clips (lossless stream copy).
  5. Run the normal iw3 conversion on each scene clip independently -- this
     gives every scene a true "fresh start" for temporal/EMA state, matching
     what was found to look correct when scenes are processed as separate
     files. Optional per-scene setting overrides can be supplied. Each
     clip's own DV handling is force-disabled since it's handled once, here.
  6. Join the converted clips back into one seamless video.
  7. Pull the matching audio slice from the pristine original source and
     mux it onto the joined video.
  8. Reinject the one RPU from step 1 onto the finished, audio-complete
     video -- the only other place DV is touched.

Intermediate files are kept in <output>.scene_batch_work/ next to the
final output, so a run can be inspected or resumed manually if needed.

Every step -- including the plain ffmpeg/dovi_tool/mkvmerge passes this
module drives directly, not just the depth/stereo conversion -- reports
itself through the same tqdm_fn mechanism the rest of iw3 already uses.
That means each step shows up identically whether this is run from the
CLI (a live progress line) or the GUI (the status bar at the bottom of
the window), so nothing looks "stuck" during a long step; it also still
prints to stderr/the log file for anyone watching a console.
"""
import os
import sys
import json
import csv
import re
import subprocess
import contextlib
from os import path

_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")


def _parse_ffmpeg_time(line):
    """Extract the 'time=HH:MM:SS.ss' progress ffmpeg prints per line, in seconds."""
    m = _FFMPEG_TIME_RE.search(line)
    if not m:
        return None
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / (10 ** len(frac))

from nunif.utils.ui import make_parent_dir
import nunif.utils.shot_boundary_detection as SBD
from nunif.utils.autocrop import AutoCrop


class _Tee:
    """Mirrors every write to a real stream (console) and a log file at once, so the
    log file ends up with a full transcript of everything printed during the run --
    including iw3's own per-scene conversion progress, not just this module's own
    ffmpeg/dovi_tool/mkvmerge steps."""
    def __init__(self, real_stream, log_fp):
        self._real = real_stream
        self._log_fp = log_fp

    def write(self, data):
        self._real.write(data)
        try:
            self._log_fp.write(data)
        except Exception:
            pass

    def flush(self):
        self._real.flush()
        try:
            self._log_fp.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def _step(args, desc, total=1):
    """total=1 gives a plain started/done indicator. Pass a real duration (seconds)
    for a step that will call _run(..., pbar=pbar, seek_total=True) so it fills in
    live as a genuine percentage -- which also gives a real ETA, since that's just
    how tqdm-style progress bars compute it from the fill rate."""
    print(f"[scene-batch] {desc}", file=sys.stderr)
    # wx.Gauge (the GUI's progress bar) requires whole numbers -- a float total/delta
    # throws inside its event handler and silently kills the status bar update for the
    # rest of the run. Round to whole seconds; real tqdm (CLI) doesn't care either way.
    total = max(1, int(round(total)))
    pbar = None
    tqdm_fn = None
    state = getattr(args, "state", None)
    if state:
        tqdm_fn = state.get("tqdm_fn")
    if tqdm_fn is not None:
        try:
            pbar = tqdm_fn(total=total, desc=desc)
        except Exception:
            pbar = None
    handle = _ProgressHandle(pbar, total)
    try:
        yield handle
    finally:
        handle.finish()


class _ProgressHandle:
    """Tracks cumulative progress against a known total independent of whatever the
    underlying pbar object actually is (real tqdm vs. the GUI's TQDMGUI), and tops
    it off to 100% on finish() so a step always ends looking complete even if the
    last ffmpeg time= line landed a hair short of the predicted total. Only ever
    emits whole-number updates to the pbar (see the note in _step() about wx.Gauge)."""
    def __init__(self, pbar, total):
        self.pbar = pbar
        self.total = total
        self.current = 0
        self._frac = 0.0

    def update(self, delta):
        if self.pbar is None or delta <= 0:
            return
        self._frac += delta
        whole = int(self._frac)
        if whole <= 0:
            return
        self._frac -= whole
        self.current = min(self.total, self.current + whole)
        try:
            self.pbar.update(whole)
        except Exception:
            pass

    def finish(self):
        if self.pbar is None:
            return
        remaining = self.total - self.current
        if remaining > 0:
            try:
                self.pbar.update(remaining)
            except Exception:
                pass
        try:
            self.pbar.close()
        except Exception:
            pass


def _bin_dir():
    return path.dirname(path.dirname(path.dirname(path.abspath(__file__))))


def _find_bin(*names):
    import shutil
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    here = _bin_dir()
    for name in names:
        candidate = path.join(here, name)
        if path.exists(candidate):
            return candidate
    return names[0]


def _ffmpeg_bin():
    return _find_bin("ffmpeg.exe", "ffmpeg")


def _ffprobe_bin():
    return _find_bin("ffprobe.exe", "ffprobe")


def _dovi_bin():
    found = _find_bin("dovi_tool.exe", "dovi_tool")
    return found if path.exists(found) or found in ("dovi_tool.exe", "dovi_tool") else None


def _mkvmerge_bin():
    candidates = ["mkvmerge.exe", "mkvmerge"]
    here = _bin_dir()
    for sub in (path.join(here, "mkvtoolnix", "mkvmerge.exe"),):
        if path.exists(sub):
            return sub
    return _find_bin(*candidates)


def _run(cmd, log_fp=None, progress=None):
    """Run a command, streaming its output live to stderr (so long ffmpeg/dovi_tool
    passes show their own native progress as they happen) while also mirroring
    everything into the log file. If `progress` (a _ProgressHandle) is given, ffmpeg's
    own 'time=HH:MM:SS.ss' lines are parsed and fed into it as real percentage/ETA
    progress instead of a flat started/done indicator."""
    if log_fp is not None:
        log_fp.write(" ".join(str(c) for c in cmd) + "\n")
        log_fp.flush()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines = []
    last_time = 0.0
    for line in proc.stdout:
        lines.append(line)
        sys.stderr.write(line)
        sys.stderr.flush()
        if log_fp is not None:
            log_fp.write(line)
        if progress is not None:
            t = _parse_ffmpeg_time(line)
            if t is not None and t > last_time:
                progress.update(t - last_time)
                last_time = t
    proc.wait()
    if log_fp is not None:
        log_fp.write("\n")
        log_fp.flush()

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}\n"
                           f"{''.join(lines[-40:])}")

    class _Result:
        pass
    result = _Result()
    result.returncode = proc.returncode
    result.stdout = "".join(lines)
    result.stderr = ""
    return result


def _ffprobe_json(input_path):
    proc = subprocess.run(
        [_ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
         str(input_path)],
        capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def _detect_hdr(input_path):
    info = _ffprobe_json(input_path)
    dv = False
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        for sd in stream.get("side_data_list", []):
            sdt = sd.get("side_data_type", "")
            if "DOVI" in sdt or "Dolby" in sdt:
                dv = True
    return dv


def _video_stream_info(input_path):
    info = _ffprobe_json(input_path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = v["r_frame_rate"].split("/")
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "r_frame_rate": v["r_frame_rate"],
        "fps": float(num) / float(den),
        "pix_fmt": v.get("pix_fmt", "yuv420p10le"),
        "color_primaries": v.get("color_primaries", "bt2020"),
        "color_transfer": v.get("color_transfer", "smpte2084"),
        "color_space": v.get("color_space", "bt2020nc"),
        "color_range": v.get("color_range", "tv"),
        "duration": float(info["format"]["duration"]),
    }


def _detect_crop(input_path, explicit_crop, vinfo, work_dir, log_fp):
    """Returns (w, h, x, y) or None (no crop needed)."""
    if explicit_crop:
        parts = explicit_crop.split(":")
        w, h = (int(v) for v in parts[0].lower().split("x"))
        if len(parts) == 3:
            x, y = int(parts[1]), int(parts[2])
        else:
            x = (vinfo["width"] - w) // 2
            y = (vinfo["height"] - h) // 2
        return (w, h, x, y)

    crop = AutoCrop.from_video_file(input_path, mode="black").get_crop()
    if crop is None:
        return None
    x, y, w, h = crop
    if w == vinfo["width"] and h == vinfo["height"]:
        return None
    return (w, h, x, y)


def _extract_rpu_once(input_path, work_dir, args, log_fp):
    """Pull the Dolby Vision RPU from the pristine source ONE time, before anything
    else happens. Held aside and reinjected once, at the very end, onto the finished
    joined+audio 3D output -- never touched per-clip."""
    if not getattr(args, "preserve_dowi", False):
        return None
    if not _detect_hdr(input_path):
        return None
    dovi = _dovi_bin()
    if not dovi or not path.exists(dovi):
        print("[scene-batch] Dolby Vision detected but dovi_tool not found, skipping DV preservation.",
              file=sys.stderr)
        return None

    raw_hevc = path.join(work_dir, "src_raw.hevc")
    rpu_path = path.join(work_dir, "src_rpu.bin")
    if path.exists(rpu_path) and path.getsize(rpu_path) > 0:
        print("[scene-batch] Dolby Vision RPU already extracted, reusing it.", file=sys.stderr)
        return rpu_path

    duration = _video_stream_info(input_path)["duration"]
    with _step(args, "extracting Dolby Vision RPU (once, from the source)...", total=duration) as progress:
        # The stream-copy pull genuinely reports real time= progress (a whole movie's
        # video track takes real time to copy even without re-encoding); dovi_tool's
        # own extract-rpu pass afterward doesn't print a parseable percentage, so the
        # bar just gets topped off to 100% by _step()'s finish() once that part is done.
        _run([_ffmpeg_bin(), "-y", "-i", input_path, "-c:v", "copy", "-an", "-f", "hevc", raw_hevc],
             log_fp, progress=progress)
        _run([dovi, "extract-rpu", "-i", raw_hevc, "-o", rpu_path], log_fp)
    try:
        os.remove(raw_hevc)
    except OSError:
        pass
    return rpu_path


def _detect_scenes(input_path, args, log_fp):
    from nunif.utils.video import pyav_init_cuda_primary_context
    from .utils import try_load_scene_cache, save_scene_cache

    # Reuse iw3's own scene-detection cache (the same one --scene-detect uses) so a
    # cancelled/restarted run doesn't repeat this multi-minute AI pass from scratch.
    if not getattr(args, "disable_scene_cache", False):
        segment_pts = try_load_scene_cache(input_path, args)
        if segment_pts is not None:
            times = sorted(p / 1000.0 for p in segment_pts)
            print(f"[scene-batch] found {len(times)} scene cuts (loaded from cache)", file=sys.stderr)
            return times

    print("[scene-batch] detecting scene cuts...", file=sys.stderr)
    pyav_init_cuda_primary_context()
    device = args.state["device"]
    segment_pts = SBD.detect_boundary(
        input_path, device=device, hwaccel=args.hwaccel,
        tqdm_fn=args.state["tqdm_fn"], tqdm_title="Scene Boundary Detection",
        stop_event=args.state["stop_event"], suspend_event=args.state["suspend_event"],
    )
    if not getattr(args, "disable_scene_cache", False):
        save_scene_cache(input_path, segment_pts, args)
    times = sorted(p / 1000.0 for p in segment_pts)
    print(f"[scene-batch] found {len(times)} scene cuts", file=sys.stderr)
    return times


def _prepare_and_key(input_path, times, work_dir, args, log_fp):
    """Crop (if needed) and force exact keyframes at every detected cut point, in a
    SINGLE re-encode pass -- not the two separate full-movie passes this used to be.
    No DV handling here -- that's done once at the very end instead, and no audio/
    subtitle remux either, since the join step pulls fresh audio straight from the
    original source later anyway. Returns the keyframed, video-only file ready for
    splitting, plus its (post-crop) stream info."""
    vinfo = _video_stream_info(input_path)
    keyed_timed = path.join(work_dir, "prepared_keyed_timed.mkv")
    if path.exists(keyed_timed) and path.getsize(keyed_timed) > 0:
        print("[scene-batch] cropped/keyframed master already prepared, reusing it.", file=sys.stderr)
        out_vinfo = _video_stream_info(keyed_timed)
        return keyed_timed, out_vinfo

    crop = _detect_crop(input_path, getattr(args, "scene_batch_crop", None), vinfo, work_dir, log_fp)

    raw_hevc = path.join(work_dir, "src_raw_nodv.hevc")
    with _step(args, "extracting source video stream...", total=vinfo["duration"]) as progress:
        _run([_ffmpeg_bin(), "-y", "-i", input_path, "-c:v", "copy", "-an", "-f", "hevc", raw_hevc],
             log_fp, progress=progress)

    keyed = path.join(work_dir, "prepared_keyed.hevc")
    seg_times = ",".join(f"{t:.3f}" for t in times) if times else None

    if crop and seg_times:
        desc = f"cropping + forcing exact keyframes at every cut point (crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]})..."
    elif crop:
        desc = f"cropping (once) + encoding master (crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]})..."
    else:
        desc = "forcing exact keyframes at every cut point..."

    ff_args = [_ffmpeg_bin(), "-y", "-i", raw_hevc]
    if crop:
        ff_args += ["-vf", f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}"]
    ff_args += ["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "14", "-b:v", "0"]
    if seg_times:
        ff_args += ["-forced-idr", "1", "-force_key_frames", seg_times]
    ff_args += [
        "-pix_fmt", vinfo["pix_fmt"],
        "-color_primaries", vinfo["color_primaries"], "-color_trc", vinfo["color_transfer"],
        "-colorspace", vinfo["color_space"], "-color_range", vinfo["color_range"],
        "-an", "-sn", "-f", "hevc", keyed,
    ]
    with _step(args, desc, total=vinfo["duration"]) as progress:
        _run(ff_args, log_fp, progress=progress)

    # A bare .hevc elementary stream carries no real timestamps of its own -- ffmpeg's
    # segment muxer needs those to do a lossless -c copy split. mkvmerge gives it proper
    # ones (using the exact fps fraction, not a rounded decimal, to avoid long-file drift).
    keyed_timed = path.join(work_dir, "prepared_keyed_timed.mkv")
    fps_num, fps_den = vinfo["r_frame_rate"].split("/")
    with _step(args, "fixing up timing before the split..."):
        _run([_mkvmerge_bin(), "-o", keyed_timed, "--default-duration", f"0:{fps_num}/{fps_den}fps",
              keyed], log_fp)

    out_vinfo = dict(vinfo)
    if crop:
        out_vinfo["width"], out_vinfo["height"] = crop[0], crop[1]
    return keyed_timed, out_vinfo


def _split_scenes(keyed_timed_path, times, vinfo, work_dir, args, log_fp):
    cut_dir = path.join(work_dir, "scene_cuts")
    os.makedirs(cut_dir, exist_ok=True)

    if times is None:
        # Scene detection was skipped entirely (the whole foundation was already built
        # -- see run_scene_batch) so there's no detection result to work from. Recover
        # each already-split clip's start time directly from the clips themselves
        # (cumulative duration) instead -- accurate, and needs no AI scan at all.
        existing = sorted(f for f in os.listdir(cut_dir) if f.endswith(".mkv"))
        if not existing:
            raise RuntimeError(f"No split scene clips found in {cut_dir}, but scene detection "
                               f"was skipped because the foundation looked ready -- this shouldn't happen.")
        print(f"[scene-batch] all {len(existing)} scene clips already split, reusing them.", file=sys.stderr)
        scene_starts = []
        t = 0.0
        for f in existing:
            scene_starts.append(t)
            t += _video_stream_info(path.join(cut_dir, f))["duration"]
        return cut_dir, scene_starts

    scene_starts = [0.0] + times

    expected_count = len(scene_starts)
    existing = sorted(f for f in os.listdir(cut_dir) if f.endswith(".mkv"))
    if len(existing) == expected_count:
        print(f"[scene-batch] all {expected_count} scene clips already split, reusing them.", file=sys.stderr)
        return cut_dir, scene_starts

    if not times:
        # single scene, nothing to split
        import shutil
        dst = path.join(cut_dir, "scene_0000.mkv")
        shutil.copyfile(keyed_timed_path, dst)
        return cut_dir, [0.0]

    seg_times = ",".join(f"{t:.3f}" for t in times)
    out_pattern = path.join(cut_dir, "scene_%04d.mkv")
    with _step(args, f"splitting into {len(times) + 1} individual scene clips...",
               total=vinfo["duration"]) as progress:
        _run([
            _ffmpeg_bin(), "-i", keyed_timed_path,
            "-f", "segment", "-segment_times", seg_times, "-reset_timestamps", "1",
            "-map", "0:v:0", "-c", "copy", "-y", out_pattern,
        ], log_fp, progress=progress)

    scene_starts = [0.0] + times
    return cut_dir, scene_starts


# Built-in EMA Decay/Buffer-by-scene-duration tables, one bucket per whole second from
# 0-1s up to 14-15s plus a 15s+ ceiling. Every scene now gets processed independently
# and resets its own temporal smoothing from nothing, so Buffer is kept safely under
# even the shortest clip in each bucket (using ~24fps: bucket N's shortest clip has
# about N*24 frames) -- otherwise the EMA never finishes "warming up" before the scene
# ends. Both Buffer and Decay increase smoothly bucket-to-bucket: short clips get light,
# fast-reacting smoothing (little footage to safely average over); long clips can afford
# progressively heavier smoothing. These largely supersede the earlier coarse 6-tier
# table (real production values 15/0.75 .. 120/0.98 at the 0-3s/10s+ ends are kept as
# anchor points here, just filled in with a step for every second in between).
#
# Two variants, chosen by which Depth Model you're using (see the "Auto EMA by Scene
# Length" model dropdown in the GUI):
#   VDA_L            -- a real video depth model with its own frame-to-frame memory,
#                        so it only needs this table's smoothing on top as a light touch.
#   Any_V3_Mono_01    -- a stills-only model with NO frame-to-frame memory of its own
#                        (prone to visible "depth breathing" without help), so this
#                        variant doubles VDA_L's Buffer at every bucket and raises Decay
#                        to match -- specifically, it halves (1 - decay), i.e. the EMA's
#                        own per-frame adaptation rate, so a doubled window is paired
#                        with proportionally heavier smoothing rather than the same
#                        smoothing spread over more frames. (One bucket, 1-2s, has its
#                        doubled Buffer trimmed back from 24 to 21 frames so it still
#                        stays safely under that bucket's shortest possible clip.)
EMA_BY_DURATION_VDA_L = [
    {"min_duration": 0, "max_duration": 1, "overrides": {"ema_buffer": 8, "ema_decay": 0.65}},
    {"min_duration": 1, "max_duration": 2, "overrides": {"ema_buffer": 12, "ema_decay": 0.70}},
    {"min_duration": 2, "max_duration": 3, "overrides": {"ema_buffer": 16, "ema_decay": 0.74}},
    {"min_duration": 3, "max_duration": 4, "overrides": {"ema_buffer": 20, "ema_decay": 0.78}},
    {"min_duration": 4, "max_duration": 5, "overrides": {"ema_buffer": 25, "ema_decay": 0.80}},
    {"min_duration": 5, "max_duration": 6, "overrides": {"ema_buffer": 30, "ema_decay": 0.83}},
    {"min_duration": 6, "max_duration": 7, "overrides": {"ema_buffer": 38, "ema_decay": 0.86}},
    {"min_duration": 7, "max_duration": 8, "overrides": {"ema_buffer": 46, "ema_decay": 0.89}},
    {"min_duration": 8, "max_duration": 9, "overrides": {"ema_buffer": 55, "ema_decay": 0.91}},
    {"min_duration": 9, "max_duration": 10, "overrides": {"ema_buffer": 65, "ema_decay": 0.93}},
    {"min_duration": 10, "max_duration": 11, "overrides": {"ema_buffer": 76, "ema_decay": 0.945}},
    {"min_duration": 11, "max_duration": 12, "overrides": {"ema_buffer": 87, "ema_decay": 0.955}},
    {"min_duration": 12, "max_duration": 13, "overrides": {"ema_buffer": 98, "ema_decay": 0.965}},
    {"min_duration": 13, "max_duration": 14, "overrides": {"ema_buffer": 108, "ema_decay": 0.972}},
    {"min_duration": 14, "max_duration": 15, "overrides": {"ema_buffer": 116, "ema_decay": 0.977}},
    {"min_duration": 15, "overrides": {"ema_buffer": 120, "ema_decay": 0.98}},
]

EMA_BY_DURATION_ANY_V3_MONO_01 = [
    {"min_duration": 0, "max_duration": 1, "overrides": {"ema_buffer": 16, "ema_decay": 0.825}},
    {"min_duration": 1, "max_duration": 2, "overrides": {"ema_buffer": 21, "ema_decay": 0.85}},
    {"min_duration": 2, "max_duration": 3, "overrides": {"ema_buffer": 32, "ema_decay": 0.87}},
    {"min_duration": 3, "max_duration": 4, "overrides": {"ema_buffer": 40, "ema_decay": 0.89}},
    {"min_duration": 4, "max_duration": 5, "overrides": {"ema_buffer": 50, "ema_decay": 0.90}},
    {"min_duration": 5, "max_duration": 6, "overrides": {"ema_buffer": 60, "ema_decay": 0.915}},
    {"min_duration": 6, "max_duration": 7, "overrides": {"ema_buffer": 76, "ema_decay": 0.93}},
    {"min_duration": 7, "max_duration": 8, "overrides": {"ema_buffer": 92, "ema_decay": 0.945}},
    {"min_duration": 8, "max_duration": 9, "overrides": {"ema_buffer": 110, "ema_decay": 0.955}},
    {"min_duration": 9, "max_duration": 10, "overrides": {"ema_buffer": 130, "ema_decay": 0.965}},
    {"min_duration": 10, "max_duration": 11, "overrides": {"ema_buffer": 152, "ema_decay": 0.9725}},
    {"min_duration": 11, "max_duration": 12, "overrides": {"ema_buffer": 174, "ema_decay": 0.9775}},
    {"min_duration": 12, "max_duration": 13, "overrides": {"ema_buffer": 196, "ema_decay": 0.9825}},
    {"min_duration": 13, "max_duration": 14, "overrides": {"ema_buffer": 216, "ema_decay": 0.986}},
    {"min_duration": 14, "max_duration": 15, "overrides": {"ema_buffer": 232, "ema_decay": 0.9885}},
    {"min_duration": 15, "overrides": {"ema_buffer": 240, "ema_decay": 0.99}},
]

EMA_BY_DURATION_TABLES = {
    "VDA_L": EMA_BY_DURATION_VDA_L,
    "Any_V3_Mono_01": EMA_BY_DURATION_ANY_V3_MONO_01,
}

# Old name kept as an alias (== the VDA_L table) in case anything else still imports it.
DEFAULT_EMA_BY_DURATION = EMA_BY_DURATION_VDA_L


def _load_scene_settings(settings_path, auto_ema_by_duration=False, auto_ema_model="VDA_L"):
    """Rules are applied in order with later matches overriding earlier ones, so the
    built-in EMA-by-duration table (if enabled) goes first -- it fills in EMA settings
    for every scene automatically, and anything the user's own JSON file explicitly
    sets (EMA included, if they want to override a specific tier, or anything else
    like Divergence) is layered on top and wins."""
    table = EMA_BY_DURATION_TABLES.get(auto_ema_model, EMA_BY_DURATION_VDA_L)
    rules = list(table) if auto_ema_by_duration else []
    if settings_path:
        with open(settings_path, "r", encoding="utf-8") as f:
            rules += json.load(f)
    return rules


def _overrides_for_scene(scene_settings, scene_index, scene_start_sec, scene_duration_sec):
    overrides = {}
    for rule in scene_settings:
        # Duration-based matching (e.g. "every scene 2-3 seconds long") is independent
        # of and can be combined with the position-based matching below -- a rule can
        # use either kind alone, or both together (both must match to apply).
        min_dur = rule.get("min_duration")
        max_dur = rule.get("max_duration")
        if min_dur is not None and scene_duration_sec < min_dur:
            continue
        if max_dur is not None and scene_duration_sec >= max_dur:
            continue

        lo = rule.get("start_scene", rule.get("start_time"))
        hi = rule.get("end_scene", rule.get("end_time"))
        if "start_scene" in rule or "end_scene" in rule:
            if lo is not None and scene_index < lo:
                continue
            if hi is not None and scene_index >= hi:
                continue
        else:
            if lo is not None and scene_start_sec < lo:
                continue
            if hi is not None and scene_start_sec >= hi:
                continue
        overrides.update(rule.get("overrides", {}))
    return overrides


def _load_scene_manifest(manifest_path):
    rows = {}
    if path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[row["filename"]] = row
    return rows


def _write_scene_manifest(manifest_path, rows):
    fieldnames = ["scene_index", "filename", "start_time_sec", "duration_sec",
                 "ema_buffer", "ema_decay", "other_overrides"]
    ordered = sorted(rows.values(), key=lambda r: int(r["scene_index"]))
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


def _process_scenes(cut_dir, converted_dir, scene_starts, scene_settings, args, depth_model, side_model):
    """Returns True if every scene was converted, False if Cancel stopped it early
    (in which case the caller must NOT go on to join -- see run_scene_batch)."""
    from .utils import process_video
    os.makedirs(converted_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(cut_dir) if f.endswith(".mkv"))
    stop_event = args.state.get("stop_event") if getattr(args, "state", None) else None

    # A per-scene record of exactly what got applied (EMA especially), so you can look
    # up any scene later instead of digging through the full log. Kept up to date after
    # every single scene (not just at the end) so a cancelled/interrupted run still
    # leaves an accurate manifest for whatever actually finished.
    manifest_path = path.join(path.dirname(converted_dir), "scene_manifest.csv")
    manifest_rows = _load_scene_manifest(manifest_path)

    # DV is handled once, outside iw3 entirely (extracted before splitting, reinjected
    # after joining) -- never per-clip, so force each individual conversion to skip its
    # own extract/reinject work regardless of what the user passed on the command line.
    #
    # Also force Resume on: iw3's per-file conversion already skips a scene whose output
    # exists (that's what Resume does), but the GUI auto-disables that checkbox for a
    # single-video-file input, since it doesn't know each scene is really its own file
    # under the hood here. Forcing it on is what actually makes "cancel and restart" not
    # throw away every scene already converted.
    saved_preserve_dowi = getattr(args, "preserve_dowi", False)
    saved_resume = getattr(args, "resume", False)
    args.preserve_dowi = False
    args.resume = True
    skipped = sum(1 for fname in files if path.exists(path.join(converted_dir, fname)))
    if skipped:
        print(f"[scene-batch] {skipped}/{len(files)} scenes already converted, skipping those.",
              file=sys.stderr)
    try:
        for i, fname in enumerate(files):
            if stop_event is not None and stop_event.is_set():
                print(f"[scene-batch] cancelled -- stopped after {i}/{len(files)} scenes. "
                      f"Nothing already converted is lost; start this same job again with the "
                      f"same Output to pick up right where this left off.", file=sys.stderr)
                return False

            scene_start = scene_starts[i] if i < len(scene_starts) else scene_starts[-1]
            scene_duration = _video_stream_info(path.join(cut_dir, fname))["duration"]
            overrides = _overrides_for_scene(scene_settings, i, scene_start, scene_duration)
            saved = {}
            for k, v in overrides.items():
                saved[k] = getattr(args, k, None)
                setattr(args, k, v)
            try:
                print(f"[scene-batch] converting scene {i + 1}/{len(files)}: {fname}"
                      + (f"  overrides={overrides}" if overrides else ""), file=sys.stderr)
                process_video(path.join(cut_dir, fname), converted_dir, args, depth_model, side_model)
            finally:
                for k, v in saved.items():
                    setattr(args, k, v)

            if stop_event is not None and stop_event.is_set():
                # Cancel landed mid-scene: process_video's own stop check left an
                # unrenamed "_tmp_"-prefixed file behind instead of the real scene
                # output (see nunif/utils/video/processor.py), so this scene did NOT
                # actually finish -- don't record it in the manifest as done, and
                # don't start another one.
                print(f"[scene-batch] cancelled -- stopped after {i}/{len(files)} scenes "
                      f"(scene {i + 1} was in progress and did not finish). Nothing already "
                      f"converted is lost; start this same job again with the same Output to "
                      f"pick up right where this left off.", file=sys.stderr)
                return False

            other = {k: v for k, v in overrides.items() if k not in ("ema_buffer", "ema_decay")}
            manifest_rows[fname] = {
                "scene_index": i,
                "filename": fname,
                "start_time_sec": f"{scene_start:.3f}",
                "duration_sec": f"{scene_duration:.3f}",
                "ema_buffer": overrides.get("ema_buffer", ""),
                "ema_decay": overrides.get("ema_decay", ""),
                "other_overrides": json.dumps(other) if other else "",
            }
            _write_scene_manifest(manifest_path, manifest_rows)
        return True
    finally:
        args.preserve_dowi = saved_preserve_dowi
        args.resume = saved_resume


def _reinject_rpu_final(output_path, rpu_path, work_dir, args, log_fp):
    """Reinject the once-extracted Dolby Vision RPU onto the finished, joined,
    audio-complete 3D output. This is the ONLY place DV is touched after extraction --
    never per-clip."""
    vinfo = _video_stream_info(output_path)
    hevc_out = path.join(work_dir, "final_out.hevc")
    hevc_dv = path.join(work_dir, "final_out_dv.hevc")
    dv_tmp = output_path + ".dv_inject.mkv"

    with _step(args, "reinjecting Dolby Vision RPU onto the finished video (once)...",
               total=vinfo["duration"]) as progress:
        _run([_ffmpeg_bin(), "-y", "-i", output_path, "-c:v", "copy", "-an", "-sn", "-f", "hevc",
              hevc_out], log_fp, progress=progress)
        _run([_dovi_bin(), "inject-rpu", "-i", hevc_out, "-r", rpu_path, "-o", hevc_dv], log_fp)

        fps_num, fps_den = vinfo["r_frame_rate"].split("/")
        timed_mkv = path.join(work_dir, "final_dv_timed.mkv")
        _run([_mkvmerge_bin(), "-o", timed_mkv, "--default-duration", f"0:{fps_num}/{fps_den}fps",
              hevc_dv], log_fp)

        _run([_ffmpeg_bin(), "-y", "-i", timed_mkv, "-i", output_path,
              "-map", "0:v:0", "-map", "1", "-map", "-1:v", "-c", "copy", dv_tmp], log_fp)
        os.replace(dv_tmp, output_path)

    for f in (hevc_out, hevc_dv):
        if path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass


def _join_and_finalize(converted_dir, original_source, output_path, args, log_fp, rpu_path=None):
    files = sorted(f for f in os.listdir(converted_dir) if f.endswith((".mkv", ".mp4")))
    if not files:
        raise RuntimeError("No converted scene clips were produced.")

    # Normalize resolution defensively (should already match since crop happened once).
    dims = set()
    total_duration = 0.0
    for f in files:
        info = _video_stream_info(path.join(converted_dir, f))
        dims.add((info["width"], info["height"]))
        total_duration += info["duration"]

    work_dir = path.dirname(converted_dir)
    video_only = path.join(work_dir, "joined_video.mkv")

    # VR Optimized Merge: skip the fast stream-copy concat path even when every
    # scene already matches resolution, and re-encode the whole timeline as one
    # continuous stream instead. Stream-copy concat is fast and lossless, but each
    # source scene clip keeps its own independent GOP/keyframe structure and
    # timestamps -- fine for normal playback, but can read as uneven/stuttery on a
    # VR headset where frame pacing matters more. Off by default (the fast path
    # remains the default for everyone not specifically targeting VR playback).
    vr_optimized_merge = bool(getattr(args, "vr_optimized_merge", False))
    vr_merge_fps = getattr(args, "vr_merge_fps", None)  # None = keep source fps

    join_desc = (f"joining {len(files)} converted scenes into one video "
                f"({'matching resolution' if len(dims) == 1 else 'normalizing mismatched resolutions'}"
                f"{', VR-optimized re-encode' if vr_optimized_merge else ''})...")
    # The fast plain-copy concat path doesn't meaningfully benefit from a live
    # percentage (it's typically done in seconds), only the NVENC re-encode fallback.
    step_total = 1 if (len(dims) == 1 and not vr_optimized_merge) else total_duration
    with _step(args, join_desc, total=step_total) as progress:
        if len(dims) == 1 and not vr_optimized_merge:
            list_file = path.join(work_dir, "concat_list.txt")
            with open(list_file, "w", encoding="utf-8", newline="\n") as f:
                for fname in files:
                    p = path.join(converted_dir, fname).replace("'", "'\\''")
                    f.write(f"file '{p}'\n")
            _run([_ffmpeg_bin(), "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                  "-c", "copy", video_only], log_fp)
        elif len(dims) == 1 and vr_optimized_merge:
            # Same fast concat demuxer to build the timeline (still avoids
            # re-decoding N separate scene files through a filter graph), but then
            # ALWAYS re-encode that concatenated stream as one pass -- this is what
            # actually regenerates clean, continuous timestamps/GOP structure and
            # lets a specific constant frame rate be forced, which stream-copy alone
            # can never do regardless of how the concat itself was built.
            list_file = path.join(work_dir, "concat_list.txt")
            with open(list_file, "w", encoding="utf-8", newline="\n") as f:
                for fname in files:
                    p = path.join(converted_dir, fname).replace("'", "'\\''")
                    f.write(f"file '{p}'\n")
            concat_cmd = [_ffmpeg_bin(), "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                          "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "16", "-b:v", "0",
                          "-pix_fmt", "yuv420p"]
            if vr_merge_fps:
                concat_cmd += ["-r", str(vr_merge_fps), "-vsync", "cfr"]
            if path.splitext(video_only)[1].lower() in (".mp4", ".mov", ".m4v"):
                concat_cmd += ["-movflags", "+faststart"]
            concat_cmd += [video_only]
            _run(concat_cmd, log_fp, progress=progress)
        else:
            # mixed resolutions: normalize via crop-or-pad to the most common size
            target_w, target_h = max(dims, key=lambda d: sum(1 for f in files
                                                              if _video_stream_info(path.join(converted_dir, f))
                                                              ["width"] == d[0]))
            args_list = []
            script_lines = []
            for i, fname in enumerate(files):
                args_list += ["-i", path.join(converted_dir, fname)]
                script_lines.append(
                    f"[{i}:v]crop=min(iw\\,{target_w}):min(ih\\,{target_h}):"
                    f"(iw-min(iw\\,{target_w}))/2:(ih-min(ih\\,{target_h}))/2,"
                    f"pad={target_w}:{target_h}:({target_w}-iw)/2:({target_h}-ih)/2:color=black[v{i}];")
            tags = "".join(f"[v{i}]" for i in range(len(files)))
            script_lines.append(f"{tags}concat=n={len(files)}:v=1:a=0[outv]")
            script_path = path.join(work_dir, "join_filter.txt")
            with open(script_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(script_lines))
            mismatch_cmd = [_ffmpeg_bin(), *args_list, "-filter_complex_script", script_path, "-map", "[outv]",
                            "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "16", "-b:v", "0"]
            if vr_optimized_merge and vr_merge_fps:
                mismatch_cmd += ["-r", str(vr_merge_fps), "-vsync", "cfr"]
            if vr_optimized_merge and path.splitext(video_only)[1].lower() in (".mp4", ".mov", ".m4v"):
                mismatch_cmd += ["-movflags", "+faststart"]
            mismatch_cmd += ["-y", video_only]
            _run(mismatch_cmd, log_fp, progress=progress)

    video_duration = _video_stream_info(video_only)["duration"]
    audio_slice = path.join(work_dir, "audio_slice.mkv")
    audio_desc = "pulling matching audio"
    audio_cmd = [_ffmpeg_bin(), "-y", "-i", original_source, "-t", f"{video_duration}", "-map", "0:a"]
    if vr_optimized_merge:
        # Re-encode to a broadly VR-headset/player-compatible spec rather than
        # passing through whatever the source audio codec is (which could be
        # something like DTS/TrueHD that not every VR player handles well) --
        # only done in this opt-in mode, since the default path already preserves
        # the original audio losslessly, which is usually the better choice.
        audio_cmd += ["-c:a", "aac", "-ar", "48000", "-b:a", "192k"]
        audio_desc += " (re-encoding to AAC 48kHz for VR compatibility)"
    else:
        audio_cmd += ["-c", "copy"]
    audio_cmd += [audio_slice]
    with _step(args, f"{audio_desc} ({video_duration:.2f}s) from the original source...",
               total=video_duration) as progress:
        _run(audio_cmd, log_fp, progress=progress)

    with _step(args, "muxing video + audio into the final output...", total=video_duration) as progress:
        make_parent_dir(output_path)
        _run([_ffmpeg_bin(), "-y", "-i", video_only, "-i", audio_slice,
              "-map", "0:v:0", "-map", "1:a", "-c", "copy", output_path], log_fp, progress=progress)

    if rpu_path is not None:
        work_dir = path.dirname(converted_dir)
        _reinject_rpu_final(output_path, rpu_path, work_dir, args, log_fp)


def _foundation_registry_path():
    return path.join(_bin_dir(), "nunif", "tmp", "scene_batch_foundations.json")


def _foundation_key(input_path):
    return path.normcase(path.abspath(input_path))


def _load_foundation_registry():
    reg_path = _foundation_registry_path()
    if not path.exists(reg_path):
        return {}
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_foundation_registry(registry):
    try:
        make_parent_dir(_foundation_registry_path())
        with open(_foundation_registry_path(), "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception:
        pass


def _is_foundation_ready(foundation_dir):
    """A foundation isn't actually usable until BOTH the crop/keyframe master AND the
    scene split are done -- checking only the master file used to let a run that got
    interrupted between those two steps look "ready" when scene_cuts was still empty,
    which made the split step blow up later with nothing to recover from."""
    if not path.exists(path.join(foundation_dir, "prepared_keyed_timed.mkv")):
        return False
    cut_dir = path.join(foundation_dir, "scene_cuts")
    if not path.isdir(cut_dir):
        return False
    return any(path.isfile(path.join(cut_dir, name)) for name in os.listdir(cut_dir))


def _find_foundation(input_path, local_work_dir):
    """Figures out where the shared, expensive-to-build crop/keyframe/scene-split
    foundation for this exact input movie should come from. Every Scene Batch run
    used to assume that foundation could only live right next to ITS OWN Output
    path, so pointing Output at a different folder (a new movie name, a new
    experiment folder, a typo) silently redid the whole crop/keyframe pass from
    scratch even though a perfectly good one already existed elsewhere. This checks
    a small registry (built up by _register_foundation below) of every foundation
    ever finished for this input file, and reuses the most recent one that's still
    actually there instead."""
    if _is_foundation_ready(local_work_dir):
        return local_work_dir

    existing = _load_foundation_registry().get(_foundation_key(input_path))
    if existing and existing != local_work_dir and _is_foundation_ready(existing):
        print(f"[scene-batch] found existing crop/keyframe/scene-split work for this movie "
              f"at {existing} -- reusing it instead of redoing that step.", file=sys.stderr)
        return existing

    # Nothing usable found (first time for this movie, or an earlier location's files
    # are gone) -- this run will build the foundation fresh, right here.
    return local_work_dir


def _register_foundation(input_path, foundation_dir):
    key = _foundation_key(input_path)
    registry = _load_foundation_registry()
    if registry.get(key) != foundation_dir:
        registry[key] = foundation_dir
        _save_foundation_registry(registry)


def run_scene_batch(args, depth_model, side_model):
    input_path = path.abspath(args.input)
    output_path = args.output
    # Defend against Output accidentally pointing at a leftover work folder from a
    # previous attempt (e.g. after a crash) -- recover the real target instead of
    # computing a nested/doubled work folder from it.
    WORK_SUFFIX = ".scene_batch_work"
    if output_path.endswith(WORK_SUFFIX):
        output_path = output_path[:-len(WORK_SUFFIX)]
    if path.isdir(output_path) or path.splitext(output_path)[1] == "":
        os.makedirs(output_path, exist_ok=True)
        output_path = path.join(output_path, path.splitext(path.basename(input_path))[0] + "_3d.mkv")

    work_dir = output_path + WORK_SUFFIX
    os.makedirs(work_dir, exist_ok=True)
    log_path = path.join(work_dir, "scene_batch.log")

    # A variant name lets a second (third, ...) pass reuse the shared, expensive-to-
    # produce foundation (RPU extraction, crop+keyframe pass, the split scene clips)
    # from a prior run of the SAME movie, while writing its own converted scenes and
    # final output to their own location -- so trying different settings (e.g. a
    # different EMA table) never touches or overwrites an earlier attempt's progress.
    variant = getattr(args, "scene_batch_variant", None)
    if variant:
        base, ext = path.splitext(output_path)
        output_path = f"{base}_{variant}{ext}"
        converted_dir = path.join(work_dir, f"variant_{variant}", "converted")
    else:
        converted_dir = path.join(work_dir, "converted")

    scene_settings = _load_scene_settings(
        getattr(args, "scene_settings", None),
        auto_ema_by_duration=getattr(args, "scene_batch_auto_ema", False),
        auto_ema_model=getattr(args, "scene_batch_auto_ema_model", None) or "VDA_L",
    )

    with open(log_path, "a", encoding="utf-8") as log_fp:
        # Mirror everything printed anywhere during the run into the log file too --
        # not just this module's own steps, but iw3's own per-scene conversion
        # progress as well, so the log ends up a complete transcript of the run.
        real_stdout, real_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(real_stdout, log_fp)
        sys.stderr = _Tee(real_stderr, log_fp)
        try:
            # Reuse an already-built crop/keyframe/scene-split foundation for this same
            # input movie wherever it actually lives on disk, instead of only ever
            # checking right next to THIS run's own Output path -- see _find_foundation.
            foundation_dir = _find_foundation(input_path, work_dir)
            if foundation_dir != work_dir:
                os.makedirs(foundation_dir, exist_ok=True)

            rpu_path = _extract_rpu_once(input_path, foundation_dir, args, log_fp)
            if rpu_path is not None and getattr(args, "video_codec", None) not in ("hevc_nvenc", "libx265"):
                # Dolby Vision reinjection only works on an HEVC bitstream (dovi_tool requirement).
                # Catch this now, before spending time converting every scene, not at the final step.
                print(f"[scene-batch] Dolby Vision requires HEVC output -- overriding Video Codec "
                      f"({getattr(args, 'video_codec', None)!r} -> hevc_nvenc) for this run.", file=sys.stderr)
                args.video_codec = "hevc_nvenc"
            if _is_foundation_ready(foundation_dir):
                # The crop/keyframe/split foundation is already fully built -- detection's
                # only purpose is telling _prepare_and_key/_split_scenes where to cut, and
                # both already skip straight past that once their real output exists, so
                # redoing a multi-minute AI scan here would find cut points nothing uses.
                times = None
            else:
                times = _detect_scenes(input_path, args, log_fp)
            keyed_timed, vinfo = _prepare_and_key(input_path, times, foundation_dir, args, log_fp)
            cut_dir, scene_starts = _split_scenes(keyed_timed, times, vinfo, foundation_dir, args, log_fp)
            # The foundation is now fully built (whether it was reused or just made fresh
            # here) -- record where it lives so any future run against this same input,
            # from any Output location, finds and reuses it too.
            _register_foundation(input_path, foundation_dir)
            completed = _process_scenes(cut_dir, converted_dir, scene_starts, scene_settings,
                                        args, depth_model, side_model)
            if not completed:
                # Cancel stopped this mid-run -- do NOT join/finalize on a partial set of
                # scenes (that used to silently produce a shortened "finished" file). Every
                # scene converted so far is kept exactly as-is on disk; starting this same
                # job again (same Output path) resumes from here via the skip-already-
                # converted check at the top of _process_scenes.
                print(f"[scene-batch] stopped by cancel -- not joining a partial result. "
                      f"Run this same job again to continue from here.", file=sys.stderr)
                return None
            _join_and_finalize(converted_dir, input_path, output_path, args, log_fp, rpu_path=rpu_path)
            print(f"[scene-batch] done: {output_path}", file=sys.stderr)
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr

    return output_path
