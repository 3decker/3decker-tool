"""iw3.direct_mvc_cli -- ADR-283: single-pass 2D-to-3D-Blu-ray-MVC conversion.

Real user request (relayed by decker): could the existing two-stage "convert to a
finished SBS file, then run iw3.sbs_to_mvc_cli on that file" pipeline instead happen
as ONE continuous pass, with no finished SBS file ever touching disk? This module is
that opt-in alternative mode, wired up as the "Direct to 3D Blu-ray MVC" checkbox in
iw3/gui.py (see on_changed_chk_direct_mvc / parse_args there).

How it works: the main iw3 conversion (iw3.utils.process_video_full()) is handed a
`raw_frame_sink` callback (nunif/utils/video/processor.py's ADR-283 addition) instead
of a real output file -- every finished packed-stereo frame's raw bytes go straight
into a downstream `ffmpeg | FRIMEncode` pipe (the exact same two-process chain
iw3.sbs_to_mvc_cli.convert() already proves out against a real finished file, see
sbs_to_mvc_cli.py:846-865 and its own module docstring for why that specific chain
exists), except ffmpeg reads from stdin ("-i -") instead of a file. FRIM's two
elementary streams are then muxed exactly the way sbs_to_mvc_cli.convert() already
does, with audio/subtitles extracted from the ORIGINAL 2D source (not any
intermediate) since that source is a real, complete, seekable file on its own.

Deliberately NOT supported in this mode (see the checkbox's own tooltip in gui.py):
  - Auto Resume: process_video_with_resume()'s whole design needs real, independently
    reopenable segment files on disk -- there is no file here to reopen.
  - RIFE frame interpolation: out of scope for this first version.
  - MVC Auto-crop: sbs_to_mvc_cli's own autocrop (detect_eye_crop()) needs random-
    access sampling across an already-finished file; there is no finished file yet
    at the point this mode would need to decide a crop.
If interrupted, the whole job must be started over -- there is no partial-progress
checkpoint for a live subprocess pipe the way there is for on-disk segment files.

HDR sources are handled the same way process_video()'s own --hdr-to-sdr already
works for the regular pipeline (iw3.utils._tonemap_hdr_to_sdr): a real 3D Blu-ray/MVC
file can never carry HDR/Dolby Vision at all (ADR-252), so this refuses outright
unless the caller already turned that on -- the GUI layer (gui.py) is expected to
catch this earlier with a pre-flight prompt (mirroring ADR-281's existing one for the
two-stage flow), but this module still refuses on its own rather than silently
producing a broken/HDR-tagged-as-SDR file.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
from copy import copy
from os import path

from .mvc_extract_cli import Cancelled, iso_to_bd_folder, _remove_stale_temp
from .sbs_to_mvc_cli import (bd_frame_rate, eye_filter, find_frim, probe_video, _plan_audio_subs,
                             _mux_mkv_via_mkvmerge, _mvc_bitrate_ceiling_mbps)
from .utils import (
    _find_tsmuxer, _get_ffmpeg_bin, _tonemap_hdr_to_sdr, _notify_stage, _StageBar,
    process_video_full, STAGE_CONVERT_MVC, STAGE_DEPTH_STEREO, log_subprocess_cmd,
)
from nunif.utils.video.metadata import parse_time

_MVC_LAYOUT_INCOMPATIBLE_FLAGS = (
    "vr180", "cross_eyed", "rgbd", "half_rgbd", "anaglyph", "export", "export_disparity", "debug_depth",
)

# Real sub-stage breakdown found inside the single "Converting to 3D Blu-ray MVC" window
# (ADR-283 Progress Reporting follow-up, relayed by decker): the real work after the main
# depth/stereo loop finishes handing frames to the pipe is (1) FRIM draining its own
# encode backlog (the ffmpeg eye-crop/scale leg has no progress signal of its own -- same
# as sbs_to_mvc_cli.py's own "encode" stage, which already covers both under one real
# frame-count signal read from FRIM's own stderr), then (2) tsMuxeR's real mux, then,
# BD-Folder output only, (3) copying the finished .iso's contents out to a folder. Labels
# match this project's existing "Converting to MVC: <sub-label>" style
# (iw3/utils.py's _MVC_STAGE_LABELS) so the two MVC pipelines read the same way to a user.
_DIRECT_MVC_STAGE_PREFIX = "Converting to 3D Blu-ray MVC: "
_FRIM_FRAME_RE = re.compile(rb"Frame number:\s*(\d+)")


def _layout_from_args(args):
    """Same real Stereo-Format-checkbox logic iw3.utils._run_mvc_conversion() already
    uses for the two-stage flow -- duplicated in full rather than imported, since that
    function's own copy is inline in the middle of building its subprocess command
    line and isn't its own callable. Both must stay in sync if a new Stereo Format is
    ever added to either MVC path."""
    if getattr(args, "half_sbs", False):
        return "half_sbs"
    if getattr(args, "tb", False):
        return "full_tb"
    if getattr(args, "half_tb", False):
        return "half_tb"
    if not any(getattr(args, f, False) for f in _MVC_LAYOUT_INCOMPATIBLE_FLAGS):
        return "full_sbs"
    return None


class _DirectMvcPipe:
    """Lazily spawns `ffmpeg | FRIMEncode` on the FIRST frame, not before -- the
    packed stereo frame's real size is decided by process_video_full()'s own
    depth/stereo pipeline (divergence/pad/resolution/etc against the SOURCE
    resolution), not something this module can compute up front the way
    sbs_to_mvc_cli.convert() can (that function probes an already-finished file's
    real dimensions before ever building its ffmpeg command)."""

    def __init__(self, ffmpeg_bin, frim_bin, layout, fps_frac, bitrate_mbps, swap_eyes,
                 base_es, dep_es, ffmpeg_log_path, stop_event, fill_mode="fit"):
        self._ffmpeg_bin = ffmpeg_bin
        self._frim_bin = frim_bin
        self._layout = layout
        self._fps_frac = fps_frac
        self._bitrate_mbps = bitrate_mbps
        self._swap_eyes = swap_eyes
        self._base_es = base_es
        self._dep_es = dep_es
        self._ffmpeg_log_path = ffmpeg_log_path
        self._stop_event = stop_event
        self._fill_mode = fill_mode
        self.ff = None
        self.fr = None
        self._ff_log = None
        self._tail = []
        self._reader = None
        self.frame_count = 0
        # Real encode progress (not just frames handed off to the pipe -- ffmpeg/FRIM may
        # still be draining a backlog well after the depth/stereo loop above has handed off
        # its very last frame): parsed from FRIM's own stderr, same "Frame number: N" line
        # and regex sbs_to_mvc_cli.py's own "encode" stage already proves out -- FRIM reads
        # from stdin ("-i -") in BOTH pipelines, so its own output format is identical here.
        self.frim_frame = 0

    def write(self, raw_bytes, width, height, pix_fmt, colorspace, color_primaries, color_trc, color_range):
        if self._stop_event is not None and self._stop_event.is_set():
            raise Cancelled()
        if self.ff is None:
            self._start(width, height, pix_fmt, colorspace, color_primaries, color_trc, color_range)
        try:
            self.ff.stdin.write(raw_bytes)
        except (BrokenPipeError, OSError) as e:
            # Real gap found from a live user crash (2026-09-26): a bare "Broken pipe" told the
            # user nothing about WHY FRIM/ffmpeg actually died -- give the same real diagnostic
            # (FRIM's own tail output + ffmpeg's log) that finish() already builds on its own
            # failure path, instead of just the OS-level symptom.
            detail = self._diagnose_failure(wait_seconds=1.0)
            raise RuntimeError(
                f"the MVC encode pipe closed unexpectedly while writing frame {self.frame_count + 1} "
                f"(FRIMEncode likely crashed or refused): {e}"
                + (f"\n{detail}" if detail else "")) from e
        self.frame_count += 1

    def _diagnose_failure(self, wait_seconds=0.0):
        """Best-effort real diagnostic: FRIM's own last output lines plus ffmpeg's log tail,
        the same shape finish() already surfaces on its own failure path. Never raises --
        called from an exception handler, so a second failure here must not mask the first."""
        try:
            if self.fr is not None:
                try:
                    self.fr.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    pass
                if self._reader is not None:
                    self._reader.join(timeout=1.0)
            frim_msg = "\n".join(self._tail[-8:]) if self._tail else \
                "(FRIMEncode produced no output at all before exiting)"
            ff_msg = ""
            if self._ff_log is not None:
                try:
                    self._ff_log.flush()
                except Exception:
                    pass
            if self._ffmpeg_log_path and path.exists(self._ffmpeg_log_path):
                with open(self._ffmpeg_log_path, "rb") as f:
                    ff_msg = f.read()[-600:].decode(errors="replace")
            parts = [f"FRIMEncode output: {frim_msg}"]
            if ff_msg.strip():
                parts.append(f"ffmpeg (likely just a downstream symptom of FRIM's pipe closing, "
                             f"not ffmpeg's own problem): {ff_msg}")
            return "\n".join(parts)
        except Exception:
            return ""

    def _start(self, width, height, pix_fmt, colorspace, color_primaries, color_trc, color_range):
        # Same ffmpeg eye-split/scale chain sbs_to_mvc_cli.convert() already uses
        # (eye_filter() -> "-pix_fmt yuv420p ... -f rawvideo -" into FRIM), except
        # this leg's OWN input is a headerless raw pipe (not a real file with its own
        # container), so it needs explicit -f rawvideo/-pix_fmt/-s/-r on the input
        # side too, unlike convert()'s "-i <real file>".
        # Real live-test finding: raw video (-f rawvideo) carries NO embedded color
        # metadata at all -- without these 4 flags telling ffmpeg how to interpret the
        # incoming YCbCr bytes, it guesses wrong, producing a visibly desaturated/
        # wrong-colored file (confirmed side-by-side against the same clip run through
        # the normal two-stage path with identical settings). The values are the real,
        # per-frame ones processor.py's output_reformatter already computed -- same
        # PyAV enum ints ffmpeg's own AVOption parser accepts directly.
        vf = eye_filter(self._layout, width, height, None, self._fill_mode)
        ff_cmd = [self._ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                  "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}", "-r", self._fps_frac,
                  "-colorspace", str(colorspace), "-color_primaries", str(color_primaries),
                  "-color_trc", str(color_trc), "-color_range", str(color_range),
                  # Real, decker-confirmed fix (same bug/fix as sbs_to_mvc_cli.convert()'s own
                  # ffmpeg->FRIM pipe): a forced output "-r" on a rawvideo target makes ffmpeg
                  # run its own internal CFR filter, which genuinely dups/drops frames in a
                  # periodic pattern. rawvideo carries no embedded timing anyway, and FRIM's
                  # own "-f fps_frac" below already declares the authoritative output rate
                  # independently, so passthrough loses nothing. (The INPUT "-r" above stays --
                  # it tells ffmpeg how to interpret this leg's headerless incoming raw pipe,
                  # a different purpose from this output flag.)
                  "-i", "-", "-an", "-sn", "-vf", vf, "-pix_fmt", "yuv420p",
                  "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
        target = int(self._bitrate_mbps * 1000)
        frim_cmd = [self._frim_bin, "-i", "-", "-o:mvc", self._base_es, self._dep_es, "-viewoutput",
                    "-sbs", "2", "-w", "1920", "-h", "1080", "-f", self._fps_frac,
                    "-profile", "high", "-level", "4.1",
                    "-vbr", str(target), str(int(target * 1.25)), "-sw"]
        if self._swap_eyes:
            frim_cmd.append("-swaplr")

        log_subprocess_cmd("direct-mvc:ffmpeg", ff_cmd)
        log_subprocess_cmd("direct-mvc:frim", frim_cmd)
        self._ff_log = open(self._ffmpeg_log_path, "wb")
        self.ff = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._ff_log)
        self.fr = subprocess.Popen(frim_cmd, stdin=self.ff.stdout, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
        self.ff.stdout.close()

        def _read():
            buf = b""
            while True:
                chunk = self.fr.stdout.read(256)
                if not chunk:
                    break
                buf += chunk
                while True:
                    m = re.search(rb"[\r\n]", buf)
                    if not m:
                        break
                    line, buf = buf[:m.start()], buf[m.end():]
                    self._tail.append(line.decode(errors="replace"))
                    del self._tail[:-20]
                    fm = _FRIM_FRAME_RE.search(line)
                    if fm:
                        self.frim_frame = int(fm.group(1))

        self._reader = threading.Thread(target=_read, daemon=True)
        self._reader.start()

    def finish(self, on_progress=None):
        """Closes ffmpeg's stdin -- the real EOF signal down the pipe, same idiom
        sbs_to_mvc_cli.convert() uses (its own `ff.stdout.close()`) to end its side of
        an equivalent pipe cleanly -- then waits for both processes and raises with
        the same real diagnostic shape convert() already uses on failure.

        on_progress(frim_frame), if given, is called about every 0.5s while waiting for
        FRIM to actually finish encoding (real backlog-draining time, which can run well
        past the moment the depth/stereo loop above already handed off its last frame),
        and once more right after -- same polling cadence this loop already used, just
        also reporting what it already knows."""
        if self.ff is None:
            raise RuntimeError("no frames were produced -- nothing to encode (the source may be "
                               "empty, or the job was cancelled before the first frame)")
        try:
            self.ff.stdin.close()
        except OSError:
            pass
        while self.fr.poll() is None:
            if self._stop_event is not None and self._stop_event.is_set():
                self.ff.kill()
                self.fr.kill()
                raise Cancelled()
            try:
                self.fr.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            if on_progress is not None:
                on_progress(self.frim_frame)
        if self._reader is not None:
            self._reader.join()
        if on_progress is not None:
            on_progress(self.frim_frame)
        self.ff.wait()
        self._ff_log.close()

        if self.fr.returncode != 0 or not (path.exists(self._base_es) and path.getsize(self._base_es) > 0
                                            and path.exists(self._dep_es) and path.getsize(self._dep_es) > 0):
            raise RuntimeError(
                f"the MVC encode failed (FRIMEncode exit code {self.fr.returncode}):\n"
                + self._diagnose_failure())

    def kill(self):
        for p in (self.ff, self.fr):
            if p is not None and p.poll() is None:
                p.kill()
        if self._ff_log is not None:
            try:
                self._ff_log.close()
            except Exception:
                pass


def convert_direct(original_source_path, output_path, args, depth_model, side_model):
    """The whole direct-to-MVC job: probes `original_source_path`, runs the main iw3
    conversion straight into an ffmpeg|FRIM pipe (no SBS file ever written), then muxes
    FRIM's output plus audio/subtitles pulled from `original_source_path` itself into
    `output_path` (.iso/.m2ts/.mkv/folder, all built through tsMuxeR's own native
    multi-format output -- ADR-303). Returns output_path on success; raises
    RuntimeError/ValueError (with a real, specific reason) on any failure or refusal,
    or mvc_extract_cli.Cancelled if args.state's stop_event fired mid-job."""
    width, height, rate, duration, hdr = probe_video(original_source_path)
    fps_text, fps_frac = bd_frame_rate(rate)  # raises ValueError naming the actual fps if illegal

    # Upper-bound frame estimate for the "encoding the real 3D (MVC) video" progress bar
    # below -- same args.start_time/args.end_time-against-probed-duration computation
    # iw3.utils.process_video_with_resume() already uses for its own segment planning, so
    # a trimmed (Start/End Time) run gets an accurate total instead of the whole source's.
    effective_start = parse_time(args.start_time) if getattr(args, "start_time", None) else 0.0
    effective_end = parse_time(args.end_time) if getattr(args, "end_time", None) else duration
    if duration:
        effective_end = min(effective_end, duration)
    total_frames = max(1, int((effective_end - effective_start) * float(fps_text))) if duration else 1

    if hdr and not getattr(args, "hdr_to_sdr", False):
        raise RuntimeError(
            "This source looks like HDR/Dolby Vision, but a real 3D Blu-ray/MVC file cannot carry "
            "HDR at all -- turn on \"Convert HDR to SDR\" first (Direct to 3D Blu-ray MVC refuses "
            "rather than silently producing a broken file).")

    layout = _layout_from_args(args)
    if layout is None:
        raise ValueError("Direct to 3D Blu-ray MVC needs Stereo Format set to Full SBS, Half SBS, "
                         "Full TB, or Half TB.")

    output_path = str(output_path)
    # ADR-295: see sbs_to_mvc_cli.py's convert() docstring for the full reasoning --
    # path.splitext() incorrectly picks up a mid-string '.' from this app's own
    # filename tags (e.g. "fs0.0") as if it were the real extension once the true
    # trailing extension is stripped, breaking folder-mode detection. Checking the
    # known suffixes directly instead sidesteps this.
    lower_output = output_path.lower()
    is_mkv_output = lower_output.endswith(".mkv")
    is_m2ts_output = lower_output.endswith(".m2ts")
    is_iso_output = lower_output.endswith(".iso")
    is_folder_output = not (is_mkv_output or is_m2ts_output or is_iso_output)
    # Same disc-legal grouping sbs_to_mvc_cli.convert() uses -- see its own real research
    # finding comment above _mvc_bitrate_ceiling_mbps() for why the bitrate ceiling (and audio-
    # codec legality, and the muxopt choice below) key off this instead of a single flat number.
    disc_legal = is_iso_output or is_folder_output

    frim = find_frim()
    if frim is None:
        raise RuntimeError("FRIMEncode not found -- run `python -m iw3.install_mvc_tools`")
    ffmpeg = _get_ffmpeg_bin()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    tsmuxer = _find_tsmuxer()
    if tsmuxer is None:
        raise RuntimeError("tsMuxeR not found -- run `python -m iw3.install_mvc_tools`")

    bitrate_mbps = float(getattr(args, "mvc_bitrate", 20.0) or 20.0)
    bitrate_ceiling = _mvc_bitrate_ceiling_mbps(disc_legal)
    if not 2 <= bitrate_mbps <= bitrate_ceiling:
        raise ValueError(
            f"bitrate must be between 2 and {bitrate_ceiling:g} Mbps "
            + ("(3D Blu-ray disc output -- .iso/BD-folder -- is combined-bitrate-limited to about "
               "60 Mbps)" if disc_legal else
               "(non-disc MVC output is limited by this encode's own AVC Level 4.1 High Profile "
               "ceiling of 62.5 Mbps)"))

    out_dir = path.dirname(path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = path.splitext(path.basename(output_path))[0]
    work_dir = path.join(out_dir, f"_direct_mvc_work_{stem}")
    os.makedirs(work_dir, exist_ok=True)

    base_es, dep_es = path.join(work_dir, "base.264"), path.join(work_dir, "dep.264")
    ffmpeg_log = path.join(work_dir, "ffmpeg.log")
    # Same real, reproduced bug class as sbs_to_mvc_cli.py's own convert() (see its
    # comment here): work_dir is named only from the output filename, so a retry after
    # any failed/cancelled run silently reuses the same dirty folder and FRIM refuses to
    # write base_es/dep_es because they already exist -- mvc_extract_cli.py's own
    # _remove_stale_temp() already solves this; this tool never called it.
    _remove_stale_temp(base_es, dep_es, ffmpeg_log)

    stop_event = (getattr(args, "state", None) or {}).get("stop_event")

    # FRIM/3D-Blu-ray only ever accepts 8-bit yuv420p -- eye_filter()'s own downstream
    # ffmpeg leg already forces "-pix_fmt yuv420p" regardless of the source, but
    # PyAV's VideoFrame.to_ndarray() (what processor.py's raw_frame_sink branch uses
    # to get raw bytes) does not support 10-bit formats at all, so a 10-bit
    # --pix-fmt/--upgrade-pix-fmt setting would crash mid-job instead of just being
    # downconverted later. Copied, not mutated, so this never leaks back into the
    # caller's own args (e.g. the GUI's live widget-backed Namespace).
    args = copy(args)
    args.pix_fmt = "yuv420p"
    args.upgrade_pix_fmt = None

    _notify_stage(args, STAGE_DEPTH_STEREO)
    hdr_tmp_file = None
    processing_source = original_source_path
    if hdr:
        processing_source, hdr_tmp_file = _tonemap_hdr_to_sdr(original_source_path, args)

    swap_eyes = getattr(args, "mvc_swap_eyes", False)
    fill_mode = getattr(args, "mvc_fill_mode", None) or "fit"
    pipe = _DirectMvcPipe(ffmpeg, frim, layout, fps_frac, bitrate_mbps,
                          swap_eyes, base_es, dep_es, ffmpeg_log, stop_event, fill_mode)
    ok = False
    try:
        try:
            process_video_full(processing_source, output_path, args, depth_model, side_model,
                               raw_frame_sink=pipe.write)
        finally:
            if hdr_tmp_file and path.exists(hdr_tmp_file):
                try:
                    os.remove(hdr_tmp_file)
                except OSError:
                    pass

        # ADR-283 Progress Reporting follow-up: _notify_stage() now fires right as the real
        # encode-backlog wait begins, not after it -- it used to fire only once pipe.finish()
        # had already returned, so the (often lengthy, FRIM is CPU/software-only) backlog
        # drain used to run silently under the previous stage's already-100%-complete bar
        # with no stage transition and no pulse timer running at all, looking identical to a
        # hang. This is a pure display-ordering fix: no processing/output changes.
        _notify_stage(args, STAGE_CONVERT_MVC)
        encode_bar = _StageBar(args, _DIRECT_MVC_STAGE_PREFIX + "encoding the real 3D (MVC) video",
                               total_frames, "frames")
        try:
            pipe.finish(on_progress=encode_bar.set)
        finally:
            encode_bar.close(complete=(pipe.fr is not None and getattr(pipe.fr, "returncode", None) == 0))

        if stop_event is not None and stop_event.is_set():
            raise Cancelled()

        av_lines, notes = _plan_audio_subs(original_source_path, work_dir, ffmpeg, True, fps_text=fps_text,
                                           disc_legal=disc_legal,
                                           allow_lossless_eac3_on_disc=getattr(
                                               args, "mvc_allow_lossless_eac3_on_disc", False))
        state = getattr(args, "state", None)
        if state is not None and notes:
            state.setdefault("mvc_notes", []).extend(notes)

        if is_mkv_output:
            # Real, confirmed bug found this session (see sbs_to_mvc_cli.py's own long
            # comment above _AV_LINE_RE for the full real evidence): tsMuxeR's bare-MUXOPT
            # native .mkv writer -- what ADR-303 switched this branch to -- has a periodic
            # PTS sawtooth (every GOP runs ~2x speed then snaps back) that a real hardware
            # MVC player paces display off, showing as a stutter about once per second.
            # Confirmed, by direct ffprobe packet-PTS comparison against a known-clean real
            # reference file, that mkvmerge (via mvc_extract_cli.interleave_mvc(), which now
            # carries BOTH the ADR-290 dropped-last-AU fix and the ADR-292 type-24-delimiter
            # fix) produces pacing matching the reference as closely as tsMuxeR's own
            # `--blu-ray` disc-authoring MUXOPT does (also independently confirmed clean) --
            # unlike the bare writer ADR-303 moved to, which still sawtooths even when fed
            # the type-24-stripped stream directly, ruling that marker out as the cause.
            # This is the same mkvmerge+interleave_mvc() mechanism ADR-303 moved this branch
            # AWAY from after a real Zidoo hardware report of doubled/overlapping video --
            # that report used an EARLIER build of this function, before ADR-290 and ADR-292
            # were both in place for this exact call site; real hardware re-verification
            # (Zidoo/Vero5/SyLC) of THIS rebuilt version is still the decisive, not-yet-done
            # step (see sbs_to_mvc_cli.py's own comment for the full reasoning).
            mux_bar = _StageBar(args, _DIRECT_MVC_STAGE_PREFIX + "writing the final file", 1000, "pct")

            def _mux_progress(stage, done, total):
                mux_bar.set(int(min(1.0, done / total) * 1000) if total else 0)
            try:
                _mux_mkv_via_mkvmerge(base_es, dep_es, fps_text, av_lines, output_path, work_dir,
                                     stop_event=stop_event, progress_cb=_mux_progress,
                                     log_prefix="direct-mvc")
            finally:
                mux_bar.close(complete=path.exists(output_path))
            ok = True
            return output_path

        fwd = lambda p: p.replace(chr(92), "/")  # noqa: E731
        # ADR-289: --blu-ray builds a full BDMV/playlist/SSIF disc structure -- a bare
        # .m2ts skips that and just needs a bare clip.
        #
        # ADR-299: --new-audio-pes must be requested explicitly on the --blu-ray branch too,
        # not just on the bare-clip one -- "normally implied by --blu-ray" is confirmed FALSE
        # for this bundled tsMuxeR by direct real testing (see sbs_to_mvc_cli.py's own ADR-299
        # comment for the full evidence). Without it, audio silently gets the legacy 0xBD PES
        # stream id instead of the real Blu-ray-standard 0xFD, and MakeMKV drops the whole
        # audio track over just that one byte despite every other declaration being correct.
        #
        # ADR-300: --maxbitrate=48000 caps the disc's declared read rate at the real BD-ROM
        # drive spec (48 Mbit/s) -- see sbs_to_mvc_cli.py's own ADR-300 comment for the full
        # real, evidence-based testing. Added unconditionally, same as the other two
        # --blu-ray MUXOPT lines.
        #
        # Only .m2ts reaches here now using the bare MUXOPT (is_mkv_output returned above,
        # through mkvmerge instead -- see the real, confirmed PTS-sawtooth bug documented
        # just above). .m2ts was already the documented-superseded, not-recognized-as-3D-by-
        # real-hardware option (ADR-289/291) before that bug was even found, and mkvmerge
        # cannot replace tsMuxeR here (a bare .m2ts isn't a Matroska target), so it is kept
        # exactly as-is -- still carrying the same unfixed defect.
        if is_m2ts_output:
            print("[direct-mvc] note: a bare .m2ts clip is a legacy option -- not recognized "
                 "as 3D by real hardware (ADR-289/291) AND still has a known real-hardware "
                 "stutter defect (the exact bug .mkv output was just fixed for) -- use .mkv or "
                 "the .iso/BD-folder options instead unless a specific player asks for a bare "
                 "clip.", file=sys.stderr)
        muxopt = ("MUXOPT --blu-ray --new-audio-pes --auto-chapters=10 --maxbitrate=48000"
                  if disc_legal else "MUXOPT --new-audio-pes")
        meta = [muxopt,
                f"V_MPEG4/ISO/AVC, {fwd(base_es)}, fps={fps_text}, insertSEI, contSPS",
                f"V_MPEG4/ISO/MVC, {fwd(dep_es)}, fps={fps_text}, insertSEI, contSPS"] + av_lines
        meta_path = path.join(work_dir, "mux.meta")
        with open(meta_path, "w", encoding="utf-8", newline="\n") as f:  # no BOM: tsMuxeR rejects one
            f.write("\n".join(meta) + "\n")
        # ADR-291: folder output builds the real .iso into a temp path inside work_dir
        # first (tsMuxeR cannot build a real 3D BDMV folder directly -- see
        # iso_to_bd_folder()'s own docstring), then copies its contents out. The temp
        # iso is cleaned up along with the rest of work_dir below, unconditionally.
        mux_target = path.join(work_dir, "_bd_temp.iso") if is_folder_output else output_path
        print(f"[direct-mvc:tsmuxer] meta file ({meta_path}):\n" + "\n".join(f"  {line}" for line in meta),
              file=sys.stderr)
        log_subprocess_cmd("direct-mvc:tsmuxer", [tsmuxer, meta_path, mux_target])
        # Real streaming percent progress -- same regex/precedent already proven in
        # sbs_to_mvc_cli.py's own convert() ("mux" stage): tsMuxeR's own stdout prints
        # incrementing "NN.N%" progress lines while it writes the disc/file, not just a
        # single line at the end.
        mux_bar = _StageBar(args, _DIRECT_MVC_STAGE_PREFIX + "writing the final file", 1000, "pct")
        mux = subprocess.Popen([tsmuxer, meta_path, mux_target], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
        mux_tail = []
        try:
            for line in mux.stdout:
                mux_tail.append(line.rstrip())
                del mux_tail[:-12]
                m = re.search(r"(\d+(?:\.\d+)?)%", line)
                if m:
                    mux_bar.set(int(min(1.0, float(m.group(1)) / 100.0) * 1000))
                if stop_event is not None and stop_event.is_set():
                    mux.kill()
                    raise Cancelled()
            mux.wait()
        finally:
            mux_bar.close(complete=(mux.returncode == 0))
        if mux.returncode != 0:
            raise RuntimeError("tsMuxeR failed:\n" + "\n".join(mux_tail))
        if is_folder_output:
            folder_bar = _StageBar(args, _DIRECT_MVC_STAGE_PREFIX + "copying the disc structure to the folder",
                                   1, "pct")
            iso_to_bd_folder(mux_target, output_path, stop_event=stop_event)
            folder_bar.close(complete=True)
        ok = True
        return output_path
    finally:
        pipe.kill()
        if not ok:
            if is_folder_output:
                shutil.rmtree(output_path, ignore_errors=True)
            else:
                try:
                    if path.exists(output_path):
                        os.remove(output_path)
                except OSError:
                    pass
        shutil.rmtree(work_dir, ignore_errors=True)


def _self_test_fps_refusal():
    from unittest import mock
    with mock.patch.object(sys.modules[__name__], "probe_video",
                           return_value=(1920, 1080, "25/1", 100.0, False)), \
         mock.patch("subprocess.Popen") as popen:
        raised = False
        try:
            convert_direct("in.mp4", "out.iso", _fake_args(), None, None)
        except ValueError as e:
            raised = True
            assert "25.000" in str(e) or "25" in str(e), str(e)
    assert raised, "an illegal source frame rate must be refused"
    assert not popen.called, "must refuse before spawning any subprocess"
    print("_self_test_fps_refusal: PASS")


def _self_test_hdr_without_sdr_refusal():
    from unittest import mock
    with mock.patch.object(sys.modules[__name__], "probe_video",
                           return_value=(1920, 1080, "24/1", 100.0, True)), \
         mock.patch("subprocess.Popen") as popen:
        args = _fake_args()
        args.hdr_to_sdr = False
        raised = False
        try:
            convert_direct("in.mp4", "out.iso", args, None, None)
        except RuntimeError as e:
            raised = True
            assert "HDR" in str(e), str(e)
    assert raised, "an HDR source without hdr_to_sdr must be refused"
    assert not popen.called, "must refuse before spawning any subprocess"
    print("_self_test_hdr_without_sdr_refusal: PASS")


def _self_test_incompatible_layout_refusal():
    from unittest import mock
    args = _fake_args()
    args.vr180 = True
    raised = False
    with mock.patch.object(sys.modules[__name__], "probe_video",
                           return_value=(1920, 1080, "24/1", 100.0, False)):
        try:
            convert_direct("in.mp4", "out.iso", args, None, None)
        except ValueError as e:
            raised = True
            assert "Stereo Format" in str(e), str(e)
    assert raised, "a Stereo Format MVC cannot use must be refused"
    print("_self_test_incompatible_layout_refusal: PASS")


class _FakeCompletedProc:
    def __init__(self, returncode=0, stdout=b""):
        self.returncode = returncode
        self.stdout = stdout


def _self_test_mocked_end_to_end():
    """No real FRIM/ffmpeg/tsMuxeR: subprocess.Popen/run are mocked, and
    process_video_full() is mocked to call the real raw_frame_sink it was handed
    with a few synthetic frames -- exactly like the real pipeline would, just
    without a real depth/stereo model. Proves: no SBS/raw intermediate file is ever
    written, and the downstream ffmpeg leg receives every frame's bytes, in order."""
    import tempfile
    from unittest import mock

    # colorspace/color_primaries/color_trc/color_range: real BT.709 tv-range values
    # (1, 1, 1, 1), matching what output_reformatter actually produces for SDR yuv420p.
    frames = [
        (bytes([10]) * 100, 64, 32, "yuv420p", 1, 1, 1, 1),
        (bytes([20]) * 100, 64, 32, "yuv420p", 1, 1, 1, 1),
        (bytes([30]) * 100, 64, 32, "yuv420p", 1, 1, 1, 1),
    ]
    written_to_ffmpeg_stdin = []

    class _FakeStdin:
        def write(self, data):
            written_to_ffmpeg_stdin.append(data)

        def close(self):
            pass

    class _FakeStdout:
        def close(self):
            pass

        def read(self, n):
            return b""

    class _FakeProc:
        def __init__(self, args, stdout=None, **kwargs):
            self.args = args
            self.stdin = _FakeStdin()
            self.stdout = stdout if stdout is not None else _FakeStdout()
            self.returncode = 0
            self._waited = False

        def poll(self):
            return 0

        def wait(self, timeout=None):
            self._waited = True
            return 0

        def kill(self):
            pass

    popen_cmds = []

    def fake_popen(cmd, **kwargs):
        popen_cmds.append(cmd)
        if "-o:mvc" in cmd:
            base_es, dep_es = cmd[cmd.index("-o:mvc") + 1], cmd[cmd.index("-o:mvc") + 2]
            with open(base_es, "wb") as f:
                f.write(b"\x00\x00\x00\x01fake-base")
            with open(dep_es, "wb") as f:
                f.write(b"\x00\x00\x00\x01fake-dep")
            return _FakeProc(cmd)
        if cmd and cmd[0] == "tsmuxer.exe":
            # text=True is passed for this one (see convert_direct()'s own streaming mux
            # loop) -- "for line in mux.stdout:" just needs a plain iterable of strings,
            # no real lines needed here (progress is covered by its own dedicated test).
            return _FakeProc(cmd, stdout=[])
        return _FakeProc(cmd)

    def fake_process_video_full(input_filename, output_path, args, depth_model, side_model, raw_frame_sink=None):
        for frame_args in frames:
            raw_frame_sink(*frame_args)
        return output_path

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = path.join(tmp_dir, "out.iso")
        with mock.patch.object(sys.modules[__name__], "probe_video",
                               return_value=(1920, 1080, "24/1", 2.0, False)), \
             mock.patch.object(sys.modules[__name__], "find_frim", return_value="frim.exe"), \
             mock.patch.object(sys.modules[__name__], "_get_ffmpeg_bin", return_value="ffmpeg.exe"), \
             mock.patch.object(sys.modules[__name__], "_find_tsmuxer", return_value="tsmuxer.exe"), \
             mock.patch.object(sys.modules[__name__], "_plan_audio_subs", return_value=([], [])), \
             mock.patch.object(sys.modules[__name__], "process_video_full", side_effect=fake_process_video_full), \
             mock.patch("subprocess.Popen", side_effect=fake_popen), \
             mock.patch("subprocess.run", return_value=_FakeCompletedProc(returncode=0)):
            result = convert_direct("in.mp4", output_path, _fake_args(), None, None)

        assert result == output_path
        assert len(written_to_ffmpeg_stdin) == len(frames), \
            f"expected {len(frames)} writes to the ffmpeg pipe, got {len(written_to_ffmpeg_stdin)}"
        assert written_to_ffmpeg_stdin == [f[0] for f in frames], "frames must reach the pipe in order"
        # Real live-test regression check: raw video carries no embedded color
        # metadata, so the ffmpeg leg MUST be told the real colorspace/primaries/
        # trc/range explicitly, or the output comes out visibly wrong-colored
        # (confirmed side-by-side against the normal two-stage path).
        ff_cmd = popen_cmds[0]
        for flag, expected in (("-colorspace", "1"), ("-color_primaries", "1"),
                               ("-color_trc", "1"), ("-color_range", "1")):
            assert flag in ff_cmd, f"ffmpeg command missing {flag}: {ff_cmd}"
            assert ff_cmd[ff_cmd.index(flag) + 1] == expected, ff_cmd
        # ADR-313 regression guard: real, decker-confirmed MVC stutter fix -- a forced
        # OUTPUT "-r" on this rawvideo target made ffmpeg run its own internal CFR filter,
        # which genuinely dup/dropped frames in a periodic pattern. It must be
        # "-fps_mode passthrough" instead (FRIM's own "-f fps_frac" already declares the
        # authoritative output rate independently). The INPUT "-r" (before "-i", "-",
        # telling ffmpeg how to interpret this leg's own headerless incoming raw pipe) is a
        # different, legitimate flag and must still be exactly the only "-r" left.
        input_i_index = ff_cmd.index("-i")
        assert ff_cmd.count("-r") == 1, f"expected exactly the INPUT -r to remain: {ff_cmd}"
        assert ff_cmd.index("-r") < input_i_index, f"the surviving -r must be the INPUT flag: {ff_cmd}"
        assert "-fps_mode" in ff_cmd and ff_cmd[ff_cmd.index("-fps_mode") + 1] == "passthrough", ff_cmd
        assert ff_cmd.index("-fps_mode") > input_i_index, ff_cmd
        # The work dir (and any stray raw/SBS file in it) is cleaned up on success --
        # the real point of this whole feature is that no such intermediate ever lands
        # anywhere durable.
        work_dir = path.join(tmp_dir, "_direct_mvc_work_out")
        assert not path.exists(work_dir), "the work directory must be cleaned up after a successful run"

    print("_self_test_mocked_end_to_end: PASS")


class _FakeTqdmBar:
    """Mimics nunif.gui.common.TQDMGUI's interface -- what args.state["tqdm_fn"] is
    really called as -- recording every (desc, total, unit-prefix) a bar is opened
    with plus the full sequence of absolute positions .update() deltas resolve to, so
    a test can assert real incremental behavior the same way the GUI's own on_tqdm()
    would see it (see iw3/gui.py's on_tqdm())."""
    def __init__(self, log, total, desc):
        self.desc = desc
        self.total = total
        self.pos = 0
        self.closed = False
        self.positions = []
        log.append(self)

    def update(self, n=1):
        self.pos += n
        self.positions.append(self.pos)

    def close(self):
        self.closed = True


def _self_test_encode_progress_matches_real_frim_output():
    """_DirectMvcPipe._read() parses FRIM's real "Frame number: N" stderr lines (same
    regex sbs_to_mvc_cli.py's own proven "encode" stage uses -- FRIM reads from stdin
    in both pipelines, so its own output format is identical) into pipe.frim_frame,
    and finish(on_progress=...) reports that value. Deterministic (joins the reader
    thread instead of racing it on a timer) -- proves correctness of the new parsing/
    wiring; real wall-clock incrementality during a real encode backlog drain was
    confirmed separately against the real bundled ffmpeg/FRIM binaries (not something
    a synthetic mock can honestly claim to reproduce -- same standard sbs_to_mvc_cli.py
    itself is held to, see ADR-221)."""
    class _FakeStdin:
        def write(self, data):
            pass

        def close(self):
            pass

    class _FakeFrimStdout:
        def __init__(self, frame_numbers):
            self._buf = b"".join(f"Frame number: {n}\r\n".encode() for n in frame_numbers)
            self._pos = 0

        def close(self):
            pass

        def read(self, n):
            chunk = self._buf[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

    class _FakeProc:
        def __init__(self, stdout=None):
            self.stdin = _FakeStdin()
            self.stdout = stdout if stdout is not None else _FakeFrimStdout([])
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_es, dep_es = path.join(tmp_dir, "base.264"), path.join(tmp_dir, "dep.264")
        for p in (base_es, dep_es):
            with open(p, "wb") as f:
                f.write(b"\x00\x00\x00\x01fake")

        procs = [_FakeProc(), _FakeProc(_FakeFrimStdout([1, 2, 3, 10]))]  # ff, then fr

        def fake_popen(cmd, **kwargs):
            return procs.pop(0)

        pipe = _DirectMvcPipe("ffmpeg.exe", "frim.exe", "full_sbs", "24/1", 20.0, False,
                              base_es, dep_es, path.join(tmp_dir, "ffmpeg.log"), None, "fit")
        with mock.patch("subprocess.Popen", side_effect=fake_popen):
            pipe.write(b"\x00" * 100, 64, 32, "yuv420p", 1, 1, 1, 1)

        seen = []
        pipe.finish(on_progress=seen.append)

        assert pipe.frim_frame == 10, f"expected the last real FRIM frame count parsed, got {pipe.frim_frame}"
        assert seen[-1] == 10, f"finish()'s on_progress must report the real final frame count, got {seen}"
        assert seen == sorted(seen), f"on_progress values must never go backwards, got {seen}"

    print("_self_test_encode_progress_matches_real_frim_output: PASS")


def _self_test_mux_progress_is_incremental():
    """convert_direct()'s tsMuxeR streaming mux loop (.iso/.m2ts/BD-folder output --
    .mkv now goes through mkvmerge instead, see
    _self_test_mkv_output_uses_mkvmerge_not_tsmuxer() below for that path's own
    progress test) reports real, incrementing percent values parsed from tsMuxeR's
    own stdout as they arrive -- not one jump at the very end -- same regex
    sbs_to_mvc_cli.py's own proven "mux" stage already uses. Deterministic: the mux
    loop is synchronous (no background thread involved), so feeding it several
    real-shaped "NN.N%" lines in order needs no timing games."""
    import tempfile
    from unittest import mock

    frames = [(bytes([10]) * 100, 64, 32, "yuv420p", 1, 1, 1, 1)]
    bars = []

    def fake_tqdm_fn(**kwargs):
        return _FakeTqdmBar(bars, kwargs["total"], kwargs.get("desc", ""))

    class _FakeStdin:
        def write(self, data):
            pass

        def close(self):
            pass

    class _FakeStdout:
        def close(self):
            pass

        def read(self, n):
            return b""

    class _FakeProc:
        def __init__(self, cmd, stdout=None):
            self.cmd = cmd
            self.stdin = _FakeStdin()
            self.stdout = stdout if stdout is not None else _FakeStdout()
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_popen(cmd, **kwargs):
        if "-o:mvc" in cmd:
            base_es, dep_es = cmd[cmd.index("-o:mvc") + 1], cmd[cmd.index("-o:mvc") + 2]
            with open(base_es, "wb") as f:
                f.write(b"\x00\x00\x00\x01fake-base")
            with open(dep_es, "wb") as f:
                f.write(b"\x00\x00\x00\x01fake-dep")
            return _FakeProc(cmd)
        if cmd and cmd[0] == "tsmuxer.exe":
            # Real-shaped tsMuxeR progress output (same "NN.N%" convention
            # sbs_to_mvc_cli.py's own proven mux-progress regex already parses),
            # arriving as several distinct lines, not one before/after pair.
            return _FakeProc(cmd, stdout=["Muxing: 0.0%\r\n", "Muxing: 12.5%\r\n",
                                          "Muxing: 61.0%\r\n", "Muxing: 100.0%\r\n"])
        return _FakeProc(cmd)

    def fake_process_video_full(input_filename, output_path, args, depth_model, side_model, raw_frame_sink=None):
        for frame_args in frames:
            raw_frame_sink(*frame_args)
        return output_path

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = path.join(tmp_dir, "out.iso")
        args = _fake_args()
        args.state = {"tqdm_fn": fake_tqdm_fn}
        with mock.patch.object(sys.modules[__name__], "probe_video",
                               return_value=(1920, 1080, "24/1", 2.0, False)), \
             mock.patch.object(sys.modules[__name__], "find_frim", return_value="frim.exe"), \
             mock.patch.object(sys.modules[__name__], "_get_ffmpeg_bin", return_value="ffmpeg.exe"), \
             mock.patch.object(sys.modules[__name__], "_find_tsmuxer", return_value="tsmuxer.exe"), \
             mock.patch.object(sys.modules[__name__], "_plan_audio_subs", return_value=([], [])), \
             mock.patch.object(sys.modules[__name__], "process_video_full", side_effect=fake_process_video_full), \
             mock.patch("subprocess.Popen", side_effect=fake_popen):
            convert_direct("in.mp4", output_path, args, None, None)

        mux_bars = [b for b in bars if "writing the final file" in b.desc]
        assert len(mux_bars) == 1, f"expected exactly one mux progress bar, got {len(mux_bars)}"
        positions = mux_bars[0].positions
        assert len(positions) >= 3, f"expected several incremental mux updates, got {positions}"
        assert positions == sorted(positions), f"mux progress must never go backwards, got {positions}"
        assert len(set(positions)) > 1, f"mux progress must show more than one distinct value, got {positions}"
        assert positions[-1] == mux_bars[0].total, f"mux progress must finish at 100%, got {positions}"
        assert mux_bars[0].closed, "the mux progress bar must be closed once the mux finishes"

    print("_self_test_mux_progress_is_incremental: PASS")


def _self_test_mkv_output_uses_mkvmerge_not_tsmuxer():
    """Real, confirmed bug fixed this session (see sbs_to_mvc_cli.py's own long comment
    above _AV_LINE_RE for the full real evidence): tsMuxeR's bare-MUXOPT native .mkv
    writer has a periodic PTS sawtooth that stutters real hardware playback about once
    per second. `.mkv` output must now go through mkvmerge (via
    mvc_extract_cli.interleave_mvc()) instead -- this asserts that code path is
    actually taken (tsMuxeR is never invoked at all for `.mkv`) and that its own mux
    progress still reports real, incrementing values, same contract as the tsMuxeR
    path _self_test_mux_progress_is_incremental() above covers for .iso/.m2ts/folder."""
    import tempfile
    from unittest import mock

    frames = [(bytes([10]) * 100, 64, 32, "yuv420p", 1, 1, 1, 1)]
    bars = []

    def fake_tqdm_fn(**kwargs):
        return _FakeTqdmBar(bars, kwargs["total"], kwargs.get("desc", ""))

    class _FakeStdin:
        def write(self, data):
            pass

        def close(self):
            pass

    class _FakeStdout:
        def close(self):
            pass

        def read(self, n):
            return b""

    class _FakeProc:
        def __init__(self, cmd, stdout=None):
            self.cmd = cmd
            self.stdin = _FakeStdin()
            self.stdout = stdout if stdout is not None else _FakeStdout()
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    popen_cmds = []

    def fake_popen(cmd, **kwargs):
        popen_cmds.append(cmd)
        if "-o:mvc" in cmd:
            base_es, dep_es = cmd[cmd.index("-o:mvc") + 1], cmd[cmd.index("-o:mvc") + 2]
            # Real AU-boundary-shaped content (type-9 AUD for the base view, type-24
            # FRIM-style delimiter for the dependent view) so interleave_mvc() -- run
            # for real here, not mocked -- has something real to recombine, same as
            # its own dedicated self-test in mvc_extract_cli.py.
            with open(base_es, "wb") as f:
                f.write(b"\x00\x00\x00\x01\x09\x00\x00\x01\x65fake-base-slice")
            with open(dep_es, "wb") as f:
                f.write(b"\x00\x00\x00\x01\x18\x00\x00\x01\x14fake-dep-slice")
            return _FakeProc(cmd)
        if cmd and cmd[0] == "tsmuxer.exe":
            raise AssertionError(f"tsMuxeR must never be invoked for .mkv output: {cmd}")
        if cmd and cmd[0] == "mkvmerge.exe":
            return _FakeProc(cmd, stdout=["Progress: 0%\r\n", "Progress: 40%\r\n",
                                          "Progress: 90%\r\n", "Progress: 100%\r\n"])
        return _FakeProc(cmd)

    def fake_process_video_full(input_filename, output_path, args, depth_model, side_model, raw_frame_sink=None):
        for frame_args in frames:
            raw_frame_sink(*frame_args)
        return output_path

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = path.join(tmp_dir, "out.mkv")
        args = _fake_args()
        args.state = {"tqdm_fn": fake_tqdm_fn}
        with mock.patch.object(sys.modules[__name__], "probe_video",
                               return_value=(1920, 1080, "24/1", 2.0, False)), \
             mock.patch.object(sys.modules[__name__], "find_frim", return_value="frim.exe"), \
             mock.patch.object(sys.modules[__name__], "_get_ffmpeg_bin", return_value="ffmpeg.exe"), \
             mock.patch.object(sys.modules[__name__], "_find_tsmuxer", return_value="tsmuxer.exe"), \
             mock.patch.object(sys.modules[__name__], "_plan_audio_subs", return_value=([], [])), \
             mock.patch.object(sys.modules[__name__], "process_video_full", side_effect=fake_process_video_full), \
             mock.patch("iw3.sbs_to_mvc_cli._find_mkvmerge", return_value="mkvmerge.exe"), \
             mock.patch("iw3.sbs_to_mvc_cli._find_mkvpropedit", return_value=None), \
             mock.patch("subprocess.Popen", side_effect=fake_popen):
            result = convert_direct("in.mp4", output_path, args, None, None)

        assert result == output_path
        assert not any(cmd and cmd[0] == "tsmuxer.exe" for cmd in popen_cmds), \
            "tsMuxeR must never be invoked for .mkv output"
        assert any(cmd and cmd[0] == "mkvmerge.exe" for cmd in popen_cmds), \
            "mkvmerge must be invoked for .mkv output"

        mux_bars = [b for b in bars if "writing the final file" in b.desc]
        assert len(mux_bars) == 1, f"expected exactly one mux progress bar, got {len(mux_bars)}"
        positions = mux_bars[0].positions
        assert len(positions) >= 3, f"expected several incremental mux updates, got {positions}"
        assert positions == sorted(positions), f"mux progress must never go backwards, got {positions}"
        assert positions[-1] == mux_bars[0].total, f"mux progress must finish at 100%, got {positions}"
        assert mux_bars[0].closed, "the mux progress bar must be closed once the mux finishes"

    print("_self_test_mkv_output_uses_mkvmerge_not_tsmuxer: PASS")


def _fake_args():
    class _Args:
        pass
    args = _Args()
    args.hdr_to_sdr = True
    args.half_sbs = args.tb = args.half_tb = False
    args.vr180 = args.cross_eyed = args.rgbd = args.half_rgbd = False
    args.anaglyph = args.export = args.export_disparity = args.debug_depth = False
    args.mvc_bitrate = 20.0
    args.pix_fmt = "yuv420p"
    args.upgrade_pix_fmt = None
    args.state = {}
    return args


def main():
    _self_test_fps_refusal()
    _self_test_hdr_without_sdr_refusal()
    _self_test_incompatible_layout_refusal()
    _self_test_mocked_end_to_end()
    _self_test_encode_progress_matches_real_frim_output()
    _self_test_mux_progress_is_incremental()
    _self_test_mkv_output_uses_mkvmerge_not_tsmuxer()
    print("ALL PASS")


if __name__ == "__main__" and "--self-test" in sys.argv:
    main()
