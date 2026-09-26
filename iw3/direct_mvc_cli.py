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

from .mvc_extract_cli import Cancelled, interleave_mvc
from .sbs_to_mvc_cli import bd_frame_rate, eye_filter, find_frim, probe_video, _plan_audio_subs, _extract_all_av_for_mkv
from .utils import (
    _find_tsmuxer, _find_mkvmerge, _get_ffmpeg_bin, _tonemap_hdr_to_sdr, _notify_stage,
    process_video_full, STAGE_CONVERT_MVC, STAGE_DEPTH_STEREO,
)

_MVC_LAYOUT_INCOMPATIBLE_FLAGS = (
    "vr180", "cross_eyed", "rgbd", "half_rgbd", "anaglyph", "export", "export_disparity", "debug_depth",
)


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
                 base_es, dep_es, ffmpeg_log_path, stop_event):
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
        self.ff = None
        self.fr = None
        self._ff_log = None
        self._tail = []
        self._reader = None
        self.frame_count = 0

    def write(self, raw_bytes, width, height, pix_fmt, colorspace, color_primaries, color_trc, color_range):
        if self._stop_event is not None and self._stop_event.is_set():
            raise Cancelled()
        if self.ff is None:
            self._start(width, height, pix_fmt, colorspace, color_primaries, color_trc, color_range)
        try:
            self.ff.stdin.write(raw_bytes)
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(
                f"the MVC encode pipe closed unexpectedly while writing frame {self.frame_count + 1} "
                f"(FRIMEncode likely crashed or refused): {e}") from e
        self.frame_count += 1

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
        vf = eye_filter(self._layout, width, height, None)
        ff_cmd = [self._ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                  "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}", "-r", self._fps_frac,
                  "-colorspace", str(colorspace), "-color_primaries", str(color_primaries),
                  "-color_trc", str(color_trc), "-color_range", str(color_range),
                  "-i", "-", "-an", "-sn", "-vf", vf, "-pix_fmt", "yuv420p",
                  "-r", self._fps_frac, "-f", "rawvideo", "-"]
        target = int(self._bitrate_mbps * 1000)
        frim_cmd = [self._frim_bin, "-i", "-", "-o:mvc", self._base_es, self._dep_es, "-viewoutput",
                    "-sbs", "2", "-w", "1920", "-h", "1080", "-f", self._fps_frac,
                    "-profile", "high", "-level", "4.1",
                    "-vbr", str(target), str(int(target * 1.25)), "-sw"]
        if self._swap_eyes:
            frim_cmd.append("-swaplr")

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

        self._reader = threading.Thread(target=_read, daemon=True)
        self._reader.start()

    def finish(self):
        """Closes ffmpeg's stdin -- the real EOF signal down the pipe, same idiom
        sbs_to_mvc_cli.convert() uses (its own `ff.stdout.close()`) to end its side of
        an equivalent pipe cleanly -- then waits for both processes and raises with
        the same real diagnostic shape convert() already uses on failure."""
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
        if self._reader is not None:
            self._reader.join()
        self.ff.wait()
        self._ff_log.close()

        if self.fr.returncode != 0 or not (path.exists(self._base_es) and path.getsize(self._base_es) > 0
                                            and path.exists(self._dep_es) and path.getsize(self._dep_es) > 0):
            with open(self._ffmpeg_log_path, "rb") as f:
                ff_msg = f.read()[-600:].decode(errors="replace")
            frim_msg = "\n".join(self._tail[-8:]) if self._tail else \
                "(FRIMEncode produced no output at all before exiting)"
            raise RuntimeError(
                f"the MVC encode failed (FRIMEncode exit code {self.fr.returncode}):\n{frim_msg}"
                + (f"\nffmpeg (likely just a downstream symptom of FRIM's pipe closing, not "
                   f"ffmpeg's own problem): {ff_msg}" if ff_msg.strip() else ""))

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
    `output_path` (.iso via tsMuxeR, or .mkv directly via mkvmerge, same choice
    sbs_to_mvc_cli.convert() offers). Returns output_path on success; raises
    RuntimeError/ValueError (with a real, specific reason) on any failure or refusal,
    or mvc_extract_cli.Cancelled if args.state's stop_event fired mid-job."""
    width, height, rate, duration, hdr = probe_video(original_source_path)
    fps_text, fps_frac = bd_frame_rate(rate)  # raises ValueError naming the actual fps if illegal

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
    is_mkv_output = path.splitext(output_path)[1].lower() == ".mkv"
    if not is_mkv_output and path.splitext(output_path)[1].lower() != ".iso":
        raise ValueError("the output must end in .iso or .mkv")

    frim = find_frim()
    if frim is None:
        raise RuntimeError("FRIMEncode not found -- run `python -m iw3.install_mvc_tools`")
    ffmpeg = _get_ffmpeg_bin()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    tsmuxer = _find_tsmuxer()
    if tsmuxer is None and not is_mkv_output:
        raise RuntimeError("tsMuxeR not found -- run `python -m iw3.install_mvc_tools`")

    bitrate_mbps = float(getattr(args, "mvc_bitrate", 20.0) or 20.0)
    if not 2 <= bitrate_mbps <= 40:
        raise ValueError("bitrate must be between 2 and 40 Mbps (3D Blu-ray allows about 40 combined)")

    out_dir = path.dirname(path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = path.splitext(path.basename(output_path))[0]
    work_dir = path.join(out_dir, f"_direct_mvc_work_{stem}")
    os.makedirs(work_dir, exist_ok=True)

    base_es, dep_es = path.join(work_dir, "base.264"), path.join(work_dir, "dep.264")
    ffmpeg_log = path.join(work_dir, "ffmpeg.log")
    combined_es = path.join(work_dir, "combined_mvc.264")  # only written for .mkv output

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

    pipe = _DirectMvcPipe(ffmpeg, frim, layout, fps_frac, bitrate_mbps,
                          getattr(args, "mvc_swap_eyes", False), base_es, dep_es, ffmpeg_log, stop_event)
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

        pipe.finish()

        if stop_event is not None and stop_event.is_set():
            raise Cancelled()

        _notify_stage(args, STAGE_CONVERT_MVC)
        if is_mkv_output:
            # ADR-245's own technique, reused unchanged: FRIM's separated base/dependent
            # elementary streams are the exact shape interleave_mvc() already expects.
            interleave_mvc(base_es, dep_es, combined_es)
            av_files = _extract_all_av_for_mkv(original_source_path, work_dir, ffmpeg)
            mkvmerge_bin = _find_mkvmerge()
            if mkvmerge_bin is None:
                raise RuntimeError("mkvmerge not found -- it ships in this project's own mkvtoolnix/ folder")
            mux_cmd = [mkvmerge_bin, "-o", output_path, "--default-duration", f"0:{fps_frac}fps",
                      combined_es] + av_files
            result = subprocess.run(mux_cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError("mkvmerge failed:\n" + result.stdout.decode(errors="replace")[-1200:])
        else:
            av_lines, notes = _plan_audio_subs(original_source_path, work_dir, ffmpeg, True, fps_text=fps_text)
            state = getattr(args, "state", None)
            if state is not None and notes:
                state.setdefault("mvc_notes", []).extend(notes)
            fwd = lambda p: p.replace(chr(92), "/")  # noqa: E731
            meta = [f"MUXOPT --blu-ray --auto-chapters=10",
                    f"V_MPEG4/ISO/AVC, {fwd(base_es)}, fps={fps_text}, insertSEI, contSPS",
                    f"V_MPEG4/ISO/MVC, {fwd(dep_es)}, fps={fps_text}, insertSEI, contSPS"] + av_lines
            meta_path = path.join(work_dir, "mux.meta")
            with open(meta_path, "w", encoding="utf-8", newline="\n") as f:  # no BOM: tsMuxeR rejects one
                f.write("\n".join(meta) + "\n")
            result = subprocess.run([tsmuxer, meta_path, output_path], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError("tsMuxeR failed:\n" + result.stdout.decode(errors="replace")[-1200:])
        ok = True
        return output_path
    finally:
        pipe.kill()
        if not ok:
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
        def __init__(self, args, **kwargs):
            self.args = args
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout()
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
        proc = _FakeProc(cmd)
        if "-o:mvc" in cmd:
            base_es, dep_es = cmd[cmd.index("-o:mvc") + 1], cmd[cmd.index("-o:mvc") + 2]
            with open(base_es, "wb") as f:
                f.write(b"\x00\x00\x00\x01fake-base")
            with open(dep_es, "wb") as f:
                f.write(b"\x00\x00\x00\x01fake-dep")
        return proc

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
        # The work dir (and any stray raw/SBS file in it) is cleaned up on success --
        # the real point of this whole feature is that no such intermediate ever lands
        # anywhere durable.
        work_dir = path.join(tmp_dir, "_direct_mvc_work_out")
        assert not path.exists(work_dir), "the work directory must be cleaned up after a successful run"

    print("_self_test_mocked_end_to_end: PASS")


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
    print("ALL PASS")


if __name__ == "__main__" and "--self-test" in sys.argv:
    main()
