"""python -m iw3.iso_to_mvc_makemkv_cli -- rips a real 3D Blu-ray .iso (built by this
project's own sbs_to_mvc_cli.py/direct_mvc_cli.py "ISO" or "BD Folder" output, or any
other real 3D Blu-ray image) into a real .mkv using MakeMKV's own command-line tool
(makemkvcon), instead of this project hand-building the MVC container structure itself.

Why this exists (ADR-317 follow-up, Steve's own suggestion, decker-approved): a real bug
was found and fixed in this project's OWN direct-to-MVC-.mkv code path (a missing mvcC
codec-metadata box, see mvc_codec_private.py/ADR-317) -- but Steve tested that fix on his
real Samsung 3D TV and real-hardware auto-detection still did not work, so decker decided
to stop further investment in that path. This project's .iso/BD-folder output IS confirmed
reliable on real hardware (Steve: "It's working now with an iso output"), and MakeMKV is a
real, mature, widely-used commercial disc-ripping tool that correctly produces real
MVC-structured .mkv files from a real 3D Blu-ray ISO -- it is, in fact, the SAME tool that
produced the known-working reference file used to figure out what the mvcC box should look
like in the first place. So instead of fixing our own MVC muxing further, this module
chains a third, separate, real step after this project's own .iso output: hand the ISO to
MakeMKV's own CLI and let IT build the real .mkv.

MakeMKV is NEVER bundled with this project (unlike ffmpeg/mkvmerge/tsMuxeR/FRIM) -- it is
always a separate, real, user-owned install with its own real license (a permanent paid
key or an officially-rotated free beta key). _find_makemkvcon() looks for a real existing
install; this module never tries to install or license MakeMKV itself.

Real makemkvcon CLI shape, confirmed against the actual bundled v1.18.4 binary on this
machine (not guessed):
  makemkvcon -r --cache=<MB> info iso:<path>             -- scan titles (robot mode)
  makemkvcon -r --cache=<MB> --progress=-same mkv iso:<path> <title_index> <out_dir>
Robot mode (-r/--robot) prints quoted, backslash-escaped, line-based records to stdout:
  MSG:code,flags,count,"message",...      -- general messages/errors
  TCOUNT:count                            -- number of titles found
  TINFO:title_id,attr_code,...,"value"    -- per-title attributes (attr 9 = duration,
                                              "H:MM:SS" or "HH:MM:SS")
  PRGC:code,id,"name" / PRGT:code,id,"name" -- current/total operation names
  PRGV:current,total,max                  -- progress bar values (both current and
                                              total are already scaled 0..max)
A real 3D Blu-ray ISO usually has one long main-feature title plus several short
extras/trailers -- this module scans with `info` first and rips ONLY the title with the
longest real duration (not `all`), so the result is always the movie, never a guess among
several output files.
"""
import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import threading
from os import path

# Same reasoning as sbs_to_mvc_cli.py's own top-of-file import: this module runs as its own
# `python -m iw3.iso_to_mvc_makemkv_cli` process (invoked by the GUI's standalone-tool Run
# button), so it needs this patch applied itself -- iw3/gui.py importing it only covers
# subprocess calls made from the GUI's own process, not this separate one.
import nunif.gui.subprocess_patch  # noqa

from .mvc_extract_cli import Cancelled
from .utils import log_subprocess_cmd

# Confirmed real registry location MakeMKV's own Windows installer registers its CLI exe
# under (the standard "App Paths" mechanism many Windows installers use) -- verified live
# on a real machine with MakeMKV 1.18.4 installed: the default value of
# HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\makemkvcon.exe pointed straight
# at "C:\Program Files (x86)\MakeMKV\MakeMKVcon.exe". Checked as a more robust fallback than
# a hardcoded Program Files path alone (same spirit as _find_mkvmerge()/_find_tsmuxer() in
# utils.py, except MakeMKV is never bundled -- there is no project-relative candidate here).
_APP_PATHS_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
_MAKEMKVCON_NAMES = ("makemkvcon64.exe", "makemkvcon64", "makemkvcon.exe", "makemkvcon")
_PROGRAM_DIRS = (r"C:\Program Files (x86)\MakeMKV", r"C:\Program Files\MakeMKV")

# Real, confirmed-verbatim substrings MakeMKV's own robot-mode MSG text uses when its
# trial/license is the real problem (not guessed -- "Evaluation period has expired,
# shareware functionality unavailable" and "...Please purchase an activation key if you've
# found this application useful" are MakeMKV's own real strings; "key is not valid"/"key
# not found" cover the separate real registration-key-rejected case). Matched
# case-insensitively against every MSG line's own message text so a real license problem
# surfaces as an honest, specific error instead of a cryptic rip failure.
_LICENSE_PROBLEM_SUBSTRINGS = (
    "evaluation period", "shareware functionality", "purchase an activation key",
    "key is not valid", "key not found", "registration key",
)


def _find_makemkvcon():
    """Locates a real, separately-installed MakeMKV CLI binary. Unlike every other
    external-tool finder in this project (_find_tsmuxer/_find_mkvmerge/_find_ffmpeg_bin in
    utils.py), MakeMKV is never bundled -- there is no project-relative candidate to fall
    back to, so this only ever looks at real system locations: PATH, the registry's own
    "App Paths" record of where MakeMKV's installer put it, then the two real common
    Program Files locations. Returns None (never raises) if no real install is found --
    callers decide how to surface that."""
    for name in _MAKEMKVCON_NAMES:
        found = shutil.which(name)
        if found:
            return found
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for name in ("makemkvcon64.exe", "makemkvcon.exe"):
                try:
                    with winreg.OpenKey(hive, f"{_APP_PATHS_KEY}\\{name}") as key:
                        value, _ = winreg.QueryValueEx(key, None)
                        if value and path.exists(value):
                            return value
                except OSError:
                    continue
    except ImportError:
        pass  # not running on Windows
    for prog_dir in _PROGRAM_DIRS:
        for name in ("makemkvcon64.exe", "makemkvcon.exe"):
            candidate = path.join(prog_dir, name)
            if path.exists(candidate):
                return candidate
    return None


def _split_robot_fields(rest):
    """Splits one robot-mode line's own fields (everything after "TAG:") on commas outside
    quotes, undoing MakeMKV's own backslash-escaping of quotes/control characters inside a
    quoted field -- confirmed live against the real bundled binary: a quoted field
    containing a Windows path comes back with every backslash doubled (e.g.
    "C:\\\\Users\\\\decker\\\\..."), exactly matching MakeMKV's own documented "all strings are
    quoted, all control characters and quotes are backslash-escaped" rule. csv.reader
    already implements exactly this escaping convention (escapechar, doublequote=False), so
    this reuses it rather than hand-rolling a quote-aware comma splitter."""
    return next(csv.reader(io.StringIO(rest), delimiter=",", quotechar='"',
                           escapechar="\\", doublequote=False))


_ROBOT_LINE_RE = re.compile(r"^([A-Za-z]+):(.*)$")


def _parse_duration(value):
    """MakeMKV's own TINFO attribute 9 (duration) format, confirmed real: "H:MM:SS" or
    "HH:MM:SS". Returns seconds (float), or None if it doesn't parse (an unexpected/blank
    value is not a reason to crash -- that title is just treated as unknown-length)."""
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return None


def _run_makemkvcon_robot(cmd, stop_event, on_msg, on_tcount, on_tinfo, on_prgv):
    """Runs a `makemkvcon -r ...` command, dispatching each real robot-mode line it prints
    on stdout to the matching callback as it arrives (same live-streaming shape as
    sbs_to_mvc_cli._run_ffmpeg_stage, just for MakeMKV's own line format instead of ffmpeg's).
    Raises Cancelled() if stop_event fires; otherwise returns (returncode, stdout_tail) once
    the process exits, where stdout_tail is the last ~40 lines (for error reporting)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail = []
    callback_error = []

    def _read_stderr():
        for _ in proc.stderr:
            pass  # MakeMKV's real robot output is all on stdout (confirmed live); drained only to avoid a full pipe buffer stall

    def _read_stdout():
        # on_msg/on_tinfo/etc. run on THIS thread -- any exception they raise (e.g.
        # _check_license_problem's RuntimeError) must reach the caller's thread instead of
        # just killing this background thread silently (its default behaviour would leave
        # the main thread waiting on a process that keeps running normally, surfacing the
        # wrong "no titles found" error instead of the real license problem -- reproduced
        # directly while writing this module's own self-tests).
        try:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                tail.append(line)
                del tail[:-40]
                m = _ROBOT_LINE_RE.match(line)
                if not m:
                    continue
                tag, rest = m.group(1), m.group(2)
                try:
                    fields = _split_robot_fields(rest)
                except csv.Error:
                    continue
                if tag == "MSG" and len(fields) >= 4:
                    on_msg(fields[3])
                elif tag == "TCOUNT" and fields:
                    on_tcount(fields[0])
                elif tag == "TINFO" and len(fields) >= 2:
                    on_tinfo(fields[0], fields[1], fields[-1])
                elif tag == "PRGV" and len(fields) >= 3:
                    on_prgv(fields[0], fields[1], fields[2])
        except Exception as e:  # noqa
            callback_error.append(e)

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread.start()
    stdout_thread.start()
    cancelled = False
    while proc.poll() is None:
        if stop_event is not None and stop_event.is_set():
            proc.kill()
            cancelled = True
            break
        if callback_error:
            # A callback already raised (e.g. a real license problem) -- stop waiting on a
            # process that may run for a long time yet rather than hang here until it exits
            # or its stdout pipe fills on its own.
            proc.kill()
            break
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
    stdout_thread.join()
    stderr_thread.join()
    if cancelled:
        raise Cancelled()
    if callback_error:
        raise callback_error[0]
    return proc.returncode, tail


def _check_license_problem(message):
    """Raises a clear, specific RuntimeError if `message` (one MSG line's own text) looks
    like a real MakeMKV trial/license problem rather than an ordinary error -- see
    _LICENSE_PROBLEM_SUBSTRINGS above for the real, confirmed MakeMKV wording this matches.
    A no-op for any other message."""
    lower = message.lower()
    if any(s in lower for s in _LICENSE_PROBLEM_SUBSTRINGS):
        raise RuntimeError(
            f"MakeMKV's own trial/license has a problem: \"{message}\" -- open MakeMKV "
            f"itself (not this app) to renew your license or trial key, then try again.")


def _scan_main_title(makemkvcon, iso_path, cache_mb, stop_event, progress_cb):
    """Runs `makemkvcon -r info iso:<path>`, parses every title's real duration, and
    returns the title index with the longest duration -- the real main feature, never a
    short extra/trailer. Raises RuntimeError (with MakeMKV's own real error text folded in)
    if the ISO can't be opened, no titles are found, or MakeMKV's own output reports a real
    trial/license problem."""
    if progress_cb:
        progress_cb("scan", 0, 1)
    cmd = [makemkvcon, "-r", f"--cache={cache_mb}", "info", f"iso:{iso_path}"]
    log_subprocess_cmd("makemkv-info", cmd)
    durations = {}
    messages = []

    def on_msg(message):
        messages.append(message)
        _check_license_problem(message)

    def on_tcount(count_str):
        pass  # informational only -- the real title set is built from TINFO below

    def on_tinfo(title_id, attr_code, value):
        if attr_code == "9":
            seconds = _parse_duration(value)
            if seconds is not None:
                durations[int(title_id)] = seconds

    def on_prgv(current, total, max_):
        pass  # the scan itself has no per-title progress worth surfacing

    returncode, tail = _run_makemkvcon_robot(cmd, stop_event, on_msg, on_tcount, on_tinfo, on_prgv)
    if not durations:
        raise RuntimeError(
            "MakeMKV found no titles on this ISO -- it may not be a valid 3D Blu-ray image, "
            "or MakeMKV's own scan failed:\n  " + "\n  ".join(messages[-5:] or tail[-5:]))
    if progress_cb:
        progress_cb("scan", 1, 1)
    return max(durations, key=durations.get)


def convert(iso_path, output_path, work_dir=None, cache_mb=1024, keep_temp=False,
            stop_event=None, progress_cb=None):
    """Full job: scans `iso_path` for its real main-feature title (longest duration), rips
    just that title with MakeMKV into a work directory, then atomically moves the result to
    `output_path`. Returns output_path on success.

    progress_cb(stage, done, total), stage in {"scan", "rip"}. Raises RuntimeError if
    MakeMKV isn't found/installed, the ISO can't be read, no titles are found, a real
    trial/license problem is detected in MakeMKV's own output, or the rip itself fails;
    raises Cancelled() if stop_event fires."""
    makemkvcon = _find_makemkvcon()
    if makemkvcon is None:
        raise RuntimeError(
            "MakeMKV not found -- this tool needs a real, separately installed and licensed "
            "copy of MakeMKV (looked for makemkvcon64.exe/makemkvcon.exe on PATH, in the "
            "registry, and under Program Files). Install MakeMKV yourself from "
            "https://www.makemkv.com/ first; this project never bundles or installs it.")
    if not path.exists(iso_path):
        raise RuntimeError(f"input file not found: {iso_path}")

    out_dir = path.dirname(path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = path.splitext(path.basename(output_path))[0]
    # Same volume as output_path (a subdirectory of its own parent folder), so the final
    # os.replace() below is a genuine atomic rename, not a cross-volume copy -- the same
    # "same-volume tmp location + atomic replace" guarantee CS-IO-001 asks for, just with
    # MakeMKV itself (not this module) doing the actual long write into that location.
    work_dir = work_dir or path.join(out_dir, f"_makemkv_work_{stem}")
    os.makedirs(work_dir, exist_ok=True)
    before = set(os.listdir(work_dir))

    try:
        main_title = _scan_main_title(makemkvcon, iso_path, cache_mb, stop_event, progress_cb)

        if progress_cb:
            progress_cb("rip", 0, 0)
        cmd = [makemkvcon, "-r", f"--cache={cache_mb}", "--progress=-same",
               "mkv", f"iso:{iso_path}", str(main_title), work_dir]
        log_subprocess_cmd("makemkv-rip", cmd)
        messages = []
        last_max = [1.0]

        def on_msg(message):
            messages.append(message)
            _check_license_problem(message)

        def on_tcount(count_str):
            pass

        def on_tinfo(title_id, attr_code, value):
            pass

        def on_prgv(current, total, max_):
            try:
                total_f, max_f = float(total), float(max_)
            except ValueError:
                return
            last_max[0] = max_f
            if progress_cb:
                progress_cb("rip", total_f, max_f)

        returncode, tail = _run_makemkvcon_robot(cmd, stop_event, on_msg, on_tcount, on_tinfo, on_prgv)
        if returncode != 0:
            raise RuntimeError(
                f"MakeMKV rip failed (exit {returncode}):\n  " + "\n  ".join(messages[-5:] or tail[-5:]))
        if progress_cb:
            progress_cb("rip", last_max[0], last_max[0])

        after = set(os.listdir(work_dir))
        ripped = sorted(f for f in (after - before) if f.lower().endswith(".mkv"))
        if not ripped:
            ripped = sorted(f for f in after if f.lower().endswith(".mkv"))
        if len(ripped) != 1:
            raise RuntimeError(
                f"MakeMKV reported success but did not leave exactly one .mkv file in "
                f"{work_dir} (found: {ripped or 'none'})")
        os.replace(path.join(work_dir, ripped[0]), output_path)
        return output_path
    finally:
        if not keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)


def main():
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
        return 0

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iso", "-i", required=True,
                        help="the 3D Blu-ray .iso to rip (this project's own ISO/BD Folder output, or any real one)")
    parser.add_argument("--output", "-o", required=True, help="the real MVC .mkv MakeMKV produces")
    parser.add_argument("--cache", type=int, default=1024,
                        help="MakeMKV's own read cache in MB (MakeMKV's own docs recommend 1024 for Blu-ray)")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--gui-progress", action="store_true",
                        help="print machine-readable 'IW3_MVC_PROGRESS <stage> <done> <total>' lines to stdout")
    args = parser.parse_args()

    if args.gui_progress:
        def show(stage, done, total):
            print(f"IW3_MVC_PROGRESS {stage} {done} {total}", flush=True)
    else:
        def show(stage, done, total):
            print(f"\r[makemkv] {stage}: {done}/{total}      ", end="", file=sys.stderr, flush=True)

    try:
        out = convert(args.iso, args.output, work_dir=args.work_dir, cache_mb=args.cache,
                      keep_temp=args.keep_temp, progress_cb=show)
    except Cancelled:
        print("\n[makemkv] cancelled", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    print(f"\n[makemkv] done: {out}", file=sys.stderr)
    return 0


# --- self-tests (no real MakeMKV install assumed -- subprocess.Popen is mocked throughout) ---

def _self_test_split_robot_fields():
    """Real captured line (this session, against the actual bundled MakeMKV v1.18.4
    binary, run read-only against a fake/invalid .iso): confirms backslash-escaped quoted
    Windows paths round-trip back to plain single backslashes."""
    rest = (r'2003,16777216,3,"Error \'Internal error\' occurred while reading '
            r'\'C:\\Users\\decker\\fake.iso\' at offset \'524288\'",'
            r'"Error \'%1\' occurred while reading \'%2\' at offset \'%3\'",'
            r'"Internal error","C:\\Users\\decker\\fake.iso","524288"')
    fields = _split_robot_fields(rest)
    assert fields[0] == "2003"
    assert fields[5] == "Internal error"
    assert fields[6] == r"C:\Users\decker\fake.iso"
    print("_self_test_split_robot_fields: PASS")


def _self_test_parse_duration():
    assert _parse_duration("1:23:45") == 1 * 3600 + 23 * 60 + 45
    assert _parse_duration("01:02:03.5") == 1 * 3600 + 2 * 60 + 3.5
    assert _parse_duration("not-a-duration") is None
    print("_self_test_parse_duration: PASS")


def _self_test_license_problem_detection():
    from unittest import mock
    try:
        _check_license_problem("Evaluation period has expired, shareware functionality unavailable")
        raise AssertionError("expected a license-problem RuntimeError")
    except RuntimeError as e:
        assert "renew your license" in str(e)
    _check_license_problem("Failed to open disc")  # unrelated message: no-op
    with mock.patch("sys.argv", sys.argv):
        pass
    print("_self_test_license_problem_detection: PASS")


def _self_test_not_found_raises():
    from unittest import mock
    with mock.patch.object(sys.modules[__name__], "_find_makemkvcon", return_value=None):
        try:
            convert("in.iso", "out.mkv")
            raise AssertionError("expected RuntimeError when MakeMKV isn't found")
        except RuntimeError as e:
            assert "MakeMKV not found" in str(e)
    print("_self_test_not_found_raises: PASS")


class _FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self.stderr = iter(())
        self.returncode = 0
        self._polled = False

    def poll(self):
        if not self._polled:
            self._polled = True
            return None
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def _info_lines(titles):
    """titles: list of (title_id, duration_str) -> synthetic robot-mode `info` output."""
    lines = [f'MSG:1005,0,1,"MakeMKV v1.18.4 win(x64-release) started","%1 started"']
    for tid, dur in titles:
        lines.append(f'TINFO:{tid},9,0,"{dur}"')
    lines.append(f"TCOUNT:{len(titles)}")
    return [l + "\n" for l in lines]


def _self_test_main_title_selection():
    """Three titles (a short extra, the real movie, a trailer) -- the longest real
    duration (title 1, 2h10m15s) must be chosen, not title 0 or the title count."""
    from unittest import mock
    lines = _info_lines([(0, "0:05:00"), (1, "2:10:15"), (2, "0:03:30")])

    def fake_popen(cmd, **kwargs):
        assert "info" in cmd and "iso:fake.iso" in cmd
        return _FakeProc(lines)

    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        chosen = _scan_main_title("makemkvcon.exe", "fake.iso", 1024, None, None)
    assert chosen == 1, chosen
    print("_self_test_main_title_selection: PASS")


def _self_test_scan_license_problem_raises():
    from unittest import mock
    lines = [l + "\n" for l in [
        'MSG:1005,0,1,"MakeMKV v1.18.4 started","%1 started"',
        'MSG:5055,0,0,"Evaluation period has expired, shareware functionality unavailable",'
        '"Evaluation period has expired, shareware functionality unavailable"',
    ]]

    def fake_popen(cmd, **kwargs):
        return _FakeProc(lines)

    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        try:
            _scan_main_title("makemkvcon.exe", "fake.iso", 1024, None, None)
            raise AssertionError("expected a license-problem RuntimeError")
        except RuntimeError as e:
            assert "renew your license" in str(e)
    print("_self_test_scan_license_problem_raises: PASS")


def _self_test_no_titles_raises():
    from unittest import mock
    lines = [l + "\n" for l in [
        'MSG:5010,0,0,"Failed to open disc","Failed to open disc"', "TCOUNT:0",
    ]]

    def fake_popen(cmd, **kwargs):
        return _FakeProc(lines)

    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        try:
            _scan_main_title("makemkvcon.exe", "fake.iso", 1024, None, None)
            raise AssertionError("expected a RuntimeError when no titles are found")
        except RuntimeError as e:
            assert "no titles" in str(e)
    print("_self_test_no_titles_raises: PASS")


def _self_test_progress_forwarded():
    """PRGV:current,total,max -> progress_cb("rip", total, max), in order."""
    from unittest import mock
    lines = [l + "\n" for l in [
        "PRGV:0,0,65536", "PRGV:10000,20000,65536", "PRGV:65536,65536,65536",
    ]]

    def fake_popen(cmd, **kwargs):
        return _FakeProc(lines)

    seen = []
    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        _run_makemkvcon_robot(["makemkvcon.exe"], None, lambda m: None, lambda c: None,
                              lambda *a: None, lambda cur, tot, mx: seen.append((float(tot), float(mx))))
    assert seen == [(0.0, 65536.0), (20000.0, 65536.0), (65536.0, 65536.0)], seen
    print("_self_test_progress_forwarded: PASS")


def _self_test_cancel_raises():
    from unittest import mock

    class _NeverEndingProc(_FakeProc):
        def poll(self):
            return None  # never exits on its own

    killed = []

    class _KillableProc(_NeverEndingProc):
        def kill(self):
            killed.append(True)

    def fake_popen(cmd, **kwargs):
        return _KillableProc([])

    class _FakeEvent:
        def is_set(self):
            return True

    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        try:
            _run_makemkvcon_robot(["makemkvcon.exe"], _FakeEvent(), lambda m: None,
                                  lambda c: None, lambda *a: None, lambda *a: None)
            raise AssertionError("expected Cancelled")
        except Cancelled:
            pass
    assert killed, "the process must be killed on cancel"
    print("_self_test_cancel_raises: PASS")


def _self_test_mocked_end_to_end():
    """No real MakeMKV/ISO: subprocess.Popen is mocked for both the info scan and the rip
    itself (which 'creates' a fake .mkv file in the work directory, the same way the real
    makemkvcon would). Proves: the main (longest) title is selected and passed to the real
    `mkv` command, the ripped file is atomically moved to output_path, and the work
    directory is cleaned up afterward."""
    import tempfile
    from unittest import mock

    rip_calls = []

    def fake_popen(cmd, **kwargs):
        if "info" in cmd:
            return _FakeProc(_info_lines([(0, "0:02:00"), (1, "1:30:00")]))
        if "mkv" in cmd:
            rip_calls.append(cmd)
            work_dir = cmd[-1]
            with open(path.join(work_dir, "the_movie_t01.mkv"), "wb") as f:
                f.write(b"fake-mkv-data")
            return _FakeProc([l + "\n" for l in ["PRGV:0,0,65536", "PRGV:65536,65536,65536"]])
        raise AssertionError(f"unexpected command: {cmd}")

    with tempfile.TemporaryDirectory() as tmp_dir, \
         mock.patch.object(sys.modules[__name__], "_find_makemkvcon", return_value="makemkvcon.exe"), \
         mock.patch("subprocess.Popen", side_effect=fake_popen):
        iso_path = path.join(tmp_dir, "movie.iso")
        open(iso_path, "wb").close()
        out_path = path.join(tmp_dir, "movie.mkv")
        progress = []
        result = convert(iso_path, out_path, progress_cb=lambda *a: progress.append(a))

        assert result == out_path
        assert path.exists(out_path) and open(out_path, "rb").read() == b"fake-mkv-data"
        assert rip_calls and rip_calls[0][rip_calls[0].index("mkv") + 2] == "1", rip_calls
        assert not path.exists(path.join(tmp_dir, "_makemkv_work_movie")), \
            "the work directory must be cleaned up after a successful run"
        assert ("scan", 0, 1) in progress and ("scan", 1, 1) in progress
        assert ("rip", 65536.0, 65536.0) in progress
    print("_self_test_mocked_end_to_end: PASS")


def _run_self_tests():
    _self_test_split_robot_fields()
    _self_test_parse_duration()
    _self_test_license_problem_detection()
    _self_test_not_found_raises()
    _self_test_main_title_selection()
    _self_test_scan_license_problem_raises()
    _self_test_no_titles_raises()
    _self_test_progress_forwarded()
    _self_test_cancel_raises()
    _self_test_mocked_end_to_end()
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
