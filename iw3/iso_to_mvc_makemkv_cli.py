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

After a successful rip, `convert()` also tags the result's video track with Matroska
StereoMode=13 via mkvpropedit (real bug found on decker's own real MOONED_3D.iso rip:
MakeMKV's own raw output had no StereoMode property at all, so no 3D-aware player/TV
had anything to auto-detect 3D from). This reuses the exact same value/mechanism this
project's own real MVC-producing tools already proved on real hardware -- see ADR-284
and sbs_to_mvc_cli.py's `_mux_mkv_via_mkvmerge()`.

Real follow-up fix (ADR-333): `convert()` used to hand MakeMKV the `iso:<path>` source
directly -- confirmed, via a real byte-level NAL scan of decker's own MOONED_3D.iso
rip, to produce a base-view-only file with ZERO real MVC data (no type-15/type-20 NAL
units at all), even though MakeMKV's own `info` scan correctly reports a real MVC
stream on the disc. Root cause, confirmed by live testing against the real ISO: TWO
separate real problems, both needed, neither sufficient alone --
1. MakeMKV's `iso:<path>` access path genuinely skips the real interleaved `.ssif`
   payload on this kind of disc. Mounting the ISO as a real virtual drive first (the
   same `mount_iso()`/`dismount_iso()` this project's own `mvc_extract_cli.py` already
   uses) and pointing MakeMKV at the resulting disc via `disc:<id>` ("direct disc
   access mode", confirmed in MakeMKV's own real log output) instead fixes this --
   `info` then reports the real `Mpeg4-MVC-3D`/`StereoHigh` stream with correct size.
2. Even in `disc:` mode, MakeMKV's own SHIPPED default track-selection rule
   (`-sel:mvcvideo` in its real `default.mmcp.xml`) EXCLUDES the MVC video stream by
   default -- confirmed this is exactly why decker's own real MakeMKV GUI left that
   track unchecked until he manually checked it. `convert()` now writes its own small
   custom MakeMKV profile (`_MVC_SELECT_PROFILE_XML` below) that flips this one rule to
   `+sel:mvcvideo` and passes it via `--profile=<path>`, confirmed live (byte-for-byte
   output size changed and a real NAL scan found genuine type-15/type-20 data) -- the
   profile's own `<name>` tag must be a real, unique string (NOT MakeMKV's own internal
   resource-id name `:5086` that its shipped default.mmcp.xml uses), confirmed live: a
   colliding name causes MakeMKV to silently keep using its built-in default profile
   with zero error/warning in robot-mode output -- the single most misleading dead end
   in this investigation.
Selection is keyed on MakeMKV's own generic `mvcvideo` stream classifier, not a
hardcoded track index, so this generalizes to any real 3D Blu-ray disc/ISO, not just
this one disc's own track layout.
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

from .mvc_extract_cli import Cancelled, mount_iso, dismount_iso
from .utils import log_subprocess_cmd, _find_mkvpropedit, _apply_stereo_mode_tag

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

# ADR-333: MakeMKV's own shipped default.mmcp.xml selection rule excludes the real MVC
# video stream by default (`-sel:mvcvideo`) -- confirmed live, this is why a real 3D
# Blu-ray's MVC/StereoHigh track comes unchecked in MakeMKV's own GUI track tree.
# Everything else here is copied verbatim from that shipped default (same mkvSettings,
# same selection scoring rules) with exactly that one rule flipped to `+sel:mvcvideo`,
# so normal audio/subtitle default-track selection behaves identically to a stock
# MakeMKV install -- this only changes whether the MVC video stream is included.
# `mvcvideo` is MakeMKV's own generic stream classifier (confirmed against this
# session's real `info` scan output), not a per-disc track index, so this selection
# rule applies to any real 3D Blu-ray disc/ISO MakeMKV recognizes as having one.
# The `<name>` tag MUST be a real, unique string -- confirmed live, reusing MakeMKV's
# own internal default-profile resource-id name (":5086", what its shipped
# default.mmcp.xml uses) makes MakeMKV silently treat this as a duplicate of its own
# built-in default and keep using THAT (with zero error or warning in robot-mode
# output), silently undoing the `+sel:mvcvideo` flip above.
_MVC_SELECT_PROFILE_XML = """<?xml version="1.0" encoding="utf-8"?>
<profile>
    <name lang="eng">iw3 MVC Select</name>
    <mkvSettings
        ignoreForcedSubtitlesFlag="true"
        useISO639Type2T="false"
        setFirstAudioTrackAsDefault="true"
        setFirstSubtitleTrackAsDefault="true"
        setFirstForcedSubtitleTrackAsDefault="true"
        insertFirstChapter00IfMissing="true"
    />
    <outputSettings name="copy" outputFormat="directCopy">
        <description lang="eng">Copy track as is</description>
    </outputSettings>
    <trackSettings input="default">
        <output outputSettingsName="copy"
                defaultSelection="-sel:all,+sel:(favlang|nolang|single),-sel:(havemulti|havecore),+sel:mvcvideo,=100:all,-10:favlang">
        </output>
    </trackSettings>
</profile>
"""


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


def _run_makemkvcon_robot(cmd, stop_event, on_msg, on_tcount, on_tinfo, on_prgv, on_drv=None):
    """Runs a `makemkvcon -r ...` command, dispatching each real robot-mode line it prints
    on stdout to the matching callback as it arrives (same live-streaming shape as
    sbs_to_mvc_cli._run_ffmpeg_stage, just for MakeMKV's own line format instead of ffmpeg's).
    Raises Cancelled() if stop_event fires; otherwise returns (returncode, stdout_tail) once
    the process exits, where stdout_tail is the last ~40 lines (for error reporting).

    on_drv(fields), fields being one `DRV:id,...,"drive name","disc name","drive letter"`
    line's own fields -- optional (defaults to a no-op) since only disc-id resolution
    (ADR-333) needs per-drive info; every other existing call site ignores DRV lines,
    same as before this parameter existed."""
    if on_drv is None:
        def on_drv(fields):
            pass
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
                elif tag == "DRV":
                    on_drv(fields)
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


def _resolve_disc_index(makemkvcon, drive_root, cache_mb, stop_event):
    """ADR-333: `mount_iso()` gives a real drive letter, not a MakeMKV disc id -- MakeMKV
    enumerates its own disc ids independently (its real `DRV:` robot-mode lines), and the
    id assigned to a given drive letter is not fixed (confirmed live: it depends on this
    machine/session's own drive enumeration order, not on the drive letter itself).

    `info disc:9999` is a real, confirmed-live trick: 9999 is never a real disc id, so
    MakeMKV always reports "Failed to open disc" for it -- but it still prints one real
    `DRV:` line per optical (real or mounted-virtual) drive it sees first, which is all
    this needs. Matches the DRV line whose own drive-letter field equals `drive_root`'s
    drive letter, so this generalizes to whatever drive letter Windows happens to assign
    when mounting, instead of assuming a fixed disc id like `disc:0`."""
    letter = drive_root.rstrip("\\").upper()
    cmd = [makemkvcon, "-r", f"--cache={cache_mb}", "info", "disc:9999"]
    log_subprocess_cmd("makemkv-list-drives", cmd)
    drv_letters = {}

    def on_msg(message):
        pass  # the "Failed to open disc" MSG for the fake id 9999 is expected, not a real error

    def on_drv(fields):
        if len(fields) >= 7 and fields[6]:
            drv_letters[fields[6].rstrip("\\").upper()] = fields[0]

    _run_makemkvcon_robot(cmd, stop_event, on_msg, lambda count_str: None,
                          lambda title_id, attr_code, value: None, lambda cur, tot, mx: None,
                          on_drv=on_drv)
    disc_id = drv_letters.get(letter)
    if disc_id is None:
        raise RuntimeError(
            f"MakeMKV did not report a disc id for the mounted drive {drive_root} -- it "
            f"may have failed to recognize the ISO as a disc (drive letters MakeMKV saw: "
            f"{sorted(drv_letters) or 'none'})")
    return disc_id


def _scan_main_title(makemkvcon, source, cache_mb, stop_event, progress_cb):
    """Runs `makemkvcon -r info <source>`, parses every title's real duration, and
    returns the title index with the longest duration -- the real main feature, never a
    short extra/trailer. Raises RuntimeError (with MakeMKV's own real error text folded in)
    if the disc can't be opened, no titles are found, or MakeMKV's own output reports a
    real trial/license problem."""
    if progress_cb:
        progress_cb("scan", 0, 1)
    cmd = [makemkvcon, "-r", f"--cache={cache_mb}", "info", source]
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
        # ADR-333: `iso:<path>` genuinely skips this disc's real interleaved `.ssif` MVC
        # payload (confirmed live via a byte-level NAL scan -- see module docstring).
        # Mounting the ISO as a real virtual drive first and pointing MakeMKV at the
        # resulting disc (`disc:<id>`, MakeMKV's own "direct disc access mode") instead
        # fixes this. Same mount_iso()/dismount_iso() this project's own mvc_extract_cli.py
        # already uses for the read direction -- an ISO the user already had mounted is
        # used as-is and (mounted_by_us is False) never dismounted out from under them.
        drive_root, mounted_by_us = mount_iso(iso_path)
        try:
            if stop_event is not None and stop_event.is_set():
                raise Cancelled()
            disc_id = _resolve_disc_index(makemkvcon, drive_root, cache_mb, stop_event)
            source = f"disc:{disc_id}"

            # Second real half of ADR-333's fix: MakeMKV's own shipped default profile
            # excludes the MVC video stream by default (confirmed live) -- this custom
            # profile flips that one rule. Written into work_dir so it's cleaned up with
            # everything else in the `finally` below; a fresh file per run rather than a
            # bundled resource, since it's tiny and has no reason to vary.
            profile_path = path.join(work_dir, "_mvc_select.mmcp.xml")
            with open(profile_path, "w", encoding="utf-8") as f:
                f.write(_MVC_SELECT_PROFILE_XML)

            main_title = _scan_main_title(makemkvcon, source, cache_mb, stop_event, progress_cb)

            if progress_cb:
                progress_cb("rip", 0, 0)
            cmd = [makemkvcon, "-r", f"--cache={cache_mb}", "--progress=-same",
                   f"--profile={profile_path}", "mkv", source, str(main_title), work_dir]
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
        finally:
            if mounted_by_us:
                dismount_iso(iso_path)

        after = set(os.listdir(work_dir))
        ripped = sorted(f for f in (after - before) if f.lower().endswith(".mkv"))
        if not ripped:
            ripped = sorted(f for f in after if f.lower().endswith(".mkv"))
        if len(ripped) != 1:
            raise RuntimeError(
                f"MakeMKV reported success but did not leave exactly one .mkv file in "
                f"{work_dir} (found: {ripped or 'none'})")
        os.replace(path.join(work_dir, ripped[0]), output_path)

        # MakeMKV's own raw rip does not always carry a Matroska StereoMode tag (confirmed:
        # a real rip of decker's own MOONED_3D.iso had none at all), so a 3D-aware
        # player/TV has nothing to auto-detect 3D from. Tag it the same way this project's
        # OWN real MVC-producing tools already do (ADR-284, sbs_to_mvc_cli.py's
        # _mux_mkv_via_mkvmerge): StereoMode 13 ("both eyes in one Block, left eye first"),
        # the de facto convention real players key off for MVC content (no dedicated
        # Matroska enum exists for "H.264 MVC"). Unconditional, not gated on a per-title
        # MVC signal -- this entire tool's only real purpose is ripping real 3D Blu-ray MVC
        # content (its caller always hands it this project's own ISO/BD-folder MVC output,
        # or a real third-party 3D Blu-ray image), and MakeMKV's own robot-mode title scan
        # exposes no cheap, reliable per-title "is this genuinely MVC" attribute to gate on.
        # Base view = left eye is the real Blu-ray 3D spec convention (unlike this
        # project's own encode path, there is no swap_eyes here to account for -- MakeMKV
        # is ripping an already-authored disc, not re-packing eyes itself). Never fatal:
        # same as sbs_to_mvc_cli.py, a missing mkvpropedit or a failed tag attempt is
        # printed and left as a non-fatal note -- the file MakeMKV already produced is
        # still returned, just not auto-detected as 3D by every player.
        from types import SimpleNamespace
        mkvpropedit = _find_mkvpropedit()
        if mkvpropedit is None:
            print(f"[makemkv] note: mkvpropedit not found -- "
                  f"{path.basename(str(output_path))} was NOT tagged with a StereoMode; a "
                  f"3D-aware player may not auto-detect it as 3D", file=sys.stderr)
        else:
            _apply_stereo_mode_tag(output_path, SimpleNamespace(stereo_mode_tag=True, vr180=False),
                                   mkvpropedit_bin=mkvpropedit, value_override=13)
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


def _drv_lines(pairs):
    """pairs: list of (disc_id, drive_letter) -> synthetic robot-mode `info disc:9999`
    drive-enumeration output (same real shape confirmed live: one DRV: line per real/
    mounted-virtual optical drive, then a "Failed to open disc" MSG for the fake id)."""
    lines = [f'MSG:1005,0,1,"MakeMKV v1.18.4 win(x64-release) started","%1 started"']
    for disc_id, letter in pairs:
        lines.append(f'DRV:{disc_id},2,999,4,"Fake Drive","Fake Disc","{letter}"')
    lines.append('MSG:5010,0,0,"Failed to open disc","Failed to open disc"')
    return [l + "\n" for l in lines]


def _self_test_resolve_disc_index():
    """ADR-333: the mounted drive letter ("Z:") must resolve to ITS OWN disc id (1, not
    the first-listed drive's id 0) -- proves the match is keyed on drive letter, not on
    enumeration order/position."""
    from unittest import mock
    lines = _drv_lines([("0", "F:"), ("1", "Z:")])

    def fake_popen(cmd, **kwargs):
        assert "disc:9999" in cmd, cmd
        return _FakeProc(lines)

    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        disc_id = _resolve_disc_index("makemkvcon.exe", "Z:\\", 1024, None)
    assert disc_id == "1", disc_id
    print("_self_test_resolve_disc_index: PASS")


def _self_test_resolve_disc_index_not_found_raises():
    from unittest import mock
    lines = _drv_lines([("0", "F:")])

    def fake_popen(cmd, **kwargs):
        return _FakeProc(lines)

    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        try:
            _resolve_disc_index("makemkvcon.exe", "Z:\\", 1024, None)
            raise AssertionError("expected a RuntimeError when the drive letter isn't seen")
        except RuntimeError as e:
            assert "did not report a disc id" in str(e)
    print("_self_test_resolve_disc_index_not_found_raises: PASS")


def _self_test_main_title_selection():
    """Three titles (a short extra, the real movie, a trailer) -- the longest real
    duration (title 1, 2h10m15s) must be chosen, not title 0 or the title count."""
    from unittest import mock
    lines = _info_lines([(0, "0:05:00"), (1, "2:10:15"), (2, "0:03:30")])

    def fake_popen(cmd, **kwargs):
        assert "info" in cmd and "disc:0" in cmd
        return _FakeProc(lines)

    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        chosen = _scan_main_title("makemkvcon.exe", "disc:0", 1024, None, None)
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
            _scan_main_title("makemkvcon.exe", "disc:0", 1024, None, None)
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
            _scan_main_title("makemkvcon.exe", "disc:0", 1024, None, None)
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
    """No real MakeMKV/ISO/mount: subprocess.Popen is mocked for the drive-enumeration
    call, the info scan, and the rip itself (which 'creates' a fake .mkv file in the work
    directory, the same way the real makemkvcon would); mount_iso/dismount_iso are mocked
    too (not a real Windows mount). Proves: the mounted drive's own disc id is resolved
    and used (ADR-333, not the old `iso:<path>` source), the real forced-MVC-selection
    profile is written and passed via `--profile=`, the main (longest) title is selected
    and passed to the real `mkv` command, the ripped file is atomically moved to
    output_path, the work directory (and the profile file inside it) is cleaned up
    afterward, the ISO is dismounted exactly once, and the real StereoMode-tagging step
    (added for the real "missing stereo tag" bug, see module docstring) runs with
    StereoMode 13 after a successful rip. _find_mkvpropedit/_apply_stereo_mode_tag are
    mocked here (not real mkvpropedit) so this test stays fast/deterministic regardless
    of whether a real mkvpropedit happens to be installed on the machine running this
    suite -- the real, hands-on mkvpropedit/mkvmerge/MakeMKV check against decker's own
    actual ripped file is done separately, outside this self-test suite."""
    import tempfile
    from unittest import mock

    rip_calls = []

    def fake_popen(cmd, **kwargs):
        if "disc:9999" in cmd:
            return _FakeProc(_drv_lines([("3", "Z:")]))
        if "info" in cmd:
            assert "disc:3" in cmd, cmd
            return _FakeProc(_info_lines([(0, "0:02:00"), (1, "1:30:00")]))
        if "mkv" in cmd:
            assert "disc:3" in cmd, cmd
            profile_args = [a for a in cmd if a.startswith("--profile=")]
            assert profile_args and path.exists(profile_args[0].split("=", 1)[1]), cmd
            rip_calls.append(cmd)
            work_dir = cmd[-1]
            with open(path.join(work_dir, "the_movie_t01.mkv"), "wb") as f:
                f.write(b"fake-mkv-data")
            return _FakeProc([l + "\n" for l in ["PRGV:0,0,65536", "PRGV:65536,65536,65536"]])
        raise AssertionError(f"unexpected command: {cmd}")

    dismount_calls = []

    with tempfile.TemporaryDirectory() as tmp_dir, \
         mock.patch.object(sys.modules[__name__], "_find_makemkvcon", return_value="makemkvcon.exe"), \
         mock.patch.object(sys.modules[__name__], "_find_mkvpropedit", return_value="mkvpropedit.exe"), \
         mock.patch.object(sys.modules[__name__], "_apply_stereo_mode_tag", return_value=True) as mock_tag, \
         mock.patch.object(sys.modules[__name__], "mount_iso", return_value=("Z:\\", True)), \
         mock.patch.object(sys.modules[__name__], "dismount_iso",
                           side_effect=lambda p: dismount_calls.append(p)), \
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
            "the work directory (and the profile file written inside it) must be " \
            "cleaned up after a successful run"
        assert dismount_calls == [iso_path], dismount_calls

        mock_tag.assert_called_once()
        tag_call_args, tag_call_kwargs = mock_tag.call_args
        assert tag_call_args[0] == out_path, tag_call_args
        assert tag_call_kwargs["value_override"] == 13, tag_call_kwargs
        assert tag_call_kwargs["mkvpropedit_bin"] == "mkvpropedit.exe", tag_call_kwargs
        assert getattr(tag_call_args[1], "stereo_mode_tag", False) is True, tag_call_args[1]
        assert ("scan", 0, 1) in progress and ("scan", 1, 1) in progress
        assert ("rip", 65536.0, 65536.0) in progress
    print("_self_test_mocked_end_to_end: PASS")


def _self_test_mount_not_dismounted_if_already_mounted():
    """mount_iso() returns mounted_by_us=False when the user already had the ISO mounted
    themselves (same convention as mvc_extract_cli.py's own mount_iso docstring) --
    convert() must never dismount a drive it didn't mount, so as not to yank it out from
    under the user's own separate use of it."""
    import tempfile
    from unittest import mock

    def fake_popen(cmd, **kwargs):
        if "disc:9999" in cmd:
            return _FakeProc(_drv_lines([("0", "Z:")]))
        if "info" in cmd:
            return _FakeProc(_info_lines([(0, "1:00:00")]))
        if "mkv" in cmd:
            work_dir = cmd[-1]
            with open(path.join(work_dir, "the_movie_t00.mkv"), "wb") as f:
                f.write(b"fake-mkv-data")
            return _FakeProc([l + "\n" for l in ["PRGV:65536,65536,65536"]])
        raise AssertionError(f"unexpected command: {cmd}")

    dismount_calls = []

    with tempfile.TemporaryDirectory() as tmp_dir, \
         mock.patch.object(sys.modules[__name__], "_find_makemkvcon", return_value="makemkvcon.exe"), \
         mock.patch.object(sys.modules[__name__], "_find_mkvpropedit", return_value=None), \
         mock.patch.object(sys.modules[__name__], "_apply_stereo_mode_tag"), \
         mock.patch.object(sys.modules[__name__], "mount_iso", return_value=("Z:\\", False)), \
         mock.patch.object(sys.modules[__name__], "dismount_iso",
                           side_effect=lambda p: dismount_calls.append(p)), \
         mock.patch("subprocess.Popen", side_effect=fake_popen):
        iso_path = path.join(tmp_dir, "movie.iso")
        open(iso_path, "wb").close()
        out_path = path.join(tmp_dir, "movie.mkv")
        result = convert(iso_path, out_path)

        assert result == out_path
        assert dismount_calls == [], \
            "an ISO the user already had mounted must never be dismounted by this tool"
    print("_self_test_mount_not_dismounted_if_already_mounted: PASS")


def _self_test_tagging_missing_mkvpropedit_is_nonfatal():
    """Real tool-availability edge case: if mkvpropedit can't be found, the rip itself
    must still succeed and still return the real output_path (same non-fatal philosophy
    as sbs_to_mvc_cli.py's own StereoMode-tagging step) -- a missing optional tagging tool
    must never turn an otherwise-successful rip into a failure."""
    import tempfile
    from unittest import mock

    def fake_popen(cmd, **kwargs):
        if "disc:9999" in cmd:
            return _FakeProc(_drv_lines([("0", "Z:")]))
        if "info" in cmd:
            return _FakeProc(_info_lines([(0, "1:00:00")]))
        if "mkv" in cmd:
            work_dir = cmd[-1]
            with open(path.join(work_dir, "the_movie_t00.mkv"), "wb") as f:
                f.write(b"fake-mkv-data")
            return _FakeProc([l + "\n" for l in ["PRGV:65536,65536,65536"]])
        raise AssertionError(f"unexpected command: {cmd}")

    with tempfile.TemporaryDirectory() as tmp_dir, \
         mock.patch.object(sys.modules[__name__], "_find_makemkvcon", return_value="makemkvcon.exe"), \
         mock.patch.object(sys.modules[__name__], "_find_mkvpropedit", return_value=None), \
         mock.patch.object(sys.modules[__name__], "_apply_stereo_mode_tag") as mock_tag, \
         mock.patch.object(sys.modules[__name__], "mount_iso", return_value=("Z:\\", True)), \
         mock.patch.object(sys.modules[__name__], "dismount_iso"), \
         mock.patch("subprocess.Popen", side_effect=fake_popen):
        iso_path = path.join(tmp_dir, "movie.iso")
        open(iso_path, "wb").close()
        out_path = path.join(tmp_dir, "movie.mkv")
        result = convert(iso_path, out_path)

        assert result == out_path
        assert path.exists(out_path)
        mock_tag.assert_not_called()
    print("_self_test_tagging_missing_mkvpropedit_is_nonfatal: PASS")


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
    _self_test_resolve_disc_index()
    _self_test_resolve_disc_index_not_found_raises()
    _self_test_mocked_end_to_end()
    _self_test_mount_not_dismounted_if_already_mounted()
    _self_test_tagging_missing_mkvpropedit_is_nonfatal()
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
