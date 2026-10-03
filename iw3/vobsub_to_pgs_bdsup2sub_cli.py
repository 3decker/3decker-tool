"""python -m iw3.vobsub_to_pgs_bdsup2sub_cli -- converts a real VobSub (DVD-era bitmap)
subtitle into a real PGS (Blu-ray bitmap) subtitle, using BDSup2Sub++'s own genuine,
OCR-free, bitmap-to-bitmap conversion (it re-packages the same decoded subtitle pixels
into the PGS container format -- it does not re-render or re-OCR the text).

Why this exists (real user report, Steve, via decker, 2026-10-03): this project's own
disc-legal MVC/Blu-ray output (iw3.sbs_to_mvc_cli._plan_audio_subs()) already converts
SRT/text subtitles into PGS automatically, but a real Blu-ray disc has no legal way to
carry a VobSub track at all, so _plan_audio_subs() has always, intentionally, SKIPPED any
VobSub track with a note ("this subtitle format cannot be put on a Blu-ray") -- that is a
real, confirmed, deliberate trade-off, not a bug, and this module does NOT change it.
Steve hit this with a real movie whose only subtitle track was VobSub and manually
converted it with a different tool first. This module automates that same real conversion
step as its own standalone tool (v1: standalone only, same deliberate scope decision
iw3.iso_to_mvc_makemkv_cli made before its own later pipeline auto-chain was added --
wiring this into _plan_audio_subs()'s own pipeline is a later, separate decision once this
tool itself is proven against a real file, see the real, disclosed gap below).

BDSup2Sub++ (github.com/prinsbert/dvd-subtitle-utils, Apache-2.0/LGPL-3.0) was chosen over
Subtitle Edit (what Steve used by hand) because Subtitle Edit's own CLI is text-subtitle
-only -- confirmed not viable for a real bitmap-to-bitmap VobSub->PGS conversion. Its
license genuinely permits redistribution (unlike MakeMKV's separate-install/licensed
model -- see iw3.iso_to_mvc_makemkv_cli._find_makemkvcon()'s docstring for that contrast),
so, like ffmpeg/mkvmerge/tsMuxeR/FRIM, it is BUNDLED directly with this project (see
iw3.utils._find_bdsup2sub()) instead of requiring a separate user install -- zero extra
setup, and the ~71MB bundled build (Qt6 DLLs included) is small next to the ~220MB
mkvtoolnix/ or ~170MB ffmpeg.exe already bundled here.

Real, confirmed-from-source CLI shape: `bdsup2sub++.exe -o <output> <input>` -- the output
FORMAT is selected purely by -o's own file extension (.sup for PGS/BD-SUP; also
.sub/.idx for VobSub, .xml/.ifo for others). An unrecognized flag produces a real, specific
"Unknown option '<name>'." on stderr (confirmed real error handling against the actual
downloaded build) -- but a genuinely malformed/unusual VobSub SOURCE can instead fail
completely silently: confirmed directly, hands-on, against a real public VobSub test
sample (ffmpeg's own sample archive, Traffic2.idx/.sub) on BOTH the current build and a
2018 legacy build -- both print "Loading <path>" then exit 1 with ZERO further output
(no error text at all) and no output file. Inspecting that sample's own .idx text directly
shows why: its `langidx: 0` points at subtitle stream index 0, but the file only defines
`id: es, index: 1` -- there IS no index 0, a genuinely malformed/incomplete real-world
VobSub file, not a bug in this module or in BDSup2Sub++'s own CLI argument handling (which
was separately confirmed correct via --badflag above). RuntimeError below surfaces this
distinction honestly (empty tool output is reported as such, not papered over).

Real, disclosed gap: no FULL successful VobSub->PGS conversion has been demonstrated
end-to-end against a real, well-formed VobSub file in this environment -- no such file was
available locally beyond the malformed sample above. This module's own detection,
invocation, and error-handling paths are verified for real (see self-tests below plus the
hands-on binary checks in this docstring); the actual bitmap-to-bitmap conversion math
inside BDSup2Sub++ itself is not independently re-verified here, only exercised via its
documented and source-confirmed CLI contract.

Real extraction from an embedded MKV VobSub track: this project's own bundled ffmpeg build
can only DEMUX VobSub (read it) -- confirmed directly (`ffmpeg -muxers` lists no "vobsub"
muxer at all, only `-demuxers` does) -- so it cannot write a standalone .idx/.sub pair
itself the way it already does for PGS/text subtitles in sbs_to_mvc_cli.py. mkvextract
(already bundled alongside mkvmerge, see iw3.utils._find_mkvextract()) is used instead:
its `tracks <TID>:out.idx` mode writes the matching out.sub alongside automatically for a
real S_VOBSUB track (real, established MKVToolNix behavior). The real Matroska track ID
mkvextract needs is read from `mkvmerge -J <input>`'s own JSON identify output (each
track's top-level "id" field), NOT ffprobe's per-type 0:s:N index -- the two numbering
schemes are not interchangeable (same reasoning sbs_to_mvc_cli.py's own
_ffprobe_list_tracks() docstring gives for a different pair of tools)."""
import argparse
import json
import os
import subprocess
import sys
from os import path

# Same reasoning as iso_to_mvc_makemkv_cli.py's own top-of-file import: this module runs as
# its own `python -m iw3.vobsub_to_pgs_bdsup2sub_cli` process (invoked by the GUI's
# standalone-tool Run button), so it needs this patch applied itself -- iw3/gui.py
# importing it only covers subprocess calls made from the GUI's own process, not this
# separate one.
import nunif.gui.subprocess_patch  # noqa

from .mvc_extract_cli import Cancelled
from .utils import _find_bdsup2sub, _find_mkvextract, _find_mkvmerge, log_subprocess_cmd


def _check_cancel(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise Cancelled()


def _identify_vobsub_tracks(mkv_path, mkvmerge_bin):
    """Returns every real S_VOBSUB subtitle track in mkv_path as a list of
    (track_id, lang) tuples, track_id being the real Matroska track ID (mkvmerge's own
    top-level "id" field) that mkvextract's `tracks <TID>:out` mode expects -- NOT
    ffprobe's/ffmpeg's separate per-type 0:s:N subtitle index. Raises RuntimeError if
    mkvmerge itself can't read the file."""
    r = subprocess.run([mkvmerge_bin, "-J", mkv_path], capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(f"mkvmerge could not read {mkv_path}: {(r.stderr or r.stdout).strip()[-300:]}")
    try:
        info = json.loads(r.stdout)
    except ValueError as e:
        raise RuntimeError(f"mkvmerge -J produced output that could not be parsed as JSON: {e}")
    tracks = []
    for t in info.get("tracks", []):
        props = t.get("properties", {}) or {}
        if t.get("type") == "subtitles" and props.get("codec_id") == "S_VOBSUB":
            lang = props.get("language_ietf") or props.get("language") or ""
            lang = "" if lang in ("und", "") else lang
            tracks.append((t.get("id"), lang))
    return tracks


def extract_vobsub_from_mkv(mkv_path, out_idx_path, track=None, mkvmerge_bin=None,
                            mkvextract_bin=None):
    """Extracts one real embedded S_VOBSUB subtitle track from mkv_path into a standalone
    out_idx_path (+ the matching out_idx_path-with-.sub-extension mkvextract writes
    alongside it automatically for a VobSub track). `track`, if given, is the real
    Matroska track ID (not an ffprobe/ffmpeg subtitle index) to extract; if omitted, the
    single VobSub track found is used automatically -- raises a clear RuntimeError (naming
    every track found) if there's zero or more than one, so this never silently guesses
    among several. Returns (idx_path, sub_path)."""
    mkvmerge_bin = mkvmerge_bin or _find_mkvmerge()
    mkvextract_bin = mkvextract_bin or _find_mkvextract()
    if mkvmerge_bin is None or mkvextract_bin is None:
        raise RuntimeError(
            "mkvmerge/mkvextract not found -- this project bundles them at "
            "<install root>/mkvtoolnix/; the bundled install looks damaged or incomplete.")
    found = _identify_vobsub_tracks(mkv_path, mkvmerge_bin)
    if not found:
        raise RuntimeError(f"no VobSub (S_VOBSUB) subtitle track found in {mkv_path}")
    if track is None:
        if len(found) > 1:
            listing = ", ".join(f"track {tid} ({lang or 'unknown language'})" for tid, lang in found)
            raise RuntimeError(
                f"{mkv_path} has {len(found)} VobSub subtitle tracks ({listing}) -- pass "
                f"--track with the one to convert.")
        track = found[0][0]
    elif track not in (tid for tid, _ in found):
        listing = ", ".join(str(tid) for tid, _ in found) or "none"
        raise RuntimeError(
            f"track {track} is not a VobSub subtitle track in {mkv_path} "
            f"(real VobSub track IDs found: {listing})")

    cmd = [mkvextract_bin, mkv_path, "tracks", f"{track}:{out_idx_path}"]
    log_subprocess_cmd("mkvextract-vobsub", cmd)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out_sub_path = path.splitext(out_idx_path)[0] + ".sub"
    if (r.returncode != 0 or not path.exists(out_idx_path) or not path.exists(out_sub_path)
            or path.getsize(out_sub_path) == 0):
        tail = (r.stderr or r.stdout or "").strip()[-500:]
        raise RuntimeError(
            f"mkvextract failed to extract VobSub track {track} from {mkv_path} "
            f"(exit {r.returncode}): {tail or '(no output)'}")
    return out_idx_path, out_sub_path


def convert(input_path, output_path, track=None, work_dir=None, keep_temp=False,
            stop_event=None, progress_cb=None):
    """Full job: if input_path is an .mkv, first extracts its one real VobSub subtitle
    track (see extract_vobsub_from_mkv() above; `track` selects which one when there's more
    than one) into a standalone .idx/.sub pair in a work directory; if input_path is
    already a standalone .idx file (with its own real .sub alongside, same basename --
    confirmed this is how BDSup2Sub++ itself expects a standalone VobSub source), it's used
    directly. Either way, runs the real, confirmed `bdsup2sub++.exe -o <output> <input>`
    conversion, then atomically moves the result to output_path (which must end in .sup --
    BDSup2Sub++ itself picks the output FORMAT from this extension). Returns output_path.

    progress_cb(stage, done, total), stage in {"extract", "convert"} ("extract" only fires
    for an .mkv input). Raises RuntimeError if BDSup2Sub++/mkvextract/mkvmerge aren't found,
    the input can't be read, no VobSub track is found/selected, or the real conversion
    itself fails (including a genuinely malformed source producing no real output with no
    error text -- see this module's own docstring for a real, confirmed example); raises
    Cancelled() if stop_event fires between stages."""
    bdsup2sub_bin = _find_bdsup2sub()
    if bdsup2sub_bin is None:
        raise RuntimeError(
            "BDSup2Sub++ not found -- this project bundles it at <install root>/bdsup2sub/"
            "bdsup2sub++.exe; the bundled install looks damaged or incomplete.")
    if not output_path.lower().endswith(".sup"):
        raise ValueError(f"output_path must end in .sup (BDSup2Sub++ picks the output "
                         f"format from this extension): {output_path}")
    _check_cancel(stop_event)
    if not path.exists(input_path):
        raise RuntimeError(f"input file not found: {input_path}")

    out_dir = path.dirname(path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = path.splitext(path.basename(output_path))[0]
    # Same volume as output_path, so the final os.replace() below is a genuine atomic
    # rename, not a cross-volume copy (same "same-volume tmp + atomic replace" convention
    # as iso_to_mvc_makemkv_cli.convert()).
    work_dir = work_dir or path.join(out_dir, f"_vobsub_pgs_work_{stem}")
    os.makedirs(work_dir, exist_ok=True)

    try:
        _check_cancel(stop_event)
        lower = input_path.lower()
        if lower.endswith(".mkv"):
            if progress_cb:
                progress_cb("extract", 0, 1)
            idx_path, _sub_path = extract_vobsub_from_mkv(
                input_path, path.join(work_dir, "extracted.idx"), track=track)
            if progress_cb:
                progress_cb("extract", 1, 1)
        elif lower.endswith(".idx"):
            idx_path = input_path
            sub_path = path.splitext(input_path)[0] + ".sub"
            if not path.exists(sub_path):
                raise RuntimeError(
                    f"{input_path} has no matching .sub file alongside it ({sub_path}) -- "
                    f"BDSup2Sub++ needs the real .idx/.sub pair together.")
        else:
            raise ValueError(
                f"input must be a standalone VobSub .idx file (with its .sub alongside) or "
                f"an .mkv containing an embedded VobSub track: {input_path}")

        _check_cancel(stop_event)
        if progress_cb:
            progress_cb("convert", 0, 1)
        tmp_sup = path.join(work_dir, f"{stem}.tmp.sup")
        cmd = [bdsup2sub_bin, "-o", tmp_sup, idx_path]
        log_subprocess_cmd("bdsup2sub", cmd)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not path.exists(tmp_sup) or path.getsize(tmp_sup) == 0:
            tail = (r.stderr or r.stdout or "").strip()[-500:]
            raise RuntimeError(
                f"BDSup2Sub++ failed converting {idx_path} to PGS (exit {r.returncode}): "
                f"{tail or '(no output -- this can happen with a malformed/unusual VobSub source)'}")
        os.replace(tmp_sup, output_path)
        if progress_cb:
            progress_cb("convert", 1, 1)
        return output_path
    finally:
        if not keep_temp:
            import shutil as _shutil
            _shutil.rmtree(work_dir, ignore_errors=True)


def main():
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
        return 0

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True,
                        help="a standalone VobSub .idx file (its .sub expected alongside, same "
                             "basename) or an .mkv file containing an embedded VobSub subtitle track")
    parser.add_argument("--output", "-o", required=True, help="the real PGS .sup file to write")
    parser.add_argument("--track", type=int, default=None,
                        help="only for an .mkv --input with more than one VobSub track: the real "
                             "Matroska track ID (from `mkvmerge -J`) of the one to convert")
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
            print(f"\r[vobsub-to-pgs] {stage}: {done}/{total}      ", end="", file=sys.stderr, flush=True)

    try:
        out = convert(args.input, args.output, track=args.track, work_dir=args.work_dir,
                      keep_temp=args.keep_temp, progress_cb=show)
    except Cancelled:
        print("\n[vobsub-to-pgs] cancelled", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    print(f"\n[vobsub-to-pgs] done: {out}", file=sys.stderr)
    return 0


# --- self-tests (no real BDSup2Sub++/mkvmerge/mkvextract install assumed -- subprocess.run
# is mocked throughout, matching iso_to_mvc_makemkv_cli.py's own established approach) ---

def _self_test_not_found_raises():
    from unittest import mock
    with mock.patch.object(sys.modules[__name__], "_find_bdsup2sub", return_value=None):
        try:
            convert("in.idx", "out.sup")
            raise AssertionError("expected RuntimeError when BDSup2Sub++ isn't found")
        except RuntimeError as e:
            assert "BDSup2Sub++ not found" in str(e)
    print("_self_test_not_found_raises: PASS")


def _self_test_output_extension_validated():
    from unittest import mock
    with mock.patch.object(sys.modules[__name__], "_find_bdsup2sub", return_value="bdsup2sub++.exe"):
        try:
            convert("in.idx", "out.sub")
            raise AssertionError("expected ValueError for a non-.sup output")
        except ValueError as e:
            assert ".sup" in str(e)
    print("_self_test_output_extension_validated: PASS")


def _self_test_standalone_idx_requires_sub():
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp_dir, \
         mock.patch.object(sys.modules[__name__], "_find_bdsup2sub", return_value="bdsup2sub++.exe"):
        idx = path.join(tmp_dir, "movie.idx")
        open(idx, "wb").close()
        try:
            convert(idx, path.join(tmp_dir, "out.sup"))
            raise AssertionError("expected RuntimeError when the companion .sub is missing")
        except RuntimeError as e:
            assert "no matching .sub file" in str(e)
    print("_self_test_standalone_idx_requires_sub: PASS")


def _self_test_identify_vobsub_tracks_filters_codec():
    from unittest import mock
    payload = json.dumps({"tracks": [
        {"id": 0, "type": "video", "properties": {"codec_id": "V_MPEG4/ISO/AVC"}},
        {"id": 1, "type": "audio", "properties": {"codec_id": "A_AC3"}},
        {"id": 2, "type": "subtitles", "properties": {"codec_id": "S_TEXT/UTF8", "language": "eng"}},
        {"id": 3, "type": "subtitles", "properties": {"codec_id": "S_VOBSUB", "language": "fre"}},
    ]})

    def fake_run(cmd, **kwargs):
        assert cmd[1] == "-J"
        return subprocess.CompletedProcess(cmd, 0, payload, "")

    with mock.patch("subprocess.run", side_effect=fake_run):
        found = _identify_vobsub_tracks("movie.mkv", "mkvmerge.exe")
    assert found == [(3, "fre")], found
    print("_self_test_identify_vobsub_tracks_filters_codec: PASS")


def _self_test_extract_requires_track_when_ambiguous():
    from unittest import mock
    payload = json.dumps({"tracks": [
        {"id": 2, "type": "subtitles", "properties": {"codec_id": "S_VOBSUB", "language": "eng"}},
        {"id": 3, "type": "subtitles", "properties": {"codec_id": "S_VOBSUB", "language": "fre"}},
    ]})

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, payload, "")

    with mock.patch("subprocess.run", side_effect=fake_run):
        try:
            extract_vobsub_from_mkv("movie.mkv", "out.idx", mkvmerge_bin="mkvmerge.exe",
                                    mkvextract_bin="mkvextract.exe")
            raise AssertionError("expected RuntimeError for 2 VobSub tracks with no --track")
        except RuntimeError as e:
            assert "2 VobSub subtitle tracks" in str(e)
    print("_self_test_extract_requires_track_when_ambiguous: PASS")


def _self_test_extract_mocked_end_to_end():
    """No real mkvmerge/mkvextract: subprocess.run is mocked for both the -J identify call
    and the real extraction call (which 'creates' the real .idx/.sub pair the same way
    mkvextract itself would for a VobSub track). Confirms the single VobSub track is
    auto-selected and the real Matroska track ID (3, not an ffprobe subtitle index) is
    passed straight through to mkvextract's own `tracks TID:out` argument."""
    import tempfile
    from unittest import mock
    payload = json.dumps({"tracks": [
        {"id": 0, "type": "video", "properties": {"codec_id": "V_MPEG4/ISO/AVC"}},
        {"id": 3, "type": "subtitles", "properties": {"codec_id": "S_VOBSUB", "language": "eng"}},
    ]})
    extract_calls = []

    def fake_run(cmd, **kwargs):
        if "-J" in cmd:
            return subprocess.CompletedProcess(cmd, 0, payload, "")
        extract_calls.append(cmd)
        spec = cmd[cmd.index("tracks") + 1]
        tid, out_idx = spec.split(":", 1)
        with open(out_idx, "wb") as f:
            f.write(b"fake-idx-data")
        with open(path.splitext(out_idx)[0] + ".sub", "wb") as f:
            f.write(b"fake-sub-data")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with tempfile.TemporaryDirectory() as tmp_dir, mock.patch("subprocess.run", side_effect=fake_run):
        idx_path, sub_path = extract_vobsub_from_mkv(
            "movie.mkv", path.join(tmp_dir, "out.idx"), mkvmerge_bin="mkvmerge.exe",
            mkvextract_bin="mkvextract.exe")
        assert path.exists(idx_path) and path.exists(sub_path)
        spec = extract_calls[0][extract_calls[0].index("tracks") + 1]
        assert spec.startswith("3:"), spec
    print("_self_test_extract_mocked_end_to_end: PASS")


def _self_test_convert_mocked_end_to_end_from_mkv():
    """No real BDSup2Sub++/mkvmerge/mkvextract: every subprocess.run call is mocked (-J
    identify, mkvextract, bdsup2sub++ itself, which 'creates' a fake .sup the same way the
    real tool would). Confirms: the embedded VobSub track is extracted first, the real
    `-o <tmp>.sup <idx>` conversion command is built, the result is atomically moved to
    output_path, progress fires for both stages, and the work directory is cleaned up."""
    import tempfile
    from unittest import mock
    payload = json.dumps({"tracks": [
        {"id": 5, "type": "subtitles", "properties": {"codec_id": "S_VOBSUB", "language": "eng"}},
    ]})
    bdsup2sub_calls = []

    def fake_run(cmd, **kwargs):
        if "-J" in cmd:
            return subprocess.CompletedProcess(cmd, 0, payload, "")
        if "tracks" in cmd:
            spec = cmd[cmd.index("tracks") + 1]
            _, out_idx = spec.split(":", 1)
            open(out_idx, "wb").write(b"fake-idx")
            open(path.splitext(out_idx)[0] + ".sub", "wb").write(b"fake-sub")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        bdsup2sub_calls.append(cmd)
        out_sup = cmd[cmd.index("-o") + 1]
        open(out_sup, "wb").write(b"fake-pgs-data")
        return subprocess.CompletedProcess(cmd, 0, "Loading ...\n", "")

    with tempfile.TemporaryDirectory() as tmp_dir, \
         mock.patch.object(sys.modules[__name__], "_find_bdsup2sub", return_value="bdsup2sub++.exe"), \
         mock.patch("subprocess.run", side_effect=fake_run):
        mkv_path = path.join(tmp_dir, "movie.mkv")
        open(mkv_path, "wb").close()
        out_path = path.join(tmp_dir, "movie.sup")
        progress = []
        result = convert(mkv_path, out_path, progress_cb=lambda *a: progress.append(a))

        assert result == out_path
        assert path.exists(out_path) and open(out_path, "rb").read() == b"fake-pgs-data"
        assert bdsup2sub_calls and bdsup2sub_calls[0][0] == "bdsup2sub++.exe"
        assert bdsup2sub_calls[0][1] == "-o"
        assert not path.exists(path.join(tmp_dir, "_vobsub_pgs_work_movie")), \
            "the work directory must be cleaned up after a successful run"
        assert ("extract", 0, 1) in progress and ("extract", 1, 1) in progress
        assert ("convert", 0, 1) in progress and ("convert", 1, 1) in progress
    print("_self_test_convert_mocked_end_to_end_from_mkv: PASS")


def _self_test_convert_failure_reports_empty_output():
    """Real, confirmed-live behavior (see this module's own docstring): BDSup2Sub++ can
    fail with exit code 1 and ZERO output on both stdout and stderr for a malformed real
    VobSub source. This must still raise a clear RuntimeError, never crash on an
    empty/None string."""
    import tempfile
    from unittest import mock

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "")

    with tempfile.TemporaryDirectory() as tmp_dir, \
         mock.patch.object(sys.modules[__name__], "_find_bdsup2sub", return_value="bdsup2sub++.exe"), \
         mock.patch("subprocess.run", side_effect=fake_run):
        idx = path.join(tmp_dir, "movie.idx")
        open(idx, "wb").close()
        open(path.join(tmp_dir, "movie.sub"), "wb").write(b"x")
        try:
            convert(idx, path.join(tmp_dir, "out.sup"))
            raise AssertionError("expected RuntimeError on a real BDSup2Sub++ failure")
        except RuntimeError as e:
            assert "BDSup2Sub++ failed" in str(e) and "no output" in str(e)
    print("_self_test_convert_failure_reports_empty_output: PASS")


def _self_test_cancel_before_stage_raises():
    from unittest import mock

    class _FakeEvent:
        def is_set(self):
            return True

    with mock.patch.object(sys.modules[__name__], "_find_bdsup2sub", return_value="bdsup2sub++.exe"):
        try:
            convert("in.idx", "out.sup", stop_event=_FakeEvent())
            raise AssertionError("expected Cancelled")
        except Cancelled:
            pass
    print("_self_test_cancel_before_stage_raises: PASS")


def _run_self_tests():
    _self_test_not_found_raises()
    _self_test_output_extension_validated()
    _self_test_standalone_idx_requires_sub()
    _self_test_identify_vobsub_tracks_filters_codec()
    _self_test_extract_requires_track_when_ambiguous()
    _self_test_extract_mocked_end_to_end()
    _self_test_convert_mocked_end_to_end_from_mkv()
    _self_test_convert_failure_reports_empty_output()
    _self_test_cancel_before_stage_raises()
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
