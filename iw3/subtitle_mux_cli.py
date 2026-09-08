"""python -m iw3.subtitle_mux_cli -- standalone tool to add an SRT subtitle file as a
NEW track to an already-converted 3D video, preserving every existing track (video,
audio, existing subtitles) untouched.

Real-world use case: a user has already run an iw3 conversion (SBS or TB) and later
obtains/creates an SRT subtitle file for it, and wants to add it as a normal soft
subtitle track without re-muxing by hand or disturbing anything already in the file.

IMPORTANT -- why this is a PLAIN soft-subtitle mux, not a stereo-baked one (do not
"improve" this without re-reading this note): this project's own iw3-player
(iw3/player/media_library.py's extract_subtitle()/get_subtitles(), and
iw3/player/public/js/window_subtitle.js's SubtitleWindow) already does CLIENT-SIDE,
playback-time stereo rendering of plain subtitle text -- it extracts only the plain
text of each cue (discarding any ASS position/style overrides) and draws it
independently on two separate render layers (left eye / right eye) with its own
configurable position/depth/scale. If this tool pre-baked stereo-duplicated/
positioned cues into the file (e.g. a fancy ASS track with left-half/right-half
copies), it would BREAK iw3-player's own playback -- the player would re-duplicate
already-duplicated text, producing doubled, wrongly-positioned captions. So this tool
muxes the SRT in as an ordinary, single soft-subtitle track, which is correct both for
iw3-player (works natively) and for any other normal media player.

Scope (deliberately narrow, see docs/ai/AI_DECISIONS.md ADR-032):
- MKV output only. If --input is not a .mkv file, this tool refuses -- no format
  conversion is attempted.
- Soft subtitle track only. No burned-in/hardcoded subtitle mode.
- One SRT per run. No multi-language/multi-file support.
- --format must resolve to a concrete value before proceeding, covering every iw3
  Stereo Format that's an actual watchable video layout: half_sbs, full_sbs, half_tb,
  full_tb, cross_eyed, vr90, rgbd, half_rgbd, anaglyph (Export/Export-disparity/
  Debug-Depth are intentionally excluded -- those are data-export formats, not
  something anyone adds a subtitle track to). In --format auto (the default),
  filename-tag detection is attempted using the exact same suffix tags
  iw3.utils.make_output_filename() writes (FULL_SBS_SUFFIX/HALF_SBS_SUFFIX/
  FULL_TB_SUFFIX/HALF_TB_SUFFIX/CROSS_EYED_SUFFIX/VR180_SUFFIX/RGBD_SUFFIX/
  HALF_RGBD_SUFFIX/ANAGLYPH_SUFFIX) -- the same tags iw3/player/stereo_detector.py's
  detect_stereo_format() parses back out of a filename. Since the mux itself doesn't
  care which of these layouts the video actually is (see above), the format value is
  purely a safety/informational resolution step, not something that changes behavior.
  If detection is inconclusive, this tool HARD-REFUSES and asks the user to pass
  --format explicitly, rather than guessing.

Note on iw3.player: this module deliberately does NOT import anything from iw3.player
to reuse its filename-tag matching, even though iw3/player/stereo_detector.py has a
similar TAG_MAP -- iw3/player has no __init__.py and its modules (e.g.
media_library.py) import fastapi directly, which is an unnecessary heavy dependency
for a lightweight muxing CLI. Instead this module duplicates the matching logic
needed against iw3.utils's own canonical suffix constants, which are already the
source of truth iw3 itself writes into output filenames.
"""
import argparse
import os
import shutil
import subprocess
import sys
from os import path

from .utils import (
    _find_mkvmerge, FULL_SBS_SUFFIX, HALF_SBS_SUFFIX, FULL_TB_SUFFIX, HALF_TB_SUFFIX,
    CROSS_EYED_SUFFIX, RGBD_SUFFIX, HALF_RGBD_SUFFIX, VR180_SUFFIX, ANAGLYPH_SUFFIX,
)


# Maps each iw3.utils canonical filename suffix tag to the --format value a user would
# pass for that same layout. Checked longest-tag-first (see _detect_format_from_filename)
# so e.g. FULL_SBS_SUFFIX ("_LRF_Full_SBS") is matched before HALF_SBS_SUFFIX ("_LR"),
# which is a substring of it. Covers every iw3 Stereo Format that's an actual watchable
# video layout -- deliberately excludes Export/Export disparity/Debug Depth, which are
# data-export/debug formats, not something anyone adds subtitles to for viewing.
# ANAGLYPH_SUFFIX has a per-color-recipe suffix appended after it at render time (e.g.
# "_redcyan_dubois2") -- a plain "in name" substring check still matches those fine.
_SUFFIX_TO_FORMAT = {
    FULL_SBS_SUFFIX: "full_sbs",
    HALF_SBS_SUFFIX: "half_sbs",
    FULL_TB_SUFFIX: "full_tb",
    HALF_TB_SUFFIX: "half_tb",
    CROSS_EYED_SUFFIX: "cross_eyed",
    VR180_SUFFIX: "vr90",
    RGBD_SUFFIX: "rgbd",
    HALF_RGBD_SUFFIX: "half_rgbd",
    ANAGLYPH_SUFFIX: "anaglyph",
}
_SORTED_SUFFIXES = sorted(_SUFFIX_TO_FORMAT.keys(), key=len, reverse=True)

_VALID_FORMATS = ("auto",) + tuple(dict.fromkeys(_SUFFIX_TO_FORMAT.values()))


def _format_cmd(cmd):
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.subtitle_mux_cli",
        description=(
            "Add an SRT subtitle file as a new soft-subtitle track to an already-converted "
            "3D (SBS/TB) MKV video, preserving every existing track (video, audio, existing "
            "subtitles) untouched. Never modifies --input -- always writes a new file at "
            "--output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                         help="Path to the already-converted 3D video. Must be a .mkv file -- "
                              "this tool does not convert containers. Read-only, never modified.")
    parser.add_argument("--srt", "-s", type=str, required=True,
                         help="Path to the .srt subtitle file to add as a new track.")
    parser.add_argument("--output", "-o", type=str, required=True,
                         help="Path to write the new file to. Must be a different path from "
                              "--input -- this tool never overwrites the input.")
    parser.add_argument("--language", type=str, default="eng",
                         help="ISO 639-2 language code for the new subtitle track (e.g. eng, "
                              "jpn, fre). Purely metadata -- does not affect muxing.")
    parser.add_argument("--track-name", type=str, default=None,
                         help="Display name for the new subtitle track, shown in player track "
                              "menus. Default: the SRT file's own name (without extension).")
    parser.add_argument("--format", type=str, default="auto", choices=list(_VALID_FORMATS),
                         help="Stereo/output layout of --input -- any of iw3's own watchable Stereo "
                              "Format outputs: half_sbs, full_sbs, half_tb, full_tb, cross_eyed, "
                              "vr90, rgbd, half_rgbd, anaglyph. (Export/Export-disparity/Debug-Depth "
                              "outputs are intentionally not covered here -- those are data-export "
                              "formats, not something you'd add a subtitle track to.) "
                              "'auto' (default) tries to detect this from --input's filename using "
                              "the same suffix tags iw3 itself writes (e.g. '_LR', '_TB', "
                              "'_LRF_Full_SBS', '_TBF_fulltb', '_RLF_cross', '_180x180_LR', '_RGBD', "
                              "'_HRGBD', '_redcyan'). If detection is inconclusive, this "
                              "tool refuses and asks you to pass this explicitly -- it never "
                              "guesses. (Note: this value is not currently used to alter the mux "
                              "itself -- a plain soft-subtitle track is geometry-agnostic regardless "
                              "of which of these layouts the video uses -- but resolving it to a "
                              "concrete value up front is a required safety check per this tool's "
                              "design.)")
    return parser


def _detect_format_from_filename(filename):
    """Returns one of the values in _SUFFIX_TO_FORMAT if a known iw3 filename
    suffix tag is found (case-insensitive, longest tag checked first so e.g. the full-SBS
    tag isn't masked by the half-SBS tag that is a substring of it, and VR90's own tag --
    which itself ends in the half-SBS tag -- is matched first for the same reason), or None
    if inconclusive."""
    name = path.basename(str(filename)).lower()
    for suffix in _SORTED_SUFFIXES:
        if suffix.lower() in name:
            return _SUFFIX_TO_FORMAT[suffix]
    return None


def _resolve_format(requested_format, input_path):
    """Returns (resolved_format_or_None, error_message_or_None)."""
    if requested_format != "auto":
        return requested_format, None

    detected = _detect_format_from_filename(input_path)
    if detected is None:
        return None, (
            f"ERROR: --format auto could not confidently determine the stereo layout (SBS or TB) "
            f"of '{input_path}' from its filename -- none of the known iw3 filename tags "
            f"({', '.join(sorted(_SUFFIX_TO_FORMAT.keys()))}) were found.\n"
            f"Please re-run with --format explicitly set to one of: "
            f"{', '.join(f for f in _VALID_FORMATS if f != 'auto')}.")
    return detected, None


def _validate_srt(srt_path):
    """Loads --srt with pysubs2 so a malformed SRT produces a clear Python-level error
    here rather than an opaque mkvmerge failure later. Returns an error message string,
    or None if the file loaded and looks valid."""
    if not path.exists(srt_path):
        return f"ERROR: --srt file does not exist: {srt_path}"
    try:
        import pysubs2
        subs = pysubs2.load(srt_path)
    except Exception as e:
        return f"ERROR: failed to parse --srt as a subtitle file ('{srt_path}'): {e}"
    if len(subs) == 0:
        return f"ERROR: --srt file '{srt_path}' parsed successfully but contains no subtitle events."
    return None


def run(args):
    input_path = str(args.input)
    srt_path = str(args.srt)
    output_path = str(args.output)

    if not path.exists(input_path):
        print(f"ERROR: --input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if path.splitext(input_path)[1].lower() != ".mkv":
        print(
            f"ERROR: --input must be an .mkv file (got '{input_path}'). This tool only supports "
            f"MKV output/input -- it does not convert containers. Re-run with an .mkv input, or "
            f"remux your existing file to .mkv first (e.g. with mkvmerge) before using this tool.",
            file=sys.stderr)
        return 1

    if path.abspath(output_path) == path.abspath(input_path):
        print("ERROR: --output must be a different path from --input -- this tool never "
              "overwrites the input.", file=sys.stderr)
        return 1

    srt_error = _validate_srt(srt_path)
    if srt_error:
        print(srt_error, file=sys.stderr)
        return 1

    resolved_format, format_error = _resolve_format(args.format, input_path)
    if format_error:
        print(format_error, file=sys.stderr)
        return 1
    print(f"[subtitle-mux] resolved stereo format: {resolved_format}"
          f"{' (auto-detected from filename)' if args.format == 'auto' else ' (explicit)'}",
          file=sys.stderr)

    mkvmerge_bin = _find_mkvmerge()
    if not mkvmerge_bin:
        print("ERROR: mkvmerge (MKVToolNix) was not found -- cannot mux. Expected it bundled "
              "alongside ffmpeg, or on PATH.", file=sys.stderr)
        return 1

    track_name = args.track_name or path.splitext(path.basename(srt_path))[0]

    out_dir = path.dirname(path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output = path.splitext(output_path)[0] + ".submux_tmp" + path.splitext(output_path)[1]

    # mkvmerge: per-file options (--language/--track-name) must appear BEFORE the file
    # they apply to. "0:" addresses track 0 OF THAT FILE (an .srt has exactly one track).
    # <input_path> is given last with no options preceding it, which is mkvmerge's default
    # "copy every track from this file, unmodified" behavior -- this is what preserves the
    # original video/audio/existing-subtitle tracks untouched. Verified against real
    # mkvmerge behavior in this tool's own smoke test (see docs/ai/AI_DECISIONS.md ADR-032).
    cmd = [
        mkvmerge_bin, "-o", tmp_output,
        "--language", f"0:{args.language}",
        "--track-name", f"0:{track_name}",
        srt_path,
        input_path,
    ]
    print(f"[subtitle-mux] running: {_format_cmd(cmd)}", file=sys.stderr)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        print(f"ERROR: failed to run mkvmerge: {e}", file=sys.stderr)
        return 1

    if proc.stdout:
        print(proc.stdout, file=sys.stderr)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    # mkvmerge exit codes: 0 = success, 1 = success with warnings, 2+ = failure.
    if proc.returncode >= 2:
        print(f"ERROR: mkvmerge failed (exit code {proc.returncode}).", file=sys.stderr)
        if path.exists(tmp_output):
            try:
                os.remove(tmp_output)
            except Exception:
                pass
        return 1

    if not path.exists(tmp_output):
        print("ERROR: mkvmerge reported success but produced no output file.", file=sys.stderr)
        return 1

    os.replace(tmp_output, output_path)
    print(f"[subtitle-mux] done -- wrote {output_path}", file=sys.stderr)
    return 0


def _self_test_format_detection():
    """Synthetic filename-only test of _detect_format_from_filename / _resolve_format --
    no GPU or real video needed."""
    cases = {
        "movie_LR.mkv": "half_sbs",
        "movie_LRF_Full_SBS.mkv": "full_sbs",
        "movie_TB.mkv": "half_tb",
        "movie_TBF_fulltb.mkv": "full_tb",
        "MOVIE_lr.MKV": "half_sbs",  # case-insensitive
    }
    for filename, expected in cases.items():
        got = _detect_format_from_filename(filename)
        assert got == expected, f"{filename}: expected {expected}, got {got}"

    # Ambiguous / unmatched filename -> None (must trigger the hard-refuse path)
    assert _detect_format_from_filename("movie_converted.mkv") is None

    resolved, err = _resolve_format("auto", "movie_LR.mkv")
    assert resolved == "half_sbs" and err is None

    resolved, err = _resolve_format("auto", "movie_converted.mkv")
    assert resolved is None and err is not None and "explicitly" in err

    resolved, err = _resolve_format("full_tb", "movie_converted.mkv")
    assert resolved == "full_tb" and err is None

    print("_self_test_format_detection: PASS")


def _self_test_srt_validation():
    """Synthetic valid/malformed SRT test using pysubs2 -- no real movie footage needed."""
    import tempfile

    valid_srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nSecond line\n\n"
    )
    malformed_srt = "this is not a valid subtitle file at all\nno timestamps, no structure\n"

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        valid_path = path.join(tmpdir, "valid.srt")
        malformed_path = path.join(tmpdir, "malformed.srt")
        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(valid_srt)
        with open(malformed_path, "w", encoding="utf-8") as f:
            f.write(malformed_srt)

        assert _validate_srt(valid_path) is None
        assert _validate_srt(malformed_path) is not None
        assert _validate_srt(path.join(tmpdir, "does_not_exist.srt")) is not None

    print("_self_test_srt_validation: PASS")


def _self_test_run_gating():
    """Synthetic/mocked test of run()'s pre-mkvmerge gates (non-mkv input, output==input,
    invalid SRT, inconclusive auto-format) -- proves each refuses before mkvmerge is ever
    invoked."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie_LR.mkv")
        mp4_input = path.join(tmpdir, "movie_LR.mp4")
        ambiguous_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        output_path = path.join(tmpdir, "out.mkv")

        for p in (mkv_input, mp4_input, ambiguous_input):
            with open(p, "wb") as f:
                f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        def _args(**overrides):
            base = dict(input=mkv_input, srt=srt_path, output=output_path,
                        language="eng", track_name=None, format="auto")
            base.update(overrides)
            return argparse.Namespace(**base)

        with patch(f"{__name__}._find_mkvmerge") as mock_find:
            # non-.mkv input -> refuse before ever looking for mkvmerge
            rc = run(_args(input=mp4_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # output == input -> refuse
            rc = run(_args(output=mkv_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # malformed SRT -> refuse before mkvmerge lookup
            bad_srt = path.join(tmpdir, "bad.srt")
            with open(bad_srt, "w", encoding="utf-8") as f:
                f.write("not a subtitle file")
            rc = run(_args(srt=bad_srt))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # ambiguous filename with format=auto -> refuse before mkvmerge lookup
            rc = run(_args(input=ambiguous_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

        # explicit --format bypasses filename ambiguity and reaches the mkvmerge stage
        with patch(f"{__name__}._find_mkvmerge", return_value=None):
            rc = run(_args(input=ambiguous_input, format="full_tb"))
            assert rc == 1, rc  # refuses because mkvmerge isn't found, but got past format gate

    print("_self_test_run_gating: PASS")


def _run_self_tests():
    _self_test_format_detection()
    _self_test_srt_validation()
    _self_test_run_gating()
    print("All subtitle_mux_cli self-tests PASSED")


def main(argv=None):
    # Special-cased ahead of the real parser (rather than added as a parser argument) so
    # it can run without also satisfying --input/--srt/--output=required -- see
    # docs/ai/TEST_MATRIX.md's "isolated (no-GPU) test pattern" convention.
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
