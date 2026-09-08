"""python -m iw3.stereo_mode_tag_cli -- standalone retroactive MKV StereoMode tagging tool.

Real-world use case (see docs/ai/AI_DECISIONS.md ADR-033): a user already has an iw3 3D
.mkv output that was made BEFORE the "Tag MKV as 3D (StereoMode)" setting existed, or
was made with it turned off, and wants to retroactively add the Matroska StereoMode
property so 3D-aware players/TVs (VLC, Kodi, compatible smart TVs) auto-detect the 3D
packing/eye-order -- without re-running the conversion or re-muxing the whole file.

This is a genuinely standalone tool, invoked as its own subprocess (same "separate
subprocess, never sharing state with the main app" convention as iw3.rife_cli /
iw3.reinject_hdr_cli / iw3.subtitle_mux_cli -- see docs/ai/CODING_STANDARDS.md
CS-SUBPROCESS-001), even though it needs no GPU at all.

Unlike the other standalone tools in this family (reinject_hdr_cli, subtitle_mux_cli),
which always write a NEW output file and never touch their input, this tool edits
--input IN PLACE via mkvpropedit. That is deliberate, not a shortcut: mkvpropedit is
specifically a container-metadata editor -- it rewrites only the small metadata region
of the file, never re-encodes or re-muxes the actual audio/video streams, so an in-place
edit is both the correct and the fast way to use it (a copy-then-edit-then-replace
dance here would defeat the entire reason to prefer mkvpropedit over mkvmerge for this
job -- see the task's own framing and _apply_stereo_mode_tag's docstring in
iw3/utils.py). An optional --backup flag is provided for users who want a safety copy
anyway; the default fast in-place path is unaffected either way.

Scope (mirrors the main pipeline's own rules -- see ADR-033):
- .mkv input only. If --input is not a .mkv file, this tool refuses outright.
- --format must resolve to a concrete value before any edit is attempted. In --format
  auto (the default), detection reuses iw3.utils's own canonical filename-suffix
  constants -- the exact tags iw3 itself writes into output filenames -- checked
  longest-tag-first so e.g. the full-SBS tag isn't masked by the half-SBS tag that is
  a substring of it. If no tag matches, the tool refuses and asks for --format to be
  passed explicitly -- it never guesses.
- Only half_sbs/full_sbs/full_tb/half_tb/cross_eyed/vr90 are real two-eye stereo pairs
  Matroska's StereoMode can describe. rgbd/half_rgbd/anaglyph are valid --format values
  (so auto-detection or an explicit choice can name them) but always produce a clear
  refusal message rather than a meaningless tag -- see _resolve_stereo_mode_value in
  iw3/utils.py for why (RGB-D/Half RGB-D store a depth channel, not a second eye view;
  Anaglyph is already a single merged image correct on any player without tagging).

Eye-order mapping is NOT re-derived here -- it reuses iw3.utils._resolve_stereo_mode_value
and _apply_stereo_mode_tag directly (via a minimal shim args object carrying just the
one resolved format's flags), the same single source of truth the main conversion
pipeline uses, so this tool can never drift out of sync with it.
"""
import argparse
import shutil
import sys
from os import path

from .utils import (
    _find_mkvpropedit, _resolve_stereo_mode_value, _apply_stereo_mode_tag,
    FULL_SBS_SUFFIX, HALF_SBS_SUFFIX, FULL_TB_SUFFIX, HALF_TB_SUFFIX,
    CROSS_EYED_SUFFIX, RGBD_SUFFIX, HALF_RGBD_SUFFIX, VR180_SUFFIX, ANAGLYPH_SUFFIX,
)


# Same canonical suffix-tag table as iw3.subtitle_mux_cli's own _SUFFIX_TO_FORMAT --
# duplicated rather than imported cross-module (these standalone CLI tools each import
# only from iw3.utils, never from each other) but built from the exact same iw3.utils
# constants, so it can never silently drift from what iw3 itself actually writes.
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

# Only these formats are a real two-eye stereo pair Matroska's StereoMode can describe
# (see _resolve_stereo_mode_value in iw3/utils.py). rgbd/half_rgbd/anaglyph are valid
# --format values so they can be named (by auto-detection or explicitly) but always
# lead to a clear refusal rather than a meaningless tag.
_TAGGABLE_FORMATS = frozenset({"full_sbs", "half_sbs", "full_tb", "half_tb", "cross_eyed", "vr90"})

# Minimal iw3.utils args-flag shims for each taggable format -- passed to
# _resolve_stereo_mode_value / _apply_stereo_mode_tag as the single source of truth
# for the eye-order mapping (see module docstring). full_sbs/half_sbs/vr90 all resolve
# via the same unflagged default branch, so they need no flags of their own (vr90 sets
# vr180=True purely so _apply_stereo_mode_tag can print its VR-specific limitation note).
_FORMAT_TO_ARGS_FLAGS = {
    "full_sbs": {},
    "half_sbs": {},
    "vr90": {"vr180": True},
    "full_tb": {"tb": True},
    "half_tb": {"half_tb": True},
    "cross_eyed": {"cross_eyed": True},
    "rgbd": {"rgbd": True},
    "half_rgbd": {"half_rgbd": True},
    "anaglyph": {"anaglyph": "dubois"},
}


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.stereo_mode_tag_cli",
        description=(
            "Retroactively tag an already-converted iw3 .mkv 3D output with the Matroska "
            "StereoMode property (via mkvpropedit), so 3D-aware players/TVs auto-detect the "
            "3D packing/eye-order. Edits --input IN PLACE -- see module docstring for why "
            "that is correct here, unlike this tool family's other members."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                         help="Path to the already-converted 3D video. Must be a .mkv file -- "
                              "this tool does not convert containers. Edited in place.")
    parser.add_argument("--format", type=str, default="auto", choices=list(_VALID_FORMATS),
                         help="Stereo/output layout of --input. 'auto' (default) tries to detect "
                              "this from --input's filename using the same suffix tags iw3 itself "
                              "writes (e.g. '_LR', '_TB', '_LRF_Full_SBS', '_TBF_fulltb', "
                              "'_RLF_cross', '_180x180_LR', '_RGBD', '_HRGBD', '_redcyan'). If "
                              "detection is inconclusive, this tool refuses and asks you to pass "
                              "this explicitly -- it never guesses. Note: rgbd/half_rgbd/anaglyph "
                              "are accepted values (so detection/explicit choice can name them) but "
                              "always refuse to tag -- see --help output above and "
                              "docs/ai/AI_DECISIONS.md ADR-033 for why.")
    parser.add_argument("--backup", action="store_true",
                         help="Copy --input to '<input>.bak' before editing, as an extra safety "
                              "net. Off by default -- mkvpropedit's in-place edit only rewrites "
                              "container metadata (never re-encodes/re-muxes the actual streams), "
                              "so this is optional insurance, not a correctness requirement.")
    return parser


def _detect_format_from_filename(filename):
    """Returns one of the values in _SUFFIX_TO_FORMAT if a known iw3 filename suffix tag
    is found (case-insensitive, longest tag checked first so e.g. the full-SBS tag isn't
    masked by the half-SBS tag that is a substring of it), or None if inconclusive."""
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
            f"ERROR: --format auto could not confidently determine the stereo layout of "
            f"'{input_path}' from its filename -- none of the known iw3 filename tags "
            f"({', '.join(sorted(_SUFFIX_TO_FORMAT.keys()))}) were found.\n"
            f"Please re-run with --format explicitly set to one of: "
            f"{', '.join(f for f in _VALID_FORMATS if f != 'auto')}.")
    return detected, None


def run(args):
    input_path = str(args.input)

    if not path.exists(input_path):
        print(f"ERROR: --input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if path.splitext(input_path)[1].lower() != ".mkv":
        print(
            f"ERROR: --input must be an .mkv file (got '{input_path}'). StereoMode is a "
            f"Matroska-only property -- there is nothing to tag on a non-MKV container.",
            file=sys.stderr)
        return 1

    resolved_format, format_error = _resolve_format(args.format, input_path)
    if format_error:
        print(format_error, file=sys.stderr)
        return 1
    print(f"[stereo-mode-tag] resolved stereo format: {resolved_format}"
          f"{' (auto-detected from filename)' if args.format == 'auto' else ' (explicit)'}",
          file=sys.stderr)

    if resolved_format not in _TAGGABLE_FORMATS:
        print(
            f"ERROR: '{resolved_format}' is not a two-eye stereo pair Matroska's StereoMode can "
            f"describe -- RGB-D/Half RGB-D store a depth channel (not a second eye view), and "
            f"Anaglyph is already a single merged image that displays correctly on any player "
            f"without tagging. Nothing was changed in '{input_path}'.",
            file=sys.stderr)
        return 1

    mkvpropedit_bin = _find_mkvpropedit()
    if not mkvpropedit_bin:
        print("ERROR: mkvpropedit (MKVToolNix) was not found -- cannot tag. Expected it bundled "
              "alongside mkvmerge, or on PATH.", file=sys.stderr)
        return 1

    if args.backup:
        backup_path = input_path + ".bak"
        print(f"[stereo-mode-tag] --backup: copying to {backup_path} before editing...",
              file=sys.stderr)
        shutil.copyfile(input_path, backup_path)

    # Single source of truth: reuse iw3.utils's own eye-order mapping and mkvpropedit
    # invocation via a minimal shim args object carrying just this one resolved format's
    # flags, rather than re-deriving the mapping here (see module docstring).
    shim_args = argparse.Namespace(stereo_mode_tag=True, **_FORMAT_TO_ARGS_FLAGS[resolved_format])
    stereo_value = _resolve_stereo_mode_value(shim_args)
    assert stereo_value is not None, resolved_format  # guaranteed by the _TAGGABLE_FORMATS check above

    ok = _apply_stereo_mode_tag(input_path, shim_args, mkvpropedit_bin=mkvpropedit_bin)
    if not ok:
        print(f"ERROR: StereoMode tagging failed on '{input_path}' -- see the message above for "
              f"why.", file=sys.stderr)
        return 1

    print(f"[stereo-mode-tag] done -- tagged {input_path} in place.", file=sys.stderr)
    return 0


def _self_test_format_detection():
    """Synthetic filename-only test of _detect_format_from_filename / _resolve_format --
    no GPU or real video needed."""
    cases = {
        "movie_LR.mkv": "half_sbs",
        "movie_LRF_Full_SBS.mkv": "full_sbs",
        "movie_TB.mkv": "half_tb",
        "movie_TBF_fulltb.mkv": "full_tb",
        "movie_RLF_cross.mkv": "cross_eyed",
        "movie_180x180_LR.mkv": "vr90",
        "movie_RGBD.mkv": "rgbd",
        "movie_HRGBD.mkv": "half_rgbd",
        "movie_redcyan_dubois2.mkv": "anaglyph",
        "MOVIE_lr.MKV": "half_sbs",  # case-insensitive
    }
    for filename, expected in cases.items():
        got = _detect_format_from_filename(filename)
        assert got == expected, f"{filename}: expected {expected}, got {got}"

    assert _detect_format_from_filename("movie_converted.mkv") is None

    resolved, err = _resolve_format("auto", "movie_LR.mkv")
    assert resolved == "half_sbs" and err is None

    resolved, err = _resolve_format("auto", "movie_converted.mkv")
    assert resolved is None and err is not None and "explicitly" in err

    resolved, err = _resolve_format("full_tb", "movie_converted.mkv")
    assert resolved == "full_tb" and err is None

    print("_self_test_format_detection: PASS")


def _self_test_stereo_value_mapping():
    """Confirms every taggable format resolves to the expected StereoMode value via the
    SAME shim-args path run() actually uses -- proves this CLI can never drift from
    iw3.utils._resolve_stereo_mode_value's single source of truth."""
    expected = {
        "full_sbs": 1, "half_sbs": 1, "vr90": 1,
        "full_tb": 3, "half_tb": 3,
        "cross_eyed": 11,
    }
    for fmt, expected_value in expected.items():
        shim_args = argparse.Namespace(stereo_mode_tag=True, **_FORMAT_TO_ARGS_FLAGS[fmt])
        got = _resolve_stereo_mode_value(shim_args)
        assert got == expected_value, f"{fmt}: expected {expected_value}, got {got}"

    for fmt in ("rgbd", "half_rgbd", "anaglyph"):
        assert fmt not in _TAGGABLE_FORMATS, fmt
        shim_args = argparse.Namespace(stereo_mode_tag=True, **_FORMAT_TO_ARGS_FLAGS[fmt])
        assert _resolve_stereo_mode_value(shim_args) is None, fmt

    print("_self_test_stereo_value_mapping: PASS")


def _self_test_run_gating():
    """Synthetic/mocked test of run()'s pre-mkvpropedit gates (missing input, non-mkv
    input, inconclusive auto-format, non-taggable format) -- proves each refuses before
    mkvpropedit is ever invoked."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_stereotag_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie_LR.mkv")
        mp4_input = path.join(tmpdir, "movie_LR.mp4")
        ambiguous_input = path.join(tmpdir, "movie.mkv")
        rgbd_input = path.join(tmpdir, "movie_RGBD.mkv")
        missing_input = path.join(tmpdir, "does_not_exist.mkv")

        for p in (mkv_input, mp4_input, ambiguous_input, rgbd_input):
            with open(p, "wb") as f:
                f.write(b"0")

        def _args(**overrides):
            base = dict(input=mkv_input, format="auto", backup=False)
            base.update(overrides)
            return argparse.Namespace(**base)

        with patch(f"{__name__}._find_mkvpropedit") as mock_find:
            # missing input -> refuse before ever looking for mkvpropedit
            rc = run(_args(input=missing_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # non-.mkv input -> refuse before ever looking for mkvpropedit
            rc = run(_args(input=mp4_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # ambiguous filename with format=auto -> refuse before mkvpropedit lookup
            rc = run(_args(input=ambiguous_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # RGB-D is a valid format value but never taggable -> refuse with a clear
            # message, before mkvpropedit lookup
            rc = run(_args(input=rgbd_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

        # explicit taggable --format bypasses filename ambiguity and reaches mkvpropedit
        with patch(f"{__name__}._find_mkvpropedit", return_value=None):
            rc = run(_args(input=ambiguous_input, format="full_tb"))
            assert rc == 1, rc  # refuses because mkvpropedit isn't found, but got past format gate

        # happy path: mkvpropedit found and the underlying apply succeeds
        with patch(f"{__name__}._find_mkvpropedit", return_value="mkvpropedit.exe"), \
             patch(f"{__name__}._apply_stereo_mode_tag", return_value=True) as mock_apply:
            rc = run(_args(input=mkv_input))
            assert rc == 0, rc
            mock_apply.assert_called_once()
            call_args = mock_apply.call_args
            assert call_args[0][0] == mkv_input
            assert call_args[1]["mkvpropedit_bin"] == "mkvpropedit.exe"

    print("_self_test_run_gating: PASS")


def _run_self_tests():
    _self_test_format_detection()
    _self_test_stereo_value_mapping()
    _self_test_run_gating()
    print("All stereo_mode_tag_cli self-tests PASSED")


def main(argv=None):
    # Special-cased ahead of the real parser (rather than added as a parser argument) so
    # it can run without also satisfying --input=required -- see docs/ai/TEST_MATRIX.md's
    # "isolated (no-GPU) test pattern" convention.
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
