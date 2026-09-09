"""python -m iw3.subtitle_mux_cli -- standalone tool to add an SRT subtitle file as a
NEW track to an already-converted 3D video, preserving every existing track (video,
audio, existing subtitles) untouched.

Real-world use case: a user has already run an iw3 conversion (SBS or TB) and later
obtains/creates an SRT subtitle file for it, and wants to add it as a normal soft
subtitle track without re-muxing by hand or disturbing anything already in the file.

IMPORTANT -- why the DEFAULT is a PLAIN soft-subtitle mux, not a stereo-baked one (do
not "improve" this without re-reading this note, and see ADR-053 before changing the
--dual-eye-subtitles handling below): this project's own iw3-player
(iw3/player/media_library.py's extract_subtitle()/get_subtitles(), and
iw3/player/public/js/window_subtitle.js's SubtitleWindow) already does CLIENT-SIDE,
playback-time stereo rendering of plain subtitle text -- it extracts only the plain
text of each cue (discarding any ASS position/style overrides) and draws it
independently on two separate render layers (left eye / right eye) with its own
configurable position/depth/scale. If this tool pre-baked stereo-duplicated/
positioned cues into the file by default (e.g. a fancy ASS track with left-half/
right-half copies), it would BREAK iw3-player's own playback -- the player would
re-duplicate already-duplicated text, producing doubled, wrongly-positioned captions.
So the DEFAULT (no flag) behavior stays an ordinary, single soft-subtitle track for
every --format, split-eye or not -- byte-for-byte the same ADR-032 output this tool
has always produced. ADR-053 does NOT change this default.

ADR-032 also originally claimed that plain-track default was "correct for any other
normal media player" too -- that part turned out to be wrong specifically for
split-eye layouts played in an external (non-iw3-player) app. A standard player (VLC
etc.) has no idea the video frame is actually two eye-views packed side-by-side or
top-and-bottom -- it centers a plain subtitle against the FULL frame, which lands
text dead-center on the seam between the two eyes on a genuine split-eye layout
(half_sbs/full_sbs/cross_eyed/vr90 horizontally, half_tb/full_tb vertically), tearing
every line of text in half. ADR-053 adds an explicit, OPT-IN `--dual-eye-subtitles`
flag (default False) for exactly this case: when set (and only when set, and only for
a genuine split-eye --format), --srt is converted into an ASS track with each cue
duplicated into two positioned copies, one centered in each eye-half -- see
_build_stereo_positioned_subs(). Leaving the flag off preserves the original
behavior for iw3-player users and anyone not asking for this. rgbd/half_rgbd (a
color|depth-map split, not two eye-views -- see apply_rgbd() in iw3/utils.py) and
anaglyph (already a single merged 2D frame) are unaffected by the flag either way --
it's a no-op for those formats since there's no seam to fix.

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

ADR-054 -- optional --start-time/--end-time trim+rebase (purely additive, does not touch
anything above): real-world use case -- a user converts only a trimmed CLIP of a full
movie (e.g. `python -m iw3 ... --start-time 00:01:15 --end-time 00:06:15`, using iw3's
own --start-time/--end-time -- see iw3/utils.py's create_parser() and parse_time() in
nunif.utils.video.metadata, the exact same flag names/parsing this tool now reuses), but
--srt (e.g. downloaded via this same GUI's Search Subtitles tool above) has timestamps
for the FULL movie. When --start-time/--end-time are given here, cues are trimmed to
those overlapping the window and REBASED (shifted so the window's start becomes 0) before
muxing -- see _trim_and_rebase_subs(). Omitting both (the default) is byte-for-byte
unchanged from pre-ADR-054 output; the rebase happens on the already-loaded `subs` before
any --dual-eye-subtitles positioning, so both features compose correctly together.

ADR-055 -- default Fontsize scaled to the real probed per-eye video height, plus an
optional --font-size override (both purely additive, only affect the --dual-eye-subtitles
ASS conversion path -- see _build_stereo_positioned_subs()): real user report -- "the
subtitles are very small" -- traced to _build_stereo_positioned_subs() previously leaving
the ASS style's Fontsize at pysubs2's own SSAStyle default (20px, see
pysubs2/ssastyle.py), a value never designed against this project's own output
resolutions. ADR-053 already correctly sets PlayResX/PlayResY to the REAL probed frame
size (e.g. 3840x2076 for a double-wide 4K SBS conversion) instead of leaving it at
libass's small implicit default (384x288) -- but leaving Fontsize itself untouched meant
20px against a real PlayResY of 2076 rendered as under 1% of frame height, effectively
invisible. The fix: when --font-size is not given, Fontsize now defaults to ~4.5% of the
per-eye vertical resolution (_default_font_size()/_eye_height()) -- unchanged (full
PlayResY) for a horizontal split (SBS-family), HALVED (PlayResY / 2) for a vertical split
(TB-family), mirroring the same per-eye halving _build_stereo_positioned_subs() already
applies to its own y-position math for TB formats, so a TB conversion's text isn't sized
for a full frame it's only ever shown in half of. --font-size (a plain float, no
auto-scaling) lets a user pin an exact pixel size instead. Both are no-ops outside the
--dual-eye-subtitles ASS path -- the plain SRT track this tool mostly still produces by
default has no Fontsize field to control at all.

Note on iw3.player: this module deliberately does NOT import anything from iw3.player
to reuse its filename-tag matching, even though iw3/player/stereo_detector.py has a
similar TAG_MAP -- iw3/player has no __init__.py and its modules (e.g.
media_library.py) import fastapi directly, which is an unnecessary heavy dependency
for a lightweight muxing CLI. Instead this module duplicates the matching logic
needed against iw3.utils's own canonical suffix constants, which are already the
source of truth iw3 itself writes into output filenames.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from os import path

from nunif.utils.video.metadata import parse_time
from .utils import (
    _find_mkvmerge, _find_ffprobe, FULL_SBS_SUFFIX, HALF_SBS_SUFFIX, FULL_TB_SUFFIX,
    HALF_TB_SUFFIX, CROSS_EYED_SUFFIX, RGBD_SUFFIX, HALF_RGBD_SUFFIX, VR180_SUFFIX,
    ANAGLYPH_SUFFIX,
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


# Resolved --format values whose watchable frame is a genuine split-eye layout, and the
# axis it's split on -- see ADR-053. half_sbs/full_sbs/cross_eyed/vr90 all pack left-eye
# and right-eye side by side (HORIZONTAL split, seam at width/2); half_tb/full_tb stack
# them top over bottom (VERTICAL split, seam at height/2). Confirmed against how iw3
# actually assembles each of these frames before relying on the name alone.
#
# rgbd/half_rgbd and anaglyph are deliberately NOT in either set, so they fall through to
# the original plain-track behavior, which is already correct for them:
# - rgbd/half_rgbd: iw3.utils.apply_rgbd() shows the "right" half is the DEPTH MAP, not a
#   second eye view of the movie -- duplicating subtitle text onto a depth map isn't
#   something a viewer reads, so there's nothing to position there.
# - anaglyph: iw3.utils.py's own comment on this format says it plainly -- "Anaglyph is
#   already a single merged image correct on any player" -- there is no seam to tear on.
_HORIZONTAL_SPLIT_FORMATS = frozenset({"half_sbs", "full_sbs", "cross_eyed", "vr90"})
_VERTICAL_SPLIT_FORMATS = frozenset({"half_tb", "full_tb"})


# ADR-055: default ASS Fontsize (when --font-size is not given) as a fraction of the
# real probed PER-EYE vertical resolution -- see _default_font_size()/_eye_height().
# ~4.5% of vertical resolution is a common, readable convention for positioned/burned
# subtitle text (roughly what most 1080p ASS scripts use in practice, ~45-55px) --
# chosen to replace pysubs2's own fixed SSAStyle default of 20px, which was never
# designed against this project's typical (much larger, e.g. 3840x2076 SBS) output
# resolutions and rendered as under 1% of frame height there -- the real root cause of
# the "subtitles are very small" bug report this ADR fixes.
_DEFAULT_FONT_SIZE_HEIGHT_RATIO = 0.045


# ISO 639-1 (2-letter) -> ISO 639-2/B (3-letter, "bibliographic" form -- e.g. "ger"/
# "fre"/"chi", not the alternative "terminological" deu/fra/zho) lookup, needed ONLY
# here at the exact point --language is handed to mkvmerge, which expects the 3-letter
# form for MKV track metadata. See docs/ai/AI_DECISIONS.md ADR-042: this whole tool's
# user-facing --language input was standardized on ISO 639-1 (the same 2-letter format
# subtitle_search_cli.py's own --language already uses, and what iw3/gui.py's Search
# Subtitles/Add Subtitle Track fields both now share) so a user never has to remember
# which of two different code spaces a given "Language" field wants.
# No stdlib or already-installed dependency ships a maintained ISO 639-1<->639-2 table
# (checked: stdlib has none; pysubs2, this tool's only subtitle-format dependency,
# ships no language table either) -- adding a new pip dependency just for this lookup
# was judged not worth it, so this is a hand-written table. It deliberately covers
# only commonly-used languages, not the full ISO 639 set (see AI_DECISIONS.md for the
# full reasoning) -- _iso639_1_to_2() below falls back to passing an unlisted code
# through unchanged (lowercased) rather than erroring out, so an unusual-but-valid
# code still reaches mkvmerge instead of being rejected here.
_ISO_639_1_TO_2 = {
    "en": "eng", "es": "spa", "fr": "fre", "de": "ger", "it": "ita", "pt": "por",
    "ja": "jpn", "zh": "chi", "ko": "kor", "ru": "rus", "ar": "ara", "hi": "hin",
    "nl": "dut", "sv": "swe", "no": "nor", "da": "dan", "pl": "pol", "tr": "tur",
    "fi": "fin", "el": "gre", "he": "heb", "th": "tha", "vi": "vie", "id": "ind",
    "ms": "may", "cs": "cze", "hu": "hun", "ro": "rum", "uk": "ukr", "bg": "bul",
    "hr": "hrv", "sk": "slo", "sl": "slv", "sr": "srp", "lt": "lit", "lv": "lav",
    "et": "est", "fa": "per", "ur": "urd", "bn": "ben", "ta": "tam", "te": "tel",
    "ml": "mal", "mr": "mar", "gu": "guj", "pa": "pan", "sw": "swa", "af": "afr",
    "is": "ice", "ga": "gle", "cy": "wel", "ca": "cat", "eu": "baq", "gl": "glg",
    "sq": "alb", "hy": "arm", "ka": "geo", "az": "aze", "kk": "kaz", "uz": "uzb",
    "mn": "mon", "km": "khm", "lo": "lao", "my": "bur", "ne": "nep", "si": "sin",
    "am": "amh", "zu": "zul", "xh": "xho", "yo": "yor", "ig": "ibo", "ha": "hau",
}


def _iso639_1_to_2(code):
    """Converts an ISO 639-1 (2-letter) language code to its ISO 639-2/B (3-letter)
    equivalent for mkvmerge's --language argument. Falls back to the input, lowercased/
    stripped, if it isn't in _ISO_639_1_TO_2 -- covers both an already-3-letter code
    (e.g. someone still typing the old "eng") and any valid-but-uncommon ISO 639-1 code
    this table doesn't happen to list, so this never hard-errors on an unusual input."""
    normalized = str(code).strip().lower()
    return _ISO_639_1_TO_2.get(normalized, normalized)


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
    parser.add_argument("--language", type=str, default="en",
                         help="ISO 639-1 (2-letter) language code for the new subtitle track (e.g. "
                              "en, ja, fr, de, es) -- the same format subtitle_search_cli.py's own "
                              "--language expects (ADR-042). Converted internally to the ISO 639-2 "
                              "(3-letter) code mkvmerge actually needs for MKV track metadata (e.g. "
                              "en -> eng); an unrecognized code is passed through unchanged rather "
                              "than erroring out. Purely metadata -- does not affect muxing.")
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
                              "guesses. (Note: by itself this value does NOT change how the subtitle "
                              "is muxed -- a plain soft-subtitle track is added the same way "
                              "regardless of layout, resolving this is purely a safety/informational "
                              "check. It only matters together with --dual-eye-subtitles below, "
                              "which needs to know which formats are a genuine two-eye split.)")
    parser.add_argument("--dual-eye-subtitles", action="store_true", default=False,
                         help="Opt-in (default: off). See ADR-053. When --input's --format is a "
                              "genuine split-eye layout (half_sbs, full_sbs, cross_eyed, vr90, "
                              "half_tb, full_tb), converts --srt into an ASS track with each cue "
                              "duplicated into two copies, one positioned in each eye-half, instead "
                              "of the default plain track -- so the subtitle isn't torn in half by "
                              "the seam between the two eyes when played in an external "
                              "(non-iw3-player) media player. Leave this OFF if you play the result "
                              "in iw3-player, which already does its own correct per-eye rendering "
                              "of a plain track and would double-render a pre-positioned one. Turn "
                              "it ON only if you know you'll play the file in VLC/MPC-HC/a TV's "
                              "built-in player/etc. No effect for rgbd, half_rgbd, or anaglyph "
                              "(printed as a no-op note if passed with one of those) -- those aren't "
                              "a two-eye split, so there's no seam to fix.")
    parser.add_argument("--font-size", type=float, default=None,
                         help="Optional exact ASS Fontsize (in pixels) for --dual-eye-subtitles' "
                              "positioned track. Omit (default) to use a size auto-scaled to ~"
                              f"{_DEFAULT_FONT_SIZE_HEIGHT_RATIO * 100:.1f}% of --input's real "
                              "per-eye video height (ADR-055) -- this now looks reasonable at any "
                              "resolution with zero configuration, fixing a real bug where the "
                              "previous fixed default (pysubs2's own 20px) rendered as barely-"
                              "visible tiny text on typical iw3 output resolutions (e.g. a "
                              "3840x2076 double-wide SBS frame). Set this explicitly only if you "
                              "want a specific exact size instead of the auto-scaled default. Only "
                              "has an effect together with --dual-eye-subtitles on a genuine "
                              "split-eye --format -- the default plain SRT track has no Fontsize "
                              "field to control at all, so this is a documented no-op otherwise "
                              "(printed as a note, not silently ignored).")
    parser.add_argument("--start-time", type=str, default=None,
                         help="Start time WITHIN --srt's own timeline to keep (HH:MM:SS, MM:SS, or "
                              "seconds -- the exact same format/parser as iw3's own --start-time). "
                              "For when --srt has timestamps for a longer source (e.g. a full movie) "
                              "than --input actually covers (e.g. a trimmed clip converted with iw3's "
                              "own --start-time/--end-time): cues are trimmed to this window and "
                              "REBASED so this point becomes 0:00 in the output track, lining up with "
                              "--input's own first frame. A cue straddling this point is clipped to "
                              "begin at 0 rather than dropped or left with a negative timestamp. "
                              "Default: start of --srt (ADR-054). If only --end-time is given, this "
                              "defaults to 00:00:00 (trim from the beginning) -- see --end-time.")
    parser.add_argument("--end-time", type=str, default=None,
                         help="End time within --srt's own timeline to keep. Cues extending past this "
                              "point are clipped to end at (--end-time - --start-time) in the rebased "
                              "output; cues entirely after it are dropped. Default: end of --srt (the "
                              "whole file is used, unchanged, if both --start-time and --end-time are "
                              "omitted -- ADR-054). Given without --start-time, --start-time is treated "
                              "as 00:00:00 (there's no ambiguity in that direction -- trimming only the "
                              "tail end is well-defined).")
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
    here rather than an opaque mkvmerge failure later. Returns (subs, error_message):
    on success, (subs, None) where subs is the loaded pysubs2.SSAFile -- reused directly
    by run() for ADR-053's stereo-positioned-ASS conversion rather than reloading the
    file from disk a second time. On failure, (None, error_message)."""
    if not path.exists(srt_path):
        return None, f"ERROR: --srt file does not exist: {srt_path}"
    try:
        import pysubs2
        subs = pysubs2.load(srt_path)
    except Exception as e:
        return None, f"ERROR: failed to parse --srt as a subtitle file ('{srt_path}'): {e}"
    if len(subs) == 0:
        return None, f"ERROR: --srt file '{srt_path}' parsed successfully but contains no subtitle events."
    return subs, None


def _parse_time_range(start_time, end_time):
    """Parses --start-time/--end-time (see ADR-054) via the same parse_time() iw3's own
    --start-time/--end-time uses (nunif.utils.video.metadata), and validates the pair.

    --end-time without --start-time is treated as --start-time 00:00:00 (trim only the
    tail end -- unambiguous). Returns (start_sec, end_sec_or_None, error_message_or_None)
    -- end_sec is None when --end-time was not given (no upper bound). On any parse/
    validation failure, returns (None, None, error_message).
    """
    try:
        start_sec = parse_time(start_time) if start_time else 0.0
    except ValueError as e:
        return None, None, f"ERROR: could not parse --start-time ({start_time!r}): {e}"

    end_sec = None
    if end_time:
        try:
            end_sec = parse_time(end_time)
        except ValueError as e:
            return None, None, f"ERROR: could not parse --end-time ({end_time!r}): {e}"
        if end_sec <= start_sec:
            return None, None, (
                f"ERROR: --end-time ({end_time}) must be after --start-time "
                f"({start_time if start_time else '00:00:00'}).")
    return start_sec, end_sec, None


def _trim_and_rebase_subs(subs, start_sec, end_sec):
    """Returns a NEW pysubs2.SSAFile built from `subs` (the already pysubs2.load()-
    validated SSAFile returned by _validate_srt) containing only the cues overlapping
    [start_sec, end_sec) -- end_sec=None means no upper bound -- with every remaining
    cue's start/end REBASED by subtracting start_sec, so a cue that began at 1:16 in the
    source becomes 0:01 when start_sec=75 (see ADR-054: real use case is a subtitle file
    downloaded for a full movie being applied to a trimmed clip converted with iw3's own
    --start-time/--end-time).

    Boundary handling:
    - A cue that starts before start_sec but overlaps it is clipped to begin at 0 (never
      left negative, never dropped).
    - A cue that ends after end_sec is clipped to end at (end_sec - start_sec).
    - A cue with no overlap with [start_sec, end_sec) at all is dropped entirely.

    Returns (new_subs, kept_count, dropped_count).
    """
    import pysubs2

    out = pysubs2.SSAFile()
    out.info.update(subs.info)
    out.styles.update(subs.styles)

    kept = 0
    dropped = 0
    for event in subs:
        cue_start_sec = event.start / 1000.0
        cue_end_sec = event.end / 1000.0
        if cue_end_sec <= start_sec:
            dropped += 1
            continue
        if end_sec is not None and cue_start_sec >= end_sec:
            dropped += 1
            continue

        new_start_sec = max(cue_start_sec, start_sec) - start_sec
        clipped_end_sec = min(cue_end_sec, end_sec) if end_sec is not None else cue_end_sec
        new_end_sec = clipped_end_sec - start_sec

        new_event = event.copy()
        new_event.start = round(new_start_sec * 1000.0)
        new_event.end = round(new_end_sec * 1000.0)
        out.append(new_event)
        kept += 1

    return out, kept, dropped


def _probe_video_dimensions(input_path, ffprobe_bin, timeout=60):
    """Probes --input's REAL video stream width/height via ffprobe -- see ADR-053: a
    PlayResX/PlayResY header mismatched against the actual frame size is exactly the
    kind of subtle mispositioning bug this fix must not introduce, so it is never
    guessed/hardcoded here, only read from the real file. Same subprocess convention as
    iw3.reinject_hdr_cli._probe_frames_and_duration (CS-SUBPROCESS-001): a list command,
    never shell=True, and the ffprobe binary is resolved by the caller via
    iw3.utils._find_ffprobe() rather than assumed to be on PATH.

    Returns (width_or_None, height_or_None, error_message_or_None).
    """
    cmd = [ffprobe_bin, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "json", str(input_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return None, None, (
            f"ERROR: failed to run ffprobe to determine --input's video dimensions "
            f"('{_format_cmd(cmd)}'): {e}")
    try:
        data = json.loads(proc.stdout) if proc.stdout else {}
        streams = data.get("streams", [])
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
    except Exception as e:
        return None, None, (
            f"ERROR: could not determine --input's video width/height via ffprobe -- needed to "
            f"correctly position stereo subtitles for a split-eye layout (ADR-053). Command run: "
            f"'{_format_cmd(cmd)}'. ffprobe stdout: {proc.stdout!r} stderr: {proc.stderr!r} "
            f"({e})")
    if width <= 0 or height <= 0:
        return None, None, (
            f"ERROR: ffprobe reported invalid video dimensions ({width}x{height}) for {input_path}")
    return width, height, None


def _eye_height(resolved_format, height):
    """Returns the effective PER-EYE vertical resolution for resolved_format -- the
    height each eye's rendered subtitle text is actually viewed against (see ADR-055).
    Unchanged for a horizontal split (SBS-family: only width is split between the two
    eyes, each eye's height is the full frame height); HALVED for a vertical split
    (TB-family: each eye occupies only the top or bottom half of the frame) -- mirrors
    the same halving _build_stereo_positioned_subs already applies to its own y-position
    math for TB formats below."""
    if resolved_format in _VERTICAL_SPLIT_FORMATS:
        return height / 2.0
    return float(height)


def _default_font_size(resolved_format, height):
    """Returns the default ASS Fontsize (pixels) when --font-size is not given (ADR-055):
    _DEFAULT_FONT_SIZE_HEIGHT_RATIO of the real per-eye vertical resolution
    (_eye_height), rounded to the nearest pixel. Replaces pysubs2's own fixed SSAStyle
    default (20px), which was never designed against this project's typical (much
    larger) output resolutions and rendered as barely-visible tiny text there -- the
    real root cause of the "subtitles are very small" bug report."""
    return round(_eye_height(resolved_format, height) * _DEFAULT_FONT_SIZE_HEIGHT_RATIO)


def _build_stereo_positioned_subs(subs, resolved_format, width, height, font_size=None):
    """Returns a NEW pysubs2.SSAFile built from `subs` (the already pysubs2.load()-
    validated SSAFile returned by _validate_srt) where every cue is duplicated into two
    SSAEvents sharing identical timing/text, each carrying an inline ASS '{\\pos(x,y)}'
    override placing one copy centered within each eye-half of a split-eye stereo frame.
    See ADR-053 for why this is needed: a plain centered subtitle on a split-eye frame
    lands exactly on the seam between the two eyes on any normal (non-iw3-player) media
    player, tearing every line of text in half.

    Only meaningful for resolved_format in _HORIZONTAL_SPLIT_FORMATS/
    _VERTICAL_SPLIT_FORMATS -- callers are responsible for only calling this for those
    formats and leaving every other format's original plain SRT track untouched.

    PlayResX/PlayResY in the returned file's script-info header are set to the REAL
    probed `width`/`height` (never left unset/default) so libass (or any ASS renderer)
    interprets the '\\pos(x,y)' coordinates below against the actual frame it draws over
    -- a PlayResX/Y mismatch would silently shift every position, which is the other
    plausible root cause ADR-053 explicitly ruled out by probing real dimensions here.

    font_size (ADR-055): every style's Fontsize is set to `font_size` if given (used
    as-is, no scaling -- an explicit --font-size override), otherwise to
    _default_font_size(resolved_format, height) -- never left at pysubs2's own small
    fixed default, which is what made subtitles look tiny on real (large) iw3 output
    resolutions before this fix.

    Positioning: '{\\pos(x,y)}' with the default/copied style's Alignment=2 (bottom-
    center, pysubs2's own default and what --srt files load as) anchors the BOTTOM-
    CENTER of the rendered text at (x, y) -- so y values below are chosen near the
    bottom of each eye-half (0.92 of its height), matching how subtitles are
    conventionally placed.
    - Horizontal split (SBS-family: half_sbs/full_sbs/cross_eyed/vr90): left copy at
      x=width/4, right copy at x=3*width/4, both at y=height*0.92.
    - Vertical split (TB-family: half_tb/full_tb): x=width/2 unchanged, top copy at
      y=(height/2)*0.92, bottom copy at y=height/2 + (height/2)*0.92.
    """
    import pysubs2

    if resolved_format in _HORIZONTAL_SPLIT_FORMATS:
        y = height * 0.92
        positions = ((width / 4.0, y), (width * 3.0 / 4.0, y))
    elif resolved_format in _VERTICAL_SPLIT_FORMATS:
        x = width / 2.0
        half_y = (height / 2.0) * 0.92
        positions = ((x, half_y), (x, height / 2.0 + half_y))
    else:
        raise ValueError(
            f"_build_stereo_positioned_subs called for non-split-eye format: {resolved_format!r}")

    resolved_font_size = font_size if font_size is not None else _default_font_size(resolved_format, height)

    out = pysubs2.SSAFile()
    out.info.update(subs.info)
    out.info["PlayResX"] = str(int(width))
    out.info["PlayResY"] = str(int(height))
    out.styles.update(subs.styles)
    if "Default" not in out.styles:
        out.styles["Default"] = pysubs2.SSAStyle()
    for style in out.styles.values():
        style.fontsize = resolved_font_size

    for event in subs:
        plain_text = event.plaintext
        style_name = event.style if event.style in out.styles else "Default"
        for (px, py) in positions:
            out.append(pysubs2.SSAEvent(
                start=event.start, end=event.end, style=style_name,
                text=f"{{\\pos({px:.1f},{py:.1f})}}{plain_text}",
            ))
    return out


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

    subs, srt_error = _validate_srt(srt_path)
    if srt_error:
        print(srt_error, file=sys.stderr)
        return 1

    # ADR-054: optional --start-time/--end-time trim+rebase, validated before anything
    # else touches --srt further (dual-eye positioning below operates on the already
    # trimmed/rebased `subs`, so the two features compose correctly). Omitting both
    # flags (the default) leaves `subs`/trimming_requested untouched -- see
    # _self_test_run_default_flag_off_time_range_unchanged.
    start_sec, end_sec, time_range_error = _parse_time_range(
        getattr(args, "start_time", None), getattr(args, "end_time", None))
    if time_range_error:
        print(time_range_error, file=sys.stderr)
        return 1
    trimming_requested = bool(getattr(args, "start_time", None)) or bool(getattr(args, "end_time", None))
    if trimming_requested:
        subs, kept, dropped = _trim_and_rebase_subs(subs, start_sec, end_sec)
        print(f"[subtitle-mux] trimmed --srt to [{start_sec}s, "
              f"{'end' if end_sec is None else f'{end_sec}s'}) and rebased so {start_sec}s -> 0s "
              f"in the output track -- {kept} cue(s) kept, {dropped} cue(s) dropped (ADR-054)",
              file=sys.stderr)
        if kept == 0:
            print(
                "ERROR: after trimming to --start-time/--end-time, no subtitle cues remain within "
                "that window -- refusing to mux an empty subtitle track. Double check the range "
                "against --srt's own (untrimmed) timestamps.", file=sys.stderr)
            return 1

    # ADR-055: optional --font-size override, validated before mkvmerge is looked up --
    # same early-refusal convention as --start-time/--end-time above. Only meaningful
    # together with --dual-eye-subtitles on a genuine split-eye --format (see below);
    # validated here regardless so a bad value is caught immediately either way.
    font_size = getattr(args, "font_size", None)
    if font_size is not None and font_size <= 0:
        print(f"ERROR: --font-size must be a positive number (got {font_size}).", file=sys.stderr)
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

    # ADR-053: DEFAULT (--dual-eye-subtitles absent/False) is exactly the original
    # ADR-032 plain-track behavior for every --format, split-eye or not -- no ffprobe
    # probe, no .ass conversion, nothing. Only when --dual-eye-subtitles is explicitly
    # passed AND the resolved format is a genuine split-eye layout does this convert
    # --srt into a stereo-positioned ASS track (two positioned copies per cue) and mux
    # THAT instead -- for an external (non-iw3-player) player, where a plain centered
    # subtitle track lands exactly on the seam between the two eyes, tearing every line
    # of text in half. rgbd/half_rgbd/anaglyph are unaffected by the flag either way --
    # see _HORIZONTAL_SPLIT_FORMATS/_VERTICAL_SPLIT_FORMATS above.
    mux_srt_path = srt_path
    tmp_ass_path = None
    tmp_trimmed_srt_path = None
    # ADR-054: when trimming was applied above, `subs` is already the trimmed/rebased
    # SSAFile -- write it to a new temp file (same extension as --srt) and mux THAT
    # instead of the untouched original --srt. Done here (before the dual-eye block
    # below) so --dual-eye-subtitles, if also requested, builds its stereo-positioned
    # ASS track from the already-trimmed `subs` -- both features compose correctly.
    if trimming_requested:
        try:
            srt_ext = path.splitext(srt_path)[1] or ".srt"
            fd, tmp_trimmed_srt_path = tempfile.mkstemp(
                suffix=srt_ext, prefix="iw3_submux_trim_", dir=out_dir)
            os.close(fd)
            subs.save(tmp_trimmed_srt_path)
        except Exception as e:
            print(f"ERROR: failed to write trimmed/rebased subtitle file: {e}", file=sys.stderr)
            if tmp_trimmed_srt_path and path.exists(tmp_trimmed_srt_path):
                try:
                    os.remove(tmp_trimmed_srt_path)
                except Exception:
                    pass
            return 1
        mux_srt_path = tmp_trimmed_srt_path
    dual_eye_subtitles = bool(getattr(args, "dual_eye_subtitles", False))
    is_split_format = (resolved_format in _HORIZONTAL_SPLIT_FORMATS
                        or resolved_format in _VERTICAL_SPLIT_FORMATS)
    if dual_eye_subtitles and not is_split_format:
        print(f"[subtitle-mux] --dual-eye-subtitles has no effect for --format "
              f"'{resolved_format}' -- it isn't a two-eye split layout (see ADR-053), so "
              f"there's no seam to fix. Muxing the plain --srt track as usual.",
              file=sys.stderr)
    # ADR-055: --font-size only has an effect on the ASS track built below -- the plain
    # SRT track (dual_eye_subtitles off, or a non-split --format) has no Fontsize field
    # to control at all. Documented no-op, not a silent ignore, mirroring the
    # --dual-eye-subtitles no-op message immediately above.
    if font_size is not None and not (dual_eye_subtitles and is_split_format):
        print(f"[subtitle-mux] --font-size has no effect here -- it only applies to "
              f"--dual-eye-subtitles' positioned ASS track (see ADR-055), and this run is "
              f"muxing a plain --srt track instead. Ignoring --font-size.", file=sys.stderr)
    if dual_eye_subtitles and is_split_format:
        ffprobe_bin = _find_ffprobe()
        width, height, probe_error = _probe_video_dimensions(input_path, ffprobe_bin)
        if probe_error:
            print(probe_error, file=sys.stderr)
            print("ERROR: cannot correctly position stereo subtitles for a split-eye layout "
                  "without --input's real video dimensions -- refusing rather than muxing "
                  "possibly-mispositioned subtitles.", file=sys.stderr)
            return 1
        print(f"[subtitle-mux] --input video dimensions (ffprobe): {width}x{height}",
              file=sys.stderr)

        resolved_font_size = font_size if font_size is not None else _default_font_size(resolved_format, height)
        print(f"[subtitle-mux] font size: {resolved_font_size}px "
              f"({'explicit --font-size' if font_size is not None else 'auto-scaled to '
                f'{_DEFAULT_FONT_SIZE_HEIGHT_RATIO * 100:.1f}% of per-eye video height, ADR-055'})",
              file=sys.stderr)

        positioned_subs = _build_stereo_positioned_subs(subs, resolved_format, width, height,
                                                          font_size=font_size)
        try:
            fd, tmp_ass_path = tempfile.mkstemp(suffix=".ass", prefix="iw3_submux_", dir=out_dir)
            os.close(fd)
            positioned_subs.save(tmp_ass_path)
        except Exception as e:
            print(f"ERROR: failed to write stereo-positioned ASS subtitle file: {e}", file=sys.stderr)
            if tmp_ass_path and path.exists(tmp_ass_path):
                try:
                    os.remove(tmp_ass_path)
                except Exception:
                    pass
            return 1
        mux_srt_path = tmp_ass_path
        print(f"[subtitle-mux] {resolved_format} is a split-eye layout -- converted --srt into a "
              f"stereo-positioned ASS track ({len(subs)} cues -> {len(positioned_subs)} positioned "
              f"events) at {tmp_ass_path}", file=sys.stderr)

    try:
        # mkvmerge: per-file options (--language/--track-name) must appear BEFORE the file
        # they apply to. "0:" addresses track 0 OF THAT FILE (an .srt/.ass has exactly one
        # track). <input_path> is given last with no options preceding it, which is
        # mkvmerge's default "copy every track from this file, unmodified" behavior -- this
        # is what preserves the original video/audio/existing-subtitle tracks untouched.
        # Verified against real mkvmerge behavior in this tool's own smoke test (see
        # docs/ai/AI_DECISIONS.md ADR-032). mkvmerge auto-detects S_TEXT/ASS from the
        # ".ass" extension exactly the same way it auto-detects S_TEXT/UTF8 from ".srt".
        mkvmerge_language = _iso639_1_to_2(args.language)
        cmd = [
            mkvmerge_bin, "-o", tmp_output,
            "--language", f"0:{mkvmerge_language}",
            "--track-name", f"0:{track_name}",
            mux_srt_path,
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
    finally:
        # CS-IO-001: clean up intermediate temp files regardless of outcome -- neither is
        # ever the thing --output points at (that's tmp_output/os.replace() above).
        for _tmp_path in (tmp_ass_path, tmp_trimmed_srt_path):
            if _tmp_path and path.exists(_tmp_path):
                try:
                    os.remove(_tmp_path)
                except Exception:
                    pass


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


def _self_test_iso639_lookup():
    """Synthetic test of the ISO 639-1 -> ISO 639-2/B lookup table (ADR-042) -- no
    network/mkvmerge needed. Covers the common-language cases the task explicitly
    called out, plus the fallback behavior for a code not in the table."""
    common_cases = {
        "en": "eng", "es": "spa", "fr": "fre", "de": "ger", "it": "ita", "pt": "por",
        "ja": "jpn", "zh": "chi", "ko": "kor", "ru": "rus",
    }
    for code, expected in common_cases.items():
        assert _iso639_1_to_2(code) == expected, f"{code}: expected {expected}, got {_iso639_1_to_2(code)}"

    # case-insensitive / whitespace-tolerant
    assert _iso639_1_to_2("EN") == "eng"
    assert _iso639_1_to_2(" ja ") == "jpn"

    # Fallback: a code not in the table is passed through unchanged (lowercased),
    # never raises -- covers both an already-3-letter code (pre-ADR-042 habit) and a
    # genuinely valid-but-uncommon ISO 639-1 code this table doesn't list.
    assert _iso639_1_to_2("eng") == "eng"
    assert _iso639_1_to_2("xx") == "xx"
    assert _iso639_1_to_2("XX") == "xx"

    print("_self_test_iso639_lookup: PASS")


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

        subs, err = _validate_srt(valid_path)
        assert err is None and subs is not None and len(subs) == 2, (subs, err)

        subs, err = _validate_srt(malformed_path)
        assert subs is None and err is not None, (subs, err)

        subs, err = _validate_srt(path.join(tmpdir, "does_not_exist.srt"))
        assert subs is None and err is not None, (subs, err)

    print("_self_test_srt_validation: PASS")


def _self_test_probe_video_dimensions():
    """Synthetic/mocked test of _probe_video_dimensions's ffprobe JSON parsing (no real
    ffmpeg/GPU needed -- CS-TEST-001), mirroring iw3.reinject_hdr_cli's own
    _self_test_probe_frames_and_duration convention for the same kind of ffprobe call."""
    from unittest.mock import patch, MagicMock

    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"streams": [{"width": "3840", "height": "2076"}]}),
            returncode=0)
        width, height, err = _probe_video_dimensions("fake.mkv", "ffprobe")
        assert (width, height, err) == (3840, 2076, None), (width, height, err)

    with patch.object(subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(stdout="not json", returncode=1)
        width, height, err = _probe_video_dimensions("fake.mkv", "ffprobe")
        assert width is None and height is None and err is not None, (width, height, err)

    with patch.object(subprocess, "run", side_effect=OSError("boom")):
        width, height, err = _probe_video_dimensions("fake.mkv", "ffprobe")
        assert width is None and height is None and "boom" in err, (width, height, err)

    print("_self_test_probe_video_dimensions: PASS")


def _self_test_default_font_size():
    """Synthetic test of ADR-055's _default_font_size()/_eye_height() scaling math at a
    few real resolutions this project actually produces -- no GPU/real video needed.
    Covers 1080p, the real 3840x2076 double-wide 4K SBS case from the bug report, and a
    hypothetical 8K output, for both a horizontal-split (SBS) and vertical-split (TB)
    format, proving the default: (1) is always well above pysubs2's own tiny fixed 20px
    default (the actual root cause of "subtitles are very small"), (2) scales up with
    resolution rather than staying fixed, and (3) is smaller for a TB format than an SBS
    format at the same width/height (each TB eye only gets half the vertical space)."""
    cases = [
        # (width, height) -- height is what matters for this math.
        (1920, 1080),   # 1080p half_sbs-style packed frame
        (3840, 2076),   # the real bug-report geometry (double-wide 4K SBS)
        (7680, 4320),   # hypothetical 8K
    ]
    prev_horiz_size = 0
    for width, height in cases:
        horiz_size = _default_font_size("half_sbs", height)
        vert_size = _default_font_size("half_tb", height)
        assert horiz_size > 20, (
            f"{width}x{height}: default font size {horiz_size}px must exceed pysubs2's own "
            f"tiny fixed 20px default -- that fixed default is the actual bug being fixed")
        assert horiz_size > prev_horiz_size, (
            "default font size must scale UP with resolution, not stay fixed",
            width, height, horiz_size, prev_horiz_size)
        assert 0 < vert_size < horiz_size, (
            "a TB (vertical-split) default must be smaller than an SBS (horizontal-split) "
            "default at the same width/height -- each TB eye only gets half the height",
            vert_size, horiz_size)
        assert horiz_size == round(height * _DEFAULT_FONT_SIZE_HEIGHT_RATIO), (width, height, horiz_size)
        assert vert_size == round((height / 2.0) * _DEFAULT_FONT_SIZE_HEIGHT_RATIO), (width, height, vert_size)
        prev_horiz_size = horiz_size

    print("_self_test_default_font_size: PASS")


def _self_test_stereo_positioning():
    """Synthetic test of ADR-053's split-eye subtitle positioning -- no real video/ffmpeg
    needed. Covers: PlayResX/PlayResY set from the real probed width/height (never left
    default/unset), correct \\pos coordinates straddling the true seam for one
    horizontal-split format (half_sbs) and one vertical-split format (half_tb), and that
    the non-split formats (rgbd/half_rgbd/anaglyph) are excluded from both split-format
    sets so run() leaves them untouched."""
    import re
    import tempfile as _tempfile

    srt_text = (
        "1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nSecond line\n\n"
    )
    with _tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        srt_path = path.join(tmpdir, "subs.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        subs, err = _validate_srt(srt_path)
        assert err is None, err

    width, height = 3840, 2076
    pos_re = re.compile(r"^\{\\pos\(([-\d.]+),([-\d.]+)\)\}(.*)$")

    # Horizontal split (half_sbs): left/right copies must straddle the vertical seam at
    # width/2 -- x=1920 for this geometry, the exact seam the real bug report reproduces
    # on (movie_LR.mkv-style filenames resolve to half_sbs).
    horiz = _build_stereo_positioned_subs(subs, "half_sbs", width, height)
    assert horiz.info["PlayResX"] == str(width), horiz.info
    assert horiz.info["PlayResY"] == str(height), horiz.info
    assert len(horiz) == len(subs) * 2, len(horiz)
    # ADR-055: default Fontsize is scaled to the real per-eye height (unchanged for a
    # horizontal split), never left at pysubs2's own fixed 20px default.
    expected_horiz_font_size = _default_font_size("half_sbs", height)
    assert expected_horiz_font_size != 20, "test would be meaningless if the scaled default happened to equal pysubs2's own fixed default"
    assert horiz.styles["Default"].fontsize == expected_horiz_font_size, horiz.styles["Default"].fontsize
    seam_x = width / 2.0
    for i in range(0, len(horiz), 2):
        left, right = horiz[i], horiz[i + 1]
        assert left.start == right.start == subs[i // 2].start
        lm = pos_re.match(left.text)
        rm = pos_re.match(right.text)
        assert lm and rm, (left.text, right.text)
        left_x, right_x = float(lm.group(1)), float(rm.group(1))
        assert left_x < seam_x < right_x, (left_x, seam_x, right_x)
        assert lm.group(3) == subs[i // 2].plaintext
        assert rm.group(3) == subs[i // 2].plaintext

    # Vertical split (half_tb): top/bottom copies must straddle the horizontal seam at
    # height/2; x stays at width/2 (unchanged) on both copies.
    vert = _build_stereo_positioned_subs(subs, "half_tb", width, height)
    # ADR-055: default Fontsize for a vertical (TB) split is scaled to HALF the frame
    # height (the per-eye height, _eye_height) -- roughly half the horizontal-split
    # default above at the same width/height (allow +/-1px for independent rounding),
    # since each TB eye only gets half the vertical space a horizontal-split eye gets.
    expected_vert_font_size = _default_font_size("half_tb", height)
    assert _eye_height("half_tb", height) == height / 2.0
    assert abs(expected_vert_font_size - expected_horiz_font_size / 2.0) <= 1, (
        expected_vert_font_size, expected_horiz_font_size)
    assert vert.styles["Default"].fontsize == expected_vert_font_size, vert.styles["Default"].fontsize
    seam_y = height / 2.0
    for i in range(0, len(vert), 2):
        top, bottom = vert[i], vert[i + 1]
        tm = pos_re.match(top.text)
        bm = pos_re.match(bottom.text)
        assert tm and bm, (top.text, bottom.text)
        top_x, top_y = float(tm.group(1)), float(tm.group(2))
        bottom_x, bottom_y = float(bm.group(1)), float(bm.group(2))
        assert top_x == bottom_x == width / 2.0, (top_x, bottom_x)
        assert top_y < seam_y < bottom_y, (top_y, seam_y, bottom_y)

    # Non-split formats: excluded from both split-format sets (callers must leave these
    # untouched -- rgbd/half_rgbd is a color|depth split, not two eye-views; anaglyph is
    # already a single merged 2D frame).
    for fmt in ("rgbd", "half_rgbd", "anaglyph"):
        assert fmt not in _HORIZONTAL_SPLIT_FORMATS and fmt not in _VERTICAL_SPLIT_FORMATS, fmt
    try:
        _build_stereo_positioned_subs(subs, "anaglyph", width, height)
        assert False, "expected ValueError for a non-split format"
    except ValueError:
        pass

    # ADR-055: explicit font_size override is used AS-IS, no scaling -- must differ from
    # both scaled defaults above to prove it isn't silently ignored/recomputed.
    explicit = _build_stereo_positioned_subs(subs, "half_sbs", width, height, font_size=72)
    assert explicit.styles["Default"].fontsize == 72, explicit.styles["Default"].fontsize

    print("_self_test_stereo_positioning: PASS")


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
                        language="en", track_name=None, format="auto",
                        dual_eye_subtitles=False)
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

        # explicit --format bypasses filename ambiguity and reaches the mkvmerge stage.
        # format="full_tb" is a split-eye (vertical) format under ADR-053, but mkvmerge
        # lookup happens BEFORE the new split-format/ffprobe branch (see run()), so this
        # still refuses at the same "mkvmerge not found" point as before -- ADR-053 did
        # not move the mkvmerge-lookup gate.
        with patch(f"{__name__}._find_mkvmerge", return_value=None):
            rc = run(_args(input=ambiguous_input, format="full_tb"))
            assert rc == 1, rc  # refuses because mkvmerge isn't found, but got past format gate

    print("_self_test_run_gating: PASS")


def _self_test_run_flag_on_split_format_conversion():
    """Synthetic/mocked end-to-end test of run()'s ADR-053 --dual-eye-subtitles path:
    proves that with the flag ON for a split-eye --format, run() converts --srt into an
    intermediate .ass file and invokes mkvmerge with THAT path (not the original .srt),
    and that the intermediate .ass temp file is cleaned up afterward regardless of
    success. Mocks _find_mkvmerge and _probe_video_dimensions (ffprobe's own JSON
    parsing is covered separately by _self_test_probe_video_dimensions) and
    subprocess.run for the mkvmerge invocation -- per CS-TEST-001, the mocked mkvmerge
    actually writes the tmp_output file the next step (os.replace onto --output)
    expects, not just a bare success return code."""
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        output_path = path.join(tmpdir, "out.mkv")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        args = argparse.Namespace(
            input=mkv_input, srt=srt_path, output=output_path,
            language="en", track_name=None, format="half_sbs",
            dual_eye_subtitles=True)

        seen = {}

        def _fake_mkvmerge_run(cmd, **kwargs):
            seen["cmd"] = cmd
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as f:
                f.write(b"fake mkv output")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge"), \
             patch(f"{__name__}._probe_video_dimensions", return_value=(3840, 2076, None)), \
             patch.object(subprocess, "run", side_effect=_fake_mkvmerge_run):
            rc = run(args)

        assert rc == 0, rc
        assert path.exists(output_path)
        cmd = seen["cmd"]
        # mux_srt_path is the element immediately before input_path in the mkvmerge cmd.
        muxed_path = cmd[cmd.index(mkv_input) - 1]
        assert muxed_path != srt_path, "expected the converted .ass, not the original .srt"
        assert muxed_path.lower().endswith(".ass"), muxed_path
        assert not path.exists(muxed_path), "intermediate .ass temp file should be cleaned up"

    print("_self_test_run_flag_on_split_format_conversion: PASS")


def _self_test_run_font_size_default_and_explicit():
    """End-to-end (mocked mkvmerge/ffprobe, real pysubs2 round-trip) test of ADR-055's
    run() wiring: with --dual-eye-subtitles on a split --format and no --font-size,
    the produced .ass file's real on-disk Fontsize matches the auto-scaled default for
    the probed dimensions; with --font-size given, it's used exactly as given instead.
    Reads the intermediate .ass file's actual content (via the mocked mkvmerge call,
    before run()'s own finally block deletes it -- same pattern as
    _self_test_run_time_range_trims_and_rebases_end_to_end), not just in-memory state."""
    import tempfile
    from unittest.mock import patch, MagicMock
    import pysubs2

    def _run_and_capture_ass(tmpdir, font_size):
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        output_path = path.join(tmpdir, f"out_{font_size}.mkv")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        args = argparse.Namespace(
            input=mkv_input, srt=srt_path, output=output_path,
            language="en", track_name=None, format="half_sbs",
            dual_eye_subtitles=True, font_size=font_size)

        seen = {}

        def _fake_mkvmerge_run(cmd, **kwargs):
            muxed_path = cmd[cmd.index(mkv_input) - 1]
            with open(muxed_path, "r", encoding="utf-8") as f:
                seen["ass_content"] = f.read()
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as f:
                f.write(b"fake mkv output")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge"), \
             patch(f"{__name__}._probe_video_dimensions", return_value=(3840, 2076, None)), \
             patch.object(subprocess, "run", side_effect=_fake_mkvmerge_run):
            rc = run(args)
        assert rc == 0, rc
        return pysubs2.SSAFile.from_string(seen["ass_content"])

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        default_subs = _run_and_capture_ass(tmpdir, font_size=None)
        expected_default = _default_font_size("half_sbs", 2076)
        assert expected_default != 20, "test would be meaningless if the scaled default equaled pysubs2's fixed default"
        assert default_subs.styles["Default"].fontsize == expected_default, \
            default_subs.styles["Default"].fontsize

        explicit_subs = _run_and_capture_ass(tmpdir, font_size=64.0)
        assert explicit_subs.styles["Default"].fontsize == 64.0, explicit_subs.styles["Default"].fontsize

    print("_self_test_run_font_size_default_and_explicit: PASS")


def _self_test_run_font_size_noop_without_dual_eye():
    """ADR-055: --font-size must be a no-op (plain --srt muxed unchanged, no .ass built)
    when --dual-eye-subtitles is off, even though --font-size was explicitly given --
    the plain track has no Fontsize field to control at all. Confirms this both by the
    muxed path staying the original --srt and by the documented no-op note appearing in
    stdout/stderr (captured via a MagicMock print check would be overkill here -- the
    muxed-path check alone proves behavior; the message itself is covered by reading
    run()'s source, consistent with how the sibling --dual-eye-subtitles no-op note is
    verified in _self_test_run_flag_on_non_split_format_noop)."""
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        output_path = path.join(tmpdir, "out.mkv")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        args = argparse.Namespace(
            input=mkv_input, srt=srt_path, output=output_path,
            language="en", track_name=None, format="half_sbs",
            dual_eye_subtitles=False, font_size=64.0)

        seen = {}

        def _fake_mkvmerge_run(cmd, **kwargs):
            seen["cmd"] = cmd
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as f:
                f.write(b"fake mkv output")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge"), \
             patch(f"{__name__}._probe_video_dimensions") as mock_probe, \
             patch.object(subprocess, "run", side_effect=_fake_mkvmerge_run):
            rc = run(args)
            mock_probe.assert_not_called()

        assert rc == 0, rc
        muxed_path = seen["cmd"][seen["cmd"].index(mkv_input) - 1]
        assert muxed_path == srt_path, (
            "--font-size without --dual-eye-subtitles must still mux the original --srt unchanged "
            f"-- got {muxed_path!r}")

    print("_self_test_run_font_size_noop_without_dual_eye: PASS")


def _self_test_run_default_flag_off_split_format_unchanged():
    """THE key regression check for ADR-053's scope correction: with --dual-eye-subtitles
    absent/False (the DEFAULT), a split-eye --format must produce EXACTLY the original
    ADR-032 plain-track behavior, unchanged -- mkvmerge invoked with the ORIGINAL --srt
    path (no .ass conversion at all), and _probe_video_dimensions (ffprobe) never even
    called. Covers both an explicit dual_eye_subtitles=False and the attribute being
    absent entirely (an older Namespace / getattr fallback), so the default is provably
    off both ways, not just when the flag happens to be spelled out."""
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        def _fake_mkvmerge_run(cmd, **kwargs):
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as f:
                f.write(b"fake mkv output")
            return MagicMock(returncode=0, stdout="", stderr="")

        for case_name, args_kwargs in (
            ("explicit dual_eye_subtitles=False", dict(dual_eye_subtitles=False)),
            ("dual_eye_subtitles attribute absent entirely", {}),
        ):
            output_path = path.join(tmpdir, f"out_{len(args_kwargs)}.mkv")
            base = dict(input=mkv_input, srt=srt_path, output=output_path,
                        language="en", track_name=None, format="half_sbs")
            base.update(args_kwargs)
            args = argparse.Namespace(**base)

            seen = {}

            def _capture(cmd, **kwargs):
                seen["cmd"] = cmd
                return _fake_mkvmerge_run(cmd, **kwargs)

            with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge"), \
                 patch(f"{__name__}._probe_video_dimensions") as mock_probe, \
                 patch.object(subprocess, "run", side_effect=_capture):
                rc = run(args)
                mock_probe.assert_not_called()

            assert rc == 0, (case_name, rc)
            cmd = seen["cmd"]
            muxed_path = cmd[cmd.index(mkv_input) - 1]
            assert muxed_path == srt_path, (
                f"{case_name}: default (flag off) must mux the ORIGINAL --srt unchanged, "
                f"even for a split-eye format -- got {muxed_path!r}")

    print("_self_test_run_default_flag_off_split_format_unchanged: PASS")


def _self_test_run_flag_on_non_split_format_noop():
    """Proves --dual-eye-subtitles is a no-op for a non-split --format (anaglyph): even
    with the flag ON, mkvmerge is invoked with the ORIGINAL --srt path (no .ass
    conversion), and _probe_video_dimensions (ffprobe) is never even called -- there's no
    seam to fix for rgbd/half_rgbd/anaglyph, so the flag must not change their output."""
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        output_path = path.join(tmpdir, "out.mkv")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        args = argparse.Namespace(
            input=mkv_input, srt=srt_path, output=output_path,
            language="en", track_name=None, format="anaglyph",
            dual_eye_subtitles=True)

        seen = {}

        def _fake_mkvmerge_run(cmd, **kwargs):
            seen["cmd"] = cmd
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as f:
                f.write(b"fake mkv output")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge"), \
             patch(f"{__name__}._probe_video_dimensions") as mock_probe, \
             patch.object(subprocess, "run", side_effect=_fake_mkvmerge_run):
            rc = run(args)
            mock_probe.assert_not_called()

        assert rc == 0, rc
        cmd = seen["cmd"]
        muxed_path = cmd[cmd.index(mkv_input) - 1]
        assert muxed_path == srt_path, muxed_path

    print("_self_test_run_flag_on_non_split_format_noop: PASS")


def _self_test_run_flag_on_probe_failure_refuses_before_mkvmerge():
    """A --dual-eye-subtitles run() for a split-format where ffprobe dimension probing
    fails must refuse (return 1) WITHOUT ever invoking mkvmerge to actually mux --
    proves the gating order still holds with ADR-053's dimension-probe step inserted
    before the mux itself."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        output_path = path.join(tmpdir, "out.mkv")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        args = argparse.Namespace(
            input=mkv_input, srt=srt_path, output=output_path,
            language="en", track_name=None, format="half_tb",
            dual_eye_subtitles=True)

        with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge"), \
             patch(f"{__name__}._probe_video_dimensions",
                   return_value=(None, None, "ERROR: ffprobe failed")), \
             patch.object(subprocess, "run") as mock_run:
            rc = run(args)
            assert rc == 1, rc
            mock_run.assert_not_called()
        assert not path.exists(output_path)

    print("_self_test_run_flag_on_probe_failure_refuses_before_mkvmerge: PASS")


def _self_test_parse_time_range():
    """Synthetic test of ADR-054's _parse_time_range: valid pair, --end-time without
    --start-time defaulting to 00:00:00, end<=start rejected, and an unparseable time
    rejected with a clear message -- no GPU/real files needed."""
    start_sec, end_sec, err = _parse_time_range("00:01:15", "00:06:15")
    assert err is None and start_sec == 75.0 and end_sec == 375.0, (start_sec, end_sec, err)

    # neither given -> no trimming window at all
    start_sec, end_sec, err = _parse_time_range(None, None)
    assert err is None and start_sec == 0.0 and end_sec is None, (start_sec, end_sec, err)

    # --end-time without --start-time -> start defaults to 00:00:00
    start_sec, end_sec, err = _parse_time_range(None, "00:00:30")
    assert err is None and start_sec == 0.0 and end_sec == 30.0, (start_sec, end_sec, err)

    # end <= start -> clear rejection, mentions both flags
    start_sec, end_sec, err = _parse_time_range("00:00:30", "00:00:30")
    assert start_sec is None and end_sec is None and err is not None and "--end-time" in err \
        and "--start-time" in err, (start_sec, end_sec, err)
    start_sec, end_sec, err = _parse_time_range("00:00:30", "00:00:10")
    assert start_sec is None and err is not None, (start_sec, end_sec, err)

    # unparseable time -> clear rejection naming the offending flag
    start_sec, end_sec, err = _parse_time_range("not-a-time", None)
    assert start_sec is None and err is not None and "--start-time" in err, (start_sec, end_sec, err)
    start_sec, end_sec, err = _parse_time_range(None, "not-a-time")
    assert start_sec is None and err is not None and "--end-time" in err, (start_sec, end_sec, err)

    print("_self_test_parse_time_range: PASS")


def _self_test_trim_and_rebase_subs():
    """Synthetic test of ADR-054's _trim_and_rebase_subs, covering a cue entirely
    before the window, one straddling the start (clipped to begin at 0, never
    negative), one entirely inside (rebased), one straddling the end (clipped to
    end - start), one entirely after, one that exactly touches the start boundary
    (dropped, zero overlap), and one that exactly touches the end boundary (dropped)
    -- plus the no-upper-bound (end_sec=None) case. No GPU/real video needed."""
    import pysubs2

    srt_text = (
        "1\n00:00:00,000 --> 00:00:05,000\nBefore window\n\n"
        "2\n00:00:05,000 --> 00:00:10,000\nTouches start boundary exactly\n\n"
        "3\n00:00:08,000 --> 00:00:12,000\nStraddles start\n\n"
        "4\n00:00:12,000 --> 00:00:15,000\nEntirely inside\n\n"
        "5\n00:00:18,000 --> 00:00:25,000\nStraddles end\n\n"
        "6\n00:00:20,000 --> 00:00:22,000\nTouches end boundary exactly\n\n"
        "7\n00:00:25,000 --> 00:00:30,000\nAfter window\n\n"
    )
    subs = pysubs2.SSAFile.from_string(srt_text)
    assert len(subs) == 7

    trimmed, kept, dropped = _trim_and_rebase_subs(subs, start_sec=10.0, end_sec=20.0)
    assert kept == 3 and dropped == 4, (kept, dropped)
    assert len(trimmed) == 3

    straddle_start, inside, straddle_end = trimmed[0], trimmed[1], trimmed[2]
    assert straddle_start.plaintext == "Straddles start"
    assert straddle_start.start == 0 and straddle_start.end == 2000, (
        straddle_start.start, straddle_start.end)  # 8-12s -> clipped to 0-2s, never negative

    assert inside.plaintext == "Entirely inside"
    assert inside.start == 2000 and inside.end == 5000, (inside.start, inside.end)  # 12-15s -> 2-5s

    assert straddle_end.plaintext == "Straddles end"
    assert straddle_end.start == 8000 and straddle_end.end == 10000, (
        straddle_end.start, straddle_end.end)  # 18-25s -> clipped to 8-10s (end-start)

    # No upper bound (end_sec=None): the "after window" cue is now kept too, rebased.
    trimmed_no_end, kept_no_end, dropped_no_end = _trim_and_rebase_subs(
        subs, start_sec=10.0, end_sec=None)
    assert dropped_no_end == 2, dropped_no_end  # only the two before-window cues drop
    assert kept_no_end == 5, kept_no_end
    last = trimmed_no_end[-1]
    assert last.plaintext == "After window"
    assert last.start == 15000 and last.end == 20000, (last.start, last.end)  # 25-30s -> 15-20s

    print("_self_test_trim_and_rebase_subs: PASS")


def _self_test_run_time_range_default_unchanged():
    """THE key ADR-054 regression check: with --start-time/--end-time both absent (the
    DEFAULT), run() must produce EXACTLY the pre-ADR-054 behavior -- mkvmerge invoked
    with the ORIGINAL --srt path unchanged, and _trim_and_rebase_subs never even
    called. Covers both explicit None and the attributes being absent entirely (an
    older Namespace), mirroring _self_test_run_default_flag_off_split_format_unchanged's
    convention for ADR-053."""
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        def _fake_mkvmerge_run(cmd, **kwargs):
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as f:
                f.write(b"fake mkv output")
            return MagicMock(returncode=0, stdout="", stderr="")

        for case_name, args_kwargs in (
            ("explicit start_time/end_time=None", dict(start_time=None, end_time=None)),
            ("start_time/end_time attributes absent entirely", {}),
        ):
            output_path = path.join(tmpdir, f"out_{len(args_kwargs)}.mkv")
            base = dict(input=mkv_input, srt=srt_path, output=output_path,
                        language="en", track_name=None, format="half_sbs",
                        dual_eye_subtitles=False)
            base.update(args_kwargs)
            args = argparse.Namespace(**base)

            seen = {}

            def _capture(cmd, **kwargs):
                seen["cmd"] = cmd
                return _fake_mkvmerge_run(cmd, **kwargs)

            with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge"), \
                 patch(f"{__name__}._trim_and_rebase_subs") as mock_trim, \
                 patch.object(subprocess, "run", side_effect=_capture):
                rc = run(args)
                mock_trim.assert_not_called()

            assert rc == 0, (case_name, rc)
            cmd = seen["cmd"]
            muxed_path = cmd[cmd.index(mkv_input) - 1]
            assert muxed_path == srt_path, (
                f"{case_name}: default (both flags absent) must mux the ORIGINAL --srt "
                f"unchanged -- got {muxed_path!r}")

    print("_self_test_run_time_range_default_unchanged: PASS")


def _self_test_run_time_range_trims_and_rebases_end_to_end():
    """End-to-end (real pysubs2, real temp files, only mkvmerge itself mocked) test of
    ADR-054's run() path: with --start-time/--end-time given, proves mkvmerge is
    invoked with a NEW temp subtitle file (not the original --srt), and that file's
    actual on-disk content is correctly trimmed and rebased -- read back and parsed
    with pysubs2 directly, not just trusted from in-memory state."""
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        output_path = path.join(tmpdir, "out.mkv")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(
                "1\n00:00:00,000 --> 00:00:05,000\nBefore clip\n\n"
                "2\n00:01:16,000 --> 00:01:18,000\nInside clip\n\n"
                "3\n00:10:00,000 --> 00:10:05,000\nAfter clip\n\n"
            )

        args = argparse.Namespace(
            input=mkv_input, srt=srt_path, output=output_path,
            language="en", track_name=None, format="anaglyph",
            dual_eye_subtitles=False,
            start_time="00:01:15", end_time="00:06:15")

        seen = {}

        def _fake_mkvmerge_run(cmd, **kwargs):
            seen["cmd"] = cmd
            muxed_path = cmd[cmd.index(mkv_input) - 1]
            # Read the trimmed temp file's real on-disk content HERE, while it still
            # exists -- run()'s own finally block deletes it (CS-IO-001) as soon as
            # run() returns, before this test function gets control back.
            with open(muxed_path, "r", encoding="utf-8") as f:
                seen["muxed_content"] = f.read()
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as f:
                f.write(b"fake mkv output")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge"), \
             patch.object(subprocess, "run", side_effect=_fake_mkvmerge_run):
            rc = run(args)

        assert rc == 0, rc
        cmd = seen["cmd"]
        muxed_path = cmd[cmd.index(mkv_input) - 1]
        assert muxed_path != srt_path, "expected a new trimmed temp file, not the original --srt"

        import pysubs2
        result_subs = pysubs2.SSAFile.from_string(seen["muxed_content"])
        # "Before clip" (0:00-0:05) and "After clip" (10:00-10:05) are entirely outside
        # [1:15, 6:15) -> dropped. "Inside clip" (1:16-1:18) -> rebased by -75s -> 0:01-0:03.
        assert len(result_subs) == 1, [e.plaintext for e in result_subs]
        kept = result_subs[0]
        assert kept.plaintext == "Inside clip"
        assert kept.start == 1000 and kept.end == 3000, (kept.start, kept.end)

        # The intermediate trimmed temp file is cleaned up afterward (CS-IO-001).
        assert not path.exists(muxed_path)

    print("_self_test_run_time_range_trims_and_rebases_end_to_end: PASS")


def _self_test_run_time_range_validation_refuses_before_mkvmerge():
    """Bad --start-time/--end-time input (end<=start, unparseable) must refuse (return
    1) WITHOUT ever looking up mkvmerge -- same gating-order convention as this
    module's other pre-flight validation (_self_test_run_gating)."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_submux_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie.mkv")
        srt_path = path.join(tmpdir, "subs.srt")
        output_path = path.join(tmpdir, "out.mkv")
        with open(mkv_input, "wb") as f:
            f.write(b"0")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        def _args(**overrides):
            base = dict(input=mkv_input, srt=srt_path, output=output_path,
                        language="en", track_name=None, format="anaglyph",
                        dual_eye_subtitles=False, start_time=None, end_time=None)
            base.update(overrides)
            return argparse.Namespace(**base)

        with patch(f"{__name__}._find_mkvmerge") as mock_find:
            rc = run(_args(start_time="00:00:30", end_time="00:00:10"))
            assert rc == 1, rc
            mock_find.assert_not_called()

            rc = run(_args(start_time="not-a-time"))
            assert rc == 1, rc
            mock_find.assert_not_called()

            rc = run(_args(end_time="not-a-time"))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # a window with no overlapping cues at all -> refuses with a clear reason,
            # never reaching mkvmerge
            rc = run(_args(start_time="01:00:00", end_time="01:00:05"))
            assert rc == 1, rc
            mock_find.assert_not_called()

    print("_self_test_run_time_range_validation_refuses_before_mkvmerge: PASS")


def _run_self_tests():
    _self_test_format_detection()
    _self_test_iso639_lookup()
    _self_test_srt_validation()
    _self_test_probe_video_dimensions()
    _self_test_default_font_size()
    _self_test_stereo_positioning()
    _self_test_run_gating()
    _self_test_run_default_flag_off_split_format_unchanged()
    _self_test_run_flag_on_split_format_conversion()
    _self_test_run_font_size_default_and_explicit()
    _self_test_run_font_size_noop_without_dual_eye()
    _self_test_run_flag_on_non_split_format_noop()
    _self_test_run_flag_on_probe_failure_refuses_before_mkvmerge()
    _self_test_parse_time_range()
    _self_test_trim_and_rebase_subs()
    _self_test_run_time_range_default_unchanged()
    _self_test_run_time_range_trims_and_rebases_end_to_end()
    _self_test_run_time_range_validation_refuses_before_mkvmerge()
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
