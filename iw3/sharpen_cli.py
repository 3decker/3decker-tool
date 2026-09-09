"""python -m iw3.sharpen_cli -- standalone post-process Sharpen tool.

Real-world use case (see docs/ai/AI_DECISIONS.md ADR-063): a user already ran an iw3
conversion and has a finished SBS/TB/RGBD/Anaglyph 3D video, and now wants ADR-061's
Sharpen filter applied to it -- without re-running the expensive depth/stereo
conversion a second time (e.g. they only decided afterward that the picture looked a
little soft, or ADR-061 didn't exist yet when they originally converted).

This is a genuinely standalone tool, invoked as its own subprocess (same "separate
subprocess, never sharing state with the main app" convention as iw3.rife_cli /
iw3.reinject_hdr_cli / iw3.subtitle_mux_cli / iw3.stereo_mode_tag_cli -- see
docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001). It reuses `DE.apply_sharpen` (the
exact same edge-aware unsharp-mask function ADR-061 wired into the main pipeline's
`apply_divergence()`) directly -- never forked or re-implemented -- so this tool can
never drift out of sync with the in-pipeline filter's own behavior/bugfixes.

Per-eye handling (do not sharpen across the seam):
- Genuine two-eye split layouts (half_sbs, full_sbs, cross_eyed, vr90, half_tb,
  full_tb) are split into their left/right (or top/bottom) eye halves FIRST, each
  half is sharpened INDEPENDENTLY via `DE.apply_sharpen`, then the two results are
  rejoined -- exactly mirroring how the main pipeline already calls
  `DE.apply_sharpen(left_eye, right_eye, ...)` on two already-separate eye tensors
  (see `apply_divergence()` in iw3/utils.py). Splitting first, rather than running
  one blur/gradient pass over the whole packed frame, is what prevents the unsharp
  mask's 3x3 blur/Sobel kernels from ever reading pixels across the seam between the
  two eyes -- treating one eye's content as if it were spatially continuous with the
  other's would smear a false "detail" response right at the seam.
- rgbd / half_rgbd: per this project's own established finding (ADR-053, made for a
  different feature -- stereo-positioned subtitles -- but the same underlying fact
  about this format): `apply_rgbd()` in iw3/utils.py shows the "second half" of an
  RGBD frame is a DEPTH MAP (the source depth tensor, resized and expanded to 3
  channels), not a second eye view of the movie. Sharpening a depth map as if it
  were a picture makes no sense -- it has no photographic detail to enhance, only a
  smoothly-varying grayscale-ish gradient that an edge-aware unsharp mask would
  either leave alone (if genuinely smooth) or exaggerate into visible ringing right
  at a depth discontinuity (if not). So only the RGB half is ever sharpened here;
  the depth-map half is passed through byte-for-byte untouched.
- anaglyph: iw3.utils.py's own comment on this format says it plainly -- "Anaglyph is
  already a single merged image correct on any player" -- there is no seam, so the
  whole frame is sharpened directly as one image, no eye-splitting needed.

Format auto-detection: reuses the exact same filename-tag convention
subtitle_mux_cli.py / stereo_mode_tag_cli.py already established (`auto` default,
matched against iw3.utils's own canonical suffix constants, longest-tag-first,
explicit --format override available, hard refusal with no guessing when
inconclusive) -- duplicated locally rather than imported cross-module, the same
"each of these standalone CLI tools imports only from iw3.utils, never from each
other" convention those two modules already document.

Scope (mirrors subtitle_mux_cli.py / stereo_mode_tag_cli.py's own ADR-032/ADR-033
scope restriction, for the same reason):
- .mkv input/output only. Sharpening always re-encodes the video track (an unsharp
  mask is a per-pixel filter, not a metadata edit), so the OTHER tracks (audio,
  subtitles, chapters, attachments, tags) must be losslessly copied back in from the
  original around the freshly re-encoded video -- mkvmerge's per-file track-type
  selection flags (`-A -S -M -T --no-chapters --no-global-tags` on the sharpened
  video-only file, `--no-video` on the original) are what guarantee that swap is
  byte-for-byte exact, and that is only available for Matroska. If --input is not a
  .mkv file, this tool refuses outright rather than attempting a lossy container
  conversion no one asked for.
- Never modifies --input -- always writes a new file at --output (CS-IO-001-style
  temp-then-replace, same convention as every other standalone tool in this family).
"""
import argparse
import os
import subprocess
import sys
from os import path

import torch

import nunif.utils.video as VU
from nunif.device import create_device
from . import depth_effects as DE
from .utils import (
    _find_mkvmerge,
    FULL_SBS_SUFFIX, HALF_SBS_SUFFIX, FULL_TB_SUFFIX, HALF_TB_SUFFIX,
    CROSS_EYED_SUFFIX, RGBD_SUFFIX, HALF_RGBD_SUFFIX, VR180_SUFFIX, ANAGLYPH_SUFFIX,
)


# Same canonical suffix-tag table as iw3.subtitle_mux_cli / iw3.stereo_mode_tag_cli's
# own _SUFFIX_TO_FORMAT -- duplicated rather than imported cross-module (see module
# docstring), but built from the exact same iw3.utils constants, so it can never
# silently drift from what iw3 itself actually writes into output filenames.
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

# Resolved --format values whose watchable frame is a genuine split-eye layout, and
# the axis it's split on -- see module docstring / ADR-053's identical geometry
# finding, reused here. half_sbs/full_sbs/cross_eyed/vr90 pack left-eye and right-eye
# side by side (HORIZONTAL split, seam at width/2); half_tb/full_tb stack them top
# over bottom (VERTICAL split, seam at height/2).
_HORIZONTAL_SPLIT_FORMATS = frozenset({"half_sbs", "full_sbs", "cross_eyed", "vr90"})
_VERTICAL_SPLIT_FORMATS = frozenset({"half_tb", "full_tb"})

# rgbd/half_rgbd also pack their two halves side by side (apply_rgbd()'s output is
# always concatenated along the width axis in postprocess_image() -- iw3/utils.py:
# `sbs = torch.cat([left_eye, right_eye], dim=2)`, same "SideBySide or RGBD" branch,
# including for half_rgbd which additionally halves the packed width like half_sbs
# does) -- but the second half is a DEPTH MAP, not a second eye view (see module
# docstring), so it gets its own dedicated no-sharpen-on-the-right-half handling
# below rather than being folded into _HORIZONTAL_SPLIT_FORMATS.
_RGBD_FORMATS = frozenset({"rgbd", "half_rgbd"})

# anaglyph is deliberately in neither set above -- it is a single merged 2D image
# with no seam at all (see module docstring), handled as a distinct whole-frame case.


def _detect_format_from_filename(filename):
    """Returns one of the values in _SUFFIX_TO_FORMAT if a known iw3 filename suffix
    tag is found (case-insensitive, longest tag checked first so e.g. the full-SBS
    tag isn't masked by the half-SBS tag that is a substring of it), or None if
    inconclusive."""
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


def _axis_for_format(resolved_format):
    """Returns "sbs" (horizontal/width split) or "tb" (vertical/height split) for a
    format whose packed frame is split into two halves at all, or None for anaglyph
    (no split)."""
    if resolved_format in _HORIZONTAL_SPLIT_FORMATS or resolved_format in _RGBD_FORMATS:
        return "sbs"
    elif resolved_format in _VERTICAL_SPLIT_FORMATS:
        return "tb"
    return None


def _split_frame_tensor(x, axis):
    """Splits a decoded frame tensor (C,H,W, as produced by
    nunif.utils.video.to_tensor) into its two physical halves along the packing
    axis: "sbs" (vertical seam -- left half, right half, split on the WIDTH/last
    axis) or "tb" (horizontal seam -- top half, bottom half, split on the
    HEIGHT/second-to-last axis). An odd packed dimension gives the remainder pixel
    to the second half, so _join_frame_tensor always reconstructs the exact original
    frame with no pixel loss.

    This mirrors iw3.utils.split_stereo_frame's exact geometry (same axis
    convention, same floor-division/remainder-to-second-half rule) but is
    reimplemented here for CHW torch tensors rather than calling that function
    directly, since it operates on HWC/HW-first numpy arrays and DE.apply_sharpen
    needs torch tensors for its conv2d-based blur/gradient ops. This is the same
    domain tradeoff iw3.waifu2x_upscale_stereo_cli.py's own _split_video already
    accepts -- it mirrors the identical geometry via an ffmpeg crop filter instead
    of calling the numpy function, because ffmpeg filters are the right tool for a
    video-level crop the same way a plain tensor slice is the right tool here."""
    if axis == "sbs":
        mid = x.shape[-1] // 2
        return x[..., :mid], x[..., mid:]
    elif axis == "tb":
        mid = x.shape[-2] // 2
        return x[..., :mid, :], x[..., mid:, :]
    raise ValueError(f"unknown stereo split axis: {axis!r}")


def _join_frame_tensor(a, b, axis):
    """Inverse of _split_frame_tensor -- concatenates the two halves back along the
    same axis. Round-trips _split_frame_tensor pixel-for-pixel when neither half was
    resized in between (see _self_test_split_join_round_trip)."""
    if axis == "sbs":
        return torch.cat([a, b], dim=-1)
    elif axis == "tb":
        return torch.cat([a, b], dim=-2)
    raise ValueError(f"unknown stereo split axis: {axis!r}")


def sharpen_frame_tensor(x, resolved_format, strength, detail_percentile=95.0):
    """Applies ADR-061's DE.apply_sharpen to one decoded frame tensor (C,H,W, float
    in [0, 1], as produced by nunif.utils.video.to_tensor), respecting
    resolved_format's real per-eye/per-half geometry -- see module docstring for the
    full reasoning per format family:
      - genuine two-eye split (half_sbs/full_sbs/cross_eyed/vr90/half_tb/full_tb):
        split into the two eye halves FIRST, sharpen each independently via
        DE.apply_sharpen (never across the seam), then rejoin.
      - rgbd/half_rgbd: only the RGB (first) half is sharpened; the depth-map
        (second) half is returned byte-for-byte untouched.
      - anaglyph (axis is None): already a single merged 2D image -- sharpened as
        one whole frame, no split at all.

    strength<=0.0 is an exact no-op end to end: DE.apply_sharpen itself returns its
    inputs unchanged (by identity) at strength<=0.0, and _split_frame_tensor/
    _join_frame_tensor are a lossless pure slice+concat round trip, so the returned
    tensor is pixel-identical to `x` for every format at strength=0.0 -- proving the
    split/rejoin machinery itself introduces no distortion on its own."""
    axis = _axis_for_format(resolved_format)
    if resolved_format in _RGBD_FORMATS:
        rgb, depth = _split_frame_tensor(x, axis)
        sharpened_rgb, _ = DE.apply_sharpen(rgb.unsqueeze(0), rgb.unsqueeze(0),
                                             strength=strength, detail_percentile=detail_percentile)
        return _join_frame_tensor(sharpened_rgb.squeeze(0), depth, axis)
    elif axis is None:
        # anaglyph -- DE.apply_sharpen is a pure per-image function (see its own
        # docstring: left_eye/right_eye are sharpened completely independently, with
        # no cross-referencing between them), so passing the same whole frame as
        # both arguments sharpens it once, through the exact same code path, with no
        # forked/duplicated logic.
        sharpened, _ = DE.apply_sharpen(x.unsqueeze(0), x.unsqueeze(0),
                                         strength=strength, detail_percentile=detail_percentile)
        return sharpened.squeeze(0)
    else:
        a, b = _split_frame_tensor(x, axis)
        sharpened_a, sharpened_b = DE.apply_sharpen(a.unsqueeze(0), b.unsqueeze(0),
                                                      strength=strength,
                                                      detail_percentile=detail_percentile)
        return _join_frame_tensor(sharpened_a.squeeze(0), sharpened_b.squeeze(0), axis)


def _format_cmd(cmd):
    return " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.sharpen_cli",
        description=(
            "Apply iw3's Sharpen filter (ADR-061, an edge-aware unsharp mask) to an "
            "already-converted 3D .mkv video, without re-running depth/stereo conversion. "
            "A genuine two-eye layout (half_sbs/full_sbs/cross_eyed/vr90/half_tb/full_tb) is "
            "split into its eye halves and each is sharpened independently, never across the "
            "seam. rgbd/half_rgbd only sharpens the RGB half -- the depth-map half is left "
            "untouched. anaglyph (already a single merged image) is sharpened as one whole "
            "frame. Every other track (audio, subtitles, chapters, attachments) is copied "
            "through unchanged -- only the video is re-encoded. Never modifies --input -- "
            "always writes a new file at --output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                         help="Path to the already-converted 3D video to sharpen. Must be a "
                              ".mkv file -- this tool does not convert containers. Read-only, "
                              "never modified.")
    parser.add_argument("--output", "-o", type=str, required=True,
                         help="Path to write the new sharpened copy to. Must be a different "
                              "path from --input -- this tool never overwrites the input.")
    parser.add_argument("--format", type=str, default="auto", choices=list(_VALID_FORMATS),
                         help="Stereo/output layout of --input -- any of iw3's own watchable "
                              "Stereo Format outputs: half_sbs, full_sbs, half_tb, full_tb, "
                              "cross_eyed, vr90, rgbd, half_rgbd, anaglyph. 'auto' (default) "
                              "tries to detect this from --input's filename using the same "
                              "suffix tags iw3 itself writes (e.g. '_LR', '_TB', "
                              "'_LRF_Full_SBS', '_TBF_fulltb', '_RLF_cross', '_180x180_LR', "
                              "'_RGBD', '_HRGBD', '_redcyan'). If detection is inconclusive, "
                              "this tool refuses and asks you to pass this explicitly -- it "
                              "never guesses.")
    parser.add_argument("--sharpen-strength", type=float, default=0.5,
                         help="how strong the unsharp-mask pass is (0.0=no effect, "
                              "1.0=strongest) -- same range/default as the in-pipeline "
                              "--sharpen-strength flag (ADR-061), for consistency.")
    parser.add_argument("--crf", type=str, default="16",
                         help="x264/x265 constant-rate-factor for the re-encoded video track "
                              "(lower = higher quality/larger file).")
    parser.add_argument("--preset", type=str, default="medium",
                         help="x264/x265 encoder preset for the re-encoded video track.")
    parser.add_argument("--gpu", type=int, default=0,
                         help="GPU device index to run the sharpen filter's conv2d ops on. "
                              "-1 forces CPU.")
    return parser


def _sharpen_video(input_path, output_path, resolved_format, strength, detail_percentile,
                    device, crf, preset):
    """Single-pass decode -> sharpen (per resolved_format's geometry) -> encode, for
    the ENTIRE input video. Writes output_path as a full container (any extraneous
    audio track VU.process_video may also copy into it is irrelevant -- the caller
    only ever takes this file's VIDEO track back out via mkvmerge). Follows
    iw3.rife_cli / iw3.waifu2x_upscale_stereo_cli's exact VU.process_video
    template."""
    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            return None
        x = VU.to_tensor(frame, device=device)
        return sharpen_frame_tensor(x, resolved_format, strength, detail_percentile)

    def config_callback(sw_format):
        return VU.VideoOutputConfig(
            fps=None,
            output_fps=None,
            options={"preset": preset, "crf": crf},
        )

    VU.process_video(
        input_path, output_path, frame_callback,
        config_callback=config_callback,
        title="Sharpen", device=device,
    )


def run(args):
    input_path = str(args.input)
    output_path = str(args.output)

    if not path.exists(input_path):
        print(f"ERROR: --input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if path.splitext(input_path)[1].lower() != ".mkv":
        print(
            f"ERROR: --input must be an .mkv file (got '{input_path}'). This tool only "
            f"supports MKV input/output -- mkvmerge's per-file track selection is what "
            f"guarantees every other track (audio/subtitles/chapters/attachments) is copied "
            f"through byte-for-byte unchanged around the freshly re-encoded video, and that "
            f"is only available for Matroska (same scope restriction "
            f"iw3.subtitle_mux_cli/iw3.stereo_mode_tag_cli already use, and for the same "
            f"reason). Remux your file to .mkv first (e.g. with mkvmerge) before using this "
            f"tool.",
            file=sys.stderr)
        return 1

    if path.abspath(output_path) == path.abspath(input_path):
        print("ERROR: --output must be a different path from --input -- this tool never "
              "overwrites the input.", file=sys.stderr)
        return 1

    strength = float(args.sharpen_strength)
    if not (0.0 <= strength <= 1.0):
        print(f"ERROR: --sharpen-strength must be between 0.0 and 1.0 (got {strength}).",
              file=sys.stderr)
        return 1

    resolved_format, format_error = _resolve_format(args.format, input_path)
    if format_error:
        print(format_error, file=sys.stderr)
        return 1
    print(f"[sharpen] resolved stereo format: {resolved_format}"
          f"{' (auto-detected from filename)' if args.format == 'auto' else ' (explicit)'}",
          file=sys.stderr)

    if resolved_format in _RGBD_FORMATS:
        print(f"[sharpen] '{resolved_format}' packs a depth map in its second half, not a "
              f"second eye view (see apply_rgbd() in iw3/utils.py, and ADR-053's identical "
              f"finding for this same format) -- only the RGB half will be sharpened; the "
              f"depth-map half is left byte-for-byte untouched.", file=sys.stderr)
    elif resolved_format == "anaglyph":
        print("[sharpen] 'anaglyph' is already a single merged 2D image -- sharpening the "
              "whole frame directly, no eye-splitting needed.", file=sys.stderr)
    else:
        print(f"[sharpen] '{resolved_format}' is a genuine two-eye split -- splitting into "
              f"eye halves and sharpening each independently (never across the seam) before "
              f"rejoining.", file=sys.stderr)

    if strength <= 0.0:
        print("[sharpen] --sharpen-strength is 0.0 -- this is an exact no-op (the video "
              "track will still be fully re-decoded/re-encoded, but every pixel is left "
              "unchanged).", file=sys.stderr)

    mkvmerge_bin = _find_mkvmerge()
    if not mkvmerge_bin:
        print("ERROR: mkvmerge (MKVToolNix) was not found -- cannot remux. Expected it "
              "bundled alongside ffmpeg, or on PATH.", file=sys.stderr)
        return 1

    device = create_device(args.gpu)

    out_dir = path.dirname(path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output = path.splitext(output_path)[0] + ".sharpen_tmp" + path.splitext(output_path)[1]
    sharpened_video_tmp = path.splitext(output_path)[0] + ".sharpen_video_tmp.mkv"

    try:
        print(f"[sharpen] [1/2] decoding, sharpening (strength={strength}), and "
              f"re-encoding the video track...", file=sys.stderr)
        _sharpen_video(input_path, sharpened_video_tmp, resolved_format, strength,
                        95.0, device, args.crf, args.preset)

        print("[sharpen] [2/2] remuxing the sharpened video track back together with "
              "every other original track (audio/subtitles/chapters/attachments) "
              "unchanged...", file=sys.stderr)
        # mkvmerge: the per-file options before EACH input restrict what's taken FROM
        # that file. "-A -S -M -T --no-chapters --no-global-tags" before
        # sharpened_video_tmp strips everything but its video track (that temp file
        # may also carry a copied/re-encoded audio track from VU.process_video --
        # irrelevant, it's excluded here regardless). "--no-video" before input_path
        # strips only video from the original, keeping every other track (audio,
        # subtitles, chapters, attachments, track tags, global tags) intact and
        # untouched -- the exact swap-only-the-video-track pattern already used by
        # iw3.utils._remux_injected_hevc's own mkvmerge branch for retroactive HDR
        # reinjection.
        cmd = [
            mkvmerge_bin, "-o", tmp_output,
            "-A", "-S", "-M", "-T", "--no-chapters", "--no-global-tags",
            sharpened_video_tmp,
            "--no-video",
            input_path,
        ]
        print(f"[sharpen] running: {_format_cmd(cmd)}", file=sys.stderr)
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
            return 1
        if not path.exists(tmp_output):
            print("ERROR: mkvmerge reported success but produced no output file.", file=sys.stderr)
            return 1

        os.replace(tmp_output, output_path)
        print(f"[sharpen] done -- wrote {output_path}", file=sys.stderr)
        return 0
    finally:
        for f in (sharpened_video_tmp, tmp_output):
            if f and path.exists(f):
                try:
                    os.remove(f)
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


def _self_test_split_join_round_trip():
    """Synthetic, GPU-free proof that _split_frame_tensor/_join_frame_tensor are a
    lossless pure slice+concat round trip -- including an odd packed dimension
    (remainder pixel goes to the second half, mirroring iw3.utils.split_stereo_frame's
    own documented rule)."""
    torch.manual_seed(0)
    for axis, shape in (("sbs", (3, 40, 64)), ("tb", (3, 65, 48))):
        x = torch.rand(*shape)
        a, b = _split_frame_tensor(x, axis)
        rejoined = _join_frame_tensor(a, b, axis)
        assert rejoined.shape == x.shape, (axis, rejoined.shape, x.shape)
        assert torch.equal(rejoined, x), f"{axis}: round-trip is not pixel-identical"

    # Odd dimension on the split axis -- remainder pixel goes to the second half.
    x = torch.rand(3, 41, 63)
    a, b = _split_frame_tensor(x, "sbs")
    assert a.shape[-1] == 31 and b.shape[-1] == 32, (a.shape, b.shape)
    assert torch.equal(_join_frame_tensor(a, b, "sbs"), x)

    x = torch.rand(3, 41, 63)
    a, b = _split_frame_tensor(x, "tb")
    assert a.shape[-2] == 20 and b.shape[-2] == 21, (a.shape, b.shape)
    assert torch.equal(_join_frame_tensor(a, b, "tb"), x)

    print("_self_test_split_join_round_trip: PASS")


def _make_edge_frame(h=32, w=48):
    """Synthetic (3,H,W) frame with a real hard edge in EACH half (left/right AND
    top/bottom), so a strength=0 test can prove a true no-op, and a strength>0 test
    has genuine detail to sharpen in every quadrant."""
    x = torch.zeros(3, h, w)
    x[:, :, w // 4:] = 1.0  # vertical edge inside the left half
    x[:, :, w // 2 + w // 4:] = 0.3  # vertical edge inside the right half
    x[: , h // 4, :] = 0.6  # a horizontal streak crossing both halves
    return x.clamp(0, 1)


def _self_test_sharpen_frame_tensor_noop_at_zero_strength():
    """strength=0.0 must be an EXACT (pixel-identical) no-op end to end, for every
    format family -- proves the split/rejoin machinery itself introduces no
    distortion on its own (DE.apply_sharpen already guarantees the sharpen math
    itself is a no-op at strength<=0.0, per ADR-061; this confirms sharpen_cli's own
    split/join wrapper doesn't add any)."""
    x = _make_edge_frame()
    for fmt in ("half_sbs", "full_sbs", "cross_eyed", "vr90", "half_tb", "full_tb",
                "rgbd", "half_rgbd", "anaglyph"):
        out = sharpen_frame_tensor(x, fmt, strength=0.0)
        assert out.shape == x.shape, (fmt, out.shape, x.shape)
        assert torch.equal(out, x), f"{fmt}: strength=0.0 must be pixel-identical to input"
    print("_self_test_sharpen_frame_tensor_noop_at_zero_strength: PASS")


def _self_test_sharpen_frame_tensor_rgbd_leaves_depth_untouched():
    """rgbd/half_rgbd: at a real positive strength, the RGB (first) half must
    actually change at its own edge, while the depth-map (second) half stays
    EXACTLY byte-for-byte identical to the input -- the concrete regression guard
    for ADR-053's "the second half is a depth map, not a picture" finding."""
    for fmt in ("rgbd", "half_rgbd"):
        x = _make_edge_frame()
        rgb_in, depth_in = _split_frame_tensor(x, "sbs")
        out = sharpen_frame_tensor(x, fmt, strength=1.0)
        rgb_out, depth_out = _split_frame_tensor(out, "sbs")
        assert torch.equal(depth_out, depth_in), f"{fmt}: depth half must be left untouched"
        assert not torch.equal(rgb_out, rgb_in), f"{fmt}: RGB half must actually be sharpened"
    print("_self_test_sharpen_frame_tensor_rgbd_leaves_depth_untouched: PASS")


def _self_test_sharpen_frame_tensor_anaglyph_whole_frame():
    """anaglyph: no split at all -- the output must equal calling DE.apply_sharpen
    directly on the WHOLE frame (proving sharpen_frame_tensor's anaglyph branch
    never splits it), including across what would be a seam location for a split
    format."""
    x = _make_edge_frame()
    out = sharpen_frame_tensor(x, "anaglyph", strength=0.8)
    expected, _ = DE.apply_sharpen(x.unsqueeze(0), x.unsqueeze(0), strength=0.8)
    expected = expected.squeeze(0)
    assert torch.equal(out, expected), "anaglyph must sharpen the whole frame directly, unsplit"
    assert not torch.equal(out, x), "anaglyph at strength=0.8 must actually change the frame"
    print("_self_test_sharpen_frame_tensor_anaglyph_whole_frame: PASS")


def _self_test_sharpen_frame_tensor_seam_independence():
    """Genuine two-eye split formats: the LEFT half of the output must depend ONLY
    on the left half of the input, never on what's in the right half -- the concrete
    regression guard against "sharpening across the seam". Builds two frames that
    are IDENTICAL in their left half but different in their right half, and confirms
    the sharpened left half comes out identical between them."""
    h, w = 32, 48
    left = torch.rand(3, h, w // 2)
    right_a = torch.zeros(3, h, w - w // 2)
    right_b = torch.ones(3, h, w - w // 2)
    frame_a = torch.cat([left, right_a], dim=-1)
    frame_b = torch.cat([left, right_b], dim=-1)

    for fmt in ("half_sbs", "full_sbs"):
        out_a = sharpen_frame_tensor(frame_a, fmt, strength=1.0)
        out_b = sharpen_frame_tensor(frame_b, fmt, strength=1.0)
        left_out_a, _ = _split_frame_tensor(out_a, "sbs")
        left_out_b, _ = _split_frame_tensor(out_b, "sbs")
        assert torch.equal(left_out_a, left_out_b), (
            f"{fmt}: left eye output must be independent of the right eye's content")

    top = torch.rand(3, h // 2, w)
    bottom_a = torch.zeros(3, h - h // 2, w)
    bottom_b = torch.ones(3, h - h // 2, w)
    frame_a = torch.cat([top, bottom_a], dim=-2)
    frame_b = torch.cat([top, bottom_b], dim=-2)
    for fmt in ("half_tb", "full_tb"):
        out_a = sharpen_frame_tensor(frame_a, fmt, strength=1.0)
        out_b = sharpen_frame_tensor(frame_b, fmt, strength=1.0)
        top_out_a, _ = _split_frame_tensor(out_a, "tb")
        top_out_b, _ = _split_frame_tensor(out_b, "tb")
        assert torch.equal(top_out_a, top_out_b), (
            f"{fmt}: top eye output must be independent of the bottom eye's content")

    print("_self_test_sharpen_frame_tensor_seam_independence: PASS")


def _self_test_run_gating():
    """Synthetic/mocked test of run()'s pre-processing gates (missing input, non-mkv
    input, output collision, bad --sharpen-strength, inconclusive auto-format) --
    proves each refuses before mkvmerge/video processing is ever invoked."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_sharpen_selftest_") as tmpdir:
        mkv_input = path.join(tmpdir, "movie_LR.mkv")
        mp4_input = path.join(tmpdir, "movie_LR.mp4")
        ambiguous_input = path.join(tmpdir, "movie.mkv")
        missing_input = path.join(tmpdir, "does_not_exist.mkv")
        output = path.join(tmpdir, "movie_LR_sharpened.mkv")

        for p in (mkv_input, mp4_input, ambiguous_input):
            with open(p, "wb") as f:
                f.write(b"0")

        def _args(**overrides):
            base = dict(input=mkv_input, output=output, format="auto",
                        sharpen_strength=0.5, crf="16", preset="medium", gpu=0)
            base.update(overrides)
            return argparse.Namespace(**base)

        with patch(f"{__name__}._find_mkvmerge") as mock_find, \
             patch(f"{__name__}._sharpen_video") as mock_sharpen:
            # missing input -> refuse before ever looking for mkvmerge
            rc = run(_args(input=missing_input))
            assert rc == 1, rc
            mock_find.assert_not_called()
            mock_sharpen.assert_not_called()

            # non-.mkv input -> refuse before ever looking for mkvmerge
            rc = run(_args(input=mp4_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # output collides with input -> refuse
            rc = run(_args(output=mkv_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # bad --sharpen-strength -> refuse before format resolution/mkvmerge lookup
            rc = run(_args(sharpen_strength=1.5))
            assert rc == 1, rc
            mock_find.assert_not_called()
            rc = run(_args(sharpen_strength=-0.1))
            assert rc == 1, rc
            mock_find.assert_not_called()

            # ambiguous filename with format=auto -> refuse before mkvmerge lookup
            rc = run(_args(input=ambiguous_input))
            assert rc == 1, rc
            mock_find.assert_not_called()

        # happy path: mkvmerge found, video sharpening + mkvmerge remux both invoked
        # with the right arguments (mkvmerge itself is mocked, no real subprocess).
        with patch(f"{__name__}._find_mkvmerge", return_value="mkvmerge.exe"), \
             patch(f"{__name__}._sharpen_video") as mock_sharpen, \
             patch.object(subprocess, "run") as mock_run:
            def _fake_sharpen_video(input_path, output_path, *a, **kw):
                with open(output_path, "wb") as f:
                    f.write(b"fake-sharpened-video")
            mock_sharpen.side_effect = _fake_sharpen_video

            def _fake_mkvmerge_run(cmd, **kw):
                out_path = cmd[cmd.index("-o") + 1]
                with open(out_path, "wb") as f:
                    f.write(b"fake-merged-output")
                from unittest.mock import MagicMock
                return MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = _fake_mkvmerge_run

            rc = run(_args())
            assert rc == 0, rc
            mock_sharpen.assert_called_once()
            assert mock_sharpen.call_args[0][2] == "half_sbs"  # resolved_format
            mock_run.assert_called_once()
            called_cmd = mock_run.call_args[0][0]
            assert "-A" in called_cmd and "--no-video" in called_cmd
            assert path.exists(output)
            with open(output, "rb") as f:
                assert f.read() == b"fake-merged-output"

    print("_self_test_run_gating: PASS")


def _run_self_tests():
    _self_test_format_detection()
    _self_test_split_join_round_trip()
    _self_test_sharpen_frame_tensor_noop_at_zero_strength()
    _self_test_sharpen_frame_tensor_rgbd_leaves_depth_untouched()
    _self_test_sharpen_frame_tensor_anaglyph_whole_frame()
    _self_test_sharpen_frame_tensor_seam_independence()
    _self_test_run_gating()
    print("All sharpen_cli self-tests PASSED")


def main(argv=None):
    # Special-cased ahead of the real parser (rather than added as a parser argument)
    # so it can run without also satisfying --input/--output=required -- see
    # docs/ai/TEST_MATRIX.md's "isolated (no-GPU) test pattern" convention.
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
