"""python -m iw3.fx_overlay_cli -- "FX Pop-Out Overlay": composite a flat overlay clip
(rain/snow/fire/smoke/sparks/etc.) onto an ALREADY-FINISHED stereo video (SBS or TB),
giving the overlay its own artificial binocular parallax so it visually pops out in front
of the screen -- completely independent of iw3's own AI depth-based conversion pipeline.

Why this exists as a separate tool: the depth model can't give volumetric/particle effects
(rain, snow, fire, smoke) their own real depth -- they're thin, semi-transparent and
numerous, so the model blends them into whatever is behind them (see this project's own
research note in iw3/docs/best_quality_3d_conversion_notes.md, around "Volumetric effects
... can't be roto'd -- need separate handling"). This tool sidesteps that entirely: it
doesn't look at the scene's depth at all. It just gives the WHOLE overlay clip one constant,
adjustable "distance" in front of the screen by shifting it a different amount per eye.

Pipeline:
  1. ffmpeg splits the finished stereo video into its two eye halves, using the same crop
     geometry idea as the SBS-to-MVC tool's own eye_filter() (see eye_regions() below).
     --layout picks full_sbs/half_sbs/full_tb/half_tb, auto-detected via
     sbs_to_mvc_cli.guess_layout() (reused, not reimplemented) when not given.
  2. The overlay clip is stretched (stretch-to-fit, no aspect-ratio lock -- these are
     texture-like effects) to an OVERSIZED width of (eye_width + 2*shift_px), then cropped
     to two different eye_width-wide windows -- one starting at x=0 (left eye), one starting
     at x=2*shift_px (right eye) -- so the exact same overlay content lands at a different
     horizontal position in each eye's copy. shift_px comes from --pop-strength; see
     pop_strength_to_shift_px() for the exact formula, and docs/ai/AI_DECISIONS.md for the
     real, synthetic-test-verified proof that THIS assignment (left=0, right=2*shift_px)
     produces crossed/pop-out disparity, not recede-behind-the-screen disparity, matching
     this codebase's own near/far convention in iw3/backward_warp.py.
  3. Each eye's overlay crop is blended onto that eye's half of the real picture
     (--blend-mode: 'screen' for a black-background/additive overlay, 'alpha' for an
     overlay that carries a real alpha channel), scaled by --opacity.
  4. The two blended eye halves are stacked back into the original SBS/TB layout.
  5. Audio and subtitle streams are passed through untouched (stream copy) -- only the
     video is touched.

This is a flat, fixed-depth trick: the overlay doesn't interact with the scene's real depth
at all, it is simply pinned at one constant distance in front of the screen for its entire
duration. --start-time/--end-time trim which part of the MAIN video is processed (the
overlay itself always starts from its own frame 0 -- it is a generic repeating texture, not
footage that needs to stay time-synced to the main video).
"""
import argparse
import os
import subprocess
import sys
from os import path

from .sbs_to_mvc_cli import guess_layout, probe_video, _cuda_available
from .utils import _get_ffmpeg_bin

LAYOUTS = ("full_sbs", "half_sbs", "full_tb", "half_tb")
BLEND_MODES = ("screen", "alpha")

# pop_strength (0.0-1.0) -> shift_px = round(pop_strength * eye_width * POP_STRENGTH_SHIFT_FACTOR).
# shift_px is the ONE-EYE shift (in pixels, measured within the oversized overlay -- see
# eye_regions()/build_filtergraph()); the two eyes end up 2*shift_px apart from EACH OTHER,
# which is the real, perceived parallax. 0.05 was chosen so a full-strength (1.0) overlay
# shifts by up to 5% of the eye's own width one-sided (10% total between the two eyes) --
# comparable in scale to this project's own Divergence slider (ROW_FLOW_V3_MAX_DIVERGENCE =
# 5.0, i.e. real depth-based content can already move up to ~5% of the width at maximum
# divergence), so a full-strength overlay pops out about as far as the strongest real
# depth-based parallax this pipeline ever produces elsewhere, without needing the overlay's
# own content to ever leave the frame (the oversized scale/crop keeps every pixel covered
# for any pop_strength in [0.0, 1.0]).
POP_STRENGTH_SHIFT_FACTOR = 0.05


class FxOverlayError(Exception):
    pass


def pop_strength_to_shift_px(pop_strength, eye_width):
    """shift_px = round(pop_strength * eye_width * POP_STRENGTH_SHIFT_FACTOR). Raises
    FxOverlayError outside [0.0, 1.0] -- see the module docstring for what shift_px means."""
    if not (0.0 <= pop_strength <= 1.0):
        raise FxOverlayError(f"--pop-strength must be between 0.0 and 1.0, got {pop_strength}")
    return max(0, round(pop_strength * eye_width * POP_STRENGTH_SHIFT_FACTOR))


def eye_regions(layout, width, height):
    """(eye_w, eye_h, (left_x, left_y), (right_x, right_y)) -- the crop origin/size for each
    eye's own half of a packed SBS/TB frame. Full vs Half only changes how squeezed the
    picture inside each half is (never the crop geometry) -- and since this tool recombines
    the two halves without ever rescaling them, that distinction does not matter here, unlike
    sbs_to_mvc_cli's own eye_filter() (which DOES rescale, so it needs it). Left eye = left
    half for SBS, top half for TB (same packing convention as iw3.utils.apply_divergence's
    own torch.cat([left_eye, right_eye], ...) -- dim=2 puts left in the left half for SBS,
    dim=1 puts left on top for TB)."""
    if layout not in LAYOUTS:
        raise FxOverlayError(f"unknown layout {layout!r}; choose from {LAYOUTS}")
    if layout in ("full_sbs", "half_sbs"):
        eye_w, eye_h = width // 2, height
        return eye_w, eye_h, (0, 0), (eye_w, 0)
    eye_w, eye_h = width, height // 2
    return eye_w, eye_h, (0, 0), (0, eye_h)


def build_filtergraph(layout, width, height, shift_px, blend_mode, opacity):
    """The real -filter_complex string: split+crop the base video into its two eyes, scale
    the overlay (input 1) to an oversized width and crop two different windows out of it (one
    per eye, 2*shift_px apart), blend each eye's crop onto that eye per blend_mode/opacity,
    then stack the two results back into the original layout. Output pad is '[vout]'."""
    if blend_mode not in BLEND_MODES:
        raise FxOverlayError(f"unknown blend mode {blend_mode!r}; choose from {BLEND_MODES}")
    if not (0.0 <= opacity <= 1.0):
        raise FxOverlayError(f"--opacity must be between 0.0 and 1.0, got {opacity}")
    eye_w, eye_h, (lx, ly), (rx, ry) = eye_regions(layout, width, height)
    overlay_w = eye_w + 2 * shift_px

    parts = [
        "[0:v]split=2[fx_base_l][fx_base_r]",
        f"[fx_base_l]crop={eye_w}:{eye_h}:{lx}:{ly}[fx_eye_l]",
        f"[fx_base_r]crop={eye_w}:{eye_h}:{rx}:{ry}[fx_eye_r]",
        f"[1:v]scale={overlay_w}:{eye_h}:flags=lanczos,setsar=1,split=2[fx_ov_l][fx_ov_r]",
        # Left eye crop window: [0, eye_w) -- x-offset 0.
        f"[fx_ov_l]crop={eye_w}:{eye_h}:0:0[fx_ov_lc]",
        # Right eye crop window: [2*shift_px, 2*shift_px + eye_w) -- x-offset 2*shift_px.
        # See the module docstring / ADR for why THIS eye gets the positive offset.
        f"[fx_ov_r]crop={eye_w}:{eye_h}:{2 * shift_px}:0[fx_ov_rc]",
    ]
    if blend_mode == "screen":
        parts.append(f"[fx_eye_l][fx_ov_lc]blend=all_mode=screen:all_opacity={opacity:.4f}[fx_out_l]")
        parts.append(f"[fx_eye_r][fx_ov_rc]blend=all_mode=screen:all_opacity={opacity:.4f}[fx_out_r]")
    else:  # alpha
        parts.append(f"[fx_ov_lc]colorchannelmixer=aa={opacity:.4f}[fx_ov_la]")
        parts.append(f"[fx_ov_rc]colorchannelmixer=aa={opacity:.4f}[fx_ov_ra]")
        parts.append("[fx_eye_l][fx_ov_la]overlay=x=0:y=0[fx_out_l]")
        parts.append("[fx_eye_r][fx_ov_ra]overlay=x=0:y=0[fx_out_r]")
    if layout in ("full_sbs", "half_sbs"):
        parts.append("[fx_out_l][fx_out_r]hstack=inputs=2[vout]")
    else:
        parts.append("[fx_out_l][fx_out_r]vstack=inputs=2[vout]")
    return ";".join(parts)


def _video_codec_args():
    """GPU (hevc_nvenc) when available, same convention used throughout this project (real
    user preference, see sbs_to_mvc_cli._intermediate_video_codec_args' own reasoning) --
    falls back to libx265 (not libx264: this is the final deliverable, not a throwaway
    intermediate file, so the better/smaller-for-same-quality codec is used even on CPU)."""
    if _cuda_available():
        return ["-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", "16"]
    return ["-c:v", "libx265", "-crf", "16", "-preset", "medium"]


def build_ffmpeg_cmd(ffmpeg_bin, input_path, overlay_path, layout, width, height, shift_px,
                     blend_mode, opacity, output_path, loop=False, start_time=None,
                     end_time=None, video_codec_args=None, include_av=True,
                     extra_output_args=None):
    """Low-level command builder -- does NOT probe or compute real trim bounds itself (run()
    does that). start_time (seconds, input-seek on the MAIN input only) and end_time
    (seconds, used only to compute -t) are passed through exactly as given: both None means
    no -ss/-t at all; end_time alone means -t = end_time - (start_time or 0). --loop adds
    -stream_loop -1 on the overlay input so it never runs out before -t stops the output."""
    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    if start_time is not None and start_time > 0:
        cmd += ["-ss", f"{start_time:.3f}"]
    cmd += ["-i", input_path]
    if loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", overlay_path]
    filtergraph = build_filtergraph(layout, width, height, shift_px, blend_mode, opacity)
    cmd += ["-filter_complex", filtergraph, "-map", "[vout]"]
    if include_av:
        cmd += ["-map", "0:a?", "-map", "0:s?", "-c:a", "copy", "-c:s", "copy"]
    cmd += list(video_codec_args if video_codec_args is not None else _video_codec_args())
    if end_time is not None:
        duration = end_time - (start_time or 0.0)
        cmd += ["-t", f"{duration:.3f}"]
    cmd += list(extra_output_args or [])
    cmd += [output_path]
    return cmd


def resolve_output_path(input_path, output_arg=None):
    """Empty -> '<name>_fxoverlay<ext>' next to the input. An existing folder -> that name
    inside it. Anything else is used as the output path as given. Same convention as
    bitrate_cap_cli.resolve_output_path()."""
    base, ext = path.splitext(path.basename(input_path))
    name = f"{base}_fxoverlay{ext}"
    if not output_arg:
        return path.join(path.dirname(path.abspath(input_path)), name)
    if path.isdir(output_arg):
        return path.join(output_arg, name)
    return output_arg


def fxoverlay_tmp_path(output_path):
    """The temp file ffmpeg actually writes into before run() replaces the real output with it
    (CS-IO-001, same atomic tmp-then-os.replace() convention as
    bitrate_cap_cli.capping_tmp_path()): a crash, Cancel, or failed encode can never leave a
    partially-written file sitting at the real output name. Shared with the GUI panel so its
    Cancel cleanup removes exactly the file this tool writes mid-encode."""
    base, ext = path.splitext(output_path)
    return f"{base}_fxoverlay_tmp{ext}"


def _remove_quietly(file_path):
    try:
        if path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass


def run(input_path, overlay_path, output_path=None, layout=None, blend_mode="screen",
        pop_strength=0.5, opacity=1.0, loop=False, start_time=None, end_time=None,
        ffmpeg_bin=None):
    """Returns the output path. Raises FxOverlayError with a plain-language reason when
    refusing. Every validation below runs BEFORE the real ffprobe/ffmpeg calls, so none of
    it needs a real video file -- only the final real composite does."""
    if path.isdir(input_path):
        raise FxOverlayError("the input is a folder; this tool handles one finished video file at a time")
    if not path.isfile(input_path):
        raise FxOverlayError(f"input file not found: {input_path}")
    if not path.isfile(overlay_path):
        raise FxOverlayError(f"overlay file not found: {overlay_path}")
    if blend_mode not in BLEND_MODES:
        raise FxOverlayError(f"--blend-mode must be one of {BLEND_MODES}, got {blend_mode!r}")
    if not (0.0 <= pop_strength <= 1.0):
        raise FxOverlayError(f"--pop-strength must be between 0.0 and 1.0, got {pop_strength}")
    if not (0.0 <= opacity <= 1.0):
        raise FxOverlayError(f"--opacity must be between 0.0 and 1.0, got {opacity}")

    output = resolve_output_path(input_path, output_path)
    if path.abspath(output) == path.abspath(input_path):
        raise FxOverlayError("the output must be a different file from the input")
    if path.exists(output):
        raise FxOverlayError(f"the output already exists and will not be overwritten: {output}")
    if not path.isdir(path.dirname(path.abspath(output)) or "."):
        raise FxOverlayError(f"the output folder does not exist: {path.dirname(path.abspath(output))}")

    width, height, _fps, duration, is_hdr = probe_video(input_path)
    chosen_layout = layout or guess_layout(width, height)
    if chosen_layout not in LAYOUTS:
        raise FxOverlayError(f"--layout must be one of {LAYOUTS}, got {chosen_layout!r}")

    eye_w, eye_h, _l, _r = eye_regions(chosen_layout, width, height)
    shift_px = pop_strength_to_shift_px(pop_strength, eye_w)

    from nunif.utils.video.metadata import parse_time
    trim_start = parse_time(start_time) if start_time else 0.0
    trim_end = parse_time(end_time) if end_time is not None else (duration or 0.0)
    if duration and trim_end > duration:
        trim_end = duration
    if trim_end <= trim_start:
        raise FxOverlayError(f"--end-time ({end_time}) must be after --start-time ({start_time})")

    if is_hdr:
        print("[iw3] FX Pop-Out Overlay: the input has HDR/Dolby Vision -- this tool always "
              "re-encodes the whole video, which REMOVES any Dolby Vision/HDR10+/HDR grading. "
              "The output will be plain SDR. This tool has no Dolby Vision/HDR re-attach option "
              "(unlike Limit Bitrate's --dv-source) -- if you need to keep it, reinject "
              "Dolby Vision/HDR onto this tool's output afterward with the standalone HDR/DV "
              "Reinjection tool instead.", file=sys.stderr)

    ffmpeg = ffmpeg_bin or _get_ffmpeg_bin()
    # Encode into a temp-suffixed path and only os.replace() it into the real `output` name
    # after confirming success -- see fxoverlay_tmp_path()'s own docstring (CS-IO-001).
    tmp_output = fxoverlay_tmp_path(output)
    cmd = build_ffmpeg_cmd(
        ffmpeg, input_path, overlay_path, chosen_layout, width, height, shift_px,
        blend_mode, opacity, tmp_output, loop=loop, start_time=trim_start, end_time=trim_end,
        video_codec_args=_video_codec_args())

    print(f"[iw3] FX Pop-Out Overlay: layout={chosen_layout} eye={eye_w}x{eye_h} "
          f"shift_px={shift_px} (2*shift_px={2 * shift_px}) blend={blend_mode} "
          f"opacity={opacity} loop={loop}", file=sys.stderr)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        _remove_quietly(tmp_output)
        raise FxOverlayError(f"could not run ffmpeg ({ffmpeg}): {e}")
    if proc.returncode != 0 or not (path.exists(tmp_output) and path.getsize(tmp_output) > 0):
        _remove_quietly(tmp_output)
        raise FxOverlayError("ffmpeg failed to build the overlay composite:\n" + (proc.stderr or "")[-1200:])
    os.replace(tmp_output, output)
    print(f"[iw3] FX Pop-Out Overlay: done. Output: {output}", file=sys.stderr)
    return output


def create_parser():
    parser = argparse.ArgumentParser(
        description="Composite a rain/snow/fire/etc. overlay onto a finished stereo video, "
                    "giving it its own pop-out parallax independent of the AI depth pipeline.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="the finished stereo video (SBS or TB, a single file)")
    parser.add_argument("--overlay", type=str, required=True,
                        help="the overlay clip (rain/snow/fire/smoke/sparks/etc.)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="output file path, or an existing folder to write into. Default: "
                             "'<name>_fxoverlay<ext>' next to the input. Never overwrites an existing file.")
    parser.add_argument("--layout", type=str, default=None, choices=LAYOUTS,
                        help="SBS/TB layout override. Default: auto-detected from the input's own "
                             "picture size (same detection the SBS-to-MVC tool uses).")
    parser.add_argument("--blend-mode", type=str, default="screen", choices=BLEND_MODES,
                        help="'screen' for a black-background overlay (additive); 'alpha' for an "
                             "overlay that carries a real alpha channel. Default: screen")
    parser.add_argument("--pop-strength", type=float, default=0.5,
                        help="0.0-1.0: how far in front of the screen the overlay pops out. Default: 0.5")
    parser.add_argument("--opacity", type=float, default=1.0,
                        help="0.0-1.0: overall overlay visibility, independent of --pop-strength. Default: 1.0")
    parser.add_argument("--loop", action="store_true",
                        help="loop the overlay clip if it is shorter than the main video")
    parser.add_argument("--start-time", type=str, default=None,
                        help="trim the MAIN video to start here (HH:MM:SS or seconds). Default: the start")
    parser.add_argument("--end-time", type=str, default=None,
                        help="trim the MAIN video to end here (HH:MM:SS or seconds). Default: the end")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    try:
        run(args.input, args.overlay, args.output, args.layout, args.blend_mode,
            args.pop_strength, args.opacity, args.loop, args.start_time, args.end_time)
    except FxOverlayError as e:
        print(f"[iw3] FX Pop-Out Overlay: refused -- {e}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------------------
# Self-tests. Run with: python -m iw3.fx_overlay_cli --self-test
# All synthetic (ffmpeg lavfi test sources) -- no GPU and no real movie needed. Refusal/
# validation paths are checked by calling run()/build_filtergraph()/pop_strength_to_shift_px()
# directly rather than through parse_args() or any GUI event handler.
# --------------------------------------------------------------------------------------

def _self_test_guess_layout_reused():
    from . import sbs_to_mvc_cli
    assert guess_layout is sbs_to_mvc_cli.guess_layout, (
        "fx_overlay_cli must reuse sbs_to_mvc_cli.guess_layout, not reimplement layout detection")
    assert guess_layout(3840, 1080) == "full_sbs"
    assert guess_layout(1920, 1080) == "half_sbs"
    print("_self_test_guess_layout_reused: PASS")


def _self_test_pop_strength_formula():
    assert pop_strength_to_shift_px(0.0, 1920) == 0
    assert pop_strength_to_shift_px(1.0, 1920) == round(1920 * POP_STRENGTH_SHIFT_FACTOR)
    assert pop_strength_to_shift_px(0.5, 1920) == round(0.5 * 1920 * POP_STRENGTH_SHIFT_FACTOR)
    # The exact case the real synthetic parallax-direction self-test below relies on.
    assert pop_strength_to_shift_px(0.4, 1000) == 20
    for bad in (-0.01, 1.01):
        try:
            pop_strength_to_shift_px(bad, 1920)
        except FxOverlayError:
            pass
        else:
            raise AssertionError(f"pop_strength={bad} must be refused")
    print("_self_test_pop_strength_formula: PASS")


def _self_test_eye_regions():
    assert eye_regions("full_sbs", 3840, 1080) == (1920, 1080, (0, 0), (1920, 0))
    assert eye_regions("half_sbs", 1920, 1080) == (960, 1080, (0, 0), (960, 0))
    assert eye_regions("full_tb", 1920, 2160) == (1920, 1080, (0, 0), (0, 1080))
    assert eye_regions("half_tb", 1920, 1080) == (1920, 540, (0, 0), (0, 540))
    try:
        eye_regions("vr180", 100, 100)
    except FxOverlayError:
        pass
    else:
        raise AssertionError("an unknown layout must be refused")
    print("_self_test_eye_regions: PASS")


def _self_test_filtergraph_screen_mode():
    g = build_filtergraph("full_sbs", 2000, 100, 20, "screen", 0.8)
    assert "crop=1000:100:0:0" in g
    assert "crop=1000:100:1000:0" in g       # right eye crop from the BASE video
    assert "scale=1040:100:flags=lanczos" in g  # 1000 + 2*20 = 1040
    assert "crop=1000:100:0:0[fx_ov_lc]" in g   # left overlay crop: x-offset 0
    assert "crop=1000:100:40:0[fx_ov_rc]" in g  # right overlay crop: x-offset 2*shift_px=40
    assert "blend=all_mode=screen:all_opacity=0.8000" in g
    assert "hstack=inputs=2" in g
    print("_self_test_filtergraph_screen_mode: PASS")


def _self_test_filtergraph_alpha_mode():
    g = build_filtergraph("half_tb", 1920, 1080, 15, "alpha", 0.6)
    assert "crop=1920:540:0:0" in g
    assert "crop=1920:540:0:540" in g
    assert "colorchannelmixer=aa=0.6000" in g
    assert "overlay=x=0:y=0" in g
    assert "vstack=inputs=2" in g
    print("_self_test_filtergraph_alpha_mode: PASS")


def _self_test_filtergraph_invalid_args_refused():
    try:
        build_filtergraph("full_sbs", 2000, 100, 20, "chroma_key", 1.0)
    except FxOverlayError:
        pass
    else:
        raise AssertionError("an invalid blend mode must be refused")
    try:
        build_filtergraph("full_sbs", 2000, 100, 20, "screen", 1.5)
    except FxOverlayError:
        pass
    else:
        raise AssertionError("an out-of-range opacity must be refused")
    print("_self_test_filtergraph_invalid_args_refused: PASS")


def _self_test_loop_and_trim_flags():
    cmd = build_ffmpeg_cmd("ffmpeg.exe", "in.mp4", "ov.mp4", "half_sbs", 1920, 1080, 10,
                           "screen", 1.0, "out.mp4", loop=True, start_time=5.0, end_time=15.0,
                           video_codec_args=["-c:v", "libx265"])
    assert "-stream_loop" in cmd and cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "5.000"
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "10.000"
    assert "0:a?" in cmd and "0:s?" in cmd

    cmd2 = build_ffmpeg_cmd("ffmpeg.exe", "in.mp4", "ov.mp4", "half_sbs", 1920, 1080, 10,
                            "screen", 1.0, "out.mp4", loop=False, start_time=None, end_time=None,
                            video_codec_args=["-c:v", "libx265"])
    assert "-stream_loop" not in cmd2
    assert "-ss" not in cmd2
    assert "-t" not in cmd2
    print("_self_test_loop_and_trim_flags: PASS")


def _self_test_resolve_output_path():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        input_path = path.join(tmp, "clip.mkv")
        assert resolve_output_path(input_path) == path.join(tmp, "clip_fxoverlay.mkv")
        folder = path.join(tmp, "out")
        os.makedirs(folder, exist_ok=True)
        assert resolve_output_path(input_path, folder) == path.join(folder, "clip_fxoverlay.mkv")
        explicit = path.join(tmp, "named.mp4")
        assert resolve_output_path(input_path, explicit) == explicit
    print("_self_test_resolve_output_path: PASS")


def _self_test_run_validation_refusals():
    """Every one of these is refused BEFORE run() ever calls ffprobe/ffmpeg, so plain dummy
    bytes stand in for real video files -- exercising run() directly (not parse_args(), not
    any GUI handler), per this task's own verification convention."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dummy_main = path.join(tmp, "main.mkv")
        dummy_overlay = path.join(tmp, "overlay.mkv")
        with open(dummy_main, "wb") as f:
            f.write(b"not a real video")
        with open(dummy_overlay, "wb") as f:
            f.write(b"not a real video either")

        try:
            run(tmp, dummy_overlay)
        except FxOverlayError as e:
            assert "folder" in str(e), e
        else:
            raise AssertionError("a folder input must be refused")

        try:
            run(dummy_main, path.join(tmp, "missing_overlay.mkv"))
        except FxOverlayError as e:
            assert "overlay file not found" in str(e), e
        else:
            raise AssertionError("a missing overlay must be refused")

        try:
            run(dummy_main, dummy_overlay, blend_mode="chroma_key")
        except FxOverlayError as e:
            assert "blend-mode" in str(e), e
        else:
            raise AssertionError("an invalid blend mode must be refused")

        for bad in (-0.1, 1.1):
            try:
                run(dummy_main, dummy_overlay, pop_strength=bad)
            except FxOverlayError as e:
                assert "pop-strength" in str(e), e
            else:
                raise AssertionError(f"pop_strength={bad} must be refused")
            try:
                run(dummy_main, dummy_overlay, opacity=bad)
            except FxOverlayError as e:
                assert "opacity" in str(e), e
            else:
                raise AssertionError(f"opacity={bad} must be refused")

        existing_output = path.join(tmp, "main_fxoverlay.mkv")
        with open(existing_output, "wb") as f:
            f.write(b"already here")
        try:
            run(dummy_main, dummy_overlay)
        except FxOverlayError as e:
            assert "already exists" in str(e), e
        else:
            raise AssertionError("an existing output must be refused")
    print("_self_test_run_validation_refusals: PASS")


def _self_test_run_atomic_write():
    """Real, ffmpeg-executed check (not mocked): a successful run() must leave ONLY the final
    output in place (no leftover fxoverlay_tmp_path() temp file), and a failed run() (here,
    forced via a bogus --ffmpeg_bin) must leave NEITHER the temp file NOR the final output --
    CS-IO-001's atomic tmp-then-os.replace() convention, same as bitrate_cap_cli."""
    import tempfile
    ffmpeg = _get_ffmpeg_bin()
    with tempfile.TemporaryDirectory() as tmp:
        main_path = path.join(tmp, "main.mp4")
        overlay_path = path.join(tmp, "overlay.mp4")
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "color=c=black:s=320x100:d=1",
                        "-frames:v", "1", main_path], check=True, capture_output=True)
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "color=c=black:s=160x100:d=1",
                        "-frames:v", "1", overlay_path], check=True, capture_output=True)

        output = run(main_path, overlay_path)
        assert path.exists(output) and path.getsize(output) > 0
        assert not path.exists(fxoverlay_tmp_path(output)), (
            "a successful run must not leave its temp encode file behind")

        # Failure case: a bogus ffmpeg binary. Must leave neither the temp file nor the output.
        failed_output = path.join(tmp, "failed_output.mp4")
        try:
            run(main_path, overlay_path, output_path=failed_output,
                ffmpeg_bin=path.join(tmp, "no_such_ffmpeg.exe"))
        except FxOverlayError:
            pass
        else:
            raise AssertionError("a bogus ffmpeg binary must be refused")
        assert not path.exists(failed_output)
        assert not path.exists(fxoverlay_tmp_path(failed_output))
    print("_self_test_run_atomic_write: PASS")


def _self_test_hdr_warning_printed():
    """Real, ffmpeg-executed check (not mocked, except for forcing probe_video()'s own is_hdr
    flag True on an ordinary SDR synthetic clip -- generating a real HDR-graded test clip via
    lavfi is not practical). Confirms run() prints an explicit stderr warning that Dolby
    Vision/HDR is being removed when the source is HDR, and prints no such warning otherwise --
    this tool always re-encodes and has no DV/HDR re-attach option, so silence would otherwise
    hide a real, permanent loss of the source's HDR grading."""
    import tempfile
    import io
    import contextlib
    from unittest import mock
    ffmpeg = _get_ffmpeg_bin()
    with tempfile.TemporaryDirectory() as tmp:
        main_path = path.join(tmp, "main.mp4")
        overlay_path = path.join(tmp, "overlay.mp4")
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "color=c=black:s=320x100:d=1",
                        "-frames:v", "1", main_path], check=True, capture_output=True)
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "color=c=black:s=160x100:d=1",
                        "-frames:v", "1", overlay_path], check=True, capture_output=True)

        real_probe = probe_video(main_path)
        assert real_probe[4] is False, "this plain lavfi clip must not already look HDR"
        forced_hdr_probe = real_probe[:4] + (True,)

        buf = io.StringIO()
        with mock.patch.object(sys.modules[__name__], "probe_video", return_value=forced_hdr_probe), \
                contextlib.redirect_stderr(buf):
            output = run(main_path, overlay_path)
        warned = buf.getvalue()
        assert "Dolby Vision" in warned and "REMOVES" in warned, warned
        assert path.exists(output)
        os.remove(output)

        buf2 = io.StringIO()
        with contextlib.redirect_stderr(buf2):
            run(main_path, overlay_path)
        assert "REMOVES" not in buf2.getvalue(), buf2.getvalue()
    print("_self_test_hdr_warning_printed: PASS")


def _self_test_parallax_direction_synthetic():
    """Real, ffmpeg-executed synthetic check (not mocked): composites a thin white marker
    line over a flat black SBS test pattern and reads back the actual output pixels to
    confirm which eye's copy of the marker lands further RIGHT. Per this codebase's own
    near/far convention (iw3/backward_warp.py: apply_divergence_grid_sample shifts the LEFT
    eye's content RIGHT and the RIGHT eye's content LEFT for near/high-depth-value pixels --
    see DE.apply_depth_band_pop's foreground threshold being the HIGH end, 0.85-1.0, i.e.
    higher depth value = nearer), the near/pop-out eye assignment is: content appears
    further RIGHT in the LEFT eye than in the RIGHT eye. This test proves this tool's own
    left=offset-0 / right=offset-2*shift_px assignment reproduces exactly that."""
    import tempfile
    ffmpeg = _get_ffmpeg_bin()
    eye_w, eye_h = 1000, 100
    width, height = eye_w * 2, eye_h
    shift_px = pop_strength_to_shift_px(0.4, eye_w)
    assert shift_px == 20, shift_px

    with tempfile.TemporaryDirectory() as tmp:
        main_path = path.join(tmp, "main.mp4")
        overlay_path = path.join(tmp, "overlay.mp4")
        out_path = path.join(tmp, "out.raw")

        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"color=c=black:s={width}x{height}:d=1",
                        "-frames:v", "1", main_path], check=True, capture_output=True)
        # A thin white vertical marker at the overlay's own local center (x=500 of 1000).
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"color=c=black:s={eye_w}x{eye_h}:d=1",
                        "-vf", f"drawbox=x={eye_w // 2 - 2}:y=0:w=4:h={eye_h}:color=white:t=fill",
                        "-frames:v", "1", overlay_path], check=True, capture_output=True)

        cmd = build_ffmpeg_cmd(
            ffmpeg, main_path, overlay_path, "full_sbs", width, height, shift_px,
            "screen", 1.0, out_path, loop=False, start_time=None, end_time=None,
            video_codec_args=[], include_av=False,
            extra_output_args=["-f", "rawvideo", "-pix_fmt", "rgb24"])
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with open(out_path, "rb") as f:
            data = f.read()
        assert len(data) == width * height * 3, (len(data), width, height)

        col_brightness = [0] * width
        for y in range(height):
            row_off = y * width * 3
            for x in range(width):
                p = row_off + x * 3
                col_brightness[x] += data[p] + data[p + 1] + data[p + 2]

        left_cols = col_brightness[:eye_w]
        right_cols = col_brightness[eye_w:]
        left_peak = left_cols.index(max(left_cols))
        right_peak_local = right_cols.index(max(right_cols))

        assert left_peak > right_peak_local, (
            "near/pop-out requires the LEFT eye's marker to land to the RIGHT of the RIGHT "
            f"eye's marker (same content, same frame) -- got left={left_peak} right={right_peak_local}")
        diff = left_peak - right_peak_local
        assert abs(diff - 2 * shift_px) <= 2, (diff, shift_px)
    print("_self_test_parallax_direction_synthetic: PASS "
          f"(left marker at local x={left_peak}, right eye at local x={right_peak_local}, "
          f"diff={diff} ~= 2*shift_px={2 * shift_px} -- left eye shows the marker further "
          f"right than the right eye, confirming crossed/pop-out disparity)")


def _run_self_tests():
    _self_test_guess_layout_reused()
    _self_test_pop_strength_formula()
    _self_test_eye_regions()
    _self_test_filtergraph_screen_mode()
    _self_test_filtergraph_alpha_mode()
    _self_test_filtergraph_invalid_args_refused()
    _self_test_loop_and_trim_flags()
    _self_test_resolve_output_path()
    _self_test_run_validation_refusals()
    _self_test_run_atomic_write()
    _self_test_hdr_warning_printed()
    _self_test_parallax_direction_synthetic()
    print("All iw3.fx_overlay_cli self-tests PASSED")


if __name__ == "__main__":
    sys.exit(main())
