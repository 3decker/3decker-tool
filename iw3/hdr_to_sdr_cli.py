"""python -m iw3.hdr_to_sdr_cli -- standalone HDR/Dolby Vision to SDR tone-map tool.

Converts an HDR (HDR10, HDR10+, Dolby Vision's base layer, or HLG) video to plain SDR,
using the exact same proven zscale+tonemap filter chain (Hable tonemap operator) already
used by iw3's own main pipeline (iw3.utils._tonemap_hdr_to_sdr) and by the "Fix HDR
automatically" option in the SBS to 3D Blu-ray MVC tool (iw3.sbs_to_mvc_cli.
tonemap_hdr_to_sdr_for_bd, which this module's tonemap_hdr_to_sdr core function is shared
with -- that one is hard-locked to 8-bit output, since 3D Blu-ray cannot carry 10-bit at
all; this general-purpose standalone tool has no such constraint, so 10-bit SDR (keeps
more of the original range even after tone-mapping) is offered and is the default).

This is a real, one-way change to the picture -- the HDR grade is genuinely gone
afterward, not just a metadata strip. Never modifies --input -- always writes a new file
at --output.

Genuinely standalone: invoked as its own subprocess (same "separate subprocess, never
sharing state with the main app" convention as iw3.sbs_to_mvc_cli / iw3.rife_cli /
iw3.sharpen_cli -- see docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001). Cancellation is
external (the GUI kills this process's PID directly, the same way it already cancels SBS
to 3D Blu-ray MVC), not an internal stop_event loop.
"""
import argparse
import sys
from os import path

from .sbs_to_mvc_cli import tonemap_hdr_to_sdr, probe_video, Cancelled
from .utils import _get_ffmpeg_bin


def _final_output_codec_args():
    """Real deliverable quality (this is the user's own kept output file, not a throwaway
    intermediate re-encoded again by something else afterward, unlike
    tonemap_hdr_to_sdr_for_bd's own choice) -- hevc_nvenc (GPU) when this machine has it,
    same detect-and-prefer pattern used throughout this project (iw3.utils.
    _hdr_upscale_codec/_sdr_upscale_codec, iw3.sbs_to_mvc_cli._intermediate_video_codec_
    args), else libx265 CRF 12 -- the same quality target iw3.utils._tonemap_hdr_to_sdr
    already uses for this exact operation elsewhere in this project."""
    try:
        import av
        av.codec.Codec("hevc_nvenc", "w")
        import torch
        if torch.cuda.is_available():
            return ["-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", "12"]
    except Exception:
        pass
    return ["-c:v", "libx265", "-crf", "12", "-preset", "medium"]


def create_parser():
    parser = argparse.ArgumentParser(
        prog="python -m iw3.hdr_to_sdr_cli",
        description=(
            "Converts an HDR (HDR10/HDR10+/Dolby Vision/HLG) video to plain SDR, using the "
            "same proven zscale+tonemap filter chain iw3's own main pipeline uses (Hable "
            "tonemap operator). This is a real, one-way change to the picture -- the HDR "
            "grade is genuinely gone afterward, not just a metadata strip. Never modifies "
            "--input -- always writes a new file at --output."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="the HDR video to convert")
    parser.add_argument("--output", "-o", required=True, help="path to write the new SDR file to")
    parser.add_argument("--bit-depth", type=int, default=10, choices=(8, 10),
                        help="output bit depth -- 10-bit (default) keeps more of the original "
                             "range even after tone-mapping down to SDR; 8-bit is smaller and "
                             "more universally compatible with older players/tools")
    parser.add_argument("--gui-progress", action="store_true",
                        help="print 'IW3_MVC_PROGRESS <stage> <done> <total>' lines to stdout")
    return parser


def run(args, progress_cb=None):
    """progress_cb override is for tests -- the real CLI entry point (main()) always builds
    one from args.gui_progress instead."""
    input_path = str(args.input)
    output_path = str(args.output)

    if not path.exists(input_path):
        print(f"ERROR: --input file does not exist: {input_path}", file=sys.stderr)
        return 1
    if path.abspath(output_path) == path.abspath(input_path):
        print("ERROR: --output must be a different path from --input -- this tool never "
              "overwrites the input.", file=sys.stderr)
        return 1

    ffmpeg = _get_ffmpeg_bin()
    if ffmpeg is None:
        print("ERROR: ffmpeg not found", file=sys.stderr)
        return 1

    try:
        width, height, rate, duration, hdr = probe_video(input_path)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not hdr:
        print("[hdr-to-sdr] source is not tagged HDR (PQ/HLG) -- nothing to convert. Refusing "
              "so a perfectly good file isn't needlessly re-encoded; if you just want a plain "
              "copy, copy the file directly instead of using this tool.", file=sys.stderr)
        return 1

    if progress_cb is None:
        if args.gui_progress:
            def progress_cb(stage, done, total):
                print(f"IW3_MVC_PROGRESS {stage} {done} {total}", flush=True)
        else:
            def progress_cb(stage, done, total):
                print(f"\r[hdr-to-sdr] {stage}: {done}/{total}      ", end="", file=sys.stderr, flush=True)

    try:
        tonemap_hdr_to_sdr(input_path, output_path, ffmpeg, _final_output_codec_args(),
                          bit_depth=args.bit_depth, audio_args=["-c:a", "copy"],
                          duration=duration, progress_cb=progress_cb, stage="tonemap")
    except Cancelled:
        print("\n[hdr-to-sdr] cancelled", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    print(f"\n[hdr-to-sdr] done: {output_path} ({args.bit_depth}-bit SDR)", file=sys.stderr)
    return 0


def _self_test_run_gating():
    """Synthetic/mocked test of run()'s pre-processing gates (missing input, output
    collision, source not actually HDR) -- proves each refuses before ffmpeg is ever
    invoked, and that a genuine HDR source reaches tonemap_hdr_to_sdr with the right args."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="iw3_hdr_to_sdr_selftest_") as tmpdir:
        input_path = path.join(tmpdir, "movie.mkv")
        missing_input = path.join(tmpdir, "does_not_exist.mkv")
        output_path = path.join(tmpdir, "movie_sdr.mkv")
        with open(input_path, "wb") as f:
            f.write(b"0")

        def _args(**overrides):
            base = dict(input=input_path, output=output_path, bit_depth=10, gui_progress=False)
            base.update(overrides)
            return argparse.Namespace(**base)

        with patch(f"{__name__}._get_ffmpeg_bin", return_value="ffmpeg.exe"), \
             patch(f"{__name__}.tonemap_hdr_to_sdr") as mock_tonemap:
            # missing input -> refuse before ever probing
            rc = run(_args(input=missing_input))
            assert rc == 1, rc
            mock_tonemap.assert_not_called()

            # output collides with input -> refuse
            rc = run(_args(output=input_path))
            assert rc == 1, rc
            mock_tonemap.assert_not_called()

            # source not HDR -> refuse (don't needlessly re-encode a perfectly good file)
            with patch(f"{__name__}.probe_video", return_value=(1920, 1080, "24/1", 100.0, False)):
                rc = run(_args())
                assert rc == 1, rc
                mock_tonemap.assert_not_called()

            # genuine HDR source -> proceeds, with the requested bit depth passed through
            with patch(f"{__name__}.probe_video", return_value=(1920, 1080, "24/1", 100.0, True)):
                rc = run(_args(bit_depth=8))
                assert rc == 0, rc
                mock_tonemap.assert_called_once()
                call_kwargs = mock_tonemap.call_args.kwargs
                assert call_kwargs["bit_depth"] == 8, call_kwargs
                assert call_kwargs["duration"] == 100.0, call_kwargs

    print("_self_test_run_gating: PASS")


def _run_self_tests():
    _self_test_run_gating()
    print("All hdr_to_sdr_cli self-tests PASSED")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
