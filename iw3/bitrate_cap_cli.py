"""Standalone Limit Bitrate tool: python -m iw3.bitrate_cap_cli

Applies the same Limit Bitrate re-encode the conversion uses (iw3.utils._run_bitrate_cap) to an
already-finished video file, so a file that came out over the limit can be fixed without redoing
the whole conversion.

The input is never modified. It is copied to the output path first (written under a temp name,
then renamed into place), and the cap runs on that copy. A copy within the limit is kept as an
unchanged copy. A copy whose re-encode fails is removed, so an uncapped file is never left
looking like a capped one.
"""
import argparse
import os
import shutil
import subprocess
import sys
import types
from os import path

from . import utils as iw3_utils


CODECS = ("hevc_nvenc", "h264_nvenc")
DEFAULT_PIX_FMT = "yuv420p10le"


class BitrateCapError(Exception):
    pass


def resolve_output_path(input_path, output_arg=None):
    """Empty -> '<name>_capped<ext>' next to the input. An existing folder -> that name inside it.
    Anything else is used as the output file path as given. Shared with the GUI panel so the
    Cancel cleanup removes exactly the file this tool writes."""
    base, ext = path.splitext(path.basename(input_path))
    name = f"{base}_capped{ext}"
    if not output_arg:
        return path.join(path.dirname(path.abspath(input_path)), name)
    if path.isdir(output_arg):
        return path.join(output_arg, name)
    return output_arg


def capping_tmp_path(output_path):
    """The temp file _run_bitrate_cap() writes its re-encode into before replacing the output."""
    base, ext = path.splitext(output_path)
    return f"{base}_capping_tmp{ext}"


def build_cap_args(codec, limit_mbps, crf, gpu, pix_fmt):
    """The args-like object _run_bitrate_cap() reads, built the same way the conversion builds it."""
    pix_fmt = iw3_utils._clamp_pix_fmt_for_codec(pix_fmt, codec)
    if codec == "h264_nvenc":
        # H.264 NVENC is 8-bit only: drop the "10le" from a 10-bit source's pixel format.
        pix_fmt = pix_fmt.replace("10le", "")
    return types.SimpleNamespace(limit_bitrate=True, video_codec=codec, video_bitrate=f"{limit_mbps:g}M",
                                 crf=crf, pix_fmt=pix_fmt, gpu=[gpu], state={})


def _source_pix_fmt(input_path):
    try:
        proc = subprocess.run(
            [iw3_utils._find_ffprobe(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", input_path],
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise BitrateCapError(f"could not read the input's pixel format: {e}")
    return proc.stdout.strip() or DEFAULT_PIX_FMT


def _remove_quietly(file_path):
    try:
        if path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass


def _cap_copy(output_path, limit_mbps, codec, crf, gpu, pix_fmt, dv_source, rife_manifest):
    """Measures the copy once and re-encodes only when it is over the limit. Raises BitrateCapError
    when the copy cannot be judged or the re-encode fails (the caller removes the copy)."""
    peak_bps = iw3_utils._measure_peak_window_bitrate_bps(output_path)
    if not peak_bps:
        raise BitrateCapError("could not read the file's per-second bitrate data, so no decision could be made")
    print(f"[iw3] Limit Bitrate tool: real peak 1-second bitrate is {peak_bps / 1_000_000:.1f} Mbps, "
          f"limit is {limit_mbps:g} Mbps.", file=sys.stderr)
    if not iw3_utils._bitrate_cap_exceeded(peak_bps, limit_mbps):
        print("[iw3] Limit Bitrate tool: the file is within the limit, so no re-encode was needed. "
              "The output is an unchanged copy.", file=sys.stderr)
        return
    if not dv_source:
        print("[iw3] Limit Bitrate tool: Dolby Vision is NOT being re-attached (no --dv-source given). "
              "The re-encode removes any Dolby Vision in the file.", file=sys.stderr)
    args = build_cap_args(codec, limit_mbps, crf, gpu, pix_fmt)
    if iw3_utils._run_bitrate_cap(output_path, args, dv_source=dv_source, rife_manifest=rife_manifest) is None:
        raise BitrateCapError("the Limit Bitrate re-encode failed (see the messages above)")


def run(input_path, limit, output_path=None, crf=15, codec="hevc_nvenc", gpu=0, dv_source=None, rife_manifest=None):
    """Returns the output path. Raises BitrateCapError with a plain-language reason when refusing.
    The original input is never written to."""
    if path.isdir(input_path):
        raise BitrateCapError("the input is a folder; this tool handles one finished video file at a time")
    if not path.isfile(input_path):
        raise BitrateCapError(f"input file not found: {input_path}")
    if codec not in CODECS:
        raise BitrateCapError(f"codec '{codec}' is not supported; use hevc_nvenc or h264_nvenc "
                              f"(Limit Bitrate only runs with NVENC)")
    limit_mbps = iw3_utils._parse_bitrate_mbps(str(limit))
    if not limit_mbps or limit_mbps <= 0:
        raise BitrateCapError(f"could not read the limit '{limit}'; use a number of Mbps, e.g. 80 or 80M")
    if dv_source is not None and not path.isfile(dv_source):
        raise BitrateCapError(f"DV source file not found: {dv_source}")
    if rife_manifest is not None and not path.isfile(rife_manifest):
        raise BitrateCapError(f"RIFE manifest file not found: {rife_manifest}")

    output = resolve_output_path(input_path, output_path)
    if path.abspath(output) == path.abspath(input_path):
        raise BitrateCapError("the output must be a different file from the input")
    if path.exists(output):
        raise BitrateCapError(f"the output already exists and will not be overwritten: {output}")
    if not path.isdir(path.dirname(path.abspath(output))):
        raise BitrateCapError(f"the output folder does not exist: {path.dirname(path.abspath(output))}")
    pix_fmt = _source_pix_fmt(input_path)

    copy_tmp = output + ".copying"
    print(f"[iw3] Limit Bitrate tool: copying the original to {output} (the original is not modified)...",
          file=sys.stderr)
    try:
        shutil.copyfile(input_path, copy_tmp)
        os.replace(copy_tmp, output)
    except OSError as e:
        _remove_quietly(copy_tmp)
        raise BitrateCapError(f"could not copy the input: {e}")

    try:
        _cap_copy(output, limit_mbps, codec, crf, gpu, pix_fmt, dv_source, rife_manifest)
    except BaseException:
        _remove_quietly(output)
        _remove_quietly(capping_tmp_path(output))
        raise
    print(f"[iw3] Limit Bitrate tool: done. Output: {output}", file=sys.stderr)
    return output


def create_parser():
    parser = argparse.ArgumentParser(description="Apply Limit Bitrate to an already-finished video file.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="the finished video file (a single file, not a folder)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="output file path, or an existing folder to write into. Default: "
                             "'<name>_capped<ext>' next to the input. Never overwrites an existing file.")
    parser.add_argument("--limit", type=str, required=True,
                        help="peak 1-second bitrate limit in Mbps, e.g. 80 or 80M")
    parser.add_argument("--crf", type=int, default=15,
                        help="quality value to re-encode at (NVENC -cq). Default: 15")
    parser.add_argument("--codec", type=str, default="hevc_nvenc", choices=CODECS,
                        help="output codec. Default: hevc_nvenc")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for the re-encode. Default: 0")
    parser.add_argument("--dv-source", type=str, default=None,
                        help="original source file to re-attach Dolby Vision from, when the input had DV")
    parser.add_argument("--rife-manifest", type=str, default=None,
                        help="the .rife_manifest.json of the RIFE output, when the input came from a RIFE job. "
                             "Only used together with --dv-source.")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        _run_self_tests()
        return 0
    args = create_parser().parse_args(argv)
    try:
        run(args.input, args.limit, args.output, args.crf, args.codec, args.gpu,
            args.dv_source, args.rife_manifest)
    except BitrateCapError as e:
        print(f"[iw3] Limit Bitrate tool: refused -- {e}", file=sys.stderr)
        return 1
    return 0


def _self_test_common(tmp_dir, name="sample.mkv", content=b"synthetic video bytes " * 200):
    input_path = path.join(tmp_dir, name)
    with open(input_path, "wb") as f:
        f.write(content)
    return input_path, content


def _read_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()


def _self_test_copy_and_original_untouched_under_limit():
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        input_path, content = _self_test_common(tmp)
        with mock.patch.object(iw3_utils, "_measure_peak_window_bitrate_bps", return_value=10_000_000), \
                mock.patch.object(iw3_utils, "_run_bitrate_cap") as cap, \
                mock.patch.object(sys.modules[__name__], "_source_pix_fmt", return_value="yuv420p10le"):
            output = run(input_path, "80")
        assert not cap.called, "a file under the limit must not be re-encoded"
        assert output == path.join(tmp, "sample_capped.mkv"), output
        assert _read_bytes(output) == content, "under-limit output must be an unchanged copy"
        assert _read_bytes(input_path) == content, "original must be byte-identical"
    print("_self_test_copy_and_original_untouched_under_limit: PASS")


def _self_test_over_limit_caps_the_copy_not_the_original():
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        input_path, content = _self_test_common(tmp)
        calls = []

        def fake_cap(video_path, args, dv_source=None, rife_manifest=None):
            calls.append((video_path, args, dv_source, rife_manifest))
            with open(video_path, "wb") as f:
                f.write(b"re-encoded")
            return video_path

        with mock.patch.object(iw3_utils, "_measure_peak_window_bitrate_bps", return_value=120_000_000), \
                mock.patch.object(iw3_utils, "_run_bitrate_cap", side_effect=fake_cap), \
                mock.patch.object(sys.modules[__name__], "_source_pix_fmt", return_value="yuv420p10le"):
            output = run(input_path, "80M", crf=18)
        assert len(calls) == 1
        video_path, args, dv_source, rife_manifest = calls[0]
        assert video_path == output, "the cap must run on the copy"
        assert args.limit_bitrate is True and args.video_codec == "hevc_nvenc"
        assert args.video_bitrate == "80M" and args.crf == 18 and args.pix_fmt == "yuv420p10le"
        assert args.gpu == [0]
        assert dv_source is None and rife_manifest is None
        assert _read_bytes(output) == b"re-encoded"
        assert _read_bytes(input_path) == content, "original must be byte-identical"
    print("_self_test_over_limit_caps_the_copy_not_the_original: PASS")


def _self_test_limit_parsing():
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        input_path, _ = _self_test_common(tmp)
        for limit in ("80", "80M", " 80m "):
            with mock.patch.object(iw3_utils, "_measure_peak_window_bitrate_bps", return_value=120_000_000), \
                    mock.patch.object(iw3_utils, "_run_bitrate_cap", side_effect=lambda p, a, **kw: p) as cap, \
                    mock.patch.object(sys.modules[__name__], "_source_pix_fmt", return_value="yuv420p10le"):
                output = run(input_path, limit)
                assert cap.call_args[0][1].video_bitrate == "80M", (limit, cap.call_args)
                os.remove(output)
        for bad in ("abc", "0", ""):
            try:
                run(input_path, bad)
            except BitrateCapError:
                pass
            else:
                raise AssertionError(f"limit {bad!r} should be refused")
        assert not path.exists(path.join(tmp, "sample_capped.mkv"))
    print("_self_test_limit_parsing: PASS")


def _self_test_existing_output_refused():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        input_path, content = _self_test_common(tmp)
        existing = path.join(tmp, "sample_capped.mkv")
        with open(existing, "wb") as f:
            f.write(b"someone else's file")
        try:
            run(input_path, "80")
        except BitrateCapError as e:
            assert "already exists" in str(e), e
        else:
            raise AssertionError("an existing output must be refused")
        assert _read_bytes(existing) == b"someone else's file", "existing output must not be touched"
        assert _read_bytes(input_path) == content
    print("_self_test_existing_output_refused: PASS")


def _self_test_non_nvenc_codec_refused():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        input_path, _ = _self_test_common(tmp)
        for codec in ("libx265", "libx264", "av1_nvenc"):
            try:
                run(input_path, "80", codec=codec)
            except BitrateCapError as e:
                assert "not supported" in str(e), e
            else:
                raise AssertionError(f"codec {codec} should be refused")
        assert not path.exists(path.join(tmp, "sample_capped.mkv"))
    print("_self_test_non_nvenc_codec_refused: PASS")


def _self_test_folder_input_refused():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run(tmp, "80")
        except BitrateCapError as e:
            assert "folder" in str(e), e
        else:
            raise AssertionError("a folder input must be refused")
    print("_self_test_folder_input_refused: PASS")


def _self_test_dv_source_and_manifest_passed_through():
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        input_path, _ = _self_test_common(tmp)
        source = _self_test_common(tmp, name="original.mkv", content=b"source")[0]
        manifest = path.join(tmp, "rife.rife_manifest.json")
        with open(manifest, "w") as f:
            f.write("{}")
        with mock.patch.object(iw3_utils, "_measure_peak_window_bitrate_bps", return_value=120_000_000), \
                mock.patch.object(iw3_utils, "_run_bitrate_cap", side_effect=lambda p, a, **kw: p) as cap, \
                mock.patch.object(sys.modules[__name__], "_source_pix_fmt", return_value="yuv420p10le"):
            run(input_path, "80", dv_source=source, rife_manifest=manifest)
            assert cap.call_args[1] == {"dv_source": source, "rife_manifest": manifest}, cap.call_args
            os.remove(path.join(tmp, "sample_capped.mkv"))
            run(input_path, "80", dv_source=source)
            assert cap.call_args[1] == {"dv_source": source, "rife_manifest": None}, cap.call_args
    print("_self_test_dv_source_and_manifest_passed_through: PASS")


def _self_test_failed_reencode_removes_copy():
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        input_path, content = _self_test_common(tmp)
        with mock.patch.object(iw3_utils, "_measure_peak_window_bitrate_bps", return_value=120_000_000), \
                mock.patch.object(iw3_utils, "_run_bitrate_cap", return_value=None), \
                mock.patch.object(sys.modules[__name__], "_source_pix_fmt", return_value="yuv420p10le"):
            try:
                run(input_path, "80")
            except BitrateCapError as e:
                assert "re-encode failed" in str(e), e
            else:
                raise AssertionError("a failed re-encode must be reported")
        assert not path.exists(path.join(tmp, "sample_capped.mkv")), "no uncapped file may be left as capped"
        assert _read_bytes(input_path) == content
    print("_self_test_failed_reencode_removes_copy: PASS")


def _self_test_unreadable_peak_refused():
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        input_path, content = _self_test_common(tmp)
        with mock.patch.object(iw3_utils, "_measure_peak_window_bitrate_bps", return_value=None), \
                mock.patch.object(iw3_utils, "_run_bitrate_cap") as cap, \
                mock.patch.object(sys.modules[__name__], "_source_pix_fmt", return_value="yuv420p10le"):
            try:
                run(input_path, "80")
            except BitrateCapError as e:
                assert "bitrate data" in str(e), e
            else:
                raise AssertionError("an unreadable peak must refuse")
        assert not cap.called
        assert not path.exists(path.join(tmp, "sample_capped.mkv"))
        assert _read_bytes(input_path) == content
    print("_self_test_unreadable_peak_refused: PASS")


def _self_test_pix_fmt_per_codec():
    args = build_cap_args("hevc_nvenc", 80, 15, 0, "yuv420p10le")
    assert args.pix_fmt == "yuv420p10le", args.pix_fmt
    assert args.video_bitrate == "80M" and args.gpu == [0] and args.limit_bitrate is True
    # H.264 NVENC is 8-bit only, so a 10-bit source is written as 8-bit
    assert build_cap_args("h264_nvenc", 80, 15, 0, "yuv420p10le").pix_fmt == "yuv420p"
    assert build_cap_args("h264_nvenc", 80, 15, 0, "yuv444p10le").pix_fmt == "yuv444p"
    print("_self_test_pix_fmt_per_codec: PASS")


def _run_self_tests():
    _self_test_pix_fmt_per_codec()
    _self_test_copy_and_original_untouched_under_limit()
    _self_test_over_limit_caps_the_copy_not_the_original()
    _self_test_limit_parsing()
    _self_test_existing_output_refused()
    _self_test_non_nvenc_codec_refused()
    _self_test_folder_input_refused()
    _self_test_dv_source_and_manifest_passed_through()
    _self_test_failed_reencode_removes_copy()
    _self_test_unreadable_peak_refused()
    print("All iw3.bitrate_cap_cli self-tests PASSED")


if __name__ == "__main__":
    sys.exit(main())
