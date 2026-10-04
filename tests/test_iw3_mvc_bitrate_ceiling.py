"""Regression test for the real output-type-aware MVC bitrate ceiling fix.

Background (real research, 2026-10-04, cross-checked across several independent,
convergent doom9.org community sources -- the same community FRIM itself came from;
the official BDA spec text is paywalled and wasn't directly accessible): the old
tooltip calling --bitrate "Mbps per eye" was wrong -- FRIMEncode's own -vbr target in
-o:mvc mode is the COMBINED bitrate for both views together. The old hard 40 Mbps
ceiling in iw3.sbs_to_mvc_cli.convert() (and its duplicate in iw3.direct_mvc_cli's
own single-pass path) was also wrong: 40 Mbps is the real BD-ROM 2D/base-view-ONLY
limit, misapplied here to the 3D combined-MVC case. The real combined ceiling for a
real disc-structured target (.iso/BD-folder, played on certified Blu-ray hardware) is
60 Mbps. That hardware/compliance concern doesn't meaningfully apply to a plain
.mkv/.m2ts file read by an ordinary software player (VLC, MPC-HC, ...) -- but the
encode is still declared "-profile high -level 4.1" for every output type regardless,
so AVC Level 4.1 High Profile's own formally-defined max bitrate (62500 kbit/s =
62.5 Mbps -- Table A-1's MaxBR=50000 for Baseline/Main/Extended at Level 4.1, times
the High Profile cpbBrVclFactor of 1.25; confirmed against ffmpeg's own
h264_levels.c level-limits table) is the real ceiling for the non-disc case instead.

Synthetic/isolated, no real FRIM/tsMuxeR/ffmpeg/GPU encode involved: find_frim(),
_find_tsmuxer() and _get_ffmpeg_bin() are mocked to return a fake path so convert()
reaches its own real bitrate-ceiling check (which runs before any of those tools are
actually invoked) against a zero-byte placeholder input file.

Run directly: python tests/test_iw3_mvc_bitrate_ceiling.py (from the nunif/ dir),
or import and call main().
"""
import sys
import tempfile
from os import path
from unittest import mock

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from iw3 import sbs_to_mvc_cli as m  # noqa: E402
from iw3 import direct_mvc_cli as dm  # noqa: E402


def _bitrate_error(output_name, bitrate_mbps):
    """None if convert() got past its own bitrate-ceiling check for this output name/
    bitrate (any later failure, e.g. a missing real ffprobe, is irrelevant here and is
    swallowed); otherwise the real ValueError message convert() raised for it."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = path.join(tmp, "in.mkv")
        open(inp, "wb").close()
        out = path.join(tmp, output_name)
        with mock.patch.object(m, "find_frim", return_value="frim.exe"), \
             mock.patch.object(m, "_find_tsmuxer", return_value="tsmuxer.exe"), \
             mock.patch.object(m, "_get_ffmpeg_bin", return_value="ffmpeg.exe"):
            try:
                m.convert(inp, out, bitrate_mbps=bitrate_mbps)
            except ValueError as e:
                return str(e)
            except Exception:
                pass
            return None


def _assert_bitrate(output_name, bitrate_mbps, expect_rejected, label):
    msg = _bitrate_error(output_name, bitrate_mbps)
    rejected = msg is not None and "bitrate must be between" in msg
    assert rejected == expect_rejected, f"{label}: {msg!r}"


def _test_disc_output_ceiling_is_60():
    # ADR (this session): a value that used to be rejected under the old (wrong) 40
    # ceiling must now be accepted for disc-structured output, up to the new 60.
    _assert_bitrate("out.iso", 50, False, "50 Mbps must now be accepted for ISO output")
    _assert_bitrate("out.iso", 60, False, "60 Mbps (the new ISO ceiling) must be accepted")
    _assert_bitrate("out.iso", 60.1, True, "just above 60 Mbps must still be rejected for ISO")
    _assert_bitrate("out.iso", 1.9, True, "below the unchanged 2 Mbps floor must still be rejected")
    # BD Folder (no extension) shares the exact same disc_legal grouping as ISO.
    _assert_bitrate("out_folder", 50, False, "50 Mbps must be accepted for BD Folder output")
    _assert_bitrate("out_folder", 61, True, "above 60 Mbps must be rejected for BD Folder output")
    print("_test_disc_output_ceiling_is_60: PASS")


def _test_nondisc_output_ceiling_is_62_5():
    # Plain MKV/Bare M2TS have no real physical-player compliance concern, so they get
    # the higher ceiling grounded in this encode's own fixed AVC Level 4.1 High Profile
    # declaration (62.5 Mbps) instead of the disc ceiling.
    _assert_bitrate("out.mkv", 60, False, "60 Mbps must be accepted for Plain MKV output")
    _assert_bitrate("out.mkv", 62.5, False, "62.5 Mbps (the new MKV ceiling) must be accepted")
    _assert_bitrate("out.mkv", 62.6, True, "just above 62.5 Mbps must be rejected for Plain MKV output")
    _assert_bitrate("out.m2ts", 62.5, False, "62.5 Mbps must be accepted for Bare M2TS output")
    _assert_bitrate("out.m2ts", 63, True, "above 62.5 Mbps must be rejected for Bare M2TS output")
    print("_test_nondisc_output_ceiling_is_62_5: PASS")


def _test_direct_mvc_cli_shares_the_same_ceiling_helper():
    # iw3.direct_mvc_cli (the single-pass "Direct to 3D Blu-ray MVC" mode) used to carry
    # its own separately hardcoded "2 <= bitrate <= 40" copy of this exact check -- it
    # now imports the same real helper sbs_to_mvc_cli.convert() uses, so the two code
    # paths reachable from the GUI's own shared MVC Bitrate field can never drift apart.
    assert dm._mvc_bitrate_ceiling_mbps is m._mvc_bitrate_ceiling_mbps
    assert dm._mvc_bitrate_ceiling_mbps(True) == 60.0
    assert dm._mvc_bitrate_ceiling_mbps(False) == 62.5
    print("_test_direct_mvc_cli_shares_the_same_ceiling_helper: PASS")


def main():
    _test_disc_output_ceiling_is_60()
    _test_nondisc_output_ceiling_is_62_5()
    _test_direct_mvc_cli_shares_the_same_ceiling_helper()
    print("ALL PASS")


if __name__ == "__main__":
    main()
