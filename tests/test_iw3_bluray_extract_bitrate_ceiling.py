"""Regression test: the 3D Blu-ray ISO Extract tool's "MVC .mkv, Auto-crop applied" layout
(iw3.mvc_extract_cli.extract_and_reencode_mvc) used to hard-cap --bitrate at 40 Mbps, while
the ADR-337 combined ceiling (iw3.sbs_to_mvc_cli / iw3.mvc_extract_cli._mvc_bitrate_ceiling_mbps)
already allowed more. This layout always writes .mkv, so the non-disc ceiling (62.5 Mbps)
applies here, and the GUI field and CLI check must agree on it.

Synthetic/isolated: every external tool lookup is mocked to a fake path, and the bitrate check
runs before any file or tool is actually touched, so no FRIMEncode/tsMuxeR/ffmpeg/GPU run happens.

Run directly: python tests/test_iw3_bluray_extract_bitrate_ceiling.py (from the nunif/ dir),
or import and call main().
"""
import sys
import tempfile
from os import path
from unittest import mock

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from iw3 import mvc_extract_cli as X  # noqa: E402
from iw3 import sbs_to_mvc_cli as m  # noqa: E402


def _bitrate_error(output_name, bitrate_mbps):
    """None if extract_and_reencode_mvc() got past its own bitrate check for this bitrate;
    otherwise the real bitrate ValueError message. Any later failure (the fake SSIF path does
    not exist) is irrelevant here and is swallowed."""
    with tempfile.TemporaryDirectory() as tmp:
        out = path.join(tmp, output_name)
        with mock.patch.object(X, "_find_tsmuxer", return_value="tsmuxer.exe"), \
             mock.patch.object(X, "_find_edge264_mvc", return_value="edge264.exe"), \
             mock.patch.object(X, "_get_ffmpeg_bin", return_value="ffmpeg.exe"), \
             mock.patch.object(X, "_find_mkvmerge", return_value="mkvmerge.exe"), \
             mock.patch.object(X, "_find_frim", return_value="frim.exe"):
            try:
                X.extract_and_reencode_mvc(path.join(tmp, "fake.ssif"), 0, 1, "0s", None,
                                           path.join(tmp, "work"), out, bitrate_mbps=bitrate_mbps)
            except ValueError as e:
                return str(e)
            except Exception:
                pass
            return None


def _assert_bitrate(bitrate_mbps, expect_rejected, label):
    msg = _bitrate_error("out.mkv", bitrate_mbps)
    rejected = msg is not None and "bitrate must be between" in msg
    assert rejected == expect_rejected, f"{label}: {msg!r}"


def _test_mkv_output_ceiling_is_62_5():
    # The old hard 40 cap must be gone: 50 Mbps is accepted for this .mkv-only layout.
    _assert_bitrate(50, False, "50 Mbps must be accepted for the .mkv MVC re-encode")
    _assert_bitrate(62.5, False, "62.5 Mbps (the .mkv ceiling) must be accepted")
    _assert_bitrate(62.6, True, "just above 62.5 Mbps must be rejected")
    _assert_bitrate(1.9, True, "below the unchanged 2 Mbps floor must still be rejected")
    print("_test_mkv_output_ceiling_is_62_5: PASS")


def _test_matches_shared_helper():
    # The CLI check and the shared ADR-337 helper must not drift apart: sbs_to_mvc_cli still
    # re-exports the same single helper object, and its .mkv answer is the one used here.
    assert m._mvc_bitrate_ceiling_mbps is X._mvc_bitrate_ceiling_mbps
    assert X._mvc_bitrate_ceiling_mbps(disc_legal=False) == 62.5
    _assert_bitrate(X._mvc_bitrate_ceiling_mbps(disc_legal=False), False, "the helper's own .mkv ceiling")
    print("_test_matches_shared_helper: PASS")


def main():
    _test_mkv_output_ceiling_is_62_5()
    _test_matches_shared_helper()
    print("ALL PASS")


if __name__ == "__main__":
    main()
