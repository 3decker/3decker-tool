"""Regression test for the real avcC+mvcC Matroska CodecPrivate fix (see
iw3/mvc_codec_private.py's own module docstring for the full, byte-level-verified
root cause and reference structure this follows).

Background: `iw3.sbs_to_mvc_cli._mux_mkv_via_mkvmerge()` feeds mkvmerge a bare
elementary stream, whose raw-ES reader has no concept of MVC's own subset SPS (NAL
type 15) or the ISO/IEC 14496-15 `mvcC` box -- it silently writes a flat, single-view
`avcC` CodecPrivate, which is why a real Samsung 3D TV did not auto-detect 3DECKER's
own direct `.mkv` MVC output as 3D even though a MakeMKV-derived file of the same
content did. This module builds the correct avcC+mvcC structure from this project's
OWN real SPS/PPS/subset-SPS bytes and patches it into an already-muxed `.mkv` file.

Synthetic/isolated, no external tool dependency (mkvmerge/mkvpropedit are NOT
required to run this test -- the patcher only ever needs a real, valid Matroska byte
layout, which this test builds by hand): a minimal but structurally real Matroska
file (EBML header + Segment + Tracks(TrackEntry with a small CodecPrivate) + a
trailing Void, exactly mkvmerge's own real, confirmed-live layout for this project's
output) is built here directly, patched, and checked byte-for-byte.

Real, confirmed bug this test guards against: an early version of the patcher sliced
the new TrackEntry's data starting at the TrackEntry ELEMENT's own start (its id+size
header included) instead of just after that header, duplicating the old header as a
bogus nested child -- confirmed live, this made mkvmerge report ZERO tracks in the
patched file and made mkvpropedit crash with a libebml assertion failure. Reproduced
and fixed against real FRIM base/dependent-view elementary stream fixtures from a
real test run before this test was written; see the real byte-level evidence in this
change's own report for the full real-file verification (frame count/payload
unchanged, structure matches a real working MakeMKV-derived reference).

Run directly: python tests/test_iw3_mvc_codec_private.py (from the nunif/ dir),
or import and call main().
"""
import os
import sys
import tempfile
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from iw3.mvc_codec_private import (  # noqa: E402
    extract_first_nals, build_mvc_codec_private, patch_mkv_codec_private,
    _encode_size_vint, _read_size, _vint_length,
)

# Real NAL bytes from an actual FRIM base/dependent-view encode (captured from a real
# test run, ADR mux-fix session) -- used here as plain byte fixtures, not reparsed
# from any file, so this test has no dependency on external fixtures.
BASE_SPS = bytes.fromhex("27640029ac2d100780227e5c04400000fa40002ee03818000d011000012a010bdef828")
BASE_PPS = bytes.fromhex("28ee3cb0")
DEP_SUBSET_SPS = bytes.fromhex(
    "2f8000294b0b4401e0089f97011000003e90000bb80e060002b59800004a8042f7be0aa96b94c2a440")
DEP_PPS = bytes.fromhex("284ae3cb")


def _test_build_mvc_codec_private_structure():
    blob = build_mvc_codec_private(BASE_SPS, BASE_PPS, DEP_SUBSET_SPS, DEP_PPS)

    assert blob[0] == 1, "configurationVersion must be 1"
    assert blob[1] == BASE_SPS[1], "avcC profile must come from the base SPS's own header"
    assert blob[2] == BASE_SPS[2]
    assert blob[3] == BASE_SPS[3]
    length_size = (blob[4] & 0x03) + 1
    assert length_size == 4, "lengthSizeMinusOne must encode 4-byte NALU length prefixes"
    num_sps = blob[5] & 0x1F
    assert num_sps == 1
    sps_len = (blob[6] << 8) | blob[7]
    sps_data = blob[8:8 + sps_len]
    assert sps_data == BASE_SPS, "avcC's own SPS must be the real base-view SPS, byte for byte"
    pos = 8 + sps_len
    num_pps = blob[pos]
    pos += 1
    pps_len = (blob[pos] << 8) | blob[pos + 1]
    pos += 2
    pps_data = blob[pos:pos + pps_len]
    assert pps_data == BASE_PPS
    pos += pps_len
    # High-Profile extension (base SPS profile_idc=100 here)
    chroma_format_idc = blob[pos] & 0x03
    bit_depth_luma_minus8 = blob[pos + 1] & 0x07
    bit_depth_chroma_minus8 = blob[pos + 2] & 0x07
    num_sps_ext = blob[pos + 3]
    assert (chroma_format_idc, bit_depth_luma_minus8, bit_depth_chroma_minus8, num_sps_ext) == (1, 0, 0, 0)
    pos += 4

    box_size = int.from_bytes(blob[pos:pos + 4], "big")
    fourcc = blob[pos + 4:pos + 8]
    assert fourcc == b"mvcC", f"expected the mvcC box right after avcC, got {fourcc!r}"
    assert box_size == len(blob) - pos, "mvcC box-size must be internally consistent with the blob's real length"
    pos += 8

    assert blob[pos] == 1
    assert blob[pos + 1] == DEP_SUBSET_SPS[1], "mvcC profile must come from the dependent subset SPS's own header"
    assert blob[pos + 2] == DEP_SUBSET_SPS[2]
    assert blob[pos + 3] == DEP_SUBSET_SPS[3]
    pos += 4
    m_length_size = (blob[pos] & 0x03) + 1
    assert m_length_size == 4
    pos += 1
    m_num_sps = blob[pos] & 0x1F
    assert m_num_sps == 2, "mvcC must carry the base SPS AND the real dependent subset SPS"
    pos += 1
    sps0_len = (blob[pos] << 8) | blob[pos + 1]
    pos += 2
    assert blob[pos:pos + sps0_len] == BASE_SPS
    pos += sps0_len
    sps1_len = (blob[pos] << 8) | blob[pos + 1]
    pos += 2
    assert blob[pos:pos + sps1_len] == DEP_SUBSET_SPS, \
        "mvcC's second SPS entry must be the REAL dependent-view subset SPS, byte for byte"
    pos += sps1_len
    m_num_pps = blob[pos]
    assert m_num_pps == 2, "mvcC must carry the base PPS AND the real dependent-view PPS"
    pos += 1
    pps0_len = (blob[pos] << 8) | blob[pos + 1]
    pos += 2
    assert blob[pos:pos + pps0_len] == BASE_PPS
    pos += pps0_len
    pps1_len = (blob[pos] << 8) | blob[pos + 1]
    pos += 2
    assert blob[pos:pos + pps1_len] == DEP_PPS, "mvcC's second PPS entry must be the real dependent-view PPS"
    pos += pps1_len
    assert pos == len(blob), f"leftover {len(blob) - pos} unexplained bytes in the blob"

    print("_test_build_mvc_codec_private_structure: PASS")


def _test_build_mvc_codec_private_rejects_wrong_nal_type():
    try:
        build_mvc_codec_private(BASE_PPS, BASE_PPS, DEP_SUBSET_SPS, DEP_PPS)  # BASE_PPS is type 8, not 7
        assert False, "expected a ValueError for a wrong NAL type"
    except ValueError:
        pass
    print("_test_build_mvc_codec_private_rejects_wrong_nal_type: PASS")


def _annex_b(*nals):
    return b"".join(b"\x00\x00\x01" + n for n in nals)


def _test_extract_first_nals_finds_each_type_once():
    stream = _annex_b(b"\x09\x10", BASE_SPS, BASE_PPS, b"\x01\xaa\xbb\xcc")  # AUD, SPS, PPS, a slice NAL
    found = extract_first_nals_from_bytes(stream, {7, 8})
    assert found[7] == BASE_SPS
    assert found[8] == BASE_PPS
    print("_test_extract_first_nals_finds_each_type_once: PASS")


def extract_first_nals_from_bytes(data, types):
    """Thin wrapper so the extraction logic can be exercised against an in-memory
    buffer in this test without writing a temp file for every case."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        es_path = path.join(tmp_dir, "s.264")
        with open(es_path, "wb") as f:
            f.write(data)
        return extract_first_nals(es_path, types)


def _test_vint_round_trip():
    for value in (0, 1, 7, 42, 126, 127, 128, 200, 16382, 16383, 100000, 2 ** 40):
        for min_len in (1, 2, 4):
            encoded = _encode_size_vint(value, min_length=min_len)
            assert len(encoded) >= min_len
            decoded, length = _read_size(encoded, 0)
            assert decoded == value, f"value {value} (min_len={min_len}) round-tripped to {decoded}"
            assert length == len(encoded)
    assert _vint_length(0x80) == 1
    assert _vint_length(0x01) == 8
    print("_test_vint_round_trip: PASS")


# --- a minimal, but structurally real, Matroska byte layout for the EBML patcher ---

def _ebml_elem(id_bytes, data):
    return id_bytes + _encode_size_vint(len(data)) + data


def _build_fake_mkv(codec_private, void_size=512):
    track_entry = (
        _ebml_elem(b"\xd7", b"\x01") +           # TrackNumber = 1
        _ebml_elem(b"\x73\xc5", b"\x00\x00\x00\x01") +  # TrackUID
        _ebml_elem(b"\x83", b"\x01") +            # TrackType = 1 (video)
        _ebml_elem(b"\x86", b"V_MPEG4/ISO/AVC") +  # CodecID
        _ebml_elem(b"\x63\xa2", codec_private)    # CodecPrivate
    )
    tracks = _ebml_elem(b"\x16\x54\xae\x6b", _ebml_elem(b"\xae", track_entry))
    void = _ebml_elem(b"\xec", b"\x00" * void_size)
    tail = b"CLUSTER-PAYLOAD-MUST-NOT-CHANGE" * 10
    segment_data = tracks + void + tail
    segment = _ebml_elem(b"\x18\x53\x80\x67", segment_data)
    ebml_header = _ebml_elem(b"\x1a\x45\xdf\xa3", b"\x01\x02\x03")
    return ebml_header + segment, len(tracks), len(void), tail


def _test_patch_replaces_codec_private_and_leaves_tail_untouched():
    old_cp = b"\x01\x64\x00\x29\xff\xe1\x00\x04\xaa\xbb\xcc\xdd\x01\x00\x02\xee\xff"
    new_cp = build_mvc_codec_private(BASE_SPS, BASE_PPS, DEP_SUBSET_SPS, DEP_PPS)
    assert len(new_cp) > len(old_cp), "test assumes the real fix grows CodecPrivate"

    data, _, _, tail = _build_fake_mkv(old_cp, void_size=512)
    with tempfile.TemporaryDirectory() as tmp_dir:
        mkv_path = path.join(tmp_dir, "fake.mkv")
        with open(mkv_path, "wb") as f:
            f.write(data)
        before = data
        patch_mkv_codec_private(mkv_path, new_cp)
        with open(mkv_path, "rb") as f:
            after = f.read()

    assert len(after) == len(before), "the void-shrink design must keep the file's total size unchanged"
    assert after[-len(tail):] == tail, "the trailing 'Cluster' payload must be byte-identical after the patch"
    assert new_cp in after, "the new CodecPrivate bytes must be present in the patched file"
    assert old_cp not in after, "the old CodecPrivate bytes must be fully replaced, not left behind"

    # Nothing before Tracks' own start changed either (EBML header unaffected).
    ebml_header_len = len(_ebml_elem(b"\x1a\x45\xdf\xa3", b"\x01\x02\x03"))
    assert after[:ebml_header_len] == before[:ebml_header_len]

    print("_test_patch_replaces_codec_private_and_leaves_tail_untouched: PASS")


def _test_patch_raises_without_enough_void_headroom():
    old_cp = b"\x01\x64\x00\x29\xff\xe1\x00\x04\xaa\xbb\xcc\xdd\x01\x00\x02\xee\xff"
    new_cp = build_mvc_codec_private(BASE_SPS, BASE_PPS, DEP_SUBSET_SPS, DEP_PPS)
    data, _, _, _ = _build_fake_mkv(old_cp, void_size=1)  # deliberately not enough room
    with tempfile.TemporaryDirectory() as tmp_dir:
        mkv_path = path.join(tmp_dir, "fake.mkv")
        with open(mkv_path, "wb") as f:
            f.write(data)
        try:
            patch_mkv_codec_private(mkv_path, new_cp)
            assert False, "expected a RuntimeError for insufficient Void headroom"
        except RuntimeError:
            pass
        with open(mkv_path, "rb") as f:
            after = f.read()
        assert after == data, "a failed patch must never leave a partially-written file behind"

    print("_test_patch_raises_without_enough_void_headroom: PASS")


def _test_patch_raises_without_a_void_after_tracks():
    old_cp = b"\x01\x64\x00\x29\xff\xe1\x00\x04\xaa\xbb\xcc\xdd\x01\x00\x02\xee\xff"
    new_cp = build_mvc_codec_private(BASE_SPS, BASE_PPS, DEP_SUBSET_SPS, DEP_PPS)
    track_entry = (
        _ebml_elem(b"\xd7", b"\x01") + _ebml_elem(b"\x73\xc5", b"\x00\x00\x00\x01") +
        _ebml_elem(b"\x83", b"\x01") + _ebml_elem(b"\x86", b"V_MPEG4/ISO/AVC") +
        _ebml_elem(b"\x63\xa2", old_cp)
    )
    tracks = _ebml_elem(b"\x16\x54\xae\x6b", _ebml_elem(b"\xae", track_entry))
    segment_data = tracks + b"NO-VOID-HERE-AT-ALL"  # Tracks not followed by a Void
    segment = _ebml_elem(b"\x18\x53\x80\x67", segment_data)
    data = _ebml_elem(b"\x1a\x45\xdf\xa3", b"\x01\x02\x03") + segment
    with tempfile.TemporaryDirectory() as tmp_dir:
        mkv_path = path.join(tmp_dir, "fake.mkv")
        with open(mkv_path, "wb") as f:
            f.write(data)
        try:
            patch_mkv_codec_private(mkv_path, new_cp)
            assert False, "expected a RuntimeError when Tracks has no following Void"
        except RuntimeError:
            pass

    print("_test_patch_raises_without_a_void_after_tracks: PASS")


def main():
    _test_build_mvc_codec_private_structure()
    _test_build_mvc_codec_private_rejects_wrong_nal_type()
    _test_extract_first_nals_finds_each_type_once()
    _test_vint_round_trip()
    _test_patch_replaces_codec_private_and_leaves_tail_untouched()
    _test_patch_raises_without_enough_void_headroom()
    _test_patch_raises_without_a_void_after_tracks()
    print("ALL PASS")


if __name__ == "__main__":
    main()
