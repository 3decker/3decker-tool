"""Builds and patches in the real avcC+mvcC Matroska CodecPrivate structure that makes a
3D-aware TV/player auto-detect 3DECKER's own direct `.mkv` MVC output as real 3D.

Real, confirmed root cause (investigated separately, with byte-level evidence against a
real working reference file -- see docs/ai/AI_DECISIONS.md): `_mux_mkv_via_mkvmerge()` in
sbs_to_mvc_cli.py feeds mkvmerge a bare `.264` elementary stream. mkvmerge's raw-ES reader
only understands NAL type 7 (SPS) / type 8 (PPS) when building a video track's
CodecPrivate -- it has no concept of MVC's own subset SPS (NAL type 15) or the ISO/IEC
14496-15 `mvcC` box that declares a track as real Multiview/Stereo-High MVC, so it silently
builds a flat, single-view `avcC` (profile 100, 1 SPS, but BOTH views' PPS collected as if
they were ordinary type-8 NALs) and the one piece of data that would formally declare "this
is real MVC" is dropped entirely. Neither mkvmerge nor mkvpropedit expose any way to set
this (confirmed: mkvmerge --help has no --mvc option; mkvpropedit --list-property-names
does not list codec-private as editable) -- this module builds the correct bytes and
patches them into an already-muxed file directly.

Reference structure (ground truth, parsed byte-for-byte from a real, confirmed-working
MakeMKV-derived file -- "The War with Grandpa (2020) 3D-Clip.mkv", 202-byte CodecPrivate):
  avcC (ISO/IEC 14496-15 AVCDecoderConfigurationRecord):
    configurationVersion=1, AVCProfileIndication/profile_compatibility/AVCLevelIndication
    taken from the base-view SPS's own header bytes, lengthSizeMinusOne=3 (4-byte NALU
    length prefixes), 1 SPS (the base view's), 1 PPS (the base view's), then -- since the
    base view here is always High Profile (100) -- the standard High-Profile extension
    (chroma_format_idc/bit_depth_luma_minus8/bit_depth_chroma_minus8/numOfSPSExt=0).
  mvcC box (box-size(4) + fourcc "mvcC", ISO/IEC 14496-15 MVCDecoderConfigurationRecord):
    configurationVersion=1, AVCProfileIndication/profile_compatibility/AVCLevelIndication
    taken from the dependent view's own subset SPS header bytes (128/"Stereo High" in every
    real case this project produces), lengthSizeMinusOne=3 (matches avcC), 2 "SPS" entries
    (the base view's own SPS again, then the real dependent-view subset SPS, NAL type 15),
    2 PPS entries (the base view's own PPS again, then the dependent view's own PPS). No
    High-Profile extension block here -- MVCDecoderConfigurationRecord doesn't have one,
    confirmed by the reference's own byte count leaving nothing over.

  (The reference's own mvcC box-size field is 4 bytes short of its true content length --
  an apparent authoring quirk that real players evidently tolerate by parsing fields
  sequentially rather than trusting it. This module writes a spec-correct box-size instead
  of reproducing that inconsistency -- same fields, same order, same values, just an
  internally-consistent size.)
"""
import os
from os import path

_CHUNK = 1024 * 1024


# ---------------------------------------------------------------------------
# NAL extraction from a raw Annex-B elementary stream
# ---------------------------------------------------------------------------

def extract_first_nals(es_path, types, head_bytes=8 * 1024 * 1024):
    """Returns {nal_type: nal_bytes} for the FIRST occurrence of each requested
    `nal_unit_type` found in `es_path` (an Annex-B elementary stream, i.e. NALs
    separated by `00 00 01` / `00 00 00 01` start codes) -- `nal_bytes` includes the
    1-byte NAL header, excludes the start code and any trailing zero padding that
    belongs to the NEXT NAL's 4-byte start code variant.

    Only reads the first `head_bytes` of the file (default 8MB, deliberately
    bounded -- a real H.264/MVC elementary stream always carries its SPS/PPS/subset
    SPS within its very first access unit, so this never needs to scan a whole
    multi-GB movie file the way _find_au_boundaries() in mvc_extract_cli.py does for
    access-unit boundaries). A type genuinely missing from the first head_bytes is a
    real encoder-output problem, not something more scanning would fix.
    """
    with open(es_path, "rb") as f:
        buf = f.read(head_bytes)
    starts = []
    idx = 0
    while True:
        idx = buf.find(b"\x00\x00\x01", idx)
        if idx == -1:
            break
        starts.append(idx)
        idx += 3
    found = {}
    remaining = set(types)
    for i, s in enumerate(starts):
        if not remaining:
            break
        if s + 3 >= len(buf):
            continue
        nal_type = buf[s + 3] & 0x1F
        if nal_type not in remaining:
            continue
        end = starts[i + 1] if i + 1 < len(starts) else len(buf)
        nal_bytes = buf[s + 3:end]
        while nal_bytes and nal_bytes[-1] == 0:
            nal_bytes = nal_bytes[:-1]
        found[nal_type] = nal_bytes
        remaining.discard(nal_type)
    return found


# ---------------------------------------------------------------------------
# avcC + mvcC CodecPrivate construction
# ---------------------------------------------------------------------------

def _u16_len_prefixed(nal_bytes):
    if len(nal_bytes) > 0xFFFF:
        raise ValueError(f"NAL unit too large for a 16-bit length prefix ({len(nal_bytes)} bytes)")
    return len(nal_bytes).to_bytes(2, "big") + bytes(nal_bytes)


def build_mvc_codec_private(base_sps, base_pps, dep_subset_sps, dep_pps):
    """Builds the combined avcC+mvcC CodecPrivate blob a real 3D-aware TV/player needs to
    recognize an MKV's video track as real MVC (see module docstring for the full,
    byte-level-verified reference structure this follows). Every argument is a real NAL
    unit (including its own 1-byte NAL header) from THIS output's own encode -- `base_sps`/
    `base_pps` are the base view's (NAL type 7/8), `dep_subset_sps` is the dependent view's
    own subset SPS (NAL type 15), `dep_pps` is the dependent view's own PPS (NAL type 8).
    Never the reference file's own bytes -- only its structure is reused here.
    """
    for name, nal, expected_type in (
        ("base_sps", base_sps, 7), ("base_pps", base_pps, 8),
        ("dep_subset_sps", dep_subset_sps, 15), ("dep_pps", dep_pps, 8),
    ):
        if len(nal) < 4:
            raise ValueError(f"{name} is too short to be a real NAL unit ({len(nal)} bytes)")
        if (nal[0] & 0x1F) != expected_type:
            raise ValueError(f"{name} has NAL type {nal[0] & 0x1F}, expected {expected_type}")

    base_profile, base_compat, base_level = base_sps[1], base_sps[2], base_sps[3]
    dep_profile, dep_compat, dep_level = dep_subset_sps[1], dep_subset_sps[2], dep_subset_sps[3]

    avcc = bytearray()
    avcc += bytes([1, base_profile, base_compat, base_level])
    avcc += bytes([0xFF])  # 6 reserved bits (1) + lengthSizeMinusOne=3 (4-byte NALU lengths)
    avcc += bytes([0xE1])  # 3 reserved bits (1) + numOfSequenceParameterSets=1
    avcc += _u16_len_prefixed(base_sps)
    avcc += bytes([1])     # numOfPictureParameterSets=1
    avcc += _u16_len_prefixed(base_pps)
    if base_profile in (100, 110, 122, 144):
        # High-Profile extension (ISO/IEC 14496-15) -- this pipeline only ever produces
        # 8-bit 4:2:0 output (3D Blu-ray/MVC is hard-capped at 8-bit H.264 High Profile,
        # see sbs_to_mvc_cli.tonemap_hdr_to_sdr_for_bd()'s own docstring), so these three
        # fields are the one real value every real output needs, not a guess.
        avcc += bytes([0xFC | 1])  # chroma_format_idc=1 (4:2:0)
        avcc += bytes([0xF8 | 0])  # bit_depth_luma_minus8=0
        avcc += bytes([0xF8 | 0])  # bit_depth_chroma_minus8=0
        avcc += bytes([0])         # numOfSequenceParameterSetExt=0

    mvcc_content = bytearray()
    mvcc_content += bytes([1, dep_profile, dep_compat, dep_level])
    mvcc_content += bytes([0xFF])  # lengthSizeMinusOne=3, same NALU-length framing as avcC
    mvcc_content += bytes([0xE2])  # numOfSequenceParameterSets=2 (base SPS + dependent subset SPS)
    mvcc_content += _u16_len_prefixed(base_sps)
    mvcc_content += _u16_len_prefixed(dep_subset_sps)
    mvcc_content += bytes([2])     # numOfPictureParameterSets=2 (base PPS + dependent PPS)
    mvcc_content += _u16_len_prefixed(base_pps)
    mvcc_content += _u16_len_prefixed(dep_pps)
    mvcc_box = (8 + len(mvcc_content)).to_bytes(4, "big") + b"mvcC" + bytes(mvcc_content)

    return bytes(avcc) + mvcc_box


# ---------------------------------------------------------------------------
# EBML parsing (minimal, scoped to what this module needs to patch)
# ---------------------------------------------------------------------------

_ID_SEGMENT = b"\x18\x53\x80\x67"
_ID_TRACKS = b"\x16\x54\xae\x6b"
_ID_TRACKENTRY = b"\xae"
_ID_TRACKTYPE = b"\x83"
_ID_CODECPRIVATE = b"\x63\xa2"
_ID_VOID = b"\xec"
_TRACK_TYPE_VIDEO = 1


def _vint_length(first_byte):
    for i in range(8):
        if first_byte & (0x80 >> i):
            return i + 1
    raise ValueError("invalid EBML vint: lead byte is 0x00")


def _read_id(buf, pos):
    length = _vint_length(buf[pos])
    return bytes(buf[pos:pos + length]), length


def _read_size(buf, pos):
    length = _vint_length(buf[pos])
    marker_bit_value = 0x80 >> (length - 1)
    value_bits_in_first_byte = buf[pos] & (marker_bit_value - 1)
    raw = bytes([value_bits_in_first_byte]) + bytes(buf[pos + 1:pos + length])
    value = int.from_bytes(raw, "big")
    is_unknown = value == (1 << (7 * length)) - 1
    return (None if is_unknown else value), length


def _encode_size_vint(value, min_length=1):
    length = max(min_length, 1)
    while True:
        max_value = (1 << (7 * length)) - 2  # reserve the all-1s pattern for "unknown size"
        if 0 <= value <= max_value:
            break
        length += 1
        if length > 8:
            raise ValueError(f"value {value} does not fit in an 8-byte EBML vint")
    marker = 1 << (7 * length)
    return (value | marker).to_bytes(length, "big")


def _iter_children(buf, start, end):
    """Yields (id_bytes, size_value, header_len, data_start, data_end) for each direct
    child element in buf[start:end]. Raises on an unknown-size child -- none of the
    (always mkvmerge-produced, non-streamed) elements this module ever walks are
    expected to use it."""
    pos = start
    while pos < end:
        id_bytes, id_len = _read_id(buf, pos)
        size_value, size_len = _read_size(buf, pos + id_len)
        if size_value is None:
            raise ValueError(f"unexpected unknown-size EBML element {id_bytes.hex()} at offset {pos}")
        data_start = pos + id_len + size_len
        data_end = data_start + size_value
        yield id_bytes, size_value, id_len + size_len, data_start, data_end
        pos = data_end


def _scan_top_level_file(f, start, end):
    """Same as _iter_children, but reads only each element's small id+size header
    directly from the file via seek+read -- never the element's own (possibly
    multi-GB, for a real Cluster) data -- so this is safe to call across an entire
    Segment regardless of file size."""
    out = []
    pos = start
    while pos < end:
        f.seek(pos)
        header = f.read(12)  # an id (<=4 bytes) + a size vint (<=8 bytes) always fits
        id_bytes, id_len = _read_id(header, 0)
        size_value, size_len = _read_size(header, id_len)
        if size_value is None:
            raise ValueError(f"unexpected unknown-size top-level EBML element {id_bytes.hex()} at {pos}")
        data_start = pos + id_len + size_len
        data_end = data_start + size_value
        out.append({"id": id_bytes, "start": pos, "header_len": id_len + size_len,
                    "data_start": data_start, "data_end": data_end})
        pos = data_end
    return out


def _find_video_codec_private_span(tracks_buf):
    """Within an in-memory Tracks element's own data bytes (`tracks_buf`, 0-based
    offsets), finds the one TrackEntry whose TrackType is video (1) and returns
    (codec_private_data_start, codec_private_data_end, codec_private_header_start)
    -- all offsets relative to `tracks_buf`. Raises if there isn't exactly one such
    TrackEntry, or it has no CodecPrivate child, so a wrong guess here is a hard,
    visible failure rather than a silently-patched wrong track."""
    video_entries = []
    for te_id, te_size, te_header_len, te_data_start, te_data_end in _iter_children(
            tracks_buf, 0, len(tracks_buf)):
        if te_id != _ID_TRACKENTRY:
            continue
        track_type = None
        cp_span = None
        for c_id, c_size, c_header_len, c_data_start, c_data_end in _iter_children(
                tracks_buf, te_data_start, te_data_end):
            if c_id == _ID_TRACKTYPE:
                track_type = int.from_bytes(tracks_buf[c_data_start:c_data_end], "big")
            elif c_id == _ID_CODECPRIVATE:
                cp_span = (c_data_start - c_header_len, c_data_start, c_data_end)
        if track_type == _TRACK_TYPE_VIDEO:
            video_entries.append((te_data_start - te_header_len, te_data_end, cp_span))
    if len(video_entries) != 1:
        raise RuntimeError(f"expected exactly one video TrackEntry, found {len(video_entries)}")
    te_start, te_end, cp_span = video_entries[0]
    if cp_span is None:
        raise RuntimeError("the video TrackEntry has no CodecPrivate element to patch")
    return te_start, te_end, cp_span


def patch_mkv_codec_private(mkv_path, new_codec_private):
    """Replaces the video track's CodecPrivate in an already-muxed `mkv_path` with
    `new_codec_private`, in place, touching ONLY the Tracks element (specifically its
    video TrackEntry's CodecPrivate) and the EBML Void padding element mkvmerge
    reliably writes immediately after Tracks (confirmed directly against this
    project's own real mkvmerge output -- every sample checked reserves several KB of
    Void there, the same convention mkvpropedit itself depends on for in-place
    metadata edits). Shrinking that Void by exactly as many bytes as CodecPrivate
    grows means NOTHING else in the file moves: SeekHead's seek positions, Cues'
    cluster positions, and every Cluster's own absolute byte offset are all
    unaffected, so this never needs to touch them.

    Raises RuntimeError (never silently corrupts the file) if that Void isn't present
    or isn't big enough -- in that case the caller should warn and leave the
    mkvmerge-produced file as-is, the same way this project already handles
    mkvpropedit being missing for the StereoMode tag.

    Writes atomically (tmp-name + os.replace, CS-IO-001): the original file is only
    ever replaced once the full patched copy has been written successfully.
    """
    with open(mkv_path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        top = _scan_top_level_file(f, 0, file_size)
        segment = next((e for e in top if e["id"] == _ID_SEGMENT), None)
        if segment is None:
            raise RuntimeError("no top-level Segment element found")
        seg_children = _scan_top_level_file(f, segment["data_start"], segment["data_end"])
        tracks_idx = next((i for i, e in enumerate(seg_children) if e["id"] == _ID_TRACKS), None)
        if tracks_idx is None:
            raise RuntimeError("no Tracks element found inside Segment")
        tracks = seg_children[tracks_idx]
        void_elem = seg_children[tracks_idx + 1] if tracks_idx + 1 < len(seg_children) else None
        if void_elem is None or void_elem["id"] != _ID_VOID:
            raise RuntimeError("no EBML Void padding element immediately follows Tracks -- "
                               "cannot safely grow Tracks in place without shifting (and "
                               "re-pointing) every element after it, which this patcher "
                               "deliberately does not attempt")

        f.seek(tracks["start"])
        tracks_full = f.read(tracks["data_end"] - tracks["start"])
        void_header_len = void_elem["header_len"]
        old_void_size = void_elem["data_end"] - void_elem["data_start"]

    tracks_header_len = tracks["header_len"]
    tracks_data = tracks_full[tracks_header_len:]
    te_start, te_end, cp_span = _find_video_codec_private_span(tracks_data)
    cp_header_start, cp_data_start, cp_data_end = cp_span

    # te_start is the TrackEntry ELEMENT's start (its own id+size header included) --
    # the new TrackEntry's DATA must start right AFTER that header, or the old header
    # bytes end up duplicated as a bogus nested child once wrapped in a freshly-encoded
    # header below (a real bug caught live: mkvmerge then reported zero tracks at all).
    te_id, te_id_len = _read_id(tracks_data, te_start)
    _te_old_size, te_size_len = _read_size(tracks_data, te_start + te_id_len)
    te_data_start = te_start + te_id_len + te_size_len

    new_cp_elem = _ID_CODECPRIVATE + _encode_size_vint(len(new_codec_private)) + bytes(new_codec_private)
    new_te_data = tracks_data[te_data_start:cp_header_start] + new_cp_elem + tracks_data[cp_data_end:te_end]
    new_te_elem = te_id + _encode_size_vint(len(new_te_data)) + new_te_data

    new_tracks_data = tracks_data[:te_start] + new_te_elem + tracks_data[te_end:]
    new_tracks_elem = _ID_TRACKS + _encode_size_vint(len(new_tracks_data)) + new_tracks_data

    delta = len(new_tracks_elem) - len(tracks_full)
    new_void_size = old_void_size - delta
    if new_void_size < 0:
        raise RuntimeError(
            f"the Void padding after Tracks is only {old_void_size} bytes, but the real "
            f"MVC CodecPrivate needs {delta} more bytes than mkvmerge's own broken one -- "
            f"cannot safely patch this file in place")
    new_void_elem = _ID_VOID + _encode_size_vint(new_void_size, min_length=void_header_len - 1) + \
        b"\x00" * new_void_size
    if len(new_void_elem) != void_header_len + old_void_size - delta:
        raise RuntimeError("internal error: patched Void element size does not balance the "
                           "bytes added to Tracks -- refusing to write a possibly-corrupt file")

    tmp_path = mkv_path + ".codecprivate.tmp"
    try:
        with open(mkv_path, "rb") as src, open(tmp_path, "wb") as dst:
            while dst.tell() < tracks["start"]:
                chunk = src.read(min(_CHUNK, tracks["start"] - dst.tell()))
                if not chunk:
                    break
                dst.write(chunk)
            dst.write(new_tracks_elem)
            dst.write(new_void_elem)
            src.seek(void_elem["data_end"])
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
        os.replace(tmp_path, mkv_path)
    finally:
        if path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
