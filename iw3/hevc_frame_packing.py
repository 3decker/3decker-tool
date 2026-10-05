"""Post-encode HEVC frame-packing SEI tagging (the 3D TV auto-detect signal).

libx264 gets its Frame Packing Arrangement SEI from x264 itself (x264-params
frame-packing=3/4, see make_video_codec_option()). libx265 and hevc_nvenc cannot
write that SEI through ffmpeg (confirmed: libx265 says "Unknown option", hevc_nvenc
has no such option, and frame-level Stereo 3D side data is not written into the HEVC
bitstream). So this module inserts the SEI NAL unit (payload type 45) into the finished
HEVC file instead.

How it stays safe:
* Only the video packets are changed. Every other stream is copied packet-for-packet
  with its original timing (stream copy, no re-encode).
* The result is written to a temp file in the same folder and replaces the original
  only after it has been verified (ffprobe opens it, stream layout, video packet
  count and duration match, and the first access units carry the SEI).
* Any failure removes the temp file and leaves the original exactly as it was. The
  conversion itself is never failed by this step.

Byte layout written (identical to the SEI libx264 writes for the same layout, verified
by a byte comparison in the self-test):
  NAL header   4E 01          (nal_unit_type 39 = prefix SEI, layer 0, tid 0)
  SEI body     2D 07 ...      (payload type 45, payload size 7, 7 payload bytes,
                               then 80 = rbsp trailing bits), emulation prevention applied.
  Half SBS: arrangement type 3 -> body 2D 07 81 81 00 00 03 00 01 20 80
  Half TB:  arrangement type 4 -> body 2D 07 82 01 00 00 03 00 01 20 80
"""

import os
import json
import shutil
import subprocess

import av


FRAME_PACKING_PAYLOAD_TYPE = 45
SEI_NAL_TYPE = 39
LAYOUT_ARRANGEMENT_TYPE = {"half_sbs": 3, "half_tb": 4}

# Expected SEI bodies (after the 2-byte NAL header) as written by libx264, used by the
# self-test to prove the writer matches a real encoder byte-for-byte.
_X264_BODY = {
    "half_sbs": bytes.fromhex("2d0781810000030001" + "2080"),
    "half_tb": bytes.fromhex("2d07820100000300012080"),
}


class _BitWriter:
    def __init__(self):
        self._bits = []

    def u(self, value, nbits):
        for i in range(nbits - 1, -1, -1):
            self._bits.append((value >> i) & 1)

    def ue(self, value):
        code = value + 1
        n = code.bit_length()
        self.u(0, n - 1)
        self.u(code, n)

    def to_bytes_with_trailing(self):
        # rbsp/payload trailing: one '1' bit then zeros to the byte boundary
        self._bits.append(1)
        while len(self._bits) % 8:
            self._bits.append(0)
        out = bytearray()
        for i in range(0, len(self._bits), 8):
            v = 0
            for b in self._bits[i:i + 8]:
                v = (v << 1) | b
            out.append(v)
        return bytes(out)


def _emulation_prevent(rbsp):
    out = bytearray()
    zeros = 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    if zeros:
        out.append(3)  # a trailing zero byte would be ambiguous in a NAL unit
    return bytes(out)


def build_frame_packing_sei_nal(layout):
    """Returns one complete HEVC prefix-SEI NAL unit (2-byte header + escaped body) that
    carries a Frame Packing Arrangement message for 'half_sbs' or 'half_tb'."""
    if layout not in LAYOUT_ARRANGEMENT_TYPE:
        raise ValueError(f"unsupported frame packing layout: {layout}")
    arrangement_type = LAYOUT_ARRANGEMENT_TYPE[layout]
    bw = _BitWriter()
    bw.ue(0)                      # frame_packing_arrangement_id
    bw.u(0, 1)                    # frame_packing_arrangement_cancel_flag
    bw.u(arrangement_type, 7)     # frame_packing_arrangement_type (3 = side by side, 4 = top bottom)
    bw.u(0, 1)                    # quincunx_sampling_flag
    bw.u(1, 6)                    # content_interpretation_type (frame 0 = left, frame 1 = right)
    bw.u(0, 1)                    # spatial_flipping_flag
    bw.u(0, 1)                    # frame0_flipped_flag
    bw.u(0, 1)                    # field_views_flag
    bw.u(0, 1)                    # current_frame_is_frame0_flag
    bw.u(0, 1)                    # frame0_self_contained_flag
    bw.u(0, 1)                    # frame1_self_contained_flag
    bw.u(0, 4)                    # frame0_grid_position_x
    bw.u(0, 4)                    # frame0_grid_position_y
    bw.u(0, 4)                    # frame1_grid_position_x
    bw.u(0, 4)                    # frame1_grid_position_y
    bw.u(0, 8)                    # frame_packing_arrangement_reserved_byte
    bw.ue(1)                      # frame_packing_arrangement_repetition_period
    bw.u(0, 1)                    # frame_packing_arrangement_extension_flag
    payload = bw.to_bytes_with_trailing()
    assert len(payload) < 255
    rbsp = bytes([FRAME_PACKING_PAYLOAD_TYPE, len(payload)]) + payload + b"\x80"
    body = _emulation_prevent(rbsp)
    header = bytes([SEI_NAL_TYPE << 1, 0x01])
    return header + body


def _nal_body_for_test(layout):
    return build_frame_packing_sei_nal(layout)[2:]


def _nal_format(extradata):
    """'hvcC' (length-prefixed, MP4/MKV) returns ('len', n); Annex B returns ('annexb', 0)."""
    if extradata and len(extradata) >= 23 and extradata[0] == 1:
        return ("len", (extradata[21] & 3) + 1)
    return ("annexb", 0)


def _split_nals(data, fmt):
    kind, nal_len = fmt
    nals = []
    if kind == "len":
        i = 0
        n = len(data)
        while i < n:
            if i + nal_len > n:
                raise ValueError("truncated NAL length prefix")
            ln = int.from_bytes(data[i:i + nal_len], "big")
            i += nal_len
            if ln <= 0 or i + ln > n:
                raise ValueError("NAL length out of range")
            nals.append(data[i:i + ln])
            i += ln
    else:
        positions = []
        i = 0
        while True:
            j = data.find(b"\x00\x00\x01", i)
            if j < 0:
                break
            positions.append(j)
            i = j + 3
        if not positions or positions[0] > 1:
            raise ValueError("no Annex B start code at the start of the access unit")
        for k, j in enumerate(positions):
            end = positions[k + 1] if k + 1 < len(positions) else len(data)
            seg = data[j + 3:end].rstrip(b"\x00")
            if not seg:
                raise ValueError("empty NAL unit")
            nals.append(seg)
    return nals


def _join_nals(nals, fmt):
    kind, nal_len = fmt
    out = bytearray()
    for nal in nals:
        if kind == "len":
            out += len(nal).to_bytes(nal_len, "big")
        else:
            out += b"\x00\x00\x00\x01"
        out += nal
    return bytes(out)


def _is_frame_packing_sei(nal):
    return len(nal) > 2 and ((nal[0] >> 1) & 0x3F) == SEI_NAL_TYPE and nal[2] == FRAME_PACKING_PAYLOAD_TYPE


def tag_access_unit(data, fmt, sei_nal):
    """Returns (new_bytes, inserted). Inserts the SEI just before the first slice (VCL) NAL,
    which is where a prefix SEI must sit. If the access unit already has one, it is left
    alone (inserted=False). Raises ValueError on an unexpected layout."""
    nals = _split_nals(data, fmt)
    if any(_is_frame_packing_sei(n) for n in nals):
        return data, False
    vcl = [i for i, n in enumerate(nals) if ((n[0] >> 1) & 0x3F) < 32]
    if not vcl:
        raise ValueError("access unit has no slice NAL unit")
    first = vcl[0]
    nals = nals[:first] + [sei_nal] + nals[first:]
    return _join_nals(nals, fmt), True


def _find_ffprobe():
    from . import utils as _utils
    ffmpeg = _utils._get_ffmpeg_bin()
    if ffmpeg:
        name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        cand = os.path.join(os.path.dirname(ffmpeg), name)
        if os.path.isfile(cand):
            return cand
    found = shutil.which("ffprobe")
    if not found:
        raise FileNotFoundError("ffprobe not found")
    return found


def _probe(ffprobe, path):
    cmd = [ffprobe, "-v", "error", "-count_packets", "-show_streams", "-show_format",
           "-of", "json", path]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {os.path.basename(path)}: {res.stderr.strip()[:300]}")
    return json.loads(res.stdout)


def _rewrite(src, dst, sei_nal):
    """Stream-copies src to dst, inserting the SEI into every HEVC video access unit.
    Returns the number of video packets written."""
    inp = av.open(src)
    out = None
    try:
        vids = [s for s in inp.streams if s.type == "video"]
        if len(vids) != 1 or vids[0].codec_context.name != "hevc":
            raise ValueError("expected exactly one HEVC video stream")
        for s in inp.streams:
            if s.type not in ("video", "audio", "subtitle"):
                raise ValueError(f"unsupported stream type in output: {s.type}")
        vin = vids[0]
        fmt = _nal_format(bytes(vin.codec_context.extradata or b""))

        container_options = {"movflags": "+faststart"} if dst.lower().endswith(".mp4") else {}
        out = av.open(dst, "w", container_options=container_options)
        out.metadata.update(inp.metadata)
        ostreams = {}
        for s in inp.streams:
            o = out.add_stream_from_template(s)
            ostreams[s.index] = o

        n_video = 0
        for pkt in inp.demux(*inp.streams):
            # Only the empty end-of-stream flush packet is skipped. Real packets can have
            # dts None (MKV B-frame reordering), so dts is NOT used as a filter.
            if pkt.size == 0:
                continue
            if pkt.stream_index == vin.index:
                new_data, _ = tag_access_unit(bytes(pkt), fmt, sei_nal)
                newp = av.Packet(new_data)
                newp.pts = pkt.pts
                newp.dts = pkt.dts
                newp.duration = pkt.duration
                newp.time_base = pkt.time_base
                newp.is_keyframe = pkt.is_keyframe
                newp.stream = ostreams[pkt.stream_index]
                out.mux(newp)
                n_video += 1
            else:
                pkt.stream = ostreams[pkt.stream_index]
                out.mux(pkt)
        out.close()
        out = None
        return n_video
    finally:
        if out is not None:
            try:
                out.close()
            except Exception:
                pass
        inp.close()


def _verify_output(ffprobe, src, tmp, n_video):
    """Checks the rewritten file before it may replace the original. Raises on any mismatch."""
    orig = _probe(ffprobe, src)
    new = _probe(ffprobe, tmp)
    o_streams = orig.get("streams", [])
    n_streams = new.get("streams", [])
    if len(o_streams) != len(n_streams):
        raise ValueError("stream count changed")
    for a, b in zip(o_streams, n_streams):
        if a.get("codec_type") != b.get("codec_type") or a.get("codec_name") != b.get("codec_name"):
            raise ValueError("stream type/codec changed")
        if a.get("codec_type") == "video":
            if b.get("codec_name") != "hevc":
                raise ValueError("output video is not HEVC")
            if int(b.get("nb_read_packets", -1)) != n_video or int(a.get("nb_read_packets", -2)) != n_video:
                raise ValueError(f"video packet count mismatch (orig {a.get('nb_read_packets')}, "
                                 f"new {b.get('nb_read_packets')}, written {n_video})")
    od = float(orig.get("format", {}).get("duration", 0.0))
    nd = float(new.get("format", {}).get("duration", 0.0))
    if abs(od - nd) > 0.05:
        raise ValueError(f"duration changed ({od:.3f}s -> {nd:.3f}s)")

    # The first access units must carry the SEI before their first slice.
    inp = av.open(tmp)
    try:
        vin = [s for s in inp.streams if s.type == "video"][0]
        fmt = _nal_format(bytes(vin.codec_context.extradata or b""))
        checked = 0
        for pkt in inp.demux(vin):
            if pkt.size == 0:
                continue
            nals = _split_nals(bytes(pkt), fmt)
            sei_idx = [i for i, n in enumerate(nals) if _is_frame_packing_sei(n)]
            vcl_idx = [i for i, n in enumerate(nals) if ((n[0] >> 1) & 0x3F) < 32]
            if not sei_idx or not vcl_idx or sei_idx[0] > vcl_idx[0]:
                raise ValueError("frame-packing SEI missing from an access unit")
            checked += 1
            if checked >= 3:
                break
        if checked == 0:
            raise ValueError("no video packets to check")
    finally:
        inp.close()


def tag_hevc_frame_packing(path, layout, log=print):
    """Adds the frame-packing SEI to an HEVC file in place (fail-safe). Returns True when the
    file was tagged, False when it was left untouched (for any reason, which is logged)."""
    if layout not in LAYOUT_ARRANGEMENT_TYPE:
        raise ValueError(f"unsupported frame packing layout: {layout}")
    if not os.path.isfile(path):
        log(f"WARNING: HEVC 3D TV tag skipped, file not found: {path}")
        return False
    root, ext = os.path.splitext(path)
    tmp = root + ".fp_tmp" + ext
    done = False
    try:
        sei_nal = build_frame_packing_sei_nal(layout)
        ffprobe = _find_ffprobe()
        if os.path.exists(tmp):
            os.remove(tmp)
        n_video = _rewrite(path, tmp, sei_nal)
        _verify_output(ffprobe, path, tmp, n_video)
        os.replace(tmp, path)
        done = True
        log(f"HEVC 3D TV frame-packing tag written ({layout}).")
        return True
    except Exception as e:
        log(f"WARNING: HEVC 3D TV frame-packing tag was NOT added (the original file is kept unchanged): {e}")
        return False
    finally:
        if not done and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _self_test():
    import tempfile
    import subprocess as sp
    from . import utils as _utils

    for layout, expected in _X264_BODY.items():
        got = _nal_body_for_test(layout)
        assert got == expected, f"{layout}: got {got.hex(' ')} expected {expected.hex(' ')}"
        nal = build_frame_packing_sei_nal(layout)
        assert nal[:2] == bytes([0x4E, 0x01])

    ffmpeg = _utils._get_ffmpeg_bin()
    assert ffmpeg, "ffmpeg not found"
    tdir = tempfile.mkdtemp(prefix="hevc_fp_selftest_")
    try:
        src = os.path.join(tdir, "synthetic.mp4")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=1",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
               "-c:v", "libx265", "-x265-params", "log-level=warning", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", src]
        sp.run(cmd, check=True, capture_output=True, timeout=300)
        orig_bytes = open(src, "rb").read()
        assert tag_hevc_frame_packing(src, "half_sbs"), "tagging the synthetic clip failed"
        assert open(src, "rb").read() != orig_bytes, "file was not rewritten"
        assert not os.path.exists(src[:-4] + ".fp_tmp.mp4"), "temp file left behind"
        ff = _find_ffprobe()
        info = _probe(ff, src)
        assert any(s.get("codec_type") == "video" and s.get("codec_name") == "hevc" for s in info["streams"])
        assert any(s.get("codec_type") == "audio" for s in info["streams"])
        print("hevc_frame_packing self-test: PASS (SEI bytes match libx264, tag written, verified)")
    finally:
        shutil.rmtree(tdir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print("usage: python -m iw3.hevc_frame_packing --self-test")
