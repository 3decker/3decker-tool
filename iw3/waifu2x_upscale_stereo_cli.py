"""python -m iw3.waifu2x_upscale_stereo_cli -- standalone per-eye, stereo-aware
waifu2x upscale + RGB temporal-smoothing post-processing entry point.

Invoked as a SEPARATE subprocess by iw3.utils._run_waifu2x_upscale_stereo() only
after the main iw3 conversion has fully finished, and only when the user
explicitly opted into BOTH "Upscale with waifu2x" AND a real (non-"auto")
Target Resolution on a packed two-eye stereo output (Half/Full SBS, Half/Full
TB, Cross-Eyed, VR180) -- see docs/ai/AI_DECISIONS.md ADR-040. Mirrors
iw3.rife_cli's separate-subprocess pattern exactly, for the same reason:
waifu2x's model must never share GPU memory with iw3's own depth/stereo
models still resident in the caller's process.

Pipeline (each step is its own full pass over the video):
  1. ffmpeg splits the packed source into two independent per-eye videos at
     the exact axis boundary (a plain crop -- see iw3.utils.split_stereo_frame
     for the pixel-exact geometry this mirrors, and its --self-test below for
     a pure-array round-trip proof of that geometry).
  2. waifu2x.cli upscales each eye video independently, via the SAME
     subprocess invocation _run_waifu2x_upscale already uses
     (iw3.utils._invoke_waifu2x_cli) -- never duplicated.
  3. RGBTemporalStabilizer (iw3.depth_scaler) smooths each UPSCALED eye's own
     frame sequence independently -- never blending information across eyes --
     then resizes to the exact per-eye target resolution
     (iw3.utils.compute_stereo_upscale_plan) in the same pass.
  4. ffmpeg stacks the two smoothed/resized eye videos back together at the
     axis boundary and re-muxes the original audio track.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import threading
import time
from os import path

import numpy as np
import torch
import torch.nn.functional as F

import nunif.utils.video as VU
from nunif.device import create_device
from .depth_scaler import RGBTemporalStabilizer


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--split-axis", type=str, required=True, choices=["sbs", "tb"])
    parser.add_argument("--target-packed-width", type=int, default=0,
                        help="final packed width: 3840 (4K) or 7680 (8K). Not used with --full-4k-layout.")
    parser.add_argument("--full-4k-layout", type=str, default=None, choices=["sbs", "tb"],
                        help=("Full 4K output (ADR-207): every eye becomes a real 3840-wide picture (3840x2160 for "
                              "16:9) packed side by side (7680x2160) or top/bottom (3840x4320), whatever the "
                              "source packing was."))
    parser.add_argument("--waifu2x-method", type=str, default="noise_scale2x")
    parser.add_argument("--waifu2x-noise-level", type=int, default=1)
    parser.add_argument("--waifu2x-style", type=str, default="photo")
    parser.add_argument("--temporal-stabilize-strength", type=float, default=0.5,
                        help="flicker smoothing of each upscaled eye, 0 = off (the fastest, skips the motion analysis "
                             "that dominates the run time)")
    parser.add_argument("--smoothing-quality", type=str, default="fast", choices=["fast", "accurate"],
                        help=("motion analysis used by the flicker smoothing: 'fast' (default, several times quicker) "
                              "or 'accurate' (the original settings)"))
    parser.add_argument("--crf", type=str, default="20")
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hdr", action="store_true",
                        help=("the video is HDR / Dolby Vision (ADR-206): use 10-bit HEVC with BT.2020/PQ colour "
                              "tags for every pass instead of 8-bit H.264. Dolby Vision data is NOT copied here; "
                              "the caller puts it back afterwards."))
    return parser


_HDR_COLOR_ARGS = ["-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
                   "-color_range", "tv"]


def _has_ffmpeg_encoder(ffmpeg_bin, name):
    """True when this ffmpeg lists `name` among its video encoders."""
    try:
        out = subprocess.run([ffmpeg_bin, "-hide_banner", "-encoders"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return False
    return any(line.startswith(" V") and line.split()[1:2] == [name] for line in out.splitlines())


def _pick_hdr_codec(ffmpeg_bin, gpu=0):
    """One HEVC encoder used by EVERY pass of an HDR job: hevc_nvenc (GPU) when both PyAV and this ffmpeg have it
    and a GPU is there (and the job is not forced onto the CPU with --gpu -1), else libx265."""
    from .utils import _hdr_upscale_codec
    if gpu is not None and gpu < 0:
        return "libx265"
    codec = _hdr_upscale_codec()
    if codec == "hevc_nvenc" and not _has_ffmpeg_encoder(ffmpeg_bin, "hevc_nvenc"):
        codec = "libx265"
    return codec


def _pick_sdr_codec(ffmpeg_bin, gpu=0):
    """ADR-208: "hevc_nvenc" for the passes of an SDR job when PyAV, this ffmpeg and a GPU all have it, else None
    (software H.264, the old behaviour). Keeps the CPU free while waifu2x works the GPU."""
    from .utils import _sdr_upscale_codec
    codec = _sdr_upscale_codec(gpu)
    if codec and not _has_ffmpeg_encoder(ffmpeg_bin, codec):
        return None
    return codec


def _video_encode_args(hdr_codec, crf, preset, gpu=0, sdr_codec=None):
    """ffmpeg output options for the split / join passes: 8-bit H.264 normally (unchanged), 8-bit hevc_nvenc when the
    GPU encoder is available (sdr_codec, ADR-208), 10-bit HEVC with HDR colour tags for an HDR video
    (hdr_codec = "hevc_nvenc" or "libx265"; None = SDR)."""
    if hdr_codec is None and sdr_codec == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", crf, "-preset", "p5", "-gpu", str(gpu),
                "-pix_fmt", "yuv420p"]
    if hdr_codec is None:
        return ["-c:v", "libx264", "-crf", crf, "-preset", preset]
    if hdr_codec == "hevc_nvenc":
        return (["-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", crf, "-preset", "p5", "-gpu", str(gpu),
                 "-pix_fmt", "p010le"] + _HDR_COLOR_ARGS)
    return ["-c:v", "libx265", "-crf", crf, "-preset", preset, "-pix_fmt", "yuv420p10le"] + _HDR_COLOR_ARGS


def _progress_line(label, done, total, start):
    """A tqdm-looking line ("label: 12/340 [00:01<00:26, 11.2it/s]") so the GUI and iw3's own progress reader,
    which both look for "N/M [", treat these steps like every other one."""
    elapsed = max(1e-6, time.time() - start)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0
    fmt = lambda t: f"{int(t) // 60:02d}:{int(t) % 60:02d}"   # noqa: E731
    return f"{label}: {done}/{total} [{fmt(elapsed)}<{fmt(eta)}, {rate:.1f}it/s]"


def _run_ffmpeg_with_progress(cmd, total_frames, label):
    """subprocess.run(cmd, check=True, capture_output=True) that also prints live progress lines (frames done of
    total) to stderr, read from ffmpeg's own -progress output. Before this the split and join passes were silent."""
    full = [cmd[0], "-progress", "pipe:1", "-nostats", "-loglevel", "error"] + list(cmd[1:])
    proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err = []
    drain = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
    drain.start()
    start, total, last = time.time(), max(1, int(total_frames or 1)), -1
    try:
        for line in proc.stdout:
            if line.startswith("frame="):
                try:
                    done = min(int(line.split("=", 1)[1].strip()), total)
                except ValueError:
                    continue
                if done != last:
                    last = done
                    print(_progress_line(label, done, total, start), file=sys.stderr, flush=True)
    finally:
        proc.wait()
        drain.join(timeout=5)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, full, stderr="".join(err).encode())


def _split_video(input_path, axis, left_path, right_path, ffmpeg_bin, crf, preset,
                 hdr_codec=None, gpu=0, total_frames=None, sdr_codec=None):
    """Crops the packed source into two eye videos in ONE ffmpeg pass (both crop
    filters read from the same [0:v] input link, so the source is only decoded
    once). The crop geometry is the video-codec equivalent of
    iw3.utils.split_stereo_frame's array slicing -- x=0..w/2 / x=w/2..w for
    "sbs", y=0..h/2 / y=h/2..h for "tb"."""
    if axis == "sbs":
        left_filter = "crop=w=iw/2:h=ih:x=0:y=0"
        right_filter = "crop=w=iw-iw/2:h=ih:x=iw/2:y=0"
    else:
        left_filter = "crop=w=iw:h=ih/2:x=0:y=0"
        right_filter = "crop=w=iw:h=ih-ih/2:x=0:y=ih/2"
    filter_complex = f"[0:v]{left_filter}[l];[0:v]{right_filter}[r]"
    enc = _video_encode_args(hdr_codec, crf, preset, gpu, sdr_codec)
    cmd = [ffmpeg_bin, "-y", "-i", input_path,
           "-filter_complex", filter_complex,
           "-map", "[l]"] + enc + ["-an", left_path,
           "-map", "[r]"] + enc + ["-an", right_path]
    _run_ffmpeg_with_progress(cmd, total_frames, "[1/4] splitting")


class _SharedProgress:
    """One progress line for several passes that run at the same time (the two eyes): their frame counts are added
    together, so the window sees a single steadily rising "N/M [" line instead of two interleaved ones."""

    def __init__(self, label, total):
        self.label, self.total, self.done, self.start, self.last = label, max(1, int(total)), 0, time.time(), 0.0
        self.lock = threading.Lock()

    def make_bar(self, desc=None, total=None, ncols=None):
        return _SharedBar(self)

    def add(self, n):
        with self.lock:
            self.done = min(self.total, self.done + n)
            now = time.time()
            if self.done >= self.total or now - self.last >= 0.5:
                self.last = now
                print(_progress_line(self.label, self.done, self.total, self.start), file=sys.stderr, flush=True)


class _SharedBar:
    """The tiny tqdm-like object VU.process_video needs (update / close)."""

    def __init__(self, shared):
        self.shared = shared

    def update(self, n=1):
        self.shared.add(n)

    def close(self):
        pass


def _smooth_and_resize_eye(input_path, output_path, target_w, target_h, strength, device, crf, preset,
                           hdr_codec=None, gpu=0, sdr_codec=None, quality="fast", tqdm_fn=None):
    """Single-pass decode -> resize to the exact per-eye target -> RGBTemporalStabilizer.stabilize() -> encode, for
    ONE eye's already-upscaled video. Follows iw3.rife_cli's exact VU.process_video template.

    ADR-210: the resize now comes BEFORE the flicker smoothing. The smoothing is a CPU optical-flow analysis whose
    cost grows with the pixel count; running it on the full enlarged frame (3840x3216) took ~2 s a frame and made
    it the slowest part of the whole upscale. At the final size it has 2-4x fewer pixels, and flicker only has to
    be calm at the size that is delivered. strength <= 0 skips the smoothing completely."""
    fast = quality != "accurate"
    stabilizer = RGBTemporalStabilizer(enabled=strength > 0, strength=strength, fast=fast,
                                       flow_downscale=2 if fast else 1)

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            return None
        x = VU.to_tensor(frame, device=device)
        h, w = x.shape[-2:]
        if (w, h) != (target_w, target_h):
            x = F.interpolate(x.unsqueeze(0), size=(target_h, target_w),
                               mode="bicubic", antialias=True).squeeze(0).clamp(0, 1)
        return stabilizer.stabilize(x)

    def config_callback(sw_format):
        # ADR-212: the eye video's OWN frame rate. With fps=None the writer fell back to 24 fps, so every stereo
        # upscale of a source that was not exactly 24 fps came out at the wrong speed (a 47.95 fps RIFE movie played at
        # half speed). sw_format is the input's metadata (same object waifu2x.cli reads get_fps() from).
        fps = sw_format.get_fps()
        if hdr_codec is None and sdr_codec == "hevc_nvenc":
            return VU.VideoOutputConfig(
                fps=fps,
                output_fps=None,
                pix_fmt="yuv420p",
                video_codec="hevc_nvenc",
                options={"rc": "constqp", "qp": str(crf), "gpu": str(gpu)},
            )
        if hdr_codec is None:
            return VU.VideoOutputConfig(
                fps=fps,
                output_fps=None,
                options={"preset": preset, "crf": crf},
            )
        if hdr_codec == "hevc_nvenc":
            options = {"rc": "constqp", "qp": str(crf), "gpu": str(gpu)}
        else:
            options = {"preset": preset, "crf": str(crf)}
        return VU.VideoOutputConfig(
            fps=fps,
            output_fps=None,
            pix_fmt="yuv420p10le",
            video_codec=hdr_codec,
            colorspace="bt2020-pq-tv",
            options=options,
        )

    VU.process_video(
        input_path, output_path, frame_callback,
        config_callback=config_callback,
        title="Stereo Upscale Smooth",
        device=device,
        tqdm_fn=tqdm_fn,
    )


def _join_video(left_path, right_path, audio_source_path, axis, output_path, ffmpeg_bin, crf, preset,
                hdr_codec=None, gpu=0, total_frames=None, sdr_codec=None):
    """Inverse of _split_video -- hstack ("sbs") / vstack ("tb") the two final
    eye videos back together, the video-codec equivalent of
    iw3.utils.join_stereo_frame, and re-muxes the ORIGINAL packed source's own
    audio track (never re-encoded)."""
    stack_filter = "hstack=inputs=2" if axis == "sbs" else "vstack=inputs=2"
    if hdr_codec is not None:
        # the stack filter drops the colour tags of its inputs; without this the final .mkv comes out with an
        # "unknown" transfer / primaries and a TV would not treat it as HDR
        stack_filter += ",setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv"
    cmd = [ffmpeg_bin, "-y",
           "-i", left_path, "-i", right_path, "-i", audio_source_path,
           "-filter_complex", f"[0:v][1:v]{stack_filter}[v]",
           "-map", "[v]", "-map", "2:a?"] + _video_encode_args(hdr_codec, crf, preset, gpu, sdr_codec) + [
           "-c:a", "aac",
           output_path]
    _run_ffmpeg_with_progress(cmd, total_frames, "[4/4] joining")


def run(input_path, output_path, split_axis, target_packed_width,
        waifu2x_method="noise_scale2x", waifu2x_noise_level=1, waifu2x_style="photo",
        temporal_stabilize_strength=0.5, crf="20", preset="medium", gpu=0,
        hdr=False, full_4k_layout=None, smoothing_quality="fast"):
    from .utils import (
        _estimate_video_frames, _get_ffmpeg_bin, _invoke_waifu2x_cli, compute_full_4k_plan,
        compute_stereo_upscale_plan, resolve_waifu2x_method_for_tier,
    )

    device = create_device(gpu)
    ffmpeg_bin = _get_ffmpeg_bin()
    nunif_dir = path.dirname(path.dirname(path.abspath(__file__)))
    hdr_codec = _pick_hdr_codec(ffmpeg_bin, gpu) if hdr else None
    sdr_codec = None if hdr else _pick_sdr_codec(ffmpeg_bin, gpu)

    sw_format = VU.VideoMetadata.from_file(input_path)
    if full_4k_layout:
        plan = compute_full_4k_plan(sw_format.width, sw_format.height, split_axis, full_4k_layout)
    else:
        plan = compute_stereo_upscale_plan(sw_format.width, sw_format.height, split_axis, target_packed_width)
    out_axis = plan.get("out_axis", split_axis)     # where the two eyes are packed in the OUTPUT
    method = resolve_waifu2x_method_for_tier(waifu2x_method, plan["scale_tier"])
    total_frames = _estimate_video_frames(input_path)[0]
    extra_args = None
    if hdr_codec:
        extra_args = ["--video-codec", hdr_codec, "--pix-fmt", "yuv420p10le", "--colorspace", "bt2020-pq-tv",
                      "--crf", str(crf)]
    elif sdr_codec:
        extra_args = ["--video-codec", sdr_codec, "--crf", str(crf)]

    out_dir = path.dirname(path.abspath(output_path)) or "."
    base = path.splitext(path.basename(output_path))[0]
    left_src = path.join(out_dir, f"_{base}_left_src.mp4")
    right_src = path.join(out_dir, f"_{base}_right_src.mp4")
    left_up = path.join(out_dir, f"_{base}_left_up.mp4")
    right_up = path.join(out_dir, f"_{base}_right_up.mp4")
    left_final = path.join(out_dir, f"_{base}_left_final.mp4")
    right_final = path.join(out_dir, f"_{base}_right_final.mp4")
    tmp_files = [left_src, right_src, left_up, right_up, left_final, right_final]

    try:
        print(f"[iw3] [1/4] splitting {split_axis} into two eye videos...", file=sys.stderr, flush=True)
        _split_video(input_path, split_axis, left_src, right_src, ffmpeg_bin, crf, preset,
                     hdr_codec=hdr_codec, gpu=gpu, total_frames=total_frames, sdr_codec=sdr_codec)

        print(f"[iw3] [2/4] upscaling each eye with waifu2x ({method})...", file=sys.stderr, flush=True)
        for src, up in ((left_src, left_up), (right_src, right_up)):
            if not _invoke_waifu2x_cli(src, up, method, waifu2x_noise_level, waifu2x_style, nunif_dir,
                                        log_prefix="[iw3] stereo-upscale", forward_progress=True,
                                        extra_args=extra_args):
                raise RuntimeError(f"waifu2x upscale failed for {src}")

        print("[iw3] [3/4] temporal-smoothing + resizing each eye "
              f"to {plan['target_eye_w']}x{plan['target_eye_h']}...", file=sys.stderr, flush=True)
        # ADR-210: both eyes at once. Each pass is limited by one CPU core (decode / optical flow), so two passes
        # take about the time of one.
        shared = _SharedProgress("[3/4] smoothing", (total_frames or 0) * 2) if total_frames else None
        tqdm_fn = shared.make_bar if shared else None
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(_smooth_and_resize_eye, up, final, plan["target_eye_w"], plan["target_eye_h"],
                                temporal_stabilize_strength, device, crf, preset,
                                hdr_codec=hdr_codec, gpu=gpu, sdr_codec=sdr_codec,
                                quality=smoothing_quality, tqdm_fn=tqdm_fn)
                    for up, final in ((left_up, left_final), (right_up, right_final))]
            for job in jobs:
                job.result()

        print(f"[iw3] [4/4] recombining ({out_axis}) to {plan['target_packed_w']}x{plan['target_packed_h']}...",
              file=sys.stderr, flush=True)
        _join_video(left_final, right_final, input_path, out_axis, output_path, ffmpeg_bin, crf, preset,
                    hdr_codec=hdr_codec, gpu=gpu, total_frames=total_frames, sdr_codec=sdr_codec)
    finally:
        for f in tmp_files:
            if path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if not args.full_4k_layout and args.target_packed_width <= 0:
        parser.error("give either --target-packed-width or --full-4k-layout")
    run(
        args.input, args.output, args.split_axis, args.target_packed_width,
        waifu2x_method=args.waifu2x_method,
        waifu2x_noise_level=args.waifu2x_noise_level,
        waifu2x_style=args.waifu2x_style,
        temporal_stabilize_strength=args.temporal_stabilize_strength,
        crf=args.crf, preset=args.preset, gpu=args.gpu,
        hdr=args.hdr, full_4k_layout=args.full_4k_layout, smoothing_quality=args.smoothing_quality,
    )

def _self_test():
    """Synthetic, GPU-free regression coverage (CS-TEST-001) -- run via
    `python -m iw3.waifu2x_upscale_stereo_cli --self-test`. Covers:
      1. RGBTemporalStabilizer actually smooths a manufactured noisy frame
         sequence more than doing nothing.
      2. split_stereo_frame/join_stereo_frame round-trip a synthetic packed
         SBS and TB frame back to pixel-identical output with no upscaling.
      3. compute_stereo_upscale_plan's scale-factor math for the worked
         Half-SBS 1920x1080 -> 8K example.
      4. resolve_waifu2x_method_for_tier rewrites the 2x/4x suffix correctly.
    No GPU or real video file needed -- everything here is synthetic
    tensors/arrays."""
    from .utils import (
        compute_stereo_upscale_plan, join_stereo_frame, resolve_waifu2x_method_for_tier,
        split_stereo_frame,
    )

    # --- 1. RGB temporal smoothing actually reduces frame-to-frame noise ---
    rng = np.random.default_rng(0)
    h, w = 64, 96
    # A stable base scene (a soft gradient, so there's real structure to preserve)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (xx / w * 0.6 + yy / h * 0.3).astype(np.float32)
    base = np.stack([base, base, base], axis=0)  # (3, H, W), no motion between frames

    num_frames = 12
    noisy_frames = []
    for _ in range(num_frames):
        noise = rng.normal(0, 0.08, size=base.shape).astype(np.float32)
        noisy_frames.append(torch.from_numpy(np.clip(base + noise, 0, 1)))

    stabilizer = RGBTemporalStabilizer(enabled=True, strength=0.8)
    stabilized_frames = [stabilizer.stabilize(f) for f in noisy_frames]

    # Compare frame-to-frame pixel deltas (a direct measure of flicker) with vs
    # without stabilization -- stabilized deltas must be meaningfully smaller.
    def mean_frame_delta(frames):
        deltas = []
        for i in range(1, len(frames)):
            deltas.append((frames[i] - frames[i - 1]).abs().mean().item())
        return sum(deltas) / len(deltas)

    raw_delta = mean_frame_delta(noisy_frames)
    stabilized_delta = mean_frame_delta(stabilized_frames)
    assert stabilized_delta < raw_delta * 0.7, (
        f"RGBTemporalStabilizer did not meaningfully reduce frame-to-frame flicker: "
        f"raw={raw_delta:.5f} stabilized={stabilized_delta:.5f}"
    )
    print(f"[self-test] RGB temporal smoothing: raw_delta={raw_delta:.5f} "
          f"stabilized_delta={stabilized_delta:.5f} (PASS)")

    # A disabled stabilizer must be a pure no-op (off-by-default guarantee).
    off_stabilizer = RGBTemporalStabilizer(enabled=False)
    passthrough = off_stabilizer.stabilize(noisy_frames[0])
    assert torch.equal(passthrough, noisy_frames[0]), "disabled RGBTemporalStabilizer must be a no-op"
    print("[self-test] RGBTemporalStabilizer disabled -> exact no-op (PASS)")

    # --- 2. split/rejoin round-trip, synthetic packed frames, both axes ---
    for axis, packed_shape in (("sbs", (240, 400, 3)), ("tb", (400, 240, 3))):
        packed = rng.integers(0, 256, size=packed_shape, dtype=np.uint8)
        a, b = split_stereo_frame(packed, axis)
        rejoined = join_stereo_frame(a, b, axis)
        assert rejoined.shape == packed.shape, f"{axis}: shape mismatch after round-trip"
        assert np.array_equal(rejoined, packed), f"{axis}: round-trip is not pixel-identical"
    print("[self-test] split_stereo_frame/join_stereo_frame round-trip (sbs, tb): pixel-identical (PASS)")

    # Odd packed dimension -- still a lossless round-trip (remainder pixel goes
    # to the second half).
    odd_packed = rng.integers(0, 256, size=(101, 151, 3), dtype=np.uint8)
    a, b = split_stereo_frame(odd_packed, "sbs")
    assert np.array_equal(join_stereo_frame(a, b, "sbs"), odd_packed), "odd-width sbs round-trip failed"
    print("[self-test] split_stereo_frame/join_stereo_frame round-trip (odd width): pixel-identical (PASS)")

    # --- 3. worked example: Half-SBS 1920x1080 source -> 8K packed output ---
    plan = compute_stereo_upscale_plan(1920, 1080, "sbs", 7680)
    assert plan["target_packed_w"] == 7680, plan
    assert plan["target_packed_h"] == 4320, plan
    assert plan["src_eye_w"] == 960 and plan["src_eye_h"] == 1080, plan
    assert plan["target_eye_w"] == 3840 and plan["target_eye_h"] == 4320, plan
    assert abs(plan["needed_scale"] - 4.0) < 1e-9, plan
    assert plan["scale_tier"] == 4, plan
    print(f"[self-test] compute_stereo_upscale_plan(1920x1080 Half-SBS -> 8K): {plan} (PASS)")

    # A non-16:9-friendly / smaller source needing only a 2x tier.
    plan_2x = compute_stereo_upscale_plan(3840, 2160, "sbs", 7680)
    assert plan_2x["scale_tier"] == 2, plan_2x
    print(f"[self-test] compute_stereo_upscale_plan(3840x2160 -> 8K, expect 2x tier): "
          f"scale_tier={plan_2x['scale_tier']} (PASS)")

    # A non-16:9 source (Full-SBS, unsqueezed 32:9 packed frame): target_packed_h
    # is derived from the SOURCE's own aspect ratio (not hardcoded 16:9), so the
    # needed scale correctly comes out uniform (2x) instead of a wrong, too-large
    # tier a naive fixed-16:9-target assumption would have picked.
    plan_full_sbs = compute_stereo_upscale_plan(3840, 1080, "sbs", 7680)
    assert plan_full_sbs["target_packed_h"] == 2160, plan_full_sbs
    assert abs(plan_full_sbs["needed_scale"] - 2.0) < 1e-9, plan_full_sbs
    assert plan_full_sbs["scale_tier"] == 2, plan_full_sbs
    print(f"[self-test] compute_stereo_upscale_plan(Full-SBS 3840x1080, 32:9 -> 8K, "
          f"aspect-preserving target): scale_tier={plan_full_sbs['scale_tier']} (PASS)")

    # --- 4. method-tier rewriting ---
    assert resolve_waifu2x_method_for_tier("noise_scale2x", 4) == "noise_scale4x"
    assert resolve_waifu2x_method_for_tier("scale4x", 2) == "scale2x"
    assert resolve_waifu2x_method_for_tier("realesrgan_x4", 4) == "realesrgan_x4"
    assert resolve_waifu2x_method_for_tier("onnx:custom", 4) == "onnx:custom"
    print("[self-test] resolve_waifu2x_method_for_tier: PASS")

    print("[self-test] ALL PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        sys.exit(main())
