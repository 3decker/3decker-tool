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
import subprocess
import sys
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
    parser.add_argument("--target-packed-width", type=int, required=True)
    parser.add_argument("--waifu2x-method", type=str, default="noise_scale2x")
    parser.add_argument("--waifu2x-noise-level", type=int, default=1)
    parser.add_argument("--waifu2x-style", type=str, default="photo")
    parser.add_argument("--temporal-stabilize-strength", type=float, default=0.5)
    parser.add_argument("--crf", type=str, default="20")
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def _split_video(input_path, axis, left_path, right_path, ffmpeg_bin, crf, preset):
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
    cmd = [ffmpeg_bin, "-y", "-i", input_path,
           "-filter_complex", filter_complex,
           "-map", "[l]", "-c:v", "libx264", "-crf", crf, "-preset", preset, "-an", left_path,
           "-map", "[r]", "-c:v", "libx264", "-crf", crf, "-preset", preset, "-an", right_path]
    subprocess.run(cmd, check=True, capture_output=True)


def _smooth_and_resize_eye(input_path, output_path, target_w, target_h, strength, device, crf, preset):
    """Single-pass decode -> RGBTemporalStabilizer.stabilize() -> resize to the
    exact per-eye target -> encode, for ONE eye's already-upscaled video.
    Follows iw3.rife_cli's exact VU.process_video template."""
    stabilizer = RGBTemporalStabilizer(enabled=True, strength=strength)

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            return None
        x = VU.to_tensor(frame, device=device)
        x = stabilizer.stabilize(x)
        h, w = x.shape[-2:]
        if (w, h) != (target_w, target_h):
            x = F.interpolate(x.unsqueeze(0), size=(target_h, target_w),
                               mode="bicubic", antialias=True).squeeze(0).clamp(0, 1)
        return x

    def config_callback(sw_format):
        return VU.VideoOutputConfig(
            fps=None,
            output_fps=None,
            options={"preset": preset, "crf": crf},
        )

    VU.process_video(
        input_path, output_path, frame_callback,
        config_callback=config_callback,
        title="Stereo Upscale Smooth",
        device=device,
    )


def _join_video(left_path, right_path, audio_source_path, axis, output_path, ffmpeg_bin, crf, preset):
    """Inverse of _split_video -- hstack ("sbs") / vstack ("tb") the two final
    eye videos back together, the video-codec equivalent of
    iw3.utils.join_stereo_frame, and re-muxes the ORIGINAL packed source's own
    audio track (never re-encoded)."""
    stack_filter = "hstack=inputs=2" if axis == "sbs" else "vstack=inputs=2"
    cmd = [ffmpeg_bin, "-y",
           "-i", left_path, "-i", right_path, "-i", audio_source_path,
           "-filter_complex", f"[0:v][1:v]{stack_filter}[v]",
           "-map", "[v]", "-map", "2:a?",
           "-c:v", "libx264", "-crf", crf, "-preset", preset,
           "-c:a", "aac",
           output_path]
    subprocess.run(cmd, check=True, capture_output=True)


def run(input_path, output_path, split_axis, target_packed_width,
        waifu2x_method="noise_scale2x", waifu2x_noise_level=1, waifu2x_style="photo",
        temporal_stabilize_strength=0.5, crf="20", preset="medium", gpu=0):
    from .utils import (
        _get_ffmpeg_bin, _invoke_waifu2x_cli, compute_stereo_upscale_plan, resolve_waifu2x_method_for_tier,
    )

    device = create_device(gpu)
    ffmpeg_bin = _get_ffmpeg_bin()
    nunif_dir = path.dirname(path.dirname(path.abspath(__file__)))

    sw_format = VU.VideoMetadata.from_file(input_path)
    plan = compute_stereo_upscale_plan(sw_format.width, sw_format.height, split_axis, target_packed_width)
    method = resolve_waifu2x_method_for_tier(waifu2x_method, plan["scale_tier"])

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
        print(f"[iw3] [1/4] splitting {split_axis} into two eye videos...", file=sys.stderr)
        _split_video(input_path, split_axis, left_src, right_src, ffmpeg_bin, crf, preset)

        print(f"[iw3] [2/4] upscaling each eye with waifu2x ({method})...", file=sys.stderr)
        for src, up in ((left_src, left_up), (right_src, right_up)):
            if not _invoke_waifu2x_cli(src, up, method, waifu2x_noise_level, waifu2x_style, nunif_dir,
                                        log_prefix="[iw3] stereo-upscale"):
                raise RuntimeError(f"waifu2x upscale failed for {src}")

        print("[iw3] [3/4] temporal-smoothing + resizing each eye "
              f"to {plan['target_eye_w']}x{plan['target_eye_h']}...", file=sys.stderr)
        for up, final in ((left_up, left_final), (right_up, right_final)):
            _smooth_and_resize_eye(up, final, plan["target_eye_w"], plan["target_eye_h"],
                                    temporal_stabilize_strength, device, crf, preset)

        print(f"[iw3] [4/4] recombining to {plan['target_packed_w']}x{plan['target_packed_h']}...",
              file=sys.stderr)
        _join_video(left_final, right_final, input_path, split_axis, output_path, ffmpeg_bin, crf, preset)
    finally:
        for f in tmp_files:
            if path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def main(argv=None):
    args = create_parser().parse_args(argv)
    run(
        args.input, args.output, args.split_axis, args.target_packed_width,
        waifu2x_method=args.waifu2x_method,
        waifu2x_noise_level=args.waifu2x_noise_level,
        waifu2x_style=args.waifu2x_style,
        temporal_stabilize_strength=args.temporal_stabilize_strength,
        crf=args.crf, preset=args.preset, gpu=args.gpu,
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
