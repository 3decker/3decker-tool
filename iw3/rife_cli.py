"""python -m iw3.rife_cli -- standalone RIFE frame-interpolation entry point.

Invoked as a SEPARATE subprocess by iw3.utils._run_rife_interpolation() only
after the main iw3 conversion has fully finished, exactly mirroring how
waifu2x.cli is invoked by _run_waifu2x_upscale() -- see
docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001 and docs/ai/AI_DECISIONS.md
ADR-014/ADR-029. Running as its own process means RIFE's model never shares GPU
memory with iw3's depth/stereo models still resident in the caller's process.

Interpolates the FINAL PACKED stereo frame (both eyes already combined into one
frame, e.g. Half-SBS/Full-SBS) as a single image, not each eye separately --
simpler, and avoids any risk of left/right eye desync since both eyes always
move through interpolation together, in lockstep. NOTE: RIFE will see a seam
between the two packed eyes it was never trained on -- an accepted, known
tradeoff, not a bug (see ADR-029)."""
import argparse
import sys

import torch

import nunif.utils.video as VU
from nunif.device import create_device
from .rife_model import DEFAULT_RIFE_MODEL, RIFE_TIERS, interpolate_frame, load_rife_model


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--rife-model", type=str, default=DEFAULT_RIFE_MODEL,
                        choices=list(RIFE_TIERS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def _make_frame_callback(model, device):
    # Buffers exactly the previous decoded frame; when the next frame arrives,
    # emits [previous, interpolated-middle] so playback order stays correct.
    # The very last frame has no successor to interpolate against and is
    # emitted verbatim on flush (frame=None).
    state = {"pending": None}

    @torch.inference_mode()
    def frame_callback(frame):
        if frame is None:
            pending = state.pop("pending", None)
            return pending.squeeze(0) if pending is not None else None
        x = VU.to_tensor(frame, device=device).unsqueeze(0)
        pending = state.get("pending")
        if pending is None:
            state["pending"] = x
            return None
        middle = interpolate_frame(model, pending, x, scale=1.0)
        state["pending"] = x
        return [pending.squeeze(0), middle.squeeze(0)]

    return frame_callback


def run(input_path, output_path, rife_model=DEFAULT_RIFE_MODEL, gpu=0):
    device = create_device(gpu)
    model = load_rife_model(rife_model, device)
    frame_callback = _make_frame_callback(model, device)

    def config_callback(sw_format):
        orig_fps = sw_format.get_fps()
        return VU.VideoOutputConfig(
            fps=None,  # no input resampling -- every real decoded frame is kept
            output_fps=float(orig_fps) * 2,
            options={"preset": "medium", "crf": "16"},
        )

    VU.process_video(
        input_path,
        output_path,
        frame_callback,
        config_callback=config_callback,
        title="RIFE",
        device=device,
    )


def main(argv=None):
    args = create_parser().parse_args(argv)
    run(args.input, args.output, rife_model=args.rife_model, gpu=args.gpu)


if __name__ == "__main__":
    sys.exit(main())
