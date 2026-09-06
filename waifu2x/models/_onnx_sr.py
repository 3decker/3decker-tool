"""Wraps an onnxruntime.InferenceSession as a torch-callable model, so a custom
external ONNX super-resolution model (e.g. a SPAN-style export) can reuse this
project's existing tiled_render infrastructure directly instead of needing a
separate, hand-written tiling loop. See waifu2x/external_sr.py for discovery
(scanning a folder for .onnx files) and docs/ai/AI_DECISIONS.md for why this
exists. Requires the optional onnxruntime(-gpu) package -- imported lazily here,
not at module load time, so its absence doesn't break anything for users who
never use this feature."""
import numpy as np
import torch
from nunif.models import I2IBaseModel, register_model


@register_model
class ONNXSRModel(I2IBaseModel):
    name = "waifu2x.onnx_sr"

    def __init__(self, onnx_path, providers=None, probe_size=32):
        # Scale is determined empirically by running one real probe inference
        # BEFORE calling super().__init__ (which requires scale as a constructor
        # arg) -- an ONNX SR model's declared input/output shapes are typically
        # dynamic/symbolic for height and width, so there's no static scale factor
        # to just read from the file's own metadata; actually running it on a
        # known-size input and measuring the real output size is the only way to
        # be sure, not an assumption.
        import onnxruntime as ort
        if providers is None:
            available = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
        session = ort.InferenceSession(onnx_path, providers=providers)
        input_meta = session.get_inputs()[0]
        output_name = session.get_outputs()[0].name
        probe = np.random.rand(1, 3, probe_size, probe_size).astype(np.float32)
        try:
            result = session.run([output_name], {input_meta.name: probe})[0]
        except Exception as e:
            raise ValueError(
                f"ONNX model {onnx_path} failed a test inference with a "
                f"1x3x{probe_size}x{probe_size} probe input -- likely not a plain "
                f"image-in/image-out super-resolution model (unexpected input shape/"
                f"count, or a model that needs extra inputs this loader doesn't "
                f"provide). Underlying error: {e}") from e
        if result.ndim != 4 or result.shape[1] != 3:
            raise ValueError(
                f"ONNX model {onnx_path}'s output shape {result.shape} doesn't look "
                f"like a standard NCHW RGB image (expected 4 dims, 3 channels) -- "
                f"not usable as a super-resolution model by this loader.")
        out_h = result.shape[-2]
        if out_h <= 0 or out_h % probe_size != 0:
            raise ValueError(
                f"ONNX model {onnx_path}: output height {out_h} is not a clean "
                f"whole-number multiple of the {probe_size}px probe input height -- "
                f"cannot determine a reliable integer scale factor for tiled "
                f"rendering.")
        scale = out_h // probe_size

        super().__init__(locals(), scale=scale, offset=0, in_channels=3,
                          default_tile_size=256, default_batch_size=1)
        self.session = session
        self.input_name = input_meta.name
        self.output_name = output_name
        self.onnx_path = onnx_path
        # This model has no real torch parameters (all computation happens inside
        # the wrapped ONNX session, not as torch ops) -- but the base Model class's
        # get_device() (used by tiled_render/SeamBlending) does `next(self.
        # parameters())`, which raises StopIteration with zero parameters. A
        # harmless dummy parameter, moved correctly by the normal .to(device) call
        # ExternalSRModel already makes, fixes that without affecting inference
        # (the ONNX session ignores it entirely).
        self._device_marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        device = x.device
        x_np = x.detach().to(torch.float32).cpu().numpy()
        y_np = self.session.run([self.output_name], {self.input_name: x_np})[0]
        y = torch.from_numpy(y_np).to(device=device, dtype=x.dtype)
        if self.training:
            return y
        return torch.clamp(y, 0., 1.)
