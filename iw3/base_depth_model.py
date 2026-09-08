from abc import ABCMeta, abstractmethod
import contextlib
from nunif.utils.ui import HiddenPrints, TorchHubDir
from nunif.models.data_parallel import DeviceSwitchInference
from nunif.models.utils import compile_model
import os
import pickle
import cv2
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torchvision.transforms import functional as TF
from nunif.device import create_device
from .depth_scaler import EMAMinMaxScaler, TemporalStabilizer
from .hub_dir import HUB_MODEL_DIR


class _CompileContext():
    def __init__(self, base_model):
        self.base_model = base_model

    def __enter__(self):
        self.base_model.compile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.base_model.clear_compiled_model()
        return False


class BaseDepthModel(metaclass=ABCMeta):
    def __init__(self, model_type):
        self.device = None
        self.model = None
        self.model_backup = None  # for compile
        self.model_type = model_type
        self.scaler = self.create_depth_scaler()
        self.limit_resolution = False
        self.refine_enabled = False
        self.refine_strength = 1.0
        self.temporal_stabilizer = TemporalStabilizer()
        # Queues up a motion-adaptive activity summary every time the EMA scaler
        # resets (scene cut or end of export) instead of losing it the instant the
        # scaler moves on -- pop_motion_activity_log() is how a caller with real
        # video-position context (frame count, timestamp) turns this into a log
        # entry the user can actually read.
        self._motion_activity_log = []
        self._frame_position = 0
        self._segment_start_position = 0

    def create_depth_scaler(self):
        # This can be overridden
        return EMAMinMaxScaler(decay=0, buffer_size=1)

    def compile_context(self, enabled=True):
        if enabled:
            return _CompileContext(self)
        else:
            return contextlib.nullcontext()

    @classmethod
    @abstractmethod
    def get_name(cls):
        pass

    def loaded(self):
        return self.model is not None

    @classmethod
    @abstractmethod
    def supported(cls, model_type):
        pass

    @classmethod
    @abstractmethod
    def has_checkpoint_file(cls, model_type):
        pass

    @classmethod
    @abstractmethod
    def get_model_path(cls, model_type):
        pass

    @abstractmethod
    def is_metric(self):
        pass

    def is_image_supported(self):
        return True

    def is_video_supported(self):
        return True

    @staticmethod
    def force_update_hub(github, model):
        with TorchHubDir(HUB_MODEL_DIR):
            torch.hub.help(github, model, force_reload=True, trust_repo=True)

    @classmethod
    @abstractmethod
    def force_update(cls):
        pass

    @classmethod
    @abstractmethod
    def multi_gpu_supported(cls, name):
        pass

    @abstractmethod
    def load_model(self, model_type, resolution, device):
        pass

    def load(self, gpu=0, resolution=None, limit_resolution=False, **kwargs):
        self.device = create_device(gpu)
        self.limit_resolution = limit_resolution

        with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
            try:
                self.model = self.load_model(self.model_type, resolution=resolution, device=self.device, **kwargs)
            except (RuntimeError, pickle.PickleError) as e:
                if isinstance(e, RuntimeError):
                    do_handle = "PytorchStreamReader" in repr(e)
                else:
                    do_handle = True
                if do_handle:
                    try:
                        # delete corrupted file
                        os.unlink(self.get_model_path(self.model_type))
                    except:  # noqa
                        pass
                    raise RuntimeError(
                        f"File `{self.get_model_path(self.model_type)}` is corrupted. "
                        "This error may occur when the network is unstable or the disk is full. "
                        "Try again."
                    )
                else:
                    raise

        self.model = self.model.to(self.device).eval()

        if (isinstance(gpu, (list, tuple)) and len(gpu) > 1):
            if self.multi_gpu_supported(self.model_type):
                self.model = DeviceSwitchInference(self.model, device_ids=gpu)
            else:
                raise ValueError(f"{self.model_type} does not support Multi-GPU")

        return self

    def get_model(self):
        return self.model

    def move_to(self, device):
        """Moves the loaded model's weights (and its compiled backup, if
        torch.compile is active) plus the EMA scaler's GPU-resident state to
        `device`, without changing self.device -- the model's real target device,
        set once by load() and used to move incoming frames back to it during
        normal inference. A model wrapped in DeviceSwitchInference (multi-GPU:
        --gpu with more than one id) is left in place -- it already replicates
        itself across every configured device at construction time, so there is
        no single device to move it to and back from. Used by iw3's opt-in
        "free GPU memory while paused" feature (--pause-frees-vram / ADR-038)."""
        if self.model is not None and not isinstance(self.model, DeviceSwitchInference):
            self.model = self.model.to(device)
            if self.model_backup is not None:
                self.model_backup = self.model_backup.to(device)
        self.scaler.to(device)
        return self

    def compile(self):
        if self.model_backup is None and not isinstance(self.model, DeviceSwitchInference):
            self.model_backup = self.model
            self.model = compile_model(self.model)

    def clear_compiled_model(self):
        if self.model_backup is not None:
            self.model = self.model_backup
            self.model_backup = None

    @abstractmethod
    def infer(self, x, *kwargs):
        pass

    def enable_ema(self, decay, buffer_size=None, motion_adaptive=None, motion_spread=None):
        self.scaler.reset(decay=decay, buffer_size=buffer_size,
                           motion_adaptive=motion_adaptive, motion_spread=motion_spread)

    def get_ema_state(self):
        return self.scaler.decay, self.scaler.buffer_size

    def disable_ema(self):
        self.scaler.reset(decay=0, buffer_size=1, motion_adaptive=False)

    def reset_ema(self, decay=None, buffer_size=None, motion_adaptive=None, motion_spread=None):
        self.scaler.reset(decay=decay, buffer_size=buffer_size,
                           motion_adaptive=motion_adaptive, motion_spread=motion_spread)

    def reset_state(self):
        pass

    def reset(self):
        self.reset_ema()
        self.reset_state()

    def get_ema_buffer_size(self):
        return self.scaler.buffer_size

    def enable_refine(self, enabled=True, strength=1.0):
        self.refine_enabled = bool(enabled)
        self.refine_strength = float(strength) if strength is not None else 1.0

    def enable_temporal_stabilize(self, strength=0.7,
                                   max_shift_velocity=None, flat_region_boost=0.0, edge_protection=0.0):
        self.temporal_stabilizer.reset(enabled=True, strength=strength,
                                        max_shift_velocity=max_shift_velocity,
                                        flat_region_boost=flat_region_boost,
                                        edge_protection=edge_protection)

    def disable_temporal_stabilize(self):
        self.temporal_stabilizer.reset(enabled=False)

    def _refine_raw_depth_chw(self, depth, strength=1.0):
        """Edge-preserving spatial smoothing (bilateral filter) on the raw, per-frame
        depth map -- cleans up the speckle/noise a depth model leaves within a single
        frame, WITHOUT blurring across real edges the way a plain blur would. This is
        a different axis from EMA smoothing (which works across frames over time): this
        works within one frame only, and happens before EMA sees the frame, so a
        cleaner raw signal also means a steadier min/max range for EMA to track.

        `strength` (default 1.0) scales the bilateral filter's sigmaColor/sigmaSpace
        proportionally -- 1.0 reproduces the exact original fixed values
        (sigmaColor=0.08, sigmaSpace=5) byte-for-byte, so nothing changes for anyone
        who doesn't touch this. Higher pushes the same edge-preserving cleanup
        further (more smoothing reach before an edge is treated as "real" and left
        alone); lower pulls back toward doing less. `d` (the pixel neighborhood
        diameter) is deliberately left fixed at 5 regardless of strength -- scaling
        it too would change the filter's basic reach/cost, not just how aggressively
        it smooths, which isn't what this knob is for. strength<=0 skips the filter
        entirely (same as being disabled) rather than passing a degenerate sigma of
        0 to OpenCV."""
        if strength is None or strength <= 0:
            return depth
        arr = depth.squeeze(0).detach().float().cpu().numpy()
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-6:
            return depth
        norm = ((arr - lo) / (hi - lo)).astype(np.float32)
        smoothed = cv2.bilateralFilter(norm, d=5, sigmaColor=0.08 * strength, sigmaSpace=5 * strength)
        smoothed = smoothed * (hi - lo) + lo
        return torch.from_numpy(smoothed).unsqueeze(0).to(depth.device, dtype=depth.dtype)

    def minmax_normalize_chw(self, depth, return_minmax=False, rgb=None):
        # NOTE: an earlier version of this also ran CLAHE contrast enhancement on the
        # normalized output. Testing with real injected noise showed it backfired --
        # CLAHE amplifies whatever local variation it finds, noise included, and even
        # blended in lightly it still measurably increased noise versus bilateral
        # filtering alone. Dropped rather than shipped with a known noise-amplification
        # risk; only the proven bilateral smoothing step remains.
        self._frame_position += 1
        if self.refine_enabled:
            depth = self._refine_raw_depth_chw(depth, strength=self.refine_strength)
        if self.temporal_stabilizer.enabled:
            depth = self.temporal_stabilizer.stabilize(rgb, depth)
        return self.scaler(depth, return_minmax=return_minmax)

    def flush_minmax_normalize(self, return_minmax=False):
        # Grab whatever motion-adaptive did during this segment BEFORE calling
        # scaler.flush() -- when its own frame_queue is already empty, flush() takes
        # an internal path that calls reset() itself, which would silently wipe these
        # counters first if popped afterward instead.
        activity = self.scaler.pop_activity()
        if activity is not None:
            self._motion_activity_log.append({
                "start_frame": self._segment_start_position,
                "end_frame": self._frame_position,
                **activity,
            })
        self._segment_start_position = self._frame_position
        # Refinement runs on the raw depth before it ever reaches the scaler (see
        # minmax_normalize_chw), so leftover queued frames flushed here are already
        # refined -- nothing extra to do at flush time.
        result = self.scaler.flush(return_minmax=return_minmax)
        # This is the actual scene-boundary hook in this codebase (the EMA scaler resets
        # itself here too) -- clear the stabilizer's frame-to-frame memory so motion
        # tracking never bleeds across an unrelated cut, without touching whether it's
        # enabled or its configured strength.
        self.temporal_stabilizer.reset()
        return result

    def pop_motion_activity_log(self):
        """Returns and clears the list of per-segment motion-adaptive summaries
        accumulated since the last call -- each entry covers one scene (or the final
        tail of the export)."""
        log = self._motion_activity_log
        self._motion_activity_log = []
        return log

    def minmax_normalize(self, depth, reset_ema=None):
        assert depth.ndim == 4
        reset_ema = [False] * depth.shape[0] if reset_ema is None else reset_ema
        assert len(reset_ema) == depth.shape[0]
        normalized_depths = []
        for i in range(depth.shape[0]):
            normalized_depth = self.minmax_normalize_chw(depth[i])
            if normalized_depth is not None:
                normalized_depths.append(normalized_depth)
            if reset_ema[i]:
                normalized_depths += self.flush_minmax_normalize()
                self.reset_ema()
        return normalized_depths

    @staticmethod
    def save_normalized_depth(depth, file_path, png_info={}, min_depth_value=None, max_depth_value=None):
        if min_depth_value is not None:
            png_info.update(iw3_min_depth_value=float(min_depth_value))
        if max_depth_value is not None:
            png_info.update(iw3_max_depth_value=float(max_depth_value))

        depth = torch.clamp(depth, 0, 1)
        depth_int = (0xffff * depth).to(torch.uint16).squeeze(0).cpu().numpy()
        metadata = PngInfo()
        for k, v in png_info.items():
            metadata.add_text(k, str(v))

        im = Image.fromarray(depth_int)
        im.save(file_path, pnginfo=metadata)

    @staticmethod
    def load_depth(file_path):
        # TODO: test iw3_max_depth_value
        with Image.open(file_path) as im:
            if "iw3_min_depth_value" in im.text and "iw3_max_depth_value" in im.text:
                try:
                    min_depth_value = float(im.text["iw3_min_depth_value"])
                    max_depth_value = float(im.text["iw3_max_depth_value"])
                except (ValueError, TypeError):
                    min_depth_value = max_depth_value = None
            else:
                min_depth_value = max_depth_value = None

            depth = TF.pil_to_tensor(im)
            if depth.dtype != torch.float32:
                depth = torch.clamp(depth.to(torch.float32) / 0xffff, 0, 1)
            if depth.shape[0] != 1:
                depth = torch.mean(depth, dim=0, keepdim=True)

            if min_depth_value is not None and max_depth_value is not None:
                depth = depth * (max_depth_value - min_depth_value) + min_depth_value

            metadata = {}
            metadata.update(im.text)
            metadata.update({"filename": file_path})

            return depth, metadata


def _test():
    pass


if __name__ == "__main__":
    _test()
