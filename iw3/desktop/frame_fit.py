"""Keeps iw3-desktop's output frame size FIXED when the captured picture changes size.

The streaming page draws the stream on a canvas sized once (WIDTH x HEIGHT in views/index.html.tpl), the local
viewer and any compiled model are sized to the first frame, and the capture buffers are allocated at start-up.
So when the captured window is resized, or the monitor resolution changes, the captured picture must be mapped
INTO that fixed frame. Before this module the backends behaved differently: some crashed (screen-size mismatch
in the shared-memory backend, buffer mismatch in the PIL backend), some copied the top-left corner of a window
onto a black canvas (a bigger window got cropped), and some stretched the picture (aspect distortion).

`fit_*` scales the source to fit INSIDE the target with its aspect ratio kept, centres it, and pads black.
Small size differences that are only window effects (borders/shadows, a pixel or two either way) are NOT
treated as a change: `is_window_effect_size` lets callers keep the old cheap top-left copy for those.
"""
import numpy as np
import torch
import torch.nn.functional as F


def fit_size(src_h, src_w, dst_h, dst_w):
    """(h, w) of the source scaled to fit inside dst_h x dst_w, aspect kept, both >= 2."""
    scale = min(dst_h / src_h, dst_w / src_w)
    new_h = min(dst_h, max(2, int(round(src_h * scale))))
    new_w = min(dst_w, max(2, int(round(src_w * scale))))
    return new_h, new_w


def same_aspect(src_h, src_w, dst_h, dst_w, tolerance=0.005):
    return abs((src_w / src_h) - (dst_w / dst_h)) <= tolerance * (dst_w / dst_h)


def is_window_effect_size(src_h, src_w, dst_h, dst_w):
    """True when the source differs from the target only by a few pixels (window borders/shadows), which is
    the normal case for window capture and must keep the exact top-left copy behaviour."""
    return abs(src_h - dst_h) <= max(4, int(0.02 * dst_h)) and abs(src_w - dst_w) <= max(4, int(0.02 * dst_w))


def fit_frame_chw(frame, dst_h, dst_w):
    """frame: CHW float tensor (0..1) -> CHW float tensor exactly dst_h x dst_w, aspect kept, black padded."""
    c, h, w = frame.shape
    if (h, w) == (dst_h, dst_w):
        return frame
    if same_aspect(h, w, dst_h, dst_w):
        return F.interpolate(frame.unsqueeze(0), size=(dst_h, dst_w), mode="bilinear",
                             align_corners=False, antialias=True).squeeze(0)
    new_h, new_w = fit_size(h, w, dst_h, dst_w)
    resized = F.interpolate(frame.unsqueeze(0), size=(new_h, new_w), mode="bilinear",
                            align_corners=False, antialias=True).squeeze(0)
    canvas = torch.zeros((c, dst_h, dst_w), dtype=frame.dtype, device=frame.device)
    top = (dst_h - new_h) // 2
    left = (dst_w - new_w) // 2
    canvas[:, top:top + new_h, left:left + new_w] = resized
    return canvas


def fit_frame_hwc(dest, source):
    """Numpy/uint8 HWC version for the shared-memory capture process: writes `source` into the preallocated
    `dest` buffer (same channel count), aspect kept, centred, black padded."""
    import cv2

    dst_h, dst_w = dest.shape[:2]
    src_h, src_w = source.shape[:2]
    if same_aspect(src_h, src_w, dst_h, dst_w):
        new_h, new_w = dst_h, dst_w
    else:
        new_h, new_w = fit_size(src_h, src_w, dst_h, dst_w)
    interpolation = cv2.INTER_AREA if (new_h < src_h or new_w < src_w) else cv2.INTER_LINEAR
    resized = cv2.resize(np.ascontiguousarray(source), (new_w, new_h), interpolation=interpolation)
    if (new_h, new_w) == (dst_h, dst_w):
        dest[:] = resized
        return
    dest[:] = 0
    top = (dst_h - new_h) // 2
    left = (dst_w - new_w) // 2
    dest[top:top + new_h, left:left + new_w] = resized


def _self_test():
    # sizes: aspect kept, never larger than the target
    assert fit_size(1080, 1920, 1080, 1920) == (1080, 1920)
    assert fit_size(720, 1280, 1080, 1920) == (1080, 1920)          # 16:9 -> 16:9 fills
    assert fit_size(1000, 1000, 1080, 1920) == (1080, 1080)          # square window in a wide frame: pillarbox
    assert fit_size(2160, 1080, 1080, 1920) == (1080, 540)           # portrait window
    assert fit_size(500, 2000, 1080, 1920) == (480, 1920)            # ultra-wide: letterbox
    # window effects vs real changes
    assert is_window_effect_size(1078, 1918, 1080, 1920) and is_window_effect_size(1080, 1920, 1080, 1920)
    assert not is_window_effect_size(720, 1280, 1080, 1920)
    assert not is_window_effect_size(1440, 5120, 1080, 3840)
    # tensor fit: exact target size, content centred, bars are black, colour preserved
    src = torch.ones((3, 500, 1000)) * 0.75                          # 2:1 picture into 16:9 frame
    out = fit_frame_chw(src, 1080, 1920)
    assert out.shape == (3, 1080, 1920)
    assert torch.allclose(out[:, 540, 960], torch.full((3,), 0.75), atol=1e-3)     # centre = picture
    assert out[:, 5, 960].abs().max() == 0 and out[:, 1075, 960].abs().max() == 0  # top/bottom bars black
    assert out[:, 540, 0].abs().max() == 0 or out[:, 540, 0].mean() > 0.7           # 2:1 in 16:9 has no side bars
    src = torch.ones((3, 1000, 1000)) * 0.5                           # square: side bars
    out = fit_frame_chw(src, 1080, 1920)
    assert out[:, 540, 10].abs().max() == 0 and abs(out[:, 540, 960].mean().item() - 0.5) < 1e-3
    assert torch.equal(fit_frame_chw(out, 1080, 1920), out)          # already the right size: untouched
    # numpy fit (shared-memory backend)
    dest = np.zeros((1080, 1920, 4), dtype=np.uint8)
    source = np.full((1000, 1000, 4), 200, dtype=np.uint8)
    fit_frame_hwc(dest, source)
    assert dest[540, 960, 0] == 200 and dest[540, 5, 0] == 0 and dest[540, 1915, 0] == 0
    dest2 = np.zeros((1080, 1920, 4), dtype=np.uint8)
    fit_frame_hwc(dest2, np.full((540, 960, 4), 90, dtype=np.uint8))  # same aspect: plain resize, no bars
    assert dest2[0, 0, 0] == 90 and dest2[1079, 1919, 0] == 90
    print("frame_fit self-test: PASS")


if __name__ == "__main__":
    _self_test()
