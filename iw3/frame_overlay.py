"""ADR-233: shared debug text-overlay helper for burning a per-frame numeric value into
the corner of a stereo output frame, so a value that only exists transiently in memory
(like the Convergence Plane an auto mode picked for this exact frame) can be read back
later as real, comparable data instead of trusted blind.

Deliberately independent of any one debug overlay's own meaning -- Auto 3D Strength's
own overlay (nt_auto3d, an optional add-on) already occupies the TOP-LEFT corner with
its own "3D X.X" stamp; this stamps the OPPOSITE corner by default so the two can be
turned on together (a real use case: comparing 3D Strength and Convergence Plane
side by side on the same test render) without drawing over each other."""
import torch

_labels = {}


def _label(text, h):
    """White text on black, as a 0..1 alpha mask (H, W), h pixels tall. Cached."""
    key = (text, h)
    m = _labels.get(key)
    if m is None:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.load_default(size=h)          # Pillow >= 10.1
            x0, y0, x1, y1 = font.getbbox(text)
            im = Image.new("L", (x1 + h // 2, y1 + h // 3), 0)
            ImageDraw.Draw(im).text((h // 4, h // 8), text, fill=255, font=font)
            m = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
        except TypeError:                                   # older Pillow: bitmap font, scaled up
            font = ImageFont.load_default()
            x0, y0, x1, y1 = font.getbbox(text)
            im = Image.new("L", (x1 + 4, y1 + 3), 0)
            ImageDraw.Draw(im).text((2, 1), text, fill=255, font=font)
            m = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
            scale = max(1, round(h / max(1, y1)))
            m = m.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        _labels[key] = m
    return m


def stamp_values(frames, values, fmt="{:.2f}", corner="top-right"):
    """Writes fmt.format(value) into one corner of each frame. frames: BCHW or CHW
    tensor, values: one number per frame (shorter lists repeat their last value).
    corner: "top-left" or "top-right" -- keep these two apart from each other so
    two different overlays (e.g. this one and Auto 3D Strength's own) never collide.
    Returns a new tensor; never mutates the one passed in."""
    if frames is None or not torch.is_tensor(frames) or not values:
        return frames
    single = frames.ndim == 3
    x = frames.unsqueeze(0) if single else frames
    x = x.clone()
    H, W = x.shape[-2:]
    h = max(12, H // 22)
    y0 = h // 3
    for i in range(x.shape[0]):
        v = values[min(i, len(values) - 1)]
        m = _label(fmt.format(v), h).to(device=x.device, dtype=x.dtype)
        mh, mw = min(m.shape[0], H - y0), min(m.shape[1], W)
        if mh <= 0 or mw <= 0:
            continue
        m = m[:mh, :mw]
        x0 = h // 3 if corner == "top-left" else max(0, W - mw - h // 3)
        r = x[i, :, y0:y0 + mh, x0:x0 + mw]
        r.copy_(r * 0.3 * (1 - m) + m)      # darkened box, white text
    return x[0] if single else x
