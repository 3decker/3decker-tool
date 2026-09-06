import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_SOBEL_Y = _SOBEL_X.t()
_BLUR3 = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
_BLUR3 = _BLUR3 / _BLUR3.sum()


def apply_foreground_pop(depth, strength, threshold_percentile=0.85):
    """
    Boost pixels already in the foreground, pushing them further toward the audience.
    Pixels above threshold_percentile of depth get amplified proportionally.
    Works on any depth scale — no [0,1] assumption needed.
    strength: 0.0-1.0, how hard to push foreground pixels forward.
    """
    B = depth.shape[0]
    result = []
    for i in range(B):
        d = depth[i]
        threshold = d.flatten().quantile(threshold_percentile)
        # How far each pixel is above the threshold (0 for background pixels)
        above = (d - threshold).clamp(min=0)
        # Boost: add proportional to how far above threshold × strength
        d_boosted = d + above * strength * 2.0
        result.append(d_boosted)
    return torch.stack(result, dim=0)


def apply_background_pop(depth, strength, threshold_percentile=0.15):
    """
    Push pixels already in the background further away from the audience.
    Pixels below threshold_percentile of depth get suppressed proportionally.
    Works on any depth scale — no [0,1] assumption needed.
    strength: 0.0-1.0, how hard to push background pixels back.
    """
    B = depth.shape[0]
    result = []
    for i in range(B):
        d = depth[i]
        threshold = d.flatten().quantile(threshold_percentile)
        # How far each pixel is below the threshold (0 for foreground pixels)
        below = (threshold - d).clamp(min=0)
        # Suppress: subtract proportional to how far below threshold × strength
        d_suppressed = d - below * strength * 2.0
        result.append(d_suppressed)
    return torch.stack(result, dim=0)


def apply_background_divergence(depth, convergence, base_divergence, background_divergence,
                                 threshold_percentile=0.15):
    """
    Give the farthest threshold_percentile of pixels their own effective Divergence,
    independent of the Divergence applied to the rest of the scene.

    Mechanism: final pixel shift is proportional to (depth - convergence) * divergence.
    Rescaling a pixel's distance from the convergence plane by (background_divergence /
    base_divergence) before the shared Divergence multiplier is applied reproduces the
    same result as if that pixel alone used background_divergence. Pixels above the
    threshold (not in the farthest slice) are left untouched.
    """
    if base_divergence == 0 or background_divergence == base_divergence:
        return depth
    ratio = background_divergence / base_divergence
    B = depth.shape[0]
    result = []
    for i in range(B):
        d = depth[i]
        c = convergence[i] if torch.is_tensor(convergence) and convergence.ndim > 0 else convergence
        threshold = d.flatten().quantile(threshold_percentile)
        mask = (d < threshold).to(d.dtype)
        rescaled = c + (d - c) * ratio
        d_out = d * (1 - mask) + rescaled * mask
        result.append(d_out)
    return torch.stack(result, dim=0)


def apply_foreground_divergence(depth, convergence, base_divergence, foreground_divergence,
                                 threshold_percentile=0.85):
    """
    Give the nearest (1 - threshold_percentile) of pixels their own effective
    Divergence, independent of the Divergence applied to the rest of the scene.
    Mirror image of apply_background_divergence — same mechanism, opposite end.
    """
    if base_divergence == 0 or foreground_divergence == base_divergence:
        return depth
    ratio = foreground_divergence / base_divergence
    B = depth.shape[0]
    result = []
    for i in range(B):
        d = depth[i]
        c = convergence[i] if torch.is_tensor(convergence) and convergence.ndim > 0 else convergence
        threshold = d.flatten().quantile(threshold_percentile)
        mask = (d > threshold).to(d.dtype)
        rescaled = c + (d - c) * ratio
        d_out = d * (1 - mask) + rescaled * mask
        result.append(d_out)
    return torch.stack(result, dim=0)


def _depth_edge_band_mask(depth, band_width, edge_percentile):
    """0-1 mask (BCHW, same H/W as depth) marking a thin band right around each real
    depth discontinuity -- built the same way as depth_blend.py's _depth_edge_mask
    (Sobel gradient magnitude, normalized against that frame's own high-percentile
    gradient so it self-calibrates to each shot's contrast), except this stays in
    torch end-to-end (no cpu/numpy round trip) since, unlike that function, this one
    runs on every frame of every video when enabled, not once per export pass."""
    kx = _SOBEL_X.to(depth.device, depth.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(depth.device, depth.dtype).view(1, 1, 3, 3)
    grad_x = F.conv2d(depth, kx, padding=1)
    grad_y = F.conv2d(depth, ky, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)

    B = grad_mag.shape[0]
    masks = []
    for i in range(B):
        g = grad_mag[i]
        ref = g.flatten().quantile(edge_percentile / 100.0)
        if ref < 1e-6:
            masks.append(torch.zeros_like(g))
        else:
            masks.append(torch.clamp(g / ref, 0.0, 1.0))
    mask = torch.stack(masks, dim=0)

    if band_width > 1:
        k = band_width if band_width % 2 == 1 else band_width + 1
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    return mask


def repair_stereo_edges(left_eye, right_eye, depth, strength=0.0, band_width=2, edge_percentile=90.0):
    """Final-stage cleanup pass on the RENDERED stereo views, applied AFTER whichever
    stereo method produced them (row_flow/MLBW/monobw/forward-fill/inpaint all
    converge here) -- so it catches thin (1-3px) fringing/ghosting residue left right
    at a depth edge regardless of which method made it, instead of needing a
    method-specific fix in each warp implementation individually.

    Deliberately narrow in scope: only blends in a touch of local smoothing exactly
    within a thin band around each frame's OWN real depth edges (never anywhere else
    in the image), so it can only ever soften a hairline fringe right at a silhouette
    -- it cannot blur flat regions, and it cannot touch a region with no underlying
    depth discontinuity. strength=0.0 (the default) is an exact no-op with zero
    extra cost, so nothing changes for anyone who doesn't turn this on.

    Real bug found in production (2026-09-06): the video inpaint methods
    (forward_inpaint/mlbw_l2_inpaint/monobw_inpaint) buffer frames internally for
    temporal consistency, so the left_eye/right_eye batch this function receives can
    have a DIFFERENT frame count than the `depth` batch that was originally fed in
    for that step -- fewer when output is delayed inside the buffer, or more when a
    scene-cut flush releases several buffered frames at once with no depth tensor of
    their own available at the call site. Interpolating depth's H/W to match doesn't
    fix a mismatched BATCH size, and blending a (B_depth, 1, H, W) mask against a
    (B_eyes, 3, H, W) image crashes with a tensor-broadcast RuntimeError. Guarded
    below: silently skip repair (return unmodified) rather than crash whenever the
    batch sizes don't line up -- this only means the handful of frames right at a
    scene-cut flush miss this optional cosmetic pass, not a visible regression."""
    if strength <= 0.0 or left_eye is None or right_eye is None:
        return left_eye, right_eye

    if depth.shape[0] != left_eye.shape[0]:
        return left_eye, right_eye

    if depth.shape[2:] != left_eye.shape[2:]:
        depth = F.interpolate(depth, size=left_eye.shape[2:], mode="bilinear", align_corners=True)

    mask = _depth_edge_band_mask(depth, band_width, edge_percentile) * strength

    blur_kernel = _BLUR3.to(left_eye.device, left_eye.dtype).view(1, 1, 3, 3)

    def _clean(img):
        C = img.shape[1]
        blurred = F.conv2d(img, blur_kernel.repeat(C, 1, 1, 1), padding=1, groups=C)
        return img * (1.0 - mask) + blurred * mask

    return _clean(left_eye), _clean(right_eye)
