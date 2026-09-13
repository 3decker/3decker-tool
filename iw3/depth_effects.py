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


def apply_midground_pop(depth, strength, threshold_low=0.15, threshold_high=0.85):
    """
    Push pixels in the MIDDLE depth band (between threshold_low and threshold_high
    percentiles) toward the audience (strength > 0) or away from it (strength < 0),
    leaving true foreground (above threshold_high) and true background (below
    threshold_low) completely untouched -- the band in between what
    apply_foreground_pop/apply_background_pop each already claim at their own
    (independently configurable) thresholds.

    strength: -1.0 to 1.0. Positive amplifies each midground pixel's own distance
    from the band's FAR edge (threshold_low, bordering true background) and adds
    that as a boost, pushing it further toward the NEAR/foreground direction --
    same proportional-amplification shape as apply_foreground_pop, just anchored
    to the band's own far edge instead of the global threshold. Negative mirrors
    this, amplifying distance from the band's NEAR edge (threshold_high, bordering
    true foreground) and pushing toward the FAR/background direction, same shape
    as apply_background_pop. strength == 0 is an exact no-op (mask still computed
    harmlessly, callers should still guard with strength != 0 to skip the work
    entirely).
    """
    B = depth.shape[0]
    result = []
    for i in range(B):
        d = depth[i]
        t_low = d.flatten().quantile(threshold_low)
        t_high = d.flatten().quantile(threshold_high)
        mask = ((d >= t_low) & (d <= t_high)).to(d.dtype)
        if strength >= 0:
            offset = (d - t_low).clamp(min=0)
            d_shifted = d + offset * strength * 2.0
        else:
            offset = (t_high - d).clamp(min=0)
            d_shifted = d - offset * abs(strength) * 2.0
        d_out = d * (1 - mask) + d_shifted * mask
        result.append(d_out)
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


def _local_detail_weight(img, blur_kernel, detail_percentile):
    """0-1 per-pixel weight (B,1,H,W) marking how much genuine local detail/edge
    structure is present at each position of the RENDERED RGB image -- same
    Sobel-gradient-magnitude-normalized-against-a-high-percentile technique
    already used for TEMPORAL depth stability in depth_scaler.py's
    _optical_flow_temporal_blend (flat_region_boost/edge_protection, ADR-020)
    and for spatial depth-edge detection in _depth_edge_band_mask above,
    applied here to spatial RGB detail instead.

    Computed on a LIGHTLY PRE-BLURRED luminance, not the raw pixels: a real
    edge (an object silhouette, a multi-pixel contrast transition) mostly
    survives a light blur, but old scanned film grain -- essentially
    single-pixel-scale noise -- is exactly the kind of high-frequency content
    a blur damps hardest. Pre-blurring before measuring the gradient keeps the
    weight map anchored to real multi-pixel structure rather than individual
    noisy pixels, so a later sharpen driven by this weight stays weak across a
    grainy-but-otherwise-flat region and strong at a genuine edge, instead of
    amplifying whichever pixel happens to be locally noisiest."""
    luma = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
            if img.shape[1] >= 3 else img.mean(dim=1, keepdim=True))
    luma = F.conv2d(luma, blur_kernel, padding=1)

    kx = _SOBEL_X.to(img.device, img.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(img.device, img.dtype).view(1, 1, 3, 3)
    grad_x = F.conv2d(luma, kx, padding=1)
    grad_y = F.conv2d(luma, ky, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)

    B = grad_mag.shape[0]
    weights = []
    for i in range(B):
        g = grad_mag[i]
        ref = g.flatten().quantile(detail_percentile / 100.0)
        if ref < 1e-6:
            weights.append(torch.zeros_like(g))
        else:
            weights.append(torch.clamp(g / ref, 0.0, 1.0))
    return torch.stack(weights, dim=0)


def apply_sharpen(left_eye, right_eye, strength=0.0, detail_percentile=95.0):
    """Final-stage unsharp-mask sharpening pass on the RENDERED stereo views,
    applied AFTER whichever stereo method produced them (row_flow/MLBW/monobw/
    forward-fill/inpaint all converge here, same hook point as repair_stereo_
    edges above) -- so it enhances the actual picture being delivered, never
    an intermediate depth map, regardless of which stereo method made it.

    Runs AFTER Edge Repair in apply_divergence() (iw3/utils.py), deliberately:
    Edge Repair's whole job is smoothing away thin fringing/ghosting residue
    right at a depth edge, and running sharpening first would hand it a
    sharpened (i.e. exaggerated) version of exactly that residue to clean up,
    working against it. Sharpening after Edge Repair instead enhances detail
    across an already-cleaned frame, including a now-clean edge band, without
    re-amplifying the artifact Edge Repair just removed.

    Classic unsharp mask: sharpened = original + strength * (original -
    blurred(original)), clipped back to the valid [0, 1] range this pipeline's
    image tensors use throughout. Scaled per-pixel by an edge-aware weight
    (_local_detail_weight, computed from the rendered image's OWN structure)
    so the boost is strongest at real detail/edges and tapers to near zero in
    flat or grain-only regions -- avoids the classic unsharp-mask failure mode
    of aggressively amplifying film grain/sensor noise into ugly speckling on
    scanned/older source material. strength=0.0 (the default) is an exact
    no-op with zero extra cost, so nothing changes for anyone who doesn't turn
    this on."""
    if strength <= 0.0 or left_eye is None or right_eye is None:
        return left_eye, right_eye

    blur_kernel = _BLUR3.to(left_eye.device, left_eye.dtype).view(1, 1, 3, 3)

    def _sharpen(img):
        C = img.shape[1]
        weight = _local_detail_weight(img, blur_kernel, detail_percentile) * strength
        blurred = F.conv2d(img, blur_kernel.repeat(C, 1, 1, 1), padding=1, groups=C)
        detail = img - blurred
        return torch.clamp(img + weight * detail, 0.0, 1.0)

    return _sharpen(left_eye), _sharpen(right_eye)


def _test_apply_midground_pop():
    """Synthetic, no-GPU regression test for apply_midground_pop's actual
    directionality and band-isolation -- verifies the real behavior, not just
    that it runs. Depth convention confirmed against apply_foreground_pop/
    apply_background_pop's own code: HIGHER depth value = NEARER (foreground),
    LOWER depth value = FARTHER (background). Built this way specifically
    because an earlier draft of this function's own docstring had the near/far
    edge labels backwards (caught and fixed before shipping) -- this test
    locks the real, code-verified direction down going forward."""
    torch.manual_seed(0)
    # A predictable ramp (0..1 across 100 values, reshaped to a square-ish
    # image) gives exactly-known quantiles for threshold_low/threshold_high.
    ramp = torch.linspace(0.0, 1.0, steps=100).view(1, 1, 10, 10)

    # strength == 0 must be an exact no-op.
    out = apply_midground_pop(ramp, 0.0)
    assert torch.allclose(out, ramp), "strength=0 must not change anything"

    # True background (below threshold_low=0.15) and true foreground (above
    # threshold_high=0.85) must be untouched regardless of strength/sign.
    for strength in (0.5, -0.5):
        out = apply_midground_pop(ramp, strength, threshold_low=0.15, threshold_high=0.85)
        bg_mask = ramp < ramp.flatten().quantile(0.15)
        fg_mask = ramp > ramp.flatten().quantile(0.85)
        assert torch.allclose(out[bg_mask], ramp[bg_mask]), \
            f"true background must be untouched at strength={strength}"
        assert torch.allclose(out[fg_mask], ramp[fg_mask]), \
            f"true foreground must be untouched at strength={strength}"

    # Positive strength must push midground values UP (toward the near/
    # foreground direction -- higher depth value), never down.
    out_pos = apply_midground_pop(ramp, 0.5, threshold_low=0.15, threshold_high=0.85)
    mid_mask = (ramp >= ramp.flatten().quantile(0.15)) & (ramp <= ramp.flatten().quantile(0.85))
    assert (out_pos[mid_mask] >= ramp[mid_mask]).all(), \
        "positive strength must push midground pixels toward higher (nearer) values"
    assert (out_pos[mid_mask] > ramp[mid_mask]).any(), \
        "positive strength must actually change at least some midground pixels"

    # Negative strength must push midground values DOWN (toward the far/
    # background direction -- lower depth value), never up.
    out_neg = apply_midground_pop(ramp, -0.5, threshold_low=0.15, threshold_high=0.85)
    assert (out_neg[mid_mask] <= ramp[mid_mask]).all(), \
        "negative strength must push midground pixels toward lower (farther) values"
    assert (out_neg[mid_mask] < ramp[mid_mask]).any(), \
        "negative strength must actually change at least some midground pixels"

    # A custom, narrower threshold band must actually narrow which pixels get
    # touched -- confirms threshold_low/threshold_high are real, live inputs,
    # not silently ignored in favor of the function's own defaults.
    out_narrow = apply_midground_pop(ramp, 0.5, threshold_low=0.4, threshold_high=0.6)
    now_untouched = (ramp >= ramp.flatten().quantile(0.15)) & (ramp < ramp.flatten().quantile(0.4))
    assert torch.allclose(out_narrow[now_untouched], ramp[now_untouched]), \
        "a narrower threshold_low must exclude pixels the wider default band would have touched"

    print("_test_apply_midground_pop: PASS")


if __name__ == "__main__":
    _test_apply_midground_pop()
