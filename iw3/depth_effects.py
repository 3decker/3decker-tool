import torch


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
