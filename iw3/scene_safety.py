"""ADR-505: Scene Safety Check -- an optional, capped, once-per-scene correction on top of
the planned per-shot-type values (Pre-Plan Every Scene).

Measured on a scene's FIRST frames, in the same depth space the stereo step compares
against Convergence (Foreground Scale mapper first, then Depth Curve Match -- the real
order in iw3.utils.apply_divergence), and held for the whole scene (never changes
mid-scene):

1. Mismatch guard -> lowers that scene's 3D Strength (at most `max_strength_cut`, default
   10%). Trigger: a deep AND busy (high-texture) background. Real finding behind this
   (2026-11-26/27 renders): left/right mismatch follows the TOTAL amount of depth, not
   where the screen plane sits, so only less depth reduces it; the worst background
   mismatch (Mario 8:17 ruins) was the same with or without Edge Snap / Smear Guard.
2. Comfort guard -> raises that scene's Convergence (at most `max_conv_raise`, default
   0.10). Triggers: too much pop-out (near objects far in front of the screen), or a
   near object cut off by the left/right frame edge (window violation).

Everything is a pure function of tensors so it can be unit-tested without a model."""
import torch
import torch.nn.functional as F

# thresholds (units: "divergence-%" = 3D Strength x depth distance from the screen plane,
# i.e. proportional to the on-screen L/R shift). Calibrated (ADR-505) on all 1420 scenes of
# the Mario movie: saved full-movie VDA_L depth + one source frame per scene + D2's planned
# per-scene strength/Convergence/Foreground Scale. busy: p50 0.0016, p85 0.0070, p95 0.0121,
# p98 0.0165, max 0.030 (the 8:12-8:20 ruins = 0.0174, top 1.5%); pop: p50 1.08, p95 2.57,
# max 3.10; edge_pop: p50 0.63, p95 2.45, max 3.06 (close-ups fill the frame, so only the
# top ~5% count as a real frame-edge problem).
BUSY_RISK_LOW = 0.007     # ~top 15% of scenes start getting a (small) strength cut
BUSY_RISK_HIGH = 0.017    # ~top 2%: full cut
POP_LIMIT = 2.6           # ~top 5% pop-out -> raise Convergence
EDGE_LIMIT = 2.45         # ~top 5% frame-edge pop -> raise Convergence
EDGE_STRIP = 0.04


def _luma(im):
    if im.shape[1] == 3:
        return (0.299 * im[:, 0:1] + 0.587 * im[:, 1:2] + 0.114 * im[:, 2:3])
    return im[:, 0:1]


def texture(im, size):
    """Per-pixel local texture (Sobel gradient magnitude of luma) at the DEPTH map's size.
    The picture is shrunk to `size` FIRST: gradients measured on a full 4K frame are several
    times weaker than on a depth-sized one (same detail spread over more pixels), which made
    the real render score far below the calibration (2026-11-27 E4: 4 strength cuts instead of
    ~40, Mario 8:12 ruins missed). Shrinking first makes it resolution-independent and matches
    how the thresholds were calibrated."""
    y = _luma(im.float())
    if tuple(y.shape[-2:]) != tuple(size):
        y = F.interpolate(y, size=size, mode="area")
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=y.device).view(1, 1, 3, 3) / 8
    ky = kx.transpose(2, 3)
    y = F.pad(y, (1, 1, 1, 1), mode="replicate")  # zero padding made the picture's own border look "busy"
    return torch.sqrt(F.conv2d(y, kx) ** 2 + F.conv2d(y, ky) ** 2)


def measure(final_depth, im, convergence, divergence):
    """final_depth: (N,1,H,W) 0..1 depth as compared to Convergence; im: (N,3,h,w) 0..1.
    Returns the scene's risk numbers (averaged over the given frames)."""
    d = final_depth.float()
    n = d.shape[0]
    flat = d.reshape(n, -1)
    near = torch.quantile(flat, 0.99, dim=1)
    pop = float((divergence * (near - convergence)).clamp(min=0).mean())
    W = d.shape[-1]
    k = max(1, int(round(W * EDGE_STRIP)))
    edges = torch.cat([d[..., :k], d[..., -k:]], dim=-1).reshape(n, -1)
    edge_pop = float((divergence * (torch.quantile(edges, 0.95, dim=1) - convergence)).clamp(min=0).mean())
    behind = (convergence - d).clamp(min=0)                     # how far behind the screen
    tex = texture(im, d.shape[-2:])
    busy = float((divergence * behind * tex).mean())            # deep x textured, per pixel
    return {"pop": pop, "edge_pop": edge_pop, "busy": busy}


def auto_convergence(final_depth, front_share=0.15, prev=None, max_step=0.20):
    """ADR-507: the Convergence that puts exactly `front_share` of this scene's picture in front
    of the screen (depth above the plane pops out), so every scene gets the same feel. Limited
    to +-max_step from the previous scene's value so neighbouring scenes don't jump."""
    flat = final_depth.float().reshape(-1)
    if flat.numel() > 2_000_000:
        flat = flat[torch.randperm(flat.numel(), device=flat.device)[:2_000_000]]
    conv = float(torch.quantile(flat, 1.0 - float(front_share)))
    if prev is not None and max_step is not None and max_step > 0:
        conv = min(max(conv, prev - max_step), prev + max_step)
    return min(max(conv, 0.0), 1.0)


def decide(m, max_strength_cut=0.10, max_conv_raise=0.10, busy_low=None, busy_high=None,
           pop_limit=None, edge_limit=None):
    """Risk numbers -> (strength_factor, conv_offset, reasons). The trigger levels default to
    the D2-calibrated constants above; a run with very different depth (e.g. no curve + depth
    reset) passes its own calibrated levels (ADR-506 CLI flags)."""
    busy_low = BUSY_RISK_LOW if busy_low is None else float(busy_low)
    busy_high = BUSY_RISK_HIGH if busy_high is None else float(busy_high)
    pop_limit = POP_LIMIT if pop_limit is None else float(pop_limit)
    edge_limit = EDGE_LIMIT if edge_limit is None else float(edge_limit)
    reasons = []
    t = (m["busy"] - busy_low) / max(busy_high - busy_low, 1e-9)
    t = min(max(t, 0.0), 1.0)
    factor = 1.0 - max_strength_cut * t
    if t > 0:
        reasons.append(f"busy deep background (risk {m['busy']:.3f})")
    raise_by = 0.0
    if m["pop"] > pop_limit:
        raise_by = max(raise_by, (m["pop"] - pop_limit) / 10.0)
        reasons.append(f"pops out far (pop {m['pop']:.2f})")
    if m["edge_pop"] > edge_limit:
        raise_by = max(raise_by, (m["edge_pop"] - edge_limit) / 10.0)
        reasons.append(f"near object at the frame edge (edge pop {m['edge_pop']:.2f})")
    conv_offset = min(raise_by, max_conv_raise)
    return factor, conv_offset, reasons


def _self_test():
    torch.manual_seed(0)
    flat_im = torch.full((2, 3, 64, 96), 0.5)
    busy_im = torch.rand(2, 3, 64, 96)
    calm = torch.full((2, 1, 64, 96), 0.45)                     # everything near the plane
    m = measure(calm, flat_im, 0.4, 3.8)
    f, o, r = decide(m)
    assert f == 1.0 and o == 0.0 and not r, (m, f, o, r)
    deep = torch.full((2, 1, 64, 96), 0.02)                     # deep background...
    m = measure(deep, busy_im, 0.5, 3.8)                        # ...full of texture
    f, o, r = decide(m)
    assert 0.899 <= f < 1.0 and "busy" in r[0], (m, f, r)
    m2 = measure(deep, flat_im, 0.5, 3.8)                       # same depth, no texture
    assert decide(m2)[0] == 1.0, m2
    # resolution-independent: the same picture at 4x the size scores the same
    big = F.interpolate(busy_im, scale_factor=4, mode="nearest")
    mb = measure(deep, big, 0.5, 3.8)
    assert abs(mb["busy"] - m["busy"]) < 1e-4, (mb, m)
    near_edge = torch.full((2, 1, 64, 96), 0.3)
    near_edge[..., :3] = 1.0                                    # something near, cut by the left edge
    m = measure(near_edge, flat_im, 0.3, 3.8)
    f, o, r = decide(m)
    assert 0.0 < o <= 0.10 and any("edge" in x for x in r), (m, o, r)
    big_pop = torch.full((2, 1, 64, 96), 0.3)
    big_pop[..., 20:40, 40:60] = 1.0
    m = measure(big_pop, flat_im, 0.0, 4.3)
    f, o, r = decide(m, max_conv_raise=0.05)
    assert o == 0.05 and any("pops out" in x for x in r), (m, o, r)
    # per-run trigger levels: a higher pop limit stops the raise; a lower busy range triggers
    assert decide(m, pop_limit=99, edge_limit=99)[1] == 0.0
    mild = {"busy": 0.005, "pop": 0.0, "edge_pop": 0.0}
    assert decide(mild)[0] == 1.0 and decide(mild, busy_low=0.001, busy_high=0.005, max_strength_cut=0.3)[0] == 0.7
    # ADR-507 auto Convergence: exactly front_share in front; jump limited vs the previous scene
    ramp = torch.linspace(0, 1, 10000).view(1, 1, 100, 100)
    c = auto_convergence(ramp, 0.15)
    assert abs(c - 0.85) < 0.01 and abs(float((ramp > c).float().mean()) - 0.15) < 0.01, c
    assert abs(auto_convergence(ramp, 0.15, prev=0.5, max_step=0.2) - 0.7) < 1e-6
    assert abs(auto_convergence(ramp, 0.15, prev=0.8, max_step=0.2) - c) < 1e-6
    print("scene_safety self-test: PASS")


if __name__ == "__main__":
    _self_test()
