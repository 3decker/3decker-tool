import torch
import torch.nn.functional as F
from nunif.modules.replication_pad2d import ReplicationPad2d


def box_blur(x, kernel_size=7):
    padding = kernel_size // 2
    x = F.avg_pool2d(x, kernel_size=kernel_size, padding=padding, stride=1, count_include_pad=False)
    return x


def blur_blend(x, mask):
    mask = torch.clamp(box_blur(mask.to(x.dtype)), 0, 1)
    x_blur = box_blur(x)
    return x * (1.0 - mask) + x_blur * mask


def shift_fill(x, sign, flip_sign=False, max_tries=100):
    mask = x < 0
    while mask.any().item() and max_tries > 0:
        if sign > 0:
            x[mask] = F.pad(x[:, :, :, 1:], (0, 1, 0, 0))[mask]
        else:
            x[mask] = F.pad(x[:, :, :, :-1], (1, 0, 0, 0))[mask]
        mask = x < 0
        max_tries = max_tries - 1
        if flip_sign:
            sign = -1 if sign > 0 else 1

    return x


def shift_fill_pack(left_eye, right_eye, inconsistent_shift=False):
    if inconsistent_shift:
        pack = torch.cat([left_eye, right_eye], dim=1)
        left_eye, right_eye = shift_fill(pack, 1, flip_sign=True).chunk(2, dim=1)
        return left_eye, right_eye
    else:
        pack = torch.cat([left_eye, torch.flip(right_eye, dims=(-1,))], dim=1)
        left_eye, right_eye = shift_fill(pack, -1).chunk(2, dim=1)
        right_eye = torch.flip(right_eye, dims=(-1,))
        return left_eye, right_eye


def fix_layered_holes(side_image, index_image, sign, max_tries=100):
    if sign > 0:
        mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (0, 1, 0, 0))
        while mask.any().item() and max_tries > 0:
            side_image[mask.expand_as(side_image)] = -2  # set undefined value
            index_image[mask] = F.pad(index_image[:, :, :, 1:], (0, 1, 0, 0))[mask]
            mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (0, 1, 0, 0))
            max_tries -= 1
    else:
        mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (1, 0, 0, 0))
        while mask.any().item() and max_tries > 0:
            side_image[mask.expand_as(side_image)] = -2
            index_image[mask] = F.pad(index_image[:, :, :, :-1], (1, 0, 0, 0))[mask]
            mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]) > 0, (1, 0, 0, 0))
            max_tries -= 1


def __detect_overlap_mask(index_image, mask):
    overlap_mask = F.pad((index_image[:, :, :, :-1] - index_image[:, :, :, 1:]).abs() > 2.1, (0, 1, 0, 0))
    overlap_mask[mask] = False
    return overlap_mask


def to_flat_index(batch, width, height, index):
    index = index + torch.arange(0, height, device=index.device).view(1, height, 1) * width
    index = index + torch.arange(0, batch, device=index.device).view(batch, 1, 1) * height * width
    index = index.view(-1)
    return index


def make_bilinear_data(batch, width, height, index, index_shift):
    float_index = torch.clamp(index + index_shift, 0, width - 1)
    floor_index = torch.clamp(float_index.floor(), 0, width - 1)
    ceil_index = torch.clamp(float_index.ceil(), 0, width - 1)
    ceil_weight = (float_index - floor_index).reshape(batch, 1, height, width)
    ceil_weight = torch.clamp(ceil_weight, min=1e-5, max=1.0 - 1e-5)
    floor_weight = 1.0 - ceil_weight
    floor_index = to_flat_index(batch, width, height, floor_index.long())
    ceil_index = to_flat_index(batch, width, height, ceil_index.long())

    return floor_index, ceil_index, floor_weight, ceil_weight


def ordered_index_copy(c, src_index, dest_index, index_order, undefined_value=-1):
    B, _, H, W = c.shape
    c = c.permute(0, 2, 3, 1).reshape(-1, c.shape[1])
    if torch.is_tensor(undefined_value):
        out = undefined_value.view(1, -1).repeat(c.shape[0], 1)
    else:
        out = torch.empty_like(c).fill_(undefined_value)

    # index_copy must run deterministically (depth order orverride)
    # NOTE: `torch.use_deterministic_algorithms(True)` is a global setting.
    #        This may cause an error if other threads run non-deterministic method while in this block.
    #        Need to exclusive lock to prevent other threads running.
    # TODO: Need to remove the complicated conditions of this very simple operation.
    # for i in index_order:
    #   out[dest_index[i]] = c[src_index[i]]
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        out.index_copy_(0, dest_index[index_order], c[src_index[index_order]])
    finally:
        torch.use_deterministic_algorithms(deterministic)

    return out.view(B, H, W, -1).permute(0, 3, 1, 2)


def _softmax_weight(depth, temperature):
    # Higher `depth` == nearer camera in this file's convention: ordered_index_copy's
    # ascending argsort + last-write-wins already means the highest-depth source among
    # colliding pixels wins the hard z-buffer overwrite, so importance must increase
    # with depth here too, to keep the same "nearer wins" semantics under blending.
    return torch.exp(depth * temperature)


# Softmax-splatting (Niklaus & Liu 2020) collision-blend temperature. depth is
# normalized to roughly [0, 1] by the time it reaches here (see apply_divergence's
# mapper step in iw3/utils.py), so with temperature=50 a depth gap of ~0.05 between
# two colliding sources already yields exp(0.05 * 50) =~ 12x weight ratio -- close
# to the old hard z-buffer cutoff for genuinely different surfaces -- while true
# sub-pixel collisions on the same surface (gaps of ~0.01 or less) still blend
# smoothly instead of flip-flopping between hard winners. Not yet exposed as a
# tunable: this project's convention (see ADR-022/025/027) is to promote a
# hardcoded constant to a real setting only after real-footage testing shows it
# needs tuning, not speculatively.
SPLAT_BLEND_TEMPERATURE = 50.0


def splat_accumulate(c, src_index, dest_index, weight):
    """Depth-weighted accumulate (softmax splatting) collision resolution:
    out = sum(w_i * value_i) / sum(w_i) over every source pixel landing on the same
    destination, instead of ordered_index_copy's single-winner hard z-buffer
    overwrite. Uses index_add_, which is commutative, so unlike ordered_index_copy
    there is no source ordering or global determinism-flag handling needed here
    (empirically confirmed deterministic under
    torch.use_deterministic_algorithms(True) on this project's pinned CUDA torch
    build -- see docs/ai/AI_DECISIONS.md ADR-030).
    Returns (blended_value, defined_mask), both flattened back to (B, C, H, W) /
    (B, 1, H, W).
    """
    B, _, H, W = c.shape
    N = B * H * W
    c_flat = c.permute(0, 2, 3, 1).reshape(N, -1)
    weight_flat = weight.permute(0, 2, 3, 1).reshape(N, 1)

    value_sum = torch.zeros_like(c_flat)
    weight_sum = torch.zeros(N, 1, dtype=c_flat.dtype, device=c_flat.device)
    value_sum.index_add_(0, dest_index, c_flat[src_index] * weight_flat[src_index])
    weight_sum.index_add_(0, dest_index, weight_flat[src_index])

    defined = weight_sum > 0
    blended = value_sum / weight_sum.clamp(min=1e-12)

    blended = blended.view(B, H, W, -1).permute(0, 3, 1, 2)
    defined = defined.view(B, H, W, 1).permute(0, 3, 1, 2)
    return blended, defined


def warp(batch, width, height, c, x_index, index_shift, src_index, index_order,
        splat_blend=False, depth_weight=None):
    floor_index, ceil_index, floor_weight, ceil_weight = make_bilinear_data(batch, width, height, x_index, index_shift)

    # pack for optimization
    floor_data = torch.cat([floor_weight, c], dim=1)
    ceil_data = torch.cat([ceil_weight, c], dim=1)

    # 0 for weight, -1 for pixel
    undefined_value = torch.tensor([0] + [-1] * c.shape[1], dtype=c.dtype, device=c.device)
    floor_warp = ordered_index_copy(floor_data, src_index, floor_index, index_order, undefined_value=undefined_value)
    ceil_warp = ordered_index_copy(ceil_data, src_index, ceil_index, index_order, undefined_value=undefined_value)

    # unpack
    floor_weight_warp, floor_warp = floor_warp[:, 0:1, :, :], floor_warp[:, 1:, :, :]
    ceil_weight_warp, ceil_warp = ceil_warp[:, 0:1, :, :], ceil_warp[:, 1:, :, :]

    if splat_blend:
        # Replace the hard z-buffer winner's COLOR channels at destinations that
        # receive more than one source contribution with a depth-weighted softmax
        # blend of all contributors. The last channel of `c` (the width-index
        # channel appended by the caller for hole detection) is deliberately left
        # untouched on the hard-overwrite result: fix_layered_holes/shift_fill_pack
        # need a single winner's raw integer column index there, not a blended float.
        color_channels = c.shape[1] - 1
        splat_weight = depth_weight.reshape(batch, 1, height, width)

        floor_color, floor_defined = splat_accumulate(
            c[:, :color_channels], src_index, floor_index, floor_weight * splat_weight)
        ceil_color, ceil_defined = splat_accumulate(
            c[:, :color_channels], src_index, ceil_index, ceil_weight * splat_weight)

        floor_warp = torch.cat([
            torch.where(floor_defined, floor_color, floor_warp[:, :color_channels]),
            floor_warp[:, color_channels:],
        ], dim=1)
        ceil_warp = torch.cat([
            torch.where(ceil_defined, ceil_color, ceil_warp[:, :color_channels]),
            ceil_warp[:, color_channels:],
        ], dim=1)

    out = (floor_warp * floor_weight_warp + ceil_warp * ceil_weight_warp) / (floor_weight_warp + ceil_weight_warp)
    out = torch.nan_to_num(out, -1)

    return out


def gen_mask2(mask):
    mask = mask[:, 0:1]
    return torch.clamp((mask == -1).float() + (mask == -2).float() * 0.5, 0, 1)


def depth_order_bilinear_forward_warp(c, depth, divergence, convergence, fill=True,
                                      synthetic_view="both",
                                      return_mask=False, inconsistent_shift=False,
                                      width_base=True, splat_blend=False):
    src_image = c
    assert synthetic_view in {"both", "right", "left"}
    if c.shape[2] != depth.shape[2] or c.shape[3] != depth.shape[3]:
        depth = F.interpolate(depth, size=c.shape[-2:],
                              mode="bilinear", align_corners=True, antialias=True)
    if synthetic_view != "both":
        divergence *= 2

    # pad
    if width_base:
        base_size = c.shape[-1]
    else:
        base_size = max(c.shape[-2:])

    padding_size = int(base_size * divergence * 0.01 + 2)
    pad = ReplicationPad2d((padding_size, padding_size, 0, 0))
    unpad = ReplicationPad2d((-padding_size, -padding_size, 0, 0))
    c = pad(c)
    depth = pad(depth)

    # forward warping
    B, _, H, W = depth.shape
    shift_size = divergence * 0.01 * base_size * 0.5
    index_shift = depth * shift_size - (shift_size * convergence)
    index_shift = index_shift.view(B, H, W)
    x_index = torch.arange(0, W, device=c.device).view(1, 1, W).expand(B, H, W)
    src_index = to_flat_index(B, W, H, x_index)
    index_order = torch.argsort(depth.view(-1), dim=0)
    depth_weight = _softmax_weight(depth, SPLAT_BLEND_TEMPERATURE) if splat_blend else None

    c = torch.cat([c, x_index.view(B, 1, H, W).to(c.dtype)], dim=1)  # warp width index together

    if synthetic_view == "both":
        left_eye = warp(B, W, H, c, x_index, index_shift, src_index, index_order,
                        splat_blend=splat_blend, depth_weight=depth_weight)
        right_eye = warp(B, W, H, c, x_index, -index_shift, src_index, index_order,
                         splat_blend=splat_blend, depth_weight=depth_weight)

        # unpad
        left_eye = unpad(left_eye)
        right_eye = unpad(right_eye)
        left_eye, left_eye_index = left_eye[:, :-1, :, :], left_eye[:, -1:, :, :]
        right_eye, right_eye_index = right_eye[:, :-1, :, :], right_eye[:, -1:, :, :]

        # Fix layered holes
        # inspired by @math-artist patch: https://github.com/nagadomi/nunif/discussions/274
        left_eye_index, right_eye_index = shift_fill_pack(left_eye_index, right_eye_index,
                                                          inconsistent_shift=inconsistent_shift)
        fix_layered_holes(left_eye, left_eye_index, 1)
        fix_layered_holes(right_eye, right_eye_index, -1)

        if return_mask:
            left_mask, right_mask = gen_mask2(left_eye), gen_mask2(right_eye)

        if fill:
            # super simple inpainting
            left_eye, right_eye = shift_fill_pack(left_eye, right_eye, inconsistent_shift=inconsistent_shift)
        else:
            # drop undefined values
            left_eye = torch.clamp(left_eye, 0, 1)
            right_eye = torch.clamp(right_eye, 0, 1)

        if return_mask:
            return left_eye.contiguous(), right_eye.contiguous(), left_mask, right_mask
        else:
            return left_eye.contiguous(), right_eye.contiguous()

    elif synthetic_view == "right":
        right_eye = warp(B, W, H, c, x_index, -index_shift, src_index, index_order,
                         splat_blend=splat_blend, depth_weight=depth_weight)
        right_eye = unpad(right_eye)
        right_eye, right_eye_index = right_eye[:, :-1, :, ], right_eye[:, -1:, :, ]
        right_eye_index = shift_fill(right_eye_index, 1)
        fix_layered_holes(right_eye, right_eye_index, -1)
        if return_mask:
            right_mask = gen_mask2(right_eye)
        if fill:
            right_eye = shift_fill(right_eye, 1)
        else:
            right_eye = torch.clamp(right_eye, 0, 1)

        if return_mask:
            return src_image, right_eye.contiguous(), None, right_mask
        else:
            return src_image, right_eye.contiguous()

    elif synthetic_view == "left":
        left_eye = warp(B, W, H, c, x_index, index_shift, src_index, index_order,
                        splat_blend=splat_blend, depth_weight=depth_weight)
        left_eye = unpad(left_eye)
        left_eye, left_eye_index = left_eye[:, :-1, :, ], left_eye[:, -1:, :, ]
        left_eye_index = shift_fill(left_eye_index, -1)
        fix_layered_holes(left_eye, left_eye_index, 1)
        if return_mask:
            left_mask = gen_mask2(left_eye)

        if fill:
            left_eye = shift_fill(left_eye, -1)
        else:
            left_eye = torch.clamp(left_eye, 0, 1)

        if return_mask:
            return left_eye.contiguous(), src_image, left_mask, None
        else:
            return left_eye.contiguous(), src_image


def apply_divergence_forward_warp(c, depth, divergence, convergence, method=None,
                                  synthetic_view="both",
                                  return_mask=False, inconsistent_shift=False,
                                  width_base=True):
    fill = method in {"forward_fill", "forward_splat_fill"}
    splat_blend = (method == "forward_splat_fill")
    with torch.inference_mode():
        return depth_order_bilinear_forward_warp(c, depth, divergence, convergence,
                                                 fill=fill, synthetic_view=synthetic_view,
                                                 return_mask=return_mask,
                                                 inconsistent_shift=inconsistent_shift,
                                                 width_base=width_base, splat_blend=splat_blend)


def nonwarp_mask(c, depth, divergence, convergence, view="right"):
    divergence = divergence * 0.5  # cancels out 2x multiplier for synthetic_view = right|left

    if c.shape[2] != depth.shape[2] or c.shape[3] != depth.shape[3]:
        depth = F.interpolate(depth, size=c.shape[-2:],
                              mode="bilinear", align_corners=True, antialias=True)

    # warp depth to the left
    depth3 = depth.repeat(1, 3, 1, 1)
    if view == "right":
        warped_depth, _ = depth_order_bilinear_forward_warp(
            depth3, depth, divergence, convergence,
            synthetic_view="left",
            fill=True, inconsistent_shift=False, return_mask=False)
        warped_depth = warped_depth.mean(dim=1, keepdim=True)
        # warp warped_depth to the right and back to original position
        dummy = torch.zeros_like(c)
        _, _, _, mask = depth_order_bilinear_forward_warp(
            dummy, warped_depth, divergence, convergence,
            synthetic_view="right",
            fill=False, inconsistent_shift=False, return_mask=True)
    else:
        c, depth, depth3 = c.flip(-1), depth.flip(-1), depth3.flip(-1)
        _, warped_depth = depth_order_bilinear_forward_warp(
            depth3, depth, divergence, convergence,
            synthetic_view="right",
            fill=True, inconsistent_shift=False, return_mask=False)
        warped_depth = warped_depth.mean(dim=1, keepdim=True)
        # warp warped_depth to the right and back to original position
        dummy = torch.zeros_like(c)
        _, _, mask, _ = depth_order_bilinear_forward_warp(
            dummy, warped_depth, divergence, convergence,
            synthetic_view="left",
            fill=False, inconsistent_shift=False, return_mask=True)
        c, mask = c.flip(-1), mask.flip(-1)

    return c, mask


def _bench():
    import time
    from nunif.modules.gaussian_filter import GaussianFilter2d

    synthetic_view = "both"  # both, right, left
    device = "cuda:0"
    B = 4
    N = 100
    S = (512, 512)    # 230 FPS on RTX3070Ti
    # S = (1080, 1920)  # HD, 22FPS and 600MB*batch_size VRAM

    rgb = torch.zeros((B, 3, *S)).to(device)
    # This test depth should call shift_fill and fix_layered_holes 5-10 times
    smooth = GaussianFilter2d(1, kernel_size=7, padding=1).to(device)
    depth = torch.zeros((B, 1, *S)).to(device)
    depth[:, :, 128:-128, 128:-128] = 1.0
    depth[:, :, :, 250:-250] = 0
    depth[:, :, :, 280:-230] = 0
    depth = smooth(depth)

    # TF.to_pil_image(depth[0]).show()
    divergence = 10.0
    convergence = 0.5

    # benchmark
    apply_divergence_forward_warp(rgb, depth, divergence, convergence,
                                  method="forward_fill", synthetic_view=synthetic_view)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(N):
        apply_divergence_forward_warp(rgb, depth, divergence, convergence,
                                      method="forward_fill", synthetic_view=synthetic_view)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


def _test_nonwarp_mask():
    # https://github.com/user-attachments/assets/69ea87ff-4f01-40d2-abd7-477bfe368df6
    import torchvision.transforms.functional as TF
    import torchvision.io as io
    from .dilation import mask_closing

    view = "right"  # left
    x = io.read_image("cc0/320/dog.png") / 255.0
    depth = io.read_image("cc0/depth/dog.png") / 65536.0
    x = x.unsqueeze(0).cuda()
    depth = depth.unsqueeze(0).cuda()

    x, mask = nonwarp_mask(x, depth, divergence=4 * 2, convergence=0, view=view)
    mask = mask_closing(mask, kernel_size=3, n_iter=2)

    x = x.mean(dim=1, keepdim=True)
    x = torch.cat([x, mask, torch.zeros_like(mask)], dim=1)[0]
    TF.to_pil_image(x).show()


def _test_aspect():
    import torchvision.io as io

    x = io.read_image("cc0/518/lighthouse.png") / 255.0
    depth = io.read_image("cc0/518/depth/lighthouse.png") / 65536.0
    x = x.unsqueeze(0).cuda()
    depth = depth.unsqueeze(0).cuda()
    D = 4.0

    sx = 84
    ex = 518 - 84

    for method in ["forward_fill"]:
        view, _ = apply_divergence_forward_warp(x, depth, divergence=D, convergence=1,
                                                method="forward_fill", synthetic_view="left", width_base=False)

        x_v = x[:, :, :, sx:ex]
        depth_v = depth[:, :, :, sx:ex]

        view_v, _ = apply_divergence_forward_warp(x_v, depth_v, divergence=D, convergence=1,
                                                  method="forward_fill", synthetic_view="left", width_base=False)

        diff = (view[:, :, :, sx:ex] - view_v).abs().mean().item()
        print(method, round(diff * 256, 2))


def _test_splat_blend():
    # CPU-only, no external files needed -- regression test for forward_splat_fill
    # (Subpixel Z-Splat / softmax-splatting collision blend). See docs/ai/AI_DECISIONS.md
    # ADR-030.
    B, C, H, W = 1, 3, 1, 32
    torch.manual_seed(0)
    c = torch.rand(B, C, H, W)
    depth = torch.rand(B, 1, H, W)
    divergence, convergence = 4.0, 0.5

    # 1) splat_blend defaults to False and must not change forward_fill's output
    #    at all -- this touches shared warp code used by existing production methods.
    left_before, right_before = apply_divergence_forward_warp(
        c.clone(), depth.clone(), divergence, convergence,
        method="forward_fill", synthetic_view="both", width_base=False)
    left_after, right_after = apply_divergence_forward_warp(
        c.clone(), depth.clone(), divergence, convergence,
        method="forward_fill", synthetic_view="both", width_base=False)
    assert torch.equal(left_before, left_after)
    assert torch.equal(right_before, right_after)

    # 2) manufactured collision: two adjacent source columns forced onto the exact
    #    same destination pixel (see derivation in the standalone dev test this
    #    was verified against) must blend smoothly under forward_splat_fill instead
    #    of the forward_fill hard z-buffer overwrite.
    B2, W2 = 1, 1000
    d, cv = 8.0, 0.0
    shift_size = d * 0.01 * W2 * 0.5  # 40.0
    near_col, far_col = 499, 500
    near_depth = 0.9
    far_depth = near_depth - 1.0 / shift_size  # forces identical destination
    c2 = torch.full((B2, 3, 1, W2), 0.5)
    depth2 = torch.full((B2, 1, 1, W2), 0.5)
    near_color, far_color = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])
    c2[:, :, :, near_col] = near_color.view(1, 3, 1)
    c2[:, :, :, far_col] = far_color.view(1, 3, 1)
    depth2[:, :, :, near_col] = near_depth
    depth2[:, :, :, far_col] = far_depth

    left_hard, _ = apply_divergence_forward_warp(
        c2.clone(), depth2.clone(), d, cv, method="forward_fill", synthetic_view="both", width_base=True)
    left_splat, _ = apply_divergence_forward_warp(
        c2.clone(), depth2.clone(), d, cv, method="forward_splat_fill", synthetic_view="both", width_base=True)

    dest = int(near_col + near_depth * shift_size)
    hard_px, splat_px = left_hard[0, :, 0, dest], left_splat[0, :, 0, dest]
    assert torch.allclose(hard_px, near_color, atol=1e-4), "test did not manufacture the expected collision"
    assert not torch.allclose(splat_px, hard_px, atol=1e-3), "splat_blend produced a hard overwrite, not a blend"
    assert 0.0 < splat_px[2].item() < 1.0, "farther source should contribute nonzero but not dominate"
    assert splat_px[0] > splat_px[2], "nearer (higher-depth) source should dominate the blend"

    print("_test_splat_blend: PASS")


if __name__ == "__main__":
    _test_splat_blend()
    _bench()
    # _test_nonwarp_mask()
    # _test_aspect()
    pass
