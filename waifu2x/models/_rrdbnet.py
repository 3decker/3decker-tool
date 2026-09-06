import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model


class _ResidualDenseBlock(nn.Module):
    """5-conv residual dense block -- the core building block of the RRDBNet/
    ESRGAN-family super-resolution architecture (Wang et al., ESRGAN/Real-ESRGAN
    papers). Original implementation written from the published architecture
    description and verified against real pretrained checkpoints' tensor shapes
    (RealESRGAN_x2plus/x4plus, BSRGAN/BSRGANx2 -- see docs/ai/AI_DECISIONS.md for
    where those weights came from) -- not derived from reading any specific
    project's source code."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # 0.2 residual scaling -- part of the published architecture, stabilizes
        # training (irrelevant for inference-only use here, but must match to
        # produce numerically correct output from pretrained weights).
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    """Residual in Residual Dense Block: 3 chained _ResidualDenseBlocks."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


@register_model
class RRDBNet(I2IBaseModel):
    """RRDBNet -- the architecture behind RealESRGAN_x2plus.pth, RealESRGAN_x4plus.pth,
    and BSRGAN.pth/BSRGANx2.pth (BSRGAN uses the original ESRGAN naming convention for
    its layers -- see waifu2x/models/_bsrgan_keys.py for the key-remapping used to load
    it into this same module). Loads external, independently pretrained weights rather
    than being trained by this project; use `create_model` + `load_state_dict` directly
    (NOT nunif's own `load_model`, which expects nunif's own serialization format) to
    load a raw external .pth state dict -- see waifu2x/realesrgan.py.

    scale=2 support -- two genuinely different pretrained variants exist, both
    handled here via explicit flags rather than guessing from `scale` alone
    (confirmed empirically, not assumed upfront -- BSRGANx2.pth's actual checkpoint
    keys/shapes only became clear after a first attempt at loading it failed):
      - Real-ESRGAN's own x2plus reuses the exact x4 network internally by
        space-to-depth'ing (pixel_unshuffle) the input 2x first, then still running
        BOTH upsample stages -- confirmed necessary since RealESRGAN_x2plus.pth and
        RealESRGAN_x4plus.pth have IDENTICAL key names/shapes (702 tensors each).
      - BSRGANx2.pth (original ESRGAN-style x2) is structurally simpler: plain
        3-channel input (no pixel_unshuffle) and only ONE upsample stage
        (conv_up2/upconv2 doesn't exist in this checkpoint at all)."""
    name = "waifu2x.rrdbnet"

    def __init__(self, scale=4, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32,
                 pixel_unshuffle_input=None, num_upsample=None):
        super().__init__(locals(), scale=scale, offset=0, in_channels=num_in_ch,
                          default_tile_size=256, default_batch_size=1)
        self.scale = scale
        # Auto-derive Real-ESRGAN's own convention (pixel-unshuffle + still 2
        # upsamples for x2) unless the caller explicitly overrides -- BSRGANx2 needs
        # pixel_unshuffle_input=False (which then also implies num_upsample=1 here,
        # matching that checkpoint's actual, simpler structure).
        if pixel_unshuffle_input is None:
            pixel_unshuffle_input = (scale == 2)
        if num_upsample is None:
            num_upsample = 1 if (scale == 2 and not pixel_unshuffle_input) else 2
        self.pixel_unshuffle_input = pixel_unshuffle_input
        self.num_upsample = num_upsample

        conv_first_in_ch = num_in_ch * 4 if pixel_unshuffle_input else num_in_ch
        self.conv_first = nn.Conv2d(conv_first_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) if num_upsample >= 2 else None
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.pixel_unshuffle_input:
            x = F.pixel_unshuffle(x, downscale_factor=2)
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.conv_up2 is not None:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        if self.training:
            return out
        return torch.clamp(out, 0., 1.)


# Key-name remapping for BSRGAN.pth/BSRGANx2.pth, which use the original ESRGAN
# repo's layer naming instead of Real-ESRGAN's renamed version of the exact same
# architecture (confirmed empirically: same tensor shapes/counts, just different
# key prefixes) -- lets both weight families load into this one module instead of
# needing a near-duplicate second class.
BSRGAN_KEY_RENAME = {
    "RRDB_trunk": "body",
    "trunk_conv": "conv_body",
    "upconv1": "conv_up1",
    "upconv2": "conv_up2",
    "HRconv": "conv_hr",
}
# Within each RRDB block, BSRGAN also capitalizes the 3 inner dense-block names
# (RDB1/RDB2/RDB3) where this module (matching Real-ESRGAN's own naming) uses
# lowercase (rdb1/rdb2/rdb3) -- found by the FIRST remap attempt still failing to
# load with strict=True (missing "body.N.rdb1..." keys, unexpected
# "body.N.RDB1..." keys), not assumed upfront.
BSRGAN_INNER_RENAME = {".RDB1.": ".rdb1.", ".RDB2.": ".rdb2.", ".RDB3.": ".rdb3."}


def remap_bsrgan_state_dict(state_dict):
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in BSRGAN_KEY_RENAME.items():
            if new_k.startswith(old_prefix + "."):
                new_k = new_prefix + new_k[len(old_prefix):]
                break
        for old_inner, new_inner in BSRGAN_INNER_RENAME.items():
            new_k = new_k.replace(old_inner, new_inner)
        remapped[new_k] = v
    return remapped
