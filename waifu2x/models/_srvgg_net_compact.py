import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model


@register_model
class SRVGGNetCompact(I2IBaseModel):
    """Compact VGG-style super-resolution network -- the architecture behind
    realesr-general-x4v3.pth (Real-ESRGAN's lightweight "general" upscaler).
    Loads external, independently pretrained weights rather than being trained
    by this project (see docs/ai/AI_DECISIONS.md for where the weights came from
    and waifu2x/realesrgan.py for how they're loaded).

    Original implementation, not derived from reading any specific project's
    source: the exact layer count (num_conv=32) and structure below was
    determined empirically by inspecting the real checkpoint's own tensor keys
    and shapes one by one (body.0/body.1 = first conv+PReLU, body.2..body.65 =
    32 more conv+PReLU pairs, body.66 = final conv producing 48=3*4*4 channels
    for a 4x PixelShuffle) and cross-checked against the published architecture
    description (a sequence of Conv+PReLU blocks predicting a residual added on
    top of a plain nearest-neighbor-upsampled base image, rather than the full
    output directly)."""
    name = "waifu2x.srvgg_net_compact"

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, scale=4):
        super().__init__(locals(), scale=scale, offset=0, in_channels=num_in_ch,
                          default_tile_size=256, default_batch_size=1)
        self.scale = scale
        layers = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            layers.append(nn.PReLU(num_parameters=num_feat))
        layers.append(nn.Conv2d(num_feat, num_out_ch * (scale ** 2), 3, 1, 1))
        self.body = nn.Sequential(*layers)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        out = self.body(x)
        out = self.upsampler(out)
        # The network predicts a residual on top of a naive nearest-neighbor
        # upsample of the input, not the full output directly -- part of the
        # published "compact" design, required to reproduce correct output from
        # these pretrained weights.
        base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        out = out + base
        if self.training:
            return out
        return torch.clamp(out, 0., 1.)
