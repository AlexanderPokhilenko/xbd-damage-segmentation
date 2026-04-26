"""Reusable blocks from the original BAFUNet/FusionUNet/ResUNet++ implementations."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str = "ReLU") -> nn.Module:
    if hasattr(nn, name):
        return getattr(nn, name)()
    return nn.ReLU()


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation="ReLU"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def _make_nConv(in_channels, out_channels, nb_Conv, activation="ReLU"):
    layers = [ConvBatchNorm(in_channels, out_channels, activation)]
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """Channel-wise Cross Attention.
    
    Computes channel attention from globally-pooled g and x, then applies it to x.
    F_g and F_x can differ; output channel count equals F_x.
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(Flatten(), nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return self.relu(x * scale)


class ECA(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


def reshape_downsample(x):
    b, c, h, w = x.shape
    ret = x.new_zeros(b, c * 4, h // 2, w // 2)
    ret[:, 0::4, :, :] = x[:, :, 0::2, 0::2]
    ret[:, 1::4, :, :] = x[:, :, 0::2, 1::2]
    ret[:, 2::4, :, :] = x[:, :, 1::2, 0::2]
    ret[:, 3::4, :, :] = x[:, :, 1::2, 1::2]
    return ret


def reshape_upsample(x):
    b, c, h, w = x.shape
    assert c % 4 == 0, "channels must be multiple of 4"
    ret = x.new_zeros(b, c // 4, h * 2, w * 2)
    ret[:, :, 0::2, 0::2] = x[:, 0::4, :, :]
    ret[:, :, 0::2, 1::2] = x[:, 1::4, :, :]
    ret[:, :, 1::2, 0::2] = x[:, 2::4, :, :]
    ret[:, :, 1::2, 1::2] = x[:, 3::4, :, :]
    return ret


class DownFuseBlock(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.eca = ECA(base_channels * 2)
        self.conv1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, groups=base_channels)
        self.norm1 = nn.BatchNorm2d(base_channels * 2)
        self.fuse_conv = ConvBatchNorm(base_channels * 2, base_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, fp1, fp2):
        down = reshape_downsample(fp1)
        down = self.relu(self.norm1(self.conv1(down)))
        fp2 = self.fuse_conv(fp2 * 0.75 + down * 0.25) + fp2
        return self.eca(fp2)


class UpFuseBlock(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.eca = ECA(base_channels)
        self.conv1 = nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=1,
                               padding=1, groups=base_channels // 2)
        self.norm1 = nn.BatchNorm2d(base_channels)
        self.fuse_conv = ConvBatchNorm(base_channels, base_channels)
        self.relu = nn.ReLU()

    def forward(self, fp1, fp2):
        up = reshape_upsample(fp2)
        up = self.relu(self.norm1(self.conv1(up)))
        fp1 = self.fuse_conv(fp1 * 0.75 + up * 0.25) + fp1
        return self.eca(fp1)


class FuseBlock(nn.Module):
    """One round of two-way fusion across 4 feature levels (channels: c, 2c, 4c, 8c).

    Levels must have channel ratios 1:2:4:8 and spatial ratios 1:1/2:1/4:1/8.
    """
    def __init__(self, base_channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(base_channels)
        self.norm2 = nn.BatchNorm2d(base_channels * 2)
        self.norm3 = nn.BatchNorm2d(base_channels * 4)
        self.norm4 = nn.BatchNorm2d(base_channels * 8)

        self.up3 = UpFuseBlock(base_channels=base_channels * 4)
        self.up2 = UpFuseBlock(base_channels=base_channels * 2)
        self.up1 = UpFuseBlock(base_channels=base_channels)

        self.down1 = DownFuseBlock(base_channels=base_channels)
        self.down2 = DownFuseBlock(base_channels=base_channels * 2)
        self.down3 = DownFuseBlock(base_channels=base_channels * 4)

    def forward(self, fp1, fp2, fp3, fp4):
        fp4 = self.norm4(fp4)
        fp3 = self.norm3(fp3)
        fp2 = self.norm2(fp2)
        fp1 = self.norm1(fp1)
        fp2 = self.down1(fp1, fp2)
        fp3 = self.down2(fp2, fp3)
        fp4 = self.down3(fp3, fp4)
        fp3 = self.up3(fp3, fp4)
        fp2 = self.up2(fp2, fp3)
        fp1 = self.up1(fp1, fp2)
        return fp1, fp2, fp3, fp4


class FuseModule(nn.Module):
    def __init__(self, base_channel, nb_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [FuseBlock(base_channel) for _ in range(max(1, nb_blocks))]
        )

    def forward(self, fp1, fp2, fp3, fp4):
        for blk in self.blocks:
            fp1, fp2, fp3, fp4 = blk(fp1, fp2, fp3, fp4)
        return fp1, fp2, fp3, fp4


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling from ResUNet++."""
    def __init__(self, in_dims, out_dims, rate=(6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=r, dilation=r),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_dims),
            ) for r in rate
        ])
        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        out = torch.cat([blk(x) for blk in self.blocks], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UpBlockAttention(nn.Module):
    """Upsample decoder feature, apply CCA-attention on skip, concat, then conv block.
    
    Args:
        decoder_channels: channels of the deeper decoder input (will be upsampled).
        skip_channels: channels of the skip connection from encoder.
        out_channels: channels after the conv block.
    """
    def __init__(self, decoder_channels, skip_channels, out_channels, nb_Conv=2,
                 activation="ReLU"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.coatt = CCA(F_g=decoder_channels, F_x=skip_channels)
        self.nConvs = _make_nConv(decoder_channels + skip_channels, out_channels,
                                  nb_Conv, activation)

    def forward(self, x, skip_x):
        """x: deeper decoder feature; skip_x: encoder skip."""
        up = self.up(x)
        skip_att = self.coatt(g=up, x=skip_x)
        return self.nConvs(torch.cat([skip_att, up], dim=1))
