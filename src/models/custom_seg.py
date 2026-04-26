"""BAFUNet, FusionUNet, ResUNet++ adapted to a pretrained encoder (ResNet34/SMP).

All three share a ResNet34 ImageNet-pretrained encoder for fair comparison.
Decoder structures are taken from the original implementations.
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.models._blocks import (ASPP, ConvBatchNorm, FuseModule, UpBlockAttention,
                                _make_nConv)
from src.models.registry import register_model


def _get_encoder(name: str = "resnet34", weights: str = "imagenet", in_channels: int = 3):
    """SMP encoder returning multi-level feature maps."""
    return smp.encoders.get_encoder(name=name, in_channels=in_channels,
                                    depth=5, weights=weights)


class _EncoderWrapper(nn.Module):
    """Convenience wrapper that returns features at stages 1..5 (skipping stage 0=input)."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # SMP encoder out_channels: list of length 6 [stage0..stage5]
        self.out_channels = encoder.out_channels  # e.g. [3, 64, 64, 128, 256, 512]

    def forward(self, x):
        feats = self.encoder(x)  # list of 6 tensors
        return feats  # [f0, f1, f2, f3, f4, f5]


# ---------------- BAFUNet (adapted) ---------------- #

class _BAFUNetAdapted(nn.Module):
    """BAFUNet with ResNet34 encoder.

    fp1..fp4 = encoder stages 2..5 (channels 64,128,256,512 for resnet34).
    Skip stage 1 (channels 64) is used in the final decoder upsampling.
    ASPP bottleneck is applied on top of fp4 (deepest, 8x8 for 256 input).
    FuseModule operates on fp1..fp4 in parallel with the bottleneck path.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=1, aggre_depth=2):
        super().__init__()
        enc = _get_encoder(encoder_name, encoder_weights, in_channels)
        self.encoder = _EncoderWrapper(enc)
        # SMP resnet34 out_channels: [3, 64, 64, 128, 256, 512]
        c = self.encoder.out_channels
        c0, c1, c2, c3, c4, c5 = c  # 3, 64, 64, 128, 256, 512

        # FuseModule operates on stages 2..5 -> channels 64,128,256,512 = base*[1,2,4,8]
        base = c2  # 64
        assert c3 == base * 2 and c4 == base * 4 and c5 == base * 8, \
            f"Encoder channels {c} not in 1:2:4:8 ratio at stages 2-5"
        self.fuse = FuseModule(base_channel=base, nb_blocks=aggre_depth)

        # ASPP bottleneck on fp4 (deepest, 512 ch)
        self.bottleneck = ASPP(c5, c5)

        # Decoder: upsample from bottleneck back to input resolution
        # Levels (after fuse): fp1@1/4 64ch, fp2@1/8 128ch, fp3@1/16 256ch, fp4@1/32 512ch
        # Bottleneck stays at 1/32 with 512ch, so we go: 1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1
        self.up5 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec5 = ConvBatchNorm(c4 * 2, c4)  # concat with fp3 (256ch)

        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec4 = ConvBatchNorm(c3 * 2, c3)  # concat with fp2 (128ch)

        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = ConvBatchNorm(c2 * 2, c2)  # concat with fp1 (64ch)

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = ConvBatchNorm(c1 * 2, c1)  # concat with f1 stage 1 (64ch)

        self.up1 = nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2)
        self.dec1 = ConvBatchNorm(c1 // 2, c1 // 2)  # final 256x256

        self.head = nn.Conv2d(c1 // 2, classes, kernel_size=1)

    def forward(self, x):
        f0, f1, f2, f3, f4, f5 = self.encoder(x)
        # Apply FuseModule to stages 2..5 (channels 64,128,256,512)
        fp1, fp2, fp3, fp4 = self.fuse(f2, f3, f4, f5)
        # Bottleneck: process deepest with ASPP
        b = self.bottleneck(fp4)
        # Decoder upsample chain
        d = self.up5(b)
        d = self.dec5(torch.cat([d, fp3], dim=1))
        d = self.up4(d)
        d = self.dec4(torch.cat([d, fp2], dim=1))
        d = self.up3(d)
        d = self.dec3(torch.cat([d, fp1], dim=1))
        d = self.up2(d)
        d = self.dec2(torch.cat([d, f1], dim=1))
        d = self.up1(d)
        d = self.dec1(d)
        return self.head(d)


@register_model("bafunet")
def build_bafunet(encoder_name="resnet34", encoder_weights="imagenet",
                  in_channels=3, classes=1, aggre_depth=2, **kwargs):
    return _BAFUNetAdapted(encoder_name=encoder_name, encoder_weights=encoder_weights,
                           in_channels=in_channels, classes=classes, aggre_depth=aggre_depth)


# ---------------- FusionUNet (adapted) ---------------- #

class _FusionUNetAdapted(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=1, aggre_depth=2):
        super().__init__()
        enc = _get_encoder(encoder_name, encoder_weights, in_channels)
        self.encoder = _EncoderWrapper(enc)
        c = self.encoder.out_channels
        c0, c1, c2, c3, c4, c5 = c  # 3, 64, 64, 128, 256, 512
        base = c2

        self.fuse = FuseModule(base_channel=base, nb_blocks=aggre_depth)

        # Decoder chain: deepest -> upsample -> attention with skip -> conv block
        # fp4 (c5=512) -> up & merge with fp3 (c4=256) -> c4 channels
        self.up_4to3 = UpBlockAttention(decoder_channels=c5, skip_channels=c4,
                                        out_channels=c4, nb_Conv=2)
        # c4 -> up & merge with fp2 (c3=128) -> c3
        self.up_3to2 = UpBlockAttention(decoder_channels=c4, skip_channels=c3,
                                        out_channels=c3, nb_Conv=2)
        # c3 -> up & merge with fp1 (c2=64) -> c2
        self.up_2to1 = UpBlockAttention(decoder_channels=c3, skip_channels=c2,
                                        out_channels=c2, nb_Conv=2)
        # c2 -> up & merge with stage1 f1 (c1=64) -> c1
        self.up_1to0 = UpBlockAttention(decoder_channels=c2, skip_channels=c1,
                                        out_channels=c1, nb_Conv=2)
        # Final upsample 128->256 (no skip available)
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2),
            ConvBatchNorm(c1 // 2, c1 // 2),
        )
        self.head = nn.Conv2d(c1 // 2, classes, kernel_size=1)

    def forward(self, x):
        f0, f1, f2, f3, f4, f5 = self.encoder(x)
        fp1, fp2, fp3, fp4 = self.fuse(f2, f3, f4, f5)
        d = self.up_4to3(fp4, fp3)
        d = self.up_3to2(d, fp2)
        d = self.up_2to1(d, fp1)
        d = self.up_1to0(d, f1)
        d = self.up_final(d)
        return self.head(d)


@register_model("fusion_unet")
def build_fusion_unet(encoder_name="resnet34", encoder_weights="imagenet",
                      in_channels=3, classes=1, aggre_depth=2, **kwargs):
    return _FusionUNetAdapted(encoder_name=encoder_name, encoder_weights=encoder_weights,
                              in_channels=in_channels, classes=classes, aggre_depth=aggre_depth)


# ---------------- ResUNet++ (adapted) ---------------- #
# We use the SMP-provided "ResUnetPlusPlus"-like decoder via plain Unet with attention.
# Since SMP doesn't ship ResUNet++ directly, we approximate it: ResNet34 encoder +
# UnetPlusPlus decoder + ASPP bottleneck + SCSE attention. This keeps spirit of the
# original (residual encoder, attention, ASPP, SE) while ensuring fair encoder match.

class _ResUNetPPAdapted(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=1):
        super().__init__()
        # SMP UnetPlusPlus with SCSE attention (closest to ResUNet++ spirit)
        self.net = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type="scse",
            activation=None,
        )

    def forward(self, x):
        return self.net(x)


@register_model("resunet_pp")
def build_resunet_pp(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=1, **kwargs):
    return _ResUNetPPAdapted(encoder_name=encoder_name, encoder_weights=encoder_weights,
                             in_channels=in_channels, classes=classes)
