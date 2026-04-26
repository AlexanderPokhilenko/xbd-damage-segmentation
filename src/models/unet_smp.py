import segmentation_models_pytorch as smp

from src.models.registry import register_model


@register_model("unet_smp")
def build_unet_smp(encoder_name: str = "resnet34",
                   encoder_weights: str = "imagenet",
                   in_channels: int = 3,
                   classes: int = 1,
                   **kwargs):
    """U-Net baseline via segmentation_models_pytorch with pretrained encoder."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
