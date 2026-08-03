"""Baseline UNet ResNet50 (Encoder Frozen) - Model Definition"""

import segmentation_models_pytorch as smp

from configs.config import CFG
from utils.utils import freeze_bn


def build_baseline(pretrained=True):
    """UNet with ResNet50 encoder frozen entirely (weights + BN stats)."""
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    for p in model.encoder.parameters():
        p.requires_grad = False

    # BN running mean/var must be frozen after requires_grad=False
    freeze_bn(model.encoder)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ratio={100*trainable/total:.2f}%")

    return model.to(CFG["device"])


if __name__ == "__main__":
    m = build_baseline()
    enc_grads = sum(p.requires_grad for p in m.encoder.parameters())
    dec_grads = sum(p.requires_grad for p in m.decoder.parameters())
    print(f"Encoder trainable params: {enc_grads} (should be 0)")
    print(f"Decoder trainable params: {dec_grads} (should be > 0)")
