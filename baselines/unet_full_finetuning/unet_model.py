"""UNet (ResNet50 Encoder) Full Fine-Tuning - Model Definition"""

import torch.nn as nn
import segmentation_models_pytorch as smp

from configs.config import ENCODER, NUM_CLASSES, DROPOUT, DEVICE


def build_model():
    """UNet with ResNet50 encoder + dropout for regularisation."""
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
        decoder_use_batchnorm=True,
    )

    # Insert dropout into decoder blocks
    for block in model.decoder.blocks:
        block.conv2 = nn.Sequential(
            block.conv2,
            nn.Dropout2d(p=DROPOUT),
        )

    return model.to(DEVICE)


if __name__ == "__main__":
    m = build_model()
    print(f"UNet ResNet50 parameters: {sum(p.numel() for p in m.parameters()):,}")
    del m
