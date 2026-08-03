"""DeepLabV3+ (ResNet50 Encoder) Full Fine-Tuning - Model Definition"""

import segmentation_models_pytorch as smp

from configs.config import ENCODER, NUM_CLASSES, DEVICE


def build_model():
    """DeepLabV3+ with ResNet50 encoder, ImageNet pretrained, fully trainable."""
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model.to(DEVICE)


if __name__ == "__main__":
    m = build_model()
    print(f"DeepLabV3+ ResNet50 parameters: {sum(p.numel() for p in m.parameters()):,}")
    del m
