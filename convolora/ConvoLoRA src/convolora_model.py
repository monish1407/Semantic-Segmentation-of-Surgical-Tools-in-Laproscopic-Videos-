# baseline_model.py

from typing import List

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import loralib as lora

from .baseline_config import CFG


def freeze_bn(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def _replace_conv_with_lora(parent, child_name: str, old_conv: nn.Conv2d, rank: int):
    new_conv = lora.Conv2d(
        old_conv.in_channels,
        old_conv.out_channels,
        old_conv.kernel_size[0],
        r=rank,
        stride=old_conv.stride[0],
        padding=old_conv.padding[0],
        dilation=old_conv.dilation[0],
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.conv.weight.copy_(old_conv.weight)
        if old_conv.bias is not None:
            new_conv.conv.bias.copy_(old_conv.bias)
    setattr(parent, child_name, new_conv)


def _inject_lora_conv2d_recursive(module: nn.Module, rank: int):
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            _replace_conv_with_lora(module, child_name, child, rank)
        else:
            _inject_lora_conv2d_recursive(child, rank)


def build_convolora_unet(rank: int | None = None) -> nn.Module:
    if rank is None:
        rank = CFG["lora_rank"]

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    for p in model.encoder.parameters():
        p.requires_grad = False

    _inject_lora_conv2d_recursive(model.encoder.layer3, rank)
    _inject_lora_conv2d_recursive(model.encoder.layer4, rank)

    lora.mark_only_lora_as_trainable(model.encoder)

    for m in model.encoder.modules():
        if isinstance(m, lora.Conv2d):
            m.conv.weight.requires_grad = False
            if m.conv.bias is not None:
                m.conv.bias.requires_grad = False

    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in model.segmentation_head.parameters():
        p.requires_grad = True

    freeze_bn(model.encoder)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable : {trainable:,}")
    print(f"Total     : {total:,}")
    print(f"Ratio     : {100*trainable/total:.2f}%")

    lora_names: List[str] = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    print(f"\nEncoder trainable tensors: {len(lora_names)}")
    print("Sample names:", lora_names[:12])

    return model.to(CFG["device"])


__all__ = ["build_convolora_unet", "freeze_bn"]
