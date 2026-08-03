"""UNet (ResNet50 Encoder) Full Fine-Tuning - Loss Functions"""

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def compute_pos_weight(mask_files, image_size, device):
    """Pixel-level class balancing weight for BCE. Clamped to [1, 10]."""
    import cv2
    from tqdm import tqdm

    tool_pixel_counts = []
    for mp in tqdm(mask_files, desc="Scanning masks"):
        m = cv2.imread(mp, 0)
        m = cv2.resize(m, (image_size, image_size))
        m = (m > 10).astype(np.uint8)
        tool_pixel_counts.append(m.sum())

    tool_pixel_counts = np.array(tool_pixel_counts)
    total_pixels = len(mask_files) * image_size * image_size
    tool_pixels = tool_pixel_counts.sum()
    bg_pixels = total_pixels - tool_pixels

    w_tool = total_pixels / (2 * tool_pixels) if tool_pixels > 0 else 1.0
    w_bg = total_pixels / (2 * bg_pixels) if bg_pixels > 0 else 1.0

    raw_pos_weight = w_tool / w_bg
    clamped_pos_weight = float(np.clip(raw_pos_weight, 1.0, 10.0))
    print(f"Class weights: tool={w_tool:.4f} background={w_bg:.4f}")
    print(f"pos_weight raw={raw_pos_weight:.4f} clamped to {clamped_pos_weight:.4f}")
    return torch.tensor(clamped_pos_weight).to(device)


def build_loss_fn(pos_weight):
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    iou_loss = smp.losses.JaccardLoss(mode="binary", from_logits=True)

    def loss_fn(pred, target):
        return 0.5 * bce_loss(pred, target) + 0.5 * iou_loss(pred, target)

    return loss_fn
