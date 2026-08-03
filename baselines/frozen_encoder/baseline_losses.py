"""Baseline UNet ResNet50 (Encoder Frozen) - Loss Functions"""

import torch

from configs.config import CFG


def dice_loss(logits, targets, eps=1.0):
    probs = torch.sigmoid(logits).view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


def dynamic_wbce_loss(logits, targets, eps=1e-6):
    """Per-image dynamic weight = neg_pixels / pos_pixels, capped at 50."""
    probs = torch.sigmoid(logits).clamp(eps, 1 - eps)
    p_flat = probs.view(probs.size(0), -1)
    t_flat = targets.view(targets.size(0), -1)
    losses = []
    for p, t in zip(p_flat, t_flat):
        bce = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
        pos = t.sum()
        if pos > 0:
            neg = t.numel() - pos
            w = torch.ones_like(t)
            w[t == 0] = (neg / pos).clamp(max=50.0)
            bce = bce * w
        # uniform weight (pure BG); FP penalty handles background frames separately
        losses.append(bce.mean())
    return torch.stack(losses)  # shape (B,)


def combined_loss(logits, targets):
    """Tool frames: alpha*Dice + (1-alpha)*Dynamic-WBCE. BG frames: FP penalty
    (mean sigmoid(logits) -> 0 when model is silent, grows with false positives).
    """
    has_fg = (targets.sum(dim=(1, 2, 3)) > 0)  # (B,) bool
    l_wbce = dynamic_wbce_loss(logits, targets)  # (B,)

    if has_fg.any():
        l_dice = dice_loss(logits[has_fg], targets[has_fg])
        loss_fg = CFG["alpha"] * l_dice + (1 - CFG["alpha"]) * l_wbce[has_fg].mean()
    else:
        loss_fg = torch.tensor(0.0, device=logits.device)

    if (~has_fg).any():
        bg_probs = torch.sigmoid(logits[~has_fg])
        fp_penalty = bg_probs.mean()  # 0 when model stays quiet
        loss_bg = fp_penalty * CFG["fp_penalty_w"]
    else:
        loss_bg = torch.tensor(0.0, device=logits.device)

    n_fg, n_bg = has_fg.sum().item(), (~has_fg).sum().item()
    return (loss_fg * n_fg + loss_bg * n_bg) / (n_fg + n_bg + 1e-6)


def dice_score(logits, targets, threshold=0.5, eps=1.0):
    preds = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return (2 * inter + eps) / (union + eps)  # per-sample (B,)


def iou_score(logits, targets, threshold=0.5, eps=1.0):
    preds = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    return (inter + eps) / (union + eps)  # per-sample (B,)
