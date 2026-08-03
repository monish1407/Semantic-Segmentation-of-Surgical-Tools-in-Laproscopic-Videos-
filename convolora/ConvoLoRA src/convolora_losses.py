# baseline_losses.py

import torch

from .baseline_config import CFG


def dice_loss(logits, targets, eps: float = 1.0):
    probs = torch.sigmoid(logits).view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2 * inter + eps) / (union + eps)).mean()


def dynamic_wbce_loss(logits, targets, eps: float = 1e-6):
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
            w[t > 0] = (neg / pos).clamp(max=50.0)
            bce = bce * w
        losses.append(bce.mean())

    return torch.stack(losses)


def combined_loss(logits, targets):
    has_fg = (targets.sum(dim=[1, 2, 3]) > 0)
    l_wbce = dynamic_wbce_loss(logits, targets)

    if has_fg.any():
        l_dice = dice_loss(logits[has_fg], targets[has_fg])
        loss_fg = CFG["alpha"] * l_dice + (1 - CFG["alpha"]) * l_wbce[has_fg].mean()
    else:
        loss_fg = torch.tensor(0.0, device=logits.device)

    if (~has_fg).any():
        bg_probs = torch.sigmoid(logits[~has_fg])
        fp_penalty = bg_probs.mean()
        loss_bg = fp_penalty * CFG["fp_penalty_w"]
    else:
        loss_bg = torch.tensor(0.0, device=logits.device)

    n_fg = has_fg.sum().item()
    n_bg = (~has_fg).sum().item()
    return (loss_fg * n_fg + loss_bg * n_bg) / (n_fg + n_bg + 1e-6)


def dice_score(logits, targets, threshold: float = 0.5, eps: float = 1.0):
    preds = (torch.sigmoid(logits) >= threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return (2 * inter + eps) / (union + eps)


def iou_score(logits, targets, threshold: float = 0.5, eps: float = 1.0):
    preds = (torch.sigmoid(logits) >= threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    return (inter + eps) / (union + eps)


__all__ = [
    "dice_loss",
    "dynamic_wbce_loss",
    "combined_loss",
    "dice_score",
    "iou_score",
]
