"""DeepLabV3+ (ResNet50 Encoder) Full Fine-Tuning - Validation Loop"""

import torch
from tqdm import tqdm

from evaluation.metrics import batch_metrics


def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_metrics = {"iou": [], "hausdorff": [], "msd": [], "asd": []}

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validating"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            running_loss += loss.item() * imgs.size(0)

            batch_res = batch_metrics(preds, masks)
            for k in all_metrics:
                all_metrics[k].extend(batch_res[k])

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss, all_metrics
