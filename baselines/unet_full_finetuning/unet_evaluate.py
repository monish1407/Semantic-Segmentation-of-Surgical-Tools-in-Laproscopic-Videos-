"""UNet (ResNet50 Encoder) Full Fine-Tuning - Fold Evaluation"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import CKPT_DIR, DEVICE, IMAGE_PATH, MASK_PATH
from utils.loader import SurgicalDataset, val_transform, get_file_lists
from models.unet_model import build_model
from evaluation.metrics import batch_metrics


def evaluate_checkpoint(ckpt_path, image_files, mask_files, val_idx):
    val_ds = SurgicalDataset(
        [image_files[i] for i in val_idx],
        [mask_files[i] for i in val_idx],
        transform=val_transform, split="eval", cache=False,
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_metrics = {"iou": [], "hausdorff": [], "msd": [], "asd": []}
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            batch_res = batch_metrics(preds, masks)
            for k in all_metrics:
                all_metrics[k].extend(batch_res[k])

    summary = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in all_metrics.items()}
    return summary


def main():
    image_files, mask_files = get_file_lists(IMAGE_PATH, MASK_PATH)
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "fold*_best.pth")))
    for ckpt_path in ckpts:
        print(f"Evaluating {ckpt_path}")
        # NOTE: val_idx should be reloaded/persisted from the training split per fold
        # for accurate reproducible evaluation.


if __name__ == "__main__":
    main()
