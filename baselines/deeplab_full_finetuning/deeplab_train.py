"""DeepLabV3+ (ResNet50 Encoder) Full Fine-Tuning - Training Loop"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

from configs.config import (
    IMAGE_PATH, MASK_PATH, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    CKPT_DIR, N_FOLDS, PATIENCE_CV, SEED, DEVICE,
)
from utils.loader import SurgicalDataset, train_transform, val_transform, get_file_lists
from utils.losses import compute_pos_weight, build_loss_fn
from utils.utils import analyse_dataset, build_weighted_sampler, clear_checkpoints
from models.deeplab_model import build_model
from train.validate import validate_one_epoch


def train_one_fold(train_idx, val_idx, fold_num, image_files, mask_files,
                    has_tool, loss_fn, epochs=EPOCHS):
    print("=" * 55)
    print(f" FOLD {fold_num} | train={len(train_idx)} val={len(val_idx)}")
    print("=" * 55)

    ckpt_path = os.path.join(CKPT_DIR, f"fold{fold_num}_latest.pth")

    train_ds = SurgicalDataset(
        [image_files[i] for i in train_idx],
        [mask_files[i] for i in train_idx],
        transform=train_transform, split=f"fold{fold_num}-train", cache=False,
    )
    val_ds = SurgicalDataset(
        [image_files[i] for i in val_idx],
        [mask_files[i] for i in val_idx],
        transform=val_transform, split=f"fold{fold_num}-val", cache=False,
    )

    train_labels = [has_tool[i] for i in train_idx]
    sampler = build_weighted_sampler(train_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_iou = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, loss_fn, DEVICE)
        mean_iou = float(np.nanmean(val_metrics["iou"]))

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_iou={mean_iou:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_iou": mean_iou})

        torch.save({"model_state": model.state_dict(), "epoch": epoch,
                    "val_iou": mean_iou}, ckpt_path)

        if mean_iou > best_iou:
            best_iou = mean_iou
            patience_counter = 0
            best_path = os.path.join(CKPT_DIR, f"fold{fold_num}_best.pth")
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_iou": mean_iou}, best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_CV:
                print(f"Early stopping fold {fold_num} at epoch {epoch}")
                break

    return best_iou, history


def main():
    image_files, mask_files = get_file_lists(IMAGE_PATH, MASK_PATH)
    _, has_tool = analyse_dataset(mask_files, IMAGE_SIZE)
    pos_weight = compute_pos_weight(mask_files, IMAGE_SIZE, DEVICE)
    loss_fn = build_loss_fn(pos_weight)

    clear_checkpoints(CKPT_DIR)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(image_files), start=1):
        best_iou, history = train_one_fold(
            train_idx, val_idx, fold_num, image_files, mask_files, list(has_tool), loss_fn
        )
        fold_results.append(best_iou)

    print(f"Mean IoU across folds: {np.mean(fold_results):.4f} "
          f"+/- {np.std(fold_results):.4f}")


if __name__ == "__main__":
    main()
