"""Baseline UNet ResNet50 (Encoder Frozen) - 3-Fold Training Loop"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config import CFG
from utils.utils import freeze_bn, save_checkpoint, load_checkpoint, save_best_model
from utils.loader import InstrumentDataset, make_weighted_sampler, load_or_create_splits
from models.baseline_model import build_baseline
from utils.losses import combined_loss, dice_score, iou_score


def normalize_group(g):
    return g.strip().lower().replace("-", "").replace(" ", "")


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        freeze_bn(model.encoder)  # re-freeze BN every epoch after model.train()

    t_start = time.time()
    total_loss = total_dice = total_iou = 0.0
    n_tool_batches = 0
    group_dice = {"majority": [], "minority": [], "tipdominant": []}
    fp_bg = bg_total = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(CFG["device"])
            masks = batch["mask"].to(CFG["device"])
            groups = batch["group"]
            is_bg = batch["is_background"]

            logits = model(images)
            loss = combined_loss(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                d_per = dice_score(logits.detach(), masks)
                i_per = iou_score(logits.detach(), masks)
                preds = (torch.sigmoid(logits.detach()) > 0.5).float()

            tool_idx = [i for i, bg in enumerate(is_bg) if not bg]
            if tool_idx:
                total_dice += d_per[tool_idx].mean().item()
                total_iou += i_per[tool_idx].mean().item()
                n_tool_batches += 1

            for ib, (g, bg_flag) in enumerate(zip(groups, is_bg)):
                if bg_flag:
                    bg_total += 1
                    if preds[ib].sum() > 0:
                        fp_bg += 1
                else:
                    key = normalize_group(g)
                    if key in group_dice:
                        group_dice[key].append(d_per[ib].item())

    elapsed = time.time() - t_start
    n_loss = len(loader)
    n_metrics = max(n_tool_batches, 1)
    gmean = lambda k: float(np.mean(group_dice[k])) if group_dice[k] else float("nan")
    fp_rate = fp_bg / bg_total if bg_total > 0 else float("nan")

    return {
        "loss": total_loss / n_loss,
        "dice": total_dice / n_metrics,
        "iou": total_iou / n_metrics,
        "dice_majority": gmean("majority"),
        "dice_minority": gmean("minority"),
        "dice_tipdominant": gmean("tipdominant"),
        "fp_bg_rate": fp_rate,
        "time": elapsed,
    }


def train_fold(fold, fold_splits):
    train_df, val_df = fold_splits[fold]["train"], fold_splits[fold]["val"]
    train_ds = InstrumentDataset(train_df, augment=True)
    val_ds = InstrumentDataset(val_df, augment=False)

    sampler = make_weighted_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                               num_workers=CFG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                             num_workers=CFG["num_workers"])

    model = build_baseline()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr_decoder"], weight_decay=CFG["wd_decoder"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=CFG["lr_patience"]
    )

    resumed = load_checkpoint(fold, model, optimizer, scheduler)
    if resumed:
        start_epoch, best_val_dice, patience_count, history = resumed
    else:
        start_epoch, best_val_dice, patience_count, history = 1, -1.0, 0, []

    for epoch in range(start_epoch, CFG["num_epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader, optimizer=None)

        print(f"Fold {fold} Epoch {epoch}: "
              f"train_loss={train_metrics['loss']:.4f} "
              f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}")

        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                         **{f"val_{k}": v for k, v in val_metrics.items()}})

        scheduler.step(val_metrics["dice"])

        if val_metrics["dice"] > best_val_dice + CFG["delta"]:
            best_val_dice = val_metrics["dice"]
            patience_count = 0
            save_best_model(fold, model)
        else:
            patience_count += 1

        save_checkpoint(fold, epoch, model, optimizer, scheduler,
                         best_val_dice, patience_count, history)

        if patience_count >= CFG["patience"]:
            print(f"Early stopping fold {fold} at epoch {epoch}")
            break

    return best_val_dice, history


def main():
    fold_splits, _ = load_or_create_splits(CFG["num_folds"])
    for fold in range(CFG["num_folds"]):
        print(f"{'='*55}\n FOLD {fold} (encoder frozen)\n{'='*55}")
        train_fold(fold, fold_splits)


if __name__ == "__main__":
    main()
