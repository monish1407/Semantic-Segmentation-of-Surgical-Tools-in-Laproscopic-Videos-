# baseline_train.py

import time
from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from .baseline_config import CFG, CKPT_DIR
from .baseline_dataset import build_dataloaders
from .baseline_losses import combined_loss, dice_score, iou_score
from .baseline_model import build_convolora_unet, freeze_bn


def _normalize_group(g: str) -> str:
    return g.strip().lower().replace("_", "").replace(" ", "").replace("-", "")


def run_epoch(model, loader, optimizer=None) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train:
        freeze_bn(model.encoder)

    t_start = time.time()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_tool_batches = 0

    group_dice = {"majority": [], "minority": [], "tipdominant": []}
    fp_bg = 0
    bg_total = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(CFG["device"])
            masks = batch["mask"].to(CFG["device"])
            groups = batch["group"]
            isbg = batch["isbackground"]

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
                preds = (torch.sigmoid(logits.detach()) >= 0.5).float()

                tool_idx = [i for i, bg in enumerate(isbg) if not bg]
                if tool_idx:
                    total_dice += d_per[tool_idx].mean().item()
                    total_iou += i_per[tool_idx].mean().item()
                    n_tool_batches += 1

                for ib, (g, bg_flag) in enumerate(zip(groups, isbg)):
                    if bg_flag:
                        bg_total += 1
                        if preds[ib].sum() > 0:
                            fp_bg += 1
                    else:
                        key = _normalize_group(g)
                        if key in group_dice:
                            group_dice[key].append(d_per[ib].item())

    elapsed = time.time() - t_start
    n_loss = len(loader)
    n_metrics = max(n_tool_batches, 1)

    gmean = lambda k: float(np.mean(group_dice[k])) if group_dice[k] else float("nan")
    fp_rate = fp_bg / bg_total if bg_total > 0 else float("nan")

    return dict(
        loss=total_loss / n_loss,
        dice=total_dice / n_metrics,
        iou=total_iou / n_metrics,
        dice_majority=gmean("majority"),
        dice_minority=gmean("minority"),
        dice_tipdominant=gmean("tipdominant"),
        fp_bg_rate=fp_rate,
        time_s=elapsed,
    )


def save_checkpoint(fold, epoch, model, optimizer, scheduler, best_val_dice, patience_count, history):
    state = {
        "fold": fold,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "best_val_dice": best_val_dice,
        "patience_count": patience_count,
        "history": history,
    }
    torch.save(state, CKPT_DIR / f"fold{fold}_resume.pt")


def load_checkpoint(fold, model, optimizer, scheduler):
    path = CKPT_DIR / f"fold{fold}_resume.pt"
    if not path.exists():
        return None

    state = torch.load(path, map_location=CFG["device"])
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optim_state"])
    scheduler.load_state_dict(state["sched_state"])

    print(
        f"↺ Resumed fold {fold} from epoch {state['epoch']} "
        f"(best_dice={state['best_val_dice']:.4f}, patience={state['patience_count']})"
    )

    return (
        state["epoch"] + 1,
        state["best_val_dice"],
        state["patience_count"],
        state["history"],
    )


def save_best_model(fold, model, best_val_dice, epoch, history):
    state = {
        "model_state": model.state_dict(),
        "best_val_dice": best_val_dice,
        "fold": fold,
        "epoch": epoch,
        "history": history,
    }
    torch.save(state, CKPT_DIR / f"fold{fold}_best.pt")
    print(f"★ Best model saved → fold{fold}_best.pt (epoch {epoch}, dice={best_val_dice:.4f})")


def train_single_fold(fold: int, fold_splits):
    print(f"\nTraining fold {fold} only")

    train_loader, val_loader = build_dataloaders(fold_splits, fold)

    model = build_convolora_unet()

    lora_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": 3e-4, "weight_decay": 1e-4},
        {"params": decoder_params, "lr": CFG["lr_decoder"], "weight_decay": CFG["wd_decoder"]},
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=CFG["lr_patience"],
        factor=0.5,
    )

    resume = load_checkpoint(fold, model, optimizer, scheduler)
    if resume is None:
        start_epoch = 1
        best_val_dice = -1.0
        patience_count = 0
        history: list[dict[str, Any]] = []
    else:
        start_epoch, best_val_dice, patience_count, history = resume

    train_start_time = time.time()
    epoch_times: list[float] = []

    for epoch in range(start_epoch, CFG["num_epochs"] + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader, optimizer=None)

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        remaining_epochs = CFG["num_epochs"] - epoch
        est_remaining_hours = (avg_epoch_time * remaining_epochs) / 3600.0

        scheduler.step(val_metrics["dice"])

        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_dice_majority": val_metrics["dice_majority"],
            "val_dice_minority": val_metrics["dice_minority"],
            "val_dice_tipdominant": val_metrics["dice_tipdominant"],
            "val_fp_bg_rate": val_metrics["fp_bg_rate"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"[Fold {fold} | Ep {epoch:02d}] "
            f"Dur: {epoch_duration/60:.1f}m | "
            f"TrLoss: {train_metrics['loss']:.4f} "
            f"ValDice: {val_metrics['dice']:.4f} | "
            f"FP-bg: {val_metrics['fp_bg_rate']:.3f} | "
            f"Est. Rem: {est_remaining_hours:.2f}h"
        )

        if val_metrics["dice"] > best_val_dice + CFG["delta"]:
            best_val_dice = val_metrics["dice"]
            patience_count = 0
            save_best_model(fold, model, best_val_dice, epoch, history)
        else:
            patience_count += 1

        save_checkpoint(
            fold,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_dice,
            patience_count,
            history,
        )

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(CKPT_DIR / f"fold{fold}_history.csv", index=False)

        elapsed_hours = (time.time() - train_start_time) / 3600.0
        if elapsed_hours >= CFG["max_train_hours"]:
            print(f"Stopping early due to Kaggle time limit safeguard ({elapsed_hours:.2f} h)")
            break

        if patience_count >= CFG["patience"]:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"\nFold {fold} training finished.")


__all__ = [
    "run_epoch",
    "save_checkpoint",
    "load_checkpoint",
    "save_best_model",
    "train_single_fold",
]
