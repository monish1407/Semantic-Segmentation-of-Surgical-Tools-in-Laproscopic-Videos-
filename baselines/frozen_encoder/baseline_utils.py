"""Baseline UNet ResNet50 (Encoder Frozen) - Checkpoint & BN Freeze Utils"""

import shutil
import torch
import torch.nn as nn

from configs.config import CKPT_DIR, CFG


def freeze_bn(module):
    """Force all BatchNorm layers into eval mode and freeze their params."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def save_checkpoint(fold, epoch, model, optimizer, scheduler, best_val_dice,
                     patience_count, history):
    """Saves full training state every epoch so a killed Kaggle session can resume."""
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
    torch.save(state, CKPT_DIR / f"fold{fold}_resume.pth")


def load_checkpoint(fold, model, optimizer, scheduler):
    """Loads saved state if a resume checkpoint exists for this fold.

    Returns (start_epoch, best_val_dice, patience_count, history) or None
    for a fresh start.
    """
    path = CKPT_DIR / f"fold{fold}_resume.pth"
    if not path.exists():
        return None
    state = torch.load(path, map_location=CFG["device"])
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optim_state"])
    scheduler.load_state_dict(state["sched_state"])
    print(f"Resumed fold {fold} from epoch {state['epoch']} "
          f"best_dice={state['best_val_dice']:.4f} patience={state['patience_count']}")
    return state["epoch"] + 1, state["best_val_dice"], state["patience_count"], state["history"]


def save_best_model(fold, model):
    """Saves the best val-Dice checkpoint separately; never overwritten by resume."""
    torch.save(model.state_dict(), CKPT_DIR / f"fold{fold}_best.pth")
    print(f"Best model saved: fold{fold}_best.pth")


def restore_previous_run(prev_ckpts, prev_splits, ckpt_dir, splits_dir):
    """Copy checkpoints/splits from a previous dataset version into the working dir."""
    if prev_ckpts.exists():
        for f in sorted(prev_ckpts.iterdir()):
            if f.is_file():
                dst = ckpt_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
    if prev_splits.exists():
        for f in sorted(prev_splits.iterdir()):
            if f.is_file():
                dst = splits_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
