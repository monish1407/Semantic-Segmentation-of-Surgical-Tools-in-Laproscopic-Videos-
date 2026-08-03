# baseline_evaluate.py

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baseline_config import CFG, SPLITS_DIR, CKPT_DIR
from .baseline_dataset import InstrumentDataset
from .baseline_losses import dice_score, iou_score
from .baseline_model import build_convolora_unet, freeze_bn


def load_best_model_for_fold(fold: int):
    best_ckpt_path = CKPT_DIR / f"fold{fold}_best.pt"
    assert best_ckpt_path.exists(), f"Best checkpoint not found: {best_ckpt_path}"

    best_state = torch.load(best_ckpt_path, map_location=CFG["device"])

    model = build_convolora_unet()
    model.load_state_dict(best_state["model_state"])
    model.eval()
    freeze_bn(model.encoder)

    print(
        f"Loaded best checkpoint for fold {fold} "
        f"(epoch {best_state['epoch']}, dice={best_state['best_val_dice']:.4f})"
    )
    return model, best_state


def evaluate_test_set(fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nRunning test set evaluation for fold {fold} ...")

    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    print(f"Test frames: {len(test_df)}")
    print(test_df["group"].value_counts())

    test_ds = InstrumentDataset(test_df, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
    )

    model, best_state = load_best_model_for_fold(fold)

    model.eval()
    freeze_bn(model.encoder)

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(CFG["device"])
            masks = batch["mask"].to(CFG["device"])
            groups = batch["group"]
            isbg = batch["isbackground"]
            ids = batch["id"]

            logits = model(images)
            d_per = dice_score(logits, masks)
            i_per = iou_score(logits, masks)
            preds = (torch.sigmoid(logits) >= 0.5).float()

            for ib in range(len(ids)):
                bg_flag = bool(isbg[ib])
                rows.append({
                    "id": ids[ib],
                    "group": groups[ib],
                    "isbackground": bg_flag,
                    "dice": d_per[ib].item(),
                    "iou": i_per[ib].item(),
                    "fp": bool(preds[ib].sum() > 0) if bg_flag else False,
                })

    results_df = pd.DataFrame(rows)

    def _group_stats(df, label):
        n = len(df)
        dice = df["dice"].mean() if n > 0 else float("nan")
        iou = df["iou"].mean() if n > 0 else float("nan")
        bg_rows = df[df["isbackground"] == True]
        fp_rate = bg_rows["fp"].mean() if len(bg_rows) > 0 else float("nan")
        return {"group": label, "n": n, "Dice": dice, "IoU": iou, "FP-bg": fp_rate}

    tool_df = results_df[results_df["isbackground"] == False]
    bg_df_ = results_df[results_df["isbackground"] == True]

    table_df = pd.DataFrame([
        _group_stats(results_df, "overall"),
        _group_stats(tool_df[tool_df["group"] == "majority"], "majority"),
        _group_stats(tool_df[tool_df["group"] == "minority"], "minority"),
        _group_stats(tool_df[tool_df["group"] == "tip_dominant"], "tip_dominant"),
        _group_stats(bg_df_, "background"),
    ])

    print("\n" + "─" * 62)
    print(f"Test Results — Fold {fold} (ConvLoRA)")
    print("─" * 62)
    print(f" {'Group':<14} {'n':>5} {'Dice':>7} {'IoU':>7} {'FP-bg':>7}")
    print(f" {'─'*54}")
    for _, r in table_df.iterrows():
        fp_str = f"{r['FP-bg']:.3f}" if not np.isnan(r["FP-bg"]) else " n/a"
        print(f" {r['group']:<14} {int(r['n']):>5} {r['Dice']:>7.4f} {r['IoU']:>7.4f} {fp_str:>7}")
    print("─" * 62)

    table_df.to_csv(CKPT_DIR / f"fold{fold}_test_results.csv", index=False)
    results_df.to_csv(CKPT_DIR / f"fold{fold}_test_results_per_sample.csv", index=False)

    print(f"Saved → fold{fold}_test_results.csv")
    print(f"Saved → fold{fold}_test_results_per_sample.csv")

    return table_df, results_df


__all__ = ["load_best_model_for_fold", "evaluate_test_set"]
