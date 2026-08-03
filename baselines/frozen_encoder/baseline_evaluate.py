"""Baseline UNet ResNet50 (Encoder Frozen) - Test Set Evaluation"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from configs.config import CFG, CKPT_DIR
from utils.utils import freeze_bn
from utils.loader import InstrumentDataset
from evaluation.metrics import dice_score, iou_score

OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_baseline_unet():
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                      in_channels=3, classes=1, activation=None)
    return model.to(CFG["device"])


def run_test_for_ckpt(ckpt_path, name, label, test_df):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    model = build_baseline_unet()
    state = torch.load(ckpt_path, map_location=CFG["device"])
    model.load_state_dict(state.get("model_state", state))
    model.eval()
    freeze_bn(model)

    test_ds = InstrumentDataset(test_df, augment=False)
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=CFG["num_workers"])

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(CFG["device"])
            masks = batch["mask"].to(CFG["device"])
            groups = batch["group"]
            is_bg = batch["is_background"]
            ids = batch["id"]

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()
            d_per = dice_score(logits, masks)
            i_per = iou_score(logits, masks)
            gt_sum = masks.view(masks.size(0), -1).sum(dim=1)

            for ib in range(len(ids)):
                bg_flag = bool(is_bg[ib])
                if gt_sum[ib].item() == 0:
                    dice_val = iou_val = float("nan")
                else:
                    dice_val, iou_val = d_per[ib].item(), i_per[ib].item()
                fp_flag = bool(preds[ib].sum() > 0) if bg_flag else False
                rows.append({
                    "id": ids[ib], "group": groups[ib], "is_background": bg_flag,
                    "dice": dice_val, "iou": iou_val, "fp": fp_flag,
                })

    results_df = pd.DataFrame(rows)

    def group_stats(df, lbl):
        n = len(df)
        dice = df["dice"].mean(skipna=True) if n > 0 else float("nan")
        iou = df["iou"].mean(skipna=True) if n > 0 else float("nan")
        bg_rows = df[df["is_background"] == True]
        fp_rate = bg_rows["fp"].mean() if len(bg_rows) > 0 else float("nan")
        return {"group": lbl, "n": n, "Dice": dice, "IoU": iou, "FP-bg": fp_rate}

    tool_df = results_df[results_df["is_background"] == False]
    bg_df = results_df[results_df["is_background"] == True]
    table_df = pd.DataFrame([
        group_stats(results_df, "overall"),
        group_stats(tool_df[tool_df["group"] == "majority"], "majority"),
        group_stats(tool_df[tool_df["group"] == "minority"], "minority"),
        group_stats(tool_df[tool_df["group"] == "tipdominant"], "tipdominant"),
        group_stats(bg_df, "background"),
    ])

    overall_dice = table_df.loc[table_df["group"] == "overall", "Dice"].item()
    overall_iou = table_df.loc[table_df["group"] == "overall", "IoU"].item()

    summary_path = OUTPUT_DIR / f"{name}_test_results_summary.csv"
    persample_path = OUTPUT_DIR / f"{name}_test_results_persample.csv"
    table_df.to_csv(summary_path, index=False)
    results_df.to_csv(persample_path, index=False)

    return {"name": name, "label": label, "ckpt": str(ckpt_path),
            "overall_dice": overall_dice, "overall_iou": overall_iou,
            "summary_csv": str(summary_path), "persample_csv": str(persample_path),
            "results_df": results_df}


def evaluate_all_folds(test_df, num_folds=3):
    """Run all 3 encoder-frozen checkpoints and build fold-comparison tables."""
    folds = [
        {"name": f"fold{i}", "label": f"Fold {i}",
         "ckpt": CKPT_DIR / f"fold{i}_best.pth"}
        for i in range(num_folds)
    ]

    multi_results = [run_test_for_ckpt(fd["ckpt"], fd["name"], fd["label"], test_df)
                      for fd in folds]

    multi_df = pd.DataFrame([
        {"label": r["label"], "overall_dice": r["overall_dice"],
         "overall_iou": r["overall_iou"], "ckpt": r["ckpt"]}
        for r in multi_results
    ])
    multi_df.to_csv(OUTPUT_DIR / "encoder_frozen_all_folds_overall_comparison.csv",
                     index=False)

    all_tables = []
    for r in multi_results:
        tdf = pd.read_csv(r["summary_csv"])
        tdf["fold"] = r["label"]
        all_tables.append(tdf)
    group_df = pd.concat(all_tables, ignore_index=True)

    pivot_dice = group_df.pivot(index="group", columns="fold", values="Dice")
    pivot_iou = group_df.pivot(index="group", columns="fold", values="IoU")
    pivot_fp = group_df.pivot(index="group", columns="fold", values="FP-bg")

    pivot_dice.to_csv(OUTPUT_DIR / "encoder_frozen_groupwise_dice.csv")
    pivot_iou.to_csv(OUTPUT_DIR / "encoder_frozen_groupwise_iou.csv")
    pivot_fp.to_csv(OUTPUT_DIR / "encoder_frozen_groupwise_fpbg.csv")

    return multi_df, pivot_dice, pivot_iou, pivot_fp


if __name__ == "__main__":
    test_csv = Path("/kaggle/input/datasets/monish14072002/baseline-result/splits/test.csv")
    test_df = pd.read_csv(test_csv)
    multi_df, pivot_dice, pivot_iou, pivot_fp = evaluate_all_folds(test_df, num_folds=3)
    print(multi_df)
