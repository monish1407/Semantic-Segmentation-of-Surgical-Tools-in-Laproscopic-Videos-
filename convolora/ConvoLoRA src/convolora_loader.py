# baseline_loader.py

import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold, train_test_split

from .baseline_config import (
    RAW_IMAGES, RAW_MASKS, AUG_IMAGES, AUG_MASKS, AUG_MANIFEST,
    SPLITS_DIR, CFG,
)


def build_full_manifest() -> pd.DataFrame:
    """Create combined manifest with aug, original, and background-only rows."""
    aug_manifest = pd.read_csv(AUG_MANIFEST)

    aug_rows = aug_manifest[aug_manifest["kind"] != "original"].copy()
    orig_rows = aug_manifest[aug_manifest["kind"] == "original"].copy()

    aug_rows["source"] = "aug"
    aug_rows["isbackground"] = False

    orig_rows["source"] = "aug"  # important fix from notebook
    orig_rows["isbackground"] = False

    bg_rows = []
    for mask_path in tqdm(sorted(RAW_MASKS.glob("*.png")), desc="Scanning BG frames"):
        mask_arr = np.array(Image.open(mask_path).convert("L"))
        if mask_arr.max() == 0:
            img_name = mask_path.stem + ".jpg"
            if (RAW_IMAGES / img_name).exists():
                bg_rows.append({
                    "source_frame": img_name,
                    "source_mask": mask_path.name,
                    "new_image": img_name,
                    "new_mask": mask_path.name,
                    "kind": "original",
                    "group": "background",
                    "cluster": -1,
                    "is_tip_dominant": False,
                    "n_components": 0,
                    "transform": "none",
                    "source": "raw",
                    "isbackground": True,
                })

    bg_df = pd.DataFrame(bg_rows)

    full_manifest = pd.concat([aug_rows, orig_rows, bg_df], ignore_index=True)
    assert full_manifest["new_image"].isna().sum() == 0, "NaN in new_image!"
    assert full_manifest["new_mask"].isna().sum() == 0, "NaN in new_mask!"
    return full_manifest


def make_splits(full_manifest: pd.DataFrame) -> Dict[str, Any]:
    """Create train/val/test splits with oversampled background."""
    import shutil

    if SPLITS_DIR.exists():
        shutil.rmtree(SPLITS_DIR)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    orig_frames = full_manifest[full_manifest["kind"] == "original"].reset_index(drop=True)

    trainval_frames, test_frames = train_test_split(
        orig_frames,
        test_size=CFG["test_fraction"],
        shuffle=True,
        random_state=CFG["seed"],
        stratify=None,
    )
    trainval_frames = trainval_frames.reset_index(drop=True)
    test_frames = test_frames.reset_index(drop=True)

    test_frames.to_csv(SPLITS_DIR / "test.csv", index=False)

    skf = StratifiedKFold(
        n_splits=CFG["num_folds"],
        shuffle=True,
        random_state=CFG["seed"],
    )

    fold_assignments = np.zeros(len(trainval_frames), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(trainval_frames, trainval_frames["group"])):
        fold_assignments[val_idx] = fold_idx

    trainval_frames["fold"] = fold_assignments
    trainval_frames.to_csv(SPLITS_DIR / "fold_assignments.csv", index=False)

    aug_only = full_manifest[full_manifest["kind"] != "original"].copy()

    fold_splits = {}
    for fold in range(CFG["num_folds"]):
        val_orig = trainval_frames[trainval_frames["fold"] == fold].reset_index(drop=True)
        train_orig = trainval_frames[trainval_frames["fold"] != fold].reset_index(drop=True)

        train_tool = train_orig[train_orig["isbackground"] == False].copy()
        train_aug = aug_only.copy()
        train_bg_raw = train_orig[train_orig["isbackground"] == True].copy()

        n_tool = len(train_tool) + len(train_aug)
        target_bg = int(n_tool * CFG["bg_oversample_ratio"])
        repeats = max(1, math.ceil(target_bg / max(len(train_bg_raw), 1)))
        train_bg_over = pd.concat([train_bg_raw] * repeats, ignore_index=True).head(target_bg)
        train_bg_over["kind"] = "bg_oversampled"

        TRAIN = pd.concat([train_tool, train_aug, train_bg_over], ignore_index=True)
        VAL = val_orig

        fold_splits[fold] = {"train": TRAIN, "val": VAL}

        TRAIN.to_csv(SPLITS_DIR / f"fold{fold}_train.csv", index=False)
        VAL.to_csv(SPLITS_DIR / f"fold{fold}_val.csv", index=False)

    return {
        "trainval_frames": trainval_frames,
        "test_frames": test_frames,
        "fold_splits": fold_splits,
    }


__all__ = ["build_full_manifest", "make_splits"]
