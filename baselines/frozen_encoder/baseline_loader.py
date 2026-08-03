"""Baseline UNet ResNet50 (Encoder Frozen) - Dataset & Splits"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config import (
    RAW_IMAGES, RAW_MASKS, AUG_IMAGES, AUG_MASKS, AUG_MANIFEST,
    SPLITS_DIR, CFG, IMAGENET_MEAN, IMAGENET_STD,
)


def build_full_manifest():
    """Merge augmented, original, and raw-mask-derived background-only frames."""
    manifest = pd.read_csv(AUG_MANIFEST)
    aug_rows = manifest[manifest["kind"] != "original"].copy()
    orig_rows = manifest[manifest["kind"] == "original"].copy()
    aug_rows["source"] = "aug"
    aug_rows["is_background"] = False
    orig_rows["source"] = "aug"   # orig frame.png files live in AUG dirs
    orig_rows["is_background"] = False

    bg_rows = []
    for mask_path in sorted(RAW_MASKS.glob("*.png")):
        mask_arr = np.array(Image.open(mask_path).convert("L"))
        if mask_arr.max() == 0:
            img_name = mask_path.stem + ".jpg"
            if (RAW_IMAGES / img_name).exists():
                bg_rows.append({
                    "source_frame": img_name, "source_mask": mask_path.name,
                    "new_image": img_name, "new_mask": mask_path.name,
                    "kind": "original", "group": "background", "cluster": -1,
                    "is_tip_dominant": False, "n_components": 0,
                    "transform": "none", "source": "raw", "is_background": True,
                })
    bg_df = pd.DataFrame(bg_rows)

    full_manifest = pd.concat([aug_rows, orig_rows, bg_df], ignore_index=True)
    assert full_manifest["new_image"].isna().sum() == 0
    assert full_manifest["new_mask"].isna().sum() == 0
    return full_manifest


def load_or_create_splits(num_folds=CFG["num_folds"]):
    """Load previously generated fold split CSVs (train/val per fold + test.csv)."""
    all_exist = all(
        (SPLITS_DIR / f"fold{f}_train.csv").exists() and
        (SPLITS_DIR / f"fold{f}_val.csv").exists()
        for f in range(num_folds)
    ) and (SPLITS_DIR / "test.csv").exists()

    if not all_exist:
        raise FileNotFoundError(
            "No saved split CSVs found. Paste original split-generation code "
            "or supply fold{N}_train.csv / fold{N}_val.csv / test.csv."
        )

    test_frame = pd.read_csv(SPLITS_DIR / "test.csv")
    fold_splits = {}
    for fold in range(num_folds):
        train = pd.read_csv(SPLITS_DIR / f"fold{fold}_train.csv")
        val = pd.read_csv(SPLITS_DIR / f"fold{fold}_val.csv")
        fold_splits[fold] = {"train": train, "val": val}
    return fold_splits, test_frame


class InstrumentDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        sz = CFG["image_size"]

        if augment:
            self.spatial = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08,
                                    rotate_limit=15, border_mode=0, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.12,
                                            contrast_limit=0.12, p=0.4),
            ], additional_targets={"mask": "mask"})
        else:
            self.spatial = None

        self.post = A.Compose([
            A.Resize(sz, sz),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = str(row["source"])
        is_bg = bool(row["is_background"])
        group = str(row["group"])
        img_dir = RAW_IMAGES if src == "raw" else AUG_IMAGES
        mask_dir = RAW_MASKS if src == "raw" else AUG_MASKS

        img = np.array(Image.open(img_dir / row["new_image"]).convert("RGB"))
        mask = np.array(Image.open(mask_dir / row["new_mask"]).convert("L"))
        mask = (mask > 0).astype(np.float32)

        if self.spatial is not None:
            out = self.spatial(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        out = self.post(image=img, mask=mask)
        img = out["image"]
        mask = out["mask"].unsqueeze(0).float()

        return {
            "image": img, "mask": mask, "group": group,
            "is_background": is_bg, "id": row["new_image"],
        }


def make_weighted_sampler(df):
    """Per-row sampling weights: BG controlled, tip/minority groups boosted."""
    weights = []
    for _, row in df.iterrows():
        g = str(row["group"]).strip().lower().replace("-", "").replace(" ", "")
        if row["is_background"]:
            weights.append(CFG["bg_sampler_w"])
        elif g == "tipdominant":
            weights.append(CFG["tip_sampler_w"])
        elif g == "minority":
            weights.append(CFG["min_sampler_w"])
        else:
            weights.append(1.0)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
