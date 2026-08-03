# baseline_dataset.py

from typing import Dict, Any

import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .baseline_config import CFG, RAW_IMAGES, RAW_MASKS, AUG_IMAGES, AUG_MASKS


class InstrumentDataset(Dataset):
    def __init__(self, df, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        sz = CFG["image_size"]

        if augment:
            self.spatial = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.04,
                    scale_limit=0.08,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.7,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.12,
                    contrast_limit=0.12,
                    p=0.4,
                ),
            ])
        else:
            self.spatial = None

        self.post = A.Compose([
            A.Resize(sz, sz),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        src = str(row["source"])
        is_bg = bool(row["isbackground"])
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
            "image": img,
            "mask": mask,
            "group": group,
            "isbackground": is_bg,
            "id": row["new_image"],
        }


def make_weighted_sampler(df) -> WeightedRandomSampler:
    weights = []
    for _, row in df.iterrows():
        g = str(row["group"]).strip().lower().replace("_", "").replace(" ", "")
        if row["isbackground"]:
            weights.append(CFG["bg_sampler_w"])
        elif g == "tipdominant":
            weights.append(CFG["tip_sampler_w"])
        elif g == "minority":
            weights.append(CFG["min_sampler_w"])
        else:
            weights.append(1.0)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(fold_splits, fold: int):
    train_df = fold_splits[fold]["train"]
    val_df = fold_splits[fold]["val"]

    train_ds = InstrumentDataset(train_df, augment=True)
    val_ds = InstrumentDataset(val_df, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        sampler=make_weighted_sampler(train_df),
        num_workers=CFG["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader


__all__ = ["InstrumentDataset", "make_weighted_sampler", "build_dataloaders"]
