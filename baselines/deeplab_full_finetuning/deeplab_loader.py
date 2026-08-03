"""DeepLabV3+ (ResNet50 Encoder) Full Fine-Tuning - Dataset Loader"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from configs.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.GaussianBlur(p=0.2),
    A.CLAHE(p=0.2),
])

val_transform = A.Compose([])

IMAGENET_MEAN_ARR = np.array(IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD_ARR = np.array(IMAGENET_STD, dtype=np.float32)


def normalize(img):
    """ImageNet normalisation required for pretrained ResNet50 encoder weights."""
    return (img - IMAGENET_MEAN_ARR) / IMAGENET_STD_ARR


class SurgicalDataset(Dataset):
    """Disk-based dataset. Optionally RAM-cache for speed."""

    def __init__(self, images, masks, transform=None, split="", cache=False):
        self.transform = transform
        self.cache = cache
        self.img_paths = images
        self.mask_paths = masks
        self.imgs = []
        self.masks = []

        if cache:
            print(f"Caching {split} split ({len(images)} imgs) into RAM...")
            for ip, mp in zip(images, masks):
                img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mp, 0)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
                mask = (mask > 10).astype(np.uint8) * 255
                self.imgs.append(img)
                self.masks.append(mask)
            mb = len(self.imgs) * IMAGE_SIZE * IMAGE_SIZE * 3 / 1e6
            print(f"Done. {mb:.0f} MB")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.cache:
            img = self.imgs[idx].astype(np.float32) / 255.0
            mask = self.masks[idx].astype(np.float32) / 255.0
        else:
            img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[idx], 0)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
            mask = (mask > 10).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=(img * 255).astype(np.uint8),
                                  mask=(mask * 255).astype(np.uint8))
            img = aug["image"].astype(np.float32) / 255.0
            mask = (aug["mask"] > 10).astype(np.float32)

        img = normalize(img)
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img).float(), torch.tensor(mask).unsqueeze(0).float()


def get_file_lists(image_path, mask_path):
    image_files = sorted(os.path.join(image_path, f) for f in os.listdir(image_path)
                          if f.lower().endswith((".jpg", ".jpeg", ".png")))
    mask_files = sorted(os.path.join(mask_path, f) for f in os.listdir(mask_path)
                         if f.lower().endswith(".png"))
    assert len(image_files) == len(mask_files), \
        f"Image/mask count mismatch! images={len(image_files)} masks={len(mask_files)}"
    return image_files, mask_files
