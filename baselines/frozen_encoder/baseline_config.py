"""Baseline UNet ResNet50 (Encoder Frozen, 3-Fold CV) - Config"""

import random
from pathlib import Path
import torch

BASE = Path("/kaggle/input/datasets/monishpatil2002")
RAW_IMAGES = BASE / "raw-dataset/images/images"
RAW_MASKS = BASE / "raw-dataset/masks/masks"
CLUSTER_CSV = BASE / "clusters/stage2_clusters.csv"
AUG_IMAGES = BASE / "augmented/augmented_dataset/images"
AUG_MASKS = BASE / "augmented/augmented_dataset/masks"
AUG_MANIFEST = BASE / "augmented/augmented_dataset/metadata/augmentation_manifest.csv"

RESULTS_ROOT = BASE / "results"
PREV_CKPTS = RESULTS_ROOT / "checkpoints"
PREV_SPLITS = RESULTS_ROOT / "splits"

SPLITS_DIR = Path("/kaggle/working/splits")
CKPT_DIR = Path("/kaggle/working/checkpoints")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CFG = dict(
    image_size=512,
    batch_size=8,
    num_workers=4,
    lr_decoder=1e-3,
    wd_decoder=1e-4,
    num_epochs=40,
    alpha=0.5,             # Dice vs Dynamic-WBCE mix for foreground frames
    fp_penalty_w=5.0,      # false-positive penalty weight for background frames
    bg_sampler_w=0.4,
    tip_sampler_w=2.0,
    min_sampler_w=1.5,
    num_folds=3,           # only 3 folds trained (full encoder freezing experiment)
    seed=42,
    delta=1e-3,
    patience=5,
    lr_patience=3,
    bg_oversample_ratio=0.12,
    max_train_hours=11.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    test_fraction=0.15,
)

torch.manual_seed(CFG["seed"])
random.seed(CFG["seed"])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
