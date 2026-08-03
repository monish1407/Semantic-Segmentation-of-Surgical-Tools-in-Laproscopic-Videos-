# baseline_config.py

from pathlib import Path
import torch

# Base paths (adapt these in Kaggle vs local)
BASE = Path("/kaggle/input/datasets/monish14072002")
RAW_IMAGES = BASE / "raw-dataset/images/images"
RAW_MASKS = BASE / "raw-dataset/masks/masks"
AUG_IMAGES = BASE / "augmentation/augmented_dataset/images"
AUG_MASKS = BASE / "augmentation/augmented_dataset/masks"
AUG_MANIFEST = BASE / "augmentation/augmented_dataset/metadata/augmentation_manifest.csv"

OUTPUT_DIR = Path("./output"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR = OUTPUT_DIR / "splits"; SPLITS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)

CFG = dict(
    image_size = 512,
    batch_size = 8,
    num_workers = 4,
    lr_decoder = 1e-3,
    wd_decoder = 1e-4,
    num_epochs = 40,
    alpha = 0.5,
    fp_penalty_w = 5.0,
    bg_sampler_w = 0.4,
    tip_sampler_w = 2.0,
    min_sampler_w = 1.5,
    num_folds = 3,
    seed = 42,
    delta = 1e-3,
    patience = 5,
    lr_patience = 3,
    bg_oversample_ratio = 0.12,
    max_train_hours = 11.5,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    test_fraction = 0.15,
    lora_rank = 4,
    fold_to_run = 0,  # change 0 / 1 / 2
)

TORCH_SEED = CFG["seed"]

torch.manual_seed(TORCH_SEED)

__all__ = [
    "BASE", "RAW_IMAGES", "RAW_MASKS", "AUG_IMAGES", "AUG_MASKS", "AUG_MANIFEST",
    "OUTPUT_DIR", "SPLITS_DIR", "CKPT_DIR", "CFG",
]