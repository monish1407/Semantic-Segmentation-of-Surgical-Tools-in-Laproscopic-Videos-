"""UNet (ResNet50 Encoder) Full Fine-Tuning - Config"""

import os
import torch

IMAGE_PATH = "/kaggle/input/datasets/monish14072002images/images"
MASK_PATH = "/kaggle/input/datasets/monish14072002maskes/masks"

IMAGE_SIZE = 512          # do not downscale further, small tools disappear
BATCH_SIZE = 8
EPOCHS = 40               # train until convergence
LEARNING_RATE = 1e-4
NUM_CLASSES = 1
ENCODER = "resnet50"
DROPOUT = 0.3             # decoder dropout
SAVE_PATH = "/kaggle/working/unet_resnet50_512.pth"
CKPT_DIR = "/kaggle/working/unet_checkpoints"
N_FOLDS = 5
EPSILON = 1e-4            # convergence threshold
PATIENCE_CV = 7           # early-stop patience
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
