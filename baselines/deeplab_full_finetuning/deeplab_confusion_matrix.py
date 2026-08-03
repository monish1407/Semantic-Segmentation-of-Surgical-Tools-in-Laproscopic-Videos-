"""DeepLabV3+ (ResNet50 Encoder) Full Fine-Tuning - Confusion Matrix"""

import numpy as np
import matplotlib.pyplot as plt


def compute_pixel_confusion(pred_bin, mask_bin):
    tp = int((pred_bin & mask_bin).sum())
    fp = int((pred_bin & ~mask_bin).sum())
    fn = int((~pred_bin & mask_bin).sum())
    tn = int((~pred_bin & ~mask_bin).sum())
    return np.array([[tn, fp], [fn, tp]])


def plot_confusion_matrix(cm, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Bg", "Pred Tool"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["GT Bg", "GT Tool"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.title("Pixel-level Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
