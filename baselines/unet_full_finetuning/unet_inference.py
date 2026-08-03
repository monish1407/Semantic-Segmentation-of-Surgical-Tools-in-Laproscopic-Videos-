"""UNet (ResNet50 Encoder) Full Fine-Tuning - Inference & Visualisation"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from configs.config import DEVICE, IMAGE_SIZE
from utils.loader import normalize


def predict_mask(model, image_path, threshold=0.6):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    img_norm = normalize(img)
    img_tensor = torch.tensor(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).float().to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred_mask = (prob > threshold).astype(np.uint8)
    return img, pred_mask, prob


def visualise_overlay(img, pred_mask, gt_mask=None, title="Prediction", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.contour(pred_mask, colors="lime", linewidths=1.5)
    handles = [mpatches.Patch(color="lime", label="Prediction")]
    if gt_mask is not None:
        ax.contour(gt_mask, colors="red", linewidths=1.5)
        handles.append(mpatches.Patch(color="red", label="Ground truth"))
    ax.legend(handles=handles, loc="upper right")
    ax.set_title(title)
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
