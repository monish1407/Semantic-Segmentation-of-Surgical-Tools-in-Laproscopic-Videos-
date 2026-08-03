"""DeepLabV3+ (ResNet50 Encoder) Full Fine-Tuning - Utilities"""

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler


def analyse_dataset(mask_files, image_size, save_path="dataset_analysis.png"):
    """Measure tool pixel count, tool presence, image difficulty (stratified)."""
    import cv2

    tool_pixel_counts = []
    has_tool = []
    for mp in tqdm(mask_files, desc="Scanning masks"):
        m = cv2.imread(mp, 0)
        m = cv2.resize(m, (image_size, image_size))
        m = (m > 10).astype(np.uint8)
        count = m.sum()
        tool_pixel_counts.append(count)
        has_tool.append(int(count > 0))

    tool_pixel_counts = np.array(tool_pixel_counts)
    has_tool = np.array(has_tool)

    print(f"% with tool: {has_tool.sum() / len(has_tool) * 100:.1f}")
    print(f"Avg tool pixels/img: {tool_pixel_counts[has_tool == 1].mean():.0f}")
    print(f"Total pixels: {image_size * image_size}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(tool_pixel_counts[has_tool == 1], bins=40, color="steelblue", edgecolor="k")
    axes[0].set_title("Tool Pixel Count Distribution (images with tool)")
    axes[0].set_xlabel("Pixel count")
    axes[0].set_ylabel("Frequency")

    axes[1].bar(["No Tool", "Has Tool"], [(has_tool == 0).sum(), (has_tool == 1).sum()],
                color=["salmon", "steelblue"], edgecolor="k")
    axes[1].set_title("Class Balance: Tool vs Background Images")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print("Dataset analysis saved.")
    return tool_pixel_counts, has_tool


def build_weighted_sampler(has_tool_labels):
    """Balance batches by inverse class frequency."""
    class_counts = [has_tool_labels.count(0), has_tool_labels.count(1)]
    weights = [1.0 / class_counts[label] for label in has_tool_labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def clear_checkpoints(ckpt_dir):
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("All checkpoints cleared, will start completely fresh")
