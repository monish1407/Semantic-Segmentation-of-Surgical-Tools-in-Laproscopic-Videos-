# baseline_utils.py

"""Utility helpers for the ConvLoRA baseline project.

This module intentionally stays lightweight so that baseline_* files map
cleanly from the original Kaggle notebook structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from .baseline_config import OUTPUT_DIR, CFG


@dataclass
class CheckpointSummary:
    fold: int
    epoch: int
    best_val_dice: float


def get_device() -> torch.device:
    return torch.device(CFG["device"])


def ensure_output_tree() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


__all__ = [
    "CheckpointSummary",
    "get_device",
    "ensure_output_tree",
    "count_parameters",
]
