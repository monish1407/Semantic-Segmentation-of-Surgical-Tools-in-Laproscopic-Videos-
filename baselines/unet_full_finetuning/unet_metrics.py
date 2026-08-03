"""UNet (ResNet50 Encoder) Full Fine-Tuning - Evaluation Metrics"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def compute_iou(pred_bin, mask_bin):
    """Intersection over Union."""
    inter = (pred_bin & mask_bin).sum()
    union = (pred_bin | mask_bin).sum()
    return float(inter) / (float(union) + 1e-6)


def compute_hausdorff(pred_bin, mask_bin):
    """95th-percentile Hausdorff Distance (pixels)."""
    if pred_bin.sum() == 0 or mask_bin.sum() == 0:
        return float("nan")
    dt_pred = distance_transform_edt(~pred_bin)
    dt_mask = distance_transform_edt(~mask_bin)
    hd_pm = float(np.percentile(dt_mask[pred_bin], 95))
    hd_mp = float(np.percentile(dt_pred[mask_bin], 95))
    return float(max(hd_pm, hd_mp))


def compute_surface_distances(pred_bin, mask_bin):
    """Mean Surface Distance and Average Surface Distance."""
    if pred_bin.sum() == 0 or mask_bin.sum() == 0:
        return float("nan"), float("nan")
    dt_pred = distance_transform_edt(~pred_bin)
    dt_mask = distance_transform_edt(~mask_bin)
    msd = float((dt_mask[pred_bin].mean() + dt_pred[mask_bin].mean()) / 2)
    asd = float((dt_mask[pred_bin].sum() + dt_pred[mask_bin].sum()) /
                (pred_bin.sum() + mask_bin.sum()))
    return msd, asd


def batch_metrics(preds_logit, masks_gt, threshold=0.6):
    """Compute all metrics for a batch. Returns dict of lists."""
    probs = torch.sigmoid(preds_logit).cpu().numpy()
    masks = masks_gt.cpu().numpy()
    results = {"iou": [], "hausdorff": [], "msd": [], "asd": []}
    for b in range(probs.shape[0]):
        pb = (probs[b, 0] > threshold).astype(bool)
        mb = (masks[b, 0] > 0.5).astype(bool)
        results["iou"].append(compute_iou(pb, mb))
        results["hausdorff"].append(compute_hausdorff(pb, mb))
        msd, asd = compute_surface_distances(pb, mb)
        results["msd"].append(msd)
        results["asd"].append(asd)
    return results
