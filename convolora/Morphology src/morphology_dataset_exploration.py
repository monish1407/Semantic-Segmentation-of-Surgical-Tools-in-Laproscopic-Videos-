"""Morphology dataset exploration.

Extracted from complete-dataset-exploration.ipynb and renamed for the
morphology project.

This module includes:
- dataset path setup
- image/mask pair scanning
- background-only frame detection
- instrument-containing pair filtering
- shape/area feature extraction
- bounding-box and centroid analysis
- perimeter/compactness/solidity analysis
- connected-component analysis
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def get_paths(image_path=None, mask_path=None):
    """Return dataset paths and discovered image/mask file lists."""
    if image_path is None:
        image_path = Path("/kaggle/input/datasets/monish1407/images/images")
    else:
        image_path = Path(image_path)

    if mask_path is None:
        mask_path = Path("/kaggle/input/datasets/monish1407/maskes/masks")
    else:
        mask_path = Path(mask_path)

    image_exts = (".jpg", ".jpeg", ".png")
    mask_exts = (".png",)

    image_files = sorted(
        image_path / f for f in os.listdir(image_path) if f.lower().endswith(image_exts)
    )
    mask_files = sorted(
        mask_path / f for f in os.listdir(mask_path) if f.lower().endswith(mask_exts)
    )

    assert len(image_files) == len(mask_files), "Image/mask count mismatch!"
    return image_path, mask_path, image_files, mask_files


def scan_masks(image_files, mask_files):
    """Split frames into background-only and instrument-containing pairs."""
    bg_only_frames = []
    instrument_pairs = []
    unreadable_masks = []

    for img_path, mask_path in zip(image_files, mask_files):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            unreadable_masks.append(mask_path.name)
            continue

        if np.count_nonzero(mask) == 0:
            bg_only_frames.append(mask_path.name)
        else:
            instrument_pairs.append((img_path, mask_path))

    return bg_only_frames, instrument_pairs, unreadable_masks


def extract_area_features(instrument_pairs):
    """Extract mask area features."""
    area_records = []

    for img_path, mask_path in instrument_pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        H, W = mask.shape
        total_px = H * W
        area_px = int(np.count_nonzero(mask))
        rel_area = area_px / total_px

        area_records.append({
            "frame": img_path.name,
            "mask_file": mask_path.name,
            "H": H,
            "W": W,
            "total_px": total_px,
            "area_px": area_px,
            "rel_area": round(rel_area, 6),
        })

    return pd.DataFrame(area_records)


def extract_bbox_features(instrument_pairs):
    """Extract bounding-box shape features."""
    bbox_records = []

    for img_path, mask_path in instrument_pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        bbox_area_px = bbox_w * bbox_h
        aspect_ratio = bbox_w / bbox_h if bbox_h else np.nan
        extent = np.count_nonzero(mask) / bbox_area_px if bbox_area_px else np.nan

        bbox_records.append({
            "frame": img_path.name,
            "bbox_x": int(x0),
            "bbox_y": int(y0),
            "bbox_w": int(bbox_w),
            "bbox_h": int(bbox_h),
            "bbox_area_px": int(bbox_area_px),
            "aspect_ratio": round(float(aspect_ratio), 4),
            "extent": round(float(extent), 6),
        })

    return pd.DataFrame(bbox_records)


def extract_centroid_features(instrument_pairs):
    """Extract centroid and zoning features."""
    centroid_records = []

    for img_path, mask_path in instrument_pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        H, W = mask.shape
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        cx = xs.mean()
        cy = ys.mean()
        cx_norm = cx / W
        cy_norm = cy / H

        if cx_norm < 0.33:
            x_zone = "left"
        elif cx_norm < 0.66:
            x_zone = "center"
        else:
            x_zone = "right"

        if cy_norm < 0.33:
            y_zone = "top"
        elif cy_norm < 0.66:
            y_zone = "middle"
        else:
            y_zone = "bottom"

        centroid_records.append({
            "frame": img_path.name,
            "cx": round(float(cx), 2),
            "cy": round(float(cy), 2),
            "cx_norm": round(float(cx_norm), 4),
            "cy_norm": round(float(cy_norm), 4),
            "x_zone": x_zone,
            "y_zone": y_zone,
        })

    return pd.DataFrame(centroid_records)


def extract_contour_features(instrument_pairs):
    """Extract perimeter, compactness, convex hull area, and solidity."""
    contour_records = []

    for img_path, mask_path in instrument_pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        all_pts = np.vstack(contours)
        area = float(np.count_nonzero(mask_bin))
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))

        hull = cv2.convexHull(all_pts)
        hull_area = float(cv2.contourArea(hull)) if hull is not None and len(hull) > 0 else np.nan

        compactness = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan
        solidity = area / hull_area if hull_area and hull_area > 0 else np.nan

        contour_records.append({
            "frame": img_path.name,
            "perimeter": round(perimeter, 2),
            "compactness": round(float(compactness), 6),
            "convex_hull_area": round(hull_area, 2) if hull_area == hull_area else np.nan,
            "solidity": round(float(solidity), 6),
        })

    return pd.DataFrame(contour_records)


def extract_component_features(instrument_pairs):
    """Extract connected-component statistics."""
    component_records = []

    for img_path, mask_path in instrument_pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 0).astype(np.uint8)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        # label 0 is background
        n_components = max(0, n_labels - 1)

        if n_components > 0:
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_component_px = int(component_areas.max())
            largest_component_ratio = float(largest_component_px / component_areas.sum())
        else:
            largest_component_px = 0
            largest_component_ratio = np.nan

        component_records.append({
            "frame": img_path.name,
            "n_components": int(n_components),
            "largest_component_px": int(largest_component_px),
            "largest_component_ratio": round(float(largest_component_ratio), 6) if largest_component_ratio == largest_component_ratio else np.nan,
        })

    return pd.DataFrame(component_records)


def build_morphology_exploration(image_path=None, mask_path=None, save_csv_dir=None):
    """Run the full morphology exploration pipeline.

    Parameters
    ----------
    image_path : str or Path, optional
    mask_path : str or Path, optional
    save_csv_dir : str or Path, optional
        If provided, saves background_only_frames.csv and feature tables.

    Returns
    -------
    dict
        Keys: image_path, mask_path, image_files, mask_files, bg_only_frames,
        instrument_pairs, df_area, df_bbox, df_centroid, df_contour, df_component.
    """
    image_path, mask_path, image_files, mask_files = get_paths(image_path, mask_path)
    bg_only_frames, instrument_pairs, unreadable_masks = scan_masks(image_files, mask_files)

    df_area = extract_area_features(instrument_pairs)
    df_bbox = extract_bbox_features(instrument_pairs)
    df_centroid = extract_centroid_features(instrument_pairs)
    df_contour = extract_contour_features(instrument_pairs)
    df_component = extract_component_features(instrument_pairs)

    if save_csv_dir is not None:
        save_csv_dir = Path(save_csv_dir)
        save_csv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"mask_filename": bg_only_frames}).to_csv(save_csv_dir / "background_only_frames.csv", index=False)
        df_area.to_csv(save_csv_dir / "morphology_area_features.csv", index=False)
        df_bbox.to_csv(save_csv_dir / "morphology_bbox_features.csv", index=False)
        df_centroid.to_csv(save_csv_dir / "morphology_centroid_features.csv", index=False)
        df_contour.to_csv(save_csv_dir / "morphology_contour_features.csv", index=False)
        df_component.to_csv(save_csv_dir / "morphology_component_features.csv", index=False)

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "image_files": image_files,
        "mask_files": mask_files,
        "bg_only_frames": bg_only_frames,
        "instrument_pairs": instrument_pairs,
        "unreadable_masks": unreadable_masks,
        "df_area": df_area,
        "df_bbox": df_bbox,
        "df_centroid": df_centroid,
        "df_contour": df_contour,
        "df_component": df_component,
    }
