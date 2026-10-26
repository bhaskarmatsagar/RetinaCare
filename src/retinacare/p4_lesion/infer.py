"""Sliding window inference with Gaussian blending, TTA, and post-processing."""
from typing import Dict, Tuple, Any, List
import cv2
import numpy as np


def create_gaussian_weight_window(window_size: int = 256, sigma_scale: float = 0.125) -> np.ndarray:
    """Create a 2D Gaussian weight window for smooth patch blending."""
    sigma = window_size * sigma_scale
    g1d = cv2.getGaussianKernel(window_size, sigma)
    g2d = g1d * g1d.T
    g2d /= np.max(g2d)
    return g2d.astype(np.float32)


def sliding_window_inference(
    model,
    img_hires_rgb: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    tta: bool = True
) -> np.ndarray:
    """Sliding-window tile inference over hires image with Gaussian blending and TTA."""
    h, w, c = img_hires_rgb.shape
    num_classes = 4

    prob_map = np.zeros((h, w, num_classes), dtype=np.float32)
    weight_map = np.zeros((h, w, num_classes), dtype=np.float32)
    gauss_weight = create_gaussian_weight_window(patch_size)[:, :, np.newaxis]

    y_steps = list(range(0, h - patch_size + 1, stride))
    if y_steps[-1] != h - patch_size:
        y_steps.append(h - patch_size)

    x_steps = list(range(0, w - patch_size + 1, stride))
    if x_steps[-1] != w - patch_size:
        x_steps.append(w - patch_size)

    patches = []
    coords = []
    for y in y_steps:
        for x in x_steps:
            p = img_hires_rgb[y : y + patch_size, x : x + patch_size].astype(np.float32) / 255.0
            patches.append(p)
            coords.append((y, x))

    patches_arr = np.array(patches, dtype=np.float32)

    # Base inference
    preds = model.predict(patches_arr, batch_size=16, verbose=0)

    # Test-time augmentation (TTA) if enabled
    if tta:
        # Horizontal flip
        patches_hf = np.flip(patches_arr, axis=2)
        preds_hf = model.predict(patches_hf, batch_size=16, verbose=0)
        preds_hf = np.flip(preds_hf, axis=2)

        # Vertical flip
        patches_vf = np.flip(patches_arr, axis=1)
        preds_vf = model.predict(patches_vf, batch_size=16, verbose=0)
        preds_vf = np.flip(preds_vf, axis=1)

        preds = (preds + preds_hf + preds_vf) / 3.0

    # Accumulate with Gaussian weights
    for idx, (y, x) in enumerate(coords):
        prob_map[y : y + patch_size, x : x + patch_size] += preds[idx] * gauss_weight
        weight_map[y : y + patch_size, x : x + patch_size] += gauss_weight

    prob_map /= np.maximum(weight_map, 1e-5)
    return np.clip(prob_map, 0.0, 1.0)


def postprocess_lesion_masks(
    prob_map: np.ndarray,
    fov_mask: np.ndarray,
    thresholds: Dict[str, float] = None,
    min_area_px: Dict[str, int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Threshold probability maps, suppress outside eroded FOV, filter by min component area."""
    class_names = ["MA", "HE", "EX", "SE"]
    if thresholds is None:
        thresholds = {"MA": 0.5, "HE": 0.5, "EX": 0.5, "SE": 0.5}
    if min_area_px is None:
        min_area_px = {"MA": 3, "HE": 20, "EX": 8, "SE": 30}

    h, w = fov_mask.shape
    erode_k = max(3, int(w * 0.02))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    fov_clean = cv2.erode(fov_mask, kernel_erode)

    masks = {}
    stats = {}
    fov_area = float(np.sum(fov_mask > 0))

    for idx, name in enumerate(class_names):
        p_c = prob_map[:, :, idx]
        th = thresholds.get(name, 0.5)
        bin_m = np.where((p_c >= th) & (fov_clean > 0), 255, 0).astype(np.uint8)

        # Filter small components
        n_labels, labels, stat_arr, _ = cv2.connectedComponentsWithStats(bin_m, connectivity=8)
        clean_m = np.zeros_like(bin_m)
        min_area = min_area_px.get(name, 5)

        for i in range(1, n_labels):
            if stat_arr[i, cv2.CC_STAT_AREA] >= min_area:
                clean_m[labels == i] = 255

        masks[name] = clean_m
        comp_count = max(0, n_labels - 1)
        area_px = int(np.sum(clean_m > 0))
        area_frac = float(area_px / (fov_area + 1e-5))

        stats[name] = {
            "count": comp_count,
            "area_px": area_px,
            "area_frac_fov": round(area_frac, 6)
        }

    return masks, stats
