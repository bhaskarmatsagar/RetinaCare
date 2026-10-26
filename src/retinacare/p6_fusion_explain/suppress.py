"""Anatomical false positive suppression (optic disc and FOV border artefacts)."""
from typing import Dict, Tuple, Optional, Any
import cv2
import numpy as np


def suppress_disc_and_border_lesions(
    lesion_masks: Dict[str, np.ndarray],
    disc_mask: Optional[np.ndarray],
    fov_mask: np.ndarray,
    border_margin_frac: float = 0.03
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Remove EX/SE false positives inside dilated disc and border edge artefacts."""
    h, w = fov_mask.shape
    cleaned_masks = {}
    ex_removed_in_disc = 0
    border_removed_total = 0

    # 1. Dilated optic disc mask (10% dilation)
    if disc_mask is not None and np.sum(disc_mask > 0) > 0:
        disc_area = np.sum(disc_mask > 0)
        disc_r = np.sqrt(disc_area / np.pi)
        k_dilate = max(3, int(disc_r * 0.10 * 2))
        kernel_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
        disc_dilated = cv2.dilate(disc_mask, kernel_disc)
    else:
        disc_dilated = None

    # 2. Outer border ring (outer 3% of FOV radius)
    fov_area = np.sum(fov_mask > 0)
    fov_r = np.sqrt(fov_area / np.pi)
    erode_border_k = max(3, int(fov_r * border_margin_frac * 2))
    kernel_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_border_k, erode_border_k))
    fov_inner = cv2.erode(fov_mask, kernel_border)
    border_ring = np.where((fov_mask > 0) & (fov_inner == 0), 255, 0).astype(np.uint8)

    for name, mask in lesion_masks.items():
        clean_m = np.zeros_like(mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            # Check border ring
            if 0 <= cy < h and 0 <= cx < w and border_ring[cy, cx] > 0:
                border_removed_total += 1
                continue

            # Check optic disc suppression for bright lesions (EX and SE)
            if name in ["EX", "SE"] and disc_dilated is not None:
                if 0 <= cy < h and 0 <= cx < w and disc_dilated[cy, cx] > 0:
                    ex_removed_in_disc += 1
                    continue

            clean_m[labels == i] = 255

        cleaned_masks[name] = clean_m

    return cleaned_masks, ex_removed_in_disc, border_removed_total
