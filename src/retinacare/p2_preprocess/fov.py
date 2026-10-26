"""Retinal Field of View (FOV) detection and segmentation."""
import cv2
import numpy as np


def detect_fov(img_rgb: np.ndarray, fov_threshold: int = 10) -> np.ndarray:
    """Segment retinal FOV mask from background."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, fov_threshold, 255, cv2.THRESH_BINARY)

    h, w = gray.shape
    k_size = max(5, int(min(h, w) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    # Morphological close and open
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        # Fallback: full circular mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.48), 255, -1)
        return mask

    # Index 0 is background, find largest component among >= 1
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fov_mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    # Fill internal holes (e.g. dark retinal lesions)
    contours, _ = cv2.findContours(fov_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(fov_mask, contours, -1, 255, thickness=cv2.FILLED)

    return fov_mask
