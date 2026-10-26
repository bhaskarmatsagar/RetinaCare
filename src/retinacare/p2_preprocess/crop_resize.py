"""Crop, square padding, and multi-resolution resizing."""
from typing import Tuple, Dict, Any
import cv2
import numpy as np


def crop_to_fov(img_rgb: np.ndarray, fov_mask: np.ndarray, margin_frac: float = 0.02) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Crop image and mask to square-padded FOV bounding box."""
    orig_h, orig_w = img_rgb.shape[:2]
    y_indices, x_indices = np.where(fov_mask > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        # Fallback if empty mask
        x0, y0, x1, y1 = 0, 0, orig_w, orig_h
    else:
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

        margin_x = int((x_max - x_min) * margin_frac)
        margin_y = int((y_max - y_min) * margin_frac)

        x0 = max(0, x_min - margin_x)
        y0 = max(0, y_min - margin_y)
        x1 = min(orig_w, x_max + margin_x)
        y1 = min(orig_h, y_max + margin_y)

    crop_img = img_rgb[y0:y1, x0:x1]
    crop_mask = fov_mask[y0:y1, x0:x1]

    # Pad to square
    ch, cw = crop_img.shape[:2]
    max_dim = max(ch, cw)
    pad_top = (max_dim - ch) // 2
    pad_bottom = max_dim - ch - pad_top
    pad_left = (max_dim - cw) // 2
    pad_right = max_dim - cw - pad_left

    square_img = cv2.copyMakeBorder(crop_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    square_mask = cv2.copyMakeBorder(crop_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    frame_info = {
        "crop_box": [x0, y0, x1, y1],
        "orig_size": [orig_w, orig_h],
        "square_dim": max_dim,
        "padding": [pad_top, pad_bottom, pad_left, pad_right]
    }
    return square_img, square_mask, frame_info


def resize_products(
    square_img: np.ndarray,
    square_mask: np.ndarray,
    clf_size: int = 256,
    hires_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Resize cropped square image and mask to clf and hires dimensions."""
    orig_side = square_img.shape[0]

    img_clf = cv2.resize(square_img, (clf_size, clf_size), interpolation=cv2.INTER_AREA)
    mask_clf = cv2.resize(square_mask, (clf_size, clf_size), interpolation=cv2.INTER_NEAREST)

    img_hires = cv2.resize(square_img, (hires_size, hires_size), interpolation=cv2.INTER_AREA)
    mask_hires = cv2.resize(square_mask, (hires_size, hires_size), interpolation=cv2.INTER_NEAREST)

    scale_clf = float(clf_size / orig_side)
    scale_hires = float(hires_size / orig_side)

    return img_clf, mask_clf, img_hires, mask_hires, scale_clf, scale_hires
