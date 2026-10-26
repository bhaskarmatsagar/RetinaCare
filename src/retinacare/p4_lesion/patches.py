"""Patch extraction, sampling mix (60% lesion, 40% negative), and data augmentation."""
from typing import Tuple, List
import cv2
import numpy as np


def extract_patch(
    img: np.ndarray,
    mask: np.ndarray,
    cy: int,
    cx: int,
    patch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract patch centered at (cx, cy) with bounds clipping."""
    h, w = img.shape[:2]
    half = patch_size // 2

    y0 = max(0, cy - half)
    y1 = min(h, y0 + patch_size)
    if y1 - y0 < patch_size:
        y0 = max(0, y1 - patch_size)

    x0 = max(0, cx - half)
    x1 = min(w, x0 + patch_size)
    if x1 - x0 < patch_size:
        x0 = max(0, x1 - patch_size)

    p_img = img[y0:y1, x0:x1]
    p_mask = mask[y0:y1, x0:x1]

    # Handle boundary pad if image is smaller than patch_size
    if p_img.shape[0] < patch_size or p_img.shape[1] < patch_size:
        pad_h = patch_size - p_img.shape[0]
        pad_w = patch_size - p_img.shape[1]
        p_img = cv2.copyMakeBorder(p_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        p_mask = cv2.copyMakeBorder(p_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

    return p_img, p_mask


def sample_patch_batch(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    batch_size: int = 8,
    patch_size: int = 256,
    lesion_ratio: float = 0.60,
    augment: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample batch of patches enforcing 60% lesion-centered, 40% background/FOV."""
    rng = np.random.default_rng(seed)
    n_images = len(images)
    n_lesion = int(batch_size * lesion_ratio)
    n_random = batch_size - n_lesion

    batch_x = []
    batch_y = []

    # 1. Lesion-centered patches
    for _ in range(n_lesion):
        img_idx = rng.integers(0, n_images)
        img = images[img_idx]
        m = masks[img_idx]

        # Lesion pixels across any of the 4 channels
        any_lesion = np.any(m > 0, axis=-1)
        y_pts, x_pts = np.where(any_lesion)

        if len(y_pts) > 0:
            pt_idx = rng.integers(0, len(y_pts))
            cy, cx = y_pts[pt_idx], x_pts[pt_idx]
        else:
            cy = rng.integers(patch_size // 2, max(patch_size // 2 + 1, img.shape[0] - patch_size // 2))
            cx = rng.integers(patch_size // 2, max(patch_size // 2 + 1, img.shape[1] - patch_size // 2))

        p_img, p_mask = extract_patch(img, m, cy, cx, patch_size=patch_size)
        if augment:
            p_img, p_mask = augment_patch(p_img, p_mask, rng)
        batch_x.append(p_img)
        batch_y.append(p_mask)

    # 2. Random FOV patches
    for _ in range(n_random):
        img_idx = rng.integers(0, n_images)
        img = images[img_idx]
        m = masks[img_idx]

        cy = rng.integers(patch_size // 2, max(patch_size // 2 + 1, img.shape[0] - patch_size // 2))
        cx = rng.integers(patch_size // 2, max(patch_size // 2 + 1, img.shape[1] - patch_size // 2))

        p_img, p_mask = extract_patch(img, m, cy, cx, patch_size=patch_size)
        if augment:
            p_img, p_mask = augment_patch(p_img, p_mask, rng)
        batch_x.append(p_img)
        batch_y.append(p_mask)

    # Normalize image to [0, 1] float32
    return np.array(batch_x, dtype=np.float32) / 255.0, np.array(batch_y, dtype=np.float32)


def augment_patch(img: np.ndarray, mask: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Data augmentation: flips, 90-degree rotations, brightness/contrast jitter."""
    out_img = img.copy()
    out_mask = mask.copy()

    # Horizontal flip
    if rng.random() > 0.5:
        out_img = np.fliplr(out_img)
        out_mask = np.fliplr(out_mask)

    # Vertical flip
    if rng.random() > 0.5:
        out_img = np.flipud(out_img)
        out_mask = np.flipud(out_mask)

    # 90-degree rotations
    k_rot = rng.integers(0, 4)
    if k_rot > 0:
        out_img = np.rot90(out_img, k=k_rot)
        out_mask = np.rot90(out_mask, k=k_rot)

    # Small brightness/contrast jitter (+-15%)
    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.uniform(-20, 20))
    out_img = np.clip(out_img * alpha + beta, 0, 255).astype(np.uint8)

    return out_img, out_mask
