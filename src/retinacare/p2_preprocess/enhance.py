"""Illumination normalization (Ben Graham), CLAHE, and denoising."""
from typing import Tuple
import cv2
import numpy as np


def apply_ben_graham(img_rgb: np.ndarray, fov_mask: np.ndarray, sigma_frac: float = 0.03) -> np.ndarray:
    """Ben Graham style illumination correction inside FOV mask."""
    w = img_rgb.shape[1]
    sigma = max(1.0, float(sigma_frac * w))

    # Gaussian blur with reflection or default border
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 4*img - 4*blurred + 128
    img_f = img_rgb.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    norm = cv2.addWeighted(img_f, 4.0, blur_f, -4.0, 128.0)
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    # Mask outside FOV to 0
    mask_3c = cv2.merge([fov_mask, fov_mask, fov_mask]) if fov_mask.ndim == 2 else fov_mask
    out = np.where(mask_3c > 0, norm, 0)
    return out


def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CLAHE to L channel in LAB space; also extract CLAHE green channel."""
    # Convert to LAB
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced_rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # Extract green channel of input image and apply CLAHE directly
    green = img_rgb[:, :, 1]
    green_clahe = clahe.apply(green)

    return enhanced_rgb, green_clahe


def denoise_image(img_rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median blur light denoising."""
    if ksize <= 1:
        return img_rgb
    return cv2.medianBlur(img_rgb, ksize)
