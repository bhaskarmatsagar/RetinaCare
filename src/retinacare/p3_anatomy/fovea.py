"""Fovea localization based on anatomical geometric relation to the Optic Disc."""
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def locate_fovea(
    img_rgb: np.ndarray,
    fov_mask: np.ndarray,
    disc_center: Tuple[int, int],
    disc_diameter_px: float,
    fovea_distance_dd_range: Tuple[float, float] = (2.0, 3.0)
) -> Tuple[bool, Tuple[int, int], float]:
    """Locate fovea point using vessel suppression and darkest smooth region search."""
    h, w = img_rgb.shape[:2]
    green = img_rgb[:, :, 1]
    disc_x, disc_y = disc_center
    fov_cx, fov_cy = w // 2, h // 2

    # Vector from disc to FOV center
    v_x = fov_cx - disc_x
    v_y = fov_cy - disc_y
    v_norm = np.sqrt(v_x**2 + v_y**2)
    if v_norm < 1e-4:
        # Disc is at center, direction arbitrary (e.g. left/temporal)
        dir_x, dir_y = -1.0, 0.0
    else:
        dir_x, dir_y = v_x / v_norm, v_y / v_norm

    # Base angle towards FOV center
    base_angle = np.arctan2(dir_y, dir_x)

    # Search sector mask
    min_dist = disc_diameter_px * fovea_distance_dd_range[0]
    max_dist = disc_diameter_px * fovea_distance_dd_range[1]
    angular_tol = np.radians(25.0)

    # Blur green channel heavily to suppress vessels
    sigma = max(2.0, float(0.4 * disc_diameter_px))
    green_blur = cv2.GaussianBlur(green, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Grid of coordinates
    y_coords, x_coords = np.ogrid[:h, :w]
    dx = x_coords - disc_x
    dy = y_coords - disc_y
    dist = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)

    # Angular difference wrapped to [-pi, pi]
    angle_diff = np.arctan2(np.sin(angles - base_angle), np.cos(angles - base_angle))

    sector_mask = (
        (dist >= min_dist) &
        (dist <= max_dist) &
        (np.abs(angle_diff) <= angular_tol) &
        (fov_mask > 0)
    )

    if np.sum(sector_mask) < 20:
        # Fallback to direct geometric projection
        proj_x = int(disc_x + 2.5 * disc_diameter_px * dir_x)
        proj_y = int(disc_y + 2.5 * disc_diameter_px * dir_y)
        proj_x = int(np.clip(proj_x, 0, w - 1))
        proj_y = int(np.clip(proj_y, 0, h - 1))
        return True, (proj_x, proj_y), 0.30

    # Find minimum intensity in sector
    sector_values = green_blur[sector_mask]
    min_val = np.min(sector_values)
    mean_val = np.mean(sector_values)

    # Coordinates where value is min
    y_cand, x_cand = np.where(sector_mask & (green_blur == min_val))
    fovea_x, fovea_y = int(x_cand[0]), int(y_cand[0])

    # Confidence based on local contrast
    contrast = (mean_val - min_val) / (mean_val + 1e-5)
    confidence = float(np.clip(contrast * 3.0, 0.20, 0.95))

    return True, (fovea_x, fovea_y), confidence
