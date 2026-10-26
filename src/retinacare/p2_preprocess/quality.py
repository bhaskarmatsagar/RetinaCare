"""Image quality scoring and clinical acceptability assessment."""
from typing import Dict, Any, List
import cv2
import numpy as np


def compute_quality_score(
    cropped_rgb: np.ndarray,
    fov_mask: np.ndarray,
    weights: Dict[str, float] = None,
    pass_threshold: float = 0.50
) -> Dict[str, Any]:
    """Compute multi-factor objective image quality score inside FOV."""
    if weights is None:
        weights = {"sharpness": 0.35, "illumination": 0.25, "contrast": 0.20, "fov_coverage": 0.20}

    fov_pixels = fov_mask > 0
    num_fov = np.sum(fov_pixels)
    total_pixels = fov_mask.size

    if num_fov < 50:
        return {
            "score": 0.0,
            "sharpness": 0.0,
            "illumination": 0.0,
            "contrast": 0.0,
            "fov_coverage": 0.0,
            "pass": False,
            "reasons": ["insufficient_fov"]
        }

    green = cropped_rgb[:, :, 1]
    green_fov = green[fov_pixels]

    # 1. Sharpness: variance of Laplacian on green channel
    lap = cv2.Laplacian(green, cv2.CV_64F)
    lap_var = float(np.var(lap[fov_pixels]))
    # Normalise with smooth sigmoid saturating around ~400
    sharpness_score = float(np.clip(lap_var / 400.0, 0.0, 1.0))

    # 2. Illumination: target mean intensity ~ 110-140, penalise quadrant disparity
    mean_val = float(np.mean(green_fov))
    target_mean = 120.0
    dist_from_target = abs(mean_val - target_mean)
    mean_score = max(0.0, 1.0 - (dist_from_target / 90.0))

    # Quadrant uniformity
    h, w = cropped_rgb.shape[:2]
    mid_y, mid_x = h // 2, w // 2
    quad_means = []
    for q_mask in [
        fov_pixels[:mid_y, :mid_x],
        fov_pixels[:mid_y, mid_x:],
        fov_pixels[mid_y:, :mid_x],
        fov_pixels[mid_y:, mid_x:]
    ]:
        if np.sum(q_mask) > 10:
            quad_means.append(np.mean(green[:mid_y, :mid_x][q_mask] if q_mask.shape == green[:mid_y, :mid_x].shape else 120))
    uniformity_score = 1.0 - (float(np.std(quad_means)) / 50.0) if len(quad_means) >= 2 else 0.8
    uniformity_score = float(np.clip(uniformity_score, 0.0, 1.0))
    illumination_score = float(0.6 * mean_score + 0.4 * uniformity_score)

    # 3. Contrast: standard deviation of green channel inside FOV
    std_val = float(np.std(green_fov))
    contrast_score = float(np.clip((std_val - 15.0) / 45.0, 0.0, 1.0))

    # 4. FOV coverage: FOV area divided by square crop area
    fov_cov = float(num_fov / total_pixels)
    # Retinal circle in square crop has theoretical max ~ pi/4 = 0.785
    fov_coverage_score = float(np.clip(fov_cov / 0.70, 0.0, 1.0))

    # Weighted combination
    score = (
        weights.get("sharpness", 0.35) * sharpness_score +
        weights.get("illumination", 0.25) * illumination_score +
        weights.get("contrast", 0.20) * contrast_score +
        weights.get("fov_coverage", 0.20) * fov_coverage_score
    )
    score = float(np.clip(score, 0.0, 1.0))

    reasons = []
    if sharpness_score < 0.40:
        reasons.append("low_sharpness_blur")
    if illumination_score < 0.40:
        reasons.append("suboptimal_illumination")
    if contrast_score < 0.40:
        reasons.append("low_contrast")
    if fov_coverage_score < 0.40:
        reasons.append("truncated_fov")

    passed = bool(score >= pass_threshold)

    return {
        "score": round(score, 3),
        "sharpness": round(sharpness_score, 3),
        "illumination": round(illumination_score, 3),
        "contrast": round(contrast_score, 3),
        "fov_coverage": round(fov_coverage_score, 3),
        "pass": passed,
        "reasons": reasons
    }
