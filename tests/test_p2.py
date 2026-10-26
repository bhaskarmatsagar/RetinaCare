"""Unit tests for Pipeline P2."""
import cv2
import numpy as np
import pytest

from retinacare.p2_preprocess.fov import detect_fov
from retinacare.p2_preprocess.crop_resize import crop_to_fov, resize_products
from retinacare.p2_preprocess.enhance import apply_ben_graham, apply_clahe, denoise_image
from retinacare.p2_preprocess.quality import compute_quality_score


def create_synthetic_retina(size=400, bg_val=0, retina_color=(180, 80, 40)):
    """Create synthetic circular retina image on black background."""
    img = np.full((size, size, 3), bg_val, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.42)
    cv2.circle(img, center, radius, retina_color, -1)
    cv2.circle(mask, center, radius, 255, -1)

    # Add synthetic details / vessels
    cv2.line(img, (center[0]-50, center[1]), (center[0]+50, center[1]), (40, 20, 10), 3)
    cv2.circle(img, (center[0] + 60, center[1]), 25, (240, 200, 100), -1)  # optic disc
    cv2.circle(img, (center[0] - 40, center[1]), 12, (100, 40, 20), -1)   # fovea
    return img, mask


def test_fov_detection_synthetic():
    img, true_mask = create_synthetic_retina(size=300)
    det_mask = detect_fov(img, fov_threshold=10)
    assert det_mask.shape == true_mask.shape
    overlap = np.sum((det_mask > 0) & (true_mask > 0)) / np.sum(true_mask > 0)
    assert overlap > 0.95


def test_crop_and_square_padding():
    img, mask = create_synthetic_retina(size=350)
    square_img, square_mask, frame = crop_to_fov(img, mask)
    assert square_img.shape[0] == square_img.shape[1], "Cropped output must be square"
    assert square_mask.shape[0] == square_mask.shape[1], "Cropped mask must be square"
    assert frame["crop_box"][2] > frame["crop_box"][0]
    assert frame["crop_box"][3] > frame["crop_box"][1]


def test_quality_score_sharpness_separation():
    img, mask = create_synthetic_retina(size=300)
    q_sharp = compute_quality_score(img, mask)

    # Heavily blurred copy
    blurred = cv2.GaussianBlur(img, (25, 25), 8)
    q_blur = compute_quality_score(blurred, mask)

    assert q_sharp["sharpness"] > q_blur["sharpness"], "Sharp image must have higher sharpness score than blurred"
    assert "low_sharpness_blur" in q_blur["reasons"]


def test_quality_score_illumination_separation():
    img, mask = create_synthetic_retina(size=300)
    q_normal = compute_quality_score(img, mask)

    # Darkened copy
    dark = (img.astype(np.float32) * 0.15).astype(np.uint8)
    q_dark = compute_quality_score(dark, mask)

    assert q_normal["illumination"] > q_dark["illumination"]
    assert "suboptimal_illumination" in q_dark["reasons"]


def test_preprocessing_determinism():
    img, mask = create_synthetic_retina(size=300)
    bg1 = apply_ben_graham(img, mask, sigma_frac=0.03)
    bg2 = apply_ben_graham(img, mask, sigma_frac=0.03)
    np.testing.assert_array_equal(bg1, bg2)
