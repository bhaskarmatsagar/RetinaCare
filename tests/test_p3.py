"""Unit tests for Pipeline P3."""
import cv2
import numpy as np
import pytest

from retinacare.p3_anatomy.optic_disc import locate_optic_disc
from retinacare.p3_anatomy.fovea import locate_fovea


def test_optic_disc_localization_synthetic():
    size = 512
    img = np.full((size, size, 3), (80, 20, 10), dtype=np.uint8)
    fov_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(fov_mask, (256, 256), 240, 255, -1)
    img[fov_mask == 0] = 0

    # Plant synthetic bright optic disc at (350, 250)
    true_disc_center = (350, 250)
    true_radius = 35
    cv2.circle(img, true_disc_center, true_radius, (245, 210, 160), -1)

    found, center, diam, mask, conf, status = locate_optic_disc(img, fov_mask)

    assert found is True
    # Center should be within 15 pixels of true planted disc center
    dist = np.sqrt((center[0] - true_disc_center[0])**2 + (center[1] - true_disc_center[1])**2)
    assert dist < 15.0, f"Detected disc center {center} too far from true {true_disc_center}"
    assert diam > 30.0


def test_fovea_localization_synthetic():
    size = 512
    img = np.full((size, size, 3), (120, 60, 40), dtype=np.uint8)
    fov_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(fov_mask, (256, 256), 240, 255, -1)

    disc_center = (360, 256)
    disc_diam = 70.0
    # Expected fovea ~ 2.5 DD towards FOV center (left)
    # Plant a dark foveal depression at (185, 256)
    true_fovea = (185, 256)
    cv2.circle(img, true_fovea, 15, (50, 20, 10), -1)

    found, f_pt, conf = locate_fovea(img, fov_mask, disc_center, disc_diam)
    assert found is True
    dist = np.sqrt((f_pt[0] - true_fovea[0])**2 + (f_pt[1] - true_fovea[1])**2)
    assert dist < 25.0


def test_blank_image_fails_soft():
    size = 512
    img = np.zeros((size, size, 3), dtype=np.uint8)
    fov_mask = np.zeros((size, size), dtype=np.uint8)

    # Should not raise exception
    found, center, diam, mask, conf, status = locate_optic_disc(img, fov_mask)
    assert status in ["low_confidence", "failed"]
