"""Optic Disc localization and segmentation using classical morphology and ellipse fitting."""
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def locate_optic_disc(
    img_rgb: np.ndarray,
    fov_mask: np.ndarray,
    diameter_frac_range: Tuple[float, float] = (0.06, 0.22)
) -> Tuple[bool, Tuple[int, int], float, np.ndarray, float, str]:
    """Locate the optic disc center, diameter, and binary mask at 512x512, returning hires-scaled results."""
    orig_h, orig_w = img_rgb.shape[:2]
    work_size = 512
    scale = work_size / orig_w

    img_work = cv2.resize(img_rgb, (work_size, work_size), interpolation=cv2.INTER_AREA)
    fov_work = cv2.resize(fov_mask, (work_size, work_size), interpolation=cv2.INTER_NEAREST)

    red = img_work[:, :, 0]
    green = img_work[:, :, 1]

    # Approximate FOV diameter
    fov_area = np.sum(fov_work > 0)
    fov_diam = 2.0 * np.sqrt(fov_area / np.pi)

    min_disc_d = fov_diam * diameter_frac_range[0]
    max_disc_d = fov_diam * diameter_frac_range[1]
    min_disc_area = np.pi * (min_disc_d / 2.0) ** 2
    max_disc_area = np.pi * (max_disc_d / 2.0) ** 2

    # Erode FOV mask by ~5% of radius to prevent border reflection false positives
    erode_k = max(3, int(fov_diam * 0.05))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    fov_eroded = cv2.erode(fov_work, kernel_erode)

    # Smooth red channel
    sigma_smooth = max(1.0, float(0.01 * work_size))
    red_smooth = cv2.GaussianBlur(red, (0, 0), sigmaX=sigma_smooth, sigmaY=sigma_smooth)

    # Candidate pixels: top 1.5% brightest in eroded FOV
    eroded_pixels = red_smooth[fov_eroded > 0]
    if len(eroded_pixels) == 0:
        thresh_val = 200
    else:
        thresh_val = np.percentile(eroded_pixels, 98.5)

    candidates = np.where((red_smooth >= thresh_val) & (fov_eroded > 0), 255, 0).astype(np.uint8)
    close_k = max(3, int(min_disc_d * 0.2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)

    best_score = -1.0
    best_candidate = None

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_disc_area * 0.4 or area > max_disc_area * 2.0:
            continue

        comp_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * (area / (perimeter ** 2))

        # Mean red brightness in component
        mean_b = np.mean(red_smooth[comp_mask > 0])
        score = circularity * 0.5 + (mean_b / 255.0) * 0.5

        if score > best_score:
            best_score = score
            best_candidate = (centroids[i], area, contours[0])

    disc_mask_work = np.zeros((work_size, work_size), dtype=np.uint8)

    if best_candidate is not None:
        (cx, cy), area, cnt = best_candidate
        # Window refinement with Otsu
        r_box = int(np.sqrt(area / np.pi) * 1.5)
        x0, y0 = max(0, int(cx - r_box)), max(0, int(cy - r_box))
        x1, y1 = min(work_size, int(cx + r_box)), min(work_size, int(cy + r_box))

        sub_red = red_smooth[y0:y1, x0:x1]
        _, sub_otsu = cv2.threshold(sub_red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        sub_labels_n, sub_labels, sub_stats, sub_centroids = cv2.connectedComponentsWithStats(sub_otsu, connectivity=8)
        if sub_labels_n > 1:
            best_sub = 1 + np.argmax(sub_stats[1:, cv2.CC_STAT_AREA])
            sub_comp = (sub_labels == best_sub).astype(np.uint8)
            sub_cnts, _ = cv2.findContours(sub_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if sub_cnts and len(sub_cnts[0]) >= 5:
                ellipse = cv2.fitEllipse(sub_cnts[0])
                # Shift ellipse to global work coordinates
                shifted_ellipse = ((ellipse[0][0] + x0, ellipse[0][1] + y0), ellipse[1], ellipse[2])
                cv2.ellipse(disc_mask_work, shifted_ellipse, 255, -1)
                cx, cy = int(shifted_ellipse[0][0]), int(shifted_ellipse[0][1])
                diam = float((shifted_ellipse[1][0] + shifted_ellipse[1][1]) / 2.0)
            else:
                cv2.circle(disc_mask_work, (int(cx), int(cy)), int(np.sqrt(area / np.pi)), 255, -1)
                diam = float(2.0 * np.sqrt(area / np.pi))
        else:
            cv2.circle(disc_mask_work, (int(cx), int(cy)), int(np.sqrt(area / np.pi)), 255, -1)
            diam = float(2.0 * np.sqrt(area / np.pi))

        found = True
        status = "ok"
        confidence = float(np.clip(best_score, 0.1, 0.99))

    else:
        # Fallback
        sigma_heavy = max(2.0, float(0.04 * work_size))
        red_heavy = cv2.GaussianBlur(red, (0, 0), sigmaX=sigma_heavy, sigmaY=sigma_heavy)
        red_heavy[fov_eroded == 0] = 0
        min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(red_heavy)
        cx, cy = max_loc
        diam = float(fov_diam * 0.12)
        cv2.circle(disc_mask_work, (int(cx), int(cy)), int(diam / 2.0), 255, -1)
        found = True
        status = "low_confidence"
        confidence = 0.35

    # Scale to hires
    scale_to_hires = orig_w / work_size
    cx_hires = int(cx * scale_to_hires)
    cy_hires = int(cy * scale_to_hires)
    diam_hires = float(diam * scale_to_hires)
    disc_mask_hires = cv2.resize(disc_mask_work, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return found, (cx_hires, cy_hires), diam_hires, disc_mask_hires, confidence, status
