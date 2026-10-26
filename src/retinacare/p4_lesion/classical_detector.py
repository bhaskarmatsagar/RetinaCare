"""Classical heuristic lesion detector (HE, MA, EX, SE) using multi-scale morphology."""
from typing import Dict, Tuple, Any
import cv2
import numpy as np
from skimage.filters import frangi


def run_classical_lesion_detector(
    green_clahe: np.ndarray,
    fov_mask: np.ndarray,
    min_area_px: Dict[str, int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Detect candidate lesions using multi-scale morphological filters, Frangi vessel suppression, and disc/border masking."""
    if min_area_px is None:
        min_area_px = {"MA": 3, "HE": 20, "EX": 8, "SE": 30}

    h, w = green_clahe.shape[:2]
    fov_bool = fov_mask > 0

    # 1. Median filter
    filtered = cv2.medianBlur(green_clahe, 3)

    # 2. Vessel suppression via Frangi ridge filter (downscaled for speed then upscaled)
    work_dim = 512
    small_green = cv2.resize(filtered, (work_dim, work_dim), interpolation=cv2.INTER_AREA)
    small_fov = cv2.resize(fov_mask, (work_dim, work_dim), interpolation=cv2.INTER_NEAREST)

    # Frangi filter on green channel (black_ridges=True finds dark vessels)
    try:
        vessel_ridges = frangi(small_green, sigmas=range(1, 4), black_ridges=True)
        vessel_thresh = np.percentile(vessel_ridges[small_fov > 0], 85)
        small_vessels = (vessel_ridges > vessel_thresh).astype(np.uint8) * 255
        vessel_mask = cv2.resize(small_vessels, (w, h), interpolation=cv2.INTER_NEAREST)
        kernel_vessel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        vessel_mask = cv2.dilate(vessel_mask, kernel_vessel)
    except Exception:
        vessel_mask = np.zeros((h, w), dtype=np.uint8)

    # 3. Internal fast optic disc suppression (independent of P3)
    erode_k = max(5, int(w * 0.05))
    kernel_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    fov_inner = cv2.erode(fov_mask, kernel_border)

    # Brightest region in green_clahe
    blur_large = cv2.GaussianBlur(filtered, (0, 0), sigmaX=float(w * 0.03))
    blur_large[fov_inner == 0] = 0
    _, _, _, max_loc = cv2.minMaxLoc(blur_large)
    internal_disc_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(internal_disc_mask, max_loc, int(w * 0.09), 255, -1)

    # Outer border suppression mask (outer 5% ring)
    border_suppression_mask = cv2.bitwise_not(fov_inner)

    # 4. Red Lesions (MA, HE) via Multi-scale Black-Hat
    scales = [9, 15, 25]
    black_hat_max = np.zeros((h, w), dtype=np.uint8)
    for s in scales:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        bh = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, k)
        black_hat_max = np.maximum(black_hat_max, bh)

    # Remove vessels, disc, and border ring
    black_hat_clean = black_hat_max.copy()
    black_hat_clean[vessel_mask > 0] = 0
    black_hat_clean[internal_disc_mask > 0] = 0
    black_hat_clean[border_suppression_mask > 0] = 0
    black_hat_clean[fov_mask == 0] = 0

    # Threshold red lesions (top 0.35% brightest black-hat responses inside FOV)
    valid_red = black_hat_clean[fov_inner > 0]
    thresh_red = np.percentile(valid_red, 99.65) if len(valid_red) > 0 else 30
    red_binary = np.where(black_hat_clean >= thresh_red, 255, 0).astype(np.uint8)

    # Separate MA and HE by connected component area and eccentricity
    num_red, labels_red, stats_red, _ = cv2.connectedComponentsWithStats(red_binary, connectivity=8)
    ma_mask = np.zeros((h, w), dtype=np.uint8)
    he_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_red):
        area = stats_red[i, cv2.CC_STAT_AREA]
        bw = stats_red[i, cv2.CC_STAT_WIDTH]
        bh_dim = stats_red[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh_dim) / (min(bw, bh_dim) + 1e-4)

        if aspect > 3.0:
            # Skip elongated components (likely residual vessel fragments)
            continue

        if min_area_px["MA"] <= area < 20:
            ma_mask[labels_red == i] = 255
        elif area >= min_area_px["HE"] and area < 1500:
            he_mask[labels_red == i] = 255

    # 5. Bright Lesions (EX, SE) via Morphological Top-Hat
    k_bright = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    top_hat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, k_bright)

    top_hat_clean = top_hat.copy()
    top_hat_clean[vessel_mask > 0] = 0
    top_hat_clean[internal_disc_mask > 0] = 0
    top_hat_clean[border_suppression_mask > 0] = 0
    top_hat_clean[fov_mask == 0] = 0

    valid_bright = top_hat_clean[fov_inner > 0]
    thresh_bright = np.percentile(valid_bright, 99.5) if len(valid_bright) > 0 else 35
    bright_binary = np.where(top_hat_clean >= thresh_bright, 255, 0).astype(np.uint8)

    num_br, labels_br, stats_br, _ = cv2.connectedComponentsWithStats(bright_binary, connectivity=8)
    ex_mask = np.zeros((h, w), dtype=np.uint8)
    se_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_br):
        area = stats_br[i, cv2.CC_STAT_AREA]
        if min_area_px["EX"] <= area < 30:
            ex_mask[labels_br == i] = 255
        elif area >= min_area_px["SE"] and area < 2500:
            se_mask[labels_br == i] = 255

    masks = {"MA": ma_mask, "HE": he_mask, "EX": ex_mask, "SE": se_mask}

    # Statistics calculation
    fov_area_total = float(np.sum(fov_bool))
    stats = {}
    for name, m in masks.items():
        n_comp, _, s_comp, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        area_px = int(np.sum(m > 0))
        area_frac = float(area_px / (fov_area_total + 1e-5))
        stats[name] = {
            "count": int(max(0, n_comp - 1)),
            "area_px": area_px,
            "area_frac_fov": round(area_frac, 6)
        }

    return masks, stats
