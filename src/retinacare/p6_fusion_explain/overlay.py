"""Overlays and 2x2 multi-modal explanation panels."""
from typing import Dict, Tuple, Optional, Any
import cv2
import numpy as np

from retinacare.common.viz import PALETTE_BGR, ALPHA_CONFIG, add_legend


def create_lesion_overlay(
    img_rgb: np.ndarray,
    lesion_masks: Dict[str, np.ndarray],
    disc_mask: Optional[np.ndarray] = None,
    fovea_point: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Create clinician lesion overlay with color palette and legend."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()

    # Draw filled lesion masks
    active_items = {}
    for name, mask in lesion_masks.items():
        if np.sum(mask > 0) == 0:
            continue
        color_bgr = PALETTE_BGR[name]
        alpha = ALPHA_CONFIG.get(name, 0.45)

        m_draw = mask.copy()
        if name == "MA":
            # Dilate 2px for visibility per Appendix B
            k_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m_draw = cv2.dilate(m_draw, k_ma)

        # Blend
        color_layer = np.full_like(img_bgr, color_bgr, dtype=np.uint8)
        mask_bool = m_draw > 0
        overlay[mask_bool] = (overlay[mask_bool] * (1.0 - alpha) + color_layer[mask_bool] * alpha).astype(np.uint8)
        active_items[name] = color_bgr

    # Draw Optic Disc outline (green, 2px)
    if disc_mask is not None and np.sum(disc_mask > 0) > 0:
        contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, PALETTE_BGR["disc"], 2)
        active_items["Optic Disc"] = PALETTE_BGR["disc"]

    # Draw Fovea cross (blue)
    if fovea_point is not None and fovea_point[0] > 0 and fovea_point[1] > 0:
        fx, fy = fovea_point
        arm = 12
        cv2.line(overlay, (fx - arm, fy), (fx + arm, fy), PALETTE_BGR["fovea"], 2)
        cv2.line(overlay, (fx, fy - arm), (fx, fy + arm), PALETTE_BGR["fovea"], 2)
        cv2.circle(overlay, (fx, fy), 4, PALETTE_BGR["fovea"], 1)
        active_items["Fovea"] = PALETTE_BGR["fovea"]

    # Add legend
    out = add_legend(overlay, active_items)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def create_gradcam_overlay(
    img_rgb: np.ndarray,
    cam_heatmap_256: np.ndarray,
    fov_mask: np.ndarray,
    alpha: float = 0.40
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Upsample Grad-CAM, apply JET colormap, mask to FOV, and calculate sanity statistics."""
    h, w = img_rgb.shape[:2]
    cam_hires = cv2.resize(cam_heatmap_256, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_hires = np.clip(cam_hires, 0.0, 1.0)

    # Sanity statistics
    total_mass = float(np.sum(cam_hires)) + 1e-7
    fov_bool = fov_mask > 0
    mass_in_fov = float(np.sum(cam_hires[fov_bool])) / total_mass

    # Border ring (outer 8% radius)
    fov_area = np.sum(fov_bool)
    fov_r = np.sqrt(fov_area / np.pi)
    erode_k = max(3, int(fov_r * 0.08 * 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    fov_inner = cv2.erode(fov_mask, kernel)
    border_ring = (fov_mask > 0) & (fov_inner == 0)
    mass_near_border = float(np.sum(cam_hires[border_ring])) / total_mass

    sanity_stats = {
        "mass_in_fov": round(mass_in_fov, 4),
        "mass_near_border": round(mass_near_border, 4)
    }

    # Colormap
    cam_uint8 = (cam_hires * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend inside FOV
    img_f = img_rgb.astype(np.float32)
    heat_f = heatmap_rgb.astype(np.float32)
    blended = img_f * (1.0 - alpha) + heat_f * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out = np.where(fov_mask[:, :, np.newaxis] > 0, blended, img_rgb)
    return out, sanity_stats


def create_combined_panel(
    original_rgb: np.ndarray,
    lesion_overlay_rgb: np.ndarray,
    gradcam_overlay_rgb: np.ndarray,
    grade: int,
    confidence: float,
    p4_mode: str,
    disc_center: Optional[Tuple[int, int]] = None,
    fovea_point: Optional[Tuple[int, int]] = None,
    disc_diam: float = 0.0
) -> np.ndarray:
    """Create 2x2 summary panel with title banners."""
    target_side = 512
    p1 = cv2.resize(original_rgb, (target_side, target_side), interpolation=cv2.INTER_AREA)
    p2 = cv2.resize(lesion_overlay_rgb, (target_side, target_side), interpolation=cv2.INTER_AREA)
    p3 = cv2.resize(gradcam_overlay_rgb, (target_side, target_side), interpolation=cv2.INTER_AREA)

    # Fourth panel: Zones and anatomy
    p4 = p1.copy()
    if fovea_point is not None and disc_diam > 1.0:
        scale_f = target_side / original_rgb.shape[1]
        fx = int(fovea_point[0] * scale_f)
        fy = int(fovea_point[1] * scale_f)
        dd = disc_diam * scale_f

        # Concentric zone circles: 1 DD (macular), 2 DD (perimacular)
        cv2.circle(p4, (fx, fy), int(dd), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(p4, (fx, fy), int(2.0 * dd), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(p4, "Macular (<=1 DD)", (fx + 10, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    # Assemble 2x2 grid
    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bottom_row])

    # Add header banner
    banner_h = 60
    banner = np.full((banner_h, grid.shape[1], 3), (25, 25, 30), dtype=np.uint8)
    mode_text = "Validated Model" if p4_mode == "A" else "Heuristic Detector (Unvalidated)"
    info_text = f"RetinaCare AI | Predicted Grade: {grade} (Conf: {confidence:.2f}) | Mode: {mode_text}"
    cv2.putText(banner, info_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    combined = np.vstack([banner, grid])
    return combined
