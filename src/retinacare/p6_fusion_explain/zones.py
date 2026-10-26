"""Lesion zone analysis (Macular <= 1 DD, Perimacular 1-2 DD, Peripheral > 2 DD)."""
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


def compute_lesion_zones(
    lesion_masks: Dict[str, np.ndarray],
    fovea_point: Optional[Tuple[int, int]],
    disc_diameter_px: float,
    fov_mask: np.ndarray
) -> Dict[str, Any]:
    """Classify lesion components into macular, perimacular, and peripheral zones."""
    if fovea_point is None or disc_diameter_px <= 1.0:
        return {"status": "not_available", "zones": {}}

    fx, fy = fovea_point
    dd = float(disc_diameter_px)
    h, w = fov_mask.shape

    zones = {
        "macular": {"count": 0, "area_px": 0, "classes": {}},
        "perimacular": {"count": 0, "area_px": 0, "classes": {}},
        "peripheral": {"count": 0, "area_px": 0, "classes": {}}
    }
    total_lesion_area = 0

    for name, mask in lesion_masks.items():
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            total_lesion_area += area

            dist_px = np.sqrt((cx - fx)**2 + (cy - fy)**2)
            dist_dd = dist_px / dd

            if dist_dd <= 1.0:
                zone_name = "macular"
            elif dist_dd <= 2.0:
                zone_name = "perimacular"
            else:
                zone_name = "peripheral"

            zones[zone_name]["count"] += 1
            zones[zone_name]["area_px"] += int(area)
            zones[zone_name]["classes"].setdefault(name, {"count": 0, "area_px": 0})
            zones[zone_name]["classes"][name]["count"] += 1
            zones[zone_name]["classes"][name]["area_px"] += int(area)

    # Compute shares
    for z in zones.values():
        z["share_of_lesion_area"] = round(float(z["area_px"] / (total_lesion_area + 1e-5)), 4)

    return {"status": "ok", "total_lesion_area_px": int(total_lesion_area), "zones": zones}
