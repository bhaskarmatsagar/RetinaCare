"""Visualization helpers, overlay composition, and color palette."""
import cv2
import numpy as np
from typing import Dict, Tuple

# Appendix B: Exact Palette (RGB)
PALETTE_RGB = {
    "HE": (255, 0, 255),      # Magenta
    "MA": (255, 255, 0),      # Yellow
    "EX": (0, 255, 255),      # Cyan
    "SE": (255, 255, 255),    # White
    "disc": (0, 255, 0),      # Green
    "fovea": (0, 128, 255),   # Blue
}

# In BGR for OpenCV
PALETTE_BGR = {
    k: (v[2], v[1], v[0]) for k, v in PALETTE_RGB.items()
}

ALPHA_CONFIG = {
    "HE": 0.45,
    "MA": 0.60,
    "EX": 0.45,
    "SE": 0.45,
}


def add_legend(img_bgr: np.ndarray, items: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """Draw a compact semi-transparent legend overlay on the image."""
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    box_w = 170
    box_h = 24 * len(items) + 12
    margin = 10
    x0, y0 = w - box_w - margin, margin
    x1, y1 = w - margin, y0 + box_h

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    cv2.rectangle(out, (x0, y0), (x1, y1), (180, 180, 180), 1)

    y = y0 + 18
    for name, color_bgr in items.items():
        cv2.circle(out, (x0 + 15, y - 4), 6, color_bgr, -1)
        cv2.putText(out, name, (x0 + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        y += 24

    return out
