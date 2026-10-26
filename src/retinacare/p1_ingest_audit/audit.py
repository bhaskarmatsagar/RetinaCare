"""Audit metrics calculation, contact sheet generation, and audit report generation."""
import json
from pathlib import Path
from typing import Dict, List, Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def audit_images(inventory_df: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
    """Compute per-image statistical attributes for image quality and camera discovery."""
    image_rows = inventory_df[inventory_df["category"] == "image"].copy()
    if len(image_rows) > max_samples:
        # Sample for fast, representative audit while respecting all files in inventory
        sample_subset = image_rows.sample(n=max_samples, random_state=42)
    else:
        sample_subset = image_rows

    results = []
    for _, row in sample_subset.iterrows():
        p = Path(row["full_path"])
        try:
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue

            h, w, c = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Brightness and contrast
            mean_b = float(np.mean(gray))
            std_c = float(np.std(gray))

            # Blur / sharpness (variance of Laplacian on green channel)
            green = img[:, :, 1]
            lap_var = float(cv2.Laplacian(green, cv2.CV_64F).var())

            # Black border fraction (pixels with intensity < 10)
            black_frac = float(np.mean(gray < 10))

            results.append({
                "image_id": row["stem"],
                "path": row["path"],
                "full_path": row["full_path"],
                "source_group": row["source_group"],
                "width": w,
                "height": h,
                "aspect_ratio": round(w / h, 3),
                "mean_brightness": mean_b,
                "contrast": std_c,
                "sharpness": lap_var,
                "black_border_fraction": black_frac
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def create_contact_sheet(
    image_paths: List[str],
    out_path: Path,
    titles: List[str] = None,
    cols: int = 5,
    thumb_size: int = 200
) -> None:
    """Assemble a grid contact sheet from a list of image paths."""
    n = len(image_paths)
    if n == 0:
        return
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)

    for idx, p_str in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        p = Path(p_str)
        try:
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                if titles and idx < len(titles):
                    cv2.putText(thumb, str(titles[idx])[:18], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                sheet[r * thumb_size : (r + 1) * thumb_size, c * thumb_size : (c + 1) * thumb_size] = thumb
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
