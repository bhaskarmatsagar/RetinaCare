"""Dataset loader and preprocessor for annotated lesion segmentation datasets (e.g. IDRiD)."""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import pandas as pd


LESION_CLASSES = ["MA", "HE", "EX", "SE"]


def discover_lesion_dataset(data_root: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Scan data_root for paired images and lesion masks. Return mode ('A' or 'B') and paired records."""
    paired_items = []
    data_root = Path(data_root).resolve()

    # Look for IDRiD or similar segmentation structures
    seg_roots = []
    for root, dirs, _ in os.walk(data_root):
        r_name = Path(root).name.lower()
        if "segmentation" in r_name or "groundtruth" in r_name or "masks" in r_name:
            seg_roots.append(Path(root))

    if not seg_roots:
        return "B", []

    # Map candidate mask files
    # IDRiD layout:
    # 1. Original Images / a. Training Set, b. Testing Set
    # 2. All Segmentation Groundtruths / a. Training Set (1. Microaneurysms, 2. Haemorrhages, 3. Hard Exudates, 4. Soft Exudates)
    image_paths: Dict[str, Path] = {}
    mask_paths: Dict[str, Dict[str, Path]] = {}

    for root, _, files in os.walk(data_root):
        r_lower = root.lower()
        for f in files:
            fp = Path(root) / f
            ext = fp.suffix.lower()
            stem = fp.stem
            if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                if "original images" in r_lower or "images" in r_lower:
                    image_paths[stem] = fp
                elif any(tag in r_lower or tag in f.lower() for tag in ["microaneurysm", "haemorrhage", "hemorrhage", "hard exudate", "soft exudate", "_ma", "_he", "_ex", "_se"]):
                    # Identify lesion type
                    fl = f.lower()
                    rl = r_lower
                    cls_found = None
                    if "microaneurysm" in rl or "microaneurysm" in fl or "_ma" in fl:
                        cls_found = "MA"
                    elif "haemorrhage" in rl or "hemorrhage" in rl or "haemorrhage" in fl or "_he" in fl:
                        cls_found = "HE"
                    elif "hard exudate" in rl or "hard_exudate" in fl or "_ex" in fl:
                        cls_found = "EX"
                    elif "soft exudate" in rl or "soft_exudate" in fl or "cotton" in rl or "_se" in fl:
                        cls_found = "SE"

                    if cls_found:
                        # Extract image base stem (e.g. IDRiD_01 from IDRiD_01_MA.tif or IDRiD_01.tif)
                        base_stem = stem
                        for sfx in ["_MA", "_HE", "_EX", "_SE", "_ma", "_he", "_ex", "_se"]:
                            if base_stem.endswith(sfx):
                                base_stem = base_stem[:-len(sfx)]
                        mask_paths.setdefault(base_stem, {})[cls_found] = fp

    # Match images with masks
    for stem, img_path in image_paths.items():
        if stem in mask_paths and len(mask_paths[stem]) > 0:
            paired_items.append({
                "image_id": stem,
                "image_path": str(img_path),
                "masks": {k: str(v) for k, v in mask_paths[stem].items()},
                "split": "train" if "training" in str(img_path).lower() else "test"
            })

    mode = "A" if len(paired_items) >= 20 else "B"
    return mode, paired_items


def load_paired_sample(
    record: Dict[str, Any],
    target_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and 4-channel binary groundtruth mask [MA, HE, EX, SE] resized to target_size."""
    # Read image
    data = np.fromfile(record["image_path"], dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img_rgb.shape[:2]
    # Square crop
    max_dim = max(orig_h, orig_w)
    pad_t = (max_dim - orig_h) // 2
    pad_b = max_dim - orig_h - pad_t
    pad_l = (max_dim - orig_w) // 2
    pad_r = max_dim - orig_w - pad_l

    img_sq = cv2.copyMakeBorder(img_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_resized = cv2.resize(img_sq, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 4-channel mask
    multi_mask = np.zeros((target_size, target_size, 4), dtype=np.float32)

    for i, cls_name in enumerate(LESION_CLASSES):
        m_path = record["masks"].get(cls_name)
        if m_path and Path(m_path).is_file():
            m_data = np.fromfile(m_path, dtype=np.uint8)
            m_img = cv2.imdecode(m_data, cv2.IMREAD_GRAYSCALE)
            if m_img is not None:
                m_sq = cv2.copyMakeBorder(m_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
                m_resized = cv2.resize(m_sq, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                multi_mask[:, :, i] = (m_resized > 0).astype(np.float32)

    return img_resized, multi_mask
