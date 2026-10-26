"""File I/O and cryptographic hashing utilities."""
import hashlib
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file's byte contents."""
    p = Path(file_path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def read_image_rgb(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Read an image file safely and return in RGB uint8 format."""
    p = Path(file_path)
    if not p.is_file():
        return None
    # Use numpy fromfile to safely support Unicode paths on Windows
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def write_image_png(file_path: Union[str, Path], img_rgb_or_gray: np.ndarray) -> bool:
    """Write an image array to a lossless PNG file safely."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if img_rgb_or_gray.ndim == 3:
            img_to_encode = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2BGR)
        else:
            img_to_encode = img_rgb_or_gray
        success, buf = cv2.imencode(".png", img_to_encode)
        if not success:
            return False
        with open(p, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False
