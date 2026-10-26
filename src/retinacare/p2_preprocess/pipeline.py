"""P2 Preprocessing Pipeline implementation."""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from retinacare.common.case_record import CaseRecord
from retinacare.common.io_utils import read_image_rgb, write_image_png
from retinacare.p2_preprocess.fov import detect_fov
from retinacare.p2_preprocess.crop_resize import crop_to_fov, resize_products
from retinacare.p2_preprocess.enhance import apply_ben_graham, apply_clahe, denoise_image
from retinacare.p2_preprocess.quality import compute_quality_score


class P2_Pipeline:
    """Pipeline 2: Preprocessing and Quality Assessment."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_root = Path(config["paths"]["data_root"]).resolve()
        repo_root = Path(config.get("_repo_root", "."))
        self.artifacts_dir = repo_root / config["paths"]["artifacts"]
        self.p2_dir = self.artifacts_dir / "p2_preprocess"
        self.p2_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("retinacare.p2")

    def run_case(self, case: CaseRecord, variant_raw_crop: bool = False, force: bool = False) -> CaseRecord:
        """Run preprocessing and quality scoring for a single case."""
        case_id = case.case_id
        out_case_dir = self.p2_dir / case_id
        out_case_dir.mkdir(parents=True, exist_ok=True)

        clf_path = out_case_dir / ("clf_raw.png" if variant_raw_crop else "clf.png")
        hires_path = out_case_dir / ("hires_raw.png" if variant_raw_crop else "hires.png")
        fov_mask_path = out_case_dir / "fov_mask.png"
        green_clahe_path = out_case_dir / "green_clahe.png"
        raw_crop_path = out_case_dir / "clf_raw.png"

        # Check if already processed
        if not force and case.p2.get("status") == "ok" and clf_path.is_file() and hires_path.is_file():
            if not raw_crop_path.is_file() and not variant_raw_crop:
                # generate clf_raw.png for E4 if missing
                full_img_path = self.data_root / case.source_path
                img_rgb = read_image_rgb(full_img_path)
                if img_rgb is not None:
                    fov_m = detect_fov(img_rgb)
                    sq_img, _, _ = crop_to_fov(img_rgb, fov_m)
                    write_image_png(raw_crop_path, cv2.resize(sq_img, (256, 256), interpolation=cv2.INTER_AREA))
            return case

        full_img_path = self.data_root / case.source_path
        if not full_img_path.is_file():
            case.p2["status"] = "failed"
            case.p2["error"] = f"Image file not found: {full_img_path}"
            return case

        try:
            # 1. Load image (RGB)
            img_rgb = read_image_rgb(full_img_path)
            if img_rgb is None:
                case.p2["status"] = "failed"
                case.p2["error"] = "Failed to decode image"
                return case

            # 2. Detect FOV
            prep_cfg = self.config.get("preprocess", {})
            fov_thresh = prep_cfg.get("fov_threshold", 10)
            fov_mask = detect_fov(img_rgb, fov_threshold=fov_thresh)

            # Failure check: FOV area < 20%
            fov_frac = float(np.mean(fov_mask > 0))
            if fov_frac < 0.20:
                case.p2["status"] = "failed"
                case.p2["error"] = "fov_failed: FOV area less than 20% of image"
                return case

            # 3. Crop & Square Padding
            square_img, square_mask, frame_info = crop_to_fov(img_rgb, fov_mask, margin_frac=0.02)

            # 4. Quality Scoring
            q_cfg = prep_cfg.get("quality", {})
            q_res = compute_quality_score(
                square_img,
                square_mask,
                weights=q_cfg.get("weights"),
                pass_threshold=q_cfg.get("pass_threshold", 0.50)
            )

            # 5. Enhancement (Ben Graham + CLAHE + Denoise)
            bg_cfg = prep_cfg.get("ben_graham", {})
            if bg_cfg.get("enabled", True) and not variant_raw_crop:
                norm_img = apply_ben_graham(square_img, square_mask, sigma_frac=bg_cfg.get("sigma_frac", 0.03))
            else:
                norm_img = square_img

            clahe_cfg = prep_cfg.get("clahe", {})
            if clahe_cfg.get("enabled", True) and not variant_raw_crop:
                enhanced_img, green_clahe = apply_clahe(
                    norm_img,
                    clip_limit=clahe_cfg.get("clip", 2.0),
                    tile_grid_size=(clahe_cfg.get("tile", 8), clahe_cfg.get("tile", 8))
                )
            else:
                enhanced_img = norm_img
                green_clahe = square_img[:, :, 1]

            ksize = prep_cfg.get("median_ksize", 3)
            if not variant_raw_crop:
                final_enhanced = denoise_image(enhanced_img, ksize=ksize)
            else:
                final_enhanced = enhanced_img

            # 6. Multi-resolution resize
            clf_size = prep_cfg.get("clf_size", 256)
            hires_size = prep_cfg.get("hires_size", 1024)

            clf_img, clf_mask, hires_img, hires_mask, s_clf, s_hires = resize_products(
                final_enhanced,
                square_mask,
                clf_size=clf_size,
                hires_size=hires_size
            )

            # Also resize green_clahe to hires for P3/P4
            green_clahe_hires = cv2.resize(green_clahe, (hires_size, hires_size), interpolation=cv2.INTER_AREA)

            # 7. Save outputs
            clf_path = out_case_dir / ("clf_raw.png" if variant_raw_crop else "clf.png")
            hires_path = out_case_dir / ("hires_raw.png" if variant_raw_crop else "hires.png")
            fov_mask_path = out_case_dir / "fov_mask.png"
            green_clahe_path = out_case_dir / "green_clahe.png"

            write_image_png(clf_path, clf_img)
            write_image_png(hires_path, hires_img)
            write_image_png(fov_mask_path, hires_mask)
            write_image_png(green_clahe_path, green_clahe_hires)

            # Also save raw crop for E4 ablation
            if not variant_raw_crop:
                clf_raw = cv2.resize(square_img, (clf_size, clf_size), interpolation=cv2.INTER_AREA)
                write_image_png(out_case_dir / "clf_raw.png", clf_raw)

            # Update case record
            case.p2["status"] = "ok"
            case.p2["error"] = None
            case.p2["frame"] = {
                "crop_box": frame_info["crop_box"],
                "orig_size": frame_info["orig_size"],
                "scale_clf": s_clf,
                "scale_hires": s_hires
            }
            case.p2["paths"] = {
                "clf": str(clf_path.relative_to(self.artifacts_dir.parent).as_posix()),
                "hires": str(hires_path.relative_to(self.artifacts_dir.parent).as_posix()),
                "fov_mask": str(fov_mask_path.relative_to(self.artifacts_dir.parent).as_posix()),
                "green_clahe": str(green_clahe_path.relative_to(self.artifacts_dir.parent).as_posix())
            }
            case.p2["quality"] = q_res

        except Exception as e:
            case.p2["status"] = "failed"
            case.p2["error"] = str(e)

        return case

    def run_batch(self, cases: List[CaseRecord], variant_raw_crop: bool = False, max_workers: int = 8) -> List[CaseRecord]:
        """Run preprocessing on a batch of cases using multithreading."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        processed = []
        total = len(cases)
        self.logger.info(f"Starting P2 batch preprocessing for {total} cases (max_workers={max_workers})...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {
                executor.submit(self.run_case, case, variant_raw_crop): case
                for case in cases
            }
            completed_count = 0
            for future in as_completed(future_to_case):
                res = future.result()
                res.save(self.artifacts_dir)
                processed.append(res)
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == total:
                    self.logger.info(f"  P2 Progress: {completed_count}/{total} cases completed.")

        return processed
