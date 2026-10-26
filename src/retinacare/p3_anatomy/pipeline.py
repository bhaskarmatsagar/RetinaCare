"""P3 Anatomy Pipeline orchestrator."""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from retinacare.common.case_record import CaseRecord
from retinacare.common.io_utils import read_image_rgb, write_image_png
from retinacare.p3_anatomy.optic_disc import locate_optic_disc
from retinacare.p3_anatomy.fovea import locate_fovea


class P3_Pipeline:
    """Pipeline 3: Anatomy Localization (Optic Disc and Fovea)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        repo_root = Path(config.get("_repo_root", "."))
        self.artifacts_dir = repo_root / config["paths"]["artifacts"]
        self.p3_dir = self.artifacts_dir / "p3_anatomy"
        self.p3_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("retinacare.p3")

    def run_case(self, case: CaseRecord, force: bool = False) -> CaseRecord:
        """Run anatomy localization for a single case."""
        case_id = case.case_id
        out_case_dir = self.p3_dir / case_id
        out_case_dir.mkdir(parents=True, exist_ok=True)

        if not force and case.p3.get("status") in ["ok", "low_confidence"]:
            return case

        # Check P2 status
        if case.p2.get("status") != "ok":
            case.p3["status"] = "failed"
            case.p3["error"] = "Prerequisite P2 preprocessing failed or not run"
            return case

        hires_rel = case.p2["paths"].get("hires")
        fov_rel = case.p2["paths"].get("fov_mask")

        if not hires_rel or not fov_rel:
            case.p3["status"] = "failed"
            case.p3["error"] = "Missing P2 image paths"
            return case

        hires_path = Path(self.config.get("_repo_root", ".")) / hires_rel
        fov_path = Path(self.config.get("_repo_root", ".")) / fov_rel

        try:
            img_rgb = read_image_rgb(hires_path)
            fov_data = np.fromfile(str(fov_path), dtype=np.uint8)
            fov_mask = cv2.imdecode(fov_data, cv2.IMREAD_GRAYSCALE)

            if img_rgb is None or fov_mask is None:
                case.p3["status"] = "failed"
                case.p3["error"] = "Failed to load hires image or FOV mask"
                return case

            anat_cfg = self.config.get("anatomy", {})
            dd_range = tuple(anat_cfg.get("disc_diameter_frac_range", [0.06, 0.22]))
            fovea_range = tuple(anat_cfg.get("fovea_distance_dd_range", [2.0, 3.0]))

            # 1. Optic Disc
            disc_found, disc_center, disc_diam, disc_mask, disc_conf, disc_status = locate_optic_disc(
                img_rgb, fov_mask, diameter_frac_range=dd_range
            )

            # Save disc mask
            disc_mask_path = out_case_dir / "disc_mask.png"
            write_image_png(disc_mask_path, disc_mask)

            # 2. Fovea
            if disc_found:
                fovea_found, fovea_point, fovea_conf = locate_fovea(
                    img_rgb, fov_mask, disc_center, disc_diam, fovea_distance_dd_range=fovea_range
                )
            else:
                fovea_found, fovea_point, fovea_conf = False, (0, 0), 0.0

            # 3. Plausibility Check
            # Check disc diameter within allowed range of FOV diameter
            fov_area = np.sum(fov_mask > 0)
            fov_diam = 2.0 * np.sqrt(fov_area / np.pi)
            disc_ratio = disc_diam / (fov_diam + 1e-5)
            disc_plausible = dd_range[0] * 0.8 <= disc_ratio <= dd_range[1] * 1.2

            # Check fovea distance in DD
            fovea_dist = np.sqrt((disc_center[0] - fovea_point[0])**2 + (disc_center[1] - fovea_point[1])**2)
            dist_in_dd = fovea_dist / (disc_diam + 1e-5)
            fovea_plausible = fovea_range[0] * 0.7 <= dist_in_dd <= fovea_range[1] * 1.3

            # Check points inside FOV
            h, w = fov_mask.shape
            pts_in_fov = (
                0 <= disc_center[0] < w and 0 <= disc_center[1] < h and fov_mask[disc_center[1], disc_center[0]] > 0 and
                0 <= fovea_point[0] < w and 0 <= fovea_point[1] < h and fov_mask[fovea_point[1], fovea_point[0]] > 0
            )

            plausible = bool(disc_found and fovea_found and disc_plausible and fovea_plausible and pts_in_fov)

            overall_status = "ok" if (disc_found and fovea_found and plausible) else ("low_confidence" if disc_found else "failed")

            case.p3["status"] = overall_status
            case.p3["error"] = None
            case.p3["disc"] = {
                "found": bool(disc_found),
                "center": [int(disc_center[0]), int(disc_center[1])],
                "diameter_px": round(float(disc_diam), 1),
                "mask_path": str(disc_mask_path.relative_to(self.artifacts_dir.parent).as_posix()),
                "confidence": round(float(disc_conf), 3)
            }
            case.p3["fovea"] = {
                "found": bool(fovea_found),
                "point": [int(fovea_point[0]), int(fovea_point[1])],
                "confidence": round(float(fovea_conf), 3)
            }
            case.p3["plausible"] = plausible

        except Exception as e:
            case.p3["status"] = "failed"
            case.p3["error"] = str(e)

        return case

    def run_batch(self, cases: List[CaseRecord], max_workers: int = 8) -> List[CaseRecord]:
        """Run anatomy localization for a batch of cases using multithreading."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = []
        total = len(cases)
        self.logger.info(f"Starting P3 batch anatomy localization for {total} cases (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {executor.submit(self.run_case, c): c for c in cases}
            completed_count = 0
            for future in as_completed(future_to_case):
                res = future.result()
                res.save(self.artifacts_dir)
                results.append(res)
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == total:
                    self.logger.info(f"  P3 Progress: {completed_count}/{total} cases localized.")
        return results
