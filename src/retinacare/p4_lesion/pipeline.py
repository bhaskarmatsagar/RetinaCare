"""P4 Lesion Pipeline orchestrator."""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from retinacare.common.case_record import CaseRecord
from retinacare.common.io_utils import read_image_rgb, write_image_png
from retinacare.p4_lesion.classical_detector import run_classical_lesion_detector
from retinacare.p4_lesion.infer import sliding_window_inference, postprocess_lesion_masks


class P4_Pipeline:
    """Pipeline 4: Lesion Segmentation (MA, HE, EX, SE)."""

    def __init__(self, config: Dict[str, Any], model=None, mode: str = "A"):
        self.config = config
        self.model = model
        self.mode = mode
        repo_root = Path(config.get("_repo_root", "."))
        self.artifacts_dir = repo_root / config["paths"]["artifacts"]
        self.p4_dir = self.artifacts_dir / "p4_lesion"
        self.p4_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("retinacare.p4")

    def run_case(self, case: CaseRecord) -> CaseRecord:
        """Run lesion segmentation for a single case."""
        case_id = case.case_id
        out_case_dir = self.p4_dir / case_id
        out_case_dir.mkdir(parents=True, exist_ok=True)

        if case.p2.get("status") != "ok":
            case.p4["status"] = "failed"
            case.p4["error"] = "Prerequisite P2 preprocessing failed or not run"
            return case

        hires_rel = case.p2["paths"].get("hires")
        green_clahe_rel = case.p2["paths"].get("green_clahe")
        fov_rel = case.p2["paths"].get("fov_mask")

        repo_root = Path(self.config.get("_repo_root", "."))
        hires_path = repo_root / hires_rel if hires_rel else None
        green_clahe_path = repo_root / green_clahe_rel if green_clahe_rel else None
        fov_path = repo_root / fov_rel if fov_rel else None

        if not hires_path or not hires_path.is_file() or not fov_path or not fov_path.is_file():
            case.p4["status"] = "failed"
            case.p4["error"] = "Missing required P2 inputs"
            return case

        try:
            img_hires = read_image_rgb(hires_path)
            fov_data = np.fromfile(str(fov_path), dtype=np.uint8)
            fov_mask = cv2.imdecode(fov_data, cv2.IMREAD_GRAYSCALE)

            lesion_cfg = self.config.get("lesion", {})
            min_area_px = lesion_cfg.get("min_area_px", {"MA": 3, "HE": 20, "EX": 8, "SE": 30})

            if self.mode == "A" and self.model is not None:
                # Mode A: Pretrained / fine-tuned U-Net inference
                prob_map = sliding_window_inference(
                    self.model,
                    img_hires,
                    patch_size=lesion_cfg.get("patch_size", 256),
                    stride=lesion_cfg.get("infer_stride", 128),
                    tta=True
                )
                masks, stats = postprocess_lesion_masks(
                    prob_map,
                    fov_mask,
                    min_area_px=min_area_px
                )
                validated = True
            else:
                # Mode B / Classical baseline
                if green_clahe_path and green_clahe_path.is_file():
                    gc_data = np.fromfile(str(green_clahe_path), dtype=np.uint8)
                    green_clahe = cv2.imdecode(gc_data, cv2.IMREAD_GRAYSCALE)
                else:
                    green_clahe = img_hires[:, :, 1]

                masks, stats = run_classical_lesion_detector(
                    green_clahe,
                    fov_mask,
                    min_area_px=min_area_px
                )
                validated = False

            # Save individual lesion masks
            mask_paths = {}
            for cls_name, m_arr in masks.items():
                m_file = out_case_dir / f"{cls_name}.png"
                write_image_png(m_file, m_arr)
                mask_paths[cls_name] = str(m_file.relative_to(self.artifacts_dir.parent).as_posix())

            case.p4["status"] = "ok"
            case.p4["error"] = None
            case.p4["mode"] = self.mode
            case.p4["validated"] = validated
            case.p4["masks"] = mask_paths
            case.p4["stats"] = stats

        except Exception as e:
            case.p4["status"] = "failed"
            case.p4["error"] = str(e)

        return case

    def run_batch(self, cases: List[CaseRecord]) -> List[CaseRecord]:
        """Run lesion segmentation on a batch of cases."""
        results = []
        for c in cases:
            res = self.run_case(c)
            res.save(self.artifacts_dir)
            results.append(res)
        return results
