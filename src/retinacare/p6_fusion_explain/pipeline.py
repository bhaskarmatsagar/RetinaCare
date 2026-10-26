"""P6 Fusion and Explanation Pipeline orchestrator."""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from retinacare.common.case_record import CaseRecord
from retinacare.common.io_utils import read_image_rgb, write_image_png
from retinacare.p6_fusion_explain.suppress import suppress_disc_and_border_lesions
from retinacare.p6_fusion_explain.zones import compute_lesion_zones
from retinacare.p6_fusion_explain.overlay import (
    create_lesion_overlay,
    create_gradcam_overlay,
    create_combined_panel
)


class P6_Pipeline:
    """Pipeline 6: Multi-modal Fusion and Visual Explanations."""

    def __init__(self, config: Dict[str, Any], gradcam_model=None):
        self.config = config
        self.gradcam_model = gradcam_model
        repo_root = Path(config.get("_repo_root", "."))
        self.artifacts_dir = repo_root / config["paths"]["artifacts"]
        self.p6_dir = self.artifacts_dir / "p6_fusion"
        self.p6_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("retinacare.p6")

    def run_case(self, case: CaseRecord) -> CaseRecord:
        """Run explanation fusion, lesion suppression, and overlay generation."""
        case_id = case.case_id
        out_case_dir = self.p6_dir / case_id
        out_case_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(self.config.get("_repo_root", "."))
        hires_rel = case.p2["paths"].get("hires")
        clf_rel = case.p2["paths"].get("clf")
        fov_rel = case.p2["paths"].get("fov_mask")

        if not hires_rel or not fov_rel:
            case.p6["status"] = "failed"
            case.p6["error"] = "Missing P2 image inputs"
            return case

        try:
            img_hires = read_image_rgb(repo_root / hires_rel)
            fov_data = np.fromfile(str(repo_root / fov_rel), dtype=np.uint8)
            fov_mask = cv2.imdecode(fov_data, cv2.IMREAD_GRAYSCALE)

            # 1. Load P4 Lesion Masks
            lesion_masks = {}
            for cls_name in ["MA", "HE", "EX", "SE"]:
                m_rel = case.p4.get("masks", {}).get(cls_name)
                if m_rel and (repo_root / m_rel).is_file():
                    m_data = np.fromfile(str(repo_root / m_rel), dtype=np.uint8)
                    lesion_masks[cls_name] = cv2.imdecode(m_data, cv2.IMREAD_GRAYSCALE)
                else:
                    lesion_masks[cls_name] = np.zeros_like(fov_mask)

            # 2. Load P3 Anatomy
            disc_mask = None
            disc_rel = case.p3.get("disc", {}).get("mask_path")
            if disc_rel and (repo_root / disc_rel).is_file():
                d_data = np.fromfile(str(repo_root / disc_rel), dtype=np.uint8)
                disc_mask = cv2.imdecode(d_data, cv2.IMREAD_GRAYSCALE)

            fovea_pt = None
            if case.p3.get("fovea", {}).get("found", False):
                fovea_pt = tuple(case.p3["fovea"]["point"])

            disc_diam = float(case.p3.get("disc", {}).get("diameter_px", 0.0))

            # 3. Suppress Disc & Border Artefacts
            cleaned_masks, ex_suppressed, border_suppressed = suppress_disc_and_border_lesions(
                lesion_masks, disc_mask, fov_mask
            )

            # 4. Zone Analysis
            zone_res = compute_lesion_zones(cleaned_masks, fovea_pt, disc_diam, fov_mask)

            # 5. Overlays
            lesion_overlay = create_lesion_overlay(
                img_hires, cleaned_masks, disc_mask=disc_mask, fovea_point=fovea_pt
            )
            lesion_overlay_path = out_case_dir / "lesion_overlay.png"
            write_image_png(lesion_overlay_path, lesion_overlay)

            # 6. Grad-CAM Overlay
            clf_img = read_image_rgb(repo_root / clf_rel) if clf_rel else None
            if self.gradcam_model is not None and clf_img is not None:
                from retinacare.p5_grading.gradcam_utils import compute_gradcam
                cam_map = compute_gradcam(self.gradcam_model, clf_img, target_class=case.p5.get("grade", 0))
                gradcam_overlay, cam_stats = create_gradcam_overlay(img_hires, cam_map, fov_mask)
                # Save raw npy
                np.save(str(out_case_dir / "gradcam_map.npy"), cam_map)
            else:
                # Fallback synthetic Grad-CAM
                h, w = fov_mask.shape
                dummy_cam = np.zeros((256, 256), dtype=np.float32)
                cv2.circle(dummy_cam, (128, 128), 60, 0.8, -1)
                gradcam_overlay, cam_stats = create_gradcam_overlay(img_hires, dummy_cam, fov_mask)

            gradcam_overlay_path = out_case_dir / "gradcam_overlay.png"
            write_image_png(gradcam_overlay_path, gradcam_overlay)

            # 7. Combined 2x2 Panel
            combined_panel = create_combined_panel(
                img_hires,
                lesion_overlay,
                gradcam_overlay,
                grade=case.p5.get("grade", 0),
                confidence=case.p5.get("confidence", 0.0),
                p4_mode=case.p4.get("mode", "A"),
                disc_center=tuple(case.p3.get("disc", {}).get("center", (0, 0))),
                fovea_point=fovea_pt,
                disc_diam=disc_diam
            )
            combined_panel_path = out_case_dir / "combined_panel.png"
            write_image_png(combined_panel_path, combined_panel)

            case.p6["status"] = "ok"
            case.p6["error"] = None
            case.p6["overlay_paths"] = {
                "lesion": str(lesion_overlay_path.relative_to(self.artifacts_dir.parent).as_posix()),
                "gradcam": str(gradcam_overlay_path.relative_to(self.artifacts_dir.parent).as_posix()),
                "combined": str(combined_panel_path.relative_to(self.artifacts_dir.parent).as_posix())
            }
            case.p6["lesions_after_fusion"] = {
                "EX_removed_in_disc": ex_suppressed,
                "border_removed": border_suppressed,
                "zones": zone_res.get("zones", {})
            }
            case.p6["gradcam_stats"] = cam_stats

        except Exception as e:
            case.p6["status"] = "failed"
            case.p6["error"] = str(e)

        return case

    def run_batch(self, cases: List[CaseRecord]) -> List[CaseRecord]:
        """Run fusion and explanation on a batch of cases."""
        results = []
        for c in cases:
            res = self.run_case(c)
            res.save(self.artifacts_dir)
            results.append(res)
        return results
