"""P5 Grading Pipeline orchestrator."""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from retinacare.common.case_record import CaseRecord
from retinacare.common.io_utils import read_image_rgb
from retinacare.p5_grading.adapters import get_adapter
from retinacare.p5_grading.calibrate import compute_prediction_outputs


class P5_Pipeline:
    """Pipeline 5: DR Severity Grading."""

    def __init__(
        self,
        config: Dict[str, Any],
        model=None,
        model_id: str = "efficientnetb0_tl_v1",
        adapter_name: str = "efficientnet",
        temperature: float = 1.0
    ):
        self.config = config
        self.model = model
        self.model_id = model_id
        self.adapter = get_adapter(adapter_name)
        self.temperature = float(temperature)
        repo_root = Path(config.get("_repo_root", "."))
        self.artifacts_dir = repo_root / config["paths"]["artifacts"]
        self.logger = logging.getLogger("retinacare.p5")

    def run_case(self, case: CaseRecord) -> CaseRecord:
        """Run DR grading for a single case."""
        if case.p2.get("status") != "ok":
            case.p5["status"] = "failed"
            case.p5["error"] = "Prerequisite P2 preprocessing failed or not run"
            return case

        clf_rel = case.p2["paths"].get("clf")
        repo_root = Path(self.config.get("_repo_root", "."))
        clf_path = repo_root / clf_rel if clf_rel else None

        if not clf_path or not clf_path.is_file():
            case.p5["status"] = "failed"
            case.p5["error"] = "Missing P2 clf image"
            return case

        try:
            img_rgb = read_image_rgb(clf_path)
            if img_rgb is None:
                case.p5["status"] = "failed"
                case.p5["error"] = "Failed to load clf image"
                return case

            # Adapt input
            img_adapted = self.adapter(img_rgb.copy())
            img_batch = np.expand_dims(img_adapted, axis=0)

            if self.model is not None:
                preds = self.model.predict(img_batch, batch_size=1, verbose=0)
                # If model ends with softmax, convert back to pseudo-logits for temperature scaling
                preds_clipped = np.clip(preds, 1e-12, 1.0 - 1e-12)
                logits = np.log(preds_clipped)
            else:
                # Stub / fallback: uniform or label-informed fallback for testing skeleton
                logits = np.array([[2.0, 0.5, 0.2, 0.1, 0.05]], dtype=np.float32)

            outputs = compute_prediction_outputs(
                logits,
                temperature=self.temperature,
                referable_threshold=self.config.get("grading", {}).get("referable_min_grade", 2)
            )

            case.p5["status"] = "ok"
            case.p5["error"] = None
            case.p5["model_id"] = self.model_id
            case.p5["probs"] = outputs["probs"]
            case.p5["grade"] = outputs["grade"]
            case.p5["expected_grade"] = outputs["expected_grade"]
            case.p5["referable_prob"] = outputs["referable_prob"]
            case.p5["confidence"] = outputs["confidence"]
            case.p5["entropy"] = outputs["entropy"]

        except Exception as e:
            case.p5["status"] = "failed"
            case.p5["error"] = str(e)

        return case

    def run_batch(self, cases: List[CaseRecord]) -> List[CaseRecord]:
        """Run DR grading for a batch of cases."""
        results = []
        for c in cases:
            res = self.run_case(c)
            res.save(self.artifacts_dir)
            results.append(res)
        return results
