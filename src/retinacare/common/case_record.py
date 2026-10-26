"""Case Record data structures and JSON serialisation."""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CaseRecord:
    case_id: str
    synthetic_patient_id: str
    image_id: str
    source_path: str
    sha256: str
    dup_group_id: str
    label: Optional[int] = None
    split: str = "unlabelled_demo"
    in_sample_500: bool = False

    p2: Dict[str, Any] = field(default_factory=lambda: {
        "status": "pending", "error": None,
        "frame": {"crop_box": [0, 0, 0, 0], "orig_size": [0, 0], "scale_clf": 1.0, "scale_hires": 1.0},
        "paths": {"clf": "", "hires": "", "fov_mask": "", "green_clahe": ""},
        "quality": {"score": 0.0, "sharpness": 0.0, "illumination": 0.0, "contrast": 0.0, "fov_coverage": 0.0, "pass": False, "reasons": []}
    })

    p3: Dict[str, Any] = field(default_factory=lambda: {
        "status": "pending", "error": None,
        "disc": {"found": False, "center": [0, 0], "diameter_px": 0.0, "mask_path": "", "confidence": 0.0},
        "fovea": {"found": False, "point": [0, 0], "confidence": 0.0},
        "plausible": False
    })

    p4: Dict[str, Any] = field(default_factory=lambda: {
        "status": "pending", "error": None,
        "mode": "A", "validated": True,
        "masks": {"MA": "", "HE": "", "EX": "", "SE": ""},
        "prob_maps": {},
        "stats": {"HE": {"count": 0, "area_px": 0, "area_frac_fov": 0.0},
                  "EX": {"count": 0, "area_px": 0, "area_frac_fov": 0.0},
                  "MA": {"count": 0, "area_px": 0, "area_frac_fov": 0.0},
                  "SE": {"count": 0, "area_px": 0, "area_frac_fov": 0.0}}
    })

    p5: Dict[str, Any] = field(default_factory=lambda: {
        "status": "pending", "error": None,
        "model_id": "",
        "probs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "grade": 0,
        "expected_grade": 0.0,
        "referable_prob": 0.0,
        "confidence": 0.0,
        "entropy": 0.0
    })

    p6: Dict[str, Any] = field(default_factory=lambda: {
        "status": "pending", "error": None,
        "overlay_paths": {"lesion": "", "gradcam": "", "combined": ""},
        "lesions_after_fusion": {"EX_removed_in_disc": 0, "zones": {"macular": {}, "perimacular": {}, "peripheral": {}}},
        "gradcam_stats": {"mass_in_fov": 0.0, "mass_near_border": 0.0, "lesion_agreement": 0.0}
    })

    p7: Dict[str, Any] = field(default_factory=lambda: {
        "status": "pending", "error": None,
        "flags": [],
        "priority": "REVIEW_STANDARD",
        "review_required": True
    })

    p8: Dict[str, Any] = field(default_factory=lambda: {
        "status": "pending", "error": None,
        "pdf_path": ""
    })

    versions: Dict[str, str] = field(default_factory=lambda: {
        "p2": "1.0", "p3": "1.0", "p4": "1.0", "p5": "1.0", "p6": "1.0", "p7": "1.0", "p8": "1.0"
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, artifacts_dir: Path) -> Path:
        cases_dir = artifacts_dir / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)
        out_path = cases_dir / f"{self.case_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    @classmethod
    def load(cls, file_path: Path) -> "CaseRecord":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
