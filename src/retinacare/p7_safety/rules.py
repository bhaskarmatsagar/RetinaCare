"""Clinical safety rules S01 - S10 and review priority determination."""
from typing import Dict, Any, List, Tuple
from retinacare.common.case_record import CaseRecord


def evaluate_safety_rules(case: CaseRecord, config: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """Evaluate clinical safety rules and determine clinician review priority."""
    flags = []
    safety_cfg = config.get("safety", {})

    # Rule S01: Quality failure
    q_pass = case.p2.get("quality", {}).get("pass", True)
    reasons = case.p2.get("quality", {}).get("reasons", [])
    if not q_pass:
        flags.append({
            "id": "S01",
            "severity": "high",
            "message": f"Image quality below threshold: retake or examine clinically. Reasons: {', '.join(reasons) if reasons else 'low_score'}"
        })

    # Rule S02: Low confidence
    conf_thresh = safety_cfg.get("low_confidence", 0.60)
    conf = float(case.p5.get("confidence", 0.0))
    if conf < conf_thresh:
        flags.append({
            "id": "S02",
            "severity": "medium",
            "message": f"Low model confidence ({conf:.2f} < {conf_thresh:.2f})."
        })

    # Rule S03 & S04: Mismatch between grade and detected lesion area
    grade = int(case.p5.get("grade", 0))
    total_area_frac = sum(
        s.get("area_frac_fov", 0.0)
        for s in case.p4.get("stats", {}).values()
    )
    mismatch_high = safety_cfg.get("mismatch_lesion_area_frac_high", 0.0015)
    mismatch_low = safety_cfg.get("mismatch_lesion_area_frac_low", 0.0002)

    if grade >= 2 and total_area_frac < mismatch_low:
        flags.append({
            "id": "S03",
            "severity": "high",
            "message": f"Grade suggests significant DR ({grade}) but few lesions were detected (area frac: {total_area_frac:.5f}). Review."
        })

    if grade <= 1 and total_area_frac > mismatch_high:
        flags.append({
            "id": "S04",
            "severity": "high",
            "message": f"Many lesions detected (area frac: {total_area_frac:.5f}) but estimated grade is low ({grade}). Review."
        })

    # Rule S05: Anatomy localization failure
    disc_found = case.p3.get("disc", {}).get("found", False)
    fovea_found = case.p3.get("fovea", {}).get("found", False)
    plausible = case.p3.get("plausible", False)
    if not (disc_found and fovea_found and plausible):
        flags.append({
            "id": "S05",
            "severity": "medium",
            "message": "Anatomy could not be located reliably; lesion zones may be unavailable or approximate."
        })

    # Rule S06: Macular lesion involvement with low overall grade
    macular_count = case.p6.get("lesions_after_fusion", {}).get("zones", {}).get("macular", {}).get("count", 0)
    if macular_count > 0 and grade <= 1:
        flags.append({
            "id": "S06",
            "severity": "high",
            "message": f"Lesions detected near the macula ({macular_count} in macular zone) with low grade ({grade}). Review for macular involvement."
        })

    # Rule S07: Grad-CAM attention concentrated on border
    border_max = safety_cfg.get("border_cam_fraction_max", 0.35)
    mass_border = float(case.p6.get("gradcam_stats", {}).get("mass_near_border", 0.0))
    if mass_border > border_max:
        flags.append({
            "id": "S07",
            "severity": "medium",
            "message": f"Model attention concentrated near image border ({mass_border:.2f} > {border_max:.2f}); possible artefact."
        })

    # Rule S08: Heuristic unvalidated lesion mode
    if not case.p4.get("validated", True):
        flags.append({
            "id": "S08",
            "severity": "info",
            "message": "Lesion map comes from a heuristic detector and is not validated against pixel-level ground truth."
        })

    # Rule S09: Component failure
    for stage_name, block in [("p2", case.p2), ("p3", case.p3), ("p4", case.p4), ("p5", case.p5), ("p6", case.p6)]:
        if block.get("status") == "failed":
            flags.append({
                "id": "S09",
                "severity": "high",
                "message": f"Component {stage_name} failed: {block.get('error', 'unknown error')}; result incomplete."
            })

    # Rule S10: Close probability between top-2 grades
    probs = case.p5.get("probs", [])
    if len(probs) >= 2:
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        diff = sorted_probs[0][1] - sorted_probs[1][1]
        if diff < 0.15:
            flags.append({
                "id": "S10",
                "severity": "low",
                "message": f"Result is close between two grades (Grade {sorted_probs[0][0]} vs Grade {sorted_probs[1][0]}, diff: {diff:.3f})."
            })

    # Determine Review Priority per Section P7.2
    has_high = any(f["severity"] == "high" for f in flags)
    ref_prob = float(case.p5.get("referable_prob", 0.0))

    if any(f["id"] == "S01" for f in flags):
        priority = "RETAKE_RECOMMENDED"
    elif has_high or grade >= 3 or ref_prob >= 0.70:
        priority = "REVIEW_PRIORITY_HIGH"
    else:
        priority = "REVIEW_STANDARD"

    return flags, priority
