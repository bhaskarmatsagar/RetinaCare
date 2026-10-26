"""Deterministic report text templates and mandatory medical disclaimers."""

# Appendix C: Mandatory Disclaimer Text
MANDATORY_DISCLAIMER = (
    "This is an AI-assisted screening aid for research and demonstration. It is not a diagnostic device "
    "and has not been clinically validated. All results must be reviewed and confirmed by a qualified eye-care "
    "professional. A result that shows no detected abnormality does not exclude eye disease. A retinal photograph "
    "cannot assess every eye condition (for example diabetic macular oedema needs OCT). Patients with sudden vision loss, "
    "new floaters, flashes of light, a dark curtain over the vision, or eye pain should seek urgent eye care regardless "
    "of this result."
)

GRADE_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative Diabetic Retinopathy"
}

LIMITATIONS_TEXT = (
    "Technical Limitations: Trained and evaluated on research dataset sample; rare classes have limited sample size. "
    "Assessment is based strictly on fundus photography; OCT is required for diabetic macular oedema (DME) assessment. "
    "Visual overlays (lesion segmentation and Grad-CAM) are clinician interpretation aids and not standalone proof of pathology."
)


def format_plain_language_explanation(grade: int, confidence: float, lesion_count: int, p4_mode: str) -> str:
    """Deterministic plain-language explanation matching Section P8.1 requirement."""
    name = GRADE_NAMES.get(grade, "Unknown Grade")
    mode_desc = "validated segmentation model" if p4_mode == "A" else "heuristic morphological detector"
    return (
        f"The system estimates DR severity as Grade {grade} ({name}) with a calibrated confidence of {confidence:.1%}. "
        f"A total of {lesion_count} lesion regions were highlighted using the {mode_desc}. "
        "The highlighted regions and attention heatmaps are assistive visualization aids for the examining clinician "
        "and do not constitute definitive proof of disease."
    )
