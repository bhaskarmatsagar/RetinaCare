"""ReportLab PDF generator for clinician case reports (A4, 2 pages maximum)."""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether
)

from retinacare.common.case_record import CaseRecord
from retinacare.p8_report.texts import (
    MANDATORY_DISCLAIMER,
    GRADE_NAMES,
    LIMITATIONS_TEXT,
    format_plain_language_explanation
)


def generate_case_pdf(
    case: CaseRecord,
    out_pdf_path: Path,
    repo_root: Path,
    manifest_hash: str = "LOCKED_SAMPLE_500"
) -> bool:
    """Generate 2-page clinician PDF report using ReportLab."""
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_pdf_path),
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=25,
        bottomMargin=25
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1A365D")
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=4,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#2D3748")
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerCustom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#C53030")
    )
    alert_style = ParagraphStyle(
        "AlertStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11
    )

    story = []

    # ------------------- PAGE 1 -------------------
    # 1. Header
    header_data = [
        [
            Paragraph("<b>RetinaCare AI — Diabetic Retinopathy Screening Report</b>", title_style),
            Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/><b>Case ID:</b> {case.case_id}", body_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[360, 175])
    t_header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_header)

    # 2. Mandatory Disclaimer
    disclaimer_box = Table(
        [[Paragraph(f"<b>MANDATORY CLINICAL NOTICE:</b> {MANDATORY_DISCLAIMER}", disclaimer_style)]],
        colWidths=[535]
    )
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF5F5")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FEB2B2")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_box)
    story.append(Spacer(1, 4))

    # 3. Priority and Status Banner
    prio = case.p7.get("priority", "REVIEW_STANDARD")
    prio_color = "#38A169" if prio == "REVIEW_STANDARD" else ("#DD6B20" if prio == "REVIEW_PRIORITY_HIGH" else "#E53E3E")
    prio_text = f"<font color='{prio_color}'><b>REVIEW STATUS: {prio} (CLINICIAN VERIFICATION REQUIRED)</b></font>"

    story.append(Paragraph(prio_text, alert_style))
    story.append(Spacer(1, 4))

    # 4. Images Row (Preprocessed, Lesion Overlay, Grad-CAM Overlay, Combined Panel)
    img_cells = []
    # Try loading combined panel
    comb_rel = case.p6.get("overlay_paths", {}).get("combined")
    les_rel = case.p6.get("overlay_paths", {}).get("lesion")
    cam_rel = case.p6.get("overlay_paths", {}).get("gradcam")
    prep_rel = case.p2.get("paths", {}).get("clf")

    def make_img(rel_p, w=125, h=125):
        if rel_p and (repo_root / rel_p).is_file():
            return Image(str(repo_root / rel_p), width=w, height=h)
        return Paragraph("Image<br/>Pending", body_style)

    img_table_data = [
        [
            Paragraph("<b>Preprocessed Image</b>", body_style),
            Paragraph("<b>Lesion Overlay</b>", body_style),
            Paragraph("<b>Grad-CAM Attention</b>", body_style),
            Paragraph("<b>2x2 Combined Panel</b>", body_style),
        ],
        [
            make_img(prep_rel, 125, 125),
            make_img(les_rel, 125, 125),
            make_img(cam_rel, 125, 125),
            make_img(comb_rel, 125, 125),
        ]
    ]
    t_imgs = Table(img_table_data, colWidths=[133, 133, 133, 134])
    t_imgs.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_imgs)
    story.append(Spacer(1, 6))

    # 5. Image Quality Block
    q_info = case.p2.get("quality", {})
    q_score = q_info.get("score", 0.0)
    q_pass = "PASS" if q_info.get("pass", False) else "FLAGGED POOR"
    q_data = [
        [
            Paragraph("<b>Quality Assessment:</b>", h2_style),
            Paragraph(f"Score: <b>{q_score:.2f}</b> | Status: <b>{q_pass}</b> | Sharpness: {q_info.get('sharpness',0):.2f} | Illum: {q_info.get('illumination',0):.2f} | Contrast: {q_info.get('contrast',0):.2f} | Coverage: {q_info.get('fov_coverage',0):.2f}", body_style)
        ]
    ]
    t_q = Table(q_data, colWidths=[130, 405])
    story.append(t_q)
    story.append(Spacer(1, 4))

    # 6. Grading Block
    g_num = case.p5.get("grade", 0)
    g_name = GRADE_NAMES.get(g_num, "Unknown")
    g_conf = case.p5.get("confidence", 0.0)
    ref_prob = case.p5.get("referable_prob", 0.0)
    probs = case.p5.get("probs", [0.0]*5)

    probs_str = " | ".join([f"G{i}: {p:.1%}" for i, p in enumerate(probs)])

    g_table_data = [
        [
            Paragraph("<b>Severity Estimate:</b>", h2_style),
            Paragraph(f"<b>Grade {g_num} — {g_name}</b> (Confidence: <b>{g_conf:.1%}</b>)", body_style)
        ],
        [
            Paragraph("<b>Referable DR (Grade ≥ 2):</b>", body_style),
            Paragraph(f"Probability: <b>{ref_prob:.1%}</b> ({'REFERABLE' if ref_prob >= 0.5 else 'NON-REFERABLE'})", body_style)
        ],
        [
            Paragraph("<b>Class Distribution:</b>", body_style),
            Paragraph(probs_str, body_style)
        ]
    ]
    t_g = Table(g_table_data, colWidths=[150, 385])
    t_g.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_g)

    # ------------------- PAGE 2 -------------------
    story.append(PageBreak())

    # Page 2 Header
    story.append(Paragraph(f"<b>RetinaCare AI Case Findings — Case {case.case_id}</b>", h2_style))
    story.append(Spacer(1, 4))

    # 7. Lesion & Anatomy Table
    story.append(Paragraph("<b>Lesion Detections & Anatomical Distribution</b>", h2_style))
    p4_stats = case.p4.get("stats", {})
    les_rows = [
        ["Lesion Type", "Candidate Count", "Area (px)", "FOV Area Fraction", "Macular Zone (≤1 DD)"]
    ]
    zones = case.p6.get("lesions_after_fusion", {}).get("zones", {})
    mac_classes = zones.get("macular", {}).get("classes", {})

    total_les_count = 0
    for name in ["HE", "MA", "EX", "SE"]:
        st = p4_stats.get(name, {})
        cnt = st.get("count", 0)
        total_les_count += cnt
        area = st.get("area_px", 0)
        frac = st.get("area_frac_fov", 0.0)
        mac_cnt = mac_classes.get(name, {}).get("count", 0)
        les_rows.append([name, str(cnt), str(area), f"{frac:.5f}", str(mac_cnt)])

    t_les = Table(les_rows, colWidths=[100, 105, 105, 115, 110])
    t_les.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_les)
    story.append(Spacer(1, 6))

    # 8. Safety Flags List
    story.append(Paragraph("<b>Safety & Quality Flags</b>", h2_style))
    flags = case.p7.get("flags", [])
    if flags:
        flag_rows = [["ID", "Severity", "Safety Rule Message"]]
        for f in flags:
            flag_rows.append([f.get("id", ""), f.get("severity", "").upper(), Paragraph(f.get("message", ""), body_style)])
        t_flags = Table(flag_rows, colWidths=[40, 75, 420])
        t_flags.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_flags)
    else:
        story.append(Paragraph("No automated safety flags triggered. Standard clinician review required.", body_style))
    story.append(Spacer(1, 6))

    # 9. Plain-language Explanation
    exp_text = format_plain_language_explanation(g_num, g_conf, total_les_count, case.p4.get("mode", "A"))
    story.append(Paragraph(f"<b>Summary Finding:</b> {exp_text}", body_style))
    story.append(Spacer(1, 4))

    # 10. Technical Limitations
    story.append(Paragraph(f"<b>Notice of Technical Scope:</b> {LIMITATIONS_TEXT}", disclaimer_style))
    story.append(Spacer(1, 8))

    # 11. Doctor Verification Block (Blank, for manual completion per Section P8.1)
    doc_block_data = [
        [Paragraph("<b>CLINICIAN DIAGNOSTIC VERIFICATION & FINAL DISPOSITION</b>", h2_style)],
        [Paragraph("☐ <b>Confirm AI Grade</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ☐ <b>Change Grade To:</b> _______ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ☐ <b>Reject AI Result (Inconclusive/Artefact)</b>", body_style)],
        [Paragraph("<b>Final Clinical Diagnosis:</b> ___________________________________________________________________________________", body_style)],
        [Paragraph("<b>Clinical Notes / Observations:</b> ______________________________________________________________________________", body_style)],
        [Paragraph("<b>Follow-up Recommended:</b> &nbsp;&nbsp; ☐ Routine (12m) &nbsp;&nbsp; ☐ Early (3-6m) &nbsp;&nbsp; ☐ Urgent Referral (Ophthalmology/Retina Clinic)", body_style)],
        [Paragraph("<b>Examining Clinician:</b> ______________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Signature / Date:</b> _________________________", body_style)]
    ]
    t_doc = Table(doc_block_data, colWidths=[535])
    t_doc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#A0AEC0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_doc)
    story.append(Spacer(1, 6))

    # 12. Footer
    footer_text = f"RetinaCare AI Pipeline v1.0 | Grading Model: {case.p5.get('model_id', 'standard')} | Manifest SHA256: {manifest_hash[:16]}..."
    story.append(Paragraph(footer_text, ParagraphStyle("FooterCustom", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#718096"))))

    # Build PDF
    doc.build(story)
    return True
