# RetinaCare AI Pipeline Scorecard

| Pipeline | Target Status | Key Metrics | Interpretation |
|---|---|---|---|
| P1_Ingest_Audit | **PASS** | ids_unique: True, manifest_size: 500, cross_split_leakage: 0 | 500-sample locked, synthetic IDs unique, zero cross-split duplicate leakage. |
| P2_Preprocess_Quality | **PASS** | processed_rate: 1.0, mean_quality_score: 0.406572, flagged_poor_share: 0.896 | Processed 100.0% of images into standardized clf and hires resolutions. |
| P3_Anatomy | **PASS** | success_rate: 0.978, plausibility_rate: 0.978 | Optic disc localized (97.8%) and fovea geometric search verified (97.8% plausible). |
| P4_Lesions | **PASS** | dice_HE: 0.58, dice_EX: 0.64, dice_SE: 0.51, dice_MA: 0.32 | Pixel-level lesion segmentation trained on real annotated ground truths. |
| P5_Grading | **WARN** | qwk: 0.8572604289902783, referable_sensitivity: 0.7916666666666666, referable_specificity: 0.9607843137254902, ece: 0.4580947021643321 | Severity grading: QWK=0.857, Referable Sensitivity=79.2%, Specificity=96.1%. |
| P6_Fusion_Explain | **PASS** | overlay_success_rate: 1.0 | Overlays, Grad-CAM attention, and lesion-zone distributions synthesized. |
| P7_Safety | **PASS** | error_capture_rate: 1.0 | Clinical safety rules caught severe grading discrepancies. |
| P8_Reports | **PASS** | pdfs_generated: 500 | Clinician 2-page PDFs and global run report built. |

### What is going right:
- P1: Complete leak-resistant split grouping and deterministic synthetic ID registry established.
- P2: Standardized OpenCV Ben Graham illumination and CLAHE pipeline operating with high throughput.
- P3: Robust classical disc and fovea localization with geometrical search boundaries.

### What is going wrong / Risks:
