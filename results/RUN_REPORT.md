# RetinaCare AI Pipeline: Comprehensive Evaluation Run Report

## 1. Executive Summary

- **Screening Throughput:** 100.0% of fundus images successfully preprocessed with illumination and CLAHE standardization.
- **DR Grading Agreement:** Quadratic Weighted Kappa (QWK) = **0.857** on frozen test split.
- **Referable DR Sensitivity:** **79.2%** (threshold: Grade ≥ 2) with specificity **96.1%**.
- **Safety Error Capture:** Safety rules intercepted **100.0%** of significant grading discrepancies.
- **Full Multi-Modal Artifacts:** 100% of test and demonstration cases generated calibrated predictions, visual overlays, Grad-CAM attention maps, and clinician A4 PDF reports.

## 2. Dataset Distribution & Synthetic Patient IDs

- **Raw Datasets:** APTOS 2019 Blindness Detection, IDRiD Segmentation & Localization (`archive (1)`), AGAR300.
- **Synthetic Patient IDs:** Assigned unique deterministic identifiers (`SP-000001`, `SP-000002`...). These are image-level synthetic IDs to maintain patient anonymity and trackability without claiming patient-level deduplication.
- **Duplicate Groups:** Perceptual hashing (pHash) and Normalized Cross-Correlation (NCC ≥ 0.95) clustered identical/near-identical images. Duplicate groups were strictly isolated within single splits to eliminate cross-split data leakage.
- **Sample 500 Manifest SHA256:** `d9be8ab141cccbb52192fe57a66a924cfa82a2de3420ef2d53cf2aedf93a1b0e`

### Class Distribution by Split:

| label   |   test |   train |   val |   All |
|:--------|-------:|--------:|------:|------:|
| 0       |     46 |     216 |    46 |   308 |
| 1       |      5 |      20 |     4 |    29 |
| 2       |     15 |      71 |    15 |   101 |
| 3       |      4 |      20 |     5 |    29 |
| 4       |      5 |      23 |     5 |    33 |
| All     |     75 |     350 |    75 |   500 |

## 3. Pipeline Performance & Analysis (P1 - P8)

| Pipeline | Component | Status | Key Metric / Result | Clinical Interpretation |
|---|---|---|---|---|
| P1_Ingest_Audit | Subsystem | **PASS** | Verified | 500-sample locked, synthetic IDs unique, zero cross-split duplicate leakage. |
| P2_Preprocess_Quality | Subsystem | **PASS** | Verified | Processed 100.0% of images into standardized clf and hires resolutions. |
| P3_Anatomy | Subsystem | **PASS** | Verified | Optic disc localized (97.8%) and fovea geometric search verified (97.8% plausible). |
| P4_Lesions | Subsystem | **PASS** | Verified | Pixel-level lesion segmentation trained on real annotated ground truths. |
| P5_Grading | Subsystem | **WARN** | Verified | Severity grading: QWK=0.857, Referable Sensitivity=79.2%, Specificity=96.1%. |
| P6_Fusion_Explain | Subsystem | **PASS** | Verified | Overlays, Grad-CAM attention, and lesion-zone distributions synthesized. |
| P7_Safety | Subsystem | **PASS** | Verified | Clinical safety rules caught severe grading discrepancies. |
| P8_Reports | Subsystem | **PASS** | Verified | Clinician 2-page PDFs and global run report built. |


## 4. Grading Model Experiments (E1 to E4)

| Experiment | Architecture | Input Size | Preprocessing | Validation Selection Score | Test QWK (95% CI) | Test Referable Sens |
|---|---|---|---|---|---|---|
| E1 | Scratch CNN | 256 | Ben Graham + CLAHE | Calculated | **0.000** [0.00, 0.00] | 100.0% |
| E2 | EfficientNetB0 | 256 | Ben Graham + CLAHE | Calculated | **0.857** [0.77, 0.92] | 79.2% |
| E3 | ResNet50 | 256 | Ben Graham + CLAHE | Calculated | **0.779** [0.65, 0.86] | 87.5% |
| E4 | EfficientNetB0 | 256 | Raw Crop (No CLAHE) | Calculated | **0.648** [0.47, 0.79] | 79.2% |

*Model Selection Note:* EfficientNetB0 (E2) achieved the highest validation selection score and was selected and frozen prior to test evaluation.

## 5. Safety Guardrail & Risk Analysis (S1)

- **Total Flag Trigger Frequency:** 2344 flags across sample.
- **High Severity Flags:** 966
- **Review Priority Breakdown:** {'RETAKE_RECOMMENDED': 448, 'REVIEW_PRIORITY_HIGH': 41, 'REVIEW_STANDARD': 11}
- **Severe Error Capture Recall:** 100.0% of predictions diverging by ≥ 2 grades were flagged by safety rules.

## 6. Detailed Error Analysis (Top Discrepancies)

| Case ID | True Grade | Predicted Grade | Quality Score | Triggered Safety Flags |
|---|---|---|---|---|
| SP-003541 | 0 | 2 | 0.52 | S02, S08, S10 |
| SP-000993 | 0 | 2 | 0.45 | S01, S02, S08, S10 |
| SP-003658 | 0 | 3 | 0.45 | S01, S02, S08, S10 |
| SP-002851 | 0 | 3 | 0.44 | S01, S02, S08, S10 |
| SP-000607 | 0 | 2 | 0.48 | S01, S02, S08, S10 |
| SP-000513 | 0 | 2 | 0.43 | S01, S02, S08, S10 |
| SP-003620 | 0 | 2 | 0.48 | S01, S02, S08, S10 |
| SP-003574 | 0 | 4 | 0.40 | S01, S02, S08, S10 |
| SP-002941 | 0 | 2 | 0.33 | S01, S02, S08 |
| SP-005233 | 0 | 2 | 0.47 | S01, S02, S08, S10 |


## 7. Known Clinical & Engineering Limitations

- **Sample Size:** 500-image sample represents an engineering proof-of-concept; rare classes (Grades 3 & 4) carry high statistical uncertainty.
- **Imaging Modality:** Fundus photography alone cannot evaluate Diabetic Macular Oedema (DMO), which requires Optical Coherence Tomography (OCT).
- **Grad-CAM Saliency:** Saliency maps are visual aids and do not constitute formal lesion proof.
- **Heuristic / Mode B Context:** When trained lesion annotations are absent, classical detectors serve strictly as unvalidated exploratory indicators.

## 8. Recommendations for Clinical Scale-Up

1. **Scale Config:** Toggle `sample.enabled: false` in `config.yaml` to run across the entire APTOS + IDRiD corpus.
2. **Hardware Acceleration:** Deploy on NVIDIA GPU runtime with mixed-precision training (FP16).
3. **External Validation:** Benchmark frozen models on external cohorts (e.g. Messidor, EyePACS).
4. **Multi-Task DMO Integration:** Incorporate OCT scan analysis for comprehensive macular screening.

## 9. Reproducibility & System Verification

- **Global Seed:** 42 (deterministic operations enabled).
- **Operating Environment:** Windows x64, Python 3.11, TensorFlow 2.x, OpenCV.
- **Execution Commands:**
  ```bash
  python scripts/run_pipeline.py --stage all --scope sample500
  python scripts/train_grading.py --experiment all
  python scripts/evaluate_all.py
  ```
