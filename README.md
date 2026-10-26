# RetinaCare AI: End-to-End Diabetic Retinopathy Screening & Explainable AI Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/pytest-13%20passed-brightgreen.svg)](tests/)
[![QWK](https://img.shields.io/badge/Test%20QWK-0.857-success.svg)](results/RUN_REPORT.md)
[![Safety Error Capture](https://img.shields.io/badge/Safety%20Recall-100%25-red.svg)](results/RUN_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, clinical-decision-support AI pipeline for **Diabetic Retinopathy (DR)** detection, anatomical localization, lesion segmentation, severity grading, Grad-CAM attention explainability, multi-rule clinical safety verification, and automated publication-grade clinician PDF reporting.

---

## Table of Contents

- [1. Executive Summary & Verified Benchmark Scores](#1-executive-summary--verified-benchmark-scores)
- [2. System Architecture & Pipeline Phases (P1–P8)](#2-system-architecture--pipeline-phases-p1p8)
- [3. Repository Directory Structure](#3-repository-directory-structure)
- [4. Installation & Environment Setup](#4-installation--environment-setup)
- [5. Dataset Preparation & Configuration](#5-dataset-preparation--configuration)
- [6. Step-by-Step Execution Guide](#6-step-by-step-execution-guide)
  - [Step 1: Run Unit Tests](#step-1-run-unit-tests)
  - [Step 2: Run End-to-End Smoke Test](#step-2-run-end-to-end-smoke-test)
  - [Step 3: Data Ingestion & Audit (P1)](#step-3-data-ingestion--audit-p1)
  - [Step 4: Image Preprocessing & Quality Scoring (P2)](#step-4-image-preprocessing--quality-scoring-p2)
  - [Step 5: Anatomical Landmark Localization (P3)](#step-5-anatomical-landmark-localization-p3)
  - [Step 6: Train Lesion Segmentation Model (P4)](#step-6-train-lesion-segmentation-model-p4)
  - [Step 7: Train DR Severity Grading Models (P5: E1–E4)](#step-7-train-dr-severity-grading-models-p5-e1e4)
  - [Step 8: Full Pipeline Execution (P1–P8)](#step-8-full-pipeline-execution-p1p8)
  - [Step 9: Final Evaluation & Report Generation](#step-9-final-evaluation--report-generation)
- [7. Clinical Safety Net Rules (S01–S10)](#7-clinical-safety-net-rules-s01s10)
- [8. Clinician Report Sample](#8-clinician-report-sample)
- [9. Reproducibility & Determinism](#9-reproducibility--determinism)
- [10. Clinical Disclaimer](#10-clinical-disclaimer)

---

## 1. Executive Summary & Verified Benchmark Scores

RetinaCare was evaluated on a locked, group-stratified cohort of 500 fundus images ($70\%$ train, $15\%$ val, $15\%$ test) with **zero cross-split duplicate leakage**.

### 1.1 DR Severity Grading Benchmark (Frozen Test Set, $N=75$)

| Experiment | Model Architecture | Preprocessing Pipeline | Test QWK (95% CI) | Referable Sensitivity (Grade $\ge 2$) | Referable Specificity | Status |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **E2 (Champion)** | **EfficientNetB0 (Transfer Learning)** | **Ben Graham + CLAHE** | **0.857** [0.77, 0.92] | **79.2%** | **96.1%** | **Selected** |
| **E3** | ResNet50 (Transfer Learning) | Ben Graham + CLAHE | **0.779** [0.65, 0.86] | 87.5% | 88.2% | Benchmark |
| **E4 (Ablation)** | EfficientNetB0 | Raw Square Crop (No CLAHE) | **0.648** [0.47, 0.79] | 79.2% | 88.2% | Ablation |
| **E1 (Baseline)** | 4-Layer CNN from Scratch | Ben Graham + CLAHE | **0.000** [0.00, 0.00] | 100.0% | 0.0% | Baseline |

> **Key Finding:** Standardized Ben Graham illumination correction combined with CLAHE boosted Quadratic Weighted Kappa by **+0.209** over raw unenhanced inputs (E2 vs E4: $0.857$ vs $0.648$).

### 1.2 Subsystem Verification & Invariant Scorecard

| Phase | Subsystem | Verified Metric | Result | Clinical Impact |
|---|---|---|:---:|---|
| **P1** | Ingestion & Audit | Group Leakage Across Splits | **0 cases (0.0%)** | Strict mathematical isolation of near-duplicate clusters. |
| **P1** | Synthetic ID Registry | Bijection & Uniqueness | **100% Unique** | Anonymous, trackable `SP-XXXXXX` patient tokens. |
| **P2** | Preprocessing | Standardization Throughput | **100.0% (500/500)** | Fully automated FOV masking, cropping, and multi-scale export. |
| **P3** | Anatomy Localization | Optic Disc & Fovea Plausibility | **97.8% (489/500)** | Accurate anatomical landmarks for false-positive lesion gating. |
| **P4** | Lesion Segmentation | Pixel-Level Dice (IDRiD GT) | **EX: 0.64, HE: 0.58, SE: 0.51, MA: 0.32** | Identifies Microaneurysms, Haemorrhages, Hard Exudates, Cotton Wool Spots. |
| **P5** | Severity Grading | Quadratic Weighted Kappa | **0.857** | Substantial inter-rater agreement with expert ophthalmologists. |
| **P6** | Explainability | Multi-Panel Synthesis | **100.0%** | Combined $2 \times 2$ panel: Image, Lesion Mask, Grad-CAM, Diagnosis Map. |
| **P7** | Clinical Safety Net | Severe Error ($\ge 2$ grades) Capture | **100.0% Recall** | Safety rules intercepted every single major grading discrepancy. |
| **P8** | Clinician Reports | Publication A4 PDF Generation | **500 PDFs Generated** | 2-page structured diagnostic summaries ready for physician sign-off. |

---

## 2. System Architecture & Pipeline Phases (P1–P8)

```
[Raw Fundus Image]
        │
        ▼
┌──────────────────┐
│ P1: Ingestion &  │ ──> Perceptual Hash (16x16 pHash) + NCC (≥0.95) Clustered Splits
│     Audit        │ ──> Synthetic ID Mapping (SP-000001 ... SP-007094)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ P2: Preprocess & │ ──> Otsu FOV Mask + Tight Square Crop + Ben Graham Illumination
│     Quality      │ ──> Dual CLAHE (LAB + Green) + 4-Factor Quality Score [0, 1]
└─────────┬────────┘
          │
          ├──────────────────────────────────────────┐
          ▼                                          ▼
┌──────────────────┐                       ┌──────────────────┐
│ P3: Anatomical   │                       │ P4: Lesion       │
│     Landmarks    │                       │     Segmentation │
│  - Optic Disc    │                       │  - U-Net Encoder │
│  - Fovea Search  │                       │  - MA / HE/EX /SE│
└─────────┬────────┘                       └─────────┬────────┘
          │                                          │
          └────────────────────┬─────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ P5: DR Severity Grading (EfficientNetB0 Transfer)      │
│     - Calibrated Probabilities via Temperature Scaling │
│     - Expected Grade (0.0 to 4.0) + Grad-CAM Heatmap   │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ P6: Fusion & Explainability                            │
│     - Disc False-Positive Suppression (Exudate Gating) │
│     - Macular Risk Zones (<1.0 DD, 1-3 DD, >3 DD)      │
│     - Multi-Panel 2x2 Composite Visualization          │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ P7: Clinical Safety Net (Rules S01 to S10)             │
│     - Flagged Quality / Low Confidence / Discordance   │
│     - Mandatory "review_required: True" Invariant      │
└──────────────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ P8: Reporting & Scorecard                              │
│     - 2-Page Clinician A4 PDF Report (ReportLab)       │
│     - Run Scorecard (scorecard.json & RUN_REPORT.md)   │
└────────────────────────────────────────────────────────┘
```

---

## 3. Repository Directory Structure

```
RetinaCare/
├── configs/
│   └── config.yaml              # Central pipeline configuration (seeds, paths, hyperparams)
├── data/
│   ├── metadata/
│   │   ├── file_inventory.csv          # Catalog of all 7,183 scanned dataset files
│   │   ├── labels_clean.csv            # Cleaned ground-truth clinical labels
│   │   ├── dup_groups.csv              # Perceptual hash duplicate clusters
│   │   ├── id_registry.csv             # Synthetic patient ID bijection table
│   │   ├── sample_500_manifest.csv     # Locked 500-sample cohort with split assignments
│   │   ├── sample_500_manifest.sha256  # SHA-256 tamper-proof lock checksum
│   │   └── sample_500_summary.md       # Class & split balance summary table
│   └── derived/                        # [Excluded from Git] Local cached intermediate tables
├── src/
│   └── retinacare/
│       ├── __init__.py
│       ├── common/              # Shared utilities (config loader, CaseRecord schema, seeds, metrics)
│       ├── p1_ingest_audit/     # Discovery, deduplication, sampling, and splits
│       ├── p2_preprocess/       # FOV masking, Ben Graham illumination, CLAHE, quality scoring
│       ├── p3_anatomy/          # Optic disc detection & fovea geometric localization
│       ├── p4_lesion/           # U-Net lesion segmentation (Mode A) & classical fallback (Mode B)
│       ├── p5_grading/          # E1-E4 architectures, training loops, temperature scaling, Grad-CAM
│       ├── p6_fusion_explain/   # Anatomic lesion suppression, macular zones, overlay panels
│       ├── p7_safety/           # Clinical safety guardrails (S01–S10)
│       └── p8_report/           # 2-Page A4 PDF report generator & global scorecard
├── scripts/
│   ├── run_pipeline.py          # Unified CLI orchestrator (--stage, --scope, --case)
│   ├── smoke_test.py            # Rapid end-to-end smoke verification on 20 stratified cases
│   ├── train_lesion.py          # Train U-Net lesion segmentation model (Mode A)
│   ├── train_grading.py         # Train E1-E4 DR grading models and select champion
│   └── evaluate_all.py          # Compute final test metrics, scorecard, and RUN_REPORT.md
├── tests/
│   ├── fixtures/                # Synthetic test image fixtures
│   ├── test_p1.py               # 5 unit tests for P1 (split leakage, bijection, lock detection)
│   ├── test_p2.py               # 5 unit tests for P2 (FOV mask IoU, padding, quality separation)
│   └── test_p3.py               # 3 unit tests for P3 (disc localization, fovea search, blank soft-fail)
├── results/
│   ├── RUN_REPORT.md            # Comprehensive global execution report
│   ├── scorecard.json           # Machine-readable pipeline audit scorecard
│   ├── scorecard.md             # Markdown summary scorecard
│   ├── audit/                   # Audit summaries and lesion Mode A decision memorandum
│   └── experiments/             # Test evaluation metrics for E1, E2, E3, and E4
├── artifacts/                   # [Directory structure tracked via .gitkeep]
│   ├── cases/                   # CaseRecord JSON state for each processed eye
│   ├── models/                  # Saved model cards and checkpoints (.keras)
│   ├── p2_preprocess/           # Standardized clf.png and hires.png outputs
│   ├── p3_anatomy/              # Optic disc masks and landmark coordinates
│   ├── p6_fusion/               # 2x2 diagnostic overlays
│   └── reports/                 # Clinician 2-page A4 PDF reports
├── pyproject.toml               # Package build configuration & pytest metadata
├── requirements.txt             # Pinned, tested dependency manifest
├── ASSUMPTIONS.md               # Complete architectural assumptions log
├── PROGRESS.md                  # Detailed implementation log
└── README.md                    # This document
```

---

## 4. Installation & Environment Setup

### 4.1 Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Version 3.11 (tested on `3.11.9`)
- **Memory:** $\ge 8\text{ GB}$ RAM (16 GB recommended)
- **Hardware Acceleration:** Native CPU execution (with oneDNN acceleration enabled) or NVIDIA CUDA GPU.

### 4.2 Clone the Repository
```bash
git clone https://github.com/bhaskarmatsagar/RetinaCare.git
cd RetinaCare
```

### 4.3 Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.4 Install Dependencies and Package in Editable Mode
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Verify your environment by running:
```bash
python -c "import retinacare; print('RetinaCare successfully imported! Version:', retinacare.__version__)"
```

---

## 5. Dataset Preparation & Configuration

RetinaCare ingests raw retinal fundus images from public benchmark repositories:
1. **APTOS 2019 Blindness Detection** (Kaggle: `train_images/`, `train.csv`)
2. **IDRiD: Indian Diabetic Retinopathy Image Dataset** (`archive (1)`: Original Images & Ground Truth Masks for MA, HE, EX, SE, OD)
3. **AGAR300 (IEEE DataPort)** (Unlabelled test images)

### 5.1 Dataset Directory Placement
Place the unzipped dataset folders either:
- In a parent directory: `../Retina care Dataset/`
- Or inside a local `datasets/` directory:
```
RetinaCare/
├── datasets/
│   ├── aptos2019-blindness-detection/
│   │   ├── train_images/
│   │   └── train.csv
│   └── archive (1)/
│       └── ...
```

### 5.2 Configure Dataset Path
You can point to your datasets in either of two ways:
1. **Via Environment Variable (Recommended):**
   ```powershell
   $env:RETINACARE_DATA_ROOT = "C:/path/to/your/datasets"
   ```
   Or on Linux/macOS:
   ```bash
   export RETINACARE_DATA_ROOT="/path/to/your/datasets"
   ```
2. **Or directly in `configs/config.yaml`:**
   ```yaml
   paths:
     data_root: "path/to/your/datasets"
     artifacts: "artifacts"
     results: "results"
   ```

---

## 6. Step-by-Step Execution Guide

### Step 1: Run Unit Tests
Validate environment integrity and core mathematical invariants:
```bash
pytest tests/ -v
```
*Expected Result:* All 13 tests pass in under 10 seconds.

### Step 2: Run End-to-End Smoke Test
Run an end-to-end verification through all 8 phases (P1–P8) on 20 stratified cases:
```bash
python scripts/smoke_test.py
```
*Outputs:* Validates every subsystem and generates 20 sample clinician PDF reports in `artifacts/reports/`.

### Step 3: Data Ingestion & Audit (P1)
Scan all raw datasets, detect near-duplicate clusters via perceptual hashing, map synthetic patient IDs, and produce the locked 500-case manifest:
```bash
python scripts/run_pipeline.py --stage p1 --scope sample500
```
*Outputs:* `data/metadata/sample_500_manifest.csv`, `id_registry.csv`, `dup_groups.csv`.

### Step 4: Image Preprocessing & Quality Scoring (P2)
Extract FOV masks, perform tight square cropping, calculate 4-factor quality scores, and apply Ben Graham illumination + CLAHE:
```bash
python scripts/run_pipeline.py --stage p2 --scope sample500
```
*Outputs:* Preprocessed images saved in `artifacts/p2_preprocess/{case_id}/` (`clf.png`, `clf_raw.png`, `hires.png`).

### Step 5: Anatomical Landmark Localization (P3)
Detect Optic Disc boundaries and estimate foveal coordinates:
```bash
python scripts/run_pipeline.py --stage p3 --scope sample500
```
*Outputs:* `artifacts/p3_anatomy/{case_id}/disc_mask.png` and geometric coordinates updated in each case JSON.

### Step 6: Train Lesion Segmentation Model (P4)
Train a 4-class U-Net model on IDRiD pixel-level ground truth (Mode A) using Focal-Tversky loss:
```bash
python scripts/train_lesion.py
```
*Outputs:* Trained weights saved to `artifacts/models/best_lesion_unet.keras`.

### Step 7: Train DR Severity Grading Models (P5: E1–E4)
Train and benchmark all 4 grading architectures on the preprocessed 500-sample splits:
```bash
python scripts/train_grading.py --experiment all
```
*Outputs:*
- `artifacts/models/E1/` (Scratch CNN)
- `artifacts/models/E2/` (EfficientNetB0 — Champion)
- `artifacts/models/E3/` (ResNet50)
- `artifacts/models/E4/` (EfficientNetB0 on raw unenhanced inputs)
- Evaluates validation QWK and saves `artifacts/models/best_grading_model.keras`.

### Step 8: Full Pipeline Execution (P1–P8)
Run the entire screening pipeline across all 500 cases using the trained models:
```bash
python scripts/run_pipeline.py --stage all --scope sample500
```

#### Running a Single Specific Case:
```bash
python scripts/run_pipeline.py --case SP-000123
```

### Step 9: Final Evaluation & Report Generation
Evaluate the frozen test set strictly once, compile bootstrap confidence intervals, and generate global summary scorecards:
```bash
python scripts/evaluate_all.py
```
*Outputs:*
- `results/RUN_REPORT.md` (Detailed executive and technical summary)
- `results/scorecard.json` (Machine-readable metrics)
- `results/scorecard.md` (Summary table)
- `artifacts/reports/*.pdf` (500 publication-grade 2-page A4 clinician reports)

---

## 7. Clinical Safety Net Rules (S01–S10)

RetinaCare enforces a multi-tiered safety net to prevent AI overreliance and flag high-risk or ambiguous cases:

| Rule Code | Safety Trigger | Condition | Clinical Action |
|:---:|:---|:---|:---|
| **S01** | Low Image Quality | Overall Quality Score $< 0.50$ | Flag for repeat photograph / manual inspection. |
| **S02** | High Classification Uncertainty | Maximum softmax confidence $< 0.60$ | Flag borderline grade for senior ophthalmologist review. |
| **S03** | High Entropy Divergence | Shannon entropy of prediction distribution $\ge 1.2$ | Ambiguous severity; requires clinical correlation. |
| **S04** | Grade-Lesion Discordance | Predicted Grade $\le 1$ but substantial lesion area detected | Guard against false negative under-grading. |
| **S05** | High Grade Without Lesions | Predicted Grade $\ge 3$ but zero detectable lesions | Guard against false positive feature hallucination. |
| **S06** | Macular Involvement | Microaneurysms or Exudates within $< 1.0\text{ DD}$ of Fovea | Urgent referral for Diabetic Macular Oedema (DMO) workup. |
| **S07** | Landmark Localization Failure | Optic disc or fovea geometric plausibility check failed | Anatomic landmark review required. |
| **S08** | Classical Heuristic Disclaimer | Lesion detector operated in Mode B (unvalidated heuristic) | Explanatory disclaimer appended to report. |
| **S09** | Grad-CAM Attention Boundary Artifact | $\ge 35\%$ of attention mass concentrated at image boundary | Heatmap unreliable; possible edge glare artifact. |
| **S10** | Out-of-Distribution Domain Shift | Camera aspect ratio, illumination, or color shift detected | Domain shift advisory for clinician. |

> **Non-Negotiable Safety Invariant:** Every case record strictly retains `review_required = True`. AI outputs are advisory tools to empower clinicians, never autonomous replacements.

---

## 8. Clinician Report Sample

Each screened case produces a 2-page A4 PDF generated via ReportLab:

- **Page 1: Diagnostic Overview & Visual Panel**
  - Patient Metadata (Synthetic ID, Acquisition Date, Image Quality Index).
  - Diagnosis Badge: Predicted Grade (0 to 4), Referable DR Probability, Confidence Score.
  - $2 \times 2$ Visual Explainability Panel:
    1. Preprocessed standardized retinal image.
    2. Color-coded lesion segmentation overlay (MA=Red, HE=Orange, EX=Yellow, SE=Cyan).
    3. Grad-CAM visual attention heatmap.
    4. Combined diagnostic map with anatomical landmarks (Disc circle, Fovea crosshair, Macular zones).
- **Page 2: Clinical Metrics & Safety Review**
  - Lesion burden table: lesion count and surface area broken down by Macular, Perimacular, and Peripheral zones.
  - Active clinical safety flags with severity indicators.
  - Plain-language patient explanation section.
  - Mandatory clinical safety disclaimers and physician sign-off block.

---

## 9. Reproducibility & Determinism

- **Global Deterministic Seed:** `42` enforced across Python `random`, NumPy, and TensorFlow.
- **Manifest Tamper Detection:** `sample_500_manifest.csv` is locked via SHA-256 (`sample_500_manifest.sha256`). Any unexpected row or label alteration will halt execution immediately.
- **Split Grouping:** Group stratification guarantees that near-duplicate images of the same eye never appear across both training and test sets.

---

## 10. Clinical Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> RetinaCare is an investigational decision-support software prototype developed for scientific research and engineering benchmarking. It is **not** an FDA/CE-cleared medical device and is **not** certified for autonomous clinical diagnosis. All model predictions, lesion segmentations, and generated reports **must** be reviewed and validated by a licensed ophthalmologist or healthcare professional before making clinical decisions.
