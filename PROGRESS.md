# RetinaCare AI Pipeline — Comprehensive Project State & Progress Report

> **Target Audience:** Any AI model, coding agent, or human engineer inspecting this repository.
> **Guarantee:** Reading this single file provides complete context on what is built, what has run, exact verification metrics, architectural decisions, and step-by-step instructions to resume without wasting tokens or credits.
>
> **Last Updated:** 2026-10-24 13:42 IST  
> **Status Summary:** P0 (Setup), P1 (Ingest/Audit/Splits), P2 (Preprocess 500), P3 (Anatomy 500), Unit Tests (13/13), and End-to-End Smoke Test (P1–P8 on 20 cases) are **100% COMPLETE & VERIFIED**.

---

## 1. Executive Summary & Dashboard

```
[ Phase 0: Setup & Discovery       ]  ====>  100% COMPLETED (Dependencies, venv, CPU fallback, Mode A decided)
[ Phase 1: Ingest, Audit & Splits  ]  ====>  100% COMPLETED (7,183 files, 500 locked sample, 0 leakage)
[ Unit Tests (13 Tests)            ]  ====>  100% PASSED (pytest in 8.24s)
[ Smoke Test (20 Cases, P1–P8)     ]  ====>  100% PASSED (All 8 phases validated, 20 clinician PDFs)
[ Phase 2: Preprocess & Quality    ]  ====>  100% COMPLETED (All 500 cases processed: clf, hires, clf_raw)
[ Phase 3: Anatomy Localization    ]  ====>  100% COMPLETED (All 500 cases: 489 high-conf, 11 flagged)
[ Phase 4: Lesion Segmentation     ]  ====>  READY TO TRAIN (U-Net Mode A on 81 IDRiD sets; Mode B validated)
[ Phase 5: DR Severity Grading     ]  ====>  READY TO TRAIN (E1-E4 architectures ready, data ready)
[ Phase 6: Fusion & Explainability ]  ====>  READY FOR FULL RUN (Validated in smoke test)
[ Phase 7: Clinical Safety Net     ]  ====>  READY FOR FULL RUN (S01–S10 rules validated in smoke test)
[ Phase 8: Reporting & Scorecard   ]  ====>  READY FOR FULL RUN (ReportLab 2-page A4 PDFs validated)
```

---

## 2. Environment & Runtime Specifications

| Attribute | Specification | Operational Note |
|---|---|---|
| **Python** | 3.11.9 | Located at `retinacare/.venv/Scripts/python.exe` |
| **Package** | `retinacare-0.1.0` | Installed in editable mode (`pip install -e .`). Imports like `from retinacare.common...` resolve globally. |
| **Hardware** | AMD Ryzen (16 Logical Cores), Integrated Radeon GPU (No CUDA) | Native Windows. **CPU Fallback Mode is permanently ON.** |
| **TensorFlow** | 2.21.0 | CPU execution mode (`oneDNN` enabled, `train.cpu_fallback: true` in config). |
| **Core Libraries** | OpenCV 5.0.0, NumPy 2.4.6, SciPy 1.17.1, Pandas 3.1.0, Scikit-Learn 1.9.1, Scikit-Image 0.26.0, Matplotlib 3.11.2, Pillow 13.0.0, ImageHash 4.3.2, PyYAML 6.0.3, ReportLab 5.1.0, PyTest 9.1.1 | Pinned in `retinacare/requirements.txt` and verified in `retinacare/artifacts/env_snapshot.txt`. |
| **Working Root** | `C:\Users\Bhaskar\OneDrive\Desktop\Retina care Dataset` | **Strict Rule:** Raw datasets (`aptos2019-*`, `archive (1)`, `AGAR300*`) are strictly read-only. |
| **Code Root** | `C:\Users\Bhaskar\OneDrive\Desktop\Retina care Dataset\retinacare\` | All code, virtual environment, artifacts, metadata, and results reside here. |
| **Random Seed** | `42` | Enforced deterministically across Python, NumPy, and TensorFlow. |
| **Matplotlib Cache** | `retinacare/artifacts/matplotlib_cache` | Set via `MPLCONFIGDIR` to eliminate Windows permission errors on `.matplotlib/fontlist*.lock`. |

---

## 3. Dataset Audit & Sample Cohort (P1 Results)

### 3.1 Raw Discovery
- **Total Files Scanned:** 7,183
- **Total Images:** 7,094
- **Total Labelled Images:** 6,003
- **Datasets Discovered:**
  - `aptos2019-blindness-detection`: 3,662 train images (`train.csv`), 1,928 test images (`test.csv`).
  - `archive (1)` (IDRiD): 81 paired image-mask sets (54 train, 27 test) covering Microaneurysms (MA), Haemorrhages (HE), Hard Exudates (EX), Soft Exudates (SE), and Optic Disc (OD).
  - `AGAR300(IEEE DataPort)`: 28 unlabelled images.

### 3.2 Duplicate & Conflict Detection
- **Near-Duplicate Detection:** LSH-style sub-hash bucketing (4 bands on 16x16 pHash) checked **1,000 candidate pairs** (instead of 25.1 million $O(n^2)$ pairs), completing in seconds.
- **NCC Verification:** 965 near-duplicate pairs verified ($\text{NCC} \ge 0.95$).
- **Clusters:** 6,119 duplicate groups (`DG-00001` to `DG-06119`).
- **Conflict Handling:** 184 groups exhibited discordant diagnostic labels across duplicates. All 184 conflict groups were strictly excluded from the sampling pool.

### 3.3 Locked 500-Sample Manifest
- **Manifest File:** `data/metadata/sample_500_manifest.csv`
- **SHA-256 Lock:** `d9be8ab141cccbb52192fe57a66a924cfa82a2de3420ef2d53cf2aedf93a1b0e` (in `sample_500_manifest.sha256`).
- **Split Strategy:** Group-stratified (`StratifiedGroupKFold`) grouped by `dup_group_id` with **zero group leakage** verified by unit tests.
- **Split Distribution:**
  - `train`: 350 cases (70.0%)
  - `val`: 75 cases (15.0%)
  - `test`: 75 cases (15.0%)
- **Class Balance (Floor $\ge 20$ per class enforced):**
  - Grade 0 (No DR): 308 cases
  - Grade 1 (Mild): 29 cases
  - Grade 2 (Moderate): 101 cases
  - Grade 3 (Severe): 29 cases
  - Grade 4 (Proliferative): 33 cases

---

## 4. Phase-by-Phase Implementation & Verification

### Phase 0: Discovery & Decision Gates — ✅ PASS
- **Lesion Segmentation Mode Decision:** Mode A selected (justification documented in `results/audit/lesion_mode_decision.md`). IDRiD provides 81 paired ground-truth masks, surpassing the threshold of $\ge 20$.
- **CPU Fallback Configuration:** Configured in `configs/config.yaml` with reduced epoch limits for grading (Phase 1: 5, Phase 2: 10) and lesion segmentation (Stage 1: 5, Stage 2: 15).

### Phase 1: Ingestion, Audit, Registry & Splits — ✅ PASS
- Generated `data/metadata/id_registry.csv` (7,094 synthetic patient IDs: `SP-000001` to `SP-007094`).
- Generated 500 initial `CaseRecord` JSON skeletons in `artifacts/cases/`.
- Generated `results/audit/audit_summary.json` and `data/metadata/sample_500_summary.md`.

### Phase 2: Preprocessing & Quality Assessment — ✅ PASS (500/500 Succeeded)
- **Parallel Optimization:** Multi-threaded using `ThreadPoolExecutor(max_workers=8)` over 16 CPU cores, completing all 500 cases in under 20 minutes (100% success rate: 500 ok, 0 failed).
- **Processing Steps per Case:**
  1. Otsu thresholding + mathematical morphology $\to$ Field of View (FOV) mask.
  2. Square crop with tight bounding box and symmetric padding.
  3. Quality score computation: 4-factor objective assessment (Sharpness, Illumination, Contrast, FOV coverage).
  4. Illumination normalization via Ben Graham method ($4 \times I - 4 \times \text{GaussianBlur}(I, \sigma) + 128$) masked to FOV.
  5. Contrast Limited Adaptive Histogram Equalization (CLAHE) on LAB L-channel + direct green channel CLAHE.
  6. Median denoising filter ($k=3$).
  7. Multi-resolution rendering and saving:
     - `clf.png`: $256 \times 256$ enhanced RGB (grading input).
     - `clf_raw.png`: $256 \times 256$ raw crop RGB (ablation E4 input).
     - `hires.png`: $1024 \times 1024$ enhanced RGB (lesion & anatomy input).
     - `fov_mask.png`: $1024 \times 1024$ binary FOV mask.
     - `green_clahe.png`: $1024 \times 1024$ green CLAHE channel.
  8. All frame transformations, scale factors, and quality scores written back to `artifacts/cases/{case_id}.json`.

### Phase 3: Anatomy Localization — ✅ PASS (500/500 Processed)
- **Parallel Optimization:** Multi-threaded across 8 workers; completed in under 2 minutes.
- **Results:**
  - 489 cases localized with high confidence (`ok`).
  - 11 cases flagged as `low_confidence` due to borderline anatomical geometry or non-standard illumination (captured cleanly for downstream clinical safety flags).
  - 0 unhandled exceptions or crashes.
- **Outputs per Case:**
  - Optic Disc: Center coordinate $(x, y)$, diameter in pixels, fitted ellipse mask saved to `artifacts/p3_anatomy/{case_id}/disc_mask.png`, detection confidence score.
  - Fovea: Center coordinate $(x, y)$ via geometric projection vector from disc towards FOV center, refined by dark vascular convergence search.
  - Plausibility Check: Ratio of disc diameter to FOV diameter ($0.06$ to $0.22$), fovea-to-disc distance in disc diameters ($2.0$ to $3.0\text{ DD}$), and FOV inclusion bounds.

### Phase 4: Lesion Segmentation — ✅ READY TO TRAIN
- **Code:** `src/retinacare/p4_lesion/`
- **Architecture:** U-Net with pre-trained EfficientNetB0 encoder (ImageNet weights) and 4-channel binary output (MA, HE, EX, SE).
- **Loss:** Combined Focal-Tversky loss ($\alpha=0.7, \beta=0.3, \gamma=0.75$) to handle extreme lesion class imbalance.
- **Patch Extraction:** Balanced patch sampling ($256 \times 256$ patches, 60% centered on lesions, 40% background) with real-time augmentations (flips, rotations, brightness/contrast jitter).
- **Classical Heuristic Fallback (Mode B):** Validated in smoke test (Black-Hat/Top-Hat morphology with Frangi vessel filter suppression).

### Phase 5: DR Severity Grading — ✅ READY TO TRAIN
- **Code:** `src/retinacare/p5_grading/`
- **Models Implemented:**
  - `E1`: 4-layer custom CNN baseline from scratch.
  - `E2`: EfficientNetB0 (ImageNet transfer learning) — primary candidate.
  - `E3`: ResNet50 (ImageNet transfer learning) — benchmark.
  - `E4`: EfficientNetB0 trained on `clf_raw.png` (preprocessing ablation).
- **Training Strategy:** Two-phase transfer learning (frozen backbone $\to$ top-layer fine-tuning) with Quadratic Weighted Kappa (QWK) and Referable Sensitivity validation callback.
- **Post-Processing:** Temperature scaling calibration via NLL optimization, continuous expected grade calculation, and Grad-CAM explainability heatmaps.

### Phase 6: Fusion & Explainability — ✅ VALIDATED
- **Code:** `src/retinacare/p6_fusion_explain/`
- **Functionality:**
  - False positive suppression: Masks out spurious exudates inside optic disc boundary and false lesions within 15px of FOV edge.
  - Macular Zone Classification: Categorizes lesions into Macular ($<1.0\text{ DD}$), Perimacular ($1.0\text{--}3.0\text{ DD}$), and Peripheral ($>3.0\text{ DD}$).
  - Multi-panel visual overlay: Produces $2 \times 2$ explanation panel:
    1. Preprocessed retinal image.
    2. Lesion color-coded segmentation overlay (MA=Red, HE=Orange, EX=Yellow, SE=Cyan).
    3. Grad-CAM attention heatmap.
    4. Combined clinical diagnosis map with anatomical landmarks (Disc circle, Fovea crosshair, Macular zones).

### Phase 7: Clinical Safety Net — ✅ VALIDATED
- **Code:** `src/retinacare/p7_safety/`
- **10 Core Safety Rules Implemented:**
  - `S01`: Low image quality (sharpness, illumination, or contrast below threshold).
  - `S02`: Low model confidence ($\max(p) < 0.60$).
  - `S03`: High prediction entropy (uncertainty between adjacent grades).
  - `S04`: Model vs. Lesion discordance (low predicted grade but substantial lesion area).
  - `S05`: High predicted grade (3 or 4) with zero detected lesions.
  - `S06`: Macular involvement (lesions detected in central $1.0\text{ DD}$ zone).
  - `S07`: Disc / Fovea localization failure or geometric implausibility.
  - `S08`: Heuristic lesion detector disclaimer flag (Mode B).
  - `S09`: Grad-CAM mass concentrated outside FOV or on boundary artifacts.
  - `S10`: Camera/domain out-of-distribution shift.
- **Non-Negotiable Constraint:** `review_required` is **strictly True** on every single case record.

### Phase 8: Reporting & Scorecard — ✅ VALIDATED
- **Code:** `src/retinacare/p8_report/`
- **PDF Generation:** Full 2-page A4 Clinician Report generated via ReportLab:
  - Header: Synthetic Patient ID, Acquisition Date, Case Metadata.
  - Page 1: Diagnosis grade badge, referable probability, confidence score, full $2 \times 2$ visual panel, quality metrics breakdown.
  - Page 2: Lesion burden metrics table (counts and area by zone), triggered safety warnings with severity chips, plain-language patient explanation, mandatory clinical disclaimer (Appendix C), and doctor verification signature block.
- **Validation:** 20 complete PDFs generated in `artifacts/reports/` during smoke testing (~6–7 MB each, verified structurally and content-checked).

---

## 5. Verification Test Suite Results

### 5.1 PyTest Unit Test Suite (13/13 Passed in 8.24s)

```powershell
cd retinacare
.venv\Scripts\python.exe -m pytest tests/ -v
```

| Test Case | Module | Status | Verified Invariant |
|---|---|---|---|
| `test_class_allocation_exact_sum` | `test_p1.py` | **PASSED** | Proportional class allocation with min-20 floor sums exactly to requested sample size. |
| `test_id_registry_uniqueness_and_bijection` | `test_p1.py` | **PASSED** | Synthetic patient IDs (`SP-*`) are unique and form a strict 1-to-1 bijection with file paths. |
| `test_manifest_lock_and_tamper_detection` | `test_p1.py` | **PASSED** | SHA-256 locking detects bit-level tampering in manifest CSV. |
| `test_no_duplicate_group_leakage_across_splits` | `test_p1.py` | **PASSED** | Group stratification guarantees zero duplicate group intersection across train, val, and test. |
| `test_planted_duplicate_detection_fixtures` | `test_p1.py` | **PASSED** | Planted duplicate images (blurred, noise-added) are correctly clustered and flagged. |
| `test_fov_detection_synthetic` | `test_p2.py` | **PASSED** | FOV mask segmentation achieves $>95\%$ IoU on synthetic ground truth. |
| `test_crop_and_square_padding` | `test_p2.py` | **PASSED** | Cropped outputs and masks are strictly square ($H=W$). |
| `test_quality_score_sharpness_separation` | `test_p2.py` | **PASSED** | Sharpness score drops monotonically on progressively blurred inputs. |
| `test_quality_score_illumination_separation` | `test_p2.py` | **PASSED** | Illumination score correctly penalizes underexposed and overexposed retinas. |
| `test_preprocessing_determinism` | `test_p2.py` | **PASSED** | Two runs on the same input produce bitwise identical output arrays. |
| `test_optic_disc_localization_synthetic` | `test_p3.py` | **PASSED** | Classical detector localizes optic disc center within 5px of true center. |
| `test_fovea_localization_synthetic` | `test_p3.py` | **PASSED** | Geometric projection localizes fovea center within reasonable anatomical tolerance. |
| `test_blank_image_fails_soft` | `test_p3.py` | **PASSED** | Pitch-black or corrupted inputs fail gracefully without raising unhandled exceptions. |

### 5.2 End-to-End Smoke Test (20 Cases, All Phases P1–P8)

```powershell
cd retinacare
.venv\Scripts\python.exe -u scripts/smoke_test.py
```

- **Total Execution Time:** 289.01 seconds (~4.8 minutes).
- **Execution Breakdown:**
  - P1 (Ingest/Audit): 0.07s
  - P2 (Preprocess): 139.55s
  - P3 (Anatomy): 9.86s
  - P4 (Lesions Mode B): 36.46s
  - P5 (Grading): 0.17s
  - P6 (Fusion/Overlay): 24.09s
  - P7 (Safety Net): 0.00s
  - P8 (Reports): 78.71s
- **Invariants Checked:**
  - Status block for every phase (`p2`, `p3`, `p4`, `p5`, `p6`, `p7`, `p8`) populated and valid in all 20 case records.
  - 20 clinician PDF files generated on disk, each $>1,000$ bytes (average 6.5 MB).
  - Exit code: `0` (Clean pass).

---

## 6. Directory Map & File Artifact Status

```
retinacare/
├── .venv/                                        # [READY] Python 3.11.9 venv
├── pyproject.toml                                # [READY] Editable package config + pytest paths
├── requirements.txt                              # [READY] Pinned dependencies
├── ASSUMPTIONS.md                                # [READY] Log of architectural assumptions
├── PROGRESS.md                                   # [THIS FILE] Source of truth for project state
├── configs/
│   └── config.yaml                               # [READY] Central pipeline configuration
├── data/
│   ├── metadata/
│   │   ├── file_inventory.csv                    # [READY] 7,183 files cataloged
│   │   ├── labels_clean.csv                      # [READY] 6,003 labels extracted
│   │   ├── dup_groups.csv                        # [READY] 6,119 duplicate groups mapped
│   │   ├── id_registry.csv                       # [READY] 7,094 synthetic patient IDs assigned
│   │   ├── sample_500_manifest.csv               # [READY] 500-case locked cohort
│   │   ├── sample_500_manifest.sha256            # [READY] SHA-256 lock checksum
│   │   └── sample_500_summary.md                 # [READY] Cohort distribution summary
├── artifacts/
│   ├── env_snapshot.txt                          # [READY] Package versions snapshot
│   ├── matplotlib_cache/                         # [READY] Custom writable matplotlib cache
│   ├── cases/                                    # [READY] 500 JSON case records (P2 & P3 updated)
│   ├── p2_preprocess/                            # [READY] 500 folders (clf.png, clf_raw.png, hires.png, fov_mask.png, green_clahe.png)
│   ├── p3_anatomy/                               # [READY] 500 folders (disc_mask.png, coordinates in JSON)
│   ├── p4_lesion/                                # [PARTIAL] 20 smoke cases populated; full run after training
│   ├── p5_grading/                               # [PARTIAL] 20 smoke cases populated; full run after training
│   ├── p6_fusion/                                # [PARTIAL] 20 smoke cases populated (overlays, combined panels)
│   ├── p7_safety/                                # [PARTIAL] 20 smoke cases populated (flags in JSON)
│   ├── reports/                                  # [PARTIAL] 20 publication-grade PDFs; 500 after full run
│   └── models/                                   # [PENDING] Destination for trained model weights
├── results/
│   ├── audit/
│   │   ├── audit_summary.json                    # [READY] Audit counts & statistics
│   │   └── lesion_mode_decision.md               # [READY] Mode A justification document
│   ├── experiments/                              # [PENDING] Model training logs & metrics
│   └── plots/                                    # [PENDING] ROC/PR curves & confusion matrices
├── scripts/
│   ├── run_pipeline.py                           # [READY] CLI orchestrator (--stage, --scope, --case)
│   ├── smoke_test.py                             # [READY] 20-case end-to-end smoke test
│   ├── train_lesion.py                           # [READY] U-Net Mode A training script
│   ├── train_grading.py                          # [READY] Grading models E1-E4 training script
│   └── evaluate_all.py                           # [READY] Final evaluation & RUN_REPORT.md generator
└── tests/
    ├── test_p1.py                                # [READY] 5 unit tests (All Passed)
    ├── test_p2.py                                # [READY] 5 unit tests (All Passed)
    └── test_p3.py                                # [READY] 3 unit tests (All Passed)
```

---

## 7. Next Actions & Execution Guide

Any engineer or AI agent resuming this task should follow this exact sequence:

### Step 1: Train U-Net Lesion Segmentation Model (Mode A)
Train on the 81 paired IDRiD image-mask sets using the CPU fallback settings (Stage 1: 5 epochs, Stage 2: 15 epochs):
```powershell
cd "C:\Users\Bhaskar\OneDrive\Desktop\Retina care Dataset\retinacare"
.venv\Scripts\python.exe scripts/train_lesion.py
```
*Outputs:* `artifacts/models/best_lesion_unet.keras`, `results/experiments/L1/test_lesion_metrics.json`.

### Step 2: Train DR Severity Grading Models (Experiments E1 to E4)
Train the 4 grading architectures on the preprocessed 350 train / 75 val images:
```powershell
.venv\Scripts\python.exe scripts/train_grading.py --experiment all
```
*Outputs:* Models in `artifacts/models/` (`E1_scratch.keras`, `E2_efficientnet.keras`, `E3_resnet50.keras`, `E4_efficientnet_raw.keras`), temperature scaling parameters, and validation QWK metrics.

### Step 3: Run Full Pipeline on All 500 Cohort Cases
Execute stages P4 through P8 on the full `sample500` cohort using the newly trained models:
```powershell
.venv\Scripts\python.exe scripts/run_pipeline.py --stage all --scope sample500
```
*Outputs:* Complete segmentation masks, grading probabilities, Grad-CAM overlays, safety evaluations, and 500 clinician PDF reports.

### Step 4: Final Evaluation and Final Report Generation
Compute test set performance (evaluated strictly once per Rule R6), generate bootstrap 95% confidence intervals, and write the final scorecard and report:
```powershell
.venv\Scripts\python.exe scripts/evaluate_all.py
```
*Outputs:* `results/scorecard.json`, `results/scorecard.md`, and top-level `RUN_REPORT.md`.
