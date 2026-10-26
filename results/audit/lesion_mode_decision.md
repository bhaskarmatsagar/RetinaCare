# Lesion Segmentation Pipeline Mode Decision (P4)

**Decision:** **Mode A (Trained U-Net on Pixel-Level Annotations)**  
**Date of Audit:** 2026-10-24  
**Evaluator:** RetinaCare Discovery Engine (Phase 0)  

---

## 1. Summary of Evidence

Per Appendix A and Phase 0 discovery guidelines, `data_root` was recursively scanned for paired retinal fundus images and lesion masks. 

A complete, high-quality reference lesion segmentation dataset—**IDRiD (Indian Diabetic Retinopathy Image Dataset)**—was identified in the local workspace directory:
- Path: `archive (1)/A. Segmentation/A. Segmentation/`

### Groundtruth Lesions Present:
1. **Microaneurysms (MA):** `2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms` (54 masks) and `b. Testing Set/1. Microaneurysms` (27 masks).
2. **Haemorrhages (HE):** `2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages` (54 masks) and `b. Testing Set/2. Haemorrhages` (27 masks).
3. **Hard Exudates (EX):** `2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates` (54 masks) and `b. Testing Set/3. Hard Exudates` (27 masks).
4. **Soft Exudates (SE / Cotton-Wool Spots):** `2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates` (54 masks) and `b. Testing Set/4. Soft Exudates` (27 masks).
5. **Optic Disc (OD):** `2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc` (54 masks) and `b. Testing Set/5. Optic Disc` (27 masks).

### Paired Original Images:
- `1. Original Images/a. Training Set/`: 54 high-resolution retinal photographs (`.jpg`).
- `1. Original Images/b. Testing Set/`: 27 high-resolution retinal photographs (`.jpg`).
- **Total Paired (Image, ≥1 Mask) Items:** **81** (far exceeding the 20-image threshold for Mode A).

### Anatomy Groundtruth Discoveries (Bonus for P3):
In addition to lesion masks, groundtruth pixel coordinates for Optic Disc centers and Fovea centers were discovered under:
- `archive (1)/C. Localization/C. Localization/2. Groundtruths/`
  - `1. Optic Disc Center Location/` (train & test coordinate tables)
  - `2. Fovea Center Location/` (train & test coordinate tables)
This enables quantitative benchmark metrics (Dice, distance errors in disc diameters) for Pipeline P3 as well.

---

## 2. Mode Implementation Plan

1. **Pipeline P4 (Lesions):** 
   - Multi-label Keras U-Net with pretrained ImageNet encoder (`EfficientNetB0`), input shape 256×256×3, multi-head Sigmoid outputs for `[MA, HE, EX, SE]`.
   - Loss: `0.5 * Focal(γ=2) + 0.5 * Tversky(α=0.3, β=0.7)` with channel weightings `MA: 1.5, HE: 1.0, EX: 1.0, SE: 1.0`.
   - Sliding-window tile inference over 1024×1024 hires images with Gaussian blending and Test-Time Augmentation (TTA).
   - Validation and test evaluation reporting pixel-level Dice, IoU, Precision, Recall, and lesion-level hit metrics.
   - Outputs are flagged `validated: true`.

2. **Classical Detector (Baseline):**
   - The classical heuristic detector (`classical_detector.py`) will also be evaluated as an inspectable baseline against the Mode A U-Net on the official test split.
