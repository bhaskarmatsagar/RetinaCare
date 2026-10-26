# Project Assumptions Log

Every assumption made during implementation (per Operating Rule R2) is logged here with What and Why.

| Timestamp | What | Why |
|---|---|---|
| 2026-10-24 | CPU execution fallback enabled (`train.cpu_fallback: true` with adjusted epochs) | Host GPU is integrated AMD Radeon without CUDA acceleration. Spec explicitly instructs: "If no GPU is available, continue on CPU with reduced settings (`train.cpu_fallback` in config: smaller image size, fewer epochs) and record this in ASSUMPTIONS.md". |
| 2026-10-24 | Raw dataset folder located at `C:\Users\Bhaskar\OneDrive\Desktop\Retina care Dataset` containing `aptos2019-blindness-detection`, `archive (1)` (IDRiD), and `AGAR300(IEEE DataPort)` | Provided by user as active workspace. Spec requires raw data to remain strictly read-only. `retinacare/` is maintained as the self-contained pipeline repository. |
| 2026-10-24 | P4 Lesion Segmentation operating mode set to Mode A | Real pixel-level annotated ground truths for MA, HE, EX, SE found in `archive (1)` (IDRiD) with >20 paired images (54 training, 27 testing). Spec Appendix A directs Mode A when >= 20 paired images exist. |
