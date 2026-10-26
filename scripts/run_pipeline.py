"""RetinaCare CLI Orchestrator."""
import argparse
import logging
from pathlib import Path
import sys
import time

# Add src to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
import os
os.environ["MPLCONFIGDIR"] = str(repo_root / "artifacts" / "matplotlib_cache")

from retinacare.common.config import load_config
from retinacare.common.case_record import CaseRecord
from retinacare.common.logging_utils import setup_logger
from retinacare.common.seeds import set_seed
from retinacare.p1_ingest_audit.pipeline import P1_Pipeline
from retinacare.p2_preprocess.pipeline import P2_Pipeline
from retinacare.p3_anatomy.pipeline import P3_Pipeline
from retinacare.p4_lesion.pipeline import P4_Pipeline
from retinacare.p5_grading.pipeline import P5_Pipeline
from retinacare.p6_fusion_explain.pipeline import P6_Pipeline
from retinacare.p7_safety.pipeline import P7_Pipeline
from retinacare.p8_report.pipeline import P8_Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="RetinaCare AI Multi-Pipeline Runner")
    parser.add_argument("--stage", choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "all"], default="all")
    parser.add_argument("--scope", choices=["smoke20", "sample500", "full", "demo_unlabelled"], default="sample500")
    parser.add_argument("--case", type=str, default=None, help="Run single case ID (e.g. SP-000123)")
    parser.add_argument("--force", action="store_true", help="Force recomputation ignoring cache")
    parser.add_argument("--force-resample", action="store_true", help="Force regenerating sample_500_manifest.csv")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    return parser.parse_args()


def load_scope_cases(repo_root: Path, manifest_path: Path, artifacts_dir: Path, scope: str, single_case: str = None) -> list:
    """Load CaseRecord objects for the requested scope."""
    cases_dir = artifacts_dir / "cases"
    if not manifest_path.exists():
        return []

    import pandas as pd
    manifest = pd.read_csv(manifest_path)

    if single_case:
        manifest = manifest[manifest["synthetic_patient_id"] == single_case]
    elif scope == "smoke20":
        # Stratified pick of 20 (4 per grade if available)
        sampled = []
        for g in sorted(manifest["label"].unique()):
            sub = manifest[manifest["label"] == g]
            sampled.append(sub.head(4))
        manifest = pd.concat(sampled).head(20)

    cases = []
    for _, r in manifest.iterrows():
        case_id = r["synthetic_patient_id"]
        c_file = cases_dir / f"{case_id}.json"
        if c_file.exists():
            cases.append(CaseRecord.load(c_file))
        else:
            cr = CaseRecord(
                case_id=case_id,
                synthetic_patient_id=case_id,
                image_id=r["image_id"],
                source_path=r["source_path"],
                sha256=r["sha256"] if "sha256" in r else "",
                dup_group_id=r["dup_group_id"] if "dup_group_id" in r else "DG-00001",
                label=int(r["label"]) if pd.notnull(r["label"]) else None,
                split=r["split"] if "split" in r else "train",
                in_sample_500=True
            )
            cases.append(cr)

    return cases


def main():
    args = parse_args()
    cfg = load_config(args.config)
    repo_root = Path(cfg["_repo_root"])
    set_seed(cfg.get("seed", 42))

    log_dir = repo_root / "logs"
    logger = setup_logger("retinacare", log_dir=log_dir)
    logger.info(f"=== Starting RetinaCare CLI Runner (stage={args.stage}, scope={args.scope}) ===")

    artifacts_dir = repo_root / cfg["paths"]["artifacts"]
    manifest_path = repo_root / "data" / "metadata" / "sample_500_manifest.csv"

    # Step 1: P1 Ingestion / Audit
    if args.stage in ["p1", "all"]:
        p1 = P1_Pipeline(cfg)
        p1_res = p1.run(force_resample=args.force_resample)
        logger.info(f"P1 Result: {p1_res['status']}, Inventory: {p1_res['inventory_count']}, Sample: {p1_res['sample_count']}")

    if args.stage == "p1":
        logger.info("Stage p1 complete.")
        return

    # Load cases for downstream stages
    cases = load_scope_cases(repo_root, manifest_path, artifacts_dir, args.scope, args.case)
    logger.info(f"Loaded {len(cases)} cases for scope={args.scope}.")
    if not cases:
        logger.error("No cases available. Ensure P1 has run.")
        return

    # Stage P2: Preprocess
    if args.stage in ["p2", "all"]:
        logger.info(f"--- Running P2 Preprocess & Quality on {len(cases)} cases ---")
        p2 = P2_Pipeline(cfg)
        cases = p2.run_batch(cases)
        ok_cnt = sum(1 for c in cases if c.p2.get("status") == "ok")
        logger.info(f"P2 Complete: {ok_cnt}/{len(cases)} succeeded.")

    if args.stage == "p2":
        return

    # Stage P3: Anatomy
    if args.stage in ["p3", "all"]:
        logger.info(f"--- Running P3 Anatomy on {len(cases)} cases ---")
        p3 = P3_Pipeline(cfg)
        cases = p3.run_batch(cases)
        ok_cnt = sum(1 for c in cases if c.p3.get("status") == "ok")
        logger.info(f"P3 Complete: {ok_cnt}/{len(cases)} localized.")

    # Stage P4: Lesions
    if args.stage in ["p4", "all"]:
        logger.info(f"--- Running P4 Lesion Segmentation on {len(cases)} cases ---")
        # Check if trained U-Net checkpoint exists
        model_path = artifacts_dir / "models" / "best_lesion_unet.keras"
        lesion_model = None
        mode = "B"
        if model_path.exists():
            import tensorflow as tf
            from retinacare.p4_lesion.losses import combined_focal_tversky_loss
            lesion_model = tf.keras.models.load_model(str(model_path), custom_objects={"combined_focal_tversky_loss": combined_focal_tversky_loss})
            mode = "A"
        p4 = P4_Pipeline(cfg, model=lesion_model, mode=mode)
        cases = p4.run_batch(cases)
        ok_cnt = sum(1 for c in cases if c.p4.get("status") == "ok")
        logger.info(f"P4 Complete: {ok_cnt}/{len(cases)} processed (Mode {mode}).")

    # Stage P5: Grading
    if args.stage in ["p5", "all"]:
        logger.info(f"--- Running P5 Severity Grading on {len(cases)} cases ---")
        grading_path = artifacts_dir / "models" / "best_grading_model.keras"
        grading_model = None
        if grading_path.exists():
            import tensorflow as tf
            grading_model = tf.keras.models.load_model(str(grading_path))
        p5 = P5_Pipeline(cfg, model=grading_model)
        cases = p5.run_batch(cases)
        ok_cnt = sum(1 for c in cases if c.p5.get("status") == "ok")
        logger.info(f"P5 Complete: {ok_cnt}/{len(cases)} graded.")

    # Stage P6: Fusion & Explain
    if args.stage in ["p6", "all"]:
        logger.info(f"--- Running P6 Fusion & Explain on {len(cases)} cases ---")
        gradcam_m = None
        grading_path = artifacts_dir / "models" / "best_grading_model.keras"
        if grading_path.exists():
            import tensorflow as tf
            from retinacare.p5_grading.gradcam_utils import get_gradcam_model
            base_m = tf.keras.models.load_model(str(grading_path))
            try:
                gradcam_m = get_gradcam_model(base_m)
            except Exception:
                pass
        p6 = P6_Pipeline(cfg, gradcam_model=gradcam_m)
        cases = p6.run_batch(cases)
        ok_cnt = sum(1 for c in cases if c.p6.get("status") == "ok")
        logger.info(f"P6 Complete: {ok_cnt}/{len(cases)} overlays and panels generated.")

    # Stage P7: Safety
    if args.stage in ["p7", "all"]:
        logger.info(f"--- Running P7 Safety Checks on {len(cases)} cases ---")
        p7 = P7_Pipeline(cfg)
        cases = p7.run_batch(cases)
        flagged_cnt = sum(1 for c in cases if len(c.p7.get("flags", [])) > 0)
        logger.info(f"P7 Complete: {flagged_cnt}/{len(cases)} triggered safety flags.")

    # Stage P8: Reports
    if args.stage in ["p8", "all"]:
        logger.info(f"--- Running P8 Reports on {len(cases)} cases ---")
        sha_file = manifest_path.with_suffix(".sha256")
        manifest_hash = "LOCKED"
        if sha_file.exists():
            with open(sha_file, "r") as f:
                manifest_hash = f.read().split()[0].strip()
        p8 = P8_Pipeline(cfg, manifest_hash=manifest_hash)
        cases = p8.run_batch(cases)
        pdf_cnt = sum(1 for c in cases if c.p8.get("status") == "ok")
        logger.info(f"P8 Complete: {pdf_cnt}/{len(cases)} clinician PDFs generated.")

    logger.info("=== RetinaCare Run Finished Successfully ===")


if __name__ == "__main__":
    main()
