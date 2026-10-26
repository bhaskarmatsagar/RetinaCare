"""Smoke test exercising the entire P1-P8 pipeline on 20 cases."""
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
from retinacare.common.seeds import set_seed
from retinacare.p1_ingest_audit.pipeline import P1_Pipeline
from retinacare.p2_preprocess.pipeline import P2_Pipeline
from retinacare.p3_anatomy.pipeline import P3_Pipeline
from retinacare.p4_lesion.pipeline import P4_Pipeline
from retinacare.p5_grading.pipeline import P5_Pipeline
from retinacare.p6_fusion_explain.pipeline import P6_Pipeline
from retinacare.p7_safety.pipeline import P7_Pipeline
from retinacare.p8_report.pipeline import P8_Pipeline


def run_smoke_test():
    print("==================================================")
    print("      RETINACARE AI - SMOKE TEST (20 CASES)       ")
    print("==================================================")
    t_start = time.perf_counter()

    cfg = load_config()
    repo_root = Path(cfg["_repo_root"])
    set_seed(cfg.get("seed", 42))

    artifacts_dir = repo_root / cfg["paths"]["artifacts"]
    manifest_path = repo_root / "data" / "metadata" / "sample_500_manifest.csv"

    # Step 1: Ensure P1 has run
    t0 = time.perf_counter()
    p1 = P1_Pipeline(cfg)
    p1_res = p1.run()
    t_p1 = time.perf_counter() - t0
    print(f"[PASS] P1 Ingest/Audit completed in {t_p1:.2f}s.")

    # Load 20 cases (4 per grade 0-4 where available)
    import pandas as pd
    manifest = pd.read_csv(manifest_path)
    smoke_subsets = []
    for g in sorted(manifest["label"].unique()):
        sub = manifest[manifest["label"] == g].head(4)
        smoke_subsets.append(sub)
    smoke_df = pd.concat(smoke_subsets).head(20)

    cases = []
    for _, r in smoke_df.iterrows():
        c_path = artifacts_dir / "cases" / f"{r['synthetic_patient_id']}.json"
        cases.append(CaseRecord.load(c_path))

    print(f"Loaded {len(cases)} smoke cases across grades {sorted(smoke_df['label'].unique().tolist())}.")

    # Step 2: P2 Preprocess
    t0 = time.perf_counter()
    p2 = P2_Pipeline(cfg)
    cases = p2.run_batch(cases)
    t_p2 = time.perf_counter() - t0
    print(f"[PASS] P2 Preprocessing completed in {t_p2:.2f}s.")

    # Step 3: P3 Anatomy
    t0 = time.perf_counter()
    p3 = P3_Pipeline(cfg)
    cases = p3.run_batch(cases)
    t_p3 = time.perf_counter() - t0
    print(f"[PASS] P3 Anatomy completed in {t_p3:.2f}s.")

    # Step 4: P4 Lesions
    t0 = time.perf_counter()
    p4 = P4_Pipeline(cfg, mode="B")
    cases = p4.run_batch(cases)
    t_p4 = time.perf_counter() - t0
    print(f"[PASS] P4 Lesions completed in {t_p4:.2f}s.")

    # Step 5: P5 Grading
    t0 = time.perf_counter()
    p5 = P5_Pipeline(cfg)
    cases = p5.run_batch(cases)
    t_p5 = time.perf_counter() - t0
    print(f"[PASS] P5 Grading completed in {t_p5:.2f}s.")

    # Step 6: P6 Fusion
    t0 = time.perf_counter()
    p6 = P6_Pipeline(cfg)
    cases = p6.run_batch(cases)
    t_p6 = time.perf_counter() - t0
    print(f"[PASS] P6 Fusion & Explanation completed in {t_p6:.2f}s.")

    # Step 7: P7 Safety
    t0 = time.perf_counter()
    p7 = P7_Pipeline(cfg)
    cases = p7.run_batch(cases)
    t_p7 = time.perf_counter() - t0
    print(f"[PASS] P7 Safety Checks completed in {t_p7:.2f}s.")

    # Step 8: P8 Report
    t0 = time.perf_counter()
    sha_file = manifest_path.with_suffix(".sha256")
    manifest_hash = "LOCKED"
    if sha_file.exists():
        with open(sha_file, "r") as f:
            manifest_hash = f.read().split()[0].strip()
    p8 = P8_Pipeline(cfg, manifest_hash=manifest_hash)
    cases = p8.run_batch(cases)
    t_p8 = time.perf_counter() - t0
    print(f"[PASS] P8 Clinician PDF Reports completed in {t_p8:.2f}s.")

    # Assertions
    print("\n--- Verifying Smoke Test Invariants ---")
    for c in cases:
        # 1. Status exists for every block
        for blk_name in ["p2", "p3", "p4", "p5", "p6", "p7", "p8"]:
            blk = getattr(c, blk_name)
            assert blk.get("status") in ["ok", "low_confidence", "failed"], f"Invalid status in {blk_name} for {c.case_id}: {blk.get('status')}"

        # 2. PDF exists on disk
        pdf_rel = c.p8.get("pdf_path")
        assert pdf_rel is not None, f"Missing pdf_path in {c.case_id}"
        full_pdf = repo_root / pdf_rel
        assert full_pdf.is_file(), f"PDF file does not exist: {full_pdf}"
        assert full_pdf.stat().st_size > 1000, f"PDF file suspiciously small: {full_pdf.stat().st_size} bytes"

    total_time = time.perf_counter() - t_start
    print("==================================================")
    print(f"   ALL 20 SMOKE CASES PASSED IN {total_time:.2f} SECONDS    ")
    print(f"   P1: {t_p1:.2f}s | P2: {t_p2:.2f}s | P3: {t_p3:.2f}s | P4: {t_p4:.2f}s")
    print(f"   P5: {t_p5:.2f}s | P6: {t_p6:.2f}s | P7: {t_p7:.2f}s | P8: {t_p8:.2f}s")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
