"""P1 Pipeline orchestrator for ingestion, audit, synthetic IDs, and sample creation."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import pandas as pd

from retinacare.common.case_record import CaseRecord
from retinacare.common.io_utils import compute_sha256
from retinacare.p1_ingest_audit.discover import discover_files, discover_label_tables
from retinacare.p1_ingest_audit.audit import audit_images, create_contact_sheet
from retinacare.p1_ingest_audit.dedup import find_duplicate_groups
from retinacare.p1_ingest_audit.id_registry import generate_id_registry
from retinacare.p1_ingest_audit.sampling import select_500_sample, lock_manifest, verify_manifest_lock
from retinacare.p1_ingest_audit.splits import make_group_stratified_splits


class P1_Pipeline:
    """Pipeline 1: Data Ingestion and Audit."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_root = Path(config["paths"]["data_root"]).resolve()
        self.artifacts_dir = Path(config.get("_repo_root", ".")) / config["paths"]["artifacts"]
        self.results_dir = Path(config.get("_repo_root", ".")) / config["paths"]["results"]
        self.data_dir = Path(config.get("_repo_root", ".")) / "data"
        self.meta_dir = self.data_dir / "metadata"
        self.audit_dir = self.results_dir / "audit"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("retinacare.p1")

    def run(self, force_resample: bool = False) -> Dict[str, Any]:
        """Execute the full P1 pipeline."""
        self.logger.info("Executing P1: Discovery, Audit, Dedup, IDs, and Sample Selection...")

        manifest_csv = self.meta_dir / "sample_500_manifest.csv"
        manifest_locked = manifest_csv.exists() and verify_manifest_lock(manifest_csv)
        if manifest_locked and not force_resample:
            self.logger.info("Manifest lock verified. Loading existing sample 500 manifest and artifacts.")
            sample_df = pd.read_csv(manifest_csv)
            inv_path = self.meta_dir / "file_inventory.csv"
            inventory_df = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()
            audit_summary = {}
            audit_path = self.audit_dir / "audit_summary.json"
            if audit_path.exists():
                with open(audit_path, "r", encoding="utf-8") as f:
                    audit_summary = json.load(f)
            return {
                "status": "PASS",
                "inventory_count": len(inventory_df),
                "sample_count": len(sample_df),
                "manifest_path": str(manifest_csv),
                "audit_summary": audit_summary
            }

        # Step 1: Discover files
        inventory_csv = self.meta_dir / "file_inventory.csv"
        inventory_df = discover_files(self.data_root)
        inventory_df.to_csv(inventory_csv, index=False)
        self.logger.info(f"Inventory saved: {len(inventory_df)} total files discovered.")

        # Step 2: Discover labels
        labels_clean_csv = self.meta_dir / "labels_clean.csv"
        labels_df, label_meta = discover_label_tables(inventory_df, self.data_root)
        labels_df.to_csv(labels_clean_csv, index=False)
        self.logger.info(f"Labels discovered: {len(labels_df)} labelled images.")

        # Map labels to images
        label_dict = dict(zip(labels_df["image_id"], labels_df["label"]))

        # Step 3: Build image records with sha256
        image_files = inventory_df[inventory_df["category"] == "image"].copy()
        image_records = []
        total_imgs = len(image_files)
        for idx, (_, r) in enumerate(image_files.iterrows()):
            if idx % 500 == 0:
                self.logger.info(f"  SHA-256 progress: {idx}/{total_imgs}")
            fp = Path(r["full_path"])
            h = compute_sha256(fp)
            stem = r["stem"]
            lbl = label_dict.get(stem)
            image_records.append({
                "image_id": stem,
                "full_path": str(fp),
                "source_path": r["path"],
                "source_group": r["source_group"],
                "sha256": h,
                "label": lbl,
                "labelled": lbl is not None,
                "width": 0,
                "height": 0
            })
        self.logger.info(f"  SHA-256 complete for {total_imgs} images.")

        # Step 4: Duplicate & near-duplicate detection
        dedup_cfg = self.config.get("dedup", {})
        dup_df, dup_group_map, conflict_group_ids = find_duplicate_groups(
            image_records,
            phash_size=dedup_cfg.get("phash_size", 16),
            max_hamming=dedup_cfg.get("max_hamming", 6),
            verify_ncc_min=dedup_cfg.get("verify_ncc_min", 0.95)
        )
        dup_csv = self.meta_dir / "dup_groups.csv"
        dup_df.to_csv(dup_csv, index=False)
        self.logger.info(f"Duplicates clustered into {len(dup_df['dup_group_id'].unique())} groups. Conflicts: {len(conflict_group_ids)}")

        # Step 5: Generate Synthetic IDs and Registry
        id_registry_csv = self.meta_dir / "id_registry.csv"
        registry_df = generate_id_registry(image_records, dup_group_map, id_registry_csv)
        self.logger.info(f"Synthetic ID Registry generated: {len(registry_df)} unique records.")

        # Step 6: 500 Sample selection and Manifest
        manifest_csv = self.meta_dir / "sample_500_manifest.csv"
        sample_cfg = self.config.get("sample", {})
        sample_size = sample_cfg.get("size", 500)
        min_per_class = sample_cfg.get("min_per_class", 20)

        manifest_locked = manifest_csv.exists() and verify_manifest_lock(manifest_csv)
        if manifest_locked and not force_resample:
            self.logger.info("Manifest lock verified. Loading existing sample 500 manifest.")
            sample_df = pd.read_csv(manifest_csv)
        else:
            self.logger.info("Selecting 500-image stratified sample...")
            sample_df = select_500_sample(
                registry_df,
                conflict_group_ids,
                sample_size=sample_size,
                min_per_class=min_per_class,
                seed=self.config.get("seed", 42)
            )

            # Step 7: Stratified group-aware splits
            split_cfg = sample_cfg.get("split", {"train": 0.70, "val": 0.15, "test": 0.15})
            sample_df = make_group_stratified_splits(
                sample_df,
                train_frac=split_cfg.get("train", 0.70),
                val_frac=split_cfg.get("val", 0.15),
                test_frac=split_cfg.get("test", 0.15),
                seed=self.config.get("seed", 42)
            )

            sample_df.to_csv(manifest_csv, index=False)
            lock_digest = lock_manifest(manifest_csv)
            self.logger.info(f"Sample manifest locked with SHA256: {lock_digest}")

        # Step 8: Audit summary and report
        audit_df = audit_images(inventory_df, max_samples=500)
        audit_summary = {
            "total_files": len(inventory_df),
            "total_images": len(image_files),
            "total_labelled": len(labels_df),
            "duplicate_groups": len(dup_df["dup_group_id"].unique()),
            "conflict_groups": len(conflict_group_ids),
            "sample_size": len(sample_df),
            "split_distribution": sample_df["split"].value_counts().to_dict(),
            "class_distribution_sample": sample_df["label"].value_counts().to_dict()
        }
        with open(self.audit_dir / "audit_summary.json", "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)

        # Write sample_500_summary.md
        summary_md = self.meta_dir / "sample_500_summary.md"
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("# Sample 500 Summary\n\n")
            f.write(f"- Date: {pd.Timestamp.now().isoformat()}\n")
            f.write(f"- Total sample size: {len(sample_df)}\n")
            f.write(f"- Split counts:\n")
            for sp, cnt in sample_df["split"].value_counts().items():
                f.write(f"  - `{sp}`: {cnt}\n")
            f.write(f"- Class counts in sample:\n")
            for cl, cnt in sample_df["label"].value_counts().sort_index().items():
                f.write(f"  - Grade {cl}: {cnt}\n")

        # Step 9: Create CaseRecord skeletons
        case_skeletons = []
        for _, r in sample_df.iterrows():
            cr = CaseRecord(
                case_id=r["synthetic_patient_id"],
                synthetic_patient_id=r["synthetic_patient_id"],
                image_id=r["image_id"],
                source_path=r["source_path"],
                sha256=r["sha256"],
                dup_group_id=r["dup_group_id"],
                label=int(r["label"]) if pd.notnull(r["label"]) else None,
                split=r["split"],
                in_sample_500=True
            )
            cr.save(self.artifacts_dir)
            case_skeletons.append(cr)

        self.logger.info(f"P1 Complete. Saved {len(case_skeletons)} case records.")
        return {
            "status": "PASS",
            "inventory_count": len(inventory_df),
            "sample_count": len(sample_df),
            "manifest_path": str(manifest_csv),
            "audit_summary": audit_summary
        }
