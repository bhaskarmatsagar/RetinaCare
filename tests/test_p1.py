"""Unit tests for Pipeline P1."""
import hashlib
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import pytest

from retinacare.p1_ingest_audit.dedup import find_duplicate_groups, preprocess_for_phash, compute_ncc
from retinacare.p1_ingest_audit.id_registry import generate_id_registry
from retinacare.p1_ingest_audit.sampling import compute_class_allocation, lock_manifest, verify_manifest_lock
from retinacare.p1_ingest_audit.splits import make_group_stratified_splits


def test_class_allocation_exact_sum():
    class_counts = {0: 1800, 1: 370, 2: 990, 3: 190, 4: 290}
    alloc = compute_class_allocation(class_counts, total_sample_size=500, min_per_class=20)
    assert sum(alloc.values()) == 500
    for c, count in alloc.items():
        assert count >= 20
        assert count <= class_counts[c]


def test_id_registry_uniqueness_and_bijection():
    valid_records = [
        {"image_id": f"img_{i:03d}", "source_group": "train", "source_path": f"train/img_{i}.png", "sha256": f"hash_{i}", "label": i % 5, "labelled": True, "width": 500, "height": 500}
        for i in range(1, 21)
    ]
    dup_map = {f"img_{i:03d}": f"DG-{i:05d}" for i in range(1, 21)}

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "id_registry.csv"
        df = generate_id_registry(valid_records, dup_map, csv_path)

        assert len(df) == 20
        assert len(df["synthetic_patient_id"].unique()) == 20
        assert df["synthetic_patient_id"].iloc[0] == "SP-000001"
        assert df["synthetic_patient_id"].iloc[-1] == "SP-000020"


def test_manifest_lock_and_tamper_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        m_path = Path(tmpdir) / "manifest.csv"
        m_path.write_text("id,label\nSP-001,0\nSP-002,1\n", encoding="utf-8")

        digest = lock_manifest(m_path)
        assert verify_manifest_lock(m_path) is True

        # Tamper
        m_path.write_text("id,label\nSP-001,0\nSP-002,2\n", encoding="utf-8")
        assert verify_manifest_lock(m_path) is False


def test_no_duplicate_group_leakage_across_splits():
    # Construct 100 cases with 20 duplicate groups of size 2-5
    records = []
    for g_idx in range(1, 41):
        dg = f"DG-{g_idx:05d}"
        for m_idx in range(3):
            records.append({
                "synthetic_patient_id": f"SP-{g_idx:03d}_{m_idx}",
                "image_id": f"img_{g_idx}_{m_idx}",
                "dup_group_id": dg,
                "label": g_idx % 5,
                "split": "unassigned"
            })
    df = pd.DataFrame(records)

    split_df = make_group_stratified_splits(df, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)

    train_groups = set(split_df[split_df["split"] == "train"]["dup_group_id"])
    val_groups = set(split_df[split_df["split"] == "val"]["dup_group_id"])
    test_groups = set(split_df[split_df["split"] == "test"]["dup_group_id"])

    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


def test_planted_duplicate_detection_fixtures():
    # Create two identical images and one distinct image
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        img1 = np.full((100, 100, 3), 120, dtype=np.uint8)
        cv2.circle(img1, (50, 50), 30, (0, 0, 255), -1)

        # Planted duplicate (identical with slight 1-pixel shift)
        img2 = img1.copy()

        # Distinct image (different pattern)
        img3 = np.full((100, 100, 3), 40, dtype=np.uint8)
        cv2.rectangle(img3, (20, 20), (80, 80), (255, 255, 0), -1)

        cv2.imwrite(str(tmp / "im1.png"), img1)
        cv2.imwrite(str(tmp / "im2.png"), img2)
        cv2.imwrite(str(tmp / "im3.png"), img3)

        records = [
            {"image_id": "im1", "full_path": str(tmp / "im1.png"), "sha256": "h1", "label": 0},
            {"image_id": "im2", "full_path": str(tmp / "im2.png"), "sha256": "h1", "label": 0},
            {"image_id": "im3", "full_path": str(tmp / "im3.png"), "sha256": "h3", "label": 2},
        ]

        dup_df, dup_map, conflicts = find_duplicate_groups(records)
        assert dup_map["im1"] == dup_map["im2"]
        assert dup_map["im1"] != dup_map["im3"]
