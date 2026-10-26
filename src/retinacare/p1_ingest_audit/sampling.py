"""Sample selection (500 images) with proportional class allocation, minimum quotas, and manifest locking."""
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def compute_class_allocation(
    class_counts: Dict[int, int],
    total_sample_size: int = 500,
    min_per_class: int = 20
) -> Dict[int, int]:
    """Allocate counts per class proportional to distribution with a minimum floor."""
    classes = sorted(class_counts.keys())
    total_eligible = sum(class_counts.values())
    if total_eligible < total_sample_size:
        raise ValueError(f"Eligible pool ({total_eligible}) is smaller than requested sample size ({total_sample_size})")

    # Initial proportional allocation
    alloc = {}
    for c in classes:
        raw_alloc = int(round((class_counts[c] / total_eligible) * total_sample_size))
        alloc[c] = max(min_per_class, min(raw_alloc, class_counts[c]))

    # Adjust difference by trimming or expanding from largest available classes
    diff = sum(alloc.values()) - total_sample_size
    while diff != 0:
        if diff > 0:
            # Need to decrease from classes with count > min_per_class
            c_reduce = max(
                [c for c in classes if alloc[c] > min_per_class],
                key=lambda c: alloc[c]
            )
            alloc[c_reduce] -= 1
            diff -= 1
        else:
            # Need to increase
            c_increase = max(
                [c for c in classes if alloc[c] < class_counts[c]],
                key=lambda c: alloc[c]
            )
            alloc[c_increase] += 1
            diff += 1

    return alloc


def select_500_sample(
    registry_df: pd.DataFrame,
    conflict_group_ids: Set[str],
    sample_size: int = 500,
    min_per_class: int = 20,
    seed: int = 42
) -> pd.DataFrame:
    """Select the locked 500-image sample enforcing duplicate-group separation."""
    rng = np.random.default_rng(seed)

    # 1. Eligible pool
    eligible = registry_df[
        (registry_df["labelled"] == True) &
        (registry_df["label"].notnull()) &
        (~registry_df["dup_group_id"].isin(conflict_group_ids))
    ].copy()

    eligible["label"] = eligible["label"].astype(int)
    class_counts = eligible["label"].value_counts().to_dict()

    allocation = compute_class_allocation(class_counts, sample_size, min_per_class)

    selected_rows = []
    used_dup_groups: Set[str] = set()

    for c in sorted(allocation.keys()):
        target_n = allocation[c]
        c_pool = eligible[eligible["label"] == c].copy()

        # Shuffle deterministically
        indices = rng.permutation(len(c_pool))
        c_pool_shuffled = c_pool.iloc[indices]

        # First pass: pick images from unrepresented duplicate groups
        pass1 = []
        pass2 = []
        for _, row in c_pool_shuffled.iterrows():
            dg = row["dup_group_id"]
            if dg not in used_dup_groups:
                pass1.append(row)
                used_dup_groups.add(dg)
            else:
                pass2.append(row)

        chosen = pass1[:target_n]
        if len(chosen) < target_n:
            needed = target_n - len(chosen)
            chosen.extend(pass2[:needed])

        selected_rows.extend(chosen)

    sample_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    assert len(sample_df) == sample_size, f"Expected {sample_size} sample rows, got {len(sample_df)}"
    return sample_df


def lock_manifest(manifest_csv_path: Path) -> str:
    """Compute sha256 hash of manifest and save to .sha256 file."""
    h = hashlib.sha256()
    with open(manifest_csv_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    digest = h.hexdigest()
    sha_file = manifest_csv_path.with_suffix(".sha256")
    with open(sha_file, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {manifest_csv_path.name}\n")
    return digest


def verify_manifest_lock(manifest_csv_path: Path) -> bool:
    """Verify that the manifest sha256 matches its locked hash file."""
    sha_file = manifest_csv_path.with_suffix(".sha256")
    if not sha_file.exists():
        return False
    with open(sha_file, "r", encoding="utf-8") as f:
        expected = f.read().split()[0].strip()

    h = hashlib.sha256()
    with open(manifest_csv_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual = h.hexdigest()
    return actual == expected
