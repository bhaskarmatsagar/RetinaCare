"""Synthetic ID generation and registry management."""
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def generate_id_registry(
    valid_images: List[Dict[str, Any]],
    dup_group_map: Dict[str, str],
    out_csv_path: Path
) -> pd.DataFrame:
    """Assign deterministic synthetic IDs (SP-000001, ...) and write id_registry.csv."""
    # Deterministic sort order
    sorted_records = sorted(
        valid_images,
        key=lambda x: (str(x.get("source_group", "")), str(x.get("image_id", "")), str(x.get("source_path", "")))
    )

    assigned_rows = []
    seen_ids = set()

    for idx, rec in enumerate(sorted_records, start=1):
        sp_id = f"SP-{idx:06d}"
        if sp_id in seen_ids:
            raise ValueError(f"Duplicate synthetic ID generated: {sp_id}")
        seen_ids.add(sp_id)

        img_id = rec["image_id"]
        dg_id = dup_group_map.get(img_id, f"DG-{idx:05d}")

        assigned_rows.append({
            "synthetic_patient_id": sp_id,
            "image_id": img_id,
            "source_group": rec.get("source_group", "unknown"),
            "source_path": rec.get("source_path", ""),
            "sha256": rec.get("sha256", ""),
            "dup_group_id": dg_id,
            "label": rec.get("label"),
            "labelled": rec.get("labelled", False),
            "width": rec.get("width", 0),
            "height": rec.get("height", 0)
        })

    registry_df = pd.DataFrame(assigned_rows)

    # Assert uniqueness and bijection
    assert len(registry_df["synthetic_patient_id"].unique()) == len(registry_df), "Synthetic IDs are not unique!"
    assert len(registry_df["source_path"].unique()) == len(registry_df), "Source paths are not 1-to-1 with synthetic IDs!"

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    registry_df.to_csv(out_csv_path, index=False)
    return registry_df
