"""Dataset discovery and file inventory generation."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
TABLE_EXTENSIONS = {".csv", ".xlsx"}


def discover_files(data_root: Path) -> pd.DataFrame:
    """Recursively discover all files under data_root and classify them."""
    records = []
    data_root = Path(data_root).resolve()

    for root, _, files in os.walk(data_root):
        root_path = Path(root)
        # Skip hidden or environment directories
        if any(part.startswith(".") or part in {"__pycache__", "venv", ".venv"} for part in root_path.parts):
            continue

        for f in files:
            file_path = root_path / f
            ext = file_path.suffix.lower()
            try:
                size = file_path.stat().st_size
            except Exception:
                size = 0

            rel_path = file_path.relative_to(data_root).as_posix()
            source_group = file_path.relative_to(data_root).parts[0] if len(file_path.relative_to(data_root).parts) > 1 else "root"

            if ext in IMAGE_EXTENSIONS:
                category = "image"
            elif ext in TABLE_EXTENSIONS:
                category = "table"
            else:
                category = "other"

            records.append({
                "path": rel_path,
                "full_path": str(file_path),
                "filename": f,
                "stem": file_path.stem,
                "extension": ext,
                "size_bytes": size,
                "category": category,
                "source_group": source_group
            })

    return pd.DataFrame(records)


def discover_label_tables(inventory_df: pd.DataFrame, data_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detect label tables and extract (image_stem, label) mappings."""
    table_rows = inventory_df[inventory_df["category"] == "table"]
    label_records = []
    metadata_summary = {}

    for _, row in table_rows.iterrows():
        full_path = Path(row["full_path"])
        if row["extension"] == ".csv":
            try:
                df = pd.read_csv(full_path)
            except Exception:
                continue

            # Identify image id column and diagnosis column
            id_col = None
            diag_col = None

            for c in df.columns:
                cl = str(c).lower().strip()
                if cl in ["id_code", "image_id", "image id", "image_name", "image name", "image"]:
                    id_col = c
                elif cl in ["diagnosis", "dr_grade", "retinopathy grade", "retinopathy_grade", "grade", "level"]:
                    diag_col = c

            if id_col is not None and diag_col is not None:
                for _, r in df.iterrows():
                    val = r[diag_col]
                    try:
                        label_int = int(val)
                        if 0 <= label_int <= 4:
                            stem = str(r[id_col]).strip()
                            # remove extension if present in id_code
                            if "." in stem:
                                stem = Path(stem).stem
                            label_records.append({
                                "image_id": stem,
                                "label": label_int,
                                "table_source": row["path"],
                                "source_group": row["source_group"]
                            })
                    except (ValueError, TypeError):
                        continue

    labels_df = pd.DataFrame(label_records)
    if not labels_df.empty:
        # Drop duplicates keeping first
        labels_df = labels_df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    metadata_summary["total_labels_found"] = len(labels_df)
    return labels_df, metadata_summary
