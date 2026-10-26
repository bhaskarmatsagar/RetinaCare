"""Stratified group-aware train/val/test splitting."""
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_group_stratified_splits(
    sample_df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """Create leak-free train/val/test splits grouped by dup_group_id and stratified by label."""
    df = sample_df.copy()
    y = df["label"].astype(int).values
    groups = df["dup_group_id"].values

    # Step 1: Use StratifiedGroupKFold to split into Train (approx 70%) and Val+Test (approx 30%)
    # For a 70/30 split, we can use 10 folds: 7 folds for train, 3 folds for val+test
    sgkf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)

    splits = list(sgkf_outer.split(df, y, groups))
    # Combine first 7 folds into train, remaining 3 folds into temp
    train_idx = np.concatenate([splits[i][1] for i in range(7)])
    temp_idx = np.concatenate([splits[i][1] for i in range(7, 10)])

    # Step 2: Split temp into val (approx 15%) and test (approx 15%)
    temp_df = df.iloc[temp_idx].copy()
    y_temp = temp_df["label"].astype(int).values
    groups_temp = temp_df["dup_group_id"].values

    sgkf_inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    inner_splits = list(sgkf_inner.split(temp_df, y_temp, groups_temp))
    val_idx = temp_idx[inner_splits[0][1]]
    test_idx = temp_idx[inner_splits[1][1]]

    df["split"] = "unassigned"
    df.loc[df.index.isin(train_idx), "split"] = "train"
    df.loc[df.index.isin(val_idx), "split"] = "val"
    df.loc[df.index.isin(test_idx), "split"] = "test"

    # Assert no group leaks across splits
    train_groups = set(df[df["split"] == "train"]["dup_group_id"])
    val_groups = set(df[df["split"] == "val"]["dup_group_id"])
    test_groups = set(df[df["split"] == "test"]["dup_group_id"])

    assert len(train_groups.intersection(val_groups)) == 0, "Leakage: dup_group shared between train and val!"
    assert len(train_groups.intersection(test_groups)) == 0, "Leakage: dup_group shared between train and test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Leakage: dup_group shared between val and test!"

    return df
