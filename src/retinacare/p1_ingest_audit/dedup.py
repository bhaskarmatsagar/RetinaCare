"""Duplicate and near-duplicate detection with NCC verification and Union-Find clustering."""
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import logging as _logging
import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image


class UnionFind:
    """Disjoint-set / Union-Find data structure."""
    def __init__(self, elements: List[str]):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def preprocess_for_phash(img_bgr: np.ndarray, target_size: int = 256) -> Image.Image:
    """Center crop and normalize contrast for perceptual hashing."""
    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    cy, cx = h // 2, w // 2
    crop = img_bgr[cy - min_dim // 2 : cy + min_dim // 2, cx - min_dim // 2 : cx + min_dim // 2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    norm = cv2.equalizeHist(resized)
    return Image.fromarray(norm)


def compute_ncc(gray1: np.ndarray, gray2: np.ndarray) -> float:
    """Normalized cross-correlation on 128x128 grayscale images."""
    g1 = cv2.resize(gray1, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    g2 = cv2.resize(gray2, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)

    g1_mean = np.mean(g1)
    g2_mean = np.mean(g2)
    g1_std = np.std(g1)
    g2_std = np.std(g2)

    if g1_std < 1e-5 or g2_std < 1e-5:
        return 0.0

    g1_norm = (g1 - g1_mean) / g1_std
    g2_norm = (g2 - g2_mean) / g2_std
    ncc = np.mean(g1_norm * g2_norm)
    return float(ncc)


def find_duplicate_groups(
    image_records: List[Dict[str, Any]],
    phash_size: int = 16,
    max_hamming: int = 6,
    verify_ncc_min: float = 0.95
) -> Tuple[pd.DataFrame, Dict[str, str], Set[str]]:
    """Detect exact and near-duplicates, cluster via Union-Find, and flag label conflicts."""
    all_ids = [rec["image_id"] for rec in image_records]
    uf = UnionFind(all_ids)

    # 1. Exact duplicates by sha256
    sha_map: Dict[str, List[str]] = {}
    for rec in image_records:
        sha_map.setdefault(rec["sha256"], []).append(rec["image_id"])

    for s, group in sha_map.items():
        if len(group) > 1:
            for other in group[1:]:
                uf.union(group[0], other)

    # 2. Near duplicates using phash + NCC verification
    # Compute phash and cache 128x128 grayscale arrays
    hash_list = []
    gray_cache: Dict[str, np.ndarray] = {}
    _plog = _logging.getLogger("retinacare.p1.dedup")
    total_recs = len(image_records)

    for rec_idx, rec in enumerate(image_records):
        p = Path(rec["full_path"])
        if rec_idx % 500 == 0:
            _plog.info(f"  pHash progress: {rec_idx}/{total_recs}")
        try:
            # Safe read on Windows
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            pil_img = preprocess_for_phash(img)
            h = imagehash.phash(pil_img, hash_size=phash_size)
            hash_list.append((rec["image_id"], h))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_cache[rec["image_id"]] = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        except Exception:
            continue

    # Near-duplicate matching via sub-hash bucketing (avoids O(n²) full scan)
    # Split each phash into 4 sub-hashes; images sharing any sub-hash bucket
    # are candidate near-duplicates and get NCC-verified.
    import logging
    _logger = logging.getLogger("retinacare.p1.dedup")

    n = len(hash_list)
    _logger.info(f"Building sub-hash buckets for {n} images...")
    num_bands = 4
    sub_hash_buckets: Dict[Tuple, List[int]] = {}
    for idx_i, (img_id, h) in enumerate(hash_list):
        h_arr = h.hash.flatten()
        band_size = len(h_arr) // num_bands
        for band_idx in range(num_bands):
            band = tuple(h_arr[band_idx * band_size : (band_idx + 1) * band_size].tolist())
            key = (band_idx, band)
            sub_hash_buckets.setdefault(key, []).append(idx_i)

    # Gather unique candidate pairs from buckets
    candidate_pairs: Set[Tuple[int, int]] = set()
    for bucket_members in sub_hash_buckets.values():
        if len(bucket_members) > 1 and len(bucket_members) < 500:  # skip huge buckets (noise)
            for bi in range(len(bucket_members)):
                for bj in range(bi + 1, len(bucket_members)):
                    pair = (min(bucket_members[bi], bucket_members[bj]),
                            max(bucket_members[bi], bucket_members[bj]))
                    candidate_pairs.add(pair)

    _logger.info(f"Candidate pairs from bucketing: {len(candidate_pairs)} (vs {n*(n-1)//2} full)")

    verified = 0
    for i, j in candidate_pairs:
        id_i, hash_i = hash_list[i]
        id_j, hash_j = hash_list[j]
        dist = hash_i - hash_j
        if dist <= max_hamming:
            if id_i in gray_cache and id_j in gray_cache:
                ncc = compute_ncc(gray_cache[id_i], gray_cache[id_j])
                if ncc >= verify_ncc_min:
                    uf.union(id_i, id_j)
                    verified += 1

    _logger.info(f"NCC-verified near-duplicate unions: {verified}")

    # 3. Form groups
    clusters: Dict[str, List[str]] = {}
    for img_id in all_ids:
        root = uf.find(img_id)
        clusters.setdefault(root, []).append(img_id)

    # Map each image_id to DG-xxxxx
    dup_group_map: Dict[str, str] = {}
    label_map = {rec["image_id"]: rec.get("label") for rec in image_records}

    group_rows = []
    conflict_group_ids = set()

    for idx, (root, members) in enumerate(sorted(clusters.items(), key=lambda x: (len(x[1]), x[0]), reverse=True), start=1):
        dg_id = f"DG-{idx:05d}"
        labels_in_group = [label_map[m] for m in members if label_map.get(m) is not None]
        has_conflict = False
        if len(set(labels_in_group)) > 1:
            has_conflict = True
            conflict_group_ids.add(dg_id)

        for m in members:
            dup_group_map[m] = dg_id
            group_rows.append({
                "dup_group_id": dg_id,
                "image_id": m,
                "group_size": len(members),
                "label": label_map.get(m),
                "label_conflict": has_conflict
            })

    dup_df = pd.DataFrame(group_rows)
    return dup_df, dup_group_map, conflict_group_ids
