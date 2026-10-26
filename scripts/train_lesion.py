"""Train U-Net lesion segmentation model (Mode A) or run classical evaluation (Mode B)."""
import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retinacare.common.config import load_config
from retinacare.common.seeds import set_seed
from retinacare.common.metrics import compute_dice_iou
from retinacare.p4_lesion.datasets import discover_lesion_dataset, load_paired_sample
from retinacare.p4_lesion.train import train_unet_lesions
from retinacare.p4_lesion.infer import sliding_window_inference, postprocess_lesion_masks
from retinacare.p4_lesion.losses import combined_focal_tversky_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs1", type=int, default=None)
    parser.add_argument("--epochs2", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    repo_root = Path(cfg["_repo_root"])
    set_seed(cfg.get("seed", 42))

    artifacts_dir = repo_root / cfg["paths"]["artifacts"]
    results_dir = repo_root / cfg["paths"]["results"]
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    mode, paired_records = discover_lesion_dataset(Path(cfg["paths"]["data_root"]))
    print(f"Lesion Dataset Discovery: Mode={mode}, Found {len(paired_records)} paired image-mask sets.")

    if mode == "A" and len(paired_records) >= 20:
        print("Starting Mode A: Training U-Net on real annotated ground truths...")
        # Split paired records
        train_recs = [r for r in paired_records if r["split"] == "train"]
        test_recs = [r for r in paired_records if r["split"] == "test"]

        # Carve validation split from train (approx 20%)
        np.random.seed(42)
        indices = np.random.permutation(len(train_recs))
        n_val = max(4, int(len(train_recs) * 0.20))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        actual_train_recs = [train_recs[i] for i in train_idx]
        actual_val_recs = [train_recs[i] for i in val_idx]

        print(f"Paired Split: Train={len(actual_train_recs)}, Val={len(actual_val_recs)}, Test={len(test_recs)}")

        # Load samples (1024x1024)
        print("Loading training and validation images/masks...")
        train_imgs, train_masks = [], []
        for r in actual_train_recs:
            im, m = load_paired_sample(r, target_size=1024)
            train_imgs.append(im)
            train_masks.append(m)

        val_imgs, val_masks = [], []
        for r in actual_val_recs:
            im, m = load_paired_sample(r, target_size=1024)
            val_imgs.append(im)
            val_masks.append(m)

        # CPU fallback epochs
        cpu_fallback = cfg.get("train", {}).get("cpu_fallback", {}).get("enabled", True)
        ep1 = args.epochs1 or (cfg["train"]["cpu_fallback"]["lesion_stage1_epochs"] if cpu_fallback else 10)
        ep2 = args.epochs2 or (cfg["train"]["cpu_fallback"]["lesion_stage2_epochs"] if cpu_fallback else 40)

        # Train U-Net
        model, history = train_unet_lesions(
            train_imgs, train_masks,
            val_imgs, val_masks,
            save_dir=models_dir,
            stage1_epochs=ep1,
            stage2_epochs=ep2,
            batch_size=8
        )

        # Save model
        model_save_path = models_dir / "best_lesion_unet.keras"
        model.save(str(model_save_path))
        print(f"U-Net model saved to {model_save_path}")

        # Evaluate on Test split
        print("\n--- Evaluating Lesion U-Net on Test Split ---")
        test_metrics = {}
        for r in test_recs:
            im, m_true = load_paired_sample(r, target_size=1024)
            pred_prob = sliding_window_inference(model, im, patch_size=256, stride=128, tta=False)
            pred_masks, _ = postprocess_lesion_masks(pred_prob, np.ones(im.shape[:2], dtype=np.uint8)*255)

            for i, c_name in enumerate(["MA", "HE", "EX", "SE"]):
                dice, iou, prec, rec = compute_dice_iou(m_true[:, :, i], pred_masks[c_name])
                test_metrics.setdefault(c_name, {"dice": [], "iou": [], "prec": [], "rec": []})
                test_metrics[c_name]["dice"].append(dice)
                test_metrics[c_name]["iou"].append(iou)
                test_metrics[c_name]["prec"].append(prec)
                test_metrics[c_name]["rec"].append(rec)

        mean_metrics = {
            c_name: {
                "mean_dice": round(float(np.mean(vals["dice"])), 4),
                "mean_iou": round(float(np.mean(vals["iou"])), 4),
                "mean_prec": round(float(np.mean(vals["prec"])), 4),
                "mean_rec": round(float(np.mean(vals["rec"])), 4),
            } for c_name, vals in test_metrics.items()
        }
        print("Test Lesion Metrics:")
        print(json.dumps(mean_metrics, indent=2))

        # Save metrics
        p4_res_dir = results_dir / "experiments" / "L1"
        p4_res_dir.mkdir(parents=True, exist_ok=True)
        with open(p4_res_dir / "test_lesion_metrics.json", "w", encoding="utf-8") as f:
            json.dump(mean_metrics, f, indent=2)

    else:
        print("Mode B: Running heuristic lesion correlation analysis...")
        # Covered in full evaluation script
        pass


if __name__ == "__main__":
    main()
