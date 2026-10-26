"""Train DR severity grading experiments E1 to E4 and select the final model."""
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
from retinacare.common.io_utils import read_image_rgb
from retinacare.common.seeds import set_seed
from retinacare.p5_grading.adapters import get_adapter
from retinacare.p5_grading.models import build_scratch_cnn, build_efficientnetb0, build_resnet50
from retinacare.p5_grading.train import train_grading_model
from retinacare.p5_grading.calibrate import fit_temperature_scaling
from retinacare.p5_grading.evaluate import evaluate_grading_model


def load_grading_split(manifest_df: pd.DataFrame, split_name: str, repo_root: Path, artifacts_dir: Path, variant_raw: bool = False):
    """Load preprocessed clf images and labels for a specific split."""
    sub = manifest_df[manifest_df["split"] == split_name]
    x_list = []
    y_list = []
    ids = []

    img_filename = "clf_raw.png" if variant_raw else "clf.png"

    for _, r in sub.iterrows():
        case_id = r["synthetic_patient_id"]
        clf_p = artifacts_dir / "p2_preprocess" / case_id / img_filename
        if not clf_p.is_file():
            # Fallback to standard clf.png if raw not generated
            clf_p = artifacts_dir / "p2_preprocess" / case_id / "clf.png"

        img = read_image_rgb(clf_p)
        if img is not None:
            x_list.append(img)
            y_list.append(int(r["label"]))
            ids.append(case_id)

    return np.array(x_list, dtype=np.uint8), np.array(y_list, dtype=np.int32), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", "E1", "E2", "E3", "E4"], default="all")
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

    manifest_path = repo_root / "data" / "metadata" / "sample_500_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    # CPU fallback settings
    cpu_fallback = cfg.get("train", {}).get("cpu_fallback", {}).get("enabled", True)
    ep1 = args.epochs1 or (cfg["train"]["cpu_fallback"]["phase1_epochs"] if cpu_fallback else 10)
    ep2 = args.epochs2 or (cfg["train"]["cpu_fallback"]["phase2_epochs"] if cpu_fallback else 30)

    print(f"Loading data for DR grading experiments (CPU fallback: {cpu_fallback}, Phase 1: {ep1} epochs, Phase 2: {ep2} epochs)...")

    train_x, train_y, _ = load_grading_split(manifest, "train", repo_root, artifacts_dir)
    val_x, val_y, _ = load_grading_split(manifest, "val", repo_root, artifacts_dir)
    test_x, test_y, _ = load_grading_split(manifest, "test", repo_root, artifacts_dir)

    print(f"Data loaded: Train={len(train_x)}, Val={len(val_x)}, Test={len(test_x)}")

    experiments_to_run = ["E1", "E2", "E3", "E4"] if args.experiment == "all" else [args.experiment]
    results_summary = {}

    for exp_id in experiments_to_run:
        print(f"\n==================== Running Experiment {exp_id} ====================")
        exp_dir = results_dir / "experiments" / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = models_dir / exp_id

        # Setup model and adapter
        if exp_id == "E1":
            model = build_scratch_cnn()
            adapter_name = "rescaling_0_1"
        elif exp_id in ["E2", "E4"]:
            model = build_efficientnetb0()
            adapter_name = "efficientnet"
        elif exp_id == "E3":
            model = build_resnet50()
            adapter_name = "resnet50"

        adapter = get_adapter(adapter_name)
        train_x_adapted = adapter(train_x.copy())
        val_x_adapted = adapter(val_x.copy())
        test_x_adapted = adapter(test_x.copy())

        # If E4, reload raw_crop variants
        if exp_id == "E4":
            raw_train_x, raw_train_y, _ = load_grading_split(manifest, "train", repo_root, artifacts_dir, variant_raw=True)
            raw_val_x, raw_val_y, _ = load_grading_split(manifest, "val", repo_root, artifacts_dir, variant_raw=True)
            raw_test_x, raw_test_y, _ = load_grading_split(manifest, "test", repo_root, artifacts_dir, variant_raw=True)
            train_x_adapted = adapter(raw_train_x.copy())
            val_x_adapted = adapter(raw_val_x.copy())
            test_x_adapted = adapter(raw_test_x.copy())

        # Train model
        trained_model, train_history = train_grading_model(
            model,
            train_x_adapted, train_y,
            val_x_adapted, val_y,
            checkpoint_dir=ckpt_dir,
            phase1_epochs=ep1,
            phase2_epochs=ep2,
            batch_size=16
        )

        # Fit temperature scaling on validation set
        val_preds = trained_model.predict(val_x_adapted, batch_size=16, verbose=0)
        val_logits = np.log(np.clip(val_preds, 1e-12, 1.0 - 1e-12))
        optimal_t = fit_temperature_scaling(val_logits, val_y)
        print(f"Optimal Temperature T for {exp_id}: {optimal_t:.3f}")

        # Validation evaluation
        val_metrics = evaluate_grading_model(
            trained_model, val_x_adapted, val_y,
            model_id=exp_id, split_name="val", repo_root=repo_root, temperature=optimal_t
        )
        print(f"Validation Score: QWK={val_metrics['qwk']:.3f}, Ref Sensitivity={val_metrics['referable_dr']['sensitivity']:.3f}")

        # Model card
        model_card = {
            "model_id": exp_id,
            "architecture": trained_model.name,
            "input_size": 256,
            "adapter": adapter_name,
            "temperature_T": optimal_t,
            "val_metrics": val_metrics,
            "phase1_epochs": ep1,
            "phase2_epochs": ep2,
            "manifest_sha256": "LOCKED",
            "limitations": "Single-dataset sample training, rare class uncertainty, research prototype."
        }
        with open(ckpt_dir / "model_card.json", "w", encoding="utf-8") as f:
            json.dump(model_card, f, indent=2)

        results_summary[exp_id] = {
            "val_score": val_metrics["qwk"] + 0.5 * val_metrics["referable_dr"]["sensitivity"],
            "model": trained_model,
            "temperature": optimal_t,
            "model_card": model_card,
            "test_x": test_x_adapted,
            "test_y": test_y
        }

    # Model Selection strictly on validation score (Section P5.6)
    best_exp_id = max(results_summary.keys(), key=lambda k: results_summary[k]["val_score"])
    print(f"\n=======================================================")
    print(f" SELECTED BEST FINAL MODEL ON VALIDATION: {best_exp_id} ")
    print(f"=======================================================")

    # Copy / link best model to standard best_grading_model.keras
    best_info = results_summary[best_exp_id]
    best_model = best_info["model"]
    best_model.save(str(models_dir / "best_grading_model.keras"))
    with open(models_dir / "best_grading_model_card.json", "w", encoding="utf-8") as f:
        json.dump(best_info["model_card"], f, indent=2)

    # Evaluate test split exactly once for all models (Rule R6)
    print("\n--- Evaluating Test Split (Exactly Once Per Rule R6) ---")
    test_results = {}
    for exp_id in experiments_to_run:
        m_info = results_summary[exp_id]
        m = m_info["model"]
        t = m_info["temperature"]
        tx = m_info["test_x"]
        ty = m_info["test_y"]

        t_metrics = evaluate_grading_model(
            m, tx, ty, model_id=exp_id, split_name="test", repo_root=repo_root, temperature=t
        )
        test_results[exp_id] = t_metrics
        with open(results_dir / "experiments" / exp_id / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(t_metrics, f, indent=2)
        print(f"Test Set [{exp_id}]: QWK={t_metrics['qwk']:.3f} (CI: {t_metrics.get('qwk_ci_95')}), Ref Sens={t_metrics['referable_dr']['sensitivity']:.3f}")

    print("\nGrading experiments complete.")


if __name__ == "__main__":
    main()
