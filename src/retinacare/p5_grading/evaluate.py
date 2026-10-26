"""Evaluation of DR grading models with bootstrap confidence intervals and test-access logging."""
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from retinacare.common.metrics import compute_grading_metrics, bootstrap_ci, compute_qwk


def log_test_access(repo_root: Path, model_id: str, split_size: int, note: str = ""):
    """Log test-set evaluation call strictly per Rule R6."""
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    access_log = log_dir / "test_access.log"

    timestamp = datetime.now().isoformat()
    entry = f"{timestamp} | MODEL: {model_id} | SAMPLES: {split_size} | {note}\n"
    with open(access_log, "a", encoding="utf-8") as f:
        f.write(entry)


def evaluate_grading_model(
    model,
    x_data: np.ndarray,
    y_true: np.ndarray,
    model_id: str,
    split_name: str,
    repo_root: Path,
    temperature: float = 1.0,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """Run full evaluation on validation or test split."""
    if split_name.lower() == "test":
        log_test_access(repo_root, model_id, len(y_true), note="Final test evaluation")

    # Time inference
    t0 = time.perf_counter()
    raw_preds = model.predict(x_data, batch_size=16, verbose=0)
    t1 = time.perf_counter()
    inf_time_per_image = (t1 - t0) / max(1, len(x_data))

    # Apply temperature
    from retinacare.p5_grading.calibrate import softmax_with_temperature
    probs = softmax_with_temperature(raw_preds, temperature)
    y_pred = np.argmax(probs, axis=1)

    # Base metrics
    metrics = compute_grading_metrics(y_true, y_pred, probs, referable_threshold=2)
    metrics["inference_time_sec_per_image"] = float(inf_time_per_image)
    metrics["sample_count"] = int(len(y_true))
    metrics["model_id"] = model_id
    metrics["split"] = split_name
    metrics["temperature"] = float(temperature)

    # Bootstrap CIs for key metrics
    qwk_point, qwk_low, qwk_high = bootstrap_ci(
        compute_qwk, y_true, y_pred, n_resamples=n_bootstrap, ci_level=0.95
    )
    metrics["qwk_ci_95"] = [round(qwk_low, 3), round(qwk_high, 3)]

    def ref_sens_fn(yt, yp):
        tp = np.sum((yt >= 2) & (yp >= 2))
        fn = np.sum((yt >= 2) & (yp < 2))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    rs_point, rs_low, rs_high = bootstrap_ci(
        ref_sens_fn, y_true, y_pred, n_resamples=n_bootstrap, ci_level=0.95
    )
    metrics["referable_dr"]["sensitivity_ci_95"] = [round(rs_low, 3), round(rs_high, 3)]

    return metrics
