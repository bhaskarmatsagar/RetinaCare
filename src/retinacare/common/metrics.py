"""Comprehensive evaluation metrics and bootstrap confidence interval computation."""
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa score."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for multi-class predictions."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_dice_iou(mask_true: np.ndarray, mask_pred: np.ndarray, eps: float = 1e-7) -> Tuple[float, float, float, float]:
    """Compute Dice, IoU, Precision, and Recall between binary masks."""
    t = (mask_true > 0).astype(bool)
    p = (mask_pred > 0).astype(bool)

    intersection = np.logical_and(t, p).sum()
    sum_t = t.sum()
    sum_p = p.sum()

    dice = (2.0 * intersection + eps) / (sum_t + sum_p + eps)
    union = sum_t + sum_p - intersection
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (sum_p + eps)
    recall = (intersection + eps) / (sum_t + eps)

    return float(dice), float(iou), float(precision), float(recall)


def compute_grading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    referable_threshold: int = 2
) -> Dict[str, Any]:
    """Calculate all required P5 grading metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    qwk = compute_qwk(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0)
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=supp)) if np.sum(supp) > 0 else 0.0

    # Referable DR metrics (grade >= referable_threshold)
    y_true_ref = (y_true >= referable_threshold).astype(int)
    prob_ref = np.sum(probs[:, referable_threshold:], axis=1)
    y_pred_ref = (y_pred >= referable_threshold).astype(int)

    ref_tp = int(np.sum((y_true_ref == 1) & (y_pred_ref == 1)))
    ref_fn = int(np.sum((y_true_ref == 1) & (y_pred_ref == 0)))
    ref_fp = int(np.sum((y_true_ref == 0) & (y_pred_ref == 1)))
    ref_tn = int(np.sum((y_true_ref == 0) & (y_pred_ref == 0)))

    ref_sensitivity = float(ref_tp / (ref_tp + ref_fn)) if (ref_tp + ref_fn) > 0 else 0.0
    ref_specificity = float(ref_tn / (ref_tn + ref_fp)) if (ref_tn + ref_fp) > 0 else 0.0
    ref_precision = float(ref_tp / (ref_tp + ref_fp)) if (ref_tp + ref_fp) > 0 else 0.0

    try:
        ref_roc_auc = float(roc_auc_score(y_true_ref, prob_ref))
    except Exception:
        ref_roc_auc = 0.0
    try:
        ref_pr_auc = float(average_precision_score(y_true_ref, prob_ref))
    except Exception:
        ref_pr_auc = 0.0

    # Rare class recalls (grades 3 and 4)
    rec_severe = float(rec[3])
    rec_prolif = float(rec[4])

    # Ordinal errors
    grade_abs_diff = np.abs(y_true - y_pred)
    mean_abs_error = float(np.mean(grade_abs_diff))
    share_off_by_2_or_more = float(np.mean(grade_abs_diff >= 2))

    # Calibration
    ece = compute_ece(probs, y_true)
    # Brier score (one-hot multi-class)
    y_true_oh = np.eye(5)[y_true]
    brier = float(np.mean(np.sum((probs - y_true_oh) ** 2, axis=1)))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "qwk": qwk,
        "confusion_matrix": cm,
        "per_class": {
            f"grade_{i}": {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(supp[i])
            } for i in range(5)
        },
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "referable_dr": {
            "sensitivity": ref_sensitivity,
            "specificity": ref_specificity,
            "precision": ref_precision,
            "roc_auc": ref_roc_auc,
            "pr_auc": ref_pr_auc,
            "counts": {"tp": ref_tp, "fn": ref_fn, "fp": ref_fp, "tn": ref_tn}
        },
        "severe_recall": rec_severe,
        "proliferative_recall": rec_prolif,
        "ordinal_error": {
            "mean_absolute_error": mean_abs_error,
            "share_off_ge_2": share_off_by_2_or_more
        },
        "calibration": {
            "ece": ece,
            "brier_score": brier
        }
    }


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Compute point estimate and [low, high] bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    if probs is not None:
        point_est = float(metric_fn(y_true, y_pred, probs))
    else:
        point_est = float(metric_fn(y_true, y_pred))

    estimates = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        yt_boot = y_true[idx]
        yp_boot = y_pred[idx]
        if probs is not None:
            p_boot = probs[idx]
            val = metric_fn(yt_boot, yp_boot, p_boot)
        else:
            val = metric_fn(yt_boot, yp_boot)
        estimates.append(val)

    alpha = 1.0 - ci_level
    low = float(np.percentile(estimates, 100 * (alpha / 2.0)))
    high = float(np.percentile(estimates, 100 * (1.0 - alpha / 2.0)))
    return point_est, low, high
