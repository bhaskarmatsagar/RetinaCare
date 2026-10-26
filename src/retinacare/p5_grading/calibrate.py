"""Post-hoc temperature scaling calibration and uncertainty estimation."""
from typing import Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize


def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute temperature-scaled softmax probabilities."""
    t = max(1e-4, float(temperature))
    scaled = logits / t
    # Numerical stability
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
    return probs


def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit a single optimal temperature scalar T by minimizing Negative Log-Likelihood (NLL) on validation set."""
    n_samples = len(val_labels)
    y_true_onehot = np.eye(val_logits.shape[1])[val_labels]

    def nll_objective(t_arr):
        t = t_arr[0]
        if t <= 0:
            return 1e9
        probs = softmax_with_temperature(val_logits, t)
        probs = np.clip(probs, 1e-12, 1.0)
        nll = -np.sum(y_true_onehot * np.log(probs)) / n_samples
        return nll

    res = minimize(nll_objective, x0=[1.0], bounds=[(0.05, 5.0)], method="L-BFGS-B")
    optimal_t = float(res.x[0])
    return optimal_t


def compute_prediction_outputs(
    logits: np.ndarray,
    temperature: float = 1.0,
    referable_threshold: int = 2
) -> Dict[str, Any]:
    """Compute calibrated probabilities, expected grade, referable probability, confidence, and entropy."""
    probs = softmax_with_temperature(logits, temperature)
    grade = int(np.argmax(probs, axis=1)[0])
    k_vec = np.arange(probs.shape[1])
    expected_grade = float(np.sum(k_vec * probs[0]))
    referable_prob = float(np.sum(probs[0, referable_threshold:]))
    confidence = float(np.max(probs[0]))
    entropy = float(-np.sum(probs[0] * np.log(np.clip(probs[0], 1e-12, 1.0))))

    return {
        "probs": [round(float(p), 4) for p in probs[0]],
        "grade": grade,
        "expected_grade": round(expected_grade, 3),
        "referable_prob": round(referable_prob, 4),
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4)
    }
