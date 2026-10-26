"""Seed management for determinism across RetinaCare."""
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Enable deterministic ops if supported
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass
