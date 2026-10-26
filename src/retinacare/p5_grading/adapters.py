"""Model-specific input preprocessing adapters."""
import numpy as np
import tensorflow as tf


def get_adapter(adapter_name: str):
    """Return model preprocessing adapter function by name."""
    name = adapter_name.lower()
    if name == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input
    elif name == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input
    elif name in ["scratch", "rescaling_0_1"]:
        return lambda x: x.astype(np.float32) / 255.0
    else:
        # Default fallback
        return lambda x: x.astype(np.float32) / 255.0
