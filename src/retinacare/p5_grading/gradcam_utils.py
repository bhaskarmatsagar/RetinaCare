"""Grad-CAM computation utilities for CNN explainability."""
from typing import Tuple, Optional
import cv2
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the name of the last 4D convolutional/activation layer in the model."""
    for layer in reversed(model.layers):
        if hasattr(layer, "output_shape") and isinstance(layer.output_shape, tuple) and len(layer.output_shape) == 4:
            return layer.name
    # Check if there is an inner functional model (e.g. EfficientNet backbone)
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for inner_layer in reversed(layer.layers):
                if hasattr(inner_layer, "output_shape") and isinstance(inner_layer.output_shape, tuple) and len(inner_layer.output_shape) == 4:
                    return inner_layer.name
    raise ValueError("Could not find a 4D convolutional feature layer in the model.")


def get_gradcam_model(model: tf.keras.Model, last_conv_layer_name: Optional[str] = None) -> tf.keras.Model:
    """Construct a multi-output model mapping input_image -> (last_conv_output, logits/predictions)."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)
    gradcam_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    return gradcam_model


def compute_gradcam(
    gradcam_model: tf.keras.Model,
    img_array: np.ndarray,
    target_class: Optional[int] = None
) -> np.ndarray:
    """Compute normalized 2D Grad-CAM heatmap for a single preprocessed image."""
    # Ensure batch dimension: shape (1, H, W, C)
    if img_array.ndim == 3:
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)
    else:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(img_tensor)
        if target_class is None:
            target_class = int(tf.argmax(predictions[0]))
        loss = predictions[:, target_class]

    # Compute gradients of target class output w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv outputs by importance
    conv_outputs_val = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs_val), axis=-1)

    # Apply ReLU
    cam = tf.maximum(cam, 0)
    cam_np = cam.numpy()

    # Normalize to [0, 1]
    max_val = np.max(cam_np)
    if max_val > 1e-6:
        cam_np = cam_np / max_val
    else:
        cam_np = np.zeros_like(cam_np)

    # Resize to match original input size
    h, w = img_array.shape[:2]
    cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(cam_resized, 0.0, 1.0).astype(np.float32)
