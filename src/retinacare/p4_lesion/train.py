"""Two-stage transfer learning training routine for U-Net lesion segmentation."""
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from retinacare.p4_lesion.model import build_unet_lesion
from retinacare.p4_lesion.losses import combined_focal_tversky_loss
from retinacare.p4_lesion.patches import sample_patch_batch


class LesionPatchGenerator(tf.keras.utils.Sequence):
    """Keras Sequence yielding on-the-fly lesion-balanced augmented patch batches."""
    def __init__(self, images: List[np.ndarray], masks: List[np.ndarray], batch_size: int = 8, steps_per_epoch: int = 50, augment: bool = True):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.augment = augment

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        x, y = sample_patch_batch(
            self.images, self.masks, batch_size=self.batch_size, patch_size=256, lesion_ratio=0.6, augment=self.augment
        )
        return x, y


def train_unet_lesions(
    train_images: List[np.ndarray],
    train_masks: List[np.ndarray],
    val_images: List[np.ndarray],
    val_masks: List[np.ndarray],
    save_dir: Path,
    stage1_epochs: int = 5,
    stage2_epochs: int = 15,
    batch_size: int = 8
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Train U-Net in two stages (frozen backbone -> fine-tuning)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_lesion_unet.keras"

    model = build_unet_lesion(input_shape=(256, 256, 3), num_classes=4, backbone_name="efficientnetb0")

    # Stage 1: Freeze encoder
    for layer in model.layers:
        if "block" in layer.name or "stem" in layer.name or "top" in layer.name:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_focal_tversky_loss,
        metrics=["binary_accuracy"]
    )

    train_gen = LesionPatchGenerator(train_images, train_masks, batch_size=batch_size, steps_per_epoch=40, augment=True)
    val_gen = LesionPatchGenerator(val_images, val_masks, batch_size=batch_size, steps_per_epoch=15, augment=False)

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=stage1_epochs,
        verbose=1
    )

    # Stage 2: Unfreeze encoder
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_focal_tversky_loss,
        metrics=["binary_accuracy"]
    )

    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    ]

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=stage2_epochs,
        callbacks=callbacks,
        verbose=1
    )

    return model, {"history_stage1": history1.history, "history_stage2": history2.history}
