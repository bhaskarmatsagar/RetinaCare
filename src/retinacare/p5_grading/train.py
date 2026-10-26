"""Training routine for DR grading with custom QWK/Referable selection callback."""
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score


class QWKSelectionCallback(tf.keras.callbacks.Callback):
    """Custom validation callback evaluating QWK and Referable Sensitivity to checkpoint the best model."""
    def __init__(self, val_x: np.ndarray, val_y: np.ndarray, checkpoint_path: Path):
        super().__init__()
        self.val_x = val_x
        self.val_y = val_y
        self.checkpoint_path = checkpoint_path
        self.best_score = -999.0
        self.history_records = []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.val_x, batch_size=16, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        qwk = float(cohen_kappa_score(self.val_y, pred_labels, weights="quadratic"))

        # Referable sensitivity (grade >= 2)
        true_ref = self.val_y >= 2
        pred_ref = pred_labels >= 2
        tp = np.sum(true_ref & pred_ref)
        fn = np.sum(true_ref & (~pred_ref))
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        selection_score = qwk + 0.5 * sensitivity
        self.history_records.append({
            "epoch": epoch + 1,
            "qwk": qwk,
            "referable_sensitivity": sensitivity,
            "selection_score": selection_score,
            "val_loss": logs.get("val_loss") if logs else None
        })

        if selection_score > self.best_score:
            self.best_score = selection_score
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.checkpoint_path))


def compute_class_weights(train_labels: np.ndarray) -> Dict[int, float]:
    """Inverse frequency class weights normalised."""
    classes, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    weights = {}
    for c, cnt in zip(classes, counts):
        weights[int(c)] = float(total / (len(classes) * cnt))
    return weights


def train_grading_model(
    model: tf.keras.Model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    checkpoint_dir: Path,
    phase1_epochs: int = 5,
    phase2_epochs: int = 10,
    batch_size: int = 16,
    unfreeze_layers: int = 20
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Two-phase transfer learning for grading model."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_grading_model.keras"

    class_weights = compute_class_weights(train_y)

    # Phase 1: Frozen backbone
    for layer in model.layers[:-3]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    train_y_oh = tf.keras.utils.to_categorical(train_y, num_classes=5)
    val_y_oh = tf.keras.utils.to_categorical(val_y, num_classes=5)

    selection_cb = QWKSelectionCallback(val_x, val_y, best_model_path)

    h1 = model.fit(
        train_x, train_y_oh,
        validation_data=(val_x, val_y_oh),
        epochs=phase1_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[selection_cb],
        verbose=1
    )

    # Phase 2: Unfreeze last N layers (keeping BatchNormalization frozen)
    for layer in model.layers[-unfreeze_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)

    h2 = model.fit(
        train_x, train_y_oh,
        validation_data=(val_x, val_y_oh),
        epochs=phase2_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[selection_cb, reduce_lr, early_stop],
        verbose=1
    )

    # Reload best checkpoint
    if best_model_path.exists():
        best_model = tf.keras.models.load_model(str(best_model_path))
    else:
        best_model = model

    return best_model, {
        "phase1_history": h1.history,
        "phase2_history": h2.history,
        "selection_history": selection_cb.history_records,
        "best_selection_score": selection_cb.best_score
    }
