"""Model architectures for DR grading experiments E1, E2, E3."""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_scratch_cnn(input_shape: Tuple[int, int, int] = (256, 256, 3), num_classes: int = 5) -> models.Model:
    """E1 Baseline: 4-block CNN built from scratch."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    filters = [32, 64, 128, 256]
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dr_grade")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="grading_scratch_cnn")
    return model


def build_efficientnetb0(input_shape: Tuple[int, int, int] = (256, 256, 3), num_classes: int = 5) -> models.Model:
    """E2 / E4: Pretrained EfficientNetB0 with custom classification head."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    # Adapter preprocessing applied upstream or in layer
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs
    )

    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dr_grade")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="grading_efficientnetb0")
    return model


def build_resnet50(input_shape: Tuple[int, int, int] = (256, 256, 3), num_classes: int = 5) -> models.Model:
    """E3: Pretrained ResNet50 with custom classification head."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs
    )

    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dr_grade")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="grading_resnet50")
    return model
