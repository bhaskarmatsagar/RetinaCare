"""Keras U-Net implementation with pretrained EfficientNetB0 / ResNet50 encoder."""
from typing import Tuple, List
import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters: int, kernel_size: int = 3):
    """Two Conv2D - BatchNorm - ReLU layers."""
    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def decoder_block(x, skip, filters: int):
    """Upsampling2D + Concat + ConvBlock."""
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    if skip is not None:
        x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet_lesion(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 4,
    backbone_name: str = "efficientnetb0"
) -> models.Model:
    """Build U-Net with pretrained ImageNet encoder from keras.applications."""
    inputs = layers.Input(shape=input_shape, name="input_image")

    if backbone_name.lower() == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
        # Select skip connection layers:
        # skips: stem/block2 (128x128), block3 (64x64), block4 (32x32), block6 (16x16)
        skip_names = [
            "block2a_expand_activation",  # 128x128
            "block3a_expand_activation",  # 64x64
            "block4a_expand_activation",  # 32x32
            "block6a_expand_activation"   # 16x16
        ]
        skips = [base_model.get_layer(name).output for name in skip_names]
        bottleneck = base_model.get_layer("top_activation").output  # 8x8

    else:
        # ResNet50 fallback
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
        skip_names = [
            "conv1_relu",        # 128x128
            "conv2_block3_out",  # 64x64
            "conv3_block4_out",  # 32x32
            "conv4_block6_out"   # 16x16
        ]
        skips = [base_model.get_layer(name).output for name in skip_names]
        bottleneck = base_model.get_layer("conv5_block3_out").output  # 8x8

    # Decoder
    filters = [256, 128, 64, 32]
    d1 = decoder_block(bottleneck, skips[3], filters[0])  # -> 16x16
    d2 = decoder_block(d1, skips[2], filters[1])          # -> 32x32
    d3 = decoder_block(d2, skips[1], filters[2])          # -> 64x64
    d4 = decoder_block(d3, skips[0], filters[3])          # -> 128x128
    d5 = decoder_block(d4, None, 16)                      # -> 256x256

    # 1x1 Conv with Sigmoid for multi-label binary segmentation
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", name="lesion_masks")(d5)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"unet_{backbone_name}")
    return model
