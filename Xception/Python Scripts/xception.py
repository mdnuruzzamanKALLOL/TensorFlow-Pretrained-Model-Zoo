import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _sep_bn(x, filters):
    """SeparableConv2D + BatchNorm (no activation)."""
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return x


def _entry_block(x, filters, first=False):
    """Entry flow residual block — downsamples by 2x."""
    residual = layers.Conv2D(filters, 1, strides=2, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    if not first:
        x = layers.Activation('relu')(x)
    x = _sep_bn(x, filters)
    x = layers.Activation('relu')(x)
    x = _sep_bn(x, filters)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    return layers.Add()([x, residual])


def _middle_block(x):
    """Middle flow residual block — identity shortcut, 728 channels."""
    residual = x
    x = layers.Activation('relu')(x)
    x = _sep_bn(x, 728)
    x = layers.Activation('relu')(x)
    x = _sep_bn(x, 728)
    x = layers.Activation('relu')(x)
    x = _sep_bn(x, 728)
    return layers.Add()([x, residual])


def build_xception(num_classes=1000, input_shape=(299, 299, 3)):
    """
    Xception — Extreme Inception via Depthwise Separable Convolutions.
    Paper: Chollet, CVPR 2017.

    Architecture:
      Entry Flow  : 2 plain convs + 3 separable residual blocks  -> 728x19x19
      Middle Flow : 8 identical separable residual blocks         -> 728x19x19
      Exit Flow   : 1 separable residual block + 2 sep convs      -> 2048x10x10
      Head        : GlobalAvgPool -> Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=input_shape)

    # ── Entry Flow — initial convolutions ──
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)                        # 32 x 150x150

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)                        # 64 x 150x150

    # ── Entry Flow — separable residual blocks ──
    x = _entry_block(x, 128, first=True)                    # 128 x 75x75
    x = _entry_block(x, 256)                                # 256 x 38x38
    x = _entry_block(x, 728)                                # 728 x 19x19

    # ── Middle Flow — 8 identical blocks ──
    for _ in range(8):
        x = _middle_block(x)                                # 728 x 19x19

    # ── Exit Flow — downsampling block ──
    residual = layers.Conv2D(1024, 1, strides=2, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu')(x)
    x = _sep_bn(x, 728)
    x = layers.Activation('relu')(x)
    x = _sep_bn(x, 1024)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = layers.Add()([x, residual])                         # 1024 x 10x10

    # ── Exit Flow — final separable convolutions ──
    x = _sep_bn(x, 1536)
    x = layers.Activation('relu')(x)                        # 1536 x 10x10
    x = _sep_bn(x, 2048)
    x = layers.Activation('relu')(x)                        # 2048 x 10x10

    x = layers.GlobalAveragePooling2D()(x)                  # 2048
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='xception')
